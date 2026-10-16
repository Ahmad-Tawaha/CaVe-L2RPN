"""Calibrated verification ("CaVe"): check a critical decision on the identified model before it is executed.

The controller plans on the published simulator. The real grid differs from it (here: per-line reactances), and the
training stage identifies that difference as one ratio per line. A *second* simulator instance carries the identified
reactances. At every crisis decision:

1. the controller's complete action (topology + reconnection + relief redispatch) is applied on the identified model,
   followed by ``verify_rollout_steps`` steps of the controller itself (do-nothing below the crisis trigger), in the
   *current* situation: existing outages, overload counters and scheduled events, no hypothetical attack;
2. if that rollout predicts the end of the episode, alternatives are tried in this order: the action the controller
   chooses when it plans on the identified model, then the other pool members by risk; if none survives, the whole
   action library is scanned on the identified model and the best ``verify_wide_k`` by next-step peak are tried;
3. an alternative replaces the action only if it survives the rollout on **both** models; otherwise the controller's
   action stands.

The action finally returned can differ from the checked one (later stages add redispatch, a feasibility release or the
request of an attacked line). If the verifier ran in this step and the outgoing action differs from the checked one,
the outgoing action is checked too (*final recheck*); if it fails, the verifier's alternatives are tried again, each
finalised through the same downstream stages.

Controller state is snapshotted before a check and restored after it, so the rollouts never change what the controller
remembers (queued plans, counters, caches). The verifier never runs inside its own rollouts.

When the training stage switches the correction on, the main simulator itself carries the identified reactances
(exponent 1); the second simulator then carries ratio ** (1 - 1), i.e. the published grid, and the check becomes
"the action must also survive on the published model".
"""
from __future__ import annotations

import copy
from collections import deque
from types import SimpleNamespace

import numpy as np

from .actions import find_best_line_to_reconnect, library_index, library_key
from .simulation import scale_line_reactances


class VerifyMixin:
    """Methods of :class:`cave.controller.Controller` that implement the verifier."""

    def _init_verifier(self):
        self.identified_ratios = None       # {line: ratio} of the identified model; None disables the verifier
        self.main_ratio_exponent = 0.0      # the main simulator already carries ratio ** this exponent (see CaVeAgent)
        self._identified_env = None         # the second simulator (built on first use)
        self._in_verify = False
        self._verify_count = 0              # verified decisions in the current episode
        self._verify_action = None
        # what the verifier checked in the current step (for the final recheck); not part of the state snapshot
        self._last_check = SimpleNamespace(checked=None, step=-1, pool_keys=[], outage=None)

    # ---------------------------------------------------------------- the identified model
    def identified_obs(self, obs):
        """A copy of ``obs`` whose ``simulate`` runs on the identified model (built and warmed up once)."""
        if self._identified_env is None:
            env2 = obs._obs_env.copy()
            c = obs.copy()
            c._obs_env = env2
            exponent = 1.0 - float(self.main_ratio_exponent)
            ratios = {int(l): float(r) ** exponent for l, r in self.identified_ratios.items() if r > 0}
            _n, err = scale_line_reactances(c, ratios, self.cfg.ratio_min, self.cfg.ratio_max)
            if err:
                raise RuntimeError(f"identified simulator not available: {err}")
            c.simulate(self.action_space({}))                   # the first call after re-initialisation is discarded
            self._identified_env = env2
        c = obs.copy()
        c._obs_env = self._identified_env
        return c

    def reset_identified_model(self):
        self._identified_env = None

    # ---------------------------------------------------------------- state snapshot
    def _snapshot(self):
        return {k: copy.copy(v) for k, v in self.__dict__.items()
                if isinstance(v, (int, float, bool, str, type(None), dict, list, set, tuple, deque))}

    # ---------------------------------------------------------------- survival rollout
    def complete_action(self, obs, key):
        """The executable action for a pool key: topology (+ best reconnection) + relief redispatch."""
        if key == "noop":
            act = find_best_line_to_reconnect(obs=obs, original_action=self.action_space({}))
        else:
            act = find_best_line_to_reconnect(obs=obs, original_action=self.action_space.from_vect(self.actions[library_index(key), :]))
        try:
            act = self._relief_redispatch(obs, act)
        except Exception:  # noqa: BLE001
            pass
        return act

    def survives(self, o, act):
        """Under the simulator attached to ``o``: apply ``act``, then ``verify_rollout_steps`` steps of the controller.
        True = alive at the end, False = the episode ends, None = the action is illegal or ambiguous."""
        s, _, d, inf = o.simulate(act)
        if inf.get("is_illegal") or inf.get("is_ambiguous"):
            return None
        if d:
            return False
        noop = self.action_space({})
        for _ in range(self.cfg.verify_rollout_steps):
            a = noop
            if float(np.asarray(s.rho).max()) >= self.rho_danger:
                try:
                    a, _ = self.decide_crisis(s, None)
                    try:
                        a = self._relief_redispatch(s, a)
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    a = noop
            s2, _, d, inf = s.simulate(a)
            if (inf.get("is_illegal") or inf.get("is_ambiguous")) and a is not noop:
                s2, _, d, inf = s.simulate(noop)
            if d:
                return False
            s = s2
        return True

    def _wide_scan(self, oc, exclude):
        """Library indices not in ``exclude`` (pool keys), sorted by next-step peak on the identified model."""
        scan = []
        for idx in range(self.actions.shape[0]):
            if library_key(idx) in exclude:
                continue
            try:
                o1, _, d1, inf1 = oc.simulate(self.action_space.from_vect(self.actions[idx, :]))
            except Exception:  # noqa: BLE001
                continue
            if d1 or inf1.get("is_illegal") or inf1.get("is_ambiguous"):
                continue
            scan.append((float(np.asarray(o1.rho).max()), idx))
        scan.sort()
        return scan

    # ---------------------------------------------------------------- the check at a crisis decision
    def verify_decision(self, obs, scored, winner_key):
        """Check the winner of a crisis decision on the identified model; return the (possibly replaced) pool key.
        If the action the controller chooses on the identified model is the replacement, it is stored in
        ``self._verify_action`` and returned unchanged as ``winner_key``."""
        if self._verify_count >= self.cfg.verify_max_per_episode:
            return winner_key
        new_key = winner_key
        chosen_action = None
        oc = self.identified_obs(obs)                       # built before the snapshot, so that it survives the restore
        snap = self._snapshot()
        self._in_verify = True
        m = self._last_check
        m.checked = None
        m.step = self.n_steps
        finite = sorted([e for e in scored if np.isfinite(e[1]) and e[0] != winner_key], key=lambda t: t[1])
        rest = sorted([e for e in scored if not np.isfinite(e[1]) and np.isfinite(e[2]) and e[0] != winner_key], key=lambda t: t[2])
        m.pool_keys = [e[0] for e in finite] + [e[0] for e in rest]
        try:
            base = self.complete_action(obs, winner_key)
            ok = self.survives(oc, base)
            m.checked = np.asarray(base.to_vect(), dtype=float).copy()
            if ok is False:                                  # the identified model predicts that this action ends the episode
                alts = []
                try:
                    a_cal, _ = self.decide_crisis(oc, None)
                    try:
                        a_cal = self._relief_redispatch(oc, a_cal)
                    except Exception:  # noqa: BLE001
                        pass
                    alts.append(("identified_choice", a_cal))
                except Exception:  # noqa: BLE001
                    pass
                for e in (finite + rest)[: self.cfg.verify_alternatives]:
                    alts.append((e[0], None))
                found = False
                for key, act in alts:
                    if act is None:
                        act = self.complete_action(obs, key)
                    if self.survives(oc, act) is True and self.survives(obs, act) is True:
                        found = True
                        m.checked = np.asarray(act.to_vect(), dtype=float).copy()
                        if key == "identified_choice":
                            chosen_action = act
                        else:
                            new_key = key
                        break
                if not found:
                    seen = {str(k) for k, _ in alts}
                    for _, idx in self._wide_scan(oc, seen)[: self.cfg.verify_wide_k]:
                        key = library_key(idx)
                        act = self.complete_action(obs, key)
                        if self.survives(oc, act) is True and self.survives(obs, act) is True:
                            new_key = key
                            m.checked = np.asarray(act.to_vect(), dtype=float).copy()
                            break
        finally:
            self.__dict__.update(snap)
            self._in_verify = False
            self.next_actions = None
        self._verify_count += 1
        if chosen_action is not None:
            self._verify_action = (chosen_action, -1)
        return new_key

    # ---------------------------------------------------------------- final recheck of the outgoing action
    def final_recheck(self, obs, action):
        """If the verifier ran this step and the outgoing action differs from the checked one, check it too."""
        m = self._last_check
        if m.step != self.n_steps or m.checked is None or not self.identified_ratios or self._in_verify:
            return action
        v = np.asarray(action.to_vect(), dtype=float)
        if v.shape == m.checked.shape and np.allclose(v, m.checked):
            return action
        oc = self.identified_obs(obs)
        snap = self._snapshot()
        queued = self.next_actions
        self._in_verify = True
        out = action
        try:
            if self.survives(oc, action) is not False:
                return action
            keys = list(m.pool_keys)
            seen = set(keys)
            keys += [library_key(idx) for _, idx in self._wide_scan(oc, seen)[: self.cfg.verify_wide_k]]
            for key in keys:
                try:
                    raw = self.action_space({}) if key == "noop" else self.action_space.from_vect(self.actions[library_index(key), :])
                    fin = find_best_line_to_reconnect(obs=obs, original_action=raw)
                    fin = self.downstream(obs, fin, m.outage)
                    fin = self.attach_probe(obs, fin)
                except Exception:  # noqa: BLE001
                    continue
                if self.survives(oc, fin) is True and self.survives(obs, fin) is True:
                    out = fin
                    queued = None                           # a queued plan belonged to the replaced action
                    break
            return out
        finally:
            self.__dict__.update(snap)
            self._in_verify = False
            self.next_actions = queued
