"""The CaVe decision pipeline (one call of :meth:`Controller.step` per environment step).

In reading order:

1. **attacked lines** -- refresh the set of lines out of service because of an ongoing attack
   (:data:`cave.actions.SUSPECT_LINES`); no simulation may bring them back.
2. **queued plan** -- a compound topology action is executed one substation per step; continue it while legal.
3. **calm grid** (peak loading below ``rho_danger``):
     a. pre-outage mode if a planned outage is imminent and risky (:mod:`cave.maintenance`);
     b. otherwise a topology revert: repair attackable bridges first, else the relaxed or classic revert toward the base
        topology (suppressed near planned outages) (:mod:`cave.fragility`);
     c. redispatch upkeep: feasibility release, pre-outage ramp, unwind, target hygiene, price release
        (:mod:`cave.redispatch`).
4. **crisis** (:meth:`Controller.decide_crisis`): the ranker's ``top_k`` proposals are simulated; do-nothing and the best
   ``pool_size`` proposals form the pool; every member is scored by its N-1 risk (CVaR over the attackable-line
   outages, an imminent planned outage added) plus a structural-fragility penalty; near-ties are separated by a short
   look-ahead; when every member is fatal for some contingency, the contingencies fatal for all members are dropped and
   the rest decides. The winner is then checked on the identified model (:mod:`cave.verify`). Relief redispatch is added
   on top.
5. **attacked-line request** -- if the outgoing action leaves the topology alone, ask for one attacked line back (if
   the attack is over, the line returns; otherwise the request is refused and nothing is lost).
6. **final recheck** of the outgoing action on the identified model when the verifier ran in this step.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from . import actions as _actions
from .actions import (find_best_line_to_reconnect, is_legal, library_index, library_key, revert_topo, simulate_candidate,
                      simulate_rho, split_action_and_return)
from .config import CaVeConfig
from .fragility import FragilityMixin
from .maintenance import MaintenanceMixin
from .ranker import ActionRanker
from .redispatch import RedispatchMixin
from .verify import VerifyMixin


class Controller(MaintenanceMixin, RedispatchMixin, FragilityMixin, VerifyMixin):
    """Rule-based controller around the learned ranker. Holds per-episode state (queued plan, counters, caches)."""

    def __init__(self, action_space, ranker: ActionRanker, config: CaVeConfig):
        self.action_space = action_space
        self.cfg = config
        self.ranker = ranker
        self.actions = ranker.actions
        self.lines_to_attack = list(config.attackable_lines)
        self.rho_danger = float(config.rho_danger)
        self.cvar_alpha = float(config.cvar_alpha)
        self.redisp_target = float(config.redisp_target)   # switched by the agent between the normal and low-load value
        self.calm_revert = True                            # relaxed calm revert; the training stage may switch it off
        self.next_actions = None
        self.n_steps = 0
        self._bridge_cache = {}
        self._init_verifier()

    # ================================================================ episode
    def reset(self):
        self.next_actions = None
        self._verify_count = 0

    # ================================================================ risk
    def _cvar(self, rhos: np.ndarray) -> float:
        """CVaR_alpha of post-contingency peak loadings: mean of the worst ceil(alpha * n); +inf if a fatal case is there."""
        k = max(1, int(np.ceil(self.cvar_alpha * len(rhos))))
        worst = np.sort(rhos)[-k:]
        if np.any(np.isinf(worst)):
            return float("inf")
        return float(np.mean(worst))

    def n1_risk(self, obs, action_vect) -> float:
        """N-1 risk of a library action: one next-step simulation per attackable-line outage."""
        action = self.action_space.from_vect(action_vect)
        rhos = []
        for line in self.lines_to_attack:
            status = np.zeros(obs.rho.shape, dtype=int)
            status[line] = -1
            a = deepcopy(action)
            a.update({"set_line_status": status})
            o, _, done, _ = obs.simulate(a)
            rhos.append(float("inf") if done else float(o.rho.max()))
        return self._cvar(np.array(rhos))

    def _contingency_outcomes(self, obs, vect):
        """Per-contingency peak loading (inf = fatal) of a library action, in the order of ``lines_to_attack``."""
        action = self.action_space.from_vect(vect)
        out = []
        for line in self.lines_to_attack:
            status = np.zeros(obs.rho.shape, dtype=int)
            status[line] = -1
            a = deepcopy(action)
            a.update({"set_line_status": status})
            try:
                o, _, d, _ = obs.simulate(a)
                out.append(float("inf") if d else float(o.rho.max()))
            except Exception:  # noqa: BLE001
                out.append(float("inf"))
        return out

    def _all_fatal_pick(self, obs, pool, scored, penalty):
        """When every member's risk is infinite: drop the contingencies that are fatal for every valid member (the
        fatality belongs to the state, not to the candidate) and rank by (number of other fatal contingencies,
        CVaR over the remaining finite ones + fragility penalty, next-step peak). Returns the key or None."""
        valid = [e for e in scored if np.isfinite(e[2])]
        if len(valid) < 2:
            return None
        vec = {key: vect for key, act, vect, r1, o1 in pool}
        outs = {e[0]: self._contingency_outcomes(obs, vec[e[0]]) for e in valid}
        n = len(self.lines_to_attack)
        fatal = {k: {i for i, r in enumerate(v) if not np.isfinite(r)} for k, v in outs.items()}
        common = set.intersection(*fatal.values()) if fatal else set()
        keep = [i for i in range(n) if i not in common]
        if not keep:
            return None
        ranked = []
        for e in valid:
            k = e[0]
            rest = [outs[k][i] for i in keep]
            extra = sum(1 for r in rest if not np.isfinite(r))
            fin = [r for r in rest if np.isfinite(r)]
            if fin:
                kk = max(1, int(np.ceil(self.cvar_alpha * len(fin))))
                cv = float(np.mean(np.sort(np.array(fin))[-kk:]))
            else:
                cv = float("inf")
            ranked.append((extra, cv + penalty.get(k, 0.0), float(e[2]), k))
        ranked.sort()
        return ranked[0][3]

    # ================================================================ look-ahead
    def _beam_pick(self, close_call):
        """Separate near-tied candidates by a depth-3 look-ahead on simulated observations.

        close_call: list of (key, risk, rho1, obs1). Score of a candidate = rho1 + gamma * rho2 + gamma^2 * rho3 with
        rho2 over the ranker's top-k on obs1 (and holding), pruned to beam_width, and rho3 the best over the ranker's
        top-k on each depth-2 state (and holding). Returns the best key, or None if nothing can be scored."""
        cfg = self.cfg
        beam = sorted(close_call, key=lambda t: t[2])[: cfg.beam_width] if len(close_call) > cfg.beam_width else close_call
        best_score = float("inf")
        best_key = None
        penalty = cfg.no_continuation_penalty
        noop_vect = None
        for key1, _risk, rho1, obs1 in beam:
            try:
                sorted2 = self.ranker.rank(obs1)[: cfg.top_k]
            except Exception:  # noqa: BLE001
                sorted2 = []
            depth2 = []
            for idx2 in sorted2:
                try:
                    rho2, valid2, obs2 = simulate_candidate(self.action_space, obs1, self.actions[int(idx2), :])
                except Exception:  # noqa: BLE001
                    continue
                if valid2:
                    depth2.append((int(idx2), float(rho2), obs2))
            try:                                             # holding is a legitimate continuation
                noop_vect = self.action_space({}).to_vect()
                rho2h, valid2h, obs2h = simulate_candidate(self.action_space, obs1, noop_vect)
                if valid2h:
                    depth2.append((-1, float(rho2h), obs2h))
            except Exception:  # noqa: BLE001
                pass
            if not depth2:
                score = rho1 + cfg.gamma * penalty + (cfg.gamma ** 2) * penalty
                if score < best_score:
                    best_score = score
                    best_key = key1
                continue
            depth2.sort(key=lambda t: t[1])
            best_over_depth2 = float("inf")
            for _idx2, rho2, obs2 in depth2[: cfg.beam_width]:
                try:
                    sorted3 = self.ranker.rank(obs2)[: cfg.top_k]
                except Exception:  # noqa: BLE001
                    sorted3 = []
                best_rho3 = float("inf")
                try:
                    rho3h, valid3h = simulate_rho(self.action_space, obs2, self.action_space({}).to_vect())
                    if valid3h:
                        best_rho3 = float(rho3h)
                except Exception:  # noqa: BLE001
                    pass
                for idx3 in sorted3:
                    try:
                        rho3, valid3 = simulate_rho(self.action_space, obs2, self.actions[int(idx3), :])
                    except Exception:  # noqa: BLE001
                        continue
                    if valid3 and rho3 < best_rho3:
                        best_rho3 = float(rho3)
                tail = penalty if best_rho3 == float("inf") else best_rho3
                score3 = rho1 + cfg.gamma * rho2 + (cfg.gamma ** 2) * tail
                if score3 < best_over_depth2:
                    best_over_depth2 = score3
            if best_over_depth2 < best_score:
                best_score = best_over_depth2
                best_key = key1
        return best_key

    # ================================================================ crisis decision
    def decide_crisis(self, obs, outage=None):
        """Crisis decision on ``obs`` (a real or a simulated observation). Returns (action, library index or -1)."""
        cfg = self.cfg
        pool = []
        a_noop = self.action_space({})
        r1, ok, o1 = simulate_candidate(self.action_space, obs, a_noop.to_vect())
        pool.append(("noop", a_noop, a_noop.to_vect(), float(r1) if ok else float("inf"), o1))
        ranked = self.ranker.rank(obs)[: cfg.max_ranked]
        proposals = []
        for idx in ranked[: cfg.top_k]:
            vect = self.actions[idx, :]
            r1, ok, o1 = simulate_candidate(self.action_space, obs, vect)
            if ok:
                proposals.append((int(idx), float(r1), o1))
        proposals.sort(key=lambda t: t[1])
        r1_noop = pool[0][3]
        for idx, r1, o1 in proposals[: cfg.pool_size]:
            if r1 > 1.0 and np.isfinite(r1_noop) and r1 > r1_noop + cfg.worse_guard:
                continue                                     # overloaded and clearly worse than doing nothing
            pool.append((library_key(idx), self.action_space.from_vect(self.actions[idx, :]), self.actions[idx, :], r1, o1))

        # N-1 risk; an imminent planned outage joins the contingency set
        saved = list(self.lines_to_attack)
        if outage is not None:
            self.lines_to_attack = saved + [int(l) for l in outage[0] if int(l) not in saved]
        try:
            scored = [(key, float(self.n1_risk(obs, vect)), r1, o1, act) for key, act, vect, r1, o1 in pool]
        finally:
            self.lines_to_attack = saved

        # structural fragility: lambda * (attackable bridges after the action - now), on finite risks
        penalty = {}
        try:
            ls_now = np.asarray(obs.line_status, dtype=bool)
            f_now = self.bridge_count(np.asarray(obs.topo_vect), ls_now)
            vec = {key: vect for key, act, vect, r1, o1 in pool}
            for key in vec:
                f_a = f_now if key == "noop" else self.bridge_count(self._topo_after(obs, vec[key]), self._status_after(obs, vec[key]))
                penalty[key] = cfg.fragility_lambda * (f_a - f_now)
            scored = [(e[0], (e[1] + penalty.get(e[0], 0.0)) if np.isfinite(e[1]) else e[1], e[2], e[3], e[4]) for e in scored]
        except Exception:  # noqa: BLE001
            penalty = {}

        finite = [e for e in scored if np.isfinite(e[1])]
        if finite:
            finite.sort(key=lambda t: t[1])
            best = finite[0][1]
            close = [e for e in finite if e[1] - best <= cfg.tiebreak_eps]
            winner_key = finite[0][0]
            if len(close) > 1 and cfg.plan_depth >= 2:
                pick = self._beam_pick([(e[0], e[1], e[2], e[3]) for e in close])
                if pick is not None:
                    winner_key = pick
        else:
            valid = [e for e in scored if np.isfinite(e[2])]
            safe = [e for e in valid if e[2] <= self.rho_danger]
            cand = sorted(safe or valid, key=lambda t: t[2])
            winner_key = cand[0][0] if cand else "noop"
            if len(valid) >= 2:
                try:
                    saved2 = list(self.lines_to_attack)
                    if outage is not None:
                        self.lines_to_attack = saved2 + [int(l) for l in outage[0] if int(l) not in saved2]
                    try:
                        pick = self._all_fatal_pick(obs, pool, scored, penalty)
                    finally:
                        self.lines_to_attack = saved2
                    if pick is not None:
                        winner_key = pick
                except Exception:  # noqa: BLE001
                    pass

        # verification on the identified model
        self._verify_action = None
        if not self._in_verify and self.identified_ratios:
            try:
                winner_key = self.verify_decision(obs, scored, winner_key)
            except Exception:  # noqa: BLE001
                self._in_verify = False
        if self._verify_action is not None:                  # the identified model's own choice replaced the action
            self.next_actions = None
            return self._verify_action
        if winner_key not in {e[0] for e in scored}:         # a wide-scan replacement outside the pool
            idx_w = library_index(winner_key)
            try:
                simulate_candidate(self.action_space, obs, self.actions[idx_w, :])
            except Exception:  # noqa: BLE001
                pass

        self.next_actions = None
        if winner_key == "noop":
            return find_best_line_to_reconnect(obs=obs, original_action=a_noop), -1
        idx = library_index(winner_key)
        self.next_actions = split_action_and_return(obs, self.action_space, self.actions[idx, :])
        act = next(self.next_actions)
        return find_best_line_to_reconnect(obs=obs, original_action=act), idx

    def _fallback_decision(self, obs):
        """Used only if :meth:`decide_crisis` raises: the ranker's top-k, filtered to candidates below the trigger,
        ranked by N-1 risk with the look-ahead tie-break; else the first ranked action below the trigger or the one
        with the lowest peak; else do-nothing. (Index 0 of the library counts as "nothing found" in the last stage.)"""
        cfg = self.cfg
        min_rho = obs.rho.max()
        idx_chosen = None
        ranked = self.ranker.rank(obs)[: cfg.max_ranked]
        safe = []
        for idx in ranked[: cfg.top_k]:
            rho, valid, obs_after = simulate_candidate(self.action_space, obs, self.actions[idx, :])
            if valid and rho <= self.rho_danger:
                safe.append((int(idx), float(rho), obs_after))
        if safe:
            scored = sorted([(idx, float(self.n1_risk(obs, self.actions[idx, :])), rho1, obs1) for idx, rho1, obs1 in safe],
                            key=lambda t: t[1])
            close = [e for e in scored if e[1] - scored[0][1] <= cfg.tiebreak_eps]
            if len(close) <= 1 or cfg.plan_depth < 2:
                idx_chosen = scored[0][0]
            else:
                idx_chosen = self._beam_pick(close)
                if idx_chosen is None:
                    idx_chosen = scored[0][0]
            if idx_chosen is not None:
                self.next_actions = split_action_and_return(obs, self.action_space, self.actions[idx_chosen, :])
                return find_best_line_to_reconnect(obs=obs, original_action=next(self.next_actions)), idx_chosen
        for idx in ranked:
            rho, valid = simulate_rho(self.action_space, obs, self.actions[idx, :])
            if not valid:
                continue
            if rho <= self.rho_danger:
                idx_chosen = idx
                break
            if rho < min_rho:
                min_rho = rho
                idx_chosen = idx
        if idx_chosen:
            self.next_actions = split_action_and_return(obs, self.action_space, self.actions[idx_chosen, :])
            return find_best_line_to_reconnect(obs=obs, original_action=next(self.next_actions)), idx_chosen
        return find_best_line_to_reconnect(obs=obs, original_action=self.action_space({})), -1

    # ================================================================ attacked lines
    def update_suspects(self, obs):
        """Lines out of service, attackable, attacked less than ``attack_mask_steps`` ago and not in maintenance."""
        _actions.SUSPECT_LINES.clear()
        try:
            tsa = np.asarray(obs.time_since_last_attack, dtype=int)
            ids = getattr(obs, "alertable_line_ids", None)
            ids = [int(x) for x in ids] if ids is not None and len(ids) == len(tsa) else [int(x) for x in list(self.lines_to_attack)[:len(tsa)]]
            st = np.asarray(obs.line_status, dtype=bool)
            tnm = np.asarray(obs.time_next_maintenance, dtype=int)
            for j, l in enumerate(ids):
                if (not st[l]) and 0 <= int(tsa[j]) < self.cfg.attack_mask_steps and int(tnm[l]) != 0:
                    _actions.SUSPECT_LINES.add(int(l))
        except Exception:  # noqa: BLE001
            _actions.SUSPECT_LINES.clear()

    def attach_probe(self, obs, action):
        """Ask for one attacked line back, only on an action that leaves the topology alone."""
        if not _actions.SUSPECT_LINES:
            return action
        try:
            if np.any(np.asarray(action.set_bus) != 0) or np.any(np.asarray(action.change_bus)) \
                    or np.any(np.asarray(action.set_line_status) != 0) or np.any(np.asarray(action.change_line_status)):
                return action
            cd = np.asarray(obs.time_before_cooldown_line, dtype=int)
            cand = [l for l in sorted(_actions.SUSPECT_LINES) if cd[l] == 0]
            if not cand:
                return action
            arr = np.zeros(int(self.action_space.n_line), dtype=int)
            arr[cand[0]] = 1
            a2 = deepcopy(action)
            a2.update({"set_line_status": arr})
            return a2
        except Exception:  # noqa: BLE001
            return action

    # ================================================================ one step
    def step(self, obs):
        """The controller's action for this observation."""
        self.update_suspects(obs)
        action = self._decide(obs)
        action = self.attach_probe(obs, action)
        try:
            action = self.final_recheck(obs, action)
        except Exception:  # noqa: BLE001
            pass
        return action

    def downstream(self, obs, action, outage_r):
        """Stages applied to a crisis decision before it is returned: feasibility release, else pre-outage ramp (if an
        outage is imminent) or relief ramp; then target hygiene."""
        action, released = self._release_if_infeasible(obs, action)
        if released:
            pass
        elif outage_r is not None:
            action = self._preoutage_redispatch(obs, action, *outage_r)
        else:
            action = self._relief_redispatch(obs, action)
        return self._target_hygiene(obs, action)

    def _decide(self, obs):
        cfg = self.cfg
        self.n_steps += 1

        # queued plan
        if self.next_actions is not None:
            try:
                nxt = find_best_line_to_reconnect(obs=obs, original_action=next(self.next_actions))
                if is_legal(nxt, obs):
                    return nxt
            except StopIteration:
                self.next_actions = None

        try:
            outage = self._imminent_outage(obs, cfg.maint_horizon)
            outage_r = self._imminent_outage(obs, cfg.redisp_horizon)
        except Exception:  # noqa: BLE001
            outage = outage_r = None

        if obs.rho.max() < self.rho_danger:
            return self._decide_calm(obs, outage, outage_r)

        try:
            action, _idx = self.decide_crisis(obs, outage)
        except Exception:  # noqa: BLE001
            self.next_actions = None
            action, _idx = self._fallback_decision(obs)
        self._last_check.outage = outage_r
        try:
            action = self.downstream(obs, action, outage_r)
        except Exception:  # noqa: BLE001
            pass
        return action

    def _decide_calm(self, obs, outage, outage_r):
        cfg = self.cfg
        res = None
        if outage is not None:
            try:
                res = self._preoutage_decide(obs, *outage)
            except Exception:  # noqa: BLE001
                self.next_actions = None
                res = None
        if res is not None:
            action = res[0]
        else:
            try:
                suppress = self._maint_near_or_active(obs, cfg.maint_revert_horizon)
            except Exception:  # noqa: BLE001
                suppress = False
            frag_vec = None
            if not suppress and float(obs.rho.max()) < self.rho_danger:
                try:
                    frag_vec = self._fragility_revert(obs)
                except Exception:  # noqa: BLE001
                    frag_vec = None
            if frag_vec is not None:
                default_action = self.action_space.from_vect(frag_vec)
            elif not suppress:
                vect = None
                if self.calm_revert:
                    try:
                        vect = self._revert_topo_relaxed(obs)
                    except Exception:  # noqa: BLE001
                        vect = None
                if vect is None:
                    vect = revert_topo(self.action_space, obs, rho_limit=cfg.revert_rho_limit)
                default_action = self.action_space.from_vect(vect)
            else:
                default_action = self.action_space({})
            action = find_best_line_to_reconnect(obs=obs, original_action=default_action)

        released = False
        try:
            action, released = self._release_if_infeasible(obs, action)
            if released:
                pass
            elif outage_r is not None:
                action = self._preoutage_redispatch(obs, action, *outage_r)
            elif float(obs.rho.max()) < cfg.unwind_rho and not self._maint_near_or_active(obs, cfg.unwind_horizon):
                action = self._unwind_redispatch(obs, action)
            elif float(obs.rho.max()) < cfg.unwind_calm_rho:
                action = self._unwind_redispatch(obs, action)   # even near an outage: a held transfer is pure risk
            action = self._target_hygiene(obs, action)
        except Exception:  # noqa: BLE001
            pass
        if not released and outage_r is None and not self._maint_near_or_active(obs, cfg.unwind_horizon):
            try:
                action = self._price_release(obs, action, max(self.rho_danger, float(obs.rho.max())))
            except Exception:  # noqa: BLE001
                pass
        return action
