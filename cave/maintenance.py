"""Planned outages: anticipation from the maintenance forecast.

A reactive controller cannot see a planned outage coming: the line is in service and the grid calm until the moment it
is switched out, and some outages leave the remaining grid overloaded beyond repair in that same step. The observation
announces planned outages in advance (``time_next_maintenance``), so the controller probes the grid at the outage moment:

* **probe** -- if a line in service has an outage starting within ``maint_horizon`` steps, simulate do-nothing at that
  horizon with the outage lines out. If the predicted peak reaches ``maint_rho_trigger`` (or the flow diverges), the
  pre-outage mode is entered;
* **pre-outage arbitration** -- scan the whole action library on the post-outage grid (the ranker is trained on the
  current state and is blind to the outage), keep the best actions that are harmless now (next-step peak at or below
  max(rho_danger, current peak)), score them by the N-1 risk at the outage moment with the outage lines added to the
  contingency set, and execute the best one only if it lowers the post-outage peak by ``maint_margin`` or do-nothing is
  fatal. Repeated every step while the outage is imminent, so a few moves of pre-positioning emerge;
* **revert suppression** -- the calm-grid topology revert would undo the pre-positioning, so it is suppressed while an
  outage is active or starts within ``maint_revert_horizon`` steps.

Actions that touch a terminal of an outage line are excluded: a bus assignment there is a reconnection request, which
the forecast would grant although the line is in maintenance.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .actions import (find_best_line_to_reconnect, library_index, library_key, simulate_candidate,
                      split_action_and_return)


class MaintenanceMixin:
    """Methods of :class:`cave.controller.Controller` that handle planned outages."""

    def _imminent_outage(self, obs, horizon: int):
        """(lines, t_next): all in-service lines whose planned outage starts at the nearest start time within
        ``horizon`` steps (several lines often go out in the same slot), else None."""
        tnm = np.asarray(obs.time_next_maintenance, dtype=int)
        dur = np.asarray(obs.duration_next_maintenance, dtype=int)
        status = np.asarray(obs.line_status, dtype=bool)
        cand = [(int(t), int(l)) for l, t in enumerate(tnm) if 0 < t <= int(horizon) and dur[l] > 0 and status[l]]
        if not cand:
            return None
        t_next = min(t for t, _ in cand)
        lines = sorted(l for t, l in cand if t == t_next)
        return lines, t_next

    def _terminal_positions(self, lines) -> set:
        """topo_vect positions of both terminals of ``lines``."""
        pos = set()
        lo = np.asarray(self.action_space.line_or_pos_topo_vect)
        le = np.asarray(self.action_space.line_ex_pos_topo_vect)
        for l in lines:
            pos.add(int(lo[l])); pos.add(int(le[l]))
        return pos

    def _maint_near_or_active(self, obs, horizon: int) -> bool:
        tnm = np.asarray(obs.time_next_maintenance, dtype=int)
        return bool(((tnm >= 0) & (tnm <= int(horizon))).any())

    def _outage_vect(self, base_action, lines, extra=None):
        """Copy of ``base_action`` that also switches ``lines`` (and ``extra``) out."""
        a = deepcopy(base_action)
        arr = np.zeros(int(self.action_space.n_line), dtype=int)
        for l in lines:
            arr[int(l)] = -1
        if extra is not None:
            arr[int(extra)] = -1
        a.update({"set_line_status": arr})
        return a

    def _post_outage(self, obs, action, lines, t_next: int):
        """(peak loading, fatal) of ``action`` applied now with ``lines`` out, evaluated at the forecast horizon t_next
        (clipped to [1, forecast_horizon_max]). With no lines this is the ordinary next-step check."""
        ts = max(1, min(int(t_next), self.cfg.forecast_horizon_max))
        try:
            o, _, d, inf = obs.simulate(self._outage_vect(action, lines), time_step=ts)
        except Exception:  # noqa: BLE001
            return float("inf"), True
        if d or inf.get("is_illegal") or inf.get("is_ambiguous"):
            return float("inf"), True
        return float(o.rho.max()), False

    def _risk_with_outage(self, obs, action, lines, t_next: int) -> float:
        """N-1 risk (CVaR, see :meth:`Controller.n1_risk`) at the outage moment, the outage lines out in every case and
        added to the contingency set."""
        ts = max(1, min(int(t_next), self.cfg.forecast_horizon_max))
        rhos = []
        for att in list(self.lines_to_attack) + list(lines):
            try:
                o, _, d, _ = obs.simulate(self._outage_vect(action, lines, extra=att), time_step=ts)
                rhos.append(float("inf") if d else float(o.rho.max()))
            except Exception:  # noqa: BLE001
                rhos.append(float("inf"))
        return self._cvar(np.array(rhos))

    def _preoutage_decide(self, obs, lines, t_next: int):
        """Pre-outage arbitration. Returns (action, library index) or None when nothing should be done now."""
        a_noop = self.action_space({})
        r_noop, d_noop = self._post_outage(obs, a_noop, lines, t_next)
        if not d_noop and r_noop < self.cfg.maint_rho_trigger:
            return None                                          # the outage is harmless
        rho_now = float(obs.rho.max())
        now_cap = max(self.rho_danger, rho_now)
        forbidden_pos = self._terminal_positions(lines)
        scan = []
        for idx in range(self.actions.shape[0]):
            act = self.action_space.from_vect(self.actions[idx, :])
            sb = np.asarray(act._set_topo_vect)
            if any(sb[p] != 0 for p in forbidden_pos):
                continue
            rp, dp = self._post_outage(obs, act, lines, t_next)
            if not dp:
                scan.append((rp, int(idx)))
        scan.sort()
        members = [("noop", a_noop, rho_now, r_noop, d_noop)]
        for rp, idx in scan[: 4 * self.cfg.pool_size]:
            vect = self.actions[idx, :]
            r1, ok, _ = simulate_candidate(self.action_space, obs, vect)
            if not ok or r1 > now_cap + 1e-6:
                continue
            members.append((library_key(idx), self.action_space.from_vect(vect), float(r1), rp, False))
            if len(members) - 1 >= self.cfg.pool_size:
                break
        scored = []
        for key, act, r1, rp, dp in members:
            risk = float("inf") if dp else self._risk_with_outage(obs, act, lines, t_next)
            scored.append((key, risk, rp, r1, act))
        finite = [e for e in scored if np.isfinite(e[1])]
        if finite:
            finite.sort(key=lambda t: t[1])
            winner = finite[0]
        else:
            valid = [e for e in scored if np.isfinite(e[2])]
            if not valid:
                return None
            valid.sort(key=lambda t: t[2])
            winner = valid[0]
        improves = d_noop or (np.isfinite(winner[2]) and winner[2] < r_noop - self.cfg.maint_margin)
        if winner[0] == "noop" or not improves:
            return None
        idx = library_index(winner[0])
        self.next_actions = split_action_and_return(obs, self.action_space, self.actions[idx, :])
        act = next(self.next_actions)
        return find_best_line_to_reconnect(obs=obs, original_action=act), idx
