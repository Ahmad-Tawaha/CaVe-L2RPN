"""Redispatch rules: the lever that topology does not supply.

All rules add balanced redispatch to an action that the topology part of the controller already chose, and every
candidate is accepted only after an ``obs.simulate`` check. They are applied in this order by the controller:

* **feasibility release** -- if the next-step forecast of the action fails with a redispatching error (the carried
  dispatch can no longer be balanced as the schedule moves; in the real environment that ends the episode), move every
  unit's target one ramp step toward its actual dispatch;
* **pre-outage ramp** -- before a planned outage whose post-outage grid is at or above the target loading, add one ramp
  step of balanced redispatch; repeated every step until the predicted post-outage peak is under target;
* **relief ramp** -- in a crisis, the same ramp on the next-step grid; units that are off are started only when the
  running units cannot provide a ramp, first those not more expensive than the current price setter, then any;
* **unwind** -- when the grid is calm and no outage is near, ramp the carried dispatch back toward zero;
* **target hygiene** -- a unit sitting at a bound whose target differs from its actual dispatch carries an unreachable
  request, which makes the dispatch infeasible when the schedule moves; move such targets toward actual;
* **price release** -- the operating cost of a step is priced at the most expensive producing unit; a unit that
  produces only because of a tiny positive dispatch residual and sets that price is pushed to zero, balanced on cheaper
  running units.

The ramp direction comes from single-unit sensitivities at horizon 1 with the outage lines out; the transfer is
min(total room up, total room down) over the best units, allocated in proportion to room, scaled down (1, 1/2, 1/4)
if the full transfer fails the checks.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np


class RedispatchMixin:
    """Methods of :class:`cave.controller.Controller` that build redispatch."""

    # ---------------------------------------------------------------- helpers
    def _dispatch_rooms(self, obs, eligibility: str = "producing"):
        """Dispatchable units with their one-step ramp room (up, down) in MW.

        eligibility: ``producing`` -- units that are off are never started; ``price`` -- an off unit may start if its
        price does not exceed the current price setter's; ``all`` -- any unit may start."""
        sp = self.action_space
        rd = np.asarray(sp.gen_redispatchable, dtype=bool)
        p = np.asarray(obs.gen_p, dtype=float)
        rooms = []
        for g in np.where(rd)[0]:
            up = min(float(sp.gen_max_ramp_up[g]), float(sp.gen_pmax[g] - p[g]))
            dn = min(float(sp.gen_max_ramp_down[g]), float(p[g] - sp.gen_pmin[g]))
            if p[g] < 1.0:                                     # an off unit: may it be started?
                price_now = float(np.max(np.asarray(sp.gen_cost_per_MW, dtype=float)[p > 0.0])) if np.any(p > 0.0) else 0.0
                start_ok = (eligibility == "all") or (eligibility == "price" and float(sp.gen_cost_per_MW[g]) <= price_now + 1e-9)
                if not start_ok:
                    up = 0.0
            rooms.append((int(g), max(0.0, up), max(0.0, dn)))
        return rooms

    def _with_redispatch(self, base_action, vec):
        """Copy of ``base_action`` with the listed (unit, MW) redispatch requests (entries under 0.05 MW are dropped)."""
        a = deepcopy(base_action)
        a.redispatch = [(int(g), float(v)) for g, v in vec if abs(v) > 0.05]
        return a

    def _now_ok(self, obs, action, cap) -> bool:
        """The action must be legal now and keep the next-step peak loading at or below ``cap``."""
        try:
            o, _, d, inf = obs.simulate(action, time_step=1)
        except Exception:  # noqa: BLE001
            return False
        if d or inf.get("is_illegal") or inf.get("is_ambiguous"):
            return False
        return float(o.rho.max()) <= cap + 1e-6

    # ---------------------------------------------------------------- ramps
    def _preoutage_redispatch(self, obs, base_action, lines, t_next: int):
        """One balanced ramp step before a planned outage, while the post-outage grid is still at or above target.

        The target is checked at the true horizon; once dispatch is carried that probe can be falsely fatal (the forecast
        re-applies the carried dispatch against a multi-step jump of the scheduled injections), so a fatal true-horizon
        probe falls back to the outage-now probe (horizon 1)."""
        rp_h, dp_h = self._post_outage(obs, base_action, lines, t_next)
        rp1, dp1 = self._post_outage(obs, base_action, lines, 1)
        safe = (not dp_h and rp_h < self.redisp_target) or (dp_h and not dp1 and rp1 < self.redisp_target)
        if safe:
            return base_action
        return self._redispatch_ramp(obs, base_action, lines, relief=False)

    def _relief_redispatch(self, obs, base_action):
        """Crisis: one balanced ramp step on top of the topology decision while the next-step peak is at or above target."""
        r0, d0 = self._post_outage(obs, base_action, [], 1)
        if not d0 and r0 < self.redisp_target:
            return base_action
        return self._redispatch_ramp(obs, base_action, [], relief=True)

    def _redispatch_ramp(self, obs, base_action, lines, relief: bool):
        """Pre-outage ramps use producing units only. Relief ramps try producing units, then off units no more
        expensive than the price setter, then any unit, and keep the first eligibility class that yields a ramp."""
        if not relief:
            return self._redispatch_ramp_core(obs, base_action, lines)
        for eligibility in ("producing", "price", "all"):
            cand = self._redispatch_ramp_core(obs, base_action, lines, self._dispatch_rooms(obs, eligibility))
            if cand is not base_action:
                return cand
        return base_action

    def _redispatch_ramp_core(self, obs, base_action, lines, rooms=None):
        """One balanced ramp step (see module docstring). Returns ``base_action`` itself when no ramp passes the checks."""
        rp, dp = self._post_outage(obs, base_action, lines, 1)
        rho_now = float(obs.rho.max())
        now_cap = max(self.rho_danger, rho_now)
        rooms = self._dispatch_rooms(obs) if rooms is None else rooms
        if dp and len(lines) > 0:
            return base_action                                   # the outage is fatal even now: no direction to rank
        ups, downs = [], []
        for g, ru, rdn in rooms:
            if ru > 0.5:
                r, d = self._post_outage(obs, self._with_redispatch(base_action, [(g, ru)]), lines, 1)
                if not d and (dp or r < rp - 1e-4):
                    ups.append(((r - rp) / ru if not dp else -1.0 / ru, g, ru))
            if rdn > 0.5:
                r, d = self._post_outage(obs, self._with_redispatch(base_action, [(g, -rdn)]), lines, 1)
                if not d and (dp or r < rp - 1e-4):
                    downs.append(((r - rp) / rdn if not dp else -1.0 / rdn, g, rdn))
        ups.sort()
        downs.sort()
        ups = ups[: self.cfg.redisp_n_up]
        downs = downs[: self.cfg.redisp_n_down]
        if not ups or not downs:
            return base_action
        su = sum(ru for _, _, ru in ups)
        sd = sum(rdn for _, _, rdn in downs)
        tot = min(su, sd)
        carried = max(float(np.clip(np.asarray(obs.target_dispatch, dtype=float), 0.0, None).sum()),
                      0.5 * float(np.abs(np.asarray(obs.actual_dispatch, dtype=float)).sum()))
        room_cap = self.cfg.redisp_max_mw - carried
        if room_cap < 1.0:
            return base_action
        tot = min(tot, room_cap)
        for scale in (1.0, 0.5, 0.25):
            vec = [(g, ru / su * tot * scale) for _, g, ru in ups] + [(g, -rdn / sd * tot * scale) for _, g, rdn in downs]
            cand = self._with_redispatch(base_action, vec)
            if not self._now_ok(obs, cand, now_cap):
                continue
            r2, d2 = self._post_outage(obs, cand, lines, 1)
            if d2 or (not dp and r2 >= rp - 1e-3):
                continue
            return cand
        return base_action

    # ---------------------------------------------------------------- dispatch upkeep
    def _release_if_infeasible(self, obs, base_action):
        """Feasibility release. Returns (action, released)."""
        td = np.asarray(obs.target_dispatch, dtype=float)
        if not np.any(np.abs(td) > 0.5):
            return base_action, False
        try:
            o, _, d, inf = obs.simulate(base_action, time_step=1)
        except Exception:  # noqa: BLE001
            return base_action, False
        if not d or not any("edispatch" in str(e) for e in inf.get("exception", [])):
            return base_action, False
        ad = np.asarray(obs.actual_dispatch, dtype=float)
        sp = self.action_space
        vec = []
        for g in np.where(np.abs(td - ad) > 0.5)[0]:
            delta = ad[g] - td[g]
            step = float(sp.gen_max_ramp_up[g] if delta > 0 else sp.gen_max_ramp_down[g])
            vec.append((int(g), float(np.sign(delta)) * min(step, abs(delta))))
        if not vec:
            return base_action, False
        cand = self._with_redispatch(base_action, vec)
        try:
            o, _, d, inf = obs.simulate(cand, time_step=1)
        except Exception:  # noqa: BLE001
            return base_action, False
        if d or inf.get("is_illegal") or inf.get("is_ambiguous"):
            return base_action, False
        return cand, True

    def _target_hygiene(self, obs, action):
        """Move unreachable targets (unit at a bound, target != actual) one ramp step toward actual, on units the action
        leaves free. The forecast does not show the infeasibility such targets cause, hence the proactive rule."""
        td = np.asarray(obs.target_dispatch, dtype=float)
        ad = np.asarray(obs.actual_dispatch, dtype=float)
        p = np.asarray(obs.gen_p, dtype=float)
        sp = self.action_space
        used = {int(g) for g, v in enumerate(np.asarray(action.redispatch, dtype=float)) if abs(v) > 0.05}
        vec = []
        for g in np.where(np.abs(td - ad) > 0.5)[0]:
            g = int(g)
            if g in used:
                continue
            at_min = p[g] <= float(sp.gen_pmin[g]) + 0.5
            at_max = p[g] >= float(sp.gen_pmax[g]) - 0.5
            if not (at_min or at_max):
                continue
            delta = ad[g] - td[g]
            step = float(sp.gen_max_ramp_up[g] if delta > 0 else sp.gen_max_ramp_down[g])
            vec.append((g, float(np.sign(delta)) * min(step, abs(delta))))
        if not vec:
            return action
        cand = deepcopy(action)
        cand.redispatch = [(int(g), float(v)) for g, v in enumerate(np.asarray(action.redispatch, dtype=float)) if abs(v) > 0.05] + vec
        try:
            o, _, d, inf = obs.simulate(cand, time_step=1)
        except Exception:  # noqa: BLE001
            return action
        if d or inf.get("is_illegal") or inf.get("is_ambiguous"):
            return action
        return cand

    def _price_release(self, obs, action, cap):
        """Push price-setting units that run only on a dispatch residual (schedule ~0, target ~0) to zero with a small
        negative request (the environment clips at the bound), balanced on the cheapest running units. The forecast
        does not reproduce this micro-effect, so the rule relies on the observed state plus the ordinary checks
        (legal, next-step peak <= cap)."""
        sp = self.action_space
        gc = np.asarray(sp.gen_cost_per_MW, dtype=float)
        rd = np.asarray(sp.gen_redispatchable, dtype=bool)
        p = np.asarray(obs.gen_p, dtype=float)
        ad = np.asarray(obs.actual_dispatch, dtype=float)
        td = np.asarray(obs.target_dispatch, dtype=float)
        running = p > 0.0
        if not running.any():
            return action
        base_price = float(gc[running].max())
        cands = [int(g) for g in np.where(rd & running & (ad > 0.0) & (p - ad <= 0.5) & (np.abs(td) <= 0.5))[0]
                 if gc[g] >= base_price - 1e-6]
        if not cands:
            return action
        others = running.copy()
        others[cands] = False
        price_wo = float(gc[others].max()) if others.any() else 0.0
        if price_wo >= base_price - 1e-6:                     # another running unit sets the same price: no gain possible
            return action
        vec = {g: -float(min(1.0, sp.gen_max_ramp_down[g])) for g in cands}
        need = -sum(vec.values())
        cheap = [(int(g), min(float(sp.gen_max_ramp_up[g]), float(sp.gen_pmax[g] - p[g]))) for g in np.where(rd & running)[0]
                 if gc[g] <= price_wo + 1e-6 and int(g) not in cands and p[g] > 1.0]
        cheap = [(g, room) for g, room in cheap if room > 0.5]
        tot = sum(room for _, room in cheap)
        if tot < need - 1e-6 or not cheap:
            return action
        for g, room in cheap:
            vec[g] = vec.get(g, 0.0) + need * room / tot
        md = {int(g): float(v) for g, v in enumerate(np.asarray(action.redispatch, dtype=float)) if abs(v) > 0.05}
        for g, v in vec.items():
            md[g] = md.get(g, 0.0) + v
        cand = deepcopy(action)
        cand.redispatch = [(int(g), float(v)) for g, v in md.items() if abs(v) > 1e-3]
        try:
            o1, _, d1, inf1 = obs.simulate(cand, time_step=1)
        except Exception:  # noqa: BLE001
            return action
        if d1 or inf1.get("is_illegal") or inf1.get("is_ambiguous") or float(np.asarray(o1.rho).max()) > cap + 1e-6:
            return action
        return cand

    def _unwind_redispatch(self, obs, base_action):
        """One ramp step of the carried (actual) dispatch toward zero on every unit carrying more than 0.05 MW.

        Requesting the opposite of the actual value pulls it down even when the target is already zero (the dispatch
        solver keeps an actual dispatch alive otherwise). Units at a bound are skipped: a request there only piles up an
        unreachable target. Accepted if the next-step peak stays at or below the redispatch target."""
        ad = np.asarray(obs.actual_dispatch, dtype=float)
        if not np.any(np.abs(ad) > 0.05):
            return base_action
        sp = self.action_space
        p = np.asarray(obs.gen_p, dtype=float)
        vec = []
        for g in np.where(np.abs(ad) > 0.05)[0]:
            if ad[g] > 0:
                room = float(p[g] - sp.gen_pmin[g]); step = float(sp.gen_max_ramp_down[g])
            else:
                room = float(sp.gen_pmax[g] - p[g]); step = float(sp.gen_max_ramp_up[g])
            if room < 0.5:
                continue
            vec.append((int(g), -np.sign(ad[g]) * min(step, abs(ad[g]), room)))
        if not vec:
            return base_action
        cand = self._with_redispatch(base_action, vec)
        if self._now_ok(obs, cand, self.redisp_target):
            return cand
        return base_action
