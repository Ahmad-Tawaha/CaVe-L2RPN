"""Structural fragility and calm-grid topology reverts.

Splitting a substation into two buses can turn an attackable line into a *bridge* of the bus graph: a line whose loss
alone splits the grid into islands. The base topology has no attackable bridge; every such state is created by the
controller's own splits. Two rules use the count of attackable bridges (a pure graph quantity, no power flow):

* at a crisis decision, each candidate's risk is increased by ``fragility_lambda`` per attackable bridge it adds
  (decreased per bridge it removes), see :meth:`cave.controller.Controller.decide_crisis`;
* between crises, if the current topology has attackable bridges, the controller prefers the substation revert (all
  elements back to bus 1) that removes most of them, if it is legal and keeps the next-step peak at or below
  ``fragility_revert_rho``.

When there is no fragile structure to repair and the grid is calm, the controller reverts substations toward the base
topology: the *relaxed* revert takes the best single-substation revert whose next-step peak stays at or below
``calm_cap`` (no improvement required) when the peak is below ``calm_gate``; otherwise the classic revert of
:func:`cave.actions.revert_topo` applies.
"""
from __future__ import annotations

import numpy as np

from .actions import is_legal, simulate_rho


class FragilityMixin:
    """Methods of :class:`cave.controller.Controller` for bridges and calm-grid reverts."""

    def bridge_count(self, topo, line_status, extra_off=()) -> int:
        """Number of attackable lines in service whose loss alone increases the number of connected components of the
        bus graph (nodes = (substation, bus) carrying any element; edges = lines in service with both ends connected)."""
        sp = self.action_space
        topo = np.asarray(topo)
        ls = np.asarray(line_status, dtype=bool)
        ck = (topo.astype(np.int8).tobytes(), ls.tobytes(), tuple(sorted(extra_off)))
        hit = self._bridge_cache.get(ck)
        if hit is not None:
            return hit
        if len(self._bridge_cache) > 512:
            self._bridge_cache.clear()

        def components(removed):
            parent = {}

            def find(x):
                while parent.get(x, x) != x:
                    x = parent[x]
                return x
            nodes = set()
            for pos, sub in ((sp.load_pos_topo_vect, sp.load_to_subid), (sp.gen_pos_topo_vect, sp.gen_to_subid)):
                for p_, s_ in zip(pos, sub):
                    b = int(topo[p_])
                    if b > 0:
                        nodes.add((int(s_), b))
            for l in range(sp.n_line):
                if not ls[l] or l in removed:
                    continue
                bo, be = int(topo[sp.line_or_pos_topo_vect[l]]), int(topo[sp.line_ex_pos_topo_vect[l]])
                if bo <= 0 or be <= 0:
                    continue
                a, b = (int(sp.line_or_to_subid[l]), bo), (int(sp.line_ex_to_subid[l]), be)
                nodes.add(a); nodes.add(b)
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
            return len({find(x) for x in nodes})

        base = components(set(extra_off))
        attackable = [l for l in self.lines_to_attack if ls[l] and l not in extra_off]
        n = len([l for l in attackable if components({l} | set(extra_off)) > base])
        self._bridge_cache[ck] = n
        return n

    def _topo_after(self, obs, vect):
        """topo_vect after applying the bus assignments of an action (vector or action object)."""
        act = self.action_space.from_vect(vect) if not hasattr(vect, "set_bus") else vect
        sb = np.asarray(act.set_bus)
        topo = np.asarray(obs.topo_vect).copy()
        m = sb != 0
        topo[m] = sb[m]
        return topo

    def _status_after(self, obs, vect):
        """Line status after the action's own reconnections / disconnections."""
        ls = np.asarray(obs.line_status, dtype=bool).copy()
        try:
            act = self.action_space.from_vect(vect) if not hasattr(vect, "set_bus") else vect
            st = np.asarray(act.set_line_status)
            ls[st == 1] = True
            ls[st == -1] = False
            ch = getattr(act, "change_line_status", None)
            if ch is not None:
                ch = np.asarray(ch, dtype=bool)
                ls[ch] = ~ls[ch]
        except Exception:  # noqa: BLE001
            pass
        return ls

    def _fragility_revert(self, obs):
        """Substation revert that removes most attackable bridges (ties: lowest next-step peak). Returns a vector or None."""
        topo = np.asarray(obs.topo_vect)
        ls = np.asarray(obs.line_status, dtype=bool)
        f_now = self.bridge_count(topo, ls)
        if f_now == 0:
            return None
        sp = self.action_space
        best = None
        for sub_id in range(obs.n_sub):
            st = obs.sub_topology(sub_id)
            if not np.any(st == 2) or obs.time_before_cooldown_sub[sub_id] > 0:
                continue
            act = sp({"set_bus": {"substations_id": [(sub_id, np.ones(len(st), dtype=int))]}})
            f_after = self.bridge_count(self._topo_after(obs, act), ls)
            if f_after >= f_now:
                continue
            try:
                o, _, d, inf = obs.simulate(act, time_step=1)
            except Exception:  # noqa: BLE001
                continue
            if d or inf.get("is_illegal") or inf.get("is_ambiguous") or not is_legal(act, obs):
                continue
            r = float(o.rho.max())
            if r > self.cfg.fragility_revert_rho:
                continue
            key = (f_after, r)
            if best is None or key < best[0]:
                best = (key, act)
        if best is None:
            return None
        return best[1].to_vect()

    def _revert_topo_relaxed(self, obs):
        """Relaxed calm revert: best single-substation revert to bus 1 whose next-step peak stays <= calm_cap."""
        if float(obs.rho.max()) >= self.cfg.calm_gate or not np.any(np.asarray(obs.topo_vect) == 2) or not np.all(obs.line_status):
            return None
        best = None
        for sub_id in range(obs.n_sub):
            st = obs.sub_topology(sub_id)
            if not np.any(st == 2) or obs.time_before_cooldown_sub[sub_id] > 0:
                continue
            vect = self.action_space({"set_bus": {"substations_id": [(sub_id, np.ones(len(st), dtype=int))]}}).to_vect()
            rho_after, ok = simulate_rho(self.action_space, obs, vect)
            if not ok or not np.isfinite(rho_after) or rho_after > self.cfg.calm_cap:
                continue
            if best is None or rho_after < best[0]:
                best = (float(rho_after), vect)
        if best is None:
            return None
        return best[1]
