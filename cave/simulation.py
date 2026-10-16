"""Simulator-level helpers: the blind-forecast fallback and in-place reactance scaling of a simulator backend.

Blind-forecast fallback
    ``obs.simulate(action)`` predicts the next step from the *forecast* injections. Occasionally the forecast itself is
    broken for a step (the forecast redispatch is infeasible or the forecast AC power flow diverges while the real grid
    is fine): every action, including do-nothing, is then reported as fatal and the controller would choose blindly.
    When a one-step simulation reports game over and the do-nothing one-step simulation of the same observation also
    does, the call is redone as a same-state simulation (``time_step=0``). Explicit horizons (``time_step > 1``) and
    ordinary steps are untouched. The wrapper is bound to each observation the controller receives; copies of that
    observation (``obs.copy()``) carry it along, bound to the copy.

Reactance scaling
    The identified model is the published grid with each line's series reactance multiplied by its identified ratio
    (real / published). :func:`scale_line_reactances` writes absolute values x_l = x_l^published * ratio_l into the
    pandapower network behind a simulator backend and re-initialises the backend. The published values are captured
    the first time a network is seen (:func:`remember_published_reactances`), so repeated scaling never compounds.
"""
from __future__ import annotations

import types
from typing import Dict, Optional, Tuple

import numpy as np

PUBLIC_X_KEY = "_cave_x_published"


def make_blind_fallback(noop_action):
    """Return ``bind(obs)``: a ``simulate`` method for ``obs`` with the blind-forecast fallback.

    The do-nothing check is made once per observation and cached on it."""

    def _simulate(self_obs, action, time_step=1):
        base = type(self_obs).simulate
        res = base(self_obs, action, time_step)
        if time_step != 1 or not res[2]:
            return res
        blind = self_obs.__dict__.get("_cave_blind")
        if blind is None:
            try:
                blind = bool(base(self_obs, noop_action, 1)[2])
            except Exception:  # noqa: BLE001
                blind = False
            self_obs.__dict__["_cave_blind"] = blind
        if not blind:
            return res
        try:
            return base(self_obs, action, 0)
        except Exception:  # noqa: BLE001
            return res

    return lambda obs: types.MethodType(_simulate, obs)


def _pandapower_net(obs):
    oe = getattr(obs, "_obs_env", None)
    bk = getattr(oe, "backend", None)
    pp0 = getattr(bk, "_init_pp_backend", None)
    return bk, getattr(pp0, "_grid", None)


def remember_published_reactances(obs) -> None:
    """Store the published line reactances on the simulator's pandapower network (once, before any scaling)."""
    try:
        _, net = _pandapower_net(obs)
        if PUBLIC_X_KEY not in net:
            net[PUBLIC_X_KEY] = net.line["x_ohm_per_km"].to_numpy(dtype=float).copy()
    except Exception:  # noqa: BLE001
        pass


def scale_line_reactances(obs, x_ratio: Dict[int, float], ratio_min: float = 0.4,
                          ratio_max: float = 2.5) -> Tuple[int, Optional[str]]:
    """Set x_l = x_l^published * clip(ratio_l) on the simulator backend of ``obs`` (in place) and re-initialise it.

    Returns (number of scaled lines, error message or None). The pandapower network may be shared with other copies of
    the same simulator; values are always written from the stored published vector, never multiplied in place."""
    try:
        bk, net = _pandapower_net(obs)
        if net is None or not hasattr(net, "line"):
            return 0, "no pandapower network on the simulator backend"
        n = len(net.line)
        f = np.ones(n)
        for l, r in x_ratio.items():
            l = int(l)
            if 0 <= l < n and np.isfinite(r):
                f[l] = min(max(float(r), ratio_min), ratio_max)
        tl = np.array(bk.thermal_limit_a, dtype=float).copy() if getattr(bk, "thermal_limit_a", None) is not None else None
        if PUBLIC_X_KEY not in net or len(net[PUBLIC_X_KEY]) != n:
            net[PUBLIC_X_KEY] = net.line["x_ohm_per_km"].to_numpy(dtype=float).copy()
        net.line["x_ohm_per_km"] = np.asarray(net[PUBLIC_X_KEY], dtype=float) * f
        bk._aux_init_pandapower()
        if tl is not None:
            bk.set_thermal_limit(tl)
        return int((f != 1.0).sum()), None
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {str(e)[:120]}"
