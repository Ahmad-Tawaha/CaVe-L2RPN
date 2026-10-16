"""Fast checks that need no grid2op environment: weights, risk aggregation, pool keys, the line-parameter estimator,
the identification with reliability shrinkage, and the paired decision rule of the training stage.

    python tests/test_units.py      (or: pytest tests)
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_weights_load():
    from cave.ranker import ActionRanker
    r = ActionRanker()
    assert r.actions.shape == (1036, 1605)
    n_params = sum(p.numel() for p in r.model.parameters())
    assert n_params > 1_000_000
    assert not r.model.training


def test_cvar():
    from cave.controller import Controller
    me = SimpleNamespace(cvar_alpha=0.5)
    assert Controller._cvar(me, np.array([0.1, 0.2, 0.3, 0.4])) == 0.35          # mean of the worst half
    assert Controller._cvar(me, np.array([0.1, 0.2, 0.3, 0.5, 0.9])) == np.mean([0.3, 0.5, 0.9])  # ceil(2.5) = 3
    assert Controller._cvar(me, np.array([0.1, np.inf, 0.3, 0.4])) == np.inf     # fatal case in the tail
    me.cvar_alpha = 0.25
    assert Controller._cvar(me, np.array([np.inf, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])) == np.inf


def test_library_keys():
    from cave.actions import library_index, library_key
    assert library_index(library_key(17)) == 17
    assert sorted(["noop", library_key(12), library_key(5)])[0] == "noop"


def _pi_model_observation(z, bh, vo, ve):
    """Observation of one line with series impedance z (ohm), half line-charging bh (S) and per-phase terminal
    voltages vo, ve (V, complex), in the units grid2op reports (kV line-to-line, degrees, MW, MVAr)."""
    i_or = (vo - ve) / z + 1j * bh * vo
    s_or = 3 * vo * np.conj(i_or)
    return SimpleNamespace(line_status=np.array([True]), p_or=np.array([s_or.real / 1e6]), q_or=np.array([s_or.imag / 1e6]),
                           v_or=np.array([abs(vo) * np.sqrt(3) / 1e3]), v_ex=np.array([abs(ve) * np.sqrt(3) / 1e3]),
                           theta_or=np.array([np.degrees(np.angle(vo))]), theta_ex=np.array([np.degrees(np.angle(ve))]))


def test_estimator_recovers_impedance():
    from cave.identify import estimate_series_impedance
    z, bh = 2.5 + 31.0j, 1.2e-4
    vo = 138e3 / np.sqrt(3) * np.exp(1j * np.radians(4.0))
    ve = 136e3 / np.sqrt(3) * np.exp(1j * np.radians(-3.0))
    obs = _pi_model_observation(z, bh, vo, ve)
    est = estimate_series_impedance(obs, {"n_line": 1, "bh": np.array([bh])})
    r, x = est[0]
    assert abs(r - z.real) < 1e-6 * abs(z) and abs(x - z.imag) < 1e-6 * abs(z)


def test_identification_shrinkage():
    from cave.config import TrainingConfig
    from cave.identify import identify_ratios
    tcfg = TrainingConfig()
    fc = {"present": True, "x": np.array([10.0, 10.0])}
    roll = {"episodes": [{"x_est": {"0": [12.0, 10, 0.0], "1": [12.0, 10, 0.5]}}]}
    ident = identify_ratios(roll, fc, tcfg)
    assert abs(ident["ratio"][0] - 1.2) < 1e-12 and abs(ident["ratio"][1] - 1.2) < 1e-12
    assert abs(ident["ratios"][0] - 1.2) < 1e-12                       # reliable line: kept
    w = tcfg.shrink_s0 ** 2 / (tcfg.shrink_s0 ** 2 + 0.5 ** 2)
    assert abs(ident["ratios"][1] - 1.2 ** w) < 1e-12                 # noisy line: shrunk toward 1


def test_paired_decision():
    from cave.config import TrainingConfig
    from cave.identify import paired_decision
    tcfg = TrainingConfig()

    def roll(completed):
        return {"episodes": [{"chronic": f"w{i}", "seed": 1, "steps": 10 if c else 5, "max_steps": 10}
                             for i, c in enumerate(completed)]}
    base = [False] * 10 + [True] * 40
    better = [True] * 10 + [True] * 40                                  # 10 rescued, 0 lost
    assert paired_decision(roll(base), roll(better), tcfg)["switch"]
    mixed = [True] * 6 + [False] * 4 + [False] * 4 + [True] * 36       # 6 rescued, 4 lost
    assert not paired_decision(roll(base), roll(mixed), tcfg)["switch"]
    assert not paired_decision(roll(base[:30]), roll(better[:30]), tcfg)["switch"]   # too few paired weeks


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
