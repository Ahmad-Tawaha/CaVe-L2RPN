"""Identification of the real grid's line parameters, the training stage, and the online drift guard.

The controller's simulator uses the *published* grid. The real grid may differ from it. CaVe models the difference as
one reactance ratio per line and identifies it from data the agent can see:

**Per-line estimator.** From one real observation, each line in service with a non-negligible flow gives its series
impedance through the AC pi model (per phase, published half line-charging b/2):

    Z = (V_or - V_ex) / (I_or - j b/2 V_or),    I_or = conj(S_or / (3 V_or)).

Im(Z) estimates the line's reactance; the ratio to the published reactance is the quantity the controller needs.

**Training stage** (:func:`run_training_stage`, called by ``CaVeAgent.train``). The agent, as shipped, plays the
practice weeks on the real environment in parallel workers:

* *phase A* -- one estimate per line every ``sample_every`` steps; per episode the median and the relative spread
  (IQR / median); pooled over episodes, the median ratio per line and the median spread. The identified ratio is
  shrunk toward 1 by reliability: log r * s0^2 / (s0^2 + spread^2). These ratios define the identified model used by
  the verifier;
* *phase B* -- the same weeks and seeds with the identified reactances written into the main simulator (the whole
  planner uses them); a paired test decides whether this *correction* is switched on;
* *phase C* -- the same weeks and seeds with the relaxed calm revert switched off; a paired test may switch it off.

A paired test switches only if the variant completes clearly more weeks: net = rescued - lost >= ``min_net``,
rescued >= ``min_ratio`` * max(lost, 1), at least ``min_paired`` paired weeks, and net >= ``sign_z`` * sqrt(rescued +
lost) (a one-sided sign test on the weeks that changed outcome). The stage also records which lines the opponent
attacked and which it declares in its configuration, so that the contingency set can be extended.

**Drift guard** (:class:`DriftGuard`). During deployment the same estimator runs on every observation. Once enough
lines are reliably estimated, the online ratios are compared with the identified ones once per simulated day; if the
spread of their log-ratio exceeds ``drift_tau``, the agent stops correcting its main simulator and points the verifier
at the online estimate. Without drift it does nothing.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import CaVeConfig, TrainingConfig


# ==================================================================================================== parameters
def load_line_ratios(path) -> Optional[Dict[int, float]]:
    """Read per-line reactance ratios (real / published) from JSON: ``{"x_ratio": {"<line>": ratio}}`` or a flat map."""
    try:
        d = json.loads(Path(path).read_text())
        d = d.get("x_ratio", d)
        return {int(k): float(v) for k, v in d.items()}
    except Exception:  # noqa: BLE001
        return None


def forecast_line_params(env_path) -> dict:
    """Series reactance, resistance and half line-charging of every line of the agent's simulator grid
    (``grid_forecast.json``). If there is no separate simulator grid, the simulator is the real grid and there is
    nothing to estimate (``present`` False)."""
    try:
        p = Path(env_path) / "grid_forecast.json"
        if not p.exists():
            return {"present": False}
        import pandapower as pp
        ln = pp.from_json(str(p)).line
        return {"present": True, "n_line": int(len(ln)),
                "x": (ln.x_ohm_per_km * ln.length_km / ln.parallel).to_numpy(dtype=float),
                "r": (ln.r_ohm_per_km * ln.length_km / ln.parallel).to_numpy(dtype=float),
                "bh": (2 * np.pi * 50.0 * ln.c_nf_per_km * 1e-9 * ln.length_km * ln.parallel / 2).to_numpy(dtype=float)}
    except Exception as e:  # noqa: BLE001
        return {"present": None, "error": f"{type(e).__name__}: {str(e)[:120]}"}


def estimate_series_impedance(obs, fc: dict, min_mw: float = 5.0) -> Dict[int, tuple]:
    """{line: (r, x)} in ohm from one real observation (AC pi model, see the module docstring)."""
    out = {}
    for l in range(int(fc.get("n_line", 0))):
        try:
            if not obs.line_status[l] or abs(obs.p_or[l]) < min_mw:
                continue
            vo = obs.v_or[l] * 1e3 / np.sqrt(3) * np.exp(1j * np.deg2rad(obs.theta_or[l]))
            ve = obs.v_ex[l] * 1e3 / np.sqrt(3) * np.exp(1j * np.deg2rad(obs.theta_ex[l]))
            io = np.conj((obs.p_or[l] + 1j * obs.q_or[l]) * 1e6 / (3 * vo))
            z = (vo - ve) / (io - 1j * fc["bh"][l] * vo)
            if np.isfinite(z.real) and np.isfinite(z.imag) and z.imag > 0:
                out[l] = (float(z.real), float(z.imag))
        except Exception:  # noqa: BLE001
            continue
    return out


# ==================================================================================================== drift guard
class DriftGuard:
    """Online re-estimation of the reactance ratios during deployment (see the module docstring)."""

    def __init__(self, config: CaVeConfig, env_paths):
        self.cfg = config
        self.env_paths = [p for p in env_paths if p]
        self.fc = None
        self.samples: Dict[int, deque] = {}
        self.steps = 0
        self.detected = False
        self.value = None
        self.arm()

    def arm(self) -> None:
        """Load the simulator grid's line parameters from the first environment path that has them (retried later
        if not found)."""
        if self.fc is not None:
            return
        for path in self.env_paths:
            try:
                fc = forecast_line_params(path)
            except Exception:  # noqa: BLE001
                continue
            if fc.get("present"):
                self.fc = fc
                break

    @property
    def armed(self) -> bool:
        return self.fc is not None

    def observe(self, obs, reference_ratios: Optional[Dict[int, float]]):
        """Add this observation's estimates; on a check step return the online ratios if drift is established
        (only once), else None. ``reference_ratios``: the identified ratios (None: compare with the published grid)."""
        cfg = self.cfg
        self.steps += 1
        for l, (_r, x) in estimate_series_impedance(obs, self.fc).items():
            d = self.samples.get(l)
            if d is None:
                d = self.samples[l] = deque(maxlen=cfg.drift_window)
            d.append(x)
        if self.steps % cfg.drift_every:
            return None
        fx = self.fc["x"]
        online = {}
        for l, d in self.samples.items():
            if len(d) >= cfg.drift_min_samples and 0 <= l < len(fx) and fx[l] > 0:
                arr = np.asarray(d, dtype=float)
                med = float(np.median(arr))
                spread = float(np.median(np.abs(arr - med)) / max(med, 1e-9))
                if spread <= cfg.drift_max_spread and med > 0:
                    online[int(l)] = med / float(fx[l])
        if len(online) < cfg.drift_min_lines:
            return None
        train = reference_ratios
        common = [l for l in online if train and l in train and train[l] > 0] if train else []
        if train and len(common) >= cfg.drift_min_lines:
            value = float(np.std(np.log(np.array([online[l] / train[l] for l in common]))))
        else:
            value = float(np.std(np.log(np.array(list(online.values())))))
        self.value = round(value, 4)
        if self.detected or value <= cfg.drift_tau:
            return None
        self.detected = True
        return online


# ==================================================================================================== training stage
_W: dict = {}


def _worker_init(env_path, x_ratios=None, calm_revert=True, agent_config=None):
    import warnings
    warnings.filterwarnings("ignore")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass
    import grid2op
    try:
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
    except Exception:  # noqa: BLE001
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()
    env = grid2op.make(env_path, backend=backend)
    from .agent import CaVeAgent
    agent = CaVeAgent(env.action_space, env, config=agent_config or CaVeConfig())
    agent.controller.calm_revert = bool(calm_revert)
    _W.update(env=env, agent=agent, fc=forecast_line_params(env_path),
              names=[str(x).rstrip("/").split("/")[-1] for x in env.chronics_handler.real_data.subpaths])
    if x_ratios:                       # phase B: this worker's main simulator carries the identified reactances
        from .simulation import scale_line_reactances
        scale_line_reactances(env.reset(), x_ratios)


def _run_episode(task):
    """Play one practice week; record its outcome, the attacked lines and the per-line reactance estimates."""
    idx, seed, deadline, sample_every, grace_s = task
    rec = {"chronic": None, "seed": int(seed), "steps": 0, "done": False, "skipped": False}
    if time.time() > deadline:
        rec["skipped"] = True
        return rec
    env, agent = _W["env"], _W["agent"]
    rec["chronic"] = _W["names"][idx]
    try:
        env.set_id(int(idx))
        env.seed(int(seed))
        obs = env.reset()
        agent.reset(obs)
        rec["max_steps"] = int(env.max_episode_duration())
        r, done, t = 0.0, False, 0
        xest, attacked, prev = {}, {}, set()
        while not done:
            if time.time() > deadline + grace_s:
                rec["cut_by_deadline"] = True
                break
            act = agent.act(obs, r, done)
            if t % sample_every == 0 and _W["fc"].get("present"):
                try:
                    for l, (_rr, xx) in estimate_series_impedance(obs, _W["fc"]).items():
                        xest.setdefault(l, []).append(xx)
                except Exception:  # noqa: BLE001
                    pass
            obs, r, done, info = env.step(act)
            t += 1
            try:
                oal = info.get("opponent_attack_line")
                cur = set(int(i) for i in np.where(np.asarray(oal))[0]) if oal is not None else set()
                for l in cur - prev:
                    attacked[l] = attacked.get(l, 0) + 1
                prev = cur
            except Exception:  # noqa: BLE001
                pass
        rec["steps"] = int(t)
        rec["done"] = bool(done)
        rec["attacked_lines"] = {str(k): v for k, v in attacked.items()}
        # per line: [median x, number of estimates, relative spread (IQR / median)], stored at fixed precision
        rec["x_est"] = {str(l): [round(float(np.median(v)), 6), len(v),
                                 round(float((np.percentile(v, 75) - np.percentile(v, 25)) / max(abs(np.median(v)), 1e-9)), 4)]
                        for l, v in xest.items()}
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return rec


def rollouts(env, n_workers: int, budget_s: float, tcfg: TrainingConfig, x_ratios=None, calm_revert=True,
             agent_config: Optional[CaVeConfig] = None) -> dict:
    """Play every practice week once (seed ``tcfg.seed``) in ``n_workers`` processes within ``budget_s`` seconds."""
    t0 = time.time()
    deadline = t0 + budget_s
    out = {"episodes": [], "errors": []}
    try:
        env_path = str(env.get_path_env())
        n = len(env.chronics_handler.real_data.subpaths)
        if tcfg.max_chronics:
            n = min(n, int(tcfg.max_chronics))
        tasks = [(i, tcfg.seed, deadline, tcfg.sample_every, tcfg.episode_grace_s) for i in range(n)]
        with mp.get_context("fork").Pool(processes=n_workers, initializer=_worker_init,
                                         initargs=(env_path, x_ratios, calm_revert, agent_config)) as pool:
            for rec in pool.imap_unordered(_run_episode, tasks, chunksize=1):
                if not rec.get("skipped"):
                    out["episodes"].append(rec)
                if time.time() > deadline:
                    pool.terminate()
                    break
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"{type(e).__name__}: {str(e)[:300]}\n{traceback.format_exc()[-800:]}")
    out["seconds"] = round(time.time() - t0, 1)
    return out


def identify_ratios(roll: dict, fc: dict, tcfg: TrainingConfig) -> Optional[dict]:
    """Pooled per-line ratio (median over episodes of the per-episode median x, divided by the published x) and
    spread (median over episodes), then reliability shrinkage. Returns {"ratios", "ratio", "spread"} or None."""
    if not fc or not fc.get("present"):
        return None
    per = {}
    for e in roll.get("episodes", []):
        for l, vals in (e.get("x_est") or {}).items():
            per.setdefault(int(l), []).append(vals)
    lines = sorted(l for l, v in per.items() if sum(x[1] for x in v) >= tcfg.min_samples)
    if not lines:
        return None
    x_est = np.array([np.median([x[0] for x in per[l]]) for l in lines])
    spread = np.array([np.nanmedian([x[2] for x in per[l]]) for l in lines])
    ratio = x_est / fc["x"][lines]
    ratio_r = {int(l): round(float(r), 5) for l, r in zip(lines, ratio)}
    spread_r = {int(l): (None if not np.isfinite(sp) else round(float(sp), 5)) for l, sp in zip(lines, spread)}
    s0 = tcfg.shrink_s0
    shrunk = {}
    for l in lines:
        r = ratio_r[l]
        sp = spread_r[l] if spread_r[l] is not None else 1.0
        w = s0 * s0 / (s0 * s0 + (sp if np.isfinite(sp) else 1.0) ** 2)
        shrunk[int(l)] = float(np.exp(w * np.log(max(r, 1e-6))))
    return {"ratios": shrunk, "ratio": ratio_r, "spread": spread_r}


def paired_decision(roll_a: dict, roll_b: dict, tcfg: TrainingConfig) -> dict:
    """Paired test of variant B against A on the same weeks and seeds (rule in the module docstring)."""
    def key(e):
        return (e.get("chronic"), e.get("seed"))
    A = {key(e): e for e in roll_a.get("episodes", []) if "error" not in e and e.get("max_steps")}
    B = {key(e): e for e in roll_b.get("episodes", []) if "error" not in e and e.get("max_steps")}
    common = [k for k in A if k in B]

    def completed(e):
        return e["steps"] >= e["max_steps"]
    rescued = sum(1 for k in common if completed(B[k]) and not completed(A[k]))
    lost = sum(1 for k in common if completed(A[k]) and not completed(B[k]))
    net = rescued - lost
    sign_test = bool(net >= tcfg.sign_z * float(np.sqrt(max(rescued + lost, 1))))
    switch = bool(len(common) >= tcfg.min_paired and net >= tcfg.min_net and rescued >= tcfg.min_ratio * max(lost, 1) and sign_test)
    return {"switch": switch, "n_paired": len(common), "rescued": rescued, "lost": lost, "net": net, "sign_test": sign_test}


def opponent_lines(env) -> list:
    """Line ids the environment's opponent declares as attackable."""
    try:
        ko = getattr(env, "_kwargs_opponent", None) or {}
        names = ko.get("lines_attacked")
        flat = [n for g in names for n in (g if isinstance(g, (list, tuple)) else [g])] if names else []
        idx = []
        for n in flat:
            w = np.where(np.asarray(env.name_line) == n)[0]
            if len(w):
                idx.append(int(w[0]))
        return idx
    except Exception:  # noqa: BLE001
        return []


def _n_workers(num_workers, cap: Optional[int]) -> int:
    try:
        nw = abs(int(num_workers)) if num_workers else 1
    except Exception:  # noqa: BLE001
        nw = 1
    nw = max(1, min(nw, os.cpu_count() or 1))
    try:
        with open("/proc/meminfo") as fh:
            mem = {l.split(":")[0]: int(l.split()[1]) for l in fh if ":" in l}
        nw = max(1, min(nw, int(mem.get("MemAvailable", mem.get("MemTotal", 0)) / 1e6 / 1.5)))   # ~1.5 GB per worker
    except Exception:  # noqa: BLE001
        pass
    if cap:
        nw = max(1, min(nw, int(cap)))
    return nw


def run_training_stage(env, num_workers, max_training_time, tcfg: TrainingConfig = TrainingConfig(),
                       agent_config: Optional[CaVeConfig] = None) -> dict:
    """Phases A, B, C (see the module docstring). Every failure is recorded; the stage always returns on time."""
    t0 = time.time()
    result = {"identification": None, "correction": {"skipped": True}, "calm_revert": {"skipped": True},
              "attacked_lines": {}, "declared_lines": opponent_lines(env), "errors": []}
    nw = _n_workers(num_workers, tcfg.max_workers)
    cap = 0.9 * float(max_training_time) if max_training_time and float(max_training_time) > 0 else 3600.0
    budget_s = float(min(tcfg.budget_s, max(60.0, cap - 300.0)))
    try:
        fc = forecast_line_params(str(env.get_path_env()))
        t_a = time.time()
        roll_a = rollouts(env, nw, tcfg.phase_a_share * budget_s, tcfg, agent_config=agent_config)
        result["errors"] += roll_a["errors"]
        for e in roll_a["episodes"]:
            if "error" in e or not e.get("max_steps"):
                continue
            for l, c in (e.get("attacked_lines") or {}).items():
                result["attacked_lines"][str(l)] = result["attacked_lines"].get(str(l), 0) + int(c)
        ident = identify_ratios(roll_a, fc, tcfg)
        result["identification"] = ident
        if ident is not None:
            rest_b = 0.5 * (budget_s - (time.time() - t_a) - tcfg.phase_b_reserve_s)
            if rest_b > tcfg.min_phase_s:
                roll_b = rollouts(env, nw, rest_b, tcfg, x_ratios=ident["ratios"], agent_config=agent_config)
                result["correction"] = paired_decision(roll_a, roll_b, tcfg)
        rest_c = budget_s - (time.time() - t_a) - tcfg.phase_c_reserve_s
        if rest_c > tcfg.min_phase_s:
            roll_c = rollouts(env, nw, rest_c, tcfg, calm_revert=False, agent_config=agent_config)
            result["calm_revert"] = paired_decision(roll_a, roll_c, tcfg)
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"{type(e).__name__}: {str(e)[:200]}")
    result["seconds"] = round(time.time() - t0, 1)
    return result
