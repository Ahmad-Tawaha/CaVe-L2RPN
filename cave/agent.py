"""CaVeAgent: the grid2op agent (``BaseAgent``) that wires the controller, the identified model and the drift guard.

Typical use::

    from cave import CaVeAgent
    agent = CaVeAgent(env.action_space, env, line_ratios=identified)   # or train() / load_state() first
    obs = env.reset(); agent.reset(obs)
    action = agent.act(obs, reward=0.0)

``line_ratios`` -- per-line reactance ratios (real / published) of the identified model -- normally come from the
training stage (:meth:`train`, then :meth:`save_state` / :meth:`load_state`). They can also be given directly. Without
them the verifier is inactive until the drift guard establishes its own online estimate.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from grid2op.Agent import BaseAgent

from .config import CaVeConfig, TrainingConfig
from .controller import Controller
from .identify import DriftGuard, run_training_stage
from .ranker import WEIGHTS_DIR, ActionRanker
from .simulation import make_blind_fallback, remember_published_reactances, scale_line_reactances

STATE_FILE = "cave_state.json"
STATE_SCHEMA = 1


def _line_fingerprint(action_space) -> Optional[str]:
    try:
        return hashlib.sha1("|".join(map(str, action_space.name_line)).encode()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


class CaVeAgent(BaseAgent):
    """Calibrated-verification grid controller."""

    def __init__(self, action_space, env=None, line_ratios: Optional[Dict[int, float]] = None,
                 config: CaVeConfig = CaVeConfig(), weights_dir=WEIGHTS_DIR, env_path: Optional[str] = None):
        BaseAgent.__init__(self, action_space=action_space)
        self.cfg = config
        self.controller = Controller(action_space, ActionRanker(weights_dir), config)
        self.line_ratios = {int(k): float(v) for k, v in line_ratios.items()} if line_ratios else None
        # correction of the main simulator (ratio ** exponent on every line); switched on only by the training stage
        self.correction_exponent = 0.0
        self._corrected_envs: Dict[int, float] = {}
        self._low_load = False
        self._bind_blind = None
        self._state: dict = {}
        paths = []
        if env is not None:
            try:
                paths.append(str(env.get_path_env()))
            except Exception:  # noqa: BLE001
                pass
        if env_path and env_path not in paths:
            paths.append(env_path)
        self.drift = DriftGuard(config, paths)
        self._drift_late_armed = False

    # ================================================================ per-step operating point
    def _update_operating_point(self, obs) -> None:
        """Redispatch target: the low-load value while total load is below the threshold (with hysteresis)."""
        cfg = self.cfg
        low = self._low_load
        try:
            load = float(np.asarray(obs.load_p, dtype=float).sum())
            h = cfg.low_load_hysteresis_mw
            low = (load < cfg.low_load_mw - h) if not low else (load < cfg.low_load_mw + h)
        except Exception:  # noqa: BLE001
            low = False
        self._low_load = low
        self.controller.redisp_target = float(cfg.redisp_target_low_load if low else cfg.redisp_target)

    def _ensure_corrected_simulator(self, obs) -> None:
        """When the correction is on, the main simulator carries x = x_published * ratio ** exponent (each simulator
        backend is rescaled once per exponent value)."""
        ratios = self.line_ratios
        if not ratios:
            return
        oe = getattr(obs, "_obs_env", None)
        if oe is None:
            return
        key = id(oe)
        if abs(self._corrected_envs.get(key, 0.0) - self.correction_exponent) < 1e-12:
            return
        self._corrected_envs[key] = self.correction_exponent
        scale_line_reactances(obs, {l: r ** self.correction_exponent for l, r in ratios.items() if r > 0},
                              self.cfg.ratio_min, self.cfg.ratio_max)
        self.controller.main_ratio_exponent = float(self.correction_exponent)
        self.controller.reset_identified_model()

    def _drift_step(self, obs) -> None:
        reference = self.line_ratios
        stored = (self._state.get("correction") or {}).get("ratios")
        if stored:
            reference = {int(k): float(v) for k, v in stored.items()}
        online = self.drift.observe(obs, reference)
        if online is None:
            return
        # drift established: back to the published simulator, verifier on the online estimate
        if self.correction_exponent != 0.0 or self._state.get("correction_applied"):
            self.correction_exponent = 0.0
            self._update_operating_point(obs)
            self._ensure_corrected_simulator(obs)
        self.controller.identified_ratios = dict(online)
        self.controller.reset_identified_model()

    # ================================================================ grid2op interface
    def reset(self, obs):
        remember_published_reactances(obs)
        try:
            self._update_operating_point(obs)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.drift.arm()
        except Exception:  # noqa: BLE001
            pass
        if self.controller.identified_ratios is None and self.line_ratios and self.correction_exponent == 0.0:
            self.controller.identified_ratios = dict(self.line_ratios)
        self.controller.reset()

    def act(self, observation, reward=0.0, done=False):
        try:
            self._update_operating_point(observation)
            self._ensure_corrected_simulator(observation)
        except Exception:  # noqa: BLE001
            pass
        if not self.drift.armed and not self._drift_late_armed:
            self._drift_late_armed = True
            self.drift.arm()
        if self.drift.armed:
            try:
                self._drift_step(observation)
            except Exception:  # noqa: BLE001
                pass
        try:
            if self._bind_blind is None:
                self._bind_blind = make_blind_fallback(self.action_space({}))
            observation.__dict__["simulate"] = self._bind_blind(observation)
        except Exception:  # noqa: BLE001
            pass
        return self.controller.step(observation)

    # ================================================================ training stage
    def train(self, env, num_workers=None, max_training_time=None, config: TrainingConfig = TrainingConfig()):
        """Identify the line parameters and run the paired tests on the practice weeks (:mod:`cave.identify`).
        The result is kept for :meth:`save_state`; nothing changes the agent's behaviour until :meth:`load_state`."""
        try:
            self._training_result = run_training_stage(env, num_workers, max_training_time, config, self.cfg)
        except Exception as e:  # noqa: BLE001
            self._training_result = {"errors": [f"{type(e).__name__}: {str(e)[:300]}"]}
        return None

    def save_state(self, path):
        """Write the training result (atomic)."""
        res = getattr(self, "_training_result", None) or {}
        ident = res.get("identification") or {}
        corr = res.get("correction") or {}
        state = {"schema": STATE_SCHEMA, "line_names_sha1": _line_fingerprint(self.action_space),
                 "identification": ident or None,
                 # the identified ratios are adopted only if the paired correction test could be run
                 "correction": (dict(corr, ratios={str(k): v for k, v in ident["ratios"].items()})
                                if ident and not corr.get("skipped") else None),
                 "calm_revert": res.get("calm_revert"),
                 "attacked_lines": res.get("attacked_lines") or {},
                 "declared_lines": res.get("declared_lines") or [],
                 "errors": res.get("errors") or []}
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p), prefix=STATE_FILE + ".", suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh, indent=1, default=str)
        os.replace(tmp, str(p / STATE_FILE))

    def load_state(self, path):
        """Apply a saved training result: identified model for the verifier, optional correction of the main simulator,
        contingency set extended by lines the opponent attacked or declares, optional switch-off of the calm revert."""
        p = Path(path) / STATE_FILE
        state = json.loads(p.read_text())
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError(f"state schema {state.get('schema')} != {STATE_SCHEMA}")
        mismatch = state.get("line_names_sha1") not in (None, _line_fingerprint(self.action_space))
        self._state = state
        corr = state.get("correction") or {}
        ratios = corr.get("ratios") or {}
        self.controller.identified_ratios = ({int(k): float(v) for k, v in ratios.items()} if (ratios and not mismatch) else None)
        if corr.get("switch") and ratios and not mismatch:
            self.line_ratios = {int(k): float(v) for k, v in ratios.items()}
            self.correction_exponent = 1.0
            self._corrected_envs = {}
            state["correction_applied"] = True
        # contingency set: add lines attacked in the practice weeks or declared by the opponent
        own = set(self.controller.lines_to_attack)
        observed = [int(l) for l in (state.get("attacked_lines") or {})]
        declared = [int(l) for l in (state.get("declared_lines") or [])]
        extra = sorted(set(l for l in observed + declared if l not in set(self.cfg.attackable_lines)))
        n_line = int(getattr(self.action_space, "n_line", 0) or 0)
        extra = [l for l in extra if 0 <= l < n_line and l not in own][:20]
        if extra:
            self.controller.lines_to_attack = list(self.controller.lines_to_attack) + extra
        if (state.get("calm_revert") or {}).get("switch"):
            self.controller.calm_revert = False


def make_agent(env, this_directory_path=None):
    """Factory with the signature used by the L2RPN evaluation (``make_agent(env, path)``)."""
    del this_directory_path
    return CaVeAgent(env.action_space, env)
