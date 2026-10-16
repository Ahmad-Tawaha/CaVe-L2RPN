"""All constants of the CaVe controller in one place.

Every number used by a decision rule is a field of :class:`CaVeConfig`. The defaults are the configuration of the
L2RPN 2026 entry; change them only for experiments.

Loadings ("rho") are line currents divided by thermal limits, so 1.0 means a line at its limit. Times are in
environment steps (5 minutes each). Powers are in MW.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Lines the opponent of the L2RPN IDF environment may attack (grid2op power-line ids), grouped by area.
# Every crisis candidate is tested against the loss of each of them (the "N-1 test").
ATTACKABLE_LINES: Tuple[int, ...] = (
    106, 93, 88, 162, 68, 117,           # area 1
    180, 160, 136, 141, 131, 121, 125,   # area 2
    126, 110, 154,
    81, 43, 33, 37, 62, 61,              # area 3
)


@dataclass(frozen=True)
class CaVeConfig:
    # ---------------------------------------------------------------- crisis trigger and candidate pool
    rho_danger: float = 0.99
    """The controller acts on topology only when the peak loading reaches this value ("crisis")."""
    top_k: int = 10
    """Number of ranker proposals that are simulated at a crisis step."""
    max_ranked: int = 50
    """Length of the ranked list used by the fallback scan when the main decision path fails."""
    pool_size: int = 5
    """Best simulated proposals (by post-action peak loading) admitted to the candidate pool next to do-nothing."""
    worse_guard: float = 0.10
    """A proposal that leaves the grid overloaded (>1) and this much worse than do-nothing is kept out of the pool."""
    cvar_alpha: float = 0.5
    """N-1 risk = mean of the worst ceil(alpha * n) post-contingency peak loadings (CVaR); a fatal case in that tail is +inf."""
    tiebreak_eps: float = 0.02
    """Candidates whose risk is within this distance of the best one are separated by a short look-ahead."""
    beam_width: int = 5
    plan_depth: int = 3
    gamma: float = 0.95
    no_continuation_penalty: float = 2.0
    """Look-ahead: score = rho1 + gamma rho2 + gamma^2 rho3 over beam_width continuations; a missing level costs this value."""
    attackable_lines: Tuple[int, ...] = ATTACKABLE_LINES

    # ---------------------------------------------------------------- structural fragility
    fragility_lambda: float = 0.05
    """Risk penalty per attackable line that becomes a bridge of the bus graph (its loss alone would split the grid)."""
    fragility_revert_rho: float = 0.95
    """Between crises, a substation revert that removes attackable bridges is accepted if the next-step peak stays below this."""

    # ---------------------------------------------------------------- calm-grid topology revert
    revert_rho_limit: float = 0.8
    """Classic revert: back to bus 1 at one substation if that lowers the peak below min(rho, this value)."""
    calm_gate: float = 0.8
    calm_cap: float = 0.8
    """Relaxed revert: when the peak is below calm_gate, the best revert is taken if its peak stays at or below calm_cap."""

    # ---------------------------------------------------------------- attacked lines
    attack_mask_steps: int = 96
    """A line out of service and attacked less than this many steps ago is treated as still under attack."""

    # ---------------------------------------------------------------- planned outages (maintenance forecast)
    maint_horizon: int = 12
    """Outages starting within this many steps are anticipated."""
    maint_rho_trigger: float = 0.95
    """Anticipation starts when the forecast grid at the outage moment reaches this peak loading (or diverges)."""
    maint_margin: float = 0.02
    """A pre-outage topology action is taken only if it lowers the post-outage peak by at least this amount."""
    maint_revert_horizon: int = 24
    """No calm-grid topology revert while an outage is active or starts within this many steps."""
    forecast_horizon_max: int = 12
    """Longest forecast horizon used by obs.simulate."""

    # ---------------------------------------------------------------- redispatch
    redisp_target: float = 0.90
    """Redispatch ramps are added while the predicted peak is at or above this loading."""
    redisp_target_low_load: float = 0.97
    low_load_mw: float = 3000.0
    low_load_hysteresis_mw: float = 50.0
    """While total load is below low_load_mw (with hysteresis), the higher target applies: in light weeks the grid
    usually survives without help and every MW of redispatch is pure cost."""
    redisp_horizon: int = 12
    """Outages starting within this many steps receive a pre-outage redispatch ramp."""
    redisp_n_up: int = 6
    redisp_n_down: int = 8
    """Most effective units ramped up / down in one balanced ramp step."""
    redisp_max_mw: float = 400.0
    """Cap on the carried transfer (a large standing dispatch becomes infeasible when the schedule moves)."""
    unwind_rho: float = 0.90
    unwind_horizon: int = 24
    """Carried dispatch is ramped back toward zero below unwind_rho when no outage is active or within unwind_horizon."""
    unwind_calm_rho: float = 0.85
    """... and, even near an outage, below this clearly calm loading."""

    # ---------------------------------------------------------------- verification on the identified model
    verify_rollout_steps: int = 3
    """A checked action is followed by this many steps of the controller itself before survival is judged."""
    verify_alternatives: int = 4
    """Pool members tried, in risk order, when the chosen action is predicted to fail."""
    verify_wide_k: int = 10
    """If no pool member survives, the whole action library is scanned and the best ones by loading are tried."""
    verify_max_per_episode: int = 400
    """Upper bound on verified decisions per episode (keeps the decision time bounded)."""

    # ---------------------------------------------------------------- identification of the line parameters
    ratio_min: float = 0.4
    ratio_max: float = 2.5
    """Identified reactance ratios (real / published) are clipped to this range before use."""

    # ---------------------------------------------------------------- online drift guard
    drift_tau: float = 0.08
    """Drift is declared when std(log(online ratio / identified ratio)) over reliable lines exceeds this value."""
    drift_every: int = 288
    """One drift check per simulated day."""
    drift_min_samples: int = 24
    drift_min_lines: int = 80
    drift_window: int = 200
    drift_max_spread: float = 0.05
    """A line is reliable with at least drift_min_samples estimates (last drift_window kept) of relative spread <= this."""

    def with_overrides(self, **kw) -> "CaVeConfig":
        from dataclasses import replace
        return replace(self, **kw)


@dataclass(frozen=True)
class TrainingConfig:
    """Training stage: practice-week rollouts, identification and two paired tests (see :mod:`cave.identify`)."""
    budget_s: float = 10800.0
    """Wall-clock budget of the whole stage (capped further by the max_training_time passed to train())."""
    seed: int = 1
    """Environment seed of every practice-week episode (the same in all phases, so the phases are paired)."""
    max_chronics: int | None = None
    max_workers: int | None = None
    sample_every: int = 24
    """Line parameters are estimated from one real observation every sample_every steps."""
    min_samples: int = 5
    """A line needs at least this many estimates (pooled over episodes) to be identified."""
    shrink_s0: float = 0.02
    """Reliability shrinkage: log-ratio * s0^2 / (s0^2 + spread^2) with spread = IQR / median of the line's estimates."""
    min_net: int = 3
    min_ratio: float = 1.3
    min_paired: int = 40
    sign_z: float = 1.64
    """A paired test switches a mechanism only if net >= min_net, rescued >= min_ratio * max(lost, 1),
    at least min_paired paired weeks, and net >= sign_z * sqrt(rescued + lost) (one-sided sign test)."""
    phase_a_share: float = 0.34
    """Share of the budget given to phase A (identification rollouts)."""
    min_phase_s: float = 300.0
    """A paired phase runs only if at least this many seconds remain for it."""
    phase_b_reserve_s: float = 240.0
    phase_c_reserve_s: float = 120.0
    """Seconds kept free when the remaining budget is split between phase B (half of the rest) and phase C."""
    episode_grace_s: float = 120.0
    """An episode still running this long after the deadline is cut (and counts as not completed)."""
