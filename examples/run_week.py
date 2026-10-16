"""Play one week with CaVe and report its outcome.

    python examples/run_week.py --env <grid2op env dir or name> --scenario 2035-06-04_7 [--ratios ratios.json]
    python examples/run_week.py --env <env> --scenario 2035-06-04_7 --state-dir state/

``--ratios``: identified per-line reactance ratios (real / published), JSON ``{"x_ratio": {"<line>": ratio}}`` or a
flat map. ``--state-dir``: a training state written by ``CaVeAgent.save_state`` (the usual way to get the ratios).
Without either, the verifier stays off until the drift guard has identified the parameters online.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratios", default=None)
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0: the whole week)")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import grid2op
    from lightsim2grid import LightSimBackend
    from cave import CaVeAgent
    from cave.identify import load_line_ratios

    env = grid2op.make(args.env, backend=LightSimBackend())
    names = [os.path.basename(str(p).rstrip("/")) for p in env.chronics_handler.real_data.subpaths]
    env.set_id(names.index(args.scenario))
    env.seed(args.seed)
    obs = env.reset()
    ratios = load_line_ratios(args.ratios) if args.ratios else None
    agent = CaVeAgent(env.action_space, env, line_ratios=ratios)
    if args.state_dir:
        agent.load_state(args.state_dir)
    agent.reset(obs)

    done, reward, t, t0 = False, 0.0, 0, time.time()
    while not done and not (args.max_steps and t >= args.max_steps):
        obs, reward, done, _info = env.step(agent.act(obs, reward, done))
        t += 1
    total = env.max_episode_duration()
    status = "completed" if t >= total else ("stopped at --max-steps" if not done else "episode ended (game over)")
    print(f"{args.scenario}: {t}/{total} steps ({status}), {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
