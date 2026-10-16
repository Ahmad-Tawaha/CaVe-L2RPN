"""Smoke test on a real environment: the agent plays the first steps of one week without error.

Set ``CAVE_TEST_ENV`` to a local grid2op environment (directory or name, e.g. ``l2rpn_idf_2023``); the test is skipped
otherwise. ``CAVE_TEST_STEPS`` sets the number of steps (default 50).

    CAVE_TEST_ENV=l2rpn_idf_2023 python tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(env_name: str, n_steps: int) -> int:
    import warnings
    warnings.filterwarnings("ignore")
    import grid2op
    from lightsim2grid import LightSimBackend
    from grid2op.Action import BaseAction
    from cave import make_agent

    env = grid2op.make(env_name, backend=LightSimBackend())
    env.seed(0)
    obs = env.reset()
    agent = make_agent(env)
    agent.reset(obs)
    done, reward, t = False, 0.0, 0
    while not done and t < n_steps:
        action = agent.act(obs, reward, done)
        assert isinstance(action, BaseAction)
        obs, reward, done, info = env.step(action)
        assert not info.get("is_illegal"), f"illegal action at step {t}"
        t += 1
    return t


def test_smoke():
    env_name = os.environ.get("CAVE_TEST_ENV")
    if not env_name:
        import pytest
        pytest.skip("CAVE_TEST_ENV not set")
    assert run(env_name, int(os.environ.get("CAVE_TEST_STEPS", "50"))) > 0


if __name__ == "__main__":
    name = os.environ.get("CAVE_TEST_ENV")
    if not name:
        print("SKIP: CAVE_TEST_ENV not set")
        sys.exit(0)
    print("steps played:", run(name, int(os.environ.get("CAVE_TEST_STEPS", "50"))))
