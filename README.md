# CaVe: Calibrated Verification for power-grid topology control

CaVe is an agent for the L2RPN IDF power grid (118 substations, 186 branches, an opponent that attacks lines) in the
[grid2op](https://github.com/rte-france/Grid2Op) framework. It keeps the grid running by switching substation
topologies, reconnecting lines and redispatching generators. Its planning simulator is only an approximation of the
real grid. This repository contains the agent submitted by team SAGE × ANL to the L2RPN 2026 (Sim2Real) competition.

## The method in plain words

1. **A simulator-trained ranker proposes actions.** A graph-attention network scores each of the 1,036 actions of a
   fixed library from the current observation. It was trained in simulation to imitate exhaustive search (soft-label
   imitation). It only proposes; every decision is made by simulation tests.
2. **A rule-based controller decides.** When the grid is calm, the controller does three things. It reverts topology
   changes toward the base topology. It repairs structurally fragile states (an attackable line whose loss alone would
   split the grid). It keeps the carried redispatch small. A crisis is a peak line loading of 0.99 or more. In a crisis
   the controller simulates the ranker's top proposals and forms a small pool with do-nothing. It scores each member
   by its N-1 risk: the mean of the worst half of the post-contingency peak loadings over the attackable-line outages,
   where any fatal case in that tail makes the risk infinite. It adds a penalty for new attackable bridges and
   separates near-ties by a short look-ahead.
3. **Planned outages are anticipated.** The maintenance forecast announces outages in advance. Some outages would leave
   the grid overloaded. Before such an outage, the controller pre-positions the topology (scanning the whole library on
   the post-outage grid) and ramps balanced redispatch.
4. **A short training stage identifies the real grid's line parameters.** The agent plays the practice weeks. From its
   observations it estimates each line's series reactance through the AC pi model. It expresses that estimate as a
   ratio to the published value. Two paired tests on the same weeks decide two switches: whether to also correct the
   main simulator, and whether to keep one of the calm-grid rules.
5. **A verify layer checks critical decisions on the identified model.** A second simulator instance carries the
   identified reactances. Every crisis decision is replayed there: the action, then three steps of the controller
   itself. The identified model may predict that the action ends the episode. Then other candidates are tried, and
   one replaces the action only if it survives on both models. If later stages changed the action, the final action
   is checked again.
6. **Redispatch rules cover what topology cannot.** They provide:
   * pre-outage ramps;
   * relief ramps in a crisis;
   * unwinding of carried dispatch;
   * removal of unreachable dispatch targets;
   * release of units that set the operating price only because of a residual dispatch.

   Each ramp follows single-unit sensitivities measured with `obs.simulate`.

During deployment a **drift guard** keeps estimating the line reactances from the observations. If they move away from
the identified ones, it switches off the main-simulator correction and points the verifier at the online estimate.

## Sim2Real

CaVe learns about the real grid only by interacting with it: in the training stage it plays the practice weeks and
estimates the line reactances from its own observations. Its planning simulator is the published grid
(`grid_forecast.json`).

## Install

Python 3.11 with:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the tested versions:

* grid2op 1.12.5 and lightsim2grid 0.13.1;
* torch 2.12 and torch_geometric 2.8;
* scikit-learn 1.9, pandapower 3.4 and numpy 2.3.

scikit-learn is required at run time, because the ranker's feature scaler is a pickled scikit-learn `StandardScaler`.

The environment is `l2rpn_idf_2023`. A test grid built from it also works, provided it has a separate simulator grid,
`grid_forecast.json`.

## Run

```python
import grid2op
from lightsim2grid import LightSimBackend
from cave import CaVeAgent

env = grid2op.make("l2rpn_idf_2023", backend=LightSimBackend())
agent = CaVeAgent(env.action_space, env)

# training stage: practice weeks in parallel workers, result kept on disk
agent.train(env, num_workers=8, max_training_time=24 * 3600)
agent.save_state("state/")
agent.load_state("state/")

obs = env.reset(); agent.reset(obs)
done, reward = False, 0.0
while not done:
    obs, reward, done, info = env.step(agent.act(obs, reward, done))
```

The L2RPN evaluation harness creates the agent with `cave.make_agent(env, path)`, then calls `train`, `save_state` and
`load_state`. `examples/run_week.py` plays one week and reports how far the agent got:

```bash
python examples/run_week.py --env l2rpn_idf_2023 --scenario 2035-06-04_7 --state-dir state/
```

## Tests

```bash
python tests/test_units.py                                    # no environment needed
CAVE_TEST_ENV=l2rpn_idf_2023 python tests/test_smoke.py       # plays the first 50 steps of a week
```

Both also run under `pytest tests`.

* `test_units.py` checks that the weights load. It also checks the risk aggregation, the reactance estimator (against
  a synthetic pi-model line), the identification with reliability shrinkage, and the paired decision rule of the
  training stage.
* `test_smoke.py` checks that the agent plays the first steps of a week without errors or illegal actions.

## Repository layout

```
cave/
  agent.py         CaVeAgent: operating point, simulator correction, drift guard, blind-forecast fallback, train/save/load
  controller.py    the decision pipeline (calm grid, crisis decision, attacked-line requests, final recheck)
  ranker.py        GAT ranker and observation-to-graph conversion
  verify.py        verification on the identified model
  redispatch.py    redispatch rules
  maintenance.py   planned-outage anticipation
  fragility.py     attackable bridges and calm-grid topology reverts
  actions.py       legality, reconnection, unary decomposition, candidate simulation
  simulation.py    blind-forecast fallback and reactance scaling of a simulator backend
  identify.py      line-parameter estimator, training stage, drift guard
  config.py        every constant, documented
  weights/         model.pt, train_config.pkl, scaler.pkl, actions.npy (the action library)
examples/run_week.py
tests/
```

## Notes and limits

* **Grid-specific data.** The attackable lines (`config.ATTACKABLE_LINES`) and the action library belong to the L2RPN
  IDF environment. The training stage adds to the contingency set any line it sees attacked and any line the opponent
  declares.
* **What the identified model corrects.** It corrects line reactances only. Resistances, transformers and injections
  come from the published grid and the forecast.
* **Determinism.** For a given observation sequence and process, the decisions are deterministic. They may differ
  across platforms or library versions, because the simulator's floating-point results can differ.

## Authors

Team SAGE × ANL: Ahmad Al-Tawaha, Mohammad Ramadan, Adrian Maldonado, Ming Jin, Mihai Anitescu.

## Licence and acknowledgements

This repository is released under the Apache License 2.0 (`LICENSE`), with one exception: `cave/actions.py` and
`cave/ranker.py` are derived from [SoftGNN](https://github.com/AI4REALNET/soft_label_gnn) (Hassouna et al., *Learning
Topology Actions for Power Grid Control: A Graph-Based Soft-Label Imitation Learning Approach*, ECML PKDD 2025). Those
two files remain under the Mozilla Public License 2.0 (`LICENSE-MPL-2.0`).

The agent runs on [grid2op](https://github.com/rte-france/Grid2Op) and
[LightSim2Grid](https://github.com/rte-france/LightSim2Grid).
