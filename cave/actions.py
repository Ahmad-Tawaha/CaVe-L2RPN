# This file is derived from SoftGNN (https://github.com/AI4REALNET/soft_label_gnn, evaluation/utilities.py).
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was
# not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/ (see LICENSE-MPL-2.0).
"""Action helpers shared by every part of the controller: legality, validity of a simulation, line reconnection,
unary decomposition of compound topology actions, one-step candidate simulation and the calm-grid topology revert.

``SUSPECT_LINES`` holds the lines that are out of service because of an ongoing attack. The environment reports
cooldown 0 for them and ``obs.simulate`` grants their reconnection, but the real environment refuses it while the
attack lasts. They are therefore never added to a candidate action, so every simulation sees the grid that really
exists. The controller refreshes the set at the start of every step (:meth:`cave.controller.Controller.update_suspects`);
it is per-process state.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Iterator, List, Tuple

import numpy as np
from grid2op.Action import ActionSpace, BaseAction
from grid2op.Observation import BaseObservation
from grid2op.dtypes import dt_int

SUSPECT_LINES: set = set()


def remove_bus_assignment_of_lines(obs: BaseObservation, action: BaseAction, lines) -> BaseAction:
    """Drop any bus assignment of the given lines' terminals (a bus assignment on a disconnected line reconnects it)."""
    if len(lines) == 0:
        return action
    or_bus = [obs.line_or_pos_topo_vect[i] for i in lines]
    ex_bus = [obs.line_ex_pos_topo_vect[i] for i in lines]
    bus_assignment = action.set_bus.copy()
    bus_assignment[or_bus + ex_bus] = 0
    out = deepcopy(action)
    out.update({"set_bus": bus_assignment})
    return out


def is_legal(action: BaseAction, obs: BaseObservation) -> bool:
    """Cooldown check: no touched substation or line may be in cooldown, and a bus change may not touch a line that
    is out of service."""
    action_dict = action.as_dict()
    if action_dict == {}:
        return True
    kind = list(action_dict.keys())[0]
    legal = True
    if kind in ("set_bus_vect", "change_bus_vect"):
        for sub in [int(s) for s in action_dict[kind]["modif_subs_id"]]:
            if obs.time_before_cooldown_sub[sub]:
                legal = False
            for line in [val["id"] for val in action_dict[kind][str(sub)].values() if "line" in val["type"]]:
                if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
                    legal = False
    elif kind == "set_line_status":
        for line in [int(l) for l in action_dict[kind]["connected_id"]]:
            if obs.time_before_cooldown_line[line]:
                legal = False
    return legal


def is_valid(obs: BaseObservation, action: BaseAction, done: bool, info: dict) -> bool:
    """A simulated action is usable if the episode does not end, the action is legal and unambiguous, and the
    simulation raised no exception."""
    valid = True
    if done:
        valid = False
    if not is_legal(action, obs):
        valid = False
    if info["is_illegal"]:
        valid = False
    if info["is_ambiguous"]:
        valid = False
    if any(info["exception"]):
        valid = False
    return valid


def find_best_line_to_reconnect(obs: BaseObservation, original_action: BaseAction) -> BaseAction:
    """Add to ``original_action`` the reconnection of the out-of-service line (cooldown over, not under attack) that
    gives the lowest simulated peak loading. The bus assignments of all other out-of-service lines are removed."""
    disconnected_lines = np.where(obs.line_status == False)[0]  # noqa: E712
    if len(disconnected_lines) == 0:
        return original_action
    min_rho = 10
    line_to_reconnect = -1
    for line in disconnected_lines:
        if int(line) in SUSPECT_LINES:
            continue
        if not obs.time_before_cooldown_line[line]:
            reconnect_array = np.zeros_like(obs.rho, dtype=int)
            reconnect_array[line] = 1
            reconnect_action = deepcopy(original_action)
            reconnect_action.update({"set_line_status": reconnect_array})
            if not is_legal(reconnect_action, obs):
                continue
            o, _, done, info = obs.simulate(reconnect_action)
            if not is_valid(obs, reconnect_action, done, info):
                continue
            if o.rho.max() < min_rho:
                line_to_reconnect = line
                min_rho = o.rho.max()
    if line_to_reconnect != -1:
        out = deepcopy(original_action)
        reconnect_array = np.zeros_like(obs.rho, dtype=int)
        reconnect_array[line_to_reconnect] = 1
        out.update({"set_line_status": reconnect_array})
        others = disconnected_lines.copy().tolist()
        others.remove(line_to_reconnect)
        return remove_bus_assignment_of_lines(obs, out, others)
    return remove_bus_assignment_of_lines(obs, deepcopy(original_action), disconnected_lines)


def unary_actions(action_space: ActionSpace, action_vect: np.ndarray) -> List[BaseAction]:
    """Split a compound (multi-substation) action into one action per substation."""
    out = [action_space({})]
    decomposed = action_space.from_vect(action_vect).decompose_as_unary_actions()
    if decomposed:
        out = []
        for v in decomposed.values():
            out += v
    return out


def split_action_and_return(obs: BaseObservation, action_space: ActionSpace,
                            action_vect: np.ndarray) -> Iterator[BaseAction]:
    """Execute a compound action one substation per step (the rules allow one substation change per step).

    Yields the unary parts in greedy order: at each step the part that, with the best reconnection, gives the lowest
    simulated peak loading on the observation the plan started from. Yields do-nothing for an empty action or when no
    part is valid."""
    noop = action_space({})
    if not action_vect.any() or action_space.from_vect(action_vect) == noop:
        yield noop
        return
    parts = unary_actions(action_space, action_vect)
    for _ in range(len(parts)):
        best_rho = np.inf
        best = None
        for act in parts:
            act_reco = find_best_line_to_reconnect(obs, act)
            o, _, done, info = obs.simulate(act_reco)
            if not is_valid(obs, act_reco, done, info):
                continue
            if o.rho.max() < best_rho:
                best = act
                best_rho = o.rho.max()
        if best is None:
            best = noop
        yield best
        if best in parts:
            parts.remove(best)


def simulate_candidate(action_space: ActionSpace, obs: BaseObservation, action_vect) -> Tuple[float, bool, BaseObservation]:
    """Simulate a library action as it would be executed now: a one-substation (or line-only) action with the best
    reconnection, or the first step of a compound action. Returns (peak loading, valid, simulated observation)."""
    action = action_space.from_vect(action_vect)
    act_dict = action.as_dict()
    if "set_bus_vect" not in act_dict.keys():
        action = find_best_line_to_reconnect(obs=obs, original_action=action)
        obs_f, _, done, info = obs.simulate(action)
    elif len(act_dict["set_bus_vect"]["modif_subs_id"]) == 1:
        action = find_best_line_to_reconnect(obs=obs, original_action=action)
        obs_f, _, done, info = obs.simulate(action)
    else:
        action = next(split_action_and_return(obs=obs, action_space=action_space, action_vect=action_vect))
        obs_f, _, done, info = obs.simulate(action)
    return obs_f.rho.max(), is_valid(obs, action, done, info), obs_f


def simulate_rho(action_space: ActionSpace, obs: BaseObservation, action_vect) -> Tuple[float, bool]:
    """:func:`simulate_candidate` without the observation."""
    rho, valid, _ = simulate_candidate(action_space, obs, action_vect)
    return rho, valid


def library_key(idx) -> str:
    """Pool key of library action ``idx``. Keys sort after "noop"; the all-fatal ranking uses the key as its last
    tie-break."""
    return f"topo:{int(idx)}"


def library_index(key) -> int:
    """Library index of a pool key made by :func:`library_key`."""
    return int(str(key).split(":", 1)[1])


def revert_topo(action_space: ActionSpace, obs: BaseObservation, rho_limit: float = 0.8) -> np.ndarray:
    """Classic calm-grid revert: if every line is in service and some substation uses bus 2, return the vector of a
    substation revert (all its elements to bus 1) that brings the peak loading below min(current peak, rho_limit);
    otherwise the do-nothing vector. When several qualify, the last one in substation order is returned."""
    best = None
    if np.any(obs.topo_vect == 2) and np.all(obs.line_status):
        available = {}
        for sub_id in range(obs.n_sub):
            if np.any(obs.sub_topology(sub_id) == 2):
                available[sub_id] = action_space(
                    {"set_bus": {"substations_id": [(sub_id, np.ones(len(obs.sub_topology(sub_id)), dtype=dt_int))]}}
                ).to_vect()
        if any(available):
            min_rho = min([obs.rho.max(), rho_limit])
            for _sub_id, vect in available.items():
                rho, valid = simulate_rho(action_space, obs, vect)
                if not valid:
                    continue
                if rho < min_rho:
                    best = vect.copy()
    if best is not None:
        return best
    return action_space({}).to_vect()
