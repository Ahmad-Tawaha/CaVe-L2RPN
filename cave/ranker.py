# This file is derived from SoftGNN (https://github.com/AI4REALNET/soft_label_gnn, gnn/gnn_models.py and
# gnn/obs_converter.py).
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was
# not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/ (see LICENSE-MPL-2.0).
"""The learned action ranker: a graph-attention network over the bus graph that scores every action of the library.

The ranker was trained in simulation to imitate the outcome of exhaustive search (a soft target distribution over the
library, KL loss), so its output is a probability per library action. At run time it only *proposes*: the controller
simulates the top proposals and decides by its own tests.

Input graph (:func:`obs_to_graph`): one node per topological bus element position, 27 features per node (bus, cooldowns,
load / generator / line / storage attributes, maintenance timers), edges = the observation's connectivity matrix.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.nn import Linear
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import from_scipy_sparse_matrix

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


class GATRanker(torch.nn.Module):
    """Stack of GAT layers, graph pooling, MLP head; returns log-probabilities over the action library."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.global_features_size = config.get("global_features_size", 0)
        self.gat_layers = torch.nn.ModuleList()
        in_channels = config["in_channels"]
        for layer_cfg in config["gat_layers"]:
            concat = layer_cfg.get("concat", True)
            self.gat_layers.append(GATConv(in_channels, layer_cfg["out_channels"], heads=layer_cfg["heads"], concat=concat,
                                           dropout=layer_cfg.get("dropout", config["dropout"])))
            in_channels = layer_cfg["out_channels"] * layer_cfg["heads"] if concat else layer_cfg["out_channels"]
        self.linear_layers = torch.nn.ModuleList()
        self.linear_dropouts = []
        in_channels = in_channels + self.global_features_size
        for layer_cfg in config["linear_layers"]:
            self.linear_layers.append(Linear(in_channels, layer_cfg["out_features"]))
            self.linear_dropouts.append(layer_cfg.get("dropout", config["dropout"]))
            in_channels = layer_cfg["out_features"]
        self.pooling_type = config["pooling_type"]
        self.gat_activation = getattr(F, config["gat_activation"], F.elu)
        self.linear_activation = getattr(F, config["linear_activation"], F.relu)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for layer in self.gat_layers:
            x = self.gat_activation(layer(x, edge_index))
        pool = {"max": global_max_pool, "mean": global_mean_pool, "add": global_add_pool}[self.pooling_type]
        x = pool(x, data.batch)
        last = len(self.linear_layers) - 1
        for i, layer in enumerate(self.linear_layers):
            if i == last:
                x = layer(x)
            else:
                x = F.dropout(self.linear_activation(layer(x)), p=self.linear_dropouts[i], training=self.training)
        return F.log_softmax(x, dim=1)


def obs_to_graph(obs):
    """(node features (n, 27) float32 tensor, edge_index (2, m)) of a grid2op observation."""
    dat = [obs.get_obj_substations(substation_id=sub)[:, 1:] for sub in range(obs.n_sub)]
    global_ids = np.concatenate(dat)
    obs_dict = obs.to_dict()
    topo_vect = obs.topo_vect.reshape(1, -1)
    cooldown_substation = np.concatenate(
        [obs_dict["cooldown"]["substation"][obs.get_obj_substations(substation_id=sub)[:, 0]] for sub in range(obs.n_sub)]
    ).reshape(1, -1)
    loads = [np.append(obs_dict["loads"][k], 0)[global_ids[:, 0]] for k in obs_dict["loads"].keys()]
    gens = [np.append(obs_dict["gens"][k], 0)[global_ids[:, 1]] for k in obs_dict["gens"].keys()]
    gens.append(np.append(obs_dict["gen_p_before_curtail"], 0)[global_ids[:, 1]])
    gens.append(np.append(obs_dict["curtailment"], 0)[global_ids[:, 1]])
    lines_or = [np.append(obs_dict["lines_or"][k], 0)[global_ids[:, 2]] for k in obs_dict["lines_or"].keys()]
    lines_ex = [np.append(obs_dict["lines_ex"][k], 0)[global_ids[:, 3]] for k in obs_dict["lines_ex"].keys()]
    line_mask = global_ids[:, 3].copy()
    line_mask[global_ids[:, 2] != -1] = global_ids[:, 2][global_ids[:, 2] != -1]
    line_infos = [np.append(obs_dict[k], 0)[line_mask] for k in ["timestep_overflow", "line_status", "rho"]]
    line_infos.append(np.append(obs_dict["maintenance"]["time_next_maintenance"], 0)[line_mask])
    line_infos.append(np.append(obs_dict["maintenance"]["duration_next_maintenance"], 0)[line_mask])
    line_infos.append(np.append(obs_dict["cooldown"]["line"], 0)[line_mask])
    storage = [np.append(obs_dict["storage_charge"], 0)[global_ids[:, 4]],
               np.append(obs_dict["storage_power_target"], 0)[global_ids[:, 4]],
               np.append(obs_dict["storage_power"], 0)[global_ids[:, 4]]]
    out = np.concatenate([topo_vect, cooldown_substation, loads, gens, lines_or, lines_ex, line_infos, storage]).transpose()
    out = torch.tensor(out).type(torch.FloatTensor)
    edge_index = from_scipy_sparse_matrix(obs.connectivity_matrix(as_csr_matrix=True))[0]
    return out, edge_index


class ActionRanker:
    """Loads the action library, the feature scaler and the network; :meth:`rank` returns library indices, best first."""

    def __init__(self, weights_dir: Path | str = WEIGHTS_DIR):
        weights_dir = Path(weights_dir)
        self.actions: np.ndarray = np.load(str(weights_dir / "actions.npy"))
        with open(weights_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)                      # sklearn StandardScaler fitted on the training features
        with open(weights_dir / "train_config.pkl", "rb") as f:
            config = pickle.load(f)[0]
        self.model = GATRanker(config)
        self.model.load_state_dict(torch.load(weights_dir / "model.pt", map_location=torch.device("cpu")))
        self.model.eval()

    def rank(self, obs) -> np.ndarray:
        features, edge_index = obs_to_graph(obs)
        features = torch.tensor(self.scaler.transform(features.numpy()), dtype=torch.float)
        with torch.no_grad():
            probabilities = torch.exp(self.model(Data(x=features, edge_index=edge_index))).numpy().reshape(-1)
        return probabilities.argsort()[::-1]
