"""
Elliptic Bitcoin dataset loader.

Files (data/elliptic/, downloaded from https://data.pyg.org/datasets/elliptic/ or Kaggle):
  elliptic_txs_features.csv  : txId, timestep, 165 features (93 local + 72 neighbour-aggregated)
  elliptic_txs_edgelist.csv  : txId1 -> txId2 (directed payment flow)
  elliptic_txs_classes.csv   : txId, class in {1 = illicit, 2 = licit, unknown}

Protocol (standard temporal split): train on timesteps 1-34, test on 35-49, labeled nodes only.
Tuning uses timesteps 1-27 for fitting and 28-34 for validation, never test data.
Edges only connect nodes within one timestep, so per-timestep graphs are independent.
"""

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "elliptic")
TRAIN_END, VAL_START, TEST_START = 34, 28, 35


@dataclass
class EllipticData:
    features: np.ndarray        # (N, 165) float32, standardized-by-dataset as shipped
    timesteps: np.ndarray       # (N,) int, 1..49
    labels: np.ndarray          # (N,) int: 1 illicit, 0 licit, -1 unknown
    edges: np.ndarray           # (E, 2) int, directed, global node indices
    tx_ids: np.ndarray

    @property
    def num_nodes(self) -> int:
        return len(self.labels)

    def split_masks(self):
        labeled = self.labels >= 0
        t = self.timesteps
        return {
            "fit": labeled & (t < VAL_START),                 # tuning fit set
            "val": labeled & (t >= VAL_START) & (t <= TRAIN_END),
            "train": labeled & (t <= TRAIN_END),              # final fit set
            "test": labeled & (t >= TEST_START),
            "test_post_shutdown": labeled & (t >= 43),
        }

    def timestep_graph(self, t: int):
        """Returns (global_node_idx, local_edges (u, v, 1.0) list) for one timestep."""
        idx = np.where(self.timesteps == t)[0]
        local = -np.ones(self.num_nodes, dtype=np.int64)
        local[idx] = np.arange(len(idx))
        mask = (self.timesteps[self.edges[:, 0]] == t)
        e = self.edges[mask]
        lu, lv = local[e[:, 0]], local[e[:, 1]]
        keep = (lu >= 0) & (lv >= 0)
        return idx, np.column_stack([lu[keep], lv[keep]])


def load_elliptic(data_dir: str = DATA_DIR) -> EllipticData:
    feats = pd.read_csv(os.path.join(data_dir, "elliptic_txs_features.csv"), header=None)
    classes = pd.read_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"))
    edgelist = pd.read_csv(os.path.join(data_dir, "elliptic_txs_edgelist.csv"))

    tx_ids = feats[0].to_numpy()
    timesteps = feats[1].to_numpy().astype(int)
    x = feats.iloc[:, 2:].to_numpy(dtype=np.float32)

    cls = classes.set_index("txId")["class"].reindex(tx_ids).astype(str)
    labels = cls.map({"1": 1, "2": 0}).fillna(-1).to_numpy().astype(int)

    id_to_idx = pd.Series(np.arange(len(tx_ids)), index=tx_ids)
    e = np.column_stack([
        id_to_idx.reindex(edgelist["txId1"]).to_numpy(),
        id_to_idx.reindex(edgelist["txId2"]).to_numpy(),
    ])
    e = e[~np.isnan(e).any(axis=1)].astype(np.int64)
    return EllipticData(x, timesteps, labels, e, tx_ids)


def timesteps_list() -> List[int]:
    return list(range(1, 50))
