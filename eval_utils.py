"""Shared evaluation helpers: metrics, group-level splits and classical structural baselines."""

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, f1_score, roc_curve

from qia_engine import build_adjacency


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """PR-AUC (average precision), F1 @ 0.5 and true FPR at >=95% recall for the illicit class."""
    if y_true.sum() == 0:
        return {"pr_auc": float("nan"), "f1": float("nan"), "fpr_at_95rec": float("nan")}
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, (y_prob > 0.5).astype(int), zero_division=0)),
        "fpr_at_95rec": float(fpr[np.searchsorted(tpr, 0.95, side="left")]),
    }


def groups_from_starts(group_starts: List[int], num_nodes: int) -> np.ndarray:
    """Maps node ids to the generated subgraph (ring / cluster) that created them."""
    return np.searchsorted(np.array(group_starts), np.arange(num_nodes), side="right") - 1


def group_split(
    groups: np.ndarray,
    labels: np.ndarray,
    train_frac: float = 0.65,
    seed: int = 99
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train/test masks split by whole subgraph so no ring or cluster straddles the split
    (a node-level split lets labelled ring-mates leak into the test set). Illicit and
    legitimate groups are split separately so both sides contain illicit nodes whenever
    at least two illicit groups exist.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    is_illicit = np.array([labels[groups == g].max() == 1 for g in uniq])
    train_groups = []
    for flag in (True, False):
        pool = uniq[is_illicit == flag].copy()
        rng.shuffle(pool)
        if len(pool) == 0:
            continue
        n_train = int(round(train_frac * len(pool)))
        n_train = max(1, min(len(pool) - 1, n_train)) if len(pool) >= 2 else len(pool)
        train_groups.extend(pool[:n_train])
    train_mask = np.isin(groups, train_groups)
    return train_mask, ~train_mask


def structural_features(num_nodes: int, edges: List) -> np.ndarray:
    """
    Classical, non-quantum graph statistics: degrees, log-flow strengths, flow imbalance
    (the same quantity that drives the engine's potential wells), reciprocity and 2-hop reach.
    This is the fair 'is the quantum machinery doing anything?' baseline.
    """
    adj = build_adjacency(num_nodes, edges)
    binary = (adj > 0).astype(np.float64)
    in_deg = np.asarray(binary.sum(axis=0)).ravel()
    out_deg = np.asarray(binary.sum(axis=1)).ravel()
    log_adj = adj.copy()
    log_adj.data = np.log1p(log_adj.data)
    in_str = np.asarray(log_adj.sum(axis=0)).ravel()
    out_str = np.asarray(log_adj.sum(axis=1)).ravel()
    imbalance = (in_str - out_str) / (in_str + out_str + 1e-6)
    recip = np.asarray(binary.multiply(binary.T).sum(axis=1)).ravel()
    reach_out2 = np.asarray(((binary @ binary) > 0).sum(axis=1)).ravel()
    reach_in2 = np.asarray(((binary.T @ binary.T) > 0).sum(axis=1)).ravel()
    sym = ((binary + binary.T) > 0).astype(np.float64)
    deg = np.asarray(sym.sum(axis=1)).ravel()
    nbr_mean_deg = (sym @ deg) / np.maximum(deg, 1.0)
    return np.column_stack([
        in_deg, out_deg, in_str, out_str, imbalance, recip,
        reach_out2, reach_in2, nbr_mean_deg,
    ])
