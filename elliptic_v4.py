"""
Elliptic experiment for HQW-AML v4 (quantum-walk risk propagation). No GNNs.

Stage 1: out-of-fold risk r from XGBoost on raw features. Training-period nodes get r from models that
never saw their own timestep's labels (timestep-grouped folds over 1-34); test-period nodes (35-49) get r from
a model fit on all labeled training nodes. No test label is used anywhere.
Stage 2: propagate r over each timestep graph (quantum / classical heat / classical neighbour) and
feed raw + r + propagated features to the same tuned learners as elliptic_benchmark.py.
"""

import json
import os
import sys
import time

import numpy as np
import xgboost as xgb

import elliptic_benchmark as eb
from elliptic_data import load_elliptic
from qia_propagation import quantum_risk_features, heat_risk_features, neighbour_risk_features

N_FOLDS = 6
STAGE1 = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.7,
              min_child_weight=2, tree_method="hist", n_jobs=10)


def stage1_risk(data, masks, seed=0):
    raw, y, t = data.features, data.labels, data.timesteps
    train_lab = masks["train"]
    pos_ratio = float((y[train_lab] == 0).sum() / (y[train_lab] == 1).sum())
    params = dict(STAGE1, scale_pos_weight=float(np.sqrt(pos_ratio)), random_state=seed)
    risk = np.zeros(data.num_nodes)
    for k in range(N_FOLDS):
        held = (t <= 34) & (t % N_FOLDS == k)
        fit = train_lab & ~held
        m = xgb.XGBClassifier(**params).fit(raw[fit], y[fit])
        risk[held] = m.predict_proba(raw[held])[:, 1]
    m = xgb.XGBClassifier(**params).fit(raw[train_lab], y[train_lab])
    risk[t >= 35] = m.predict_proba(raw[t >= 35])[:, 1]
    return risk


def build_sets(data, risk):
    def per_ts(fn):
        out = None
        for ts in range(1, 50):
            idx, e = data.timestep_graph(ts)
            e = [(int(a), int(b)) for a, b in e]
            block = fn(len(idx), e, risk[idx])
            if out is None:
                out = np.zeros((data.num_nodes, block.shape[1]), dtype=np.float32)
            out[idx] = block
        return out

    t0 = time.time()
    q1 = per_ts(lambda n, e, r: quantum_risk_features(n, e, r, gamma=1.0))
    q0 = per_ts(lambda n, e, r: quantum_risk_features(n, e, r, gamma=0.0))
    qnw = per_ts(lambda n, e, r: quantum_risk_features(n, e, r, gamma=1.0, flow_potential_weight=0.0))
    heat = per_ts(heat_risk_features)
    nbr = per_ts(neighbour_risk_features)
    print(f"propagation features built in {time.time() - t0:.0f}s", flush=True)

    raw, r = data.features, risk[:, None].astype(np.float32)
    return {
        "raw": raw,
        "raw+risk (stacking only)": np.hstack([raw, r]),
        "raw+risk+neighbour (classical)": np.hstack([raw, r, nbr]),
        "raw+risk+heat (classical)": np.hstack([raw, r, heat]),
        "raw+risk+neighbour+heat (classical)": np.hstack([raw, r, nbr, heat]),
        "raw+risk+qwalk gamma=1 (HQW v4)": np.hstack([raw, r, q1]),
        "raw+risk+qwalk gamma=0 (ablation)": np.hstack([raw, r, q0]),
        "raw+risk+qwalk no wells (ablation)": np.hstack([raw, r, qnw]),
        "raw+risk+qwalk+neighbour+heat": np.hstack([raw, r, q1, nbr, heat]),
    }


def main():
    out_path = os.path.join(eb.RESULTS, "elliptic_v4_results.json")
    log = lambda m: print(m, flush=True)
    data = load_elliptic()
    masks = data.split_masks()
    risk = stage1_risk(data, masks)
    from sklearn.metrics import average_precision_score
    te = masks["test"]
    log(f"stage-1 risk alone: test PR-AUC {average_precision_score(data.labels[te], risk[te]):.3f}")
    sets = build_sets(data, risk)
    res = eb.load_results(out_path)
    for name in sets:
        key = f"xgb[{name}]"
        if key in res:
            continue
        res[key] = eb.run_gbdt(data, sets, "xgb", name, masks, log)
        eb.save_results(out_path, res)
    print("\n" + eb.report(res))


if __name__ == "__main__":
    main()
