"""
Drift-robustness experiment on Elliptic: per-timestep feature normalization (no labels used, only
unlabeled features within each timestep) combined with risk propagation and HQW/structural features.
No GNNs.
"""
import os

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

import elliptic_benchmark as eb
import elliptic_v4 as v4
from elliptic_data import load_elliptic


def per_timestep_rank(x, timesteps):
    out = np.zeros_like(x, dtype=np.float32)
    for t in np.unique(timesteps):
        m = timesteps == t
        out[m] = (rankdata(x[m], axis=0) - 0.5) / m.sum()
    return out


def per_timestep_zscore(x, timesteps):
    out = np.zeros_like(x, dtype=np.float32)
    for t in np.unique(timesteps):
        m = timesteps == t
        mu, sd = x[m].mean(0), x[m].std(0)
        out[m] = (x[m] - mu) / np.where(sd > 1e-6, sd, 1.0)
    return np.clip(out, -10, 10)


def main():
    out_path = os.path.join(eb.RESULTS, "elliptic_drift_results.json")
    log = lambda m: print(m, flush=True)
    data = load_elliptic()
    masks = data.split_masks()
    t = data.timesteps

    raw_rank = per_timestep_rank(data.features, t)
    raw_z = per_timestep_zscore(data.features, t)
    hqw = np.load(os.path.join(eb.CACHE, "hqw_topo.npy"))
    struct = np.load(os.path.join(eb.CACHE, "structural.npy"))
    hqw_rank, struct_rank = per_timestep_rank(hqw, t), per_timestep_rank(struct, t)

    sets = {
        "raw (absolute)": data.features,
        "raw rank-per-timestep": raw_rank,
        "raw z-score-per-timestep": raw_z,
        "rank raw + rank structural": np.hstack([raw_rank, struct_rank]),
        "rank raw + rank HQW topology": np.hstack([raw_rank, hqw_rank]),
        "rank HQW topology only": hqw_rank,
        "rank structural only": struct_rank,
    }
    for tag, X in (("rank", raw_rank), ("z", raw_z)):
        risk = v4.stage1_risk(data, masks, X=X)
        te = masks["test"]
        log(f"stage-1 risk ({tag}) alone: test PR-AUC {average_precision_score(data.labels[te], risk[te]):.3f}")
        full = v4.build_sets(data, risk, X=X)
        sets[f"{tag}: raw+risk+classical (neighbour+heat)"] = full["raw+risk+neighbour+heat (classical)"]
        sets[f"{tag}: raw+risk+qwalk (HQW v4)"] = full["raw+risk+qwalk gamma=1 (HQW v4)"]
        sets[f"{tag}: raw+risk+qwalk+classical"] = full["raw+risk+qwalk+neighbour+heat"]

    res = eb.load_results(out_path)
    for name in sets:
        key = f"xgb[{name}]"
        if key not in res:
            res[key] = eb.run_gbdt(data, sets, "xgb", name, masks, log)
            eb.save_results(out_path, res)
    print("\n" + eb.report(res))


if __name__ == "__main__":
    main()
