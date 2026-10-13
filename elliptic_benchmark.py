"""
Final benchmark on the Elliptic Bitcoin dataset (203,769 transactions, ~2% illicit).

Protocol
- Temporal split: train timesteps 1-34, test 35-49, labeled nodes only. Test labels are never
  used for tuning, early stopping or threshold selection.
- Every model gets the same tuning recipe: random hyper-parameter search, fit on timesteps 1-27,
  scored by validation PR-AUC on 28-34, then refit on 1-34 with the selected configuration.
  The decision threshold for F1 is selected on validation predictions of the tuned model.
- Stochastic models are refit with 5 seeds; results are mean +/- std.

Classical systems
- Tabular: XGBoost, LightGBM, Random Forest on raw features (local-only and all 165).
- Graph neural networks (PyTorch Geometric): GCN, GraphSAGE, GATv2 on the full transaction graph.
- Structural-statistics GBDT: raw features + degrees / flow imbalance / reciprocity / 2-hop reach.
Quantum-inspired
- HQW-AML features computed per timestep graph (topology-only and with unitary wave features),
  plus a gamma=0 ablation, all fed to the *same* tuned GBDT learners as the baselines, so the
  comparison isolates the value of the features.

Usage: python elliptic_benchmark.py [--stage features|gbdt|gnn|report|all]
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_curve

from elliptic_data import load_elliptic
from eval_utils import structural_features
from qia_engine import extract_features

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "data", "cache")
RESULTS = os.path.join(ROOT, "results")
SEEDS = [0, 1, 2, 3, 4]
N_SEARCH_GBDT = 20
N_SEARCH_RF = 8
N_SEARCH_GNN = 10
K_MODES = 192


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def metrics(y, p, threshold):
    fpr, tpr, _ = roc_curve(y, p)
    pred = (p >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float((pred[y == 1].sum()) / max(pred.sum(), 1)),
        "recall": float(pred[y == 1].mean()),
        "fpr_at_95rec": float(fpr[np.searchsorted(tpr, 0.95, side="left")]),
    }


def evaluate(y_test, p_test, post_mask, threshold):
    out = metrics(y_test, p_test, threshold)
    if post_mask is not None and y_test[post_mask].sum() > 0:
        out["pr_auc_post_shutdown"] = float(average_precision_score(y_test[post_mask], p_test[post_mask]))
    return out


# --------------------------------------------------------------------------- #
# Feature sets (cached on disk)
# --------------------------------------------------------------------------- #

def _cache_path(name):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, f"{name}.npy")


def per_timestep(data, fn):
    """Applies fn(local_n, local_edges, global_idx) per timestep and scatters the rows back."""
    out = None
    for t in range(1, 50):
        idx, e = data.timestep_graph(t)
        block = fn(len(idx), [(int(u), int(v)) for u, v in e], idx)
        if out is None:
            out = np.zeros((data.num_nodes, block.shape[1]), dtype=np.float32)
        out[idx] = block
    return out


def build_feature_sets(data):
    """Returns dict name -> (N, d) float32, computing and caching extraction as needed."""
    raw = data.features
    sets = {}

    def cached(name, compute):
        path = _cache_path(name)
        if os.path.exists(path):
            return np.load(path)
        t0 = time.time()
        arr = compute()
        np.save(path, arr)
        print(f"  computed {name}: {arr.shape} in {time.time() - t0:.0f}s", flush=True)
        return arr

    struct = cached("structural", lambda: per_timestep(data, lambda n, e, idx: structural_features(n, e)))
    hqw = cached("hqw_topo", lambda: per_timestep(
        data, lambda n, e, idx: extract_features(n, e, None, gamma=1.0, k_modes=K_MODES)))
    hqw_g0 = cached("hqw_topo_g0", lambda: per_timestep(
        data, lambda n, e, idx: extract_features(n, e, None, gamma=0.0, k_modes=K_MODES)))

    train_rows = data.timesteps <= 34
    pca = PCA(n_components=16, random_state=0).fit(raw[train_rows])
    low = pca.transform(raw).astype(np.float64)
    low /= low.std(axis=0, keepdims=True) + 1e-9
    hqw_wave = cached("hqw_wave", lambda: per_timestep(
        data, lambda n, e, idx: extract_features(n, e, low[idx], gamma=1.0, k_modes=K_MODES)))

    sets["raw_local"] = raw[:, :93]
    sets["raw_all"] = raw
    sets["structural_only"] = struct
    sets["hqw_only"] = hqw
    sets["raw+structural"] = np.hstack([raw, struct])
    sets["raw+hqw"] = np.hstack([raw, hqw])
    sets["raw+hqw_gamma0"] = np.hstack([raw, hqw_g0])
    sets["raw+hqw_wave"] = np.hstack([raw, hqw_wave])
    sets["raw+structural+hqw"] = np.hstack([raw, struct, hqw])
    return {k: np.nan_to_num(v.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6) for k, v in sets.items()}


# --------------------------------------------------------------------------- #
# GBDT / RF learners with a shared tuning recipe
# --------------------------------------------------------------------------- #

def sample_xgb(rng, pos_ratio):
    return dict(
        max_depth=int(rng.integers(3, 11)),
        learning_rate=float(10 ** rng.uniform(-1.7, -0.5)),
        subsample=float(rng.uniform(0.5, 1.0)),
        colsample_bytree=float(rng.uniform(0.3, 1.0)),
        min_child_weight=float(rng.integers(1, 21)),
        reg_lambda=float(10 ** rng.uniform(-2, 1)),
        gamma=float(rng.choice([0, 0.5, 1, 5])),
        scale_pos_weight=float(rng.choice([1.0, np.sqrt(pos_ratio), pos_ratio])),
    )


def sample_lgbm(rng, pos_ratio):
    return dict(
        num_leaves=int(rng.choice([15, 31, 63, 127, 255])),
        learning_rate=float(10 ** rng.uniform(-1.7, -0.5)),
        subsample=float(rng.uniform(0.5, 1.0)),
        subsample_freq=1,
        colsample_bytree=float(rng.uniform(0.3, 1.0)),
        min_child_samples=int(rng.integers(5, 61)),
        reg_lambda=float(10 ** rng.uniform(-2, 1)),
        scale_pos_weight=float(rng.choice([1.0, np.sqrt(pos_ratio), pos_ratio])),
    )


def sample_rf(rng, pos_ratio):
    # index into python lists: rng.choice over mixed-type lists coerces everything to strings
    pick = lambda opts: opts[int(rng.integers(len(opts)))]
    return dict(
        max_depth=pick([None, 10, 20]),
        min_samples_leaf=pick([1, 2, 5]),
        max_features=pick(["sqrt", 0.3]),
        class_weight=pick([None, "balanced_subsample"]),
    )


def fit_predict(learner, params, X_fit, y_fit, X_eval_list, seed, n_rounds=None, X_val=None, y_val=None):
    """Fits one model; with a validation set uses early stopping and returns the best round count."""
    best_iter = n_rounds
    if learner == "xgb":
        import xgboost as xgb
        kw = dict(n_estimators=n_rounds or 1000, tree_method="hist", eval_metric="aucpr",
                  random_state=seed, n_jobs=10, **params)
        if X_val is not None:
            m = xgb.XGBClassifier(early_stopping_rounds=50, **kw)
            m.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
            best_iter = int(m.best_iteration) + 1
        else:
            m = xgb.XGBClassifier(**kw)
            m.fit(X_fit, y_fit)
    elif learner == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=n_rounds or 1000, random_state=seed, n_jobs=10,
                               verbose=-1, **params)
        if X_val is not None:
            m.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            best_iter = int(m.best_iteration_) or 1
        else:
            m.fit(X_fit, y_fit)
    elif learner == "rf":
        m = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=10, **params)
        m.fit(X_fit, y_fit)
    else:
        raise ValueError(learner)
    return [m.predict_proba(X)[:, 1] for X in X_eval_list], best_iter


def run_gbdt(data, sets, learner, set_name, masks, log):
    X, y = sets[set_name], data.labels
    fit, val, train, test = masks["fit"], masks["val"], masks["train"], masks["test"]
    pos_ratio = float((y[fit] == 0).sum() / max((y[fit] == 1).sum(), 1))
    sampler = {"xgb": sample_xgb, "lgbm": sample_lgbm, "rf": sample_rf}[learner]
    n_search = N_SEARCH_RF if learner == "rf" else N_SEARCH_GBDT
    rng = np.random.default_rng(0)

    t0 = time.time()
    best = (-1.0, None, None)
    for _ in range(n_search):
        params = sampler(rng, pos_ratio)
        (p_val,), it = fit_predict(learner, params, X[fit], y[fit], [X[val]], 0,
                                   X_val=X[val], y_val=y[val]) if learner != "rf" else \
            (fit_predict(learner, params, X[fit], y[fit], [X[val]], 0)[0], None)
        score = average_precision_score(y[val], p_val[0] if isinstance(p_val, list) else p_val)
        if score > best[0]:
            best = (score, params, it)
    val_ap, params, best_iter = best

    # threshold from the tuned fit-model's validation predictions
    (p_val,), _ = fit_predict(learner, params, X[fit], y[fit], [X[val]], 0, n_rounds=best_iter)
    threshold = best_f1_threshold(y[val], p_val)

    n_rounds = int(best_iter * 1.15) if best_iter else None
    per_seed = []
    for seed in SEEDS:
        (p_test,), _ = fit_predict(learner, params, X[train], y[train], [X[test]], seed, n_rounds=n_rounds)
        per_seed.append(evaluate(y[test], p_test, masks["test_post_shutdown"][test], threshold))
    log(f"  {learner:5s} | {set_name:22s} | val AP {val_ap:.3f} | test PR-AUC "
        f"{np.mean([r['pr_auc'] for r in per_seed]):.3f} F1 {np.mean([r['f1'] for r in per_seed]):.3f} "
        f"| {time.time() - t0:.0f}s")
    return {"val_ap": float(val_ap), "params": {k: (None if v is None else (v if isinstance(v, (int, float, str)) else str(v))) for k, v in params.items()},
            "threshold": threshold, "seeds": per_seed}


# --------------------------------------------------------------------------- #
# GNN baselines (PyTorch Geometric)
# --------------------------------------------------------------------------- #

def make_gnn(kind, in_dim, hidden, layers, dropout, heads=4):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            d = in_dim
            for i in range(layers):
                last = i == layers - 1
                out = 2 if last else hidden
                if kind == "gcn":
                    conv = GCNConv(d, out)
                elif kind == "sage":
                    conv = SAGEConv(d, out)
                else:
                    conv = GATv2Conv(d, out, heads=1) if last else GATv2Conv(d, hidden // heads, heads=heads)
                self.convs.append(conv)
                if not last:
                    self.norms.append(nn.LayerNorm(hidden))
                d = hidden
            self.dropout = dropout

        def forward(self, x, ei):
            for i, conv in enumerate(self.convs):
                x = F.dropout(x, self.dropout, self.training)
                x = conv(x, ei)
                if i < len(self.convs) - 1:
                    x = F.relu(self.norms[i](x))
            return x

    return Net()


def train_gnn(kind, cfg, data, masks, ei_dir, ei_undir, seed, fixed_epochs=None, fit_name="fit",
              max_epochs=300, patience=40):
    """Trains one GNN. With fixed_epochs=None uses early stopping on val PR-AUC and returns best epoch."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.tensor(data.features, dtype=torch.float32)
    y = torch.tensor(np.maximum(data.labels, 0), dtype=torch.long)
    ei = ei_undir if cfg["undirected"] else ei_dir
    fit_idx = torch.tensor(np.where(masks[fit_name])[0])
    val_idx = np.where(masks["val"])[0]

    pos = float((data.labels[masks[fit_name]] == 1).sum())
    neg = float((data.labels[masks[fit_name]] == 0).sum())
    weight = torch.tensor([1.0, cfg["pos_weight_mult"] * np.sqrt(neg / max(pos, 1))], dtype=torch.float32)

    model = make_gnn(kind, x.shape[1], cfg["hidden"], cfg["layers"], cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    best_ap, best_epoch, best_state, bad = -1.0, 0, None, 0
    epochs = fixed_epochs or max_epochs
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(x, ei)
        loss = F.cross_entropy(out[fit_idx], y[fit_idx], weight=weight)
        loss.backward()
        opt.step()
        if fixed_epochs is None:
            model.eval()
            with torch.no_grad():
                p = torch.softmax(model(x, ei), dim=1)[:, 1].numpy()
            ap = average_precision_score(data.labels[val_idx], p[val_idx])
            if ap > best_ap:
                best_ap, best_epoch, bad = ap, epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x, ei), dim=1)[:, 1].numpy()
    return probs, best_ap, best_epoch


def run_gnn(data, masks, kind, log):
    import torch
    e = torch.tensor(data.edges.T, dtype=torch.long)
    ei_dir = e
    ei_undir = torch.cat([e, e.flip(0)], dim=1)
    y, test = data.labels, masks["test"]
    rng = np.random.default_rng(0)

    t0 = time.time()
    best = (-1.0, None, None)
    for _ in range(N_SEARCH_GNN):
        cfg = dict(
            hidden=int(rng.choice([64, 128, 256])),
            layers=int(rng.choice([2, 3])),
            dropout=float(rng.choice([0.1, 0.3, 0.5])),
            lr=float(rng.choice([0.01, 0.005, 0.002])),
            wd=float(rng.choice([1e-4, 5e-4, 1e-3])),
            pos_weight_mult=float(rng.choice([0.5, 1.0, 2.0])),
            undirected=bool(rng.choice([True, False])),
        )
        probs, ap, epoch = train_gnn(kind, cfg, data, masks, ei_dir, ei_undir, seed=0)
        if ap > best[0]:
            best = (ap, cfg, epoch, probs)
    val_ap, cfg, epoch = best[0], best[1], best[2]
    threshold = best_f1_threshold(y[masks["val"]], best[3][masks["val"]])

    per_seed = []
    for seed in SEEDS:
        probs, _, _ = train_gnn(kind, cfg, data, masks, ei_dir, ei_undir, seed, fixed_epochs=int(epoch * 1.15) + 1,
                                fit_name="train")
        per_seed.append(evaluate(y[test], probs[test], masks["test_post_shutdown"][test], threshold))
    log(f"  gnn   | {kind:22s} | val AP {val_ap:.3f} | test PR-AUC "
        f"{np.mean([r['pr_auc'] for r in per_seed]):.3f} F1 {np.mean([r['f1'] for r in per_seed]):.3f} "
        f"| {time.time() - t0:.0f}s")
    return {"val_ap": float(val_ap), "params": cfg, "epochs": int(epoch), "threshold": threshold, "seeds": per_seed}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

GBDT_PLAN = [
    ("xgb", ["raw_local", "raw_all", "structural_only", "hqw_only", "raw+structural", "raw+hqw",
             "raw+hqw_gamma0", "raw+hqw_wave", "raw+structural+hqw"]),
    ("lgbm", ["raw_all", "raw+structural", "raw+hqw", "raw+hqw_wave"]),
    ("rf", ["raw_local", "raw_all", "raw+structural", "raw+hqw"]),
]


def load_results(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def save_results(path, res):
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(path, "w"), indent=1)


def report(res):
    def cell(r, k):
        v = np.array([s[k] for s in r["seeds"] if k in s])
        return f"{v.mean():.3f}+/-{v.std():.3f}" if len(v) else "n/a"

    lines = [f"{'Model':<34} | {'PR-AUC':<13} | {'F1':<13} | {'FPR@95Rec':<13} | {'PR-AUC t>=43':<13}"]
    lines.append("-" * len(lines[0]))
    for name, r in res.items():
        lines.append(f"{name:<34} | {cell(r, 'pr_auc'):<13} | {cell(r, 'f1'):<13} | "
                     f"{cell(r, 'fpr_at_95rec'):<13} | {cell(r, 'pr_auc_post_shutdown'):<13}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["features", "gbdt", "gnn", "report", "all"])
    args = ap.parse_args()
    out_path = os.path.join(RESULTS, "elliptic_results.json")

    def log(msg):
        print(msg, flush=True)

    data = load_elliptic()
    masks = data.split_masks()
    log(f"Elliptic: {data.num_nodes} nodes, {len(data.edges)} edges | train {masks['train'].sum()} "
        f"({data.labels[masks['train']].mean():.1%} illicit) | test {masks['test'].sum()} "
        f"({data.labels[masks['test']].mean():.1%} illicit)")

    res = load_results(out_path)
    if args.stage in ("features", "gbdt", "all"):
        log("Building feature sets...")
        sets = build_feature_sets(data)
        log("  " + ", ".join(f"{k}:{v.shape[1]}" for k, v in sets.items()))
    if args.stage in ("gbdt", "all"):
        for learner, names in GBDT_PLAN:
            for name in names:
                key = f"{learner}[{name}]"
                if key in res:
                    continue
                res[key] = run_gbdt(data, sets, learner, name, masks, log)
                save_results(out_path, res)
    if args.stage in ("gnn", "all"):
        for kind in ("gcn", "sage", "gat"):
            key = f"gnn[{kind}]"
            if key in res:
                continue
            res[key] = run_gnn(data, masks, kind, log)
            save_results(out_path, res)
    if args.stage in ("report", "all"):
        print("\n" + report(res))


if __name__ == "__main__":
    main()
