"""
Adversarial Stress Test for HQW-AML v2 with Real-Time Progress Logging.

Evaluates fraud detection models across four adversarial scenarios:
Scenario A: Invisible Mules (identical feature distributions, pure topology)
Scenario B: Camouflaged Rings (bilateral legitimate cover transactions)
Scenario C: Needle in a Haystack (0.5% class imbalance inside large clusters)
Scenario D: Kitchen Sink (all combined: camouflage, 0.5% imbalance, noise)
"""

import time
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from qia_engine import HQWAMLClassifier


def generate_stress_graph(
    scenario: str,
    hop_depth: int = 4,
    seed: int = 42
) -> dict:
    """Generates adversarial transaction graphs designed to stress fraud detectors."""
    np.random.seed(seed)

    all_edges = []
    node_labels = []
    node_features = []
    current_id = 0
    mule_ids = []
    legit_ids = []

    if scenario == "invisible_mules":
        num_subgraphs = 60
        illicit_prob = 0.06
    elif scenario == "camouflaged_rings":
        num_subgraphs = 60
        illicit_prob = 0.06
    elif scenario == "needle_haystack":
        num_subgraphs = 80
        illicit_prob = 0.015
    elif scenario == "kitchen_sink":
        num_subgraphs = 80
        illicit_prob = 0.015
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    for _ in range(num_subgraphs):
        is_illicit = np.random.rand() < illicit_prob

        if is_illicit:
            if scenario == "kitchen_sink":
                ring_depth = int(np.random.choice([2, 3, hop_depth, hop_depth + 2]))
            else:
                ring_depth = hop_depth

            source = current_id
            current_id += 1
            node_labels.append(1)
            mule_ids.append(source)
            node_features.append(np.random.randn(4) * 0.8)

            curr_layer = [source]
            for step in range(ring_depth - 1):
                fan_out = np.random.randint(2, 5) if step == 0 else np.random.randint(1, 4)
                next_layer = []
                for parent in curr_layer:
                    for _ in range(fan_out):
                        child = current_id
                        current_id += 1
                        amt = float(np.random.uniform(50.0, 2000.0))
                        all_edges.append((parent, child, amt))
                        next_layer.append(child)
                        node_labels.append(1)
                        mule_ids.append(child)
                        node_features.append(np.random.randn(4) * 0.8)
                curr_layer = next_layer

            sink = current_id
            current_id += 1
            node_labels.append(1)
            mule_ids.append(sink)
            node_features.append(np.random.randn(4) * 0.8)
            for m in curr_layer:
                amt = float(np.random.uniform(50.0, 1500.0))
                all_edges.append((m, sink, amt))
        else:
            if scenario in ("needle_haystack", "kitchen_sink"):
                cluster_size = np.random.randint(30, 60)
            else:
                cluster_size = hop_depth * 4

            cluster = list(range(current_id, current_id + cluster_size))
            current_id += cluster_size
            for nid in cluster:
                node_labels.append(0)
                legit_ids.append(nid)
                node_features.append(np.random.randn(4) * 0.8)

            for _ in range(cluster_size * 3):
                u = np.random.choice(cluster)
                v = np.random.choice(cluster)
                if u != v:
                    amt = float(np.random.uniform(10.0, 2000.0))
                    all_edges.append((u, v, amt))
                    if np.random.rand() < 0.4:
                        all_edges.append((v, u, float(np.random.uniform(10.0, amt))))

    total = current_id

    # Bilateral cover traffic for camouflaged scenarios
    if scenario in ("camouflaged_rings", "kitchen_sink"):
        if len(mule_ids) > 0 and len(legit_ids) > 0:
            num_cover = len(mule_ids) * 3
            for _ in range(num_cover):
                m = np.random.choice(mule_ids)
                l = np.random.choice(legit_ids)
                amt = float(np.random.uniform(10.0, 500.0))
                all_edges.append((m, l, amt))
                if np.random.rand() < 0.5:
                    all_edges.append((l, m, float(np.random.uniform(10.0, 300.0))))

    # Background ambient network noise
    if len(legit_ids) > 10:
        for _ in range(total):
            u = np.random.choice(legit_ids)
            v = np.random.choice(total)
            if u != v:
                all_edges.append((u, v, float(np.random.uniform(5.0, 200.0))))

    labels = np.array(node_labels, dtype=np.int64)
    features = np.array(node_features, dtype=np.float64)

    illicit_count = int(np.sum(labels))
    illicit_pct = 100.0 * illicit_count / total

    return {
        "num_nodes": total,
        "edges": all_edges,
        "features": features,
        "labels": labels,
        "stats": {
            "total_nodes": total,
            "illicit_nodes": illicit_count,
            "illicit_pct": round(illicit_pct, 2),
            "total_edges": len(all_edges)
        }
    }


class SimpleGCN(nn.Module):
    """Classical spatial message-passing Graph Convolutional Network."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = norm_adj @ x
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Computes PR-AUC and F1-score for minority positive class."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall, precision))
    pred = (y_prob > 0.5).astype(int)
    f1 = float(f1_score(y_true, pred, zero_division=0))
    return pr_auc, f1


def eval_tabular(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray
) -> Tuple[float, float, float]:
    """Trains and evaluates Tabular GBDT baseline with duration timing."""
    t0 = time.time()
    clf = HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=42)
    clf.fit(features[train_mask], labels[train_mask])
    probs = clf.predict_proba(features[test_mask])[:, 1]
    elapsed = time.time() - t0
    pr_auc, f1 = eval_metrics(labels[test_mask], probs)
    return pr_auc, f1, elapsed


def eval_gcn(
    num_nodes: int,
    edges: list,
    features: np.ndarray,
    labels: np.ndarray,
    num_layers: int,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    log_prefix: str = ""
) -> Tuple[float, float, float]:
    """Trains and evaluates Classical GCN with periodic epoch progress updates."""
    t0 = time.time()
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v, _ in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u, v] = 1.0
            adj[v, u] = 1.0
    np.fill_diagonal(adj, 1.0)
    deg = np.sum(adj, axis=1)
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    norm_adj = np.diag(inv_sqrt) @ adj @ np.diag(inv_sqrt)

    x = torch.tensor(features, dtype=torch.float32)
    a = torch.tensor(norm_adj, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    model = SimpleGCN(features.shape[1], 32, 2, max(num_layers, 2))
    pos = max(int(np.sum(labels[train_mask] == 1)), 1)
    neg = max(int(np.sum(labels[train_mask] == 0)), 1)
    weight = torch.tensor([1.0, float(neg / pos)])
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    total_epochs = 100
    model.train()
    for epoch in range(1, total_epochs + 1):
        opt.zero_grad()
        out = model(x, a)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        if epoch in (25, 50, 75, 100):
            print(
                f"{log_prefix}    ... GCN training epoch {epoch}/{total_epochs} (loss={loss.item():.4f})",
                flush=True
            )

    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x, a), dim=1)[:, 1].numpy()
    elapsed = time.time() - t0
    pr_auc, f1 = eval_metrics(labels[test_mask], probs[test_mask])
    return pr_auc, f1, elapsed


def eval_hqw(
    num_nodes: int,
    edges: list,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    log_prefix: str = ""
) -> Tuple[float, float, float]:
    """Extracts quantum invariants and evaluates HQW-AML v2 engine with timing."""
    t0 = time.time()
    print(f"{log_prefix}    ... HQW-AML: constructing Hamiltonian & spectral decomposition", flush=True)
    clf = HQWAMLClassifier(max_iter=100, random_state=42)
    clf.fit(num_nodes, edges, features, labels, train_mask=train_mask)
    print(f"{log_prefix}    ... HQW-AML: evaluating inference & probability scoring", flush=True)
    probs = clf.predict_proba(num_nodes, edges, features, eval_mask=test_mask)[:, 1]
    elapsed = time.time() - t0
    pr_auc, f1 = eval_metrics(labels[test_mask], probs)
    return pr_auc, f1, elapsed


def run_stress_tests():
    """Runs the full stress testing suite with continuous terminal logging."""
    scenarios = [
        ("invisible_mules",   "A: Invisible Mules (identical feature distributions)"),
        ("camouflaged_rings", "B: Camouflaged Rings (bilateral legitimate cover traffic)"),
        ("needle_haystack",   "C: Needle in Haystack (0.5% imbalance, small rings in large clusters)"),
        ("kitchen_sink",      "D: Kitchen Sink (all stress vectors combined)"),
    ]
    depths = [2, 4, 6]
    total_evals = len(scenarios) * len(depths)
    current_eval = 0

    suite_start = time.time()
    print("HQW-AML v2 Adversarial Stress Test Suite", flush=True)
    print(f"Total workload: {len(scenarios)} scenarios x {len(depths)} hop depths = {total_evals} test runs\n", flush=True)

    summary_tables = {}

    for s_idx, (scenario_key, scenario_name) in enumerate(scenarios, start=1):
        print(f"[{s_idx}/{len(scenarios)}] SCENARIO: {scenario_name}", flush=True)
        scenario_results = {}

        for depth in depths:
            current_eval += 1
            eval_start = time.time()
            log_pfx = f"  [{current_eval}/{total_evals}]"
            remaining = total_evals - current_eval

            print(f"\n{log_pfx} Generating graph for {depth}-hop layering (scenario: {scenario_key})...", flush=True)
            data = generate_stress_graph(scenario_key, hop_depth=depth, seed=42 + depth)
            stats = data["stats"]
            print(
                f"{log_pfx} Graph constructed: {stats['total_nodes']} nodes, {stats['total_edges']} edges | "
                f"Illicit: {stats['illicit_nodes']} ({stats['illicit_pct']}%)",
                flush=True
            )

            n = data["num_nodes"]
            idx = np.arange(n)
            np.random.seed(99)
            np.random.shuffle(idx)
            split = int(0.65 * n)
            train_mask = np.zeros(n, dtype=bool)
            test_mask = np.zeros(n, dtype=bool)
            train_mask[idx[:split]] = True
            test_mask[idx[split:]] = True

            # Step 1: Tabular baseline
            print(f"{log_pfx} (1/3) Evaluating Tabular Baseline (GBDT)...", flush=True)
            tab_pr, tab_f1, tab_sec = eval_tabular(data["features"], data["labels"], train_mask, test_mask)
            print(f"{log_pfx}       Done in {tab_sec:.2f}s -> PR-AUC: {tab_pr:.4f}, F1: {tab_f1:.4f}", flush=True)

            # Step 2: Classical GCN
            print(f"{log_pfx} (2/3) Evaluating Classical GCN ({depth} layers)...", flush=True)
            gcn_pr, gcn_f1, gcn_sec = eval_gcn(
                n, data["edges"], data["features"], data["labels"],
                depth, train_mask, test_mask, log_prefix=log_pfx
            )
            print(f"{log_pfx}       Done in {gcn_sec:.2f}s -> PR-AUC: {gcn_pr:.4f}, F1: {gcn_f1:.4f}", flush=True)

            # Step 3: HQW-AML v2
            print(f"{log_pfx} (3/3) Evaluating HQW-AML v2 (Quantum-Inspired Engine)...", flush=True)
            hqw_pr, hqw_f1, hqw_sec = eval_hqw(
                n, data["edges"], data["features"], data["labels"],
                train_mask, test_mask, log_prefix=log_pfx
            )
            print(f"{log_pfx}       Done in {hqw_sec:.2f}s -> PR-AUC: {hqw_pr:.4f}, F1: {hqw_f1:.4f}", flush=True)

            eval_elapsed = time.time() - eval_start
            total_elapsed = time.time() - suite_start
            print(
                f"{log_pfx} {depth}-hop completed in {eval_elapsed:.1f}s | "
                f"Remaining runs: {remaining} | Total elapsed: {total_elapsed:.1f}s",
                flush=True
            )

            scenario_results[depth] = {
                "tab": (tab_pr, tab_f1),
                "gcn": (gcn_pr, gcn_f1),
                "hqw": (hqw_pr, hqw_f1),
            }

        summary_tables[scenario_key] = scenario_results

        # Print clean scenario checkpoint
        print(f"\n--- Checkpoint: Results for Scenario {s_idx} ({scenario_key}) ---", flush=True)
        print(f"{'Model':<22} | {'2-Hop PR-AUC':>12} | {'4-Hop PR-AUC':>12} | {'6-Hop PR-AUC':>12}", flush=True)
        for label, key in [("Tabular (GBDT)", "tab"), ("Classical GCN", "gcn"), ("HQW-AML v2", "hqw")]:
            vals = [scenario_results[d][key][0] for d in depths]
            print(f"{label:<22} | {vals[0]:>12.4f} | {vals[1]:>12.4f} | {vals[2]:>12.4f}", flush=True)
        print("", flush=True)

    suite_total_time = time.time() - suite_start
    print(f"All 4 stress test scenarios finished in {suite_total_time:.1f}s.\n", flush=True)


if __name__ == "__main__":
    run_stress_tests()
