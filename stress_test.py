"""
Adversarial Stress Test for HQW-AML v2.

Intentionally creates the hardest possible conditions for fraud detection:

Scenario A — "Invisible Mules":
  Raw features are drawn from IDENTICAL distributions for mule and legit accounts.
  Zero signal in tabular features. Only topology can save you.

Scenario B — "Camouflaged Rings":
  Mule accounts also engage in legitimate bilateral commerce with random legit
  accounts, creating cover traffic that blurs the directed fan-out/fan-in pattern.

Scenario C — "Needle in a Haystack":
  0.5% class imbalance (1 mule per 200 accounts). Tiny mule rings (3-5 nodes)
  buried inside dense legitimate commerce clusters of 50+ nodes each.

Scenario D — "Kitchen Sink" (all of the above combined):
  Identical features + camouflage traffic + 0.5% imbalance + mixed ring sizes
  (3 to 20 nodes) + heavy random noise edges between mule and legit accounts.
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
    """
    Generates adversarial transaction graphs designed to break fraud detectors.
    """
    np.random.seed(seed)

    all_edges = []
    node_labels = []
    node_features = []
    current_id = 0
    mule_ids = []
    legit_ids = []

    if scenario == "invisible_mules":
        # 60 subgraphs, ~3% illicit, IDENTICAL feature distributions
        num_subgraphs = 60
        illicit_prob = 0.06
    elif scenario == "camouflaged_rings":
        # mules have bilateral cover traffic with legit accounts
        num_subgraphs = 60
        illicit_prob = 0.06
    elif scenario == "needle_haystack":
        # 0.5% imbalance, tiny rings inside huge legit clusters
        num_subgraphs = 80
        illicit_prob = 0.015
    elif scenario == "kitchen_sink":
        # everything combined
        num_subgraphs = 80
        illicit_prob = 0.015
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    for _ in range(num_subgraphs):
        is_illicit = np.random.rand() < illicit_prob

        if is_illicit:
            # variable ring sizes for kitchen_sink
            if scenario == "kitchen_sink":
                ring_depth = np.random.choice([2, 3, hop_depth, hop_depth + 2])
            else:
                ring_depth = hop_depth

            source = current_id
            current_id += 1
            node_labels.append(1)
            mule_ids.append(source)
            # ALL scenarios: identical feature distribution to legit
            node_features.append(np.random.randn(4) * 0.8)

            curr_layer = [source]
            for step in range(ring_depth - 1):
                fan_out = np.random.randint(2, 5) if step == 0 else np.random.randint(1, 4)
                next_layer = []
                for parent in curr_layer:
                    for _ in range(fan_out):
                        child = current_id
                        current_id += 1
                        # amounts overlap with legit range
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
            if scenario == "needle_haystack" or scenario == "kitchen_sink":
                cluster_size = np.random.randint(30, 60)
            else:
                cluster_size = hop_depth * 4

            cluster = list(range(current_id, current_id + cluster_size))
            current_id += cluster_size
            for nid in cluster:
                node_labels.append(0)
                legit_ids.append(nid)
                node_features.append(np.random.randn(4) * 0.8)

            # dense bilateral commerce within cluster
            for _ in range(cluster_size * 3):
                u = np.random.choice(cluster)
                v = np.random.choice(cluster)
                if u != v:
                    amt = float(np.random.uniform(10.0, 2000.0))
                    all_edges.append((u, v, amt))
                    # bilateral: return transaction
                    if np.random.rand() < 0.4:
                        all_edges.append((v, u, float(np.random.uniform(10.0, amt))))

    total = current_id

    # camouflage: mule accounts also transact bilaterally with legit accounts
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

    # heavy background cross-traffic
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
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x, norm_adj):
        for i, layer in enumerate(self.layers):
            x = norm_adj @ x
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


def eval_metrics(y_true, y_prob):
    """Returns (PR-AUC, F1) from true labels and predicted probabilities."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall, precision))
    pred = (y_prob > 0.5).astype(int)
    f1 = float(f1_score(y_true, pred, zero_division=0))
    return pr_auc, f1


def eval_tabular(features, labels, train_mask, test_mask):
    clf = HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=42)
    clf.fit(features[train_mask], labels[train_mask])
    probs = clf.predict_proba(features[test_mask])[:, 1]
    return eval_metrics(labels[test_mask], probs)


def eval_gcn(num_nodes, edges, features, labels, num_layers, train_mask, test_mask):
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
    pos = max(np.sum(labels[train_mask] == 1), 1)
    neg = max(np.sum(labels[train_mask] == 0), 1)
    weight = torch.tensor([1.0, float(neg / pos)])
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    model.train()
    for _ in range(100):
        opt.zero_grad()
        out = model(x, a)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x, a), dim=1)[:, 1].numpy()
    return eval_metrics(labels[test_mask], probs[test_mask])


def eval_hqw(num_nodes, edges, features, labels, train_mask, test_mask):
    clf = HQWAMLClassifier(max_iter=100, random_state=42)
    clf.fit(num_nodes, edges, features, labels, train_mask=train_mask)
    probs = clf.predict_proba(num_nodes, edges, features, eval_mask=test_mask)[:, 1]
    return eval_metrics(labels[test_mask], probs)


def run_stress_tests():
    scenarios = [
        ("invisible_mules",   "A: Invisible Mules (identical features)"),
        ("camouflaged_rings", "B: Camouflaged Rings (bilateral cover traffic)"),
        ("needle_haystack",   "C: Needle in Haystack (0.5% imbalance, tiny rings)"),
        ("kitchen_sink",      "D: Kitchen Sink (all combined)"),
    ]
    depths = [2, 4, 6]

    print("=" * 90)
    print("HQW-AML v2 ADVERSARIAL STRESS TEST SUITE")
    print("=" * 90)

    for scenario_key, scenario_name in scenarios:
        print(f"\n{'─' * 90}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'─' * 90}")

        results = {}
        for depth in depths:
            data = generate_stress_graph(scenario_key, hop_depth=depth, seed=42 + depth)
            stats = data["stats"]
            print(f"\n  {depth}-hop: {stats['total_nodes']} nodes, {stats['total_edges']} edges, "
                  f"{stats['illicit_nodes']} illicit ({stats['illicit_pct']}%)")

            n = data["num_nodes"]
            idx = np.arange(n)
            np.random.seed(99)
            np.random.shuffle(idx)
            split = int(0.65 * n)
            train_mask = np.zeros(n, dtype=bool)
            test_mask = np.zeros(n, dtype=bool)
            train_mask[idx[:split]] = True
            test_mask[idx[split:]] = True

            tab_pr, tab_f1 = eval_tabular(data["features"], data["labels"], train_mask, test_mask)
            gcn_pr, gcn_f1 = eval_gcn(
                n, data["edges"], data["features"], data["labels"],
                depth, train_mask, test_mask
            )

            t0 = time.time()
            hqw_pr, hqw_f1 = eval_hqw(
                n, data["edges"], data["features"], data["labels"],
                train_mask, test_mask
            )
            elapsed = time.time() - t0

            results[depth] = {
                "tab": (tab_pr, tab_f1),
                "gcn": (gcn_pr, gcn_f1),
                "hqw": (hqw_pr, hqw_f1),
            }
            print(f"  Tabular:  PR-AUC={tab_pr:.4f}  F1={tab_f1:.4f}")
            print(f"  GCN:      PR-AUC={gcn_pr:.4f}  F1={gcn_f1:.4f}")
            print(f"  HQW-AML:  PR-AUC={hqw_pr:.4f}  F1={hqw_f1:.4f}  ({elapsed:.1f}s)")

        print(f"\n  {'Model':<22} | {'2-Hop PR-AUC':>12} | {'4-Hop PR-AUC':>12} | {'6-Hop PR-AUC':>12}")
        print(f"  {'-'*22}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
        for label, key in [("Tabular (GBDT)", "tab"), ("Classical GCN", "gcn"), ("HQW-AML v2", "hqw")]:
            vals = [results[d][key][0] for d in depths]
            print(f"  {label:<22} | {vals[0]:>12.4f} | {vals[1]:>12.4f} | {vals[2]:>12.4f}")

    print(f"\n{'=' * 90}")
    print("STRESS TEST COMPLETE")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    run_stress_tests()
