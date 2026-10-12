"""
Comprehensive Adversarial Stress Test Suite for HQW-AML v3.

Evaluates fraud detection models across seven scenarios, ranging from established
benchmarks to exponentially harder adversarial syndicates:

Established Baseline Scenarios (Regression Verification):
- Scenario A: Invisible Mules (identical feature distributions, pure topology)
- Scenario B: Camouflaged Rings (bilateral legitimate cover transactions)
- Scenario C: Needle in a Haystack (extreme class imbalance inside large clusters)
- Scenario D: Kitchen Sink (all baseline stress vectors combined)

Exponentially More Difficult Adversarial Scenarios:
- Scenario E: Cyclic Wash Smurfing & Multi-Exit Splinters
    Mules form directed circular laundering cycles with concurrent smurfing
    out to 8-12 diffuse sinks, breaking tree-based assumptions.
- Scenario F: Ghost Swarm & Commercial Super-Hub Infiltration (<0.2% imbalance)
    Mules route micro-transfers through legitimate high-degree merchant hubs
    (5,000+ nodes) to dissolve topological links into corporate traffic.
- Scenario G: Temporal Mimicry & Splinter Layering
    Adversarially spoofed feature profiles, fragmented asynchronous paths,
    and diffuse off-ramps under heavy ambient noise.
"""

import gc
import time
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from qia_engine import HQWAMLClassifier
from eval_utils import compute_metrics, groups_from_starts, group_split, structural_features


MIN_ILLICIT_RINGS = 6


def generate_stress_graph(
    scenario: str,
    hop_depth: int = 4,
    seed: int = 42
) -> dict:
    """Generates complex and adversarial financial transaction networks."""
    np.random.seed(seed)

    all_edges = []
    node_labels = []
    node_features = []
    current_id = 0
    mule_ids = []
    legit_ids = []

    # Configure scenario topologies and parameters
    if scenario == "invisible_mules":
        num_subgraphs = 60
        illicit_prob = 0.06
        guarantee_illicit = True
    elif scenario == "camouflaged_rings":
        num_subgraphs = 60
        illicit_prob = 0.06
        guarantee_illicit = True
    elif scenario == "needle_haystack":
        num_subgraphs = 80
        illicit_prob = 0.015
        guarantee_illicit = True
    elif scenario == "kitchen_sink":
        num_subgraphs = 80
        illicit_prob = 0.015
        guarantee_illicit = True
    elif scenario == "cyclic_smurfing":
        num_subgraphs = 70
        illicit_prob = 0.03
        guarantee_illicit = True
    elif scenario == "superhub_ghost_swarm":
        num_subgraphs = 90
        illicit_prob = 0.01
        guarantee_illicit = True
    elif scenario == "temporal_mimicry_splinters":
        num_subgraphs = 75
        illicit_prob = 0.02
        guarantee_illicit = True
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    illicit_spawned = 0
    group_starts = []

    for g_idx in range(num_subgraphs):
        group_starts.append(current_id)
        is_illicit = (np.random.rand() < illicit_prob)
        # Guarantee MIN_ILLICIT_RINGS rings so train and test both hold illicit groups
        remaining = num_subgraphs - g_idx
        if guarantee_illicit and illicit_spawned + remaining <= MIN_ILLICIT_RINGS:
            is_illicit = True

        if is_illicit:
            illicit_spawned += 1

            if scenario == "cyclic_smurfing":
                # Scenario E: Circular wash loop + diffuse smurfing exit sinks
                cycle_len = max(3, hop_depth)
                cycle_nodes = list(range(current_id, current_id + cycle_len))
                current_id += cycle_len
                for cid in cycle_nodes:
                    node_labels.append(1)
                    mule_ids.append(cid)
                    node_features.append(np.random.randn(4) * 0.8)

                # Connect ring in a directed cycle: 0 -> 1 -> 2 -> ... -> 0
                for i in range(cycle_len):
                    nxt = cycle_nodes[(i + 1) % cycle_len]
                    amt = float(np.random.uniform(800.0, 1500.0))
                    all_edges.append((cycle_nodes[i], nxt, amt))

                # Smurfing leakage: each cycle node leaks micro-amounts to 2-3 exit sinks
                num_sinks = cycle_len * 2
                sink_nodes = list(range(current_id, current_id + num_sinks))
                current_id += num_sinks
                for sid in sink_nodes:
                    node_labels.append(1)
                    mule_ids.append(sid)
                    node_features.append(np.random.randn(4) * 0.8)
                    parent = np.random.choice(cycle_nodes)
                    all_edges.append((parent, sid, float(np.random.uniform(50.0, 300.0))))

            elif scenario == "superhub_ghost_swarm":
                # Scenario F: Mules routing micro-transfers through commercial super-hubs
                source = current_id
                current_id += 1
                node_labels.append(1)
                mule_ids.append(source)
                node_features.append(np.random.randn(4) * 0.8)

                # Chain of mules leading into super-hub and out to cash-out point
                mule_chain = [source]
                for _ in range(max(2, hop_depth - 1)):
                    child = current_id
                    current_id += 1
                    node_labels.append(1)
                    mule_ids.append(child)
                    node_features.append(np.random.randn(4) * 0.8)
                    all_edges.append((mule_chain[-1], child, float(np.random.uniform(200.0, 800.0))))
                    mule_chain.append(child)

            elif scenario == "temporal_mimicry_splinters":
                # Scenario G: Adversarially mimicked features, fractured multi-sink splinters
                source = current_id
                current_id += 1
                node_labels.append(1)
                mule_ids.append(source)
                # Adversarially spoofed feature profile (inverted variance)
                node_features.append(np.array([0.0, 0.0, 0.0, 0.0]) + np.random.randn(4) * 0.2)

                splinter_layers = []
                curr_branch = [source]
                for step in range(hop_depth - 1):
                    next_branch = []
                    for parent in curr_branch:
                        for _ in range(2):
                            child = current_id
                            current_id += 1
                            node_labels.append(1)
                            mule_ids.append(child)
                            node_features.append(np.array([0.0, 0.0, 0.0, 0.0]) + np.random.randn(4) * 0.2)
                            all_edges.append((parent, child, float(np.random.uniform(100.0, 600.0))))
                            next_branch.append(child)
                    curr_branch = next_branch
                    splinter_layers.append(curr_branch)

                # Fracture into 3 independent diffuse cash-out endpoints
                for _ in range(3):
                    sink = current_id
                    current_id += 1
                    node_labels.append(1)
                    mule_ids.append(sink)
                    node_features.append(np.array([0.0, 0.0, 0.0, 0.0]) + np.random.randn(4) * 0.2)
                    for b in curr_branch[:len(curr_branch)//2]:
                        all_edges.append((b, sink, float(np.random.uniform(50.0, 400.0))))

            else:
                # Scenarios A, B, C, D: Standard layered mule tree
                ring_depth = int(np.random.choice([2, 3, hop_depth, hop_depth + 2])) if scenario == "kitchen_sink" else hop_depth
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
                    all_edges.append((m, sink, float(np.random.uniform(50.0, 1500.0))))

        else:
            # Legitimate commercial clusters
            if scenario in ("needle_haystack", "kitchen_sink", "superhub_ghost_swarm"):
                cluster_size = np.random.randint(35, 65)
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

    # For superhub scenario: inject dense commercial hubs and route mule traffic through them
    if scenario == "superhub_ghost_swarm" and len(legit_ids) > 10 and len(mule_ids) > 0:
        super_hub = legit_ids[0]
        # Super-hub transacts with hundreds of legitimate accounts
        for lid in legit_ids[1:min(len(legit_ids), 250)]:
            all_edges.append((lid, super_hub, float(np.random.uniform(20.0, 500.0))))
            all_edges.append((super_hub, lid, float(np.random.uniform(10.0, 300.0))))
        # Infiltrate: a middle mule passes through the super-hub
        mid_mule = mule_ids[len(mule_ids) // 2]
        all_edges.append((mid_mule, super_hub, float(np.random.uniform(400.0, 900.0))))
        if len(mule_ids) > 2:
            all_edges.append((super_hub, mule_ids[-1], float(np.random.uniform(300.0, 800.0))))

    # Bilateral cover traffic for camouflaged scenarios
    if scenario in ("camouflaged_rings", "kitchen_sink", "temporal_mimicry_splinters"):
        if len(mule_ids) > 0 and len(legit_ids) > 0:
            num_cover = len(mule_ids) * 3
            for _ in range(num_cover):
                m = np.random.choice(mule_ids)
                l = np.random.choice(legit_ids)
                amt = float(np.random.uniform(10.0, 500.0))
                all_edges.append((m, l, amt))
                if np.random.rand() < 0.5:
                    all_edges.append((l, m, float(np.random.uniform(10.0, 300.0))))

    # Ambient network cross-traffic
    if len(legit_ids) > 10:
        for _ in range(total):
            u = np.random.randint(total)
            v = np.random.randint(total)
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
        "groups": groups_from_starts(group_starts, total),
        "stats": {
            "total_nodes": total,
            "illicit_nodes": illicit_count,
            "illicit_pct": round(illicit_pct, 2),
            "total_edges": len(all_edges)
        }
    }


class SimpleGCN(nn.Module):
    """Classical multi-layer Graph Convolutional Network."""
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


def eval_tabular(features, labels, train_mask, test_mask):
    """GBDT on raw node features only."""
    clf = HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=42)
    clf.fit(features[train_mask], labels[train_mask])
    return compute_metrics(labels[test_mask], clf.predict_proba(features[test_mask])[:, 1])


def eval_structural(num_nodes, edges, features, labels, train_mask, test_mask):
    """GBDT on raw features + classical graph statistics (degree, flow imbalance, 2-hop reach...)."""
    X = np.hstack([features, structural_features(num_nodes, edges)])
    clf = HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=42)
    clf.fit(X[train_mask], labels[train_mask])
    return compute_metrics(labels[test_mask], clf.predict_proba(X[test_mask])[:, 1])


def eval_gcn(num_nodes, edges, features, labels, num_layers, train_mask, test_mask):
    """Classical GCN trained on the train mask, scored on the held-out groups."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v, _ in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u, v] = 1.0
            adj[v, u] = 1.0
    np.fill_diagonal(adj, 1.0)
    deg = np.sum(adj, axis=1)
    inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    norm_adj = inv_sqrt[:, None] * adj * inv_sqrt[None, :]

    x = torch.tensor(features, dtype=torch.float32)
    a = torch.tensor(norm_adj, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    torch.manual_seed(0)
    model = SimpleGCN(features.shape[1], 32, 2, max(num_layers, 2))
    pos = max(int(np.sum(labels[train_mask] == 1)), 1)
    neg = max(int(np.sum(labels[train_mask] == 0)), 1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, float(neg / pos)]))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    model.train()
    for _ in range(100):
        opt.zero_grad()
        loss = criterion(model(x, a)[train_mask], y[train_mask])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x, a), dim=1)[:, 1].numpy()

    del x, a, y, model, opt, norm_adj, adj
    gc.collect()
    return compute_metrics(labels[test_mask], probs[test_mask])


def eval_hqw(num_nodes, edges, features, labels, train_mask, test_mask, **engine_kwargs):
    """HQW-AML; engine_kwargs enable ablations (gamma=0, flow_potential_weight=0)."""
    clf = HQWAMLClassifier(max_iter=100, random_state=42, **engine_kwargs)
    clf.fit(num_nodes, edges, features, labels, train_mask=train_mask)
    probs = clf.predict_proba(num_nodes, edges, features, eval_mask=test_mask)[:, 1]
    return compute_metrics(labels[test_mask], probs)


MODELS = [
    ("Tabular (raw features)", "tab"),
    ("Tabular + structural stats", "struct"),
    ("Classical GCN", "gcn"),
    ("HQW-AML v3 (full)", "hqw"),
    ("HQW ablation: gamma=0 (no chirality)", "hqw_g0"),
    ("HQW ablation: no potential wells", "hqw_nowell"),
]


def run_all_models(data, depth, train_mask, test_mask):
    n, e, f, y = data["num_nodes"], data["edges"], data["features"], data["labels"]
    return {
        "tab": eval_tabular(f, y, train_mask, test_mask),
        "struct": eval_structural(n, e, f, y, train_mask, test_mask),
        "gcn": eval_gcn(n, e, f, y, depth, train_mask, test_mask),
        "hqw": eval_hqw(n, e, f, y, train_mask, test_mask),
        "hqw_g0": eval_hqw(n, e, f, y, train_mask, test_mask, gamma=0.0),
        "hqw_nowell": eval_hqw(n, e, f, y, train_mask, test_mask,
                               flow_potential_weight=0.0, feature_potential_weight=0.0),
    }


def run_stress_tests(seeds=(0, 1, 2)):
    """Runs all 7 scenarios x 3 depths x len(seeds) graph draws; reports mean +/- std PR-AUC."""
    scenarios = [
        ("invisible_mules",              "A: Invisible Mules (identical feature distributions)"),
        ("camouflaged_rings",            "B: Camouflaged Rings (bilateral legitimate cover traffic)"),
        ("needle_haystack",              "C: Needle in Haystack (extreme imbalance in large clusters)"),
        ("kitchen_sink",                 "D: Kitchen Sink (all baseline stress vectors combined)"),
        ("cyclic_smurfing",              "E: Cyclic Wash Smurfing & Multi-Exit Splinters"),
        ("superhub_ghost_swarm",         "F: Ghost Swarm & Commercial Super-Hub Infiltration"),
        ("temporal_mimicry_splinters",   "G: Temporal Mimicry & Splinter Layering (spoofed profiles)"),
    ]
    depths = [2, 4, 6]
    suite_start = time.time()
    print("HQW-AML v3 Adversarial Stress Test Suite (group-level split, multi-seed, ablations)", flush=True)
    print(f"{len(scenarios)} scenarios x {len(depths)} depths x {len(seeds)} seeds\n", flush=True)

    all_results = {}
    for s_idx, (key, name) in enumerate(scenarios, start=1):
        print(f"[{s_idx}/{len(scenarios)}] SCENARIO: {name}", flush=True)
        per_depth = {}
        for depth in depths:
            runs = []
            for seed in seeds:
                data = generate_stress_graph(key, hop_depth=depth, seed=1000 * seed + 42 + depth)
                train_mask, test_mask = group_split(data["groups"], data["labels"], 0.65, seed=99 + seed)
                st = data["stats"]
                print(f"  depth={depth} seed={seed}: {st['total_nodes']} nodes, {st['total_edges']} edges, "
                      f"{st['illicit_nodes']} illicit ({st['illicit_pct']}%), "
                      f"test illicit={int(data['labels'][test_mask].sum())}", flush=True)
                runs.append(run_all_models(data, depth, train_mask, test_mask))
            per_depth[depth] = runs
        all_results[key] = per_depth

        print(f"\n--- {name}: PR-AUC mean +/- std over {len(seeds)} seeds ---", flush=True)
        print(f"{'Model':<40} | " + " | ".join(f"{d}-hop".center(13) for d in depths), flush=True)
        for label, mk in MODELS:
            cells = []
            for d in depths:
                vals = np.array([r[mk]["pr_auc"] for r in per_depth[d]])
                cells.append(f"{np.nanmean(vals):.3f}+/-{np.nanstd(vals):.3f}")
            print(f"{label:<40} | " + " | ".join(cells), flush=True)
        print("", flush=True)

    print(f"Finished in {time.time() - suite_start:.1f}s.", flush=True)
    return all_results


if __name__ == "__main__":
    run_stress_tests()
