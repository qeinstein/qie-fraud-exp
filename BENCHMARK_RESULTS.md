# HQW-AML Benchmark Results Tracker

All results measured on synthetic AML transaction graphs with overlapping
raw features, background commercial cross-traffic, and class imbalance.

Metrics: **PR-AUC** (Precision-Recall Area Under Curve) on the illicit minority class.

---

## Run 1 — v1 Engine (2026-10-12, initial implementation)

Synthetic data: 60 subgraphs, ~2% illicit, distinct raw feature distributions
(mules had clearly different feature means from legit accounts).

| Model Architecture     | 2-Hop PR-AUC | 4-Hop PR-AUC | 6-Hop PR-AUC | Depth Drop (%) |
|------------------------|:------------:|:------------:|:------------:|:--------------:|
| Tabular Baseline (GBDT)| 0.993        | 0.980        | 0.977        | 1.6%           |
| Classical GCN          | 1.000        | 1.000        | 1.000        | 0.0%           |
| HQW-AML v1             | 1.000        | 0.989        | 0.995        | 0.5%           |

> **Verdict:** Features were too easy to separate. All models scored near-perfect.

---

## Run 2 — v1 Engine (2026-10-12, realistic feature overlap + cross-traffic)

Synthetic data: 60 subgraphs, ~2% illicit, overlapping raw features
(mule and legit accounts drawn from nearly identical distributions),
plus random background cross-traffic edges.

| Model Architecture     | 2-Hop PR-AUC | 4-Hop PR-AUC | 6-Hop PR-AUC | Depth Drop (%) |
|------------------------|:------------:|:------------:|:------------:|:--------------:|
| Tabular Baseline (GBDT)| 0.039        | 0.040        | 0.328        | -736.3%        |
| Classical GCN          | 0.346        | 0.896        | 0.932        | -169.2%        |
| HQW-AML v1             | 0.723        | 0.893        | 0.995        | -37.7%         |

> **Verdict:** When features overlap, tabular models collapse to random.
> GCN struggles at shallow depths. v1 is best but still weak at 2-hop (0.723).

---

## Run 3 — v2 Engine (2026-10-12, same realistic data as Run 2)

v2 fixes: log-normalized amounts, spectral-adaptive time scales,
quantum-classical interference contrast, spectral participation ratio,
coherent temporal variance, on-site potentials from node features.

| Model Architecture     | 2-Hop PR-AUC | 4-Hop PR-AUC | 6-Hop PR-AUC | Depth Drop (%) |
|------------------------|:------------:|:------------:|:------------:|:--------------:|
| Tabular Baseline (GBDT)| 0.039        | 0.040        | 0.328        | -736.3%        |
| Classical GCN          | 0.346        | 0.896        | 0.932        | -169.2%        |
| HQW-AML v2             | 1.000        | 1.000        | 1.000        | 0.0%           |

> **Verdict:** Perfect separation across all depth regimes.
> 2-hop jumped from 0.723 → 1.000 thanks to interference contrast features.

---

## Run 4 — v2 Engine, Adversarial Stress Test (pending)

Stress scenarios:
- Identical feature distributions (zero signal in raw features)
- Mule rings embedded inside dense legitimate commerce (not isolated)
- 0.5% class imbalance (4x harder than previous)
- Camouflaged mules with bidirectional cover traffic
- Mixed ring sizes (3-node to 20+ node)
- Heavy noise edges between mule and legit accounts

| Model Architecture     | 2-Hop PR-AUC | 4-Hop PR-AUC | 6-Hop PR-AUC | Depth Drop (%) |
|------------------------|:------------:|:------------:|:------------:|:--------------:|
| Tabular Baseline (GBDT)| —            | —            | —            | —              |
| Classical GCN          | —            | —            | —            | —              |
| HQW-AML v2             | —            | —            | —            | —              |

> **Verdict:** Pending.
