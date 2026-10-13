# HQW-AML Benchmark Results Tracker

All results measured on synthetic AML transaction graphs with overlapping
raw features, background commercial cross-traffic, and class imbalance.

Metrics: **PR-AUC** (Precision-Recall Area Under Curve) on the illicit minority class.

> **Runs 1-5 are superseded by Run 6.** They used a node-level train/test split (ring-mates in both
> sets), ambient traffic that only legitimate accounts initiated (a structural tell), a trapezoidal
> PR-AUC, an "FPR" that was really 1-precision, a single seed, and no structural or ablation baselines.
> Their headline claims (e.g. "invariant to spoofed features", "scale-invariant to 50,000 nodes")
> are not supported; see Run 6.

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

## Run 4 — v2 Engine, Adversarial Stress Test (2026-10-12)

Stress scenarios evaluated on large graphs (up to 3,757 nodes, 19,525 edges):

### Scenario A: Invisible Mules (Identical Raw Feature Distributions)
Raw tabular features carry zero discriminative signal. Pure topological detection.

| Model Architecture     | 2-Hop PR-AUC | 4-Hop PR-AUC | 6-Hop PR-AUC |
|------------------------|:------------:|:------------:|:------------:|
| Tabular Baseline (GBDT)| 0.0645       | 0.1094       | 0.2435       |
| Classical GCN          | 0.4626       | 0.8696       | 0.9881       |
| HQW-AML v2             | **0.6817**   | **0.9279**   | **0.9924**   |

> **Verdict:** HQW-AML decisively beats classical models, especially in shallow 2-hop regimes (+47% PR-AUC over GCN) where GCN lacks neighborhood aggregation depth.

---

### Scenario B: Camouflaged Rings (Bilateral Legitimate Cover Traffic)
Mule accounts actively execute reciprocal cover transfers with legitimate merchants to disguise directed fan-out/fan-in flow.

| Model Architecture     | 2-Hop PR-AUC | 4-Hop PR-AUC | 6-Hop PR-AUC |
|------------------------|:------------:|:------------:|:------------:|
| Tabular Baseline (GBDT)| 0.0645       | 0.1094       | 0.2435       |
| Classical GCN          | 0.2142       | 0.3051       | 0.7500       |
| HQW-AML v2             | **0.4458**   | **0.5561**   | **0.8542**   |

> **Verdict:** Massive breakthrough. Classical GCN is severely blinded by bidirectional cover noise (dropping to 0.21 - 0.30). HQW-AML's chiral imaginary phase (`iγ(A - A^T)`) cancels reciprocal noise while amplifying net directional flow, yielding +108% higher PR-AUC at 2-hop and +65% F1 at 6-hop.

---

### Scenario C: Needle in a Haystack (Extreme ~0.5%–1.7% Imbalance in 3,600-Node Clusters)
Tiny mule rings buried inside large commercial clusters of 50+ nodes each.

| Model Architecture     | 2-Hop PR-AUC | 4-Hop PR-AUC | 6-Hop PR-AUC |
|------------------------|:------------:|:------------:|:------------:|
| Tabular Baseline (GBDT)| 0.5000*      | 0.0108       | 0.0166       |
| Classical GCN          | 0.5000*      | **0.9607**   | **0.9908**   |
| HQW-AML v2             | 0.5000*      | 0.6708       | 0.9772       |

*Note: In 2-hop, random generation yielded 0 positive illicit nodes (degenerate baseline).*

---

### Scenario D: Kitchen Sink (All Combined Stress Vectors: Noise + Cover + Imbalance)
Variable ring sizes (3 to 20 nodes), bilateral camouflage, extreme imbalance, and ambient noise across 3,757 nodes.

| Model Architecture     | 2-Hop PR-AUC | 4-Hop PR-AUC | 6-Hop PR-AUC |
|------------------------|:------------:|:------------:|:------------:|
| Tabular Baseline (GBDT)| 0.5000*      | 0.0122       | 0.0412       |
| Classical GCN          | 0.5000*      | **0.0511**   | 0.4028       |
| HQW-AML v2             | 0.5000*      | 0.0398       | **0.5880**   |

> **Verdict:** In the most adversarial 6-hop test, Classical GCN collapses to 0.4028 PR-AUC and 0.2329 F1. HQW-AML v2 achieves **0.5880 PR-AUC and 0.5781 F1 (+148% higher F1)**, proving that constructive quantum interference preserves signal through heavy topological noise.

---

## Run 5 — HQW-AML v3 Expanded Adversarial Stress Test Suite (2026-10-12, from `result2.txt`)

Evaluation of the upgraded v3 engine across **7 scenarios** and **21 test runs** (2-hop, 4-hop, and 6-hop).
Total workload executed in **83.8 seconds** (down from >15 minutes in v2) with zero memory/OOM crashes.

### Overall Performance Matrix (PR-AUC on Illicit Minority Class)

| Scenario | Model Architecture | 2-Hop PR-AUC | 4-Hop PR-AUC | 6-Hop PR-AUC | Key Advantage / Takeaway |
|:---|:---|:---:|:---:|:---:|:---|
| **A: Invisible Mules**<br>*(identical feature distributions)* | Tabular (GBDT)<br>Classical GCN<br>**HQW-AML v3** | 0.1812<br>0.5341<br>**0.8948** | 0.0820<br>0.8998<br>**0.9124** | 0.2237<br>**0.9902**<br>0.9798 | **+67.5% at 2-hop**; overcomes GCN shallow neighborhood blindness. |
| **B: Camouflaged Rings**<br>*(bilateral legitimate cover traffic)* | Tabular (GBDT)<br>Classical GCN<br>**HQW-AML v3** | 0.1812<br>0.1903<br>**0.5106** | 0.0820<br>0.2195<br>**0.5623** | 0.2237<br>0.7263<br>**0.8390** | **+168% at 2-hop, +156% at 4-hop**; chiral imaginary flux cancels bilateral noise. |
| **C: Needle in Haystack**<br>*(0.2%–1.5% extreme imbalance)* | Tabular (GBDT)<br>Classical GCN<br>**HQW-AML v3** | 0.0224<br>0.3504<br>**0.5823** | 0.0176<br>**0.7257**<br>0.6851 | 0.0439<br>**0.9638**<br>0.9341 | **+66% at 2-hop**; F1 at 4-hop is **0.7907** (HQW) vs **0.5385** (GCN). |
| **D: Kitchen Sink**<br>*(all baseline vectors combined)* | Tabular (GBDT)<br>Classical GCN<br>**HQW-AML v3** | 0.0079<br>0.0249<br>**0.0662** | 0.0047<br>0.0050<br>**0.0895** | 0.1297<br>0.6781<br>**0.8643** | **+27.4% PR-AUC & +34.8% F1 at 6-hop**; GCN collapses at shallow hops. |
| **E: Cyclic Wash Smurfing**<br>*(directed cycles + multi-exit sinks)* | Tabular (GBDT)<br>Classical GCN<br>**HQW-AML v3** | 0.0644<br>0.2924<br>**0.8325** | 0.0301<br>**1.0000**<br>0.6732 | 0.0329<br>**1.0000**<br>0.8595 | **+184.7% at 2-hop**; coherent walk captures return cycles immediately. |
| **F: Ghost Swarm Super-Hub**<br>*(merchant hub routing, <0.2% imb.)* | Tabular (GBDT)<br>Classical GCN<br>**HQW-AML v3** | 0.0050<br>0.0907<br>**0.1385** | 0.0015<br>0.4712<br>**0.5284** | 0.0024<br>**0.9437**<br>0.3726 | **+52.7% at 2-hop, +12.1% at 4-hop**; superhub dilution challenges 6-hop walk. |
| **G: Temporal Mimicry Splinters**<br>*(spoofed profiles + diffuse exits)* | Tabular (GBDT)<br>Classical GCN<br>**HQW-AML v3** | 0.4196<br>0.0334<br>**0.4784** | 0.7555<br>0.2163<br>**0.9719** | 0.7179<br>0.1839<br>**0.9340** | **+349% at 4-hop, +408% at 6-hop**; complete breakdown of Classical GCN. |

---

### Executive Insights from Run 5

1. **Decisive Quantum Advantage in Shallow Regimes (2-Hop Across All 7 Scenarios):**
   Classical GCNs consistently suffer from neighborhood under-reach at 2 hops (PR-AUCs dropping to 0.03 - 0.53). HQW-AML v3 outperforms GCN at 2-hop across **all 7 scenarios**, reaching up to **0.8948** (Invisible Mules) and **0.8325** (Cyclic Smurfing).
2. **Resilience to Adversarial Evasion (Scenarios B & G):**
   When syndicates deploy bilateral cover traffic (Scenario B) or spoof feature profiles and splinter paths (Scenario G), Classical GCN degrades severely (0.18 - 0.22 PR-AUC). HQW-AML v3 maintains near-flawless detection (**0.9340 - 0.9719 PR-AUC**) because chiral quantum interference and flow-imbalance potential wells are invariant to bilateral cover and spoofed features.
3. **Efficiency Breakthrough:**
   Total execution time dropped from >15 minutes down to **83.8 seconds** for 21 complete graph evaluations (up to 4,469 nodes and 24,000 edges per graph), validating the $O(E \cdot K)$ sparse transition kernel and zero-overhead caching.

---

## Run 6 — Corrected protocol (`result3.txt`, `stress_test.py`)

Changes vs Run 5: group-level split (whole rings/clusters held out), ambient traffic from all nodes,
>= 6 illicit rings per graph (so imbalance is higher than the nominal scenario value),
`average_precision_score` for PR-AUC, mean +/- std over 3 graph draws, plus two new baselines/ablations:
GBDT on raw features + classical structural statistics (degrees, flow imbalance, reciprocity, 2-hop reach),
and HQW with gamma=0 (no chirality) / no potential wells. Total runtime 571 s.

PR-AUC, mean over 3 seeds (std in `result3.txt`; stds are often 0.03-0.28, so many gaps are within noise):

| Scenario | Model | 2-hop | 4-hop | 6-hop |
|:---|:---|:---:|:---:|:---:|
| **A** Invisible mules | Tabular (raw) | 0.075 | 0.142 | 0.269 |
| | Tabular + structural | 0.778 | 0.902 | **0.988** |
| | GCN | 0.056 | 0.248 | 0.876 |
| | HQW full | **0.797** | **0.908** | 0.921 |
| | HQW gamma=0 | 0.530 | 0.891 | 0.946 |
| **B** Camouflaged | Tabular + structural | 0.221 | 0.620 | **0.889** |
| | GCN | 0.050 | 0.132 | 0.508 |
| | HQW full | **0.713** | **0.828** | 0.831 |
| | HQW gamma=0 | 0.453 | 0.777 | 0.821 |
| **C** Needle | Tabular + structural | 0.460 | **0.737** | **0.982** |
| | GCN | 0.012 | 0.094 | 0.826 |
| | HQW full | 0.358 | 0.354 | 0.888 |
| | HQW gamma=0 | **0.519** | 0.697 | 0.877 |
| **D** Kitchen sink | Tabular + structural | 0.277 | 0.114 | 0.454 |
| | GCN | 0.029 | 0.032 | 0.137 |
| | HQW full | 0.650 | 0.463 | **0.791** |
| | HQW gamma=0 | **0.819** | **0.574** | 0.782 |
| **E** Cyclic smurfing | Tabular + structural | 0.772 | **0.828** | 0.848 |
| | GCN | 0.157 | 0.133 | 0.375 |
| | HQW full | 0.835 | 0.740 | 0.732 |
| | HQW gamma=0 | **0.900** | 0.766 | **0.851** |
| **F** Super-hub | Tabular + structural | 0.185 | **0.317** | **0.523** |
| | GCN | 0.112 | 0.018 | 0.030 |
| | HQW full | 0.075 | 0.260 | 0.304 |
| | HQW gamma=0 | **0.356** | 0.226 | 0.437 |
| **G** Mimicry | Tabular + structural | 0.758 | 0.931 | 0.972 |
| | GCN | 0.092 | 0.099 | 0.177 |
| | HQW full | **0.876** | **0.982** | **0.987** |
| | HQW gamma=0 | 0.828 | 0.969 | 0.980 |

### What the corrected results support
1. **HQW-AML clearly beats the classical GCN** in nearly every cell (GCN collapses under a group-level split).
2. **A GBDT on simple structural statistics is a strong baseline** and matches or beats HQW in A (6-hop),
   B (6-hop), C (4/6-hop), E (4/6-hop) and F. HQW's clear wins over it are at shallow depth in B, D and G.
3. **Chirality (gamma) is not consistently helping**: gamma=0 is better or equal in C, D, E and F, and
   HQW full only clearly benefits from it in A and B 2-hop. The "chiral phase cancels bilateral noise"
   claim is supported only for B at 2-4 hops. Potential wells give no consistent gain.
4. Scenario F (super-hub) is not solved by any model.
5. All data is synthetic and authored by us; rings are still clean trees/cycles, which any structural
   feature can detect. Real-data validation (Elliptic / AMLSim) is still outstanding.

### Scalability (`scale_test.py`, random graphs, avg degree 5)
| N | E | time | peak RSS |
|---:|---:|---:|---:|
| 5,000 | 25k | 1.5 s | 0.5 GB |
| 20,000 | 100k | 12 s | 1.4 GB |
| 50,000 | 250k | 35 s | 2.3 GB |

Above 1,200 nodes spectral features are rank-96 approximations (not "sub-second", and not equivalent
to the dense regime); dispersion entropy is column-sampled above 2,048 nodes.

---

## Run 7 — Elliptic Bitcoin dataset (real data, `elliptic_benchmark.py`)

203,769 transactions, 234,355 edges, 49 timesteps. Temporal split: train timesteps 1-34
(29,894 labeled, 11.6% illicit), test 35-49 (16,670 labeled, 6.5% illicit). Test labels were never used
for tuning, early stopping or thresholds. Every model got the same recipe: random hyper-parameter
search (20 configs GBDT, 8 RF, 10 GNN) fitting on timesteps 1-27 and scoring PR-AUC on 28-34, then a
refit on 1-34, F1 threshold chosen on validation, 5 seeds (mean +/- std). HQW features were computed
per timestep graph (rank-192 spectrum, unweighted edges since Elliptic has no amounts) and fed to the
*same* tuned learners as the baselines. Raw results: `results/elliptic_results.json`.

| Model | PR-AUC | F1 | FPR@95Rec | PR-AUC (t>=43) |
|:---|:---:|:---:|:---:|:---:|
| XGBoost, local features (93) | 0.778 | 0.634 | 0.407 | 0.034 |
| XGBoost, all features (165) | 0.803 | 0.783 | 0.335 | 0.045 |
| XGBoost, structural stats only | 0.141 | 0.181 | 0.802 | 0.035 |
| XGBoost, HQW features only | 0.117 | 0.122 | 0.860 | 0.079 |
| XGBoost, raw + structural | 0.804 | 0.780 | 0.335 | 0.047 |
| **XGBoost, raw + HQW** | 0.803 | 0.793 | 0.335 | 0.049 |
| XGBoost, raw + HQW (gamma=0) | 0.803 | 0.772 | 0.333 | 0.048 |
| XGBoost, raw + HQW wave | 0.798 | 0.774 | 0.374 | 0.043 |
| XGBoost, raw + structural + HQW | 0.802 | 0.778 | 0.308 | 0.047 |
| LightGBM, all features | 0.799 | 0.814 | 0.351 | 0.043 |
| LightGBM, raw + structural | 0.800 | 0.819 | 0.349 | 0.044 |
| LightGBM, raw + HQW | 0.803 | 0.783 | 0.334 | 0.049 |
| LightGBM, raw + HQW wave | 0.795 | 0.791 | 0.417 | 0.047 |
| Random Forest, all features | 0.794 | 0.820 | 0.384 | 0.047 |
| Random Forest, raw + HQW | 0.793 | 0.823 | 0.395 | 0.045 |
| GCN | 0.640 | 0.572 | 0.687 | 0.053 |
| GraphSAGE | 0.722 | 0.648 | 0.349 | 0.052 |
| GATv2 | 0.722 | 0.668 | 0.436 | 0.065 |

(Stds are mostly 0.001-0.03; see the JSON.)

### What the real-data results support
1. **HQW-AML does not improve on tuned classical systems on Elliptic.** Adding HQW features to raw
   features leaves PR-AUC unchanged (0.803 vs 0.803 XGBoost; 0.803 vs 0.799 LightGBM; 0.793 vs 0.794
   RF). F1 moves by +-0.03 in either direction depending on the learner, within threshold noise.
2. **Topology alone is weak on Elliptic**: HQW-only 0.117 and structural-only 0.141 PR-AUC. Elliptic's
   shipped aggregated features already carry neighbourhood information, and its graphs are near-trees
   (E ~ N) with little cycle/interference structure for a quantum walk to exploit.
3. **The earlier synthetic-data conclusions do not transfer**: the quantum-vs-GCN gap seen in Run 6
   reappears only in the sense that every tabular model, with or without HQW features, beats every GNN
   (best GNN 0.722 vs 0.80). The chirality ablation (gamma=0) is indistinguishable on PR-AUC.
4. **All models collapse after the dark-market shutdown (timesteps >= 43)**: PR-AUC 0.03-0.08, the
   known concept-drift failure on this dataset. HQW-only has the highest (0.079) but with a std of
   0.055, so no claim is made.
5. Not evaluated: AMLSim / IBM AML (no public file available without an account), amount-weighted
   edges (Elliptic has none), and a full-graph (non-per-timestep) HQW run. GNN searches were limited
   to 10 configurations each.

---

## Run 8 — HQW-AML v4: quantum-walk risk propagation (Elliptic, `elliptic_v4.py`)

Idea: label homophily on Elliptic is strong (P(next tx illicit | this illicit) = 0.52 vs 0.023 after a licit
tx), so instead of structure-only features the walk `exp(-iHt)` carries a first-stage risk score across the
directed graph (sparse Krylov, no mode truncation). Risk is out-of-fold on the training period (timestep-grouped
folds) and from a train-only model on the test period; no test label is used. Same tuned XGBoost recipe as Run 7,
5 seeds. Raw results: `results/elliptic_v4_results.json`.

| Feature set (XGBoost) | PR-AUC | F1 | FPR@95Rec | PR-AUC (t>=43) |
|:---|:---:|:---:|:---:|:---:|
| raw | 0.803 | 0.783 | 0.335 | 0.045 |
| raw + risk (stacking only) | 0.802 | 0.823 | 0.446 | 0.044 |
| raw + risk + neighbour means (classical) | 0.808 | 0.817 | 0.424 | 0.049 |
| raw + risk + heat diffusion (classical) | 0.808 | 0.816 | 0.395 | 0.051 |
| raw + risk + neighbour + heat (classical) | **0.812** | 0.822 | 0.363 | 0.053 |
| raw + risk + quantum walk, gamma=1 (HQW v4) | 0.805 | 0.814 | 0.352 | 0.050 |
| raw + risk + quantum walk, gamma=0 | 0.805 | 0.806 | 0.375 | 0.046 |
| raw + risk + quantum walk, no wells | 0.803 | 0.804 | 0.381 | 0.044 |
| raw + risk + quantum + neighbour + heat | 0.810 | 0.817 | 0.379 | 0.051 |

### Findings
1. Risk propagation helps only marginally (+0.002 to +0.009 PR-AUC over raw). **The quantum walk does not beat
   classical propagation of the same risk** (0.805 vs 0.808-0.812). Chirality and potential wells add
   ~0.000-0.002, within noise.
2. **The ceiling is temporal drift, not graph information.** XGBoost on raw features gets PR-AUC 0.96 on
   timesteps 35-38 and 0.89 on 39-42, but 0.046 on 43-49 (after the dark-market shutdown; illicit rate 2.5%).
   Giving the model half of the test-period labels lifts test PR-AUC from 0.80 to 0.96, and a random split inside the
   training period scores 0.994. Any feature that does not address drift can move the aggregate score by
   at most about 0.01-0.02.

---

## Run 9 — Drift-robustness attempt: per-timestep normalization (Elliptic, `elliptic_drift.py`)

Every feature rank-transformed (or z-scored) within its own timestep (unlabeled features only, no labels used),
optionally combined with risk propagation and structural/HQW features. XGBoost, same tuning, 5 seeds.
Raw results: `results/elliptic_drift_results.json`.

| Feature set (XGBoost) | PR-AUC | F1 | PR-AUC (t>=43) |
|:---|:---:|:---:|:---:|
| raw (absolute, baseline) | **0.803** | **0.783** | 0.045 |
| raw rank-per-timestep | 0.781 | 0.476 | 0.046 |
| raw z-score-per-timestep | 0.755 | 0.423 | 0.051 |
| rank raw + rank HQW topology | 0.780 | 0.599 | 0.041 |
| rank HQW topology only | 0.130 | 0.143 | **0.023** |
| rank structural only | 0.128 | 0.160 | **0.023** |
| rank: raw + risk + quantum walk (v4) | 0.787 | 0.563 | 0.048 |
| rank: raw + risk + classical propagation | 0.791 | 0.480 | 0.063 |

### Findings
1. **Per-timestep normalization hurts** (0.803 -> 0.78 rank, 0.755 z-score) and F1 collapses (0.78 -> 0.42-0.48),
   because the validation-chosen threshold no longer transfers and absolute magnitudes carry signal; it does not
   repair the post-shutdown collapse (t>=43 stays 0.04-0.06).
2. **Topology-only features do not resist drift**: on t>=43 they score 0.023, at the 2.5% base rate (random). The earlier
   HQW-only 0.079 +/- 0.055 in Run 7 was noise.
3. Net: after two feature-side attempts (Run 8, Run 9) there is no evidence on Elliptic that quantum-walk features
   improve on tuned classical models, and the post-shutdown collapse (t>=43) is not fixed by any feature set tried.
