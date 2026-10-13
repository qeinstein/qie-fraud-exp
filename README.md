Basically, what this repo is all about rn..

A quantum-inspired (continuous-time quantum walk) graph algorithm for detecting money-mule rings and
other coordinated fraud in directed transaction networks, benchmarked against classical baselines.

## Files
- `qia_engine.py` - HQW-AML engine (sparse Hamiltonian, spectral features, classifier, `score_subgraph`).
- `eval_utils.py` - metrics (PR-AUC, F1, true FPR@95% recall), group-level splits, classical structural baseline.
- `stress_test.py` - 7 adversarial scenarios x 3 depths x 3 seeds, with ablations (gamma=0, no potential wells) and a structural-statistics baseline.
- `benchmark.py` - simple single-seed 2/4/6-hop comparison.
- `scale_test.py` - runtime/memory on large random graphs.
- `BENCHMARK_RESULTS.md` - results log. `Concepts.txt`, `proofs_for_deck.txt`, `papers_read/` - background.

## Caveats
- All data is synthetic and written by us. The original plan to use Elliptic / AMLSim is not done yet.
- Above 1200 nodes the spectral features are rank-96 approximations; train and score in the same regime.
