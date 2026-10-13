"""Measures HQW-AML feature extraction time/memory on large random transaction graphs."""
import sys
import time
import resource
import numpy as np
from qia_engine import extract_features


def run(n: int, avg_deg: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = n * avg_deg
    uv = rng.integers(0, n, (m, 2))
    amt = rng.uniform(5, 2000, m)
    edges = [(int(a), int(b), float(w)) for (a, b), w in zip(uv, amt)]
    feats = rng.normal(size=(n, 4))
    t0 = time.time()
    X = extract_features(n, edges, feats)
    dt = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB on macOS
    print(f"N={n:>6} E={m:>7} features={X.shape} finite={np.isfinite(X).all()} "
          f"time={dt:7.1f}s peak_rss={rss:7.0f}MB", flush=True)


if __name__ == "__main__":
    for n in [int(a) for a in sys.argv[1:]] or [1000, 5000, 20000]:
        run(n)
