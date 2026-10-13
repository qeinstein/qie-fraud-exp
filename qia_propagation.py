"""
HQW-AML v4: quantum-walk risk propagation.

Instead of describing graph structure alone, a first-stage classifier assigns every node a
risk score r (out-of-fold for training nodes) and the continuous-time quantum walk carries that
signal across the directed graph: psi(t) = exp(-i H t) r, with the chiral Hamiltonian
H = A_sym + i*gamma*(A - A^T) + potential wells from qia_engine.build_hamiltonian.
The evolution uses sparse Krylov expm_multiply, so no eigendecomposition and no mode truncation.

Per node and time t we emit Re psi, Im psi and the arrival probability |psi|^2 (rms-normalized).
The imaginary skew term makes these direction-aware. Classical counterparts (neighbour means and
heat-kernel diffusion of the same risk) are provided so the benefit of the quantum dynamics can be
isolated from the benefit of propagating risk at all.
"""

from typing import List, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from qia_engine import build_hamiltonian, build_adjacency, build_sym_laplacian

DEFAULT_TIMES = (0.5, 1.0, 2.0, 4.0)


def _rms_normalize(r: np.ndarray) -> np.ndarray:
    return r / (np.sqrt(np.mean(r ** 2)) + 1e-9)


def quantum_risk_features(
    num_nodes: int,
    edges: List,
    risk: np.ndarray,
    gamma: float = 1.0,
    flow_potential_weight: float = 0.35,
    times: Sequence[float] = DEFAULT_TIMES,
) -> np.ndarray:
    """Re/Im/|.|^2 of exp(-iHt) applied to the node risk vector, for each t (3 * len(times) columns)."""
    H, _ = build_hamiltonian(num_nodes, edges, None, gamma=gamma,
                             flow_potential_weight=flow_potential_weight, sparse_threshold=0)
    z0 = _rms_normalize(risk).astype(np.complex128)
    cols = []
    for t in times:
        psi = expm_multiply(-1j * t * H, z0)
        cols += [psi.real, psi.imag, np.abs(psi) ** 2]
    return np.column_stack(cols)


def neighbour_risk_features(num_nodes: int, edges: List, risk: np.ndarray) -> np.ndarray:
    """Classical: in/out-neighbour mean and max risk, 2-hop mean (6 columns)."""
    adj = (build_adjacency(num_nodes, edges) > 0).astype(np.float64)
    in_deg = np.asarray(adj.sum(axis=0)).ravel()
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    in_mean = (adj.T @ risk) / np.maximum(in_deg, 1.0)
    out_mean = (adj @ risk) / np.maximum(out_deg, 1.0)
    sym = ((adj + adj.T) > 0).astype(np.float64).tocsr()
    sym_deg = np.asarray(sym.sum(axis=1)).ravel()
    two_hop = (sym @ ((sym @ risk) / np.maximum(sym_deg, 1.0))) / np.maximum(sym_deg, 1.0)

    def row_max(m, v):
        m = m.tocsr()
        out = np.zeros(num_nodes)
        for i in range(num_nodes):
            s, e = m.indptr[i], m.indptr[i + 1]
            if e > s:
                out[i] = v[m.indices[s:e]].max()
        return out

    return np.column_stack([in_mean, out_mean, two_hop, row_max(adj, risk), row_max(adj.T, risk),
                            row_max(sym, risk)])


def heat_risk_features(num_nodes: int, edges: List, risk: np.ndarray,
                       times: Sequence[float] = DEFAULT_TIMES) -> np.ndarray:
    """Classical: random-walk heat-kernel diffusion exp(-t (I - D^-1 A_sym)) r (len(times) columns)."""
    adj = (build_adjacency(num_nodes, edges) > 0).astype(np.float64)
    sym = ((adj + adj.T) > 0).astype(np.float64).tocsr()
    deg = np.asarray(sym.sum(axis=1)).ravel()
    walk = sp.diags(1.0 / np.maximum(deg, 1.0)) @ sym
    lap = sp.identity(num_nodes) - walk
    return np.column_stack([expm_multiply(-t * lap, risk) for t in times])
