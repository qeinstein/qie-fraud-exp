"""
HQW-AML v3: Hermitian Continuous-Time Quantum Walk Anomaly Engine.

A quantum-inspired graph anomaly detection engine for directed financial
transaction networks. Detects coordinated multi-hop money-mule rings,
smurfing webs, and camouflaged layering using:

1. Flow-Imbalance On-Site Potential Wells:
   Diagonal potentials act as attractive potential wells at cash-out sinks
   (Inflow >> Outflow) and repulsive barriers at victim sources (Inflow << Outflow).
   NOTE: the potential is a function of classical in/out flow, so a fair baseline
   must also be given flow-imbalance features (see eval_utils.structural_features).
2. Local Counterparty Interference Contrast:
   Quantum vs classical diffusion contrast evaluated over direct transaction
   counterparties only (O(E * K), no N x N matrix needed).
3. Chiral Directional Asymmetry:
   Imaginary antisymmetric phases (H = A_sym + i*gamma*(A - A^T)) encode direction.
   Set gamma=0 to ablate chirality.
4. Adaptive Eigensolving:
   Exact dense decomposition for N <= sparse_threshold (default 1200). Above it,
   the adjacency/Hamiltonian is kept sparse and only the k_modes largest-magnitude
   modes of H (and smoothest modes of the Laplacian) are computed via Lanczos.
   In that regime every spectral feature is a rank-k approximation, so feature
   distributions are not identical to the dense regime: train and score in the
   same regime. Memory is O(E + N*k); the dispersion-entropy feature is exact up to
   2048 nodes and column-sampled beyond that.
"""

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

SPARSE_THRESHOLD = 1200
K_MODES = 96
DISPERSION_MAX_COLS = 2048


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #

def _edge_arrays(num_nodes: int, edges: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalizes (u, v) / (u, v, w) tuples into valid, positive-weight edge arrays."""
    us, vs, ws = [], [], []
    for edge in edges:
        if len(edge) == 2:
            u, v, w = int(edge[0]), int(edge[1]), 1.0
        else:
            u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
        if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v and w > 0:
            us.append(u)
            vs.append(v)
            ws.append(w)
    return (np.array(us, dtype=np.int64), np.array(vs, dtype=np.int64),
            np.array(ws, dtype=np.float64))


def build_adjacency(num_nodes: int, edges: List) -> sp.csr_matrix:
    """Sparse directed amount-weighted adjacency; duplicate edges are summed."""
    u, v, w = _edge_arrays(num_nodes, edges)
    return sp.coo_matrix((w, (u, v)), shape=(num_nodes, num_nodes)).tocsr()


def _log_normalized(raw_adj: sp.csr_matrix) -> sp.csr_matrix:
    log_adj = raw_adj.copy()
    log_adj.data = np.log1p(log_adj.data)
    max_log = log_adj.data.max() if log_adj.nnz else 0.0
    if max_log > 0:
        log_adj.data /= max_log
    return log_adj


def build_hamiltonian(
    num_nodes: int,
    edges: List,
    raw_node_features: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    flow_potential_weight: float = 0.35,
    feature_potential_weight: float = 0.1,
    sparse_threshold: int = SPARSE_THRESHOLD
):
    """
    Constructs an amount-normalized Hermitian Hamiltonian with flow-imbalance potential wells.

    H = A_sym + i*gamma*(A - A^T) + diag(V_flow + V_feat), built sparsely.

    - Log-normalized amounts prevent spectral blow-up while keeping magnitude hierarchies.
    - V_flow = -flow_potential_weight * (In - Out) / (In + Out + eps): sinks become
      attractive wells (V < 0), sources repulsive barriers (V > 0).
    - V_feat shapes the landscape from normalized raw feature norms.

    Returns (H, raw_adj): H is a dense ndarray when N <= sparse_threshold, otherwise a
    scipy CSR matrix. raw_adj is always a sparse CSR matrix of raw amounts.
    """
    raw_adj = build_adjacency(num_nodes, edges)
    log_adj = _log_normalized(raw_adj)

    a_sym = 0.5 * (log_adj + log_adj.T)
    a_skew = log_adj - log_adj.T
    H = (a_sym.astype(np.complex128) + 1j * gamma * a_skew).tocsr()

    in_flow = np.asarray(log_adj.sum(axis=0)).ravel()
    out_flow = np.asarray(log_adj.sum(axis=1)).ravel()
    flow_imbalance = (in_flow - out_flow) / (in_flow + out_flow + 1e-6)
    v_diag = -flow_potential_weight * flow_imbalance

    if raw_node_features is not None and raw_node_features.shape[0] == num_nodes:
        norms = np.linalg.norm(raw_node_features, axis=1)
        max_norm = np.max(norms)
        if max_norm > 0:
            v_diag = v_diag + feature_potential_weight * (norms / max_norm)

    H = (H + sp.diags(v_diag.astype(np.complex128))).tocsr()
    if num_nodes <= sparse_threshold:
        H = H.toarray()
    return H, raw_adj


# --------------------------------------------------------------------------- #
# Spectral machinery
# --------------------------------------------------------------------------- #

def spectral_decompose(
    H,
    sparse_threshold: int = SPARSE_THRESHOLD,
    k_modes: int = K_MODES
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact eigh for N <= sparse_threshold; otherwise Lanczos (eigsh) for the k_modes
    largest-magnitude eigenpairs of the sparse Hamiltonian.
    """
    N = H.shape[0]
    if N <= sparse_threshold:
        dense = H.toarray() if sp.issparse(H) else H
        return np.linalg.eigh(dense)

    H_sparse = sp.csr_matrix(H)
    actual_k = min(k_modes, N - 2)
    eigenvalues, eigenvectors = spla.eigsh(H_sparse, k=actual_k, which='LM')
    sort_idx = np.argsort(eigenvalues)
    return eigenvalues[sort_idx], eigenvectors[:, sort_idx]


def adaptive_time_scales(eigenvalues: np.ndarray, num_steps: int = 6) -> np.ndarray:
    """Computes spectrally-adapted evolution times based on spectral bandwidth."""
    bandwidth = float(eigenvalues[-1] - eigenvalues[0])
    if bandwidth < 1e-8:
        return np.array([0.5, 1.0, 2.0])
    t_fundamental = 2.0 * np.pi / bandwidth
    return np.linspace(t_fundamental / num_steps, t_fundamental * 1.5, num_steps)


def quantum_transition_matrix(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    t: float
) -> np.ndarray:
    """Computes the dense P(t) where P[v,u] = |<v|exp(-iHt)|u>|^2 (small graphs only)."""
    phase = np.exp(-1j * eigenvalues * t)
    u_t = eigenvectors @ (phase[:, None] * eigenvectors.conj().T)
    return np.abs(u_t) ** 2


def build_sym_laplacian(raw_adj: sp.csr_matrix) -> sp.csr_matrix:
    """Sparse combinatorial Laplacian of the undirected, unweighted skeleton."""
    binary = (raw_adj > 0).astype(np.float64)
    sym_adj = 0.5 * (binary + binary.T)
    degrees = np.asarray(sym_adj.sum(axis=1)).ravel()
    return (sp.diags(degrees) - sym_adj).tocsr()


def decompose_classical_laplacian(
    sym_laplacian,
    sparse_threshold: int = SPARSE_THRESHOLD,
    k_modes: int = K_MODES
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spectrum of the symmetric Laplacian. Large graphs compute the smoothest modes (those that
    dominate the heat kernel) as the top modes of the shifted operator c*I - L, which is far
    faster than shift-invert (LU fill-in) and sturdier than which='SM'.
    """
    N = sym_laplacian.shape[0]
    if N <= sparse_threshold:
        dense = sym_laplacian.toarray() if sp.issparse(sym_laplacian) else sym_laplacian
        return np.linalg.eigh(dense)

    L_sparse = sp.csr_matrix(sym_laplacian)
    actual_k = min(k_modes, N - 2)
    shift = 2.0 * float(L_sparse.diagonal().max()) + 1.0  # Gershgorin bound on lambda_max
    shifted = (shift * sp.identity(N) - L_sparse).tocsr()
    w, V = spla.eigsh(shifted, k=actual_k, which='LA', tol=1e-4)
    w = shift - w
    order = np.argsort(w)
    w, V = w[order], V[:, order]
    return np.clip(w, 0.0, None), V


def compute_local_interference_contrast(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    laplacian_spectrum: Tuple[np.ndarray, np.ndarray],
    raw_adj: sp.csr_matrix,
    time_scales: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Local Counterparty Interference Contrast, evaluated strictly over direct
    transaction counterparties in O(E * K) per time step.

    Returns:
    - local_sink_ratio: max_t (Q_in / (C_in + eps))
    - local_sink_excess: max_t (Q_in - C_in)
    - local_source_excess: max_t (Q_out - C_out)
    - contrast_features: (N, num_time_steps) time-resolved contrast profiles
    """
    N = eigenvectors.shape[0]
    coo = raw_adj.tocoo()
    u_idx, v_idx = coo.row, coo.col
    in_degrees = np.bincount(v_idx, minlength=N).astype(float) + 1e-6
    out_degrees = np.bincount(u_idx, minlength=N).astype(float) + 1e-6

    max_q_in = np.zeros(N)
    max_c_in = np.zeros(N)
    max_excess_in = np.zeros(N)
    max_excess_out = np.zeros(N)
    contrasts = []

    w_c, V_c = laplacian_spectrum

    for t in time_scales:
        phase = np.exp(-1j * eigenvalues * t)
        v_ph = eigenvectors * phase
        u_uv = np.sum(v_ph[u_idx] * eigenvectors.conj()[v_idx], axis=1)
        u_vu = np.sum(v_ph[v_idx] * eigenvectors.conj()[u_idx], axis=1)
        q_in = np.bincount(v_idx, weights=np.abs(u_uv) ** 2, minlength=N) / in_degrees
        q_out = np.bincount(u_idx, weights=np.abs(u_vu) ** 2, minlength=N) / out_degrees

        phase_c = np.exp(-w_c * t)
        vc_ph = V_c * phase_c
        u_c = np.clip(np.sum(vc_ph[u_idx] * V_c[v_idx], axis=1), 0.0, None)
        c_in = np.bincount(v_idx, weights=u_c, minlength=N) / in_degrees
        c_out = np.bincount(u_idx, weights=u_c, minlength=N) / out_degrees

        max_q_in = np.maximum(max_q_in, q_in)
        max_c_in = np.maximum(max_c_in, c_in)
        max_excess_in = np.maximum(max_excess_in, q_in - c_in)
        max_excess_out = np.maximum(max_excess_out, q_out - c_out)
        contrasts.append(q_in - c_in)

    local_sink_ratio = max_q_in / (max_c_in + 1e-6)
    contrast_features = np.array(contrasts).T

    return local_sink_ratio, max_excess_in, max_excess_out, contrast_features


def compute_spectral_participation_ratio(eigenvectors: np.ndarray) -> np.ndarray:
    """
    Per-node inverse participation ratio across the K retained modes, normalized by K
    (K == N for the dense solver) so the value is comparable between regimes.
    """
    K = eigenvectors.shape[1]
    ipr = np.sum(np.abs(eigenvectors) ** 4, axis=1)
    return 1.0 / (K * ipr + 1e-12)


def compute_coherent_temporal_variance(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    time_scales: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Variance and non-zero peak return probabilities over time via O(N * K) diagonal projection."""
    abs_v2 = np.abs(eigenvectors) ** 2
    returns = []
    for t in time_scales:
        diag_u = abs_v2 @ np.exp(-1j * eigenvalues * t)
        returns.append(np.abs(diag_u) ** 2)
    returns = np.array(returns)
    ctv = np.var(returns, axis=0)
    peak_return = np.max(returns[1:], axis=0) if len(returns) > 1 else returns[0]
    return ctv, peak_return


def compute_quantum_amplification_ratio(
    eigenvectors: np.ndarray,
    raw_adj: sp.csr_matrix
) -> np.ndarray:
    """Faccin et al. (PRX 2013) Quantum Amplification Ratio (normalized by retained modes K)."""
    N, K = eigenvectors.shape
    quantum_centrality = (1.0 / K) * np.sum(np.abs(eigenvectors) ** 4, axis=1)
    binary = (raw_adj > 0).astype(float)
    degrees = np.asarray(binary.sum(axis=1)).ravel() + np.asarray(binary.sum(axis=0)).ravel()
    total = np.sum(degrees)
    classical = np.ones(N) / N if total == 0 else degrees / total
    return quantum_centrality / (classical + 1e-8)


def compute_weighted_chiral_flow_imbalance(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    raw_adj: sp.csr_matrix,
    t: float
) -> np.ndarray:
    """Amount-weighted directional quantum flux asymmetry computed on the sparse adjacency."""
    N = eigenvectors.shape[0]
    log_adj = _log_normalized(raw_adj)
    comb = (0.5 * (log_adj + log_adj.T)).tocoo()

    phase = np.exp(-1j * eigenvalues * t)
    v_ph = eigenvectors * phase
    u_uv = np.sum(v_ph[comb.row] * eigenvectors.conj()[comb.col], axis=1)
    u_vu = np.sum(v_ph[comb.col] * eigenvectors.conj()[comb.row], axis=1)
    p_uv = np.abs(u_uv) ** 2
    p_vu = np.abs(u_vu) ** 2

    return np.bincount(comb.col, weights=comb.data * (p_vu - p_uv), minlength=N)


def von_neumann_entropy_deficit_from_spectrum(w: np.ndarray, beta: float = 1.0) -> float:
    """Normalized Von Neumann entropy deficit from Laplacian eigenvalues (K retained modes)."""
    n_modes = len(w)
    if n_modes <= 1:
        return 0.0
    exp_w = np.exp(-beta * (w - np.min(w)))
    trace = np.sum(exp_w)
    if trace <= 0:
        return 0.0
    probs = exp_w / trace
    probs = probs[probs > 1e-12]
    entropy = -float(np.sum(probs * np.log2(probs)))
    return 1.0 - entropy / np.log2(n_modes)


def compute_von_neumann_entropy_deficit(
    laplacian,
    beta: float = 1.0,
    sparse_threshold: int = SPARSE_THRESHOLD
) -> float:
    """Entropy deficit of a Laplacian (decomposes it; prefer the _from_spectrum variant)."""
    if laplacian.shape[0] <= 1:
        return 0.0
    w, _ = decompose_classical_laplacian(laplacian, sparse_threshold)
    return von_neumann_entropy_deficit_from_spectrum(w, beta)


# --------------------------------------------------------------------------- #
# Feature extraction (single shared path for training and inference)
# --------------------------------------------------------------------------- #

def compute_spectral_context(
    num_nodes: int,
    edges: List,
    raw_node_features: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    num_time_steps: int = 6,
    flow_potential_weight: float = 0.35,
    feature_potential_weight: float = 0.1,
    sparse_threshold: int = SPARSE_THRESHOLD,
    k_modes: int = K_MODES
) -> Dict:
    """Builds H, decomposes it once, and returns everything the feature/score code needs."""
    H, raw_adj = build_hamiltonian(
        num_nodes, edges, raw_node_features,
        gamma=gamma,
        flow_potential_weight=flow_potential_weight,
        feature_potential_weight=feature_potential_weight,
        sparse_threshold=sparse_threshold
    )
    eigenvalues, eigenvectors = spectral_decompose(H, sparse_threshold, k_modes)
    del H
    times = adaptive_time_scales(eigenvalues, num_time_steps)
    sym_laplacian = build_sym_laplacian(raw_adj)
    lap_spectrum = decompose_classical_laplacian(sym_laplacian, sparse_threshold, k_modes)
    return {
        "num_nodes": num_nodes,
        "raw_adj": raw_adj,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "times": times,
        "t_res": times[len(times) // 2],
        "laplacian_spectrum": lap_spectrum,
    }


def features_from_context(ctx: Dict, raw_node_features: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-node quantum invariants and unitary wave features from a spectral context."""
    num_nodes = ctx["num_nodes"]
    raw_adj = ctx["raw_adj"]
    eigenvalues, eigenvectors = ctx["eigenvalues"], ctx["eigenvectors"]
    times, t_res = ctx["times"], ctx["t_res"]

    qar = compute_quantum_amplification_ratio(eigenvectors, raw_adj)
    loc_sink_ratio, loc_excess_in, loc_excess_out, contrast_times = compute_local_interference_contrast(
        eigenvalues, eigenvectors, ctx["laplacian_spectrum"], raw_adj, times
    )
    spr = compute_spectral_participation_ratio(eigenvectors)
    ctv, peak_return = compute_coherent_temporal_variance(eigenvalues, eigenvectors, times)
    wcfi = compute_weighted_chiral_flow_imbalance(eigenvalues, eigenvectors, raw_adj, t_res)

    entropy_deficit = von_neumann_entropy_deficit_from_spectrum(ctx["laplacian_spectrum"][0])
    entropy_col = np.full(num_nodes, entropy_deficit)

    abs_v2 = np.abs(eigenvectors) ** 2
    phase_res = np.exp(-1j * eigenvalues * t_res)
    self_return_res = np.abs(abs_v2 @ phase_res) ** 2

    # Dispersion entropy of the walker's spread. Exact (chunked) for N <= DISPERSION_MAX_COLS;
    # above that, rows are evaluated on a fixed random subset of target nodes and
    # renormalized, turning O(N^2 * K) into O(N * DISPERSION_MAX_COLS * K).
    if num_nodes > DISPERSION_MAX_COLS:
        cols = np.sort(np.random.default_rng(0).choice(num_nodes, DISPERSION_MAX_COLS, replace=False))
    else:
        cols = slice(None)
    v_cols_h = eigenvectors[cols].conj().T
    dispersion_entropy = np.zeros(num_nodes)
    chunk_size = 500
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        u_chunk = eigenvectors[start:end] @ (phase_res[:, None] * v_cols_h)
        p_chunk = np.abs(u_chunk) ** 2 + 1e-12
        p_chunk /= p_chunk.sum(axis=1, keepdims=True)
        dispersion_entropy[start:end] = -np.sum(p_chunk * np.log2(p_chunk), axis=1)

    blocks = [
        qar[:, None],
        loc_sink_ratio[:, None],
        loc_excess_in[:, None],
        loc_excess_out[:, None],
        spr[:, None],
        ctv[:, None],
        peak_return[:, None],
        wcfi[:, None],
        entropy_col[:, None],
        self_return_res[:, None],
        dispersion_entropy[:, None],
        contrast_times,
    ]

    # O(N * K * d) unitary wave propagation without N x N materialization
    if raw_node_features is not None:
        z_0 = raw_node_features.astype(np.complex128)
        blocks.append(raw_node_features)
        v_dagger_z0 = eigenvectors.conj().T @ z_0
        for t in (times[0], t_res, times[-1]):
            phase = np.exp(-1j * eigenvalues * t)
            z_t = eigenvectors @ (phase[:, None] * v_dagger_z0)
            blocks.append(np.abs(z_t))
            blocks.append(np.angle(z_t))

    return np.hstack(blocks)


def extract_features(
    num_nodes: int,
    edges: List,
    raw_node_features: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    num_time_steps: int = 6,
    flow_potential_weight: float = 0.35,
    feature_potential_weight: float = 0.1,
    sparse_threshold: int = SPARSE_THRESHOLD,
    k_modes: int = K_MODES
) -> np.ndarray:
    """Extracts quantum invariants and unitary wave features for all nodes."""
    ctx = compute_spectral_context(
        num_nodes, edges, raw_node_features,
        gamma=gamma,
        num_time_steps=num_time_steps,
        flow_potential_weight=flow_potential_weight,
        feature_potential_weight=feature_potential_weight,
        sparse_threshold=sparse_threshold,
        k_modes=k_modes
    )
    return features_from_context(ctx, raw_node_features)


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #

class HQWAMLClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid Quantum-Walk Anomaly Detector for Financial Networks (v3).

    Quantum-walk features feed a gradient-boosted decision head. Training and
    inference share one feature path (compute_spectral_context / features_from_context).
    """

    def __init__(
        self,
        gamma: float = 1.0,
        num_time_steps: int = 6,
        flow_potential_weight: float = 0.35,
        feature_potential_weight: float = 0.1,
        max_iter: int = 150,
        random_state: int = 42,
        sparse_threshold: int = SPARSE_THRESHOLD,
        k_modes: int = K_MODES
    ):
        self.gamma = gamma
        self.num_time_steps = num_time_steps
        self.flow_potential_weight = flow_potential_weight
        self.feature_potential_weight = feature_potential_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.sparse_threshold = sparse_threshold
        self.k_modes = k_modes
        self.model = None
        self._cached_X = None
        self._cached_key = None

    def _engine_kwargs(self) -> Dict:
        return dict(
            gamma=self.gamma,
            num_time_steps=self.num_time_steps,
            flow_potential_weight=self.flow_potential_weight,
            feature_potential_weight=self.feature_potential_weight,
            sparse_threshold=self.sparse_threshold,
            k_modes=self.k_modes
        )

    def _fingerprint(self, num_nodes, edges, raw_node_features) -> str:
        """Content hash of the graph, features and hyperparameters (cache key)."""
        h = hashlib.blake2b(digest_size=16)
        h.update(repr((num_nodes, sorted(self._engine_kwargs().items()))).encode())
        u, v, w = _edge_arrays(num_nodes, edges)
        for arr in (u, v, w):
            h.update(arr.tobytes())
        if raw_node_features is not None:
            feats = np.ascontiguousarray(raw_node_features, dtype=np.float64)
            h.update(repr(feats.shape).encode())
            h.update(feats.tobytes())
        return h.hexdigest()

    def _build_features(self, num_nodes, edges, raw_node_features=None):
        key = self._fingerprint(num_nodes, edges, raw_node_features)
        if self._cached_key == key and self._cached_X is not None:
            return self._cached_X
        X = extract_features(num_nodes, edges, raw_node_features, **self._engine_kwargs())
        self._cached_X = X
        self._cached_key = key
        return X

    def fit(self, num_nodes, edges, raw_node_features, labels, train_mask=None):
        X = self._build_features(num_nodes, edges, raw_node_features)
        if train_mask is not None:
            X_train, y_train = X[train_mask], labels[train_mask]
        else:
            X_train, y_train = X, labels

        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight="balanced"
        )
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, num_nodes, edges, raw_node_features=None, eval_mask=None):
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        X = self._build_features(num_nodes, edges, raw_node_features)
        if eval_mask is not None:
            X = X[eval_mask]
        return self.model.predict_proba(X)

    def predict(self, num_nodes, edges, raw_node_features=None, eval_mask=None):
        return np.argmax(
            self.predict_proba(num_nodes, edges, raw_node_features, eval_mask),
            axis=1
        )

    def score_subgraph(self, subgraph_data: Dict) -> Dict:
        """Inference endpoint for backend services and SOC console."""
        nodes = subgraph_data["nodes"]
        edges_raw = subgraph_data["edges"]
        node_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
        N = len(nodes)

        edge_tuples = []
        for e in edges_raw:
            u = node_id_to_idx.get(e["source"])
            v = node_id_to_idx.get(e["target"])
            if u is not None and v is not None:
                edge_tuples.append((u, v, float(e.get("amount", 1.0))))

        raw_feats = None
        if nodes and "features" in nodes[0]:
            raw_feats = np.array([n["features"] for n in nodes], dtype=np.float64)

        # One spectral context drives the model features, diagnostics and flux paths,
        # so inference sees exactly the feature construction used in training.
        ctx = compute_spectral_context(N, edge_tuples, raw_feats, **self._engine_kwargs())
        w, V, t_res, raw_adj = ctx["eigenvalues"], ctx["eigenvectors"], ctx["t_res"], ctx["raw_adj"]

        qar = compute_quantum_amplification_ratio(V, raw_adj)
        spr = compute_spectral_participation_ratio(V)
        wcfi = compute_weighted_chiral_flow_imbalance(w, V, raw_adj, t_res)
        entropy_deficit = von_neumann_entropy_deficit_from_spectrum(ctx["laplacian_spectrum"][0])

        if self.model is not None:
            X = features_from_context(ctx, raw_feats)
            node_risks = self.model.predict_proba(X)[:, 1]
        else:
            qar_norm = np.clip(qar / (np.max(qar) + 1e-8), 0, 1)
            wcfi_norm = np.clip(np.abs(wcfi) / (np.max(np.abs(wcfi)) + 1e-8), 0, 1)
            spr_inv = 1.0 - np.clip(spr / (np.max(spr) + 1e-8), 0, 1)
            node_risks = 0.4 * qar_norm + 0.3 * wcfi_norm + 0.3 * spr_inv

        graph_risk = float(np.max(node_risks))

        # Edge-wise flux in O(E * K): no N x N transition matrix.
        coo = raw_adj.tocoo()
        e_u, e_v, e_amt = coo.row, coo.col, coo.data
        v_ph = V * np.exp(-1j * w * t_res)
        p_vu = np.abs(np.sum(v_ph[e_v] * V.conj()[e_u], axis=1)) ** 2
        p_uv = np.abs(np.sum(v_ph[e_u] * V.conj()[e_v], axis=1)) ** 2
        fluxes = p_vu - p_uv

        suspicious_paths = []
        for u_idx, v_idx, amt, flux in zip(e_u, e_v, e_amt, fluxes):
            flux = float(flux)
            if abs(flux) > 0.05:
                suspicious_paths.append({
                    "source": nodes[u_idx]["id"],
                    "target": nodes[v_idx]["id"],
                    "quantum_flux": round(flux, 4),
                    "amount": float(amt),
                    "severity": "HIGH_EXTRACTION" if flux > 0.15 else "MODERATE"
                })
        suspicious_paths.sort(key=lambda x: abs(x["quantum_flux"]), reverse=True)

        node_results = []
        for idx, n in enumerate(nodes):
            risk = float(node_risks[idx])
            flow = float(wcfi[idx])

            if risk > 0.7 and flow > 0:
                role = "MULE_CASHOUT_SINK"
            elif risk > 0.7 and flow < 0:
                role = "COMPROMISED_SOURCE"
            elif risk > 0.4:
                role = "INTERMEDIARY_CONDUIT"
            else:
                role = "NORMAL_ACCOUNT"

            node_results.append({
                "id": n["id"],
                "anomaly_score": round(risk, 4),
                "qar": round(float(qar[idx]), 4),
                "chiral_flow": round(flow, 4),
                "spectral_localization": round(float(spr[idx]), 4),
                "role": role
            })

        return {
            "graph_risk_score": round(graph_risk, 4),
            "node_scores": node_results,
            "suspicious_paths": suspicious_paths[:10],
            "quantum_diagnostics": {
                "von_neumann_entropy_deficit": round(entropy_deficit, 4),
                "spectral_bandwidth": round(float(w[-1] - w[0]), 4),
                "resonance_time": round(float(t_res), 4),
                "num_nodes": N,
                "num_edges": int(raw_adj.nnz)
            }
        }


if __name__ == "__main__":
    print("HQW-AML v3 Self-Test")
    demo_edges = [
        (0, 1, 1000.0), (0, 2, 1000.0), (0, 3, 1000.0),
        (1, 4, 950.0),  (2, 4, 950.0),  (3, 4, 950.0),
        (5, 6, 20.0),   (6, 5, 18.0),
        (6, 7, 15.0),   (7, 6, 14.0),
    ]
    N = 8
    feats = np.random.randn(N, 4)
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0])

    clf = HQWAMLClassifier(max_iter=50)
    clf.fit(N, demo_edges, feats, labels)
    probs = clf.predict_proba(N, demo_edges, feats)

    print("Predictions computed successfully across 8 nodes.")
    payload = {
        "nodes": [{"id": f"ACC_{i}", "features": list(feats[i])} for i in range(N)],
        "edges": [{"source": f"ACC_{e[0]}", "target": f"ACC_{e[1]}", "amount": e[2]} for e in demo_edges]
    }
    result = clf.score_subgraph(payload)
    print("Graph Risk Score:", result["graph_risk_score"])
    print("Self-test complete.")
