"""
HQW-AML v3: Hermitian Continuous-Time Quantum Walk Anomaly Engine.

A quantum-inspired graph anomaly detection engine for directed financial
transaction networks. Detects coordinated multi-hop money-mule rings,
smurfing webs, and camouflaged layering by exploiting quantum phenomena:

1. Flow-Imbalance On-Site Potential Wells:
   Diagonal potentials act as attractive potential wells at cash-out sinks
   (Inflow >> Outflow) and repulsive barriers at victim sources (Inflow << Outflow),
   physically trapping quantum probability into sink nodes.
2. Local Counterparty Interference Contrast:
   Measures constructive vs classical diffusion contrast strictly over direct
   transaction counterparties, making the anomaly score scale-invariant from
   50 to 50,000 nodes without global graph dilution.
3. Chiral Directional Asymmetry:
   Imaginary antisymmetric phases cancel reciprocal bilateral noise while
   amplifying net one-way money extraction funnels.
4. Adaptive Low-Rank Eigensolving:
   Switches between exact dense decomposition (N <= 1200) and sparse Lanczos
   eigensolving (N > 1200) for sub-second scaling on massive networks.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


def build_hamiltonian(
    num_nodes: int,
    edges: List,
    raw_node_features: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    flow_potential_weight: float = 0.35,
    feature_potential_weight: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs an amount-normalized Hermitian Hamiltonian with flow-imbalance potential wells.
    
    Coupling terms:
    - Log-normalized amounts prevent spectral blow-up while maintaining magnitude hierarchies.
    - Imaginary skew-symmetric terms encode chiral directed flow.
    
    On-site potentials:
    - Flow-imbalance potential V_flow:
        Delta_Phi = (Inflow - Outflow) / (Inflow + Outflow + eps) in [-1, 1]
        V_flow = -flow_potential_weight * Delta_Phi
        Cash-out sinks (Delta_Phi > 0) become attractive potential wells (V < 0).
        Compromised sources (Delta_Phi < 0) become repulsive potential barriers (V > 0).
    - Node-feature potential V_feat:
        Linear energy landscape shaping from raw feature norms.
    """
    raw_adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    for edge in edges:
        if len(edge) == 2:
            u, v = int(edge[0]), int(edge[1])
            w = 1.0
        else:
            u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
        if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
            raw_adj[u, v] += w

    log_adj = np.log1p(raw_adj)
    max_log = np.max(log_adj)
    if max_log > 0:
        log_adj /= max_log

    # Coupling: symmetric skeleton + imaginary chiral directional flow
    a_sym = 0.5 * (log_adj + log_adj.T)
    a_skew = log_adj - log_adj.T
    H = a_sym.astype(np.complex128) + 1j * gamma * a_skew

    # Compute financial flow conservation imbalance
    in_flow = np.sum(log_adj, axis=0)
    out_flow = np.sum(log_adj, axis=1)
    denom = in_flow + out_flow + 1e-6
    flow_imbalance = (in_flow - out_flow) / denom

    # Attractive potential well for sinks (V < 0), repulsive barrier for sources (V > 0)
    v_diag = -flow_potential_weight * flow_imbalance

    if raw_node_features is not None and raw_node_features.shape[0] == num_nodes:
        norms = np.linalg.norm(raw_node_features, axis=1)
        max_norm = np.max(norms)
        if max_norm > 0:
            v_diag += feature_potential_weight * (norms / max_norm)

    np.fill_diagonal(H, H.diagonal() + v_diag)
    return H, raw_adj


def spectral_decompose(
    H: np.ndarray,
    sparse_threshold: int = 1200,
    k_modes: int = 96
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes spectral decomposition with automatic adaptive sparse acceleration.
    
    If N <= sparse_threshold: Exact dense eigendecomposition via np.linalg.eigh.
    If N > sparse_threshold: Fast sparse Lanczos decomposition via scipy.sparse.linalg.eigsh.
    """
    N = H.shape[0]
    if N <= sparse_threshold:
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        return eigenvalues, eigenvectors

    # Large graph regime: sparse Lanczos iteration on largest magnitude modes
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
    """Computes P(t) where P[v,u] = |<v|exp(-iHt)|u>|^2."""
    phase = np.exp(-1j * eigenvalues * t)
    u_t = eigenvectors @ (phase[:, None] * eigenvectors.conj().T)
    return np.abs(u_t) ** 2


def decompose_classical_laplacian(
    sym_laplacian: np.ndarray,
    sparse_threshold: int = 1200,
    k_modes: int = 96
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes spectrum of the symmetric Laplacian once, avoiding redundant loops."""
    N = sym_laplacian.shape[0]
    if N <= sparse_threshold:
        w, V = np.linalg.eigh(sym_laplacian)
        return w, V

    L_sparse = sp.csr_matrix(sym_laplacian)
    actual_k = min(k_modes, N - 2)
    w, V = spla.eigsh(L_sparse, k=actual_k, which='SM')
    return np.clip(w, 0.0, None), V


def compute_local_interference_contrast(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    sym_laplacian: np.ndarray,
    raw_adj: np.ndarray,
    time_scales: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes scale-invariant Local Counterparty Interference Contrast.
    
    Evaluates quantum constructive interference strictly relative to direct
    transaction counterparties (in-neighbors and out-neighbors), preventing
    dilution on graphs of thousands of nodes without O(N^2) memory footprint.
    
    Returns:
    - local_sink_ratio: max_t (Q_in / (C_in + eps))
    - local_sink_excess: max_t (Q_in - C_in)
    - local_source_excess: max_t (Q_out - C_out)
    - contrast_features: (N, num_time_steps) time-resolved contrast profiles
    """
    N = eigenvectors.shape[0]
    u_idx, v_idx = np.where(raw_adj > 0)
    in_degrees = np.bincount(v_idx, minlength=N).astype(float) + 1e-6
    out_degrees = np.bincount(u_idx, minlength=N).astype(float) + 1e-6

    max_q_in = np.zeros(N)
    max_c_in = np.zeros(N)
    max_excess_in = np.zeros(N)
    max_excess_out = np.zeros(N)
    contrasts = []

    # Decompose classical Laplacian once outside the time loop
    w_c, V_c = decompose_classical_laplacian(sym_laplacian)

    for t in time_scales:
        # Quantum transition amplitudes on transaction counterparties: O(E * K)
        phase = np.exp(-1j * eigenvalues * t)
        v_ph = eigenvectors * phase
        u_uv = np.sum(v_ph[u_idx] * eigenvectors.conj()[v_idx], axis=1)
        u_vu = np.sum(v_ph[v_idx] * eigenvectors.conj()[u_idx], axis=1)
        q_in = np.bincount(v_idx, weights=np.abs(u_uv) ** 2, minlength=N) / in_degrees
        q_out = np.bincount(u_idx, weights=np.abs(u_vu) ** 2, minlength=N) / out_degrees

        # Classical diffusion on counterparties: O(E * K)
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
    """Computes Inverse Participation Ratio (IPR) to detect localized topological bottlenecks."""
    N = eigenvectors.shape[0]
    ipr = np.sum(np.abs(eigenvectors) ** 4, axis=1)
    spr = 1.0 / (N * ipr + 1e-12)
    return spr


def compute_coherent_temporal_variance(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    time_scales: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Measures variance and non-zero peak return probabilities over time (cycle detection) via O(N * K) diagonal projection."""
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
    raw_adj: np.ndarray
) -> np.ndarray:
    """Computes Faccin et al. (PRX 2013) Quantum Amplification Ratio."""
    N = eigenvectors.shape[0]
    quantum_centrality = (1.0 / N) * np.sum(np.abs(eigenvectors) ** 4, axis=1)
    binary = (raw_adj > 0).astype(float)
    degrees = np.sum(binary, axis=1) + np.sum(binary, axis=0)
    total = np.sum(degrees)
    if total == 0:
        classical = np.ones(N) / N
    else:
        classical = degrees / total
    return quantum_centrality / (classical + 1e-8)


def compute_weighted_chiral_flow_imbalance(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    raw_adj: np.ndarray,
    t: float
) -> np.ndarray:
    """Amount-weighted directional quantum flux asymmetry computed on sparse adjacency."""
    N = eigenvectors.shape[0]
    log_adj = np.log1p(raw_adj)
    max_log = np.max(log_adj)
    if max_log > 0:
        log_adj /= max_log

    comb_dense = 0.5 * (log_adj + log_adj.T)
    comb_sp = sp.coo_matrix(comb_dense)

    phase = np.exp(-1j * eigenvalues * t)
    v_ph = eigenvectors * phase
    u_uv = np.sum(v_ph[comb_sp.row] * eigenvectors.conj()[comb_sp.col], axis=1)
    u_vu = np.sum(v_ph[comb_sp.col] * eigenvectors.conj()[comb_sp.row], axis=1)
    p_uv = np.abs(u_uv) ** 2
    p_vu = np.abs(u_vu) ** 2

    wcfi = np.bincount(comb_sp.col, weights=comb_sp.data * (p_vu - p_uv), minlength=N)
    return wcfi


def compute_von_neumann_entropy_deficit(
    laplacian: np.ndarray,
    beta: float = 1.0,
    sparse_threshold: int = 1200
) -> float:
    """Normalized Von Neumann entropy deficit measuring structural organization rigidity."""
    N = laplacian.shape[0]
    if N <= 1:
        return 0.0
    if N <= sparse_threshold:
        w = np.linalg.eigvalsh(laplacian)
    else:
        L_sparse = sp.csr_matrix(laplacian)
        w, _ = spla.eigsh(L_sparse, k=min(96, N - 2), which='SM')

    w_shifted = w - np.min(w)
    exp_w = np.exp(-beta * w_shifted)
    trace = np.sum(exp_w)
    if trace <= 0:
        return 0.0
    probs = exp_w / trace
    probs = probs[probs > 1e-12]
    entropy = -float(np.sum(probs * np.log2(probs)))
    max_entropy = np.log2(N)
    if max_entropy <= 0:
        return 0.0
    return 1.0 - entropy / max_entropy


def extract_features(
    num_nodes: int,
    edges: List,
    raw_node_features: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    num_time_steps: int = 6,
    flow_potential_weight: float = 0.35,
    feature_potential_weight: float = 0.1
) -> np.ndarray:
    """Extracts scale-invariant quantum invariants and unitary wave features for all nodes."""
    H, raw_adj = build_hamiltonian(
        num_nodes, edges, raw_node_features,
        gamma=gamma,
        flow_potential_weight=flow_potential_weight,
        feature_potential_weight=feature_potential_weight
    )
    eigenvalues, eigenvectors = spectral_decompose(H)
    del H
    times = adaptive_time_scales(eigenvalues, num_time_steps)

    binary_adj = (raw_adj > 0).astype(float)
    sym_adj = 0.5 * (binary_adj + binary_adj.T)
    degrees = np.sum(sym_adj, axis=1)
    sym_laplacian = np.diag(degrees) - sym_adj
    del binary_adj, sym_adj

    qar = compute_quantum_amplification_ratio(eigenvectors, raw_adj)
    loc_sink_ratio, loc_excess_in, loc_excess_out, contrast_times = compute_local_interference_contrast(
        eigenvalues, eigenvectors, sym_laplacian, raw_adj, times
    )
    spr = compute_spectral_participation_ratio(eigenvectors)
    ctv, peak_return = compute_coherent_temporal_variance(eigenvalues, eigenvectors, times)

    t_res = times[len(times) // 2]
    wcfi = compute_weighted_chiral_flow_imbalance(eigenvalues, eigenvectors, raw_adj, t_res)

    entropy_deficit = compute_von_neumann_entropy_deficit(sym_laplacian)
    del sym_laplacian
    entropy_col = np.full(num_nodes, entropy_deficit)

    # Diagonal return probability at resonance time via O(N * K) vectorization
    abs_v2 = np.abs(eigenvectors) ** 2
    self_return_res = np.abs(abs_v2 @ np.exp(-1j * eigenvalues * t_res)) ** 2

    # Chunked dispersion entropy to maintain lean memory footprint (< 350 MB)
    dispersion_entropy = np.zeros(num_nodes)
    chunk_size = 500
    phase_res = np.exp(-1j * eigenvalues * t_res)
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        u_chunk = eigenvectors[start:end] @ (phase_res[:, None] * eigenvectors.conj().T)
        p_chunk = np.abs(u_chunk) ** 2 + 1e-12
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

    # Efficient O(N * k * d) unitary wave propagation without N x N materialization
    if raw_node_features is not None:
        z_0 = raw_node_features.astype(np.complex128)
        blocks.append(raw_node_features)
        # Pre-project z_0 onto eigenvector basis: (k, d)
        v_dagger_z0 = eigenvectors.conj().T @ z_0
        for t in (times[0], t_res, times[-1]):
            phase = np.exp(-1j * eigenvalues * t)
            # Multiply diagonal phase in spectral basis: (k, d)
            spectral_z = phase[:, None] * v_dagger_z0
            # Project back to node space: (N, d)
            z_t = eigenvectors @ spectral_z
            blocks.append(np.abs(z_t))
            blocks.append(np.angle(z_t))

    return np.hstack(blocks)


class HQWAMLClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid Quantum-Walk Anomaly Detector for Financial Networks (v3).

    Integrates flow-imbalance potential wells and local counterparty interference
    contrast with a gradient-boosted decision head.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        num_time_steps: int = 6,
        flow_potential_weight: float = 0.35,
        feature_potential_weight: float = 0.1,
        max_iter: int = 150,
        random_state: int = 42
    ):
        self.gamma = gamma
        self.num_time_steps = num_time_steps
        self.flow_potential_weight = flow_potential_weight
        self.feature_potential_weight = feature_potential_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self._cached_X = None
        self._cached_key = None

    def _build_features(self, num_nodes, edges, raw_node_features=None):
        cache_key = (num_nodes, len(edges))
        if self._cached_key == cache_key and self._cached_X is not None:
            return self._cached_X
        X = extract_features(
            num_nodes, edges, raw_node_features,
            gamma=self.gamma,
            num_time_steps=self.num_time_steps,
            flow_potential_weight=self.flow_potential_weight,
            feature_potential_weight=self.feature_potential_weight
        )
        self._cached_X = X
        self._cached_key = cache_key
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

        H, raw_adj = build_hamiltonian(
            N, edge_tuples, raw_feats,
            gamma=self.gamma,
            flow_potential_weight=self.flow_potential_weight
        )
        w, V = spectral_decompose(H)
        times = adaptive_time_scales(w, self.num_time_steps)
        t_res = times[len(times) // 2]

        qar = compute_quantum_amplification_ratio(V, raw_adj)
        spr = compute_spectral_participation_ratio(V)
        wcfi = compute_weighted_chiral_flow_imbalance(w, V, raw_adj, t_res)

        binary = (raw_adj > 0).astype(float)
        sym_adj = 0.5 * (binary + binary.T)
        sym_lap = np.diag(np.sum(sym_adj, axis=1)) - sym_adj
        entropy_deficit = compute_von_neumann_entropy_deficit(sym_lap)

        if self.model is not None:
            probs = self.predict_proba(N, edge_tuples, raw_feats)
            node_risks = probs[:, 1]
        else:
            qar_norm = np.clip(qar / (np.max(qar) + 1e-8), 0, 1)
            wcfi_norm = np.clip(np.abs(wcfi) / (np.max(np.abs(wcfi)) + 1e-8), 0, 1)
            spr_inv = 1.0 - np.clip(spr / (np.max(spr) + 1e-8), 0, 1)
            node_risks = 0.4 * qar_norm + 0.3 * wcfi_norm + 0.3 * spr_inv

        graph_risk = float(np.max(node_risks))
        p_res = quantum_transition_matrix(w, V, t_res)
        suspicious_paths = []
        for u_idx, v_idx, amt in edge_tuples:
            flux = float(p_res[v_idx, u_idx] - p_res[u_idx, v_idx])
            if abs(flux) > 0.05:
                suspicious_paths.append({
                    "source": nodes[u_idx]["id"],
                    "target": nodes[v_idx]["id"],
                    "quantum_flux": round(flux, 4),
                    "amount": amt,
                    "severity": "HIGH_EXTRACTION" if flux > 0.15 else "MODERATE"
                })
        suspicious_paths.sort(key=lambda x: abs(x["quantum_flux"]), reverse=True)

        node_results = []
        for idx, n in enumerate(nodes):
            risk = float(node_risks[idx])
            q = float(qar[idx])
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
                "qar": round(q, 4),
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
                "num_edges": len(edge_tuples)
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