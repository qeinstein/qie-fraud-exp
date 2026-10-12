"""
HQW-AML v2: Hermitian Quantum Walk Anomaly Engine.

A quantum-inspired graph anomaly detection engine for directed financial
transaction networks. Detects coordinated multi-hop money-mule rings by
exploiting three quantum phenomena with no classical analog:

1. Constructive phase interference at topological reconvergence sinks
2. Ballistic norm-preserving propagation immune to over-smoothing
3. Chiral asymmetry encoding causal direction of money flow

The core innovation is the Quantum-Classical Interference Contrast:
for every node we compute BOTH quantum walk and classical diffusion
probabilities, then measure the DIFFERENCE. This difference is the
pure interference signal that only exists in mule ring topologies.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


def build_hamiltonian(
    num_nodes: int,
    edges: List,
    raw_node_features: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    feature_potential_weight: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs an amount-normalized Hermitian Hamiltonian with on-site potentials.

    Coupling terms use log-normalized transaction amounts to prevent spectral
    blow-up while preserving relative magnitude ordering. Directed edges are
    encoded as imaginary antisymmetric phases (Zimborás et al. 2013).

    On-site potentials from raw node features shape the quantum energy landscape
    so the walker's dynamics are influenced by account-level risk priors.

    Returns (H, raw_adjacency) where H is complex Hermitian and raw_adjacency
    is the unnormalized directed weight matrix.
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

    # log-normalize amounts so eigenvalues are bounded
    log_adj = np.log1p(raw_adj)
    max_log = np.max(log_adj)
    if max_log > 0:
        log_adj /= max_log

    # symmetric coupling + antisymmetric chiral flow
    a_sym = 0.5 * (log_adj + log_adj.T)
    a_skew = log_adj - log_adj.T
    H = a_sym.astype(np.complex128) + 1j * gamma * a_skew

    # on-site potentials from node features
    if raw_node_features is not None and raw_node_features.shape[0] == num_nodes:
        # project features to a scalar potential per node via L2 norm
        # this creates energy wells/barriers that redirect quantum flow
        norms = np.linalg.norm(raw_node_features, axis=1)
        if np.max(norms) > 0:
            norms = norms / np.max(norms)
        np.fill_diagonal(H, feature_potential_weight * norms)

    return H, raw_adj


def spectral_decompose(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns eigenvalues (real) and eigenvectors (complex unitary) of H."""
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors


def adaptive_time_scales(eigenvalues: np.ndarray, num_steps: int = 6) -> np.ndarray:
    """
    Computes spectrally-adapted evolution times from the Hamiltonian spectrum.

    The fundamental resonance period is T_0 = 2π / spectral_bandwidth.
    We sample num_steps evenly spaced fractions of T_0 so that features
    capture at least one full interference cycle regardless of graph structure.
    """
    bandwidth = float(eigenvalues[-1] - eigenvalues[0])
    if bandwidth < 1e-8:
        return np.array([0.5, 1.0, 2.0])
    t_fundamental = 2.0 * np.pi / bandwidth
    return np.linspace(t_fundamental / num_steps, t_fundamental, num_steps)


def quantum_transition_matrix(eigenvalues, eigenvectors, t):
    """P(t) where P[v,u] = |<v|exp(-iHt)|u>|^2."""
    phase = np.exp(-1j * eigenvalues * t)
    U_t = eigenvectors @ (phase[:, None] * eigenvectors.conj().T)
    return np.abs(U_t) ** 2


def classical_diffusion_matrix(laplacian, t):
    """Heat kernel exp(-L*t) for classical random walk comparison."""
    w, V = np.linalg.eigh(laplacian)
    return V @ (np.exp(-w * t)[:, None] * V.T)


def compute_interference_contrast(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    sym_laplacian: np.ndarray,
    time_scales: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes quantum-classical interference contrast at each node.

    For every time scale, we compute:
    - Quantum incoming probability (column sum of P_quantum)
    - Classical incoming probability (column sum of P_classical)
    - Their difference isolates the pure interference signal

    Returns:
        icr: Interference Contrast Ratio per node (max_t quantum / classical)
        pqe: Peak Quantum Excess per node (max_t quantum - classical)
        contrast_features: (N, len(time_scales)) raw contrast values
    """
    N = len(eigenvalues)
    max_q = np.zeros(N)
    max_c = np.zeros(N)
    contrasts = []

    for t in time_scales:
        p_q = quantum_transition_matrix(eigenvalues, eigenvectors, t)
        p_c = classical_diffusion_matrix(sym_laplacian, t)
        p_c = np.clip(p_c, 0, None)

        # incoming probability at each node (sum over source column)
        q_incoming = np.sum(p_q, axis=0) / N
        c_incoming = np.sum(p_c, axis=0) / max(np.sum(p_c), 1e-12) * N
        c_incoming = np.clip(c_incoming / N, 1e-12, None)

        max_q = np.maximum(max_q, q_incoming)
        max_c = np.maximum(max_c, c_incoming)
        contrasts.append(q_incoming - c_incoming)

    icr = max_q / (max_c + 1e-8)
    pqe = np.max(np.array(contrasts), axis=0)
    contrast_features = np.array(contrasts).T

    return icr, pqe, contrast_features


def compute_spectral_participation_ratio(eigenvectors: np.ndarray) -> np.ndarray:
    """
    Inverse Participation Ratio from condensed matter physics.

    SPR(v) = 1 / (N * Σ_n |V_{v,n}|^4)

    Low SPR → node is spectrally localized (topological endpoint/bottleneck).
    High SPR → node is delocalized across many eigenmodes (normal account).
    """
    N = eigenvectors.shape[0]
    ipr = np.sum(np.abs(eigenvectors) ** 4, axis=1)
    spr = 1.0 / (N * ipr + 1e-12)
    return spr


def compute_coherent_temporal_variance(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    time_scales: np.ndarray
) -> np.ndarray:
    """
    Measures how strongly the return probability oscillates over time.

    CTV(v) = Var_t[ P_{vv}(t) ]

    High CTV → few eigenmodes dominate → rigid mule structure with coherent beating.
    Low CTV → many eigenmodes → rapid dephasing → normal random commerce.
    """
    N = len(eigenvalues)
    returns = []
    for t in time_scales:
        p_t = quantum_transition_matrix(eigenvalues, eigenvectors, t)
        returns.append(np.diag(p_t))
    returns = np.array(returns)  # (num_times, N)
    ctv = np.var(returns, axis=0)
    return ctv


def compute_quantum_amplification_ratio(
    eigenvectors: np.ndarray,
    raw_adj: np.ndarray
) -> np.ndarray:
    """
    Faccin et al. (PRX 2013) Quantum Amplification Ratio.

    Ratio of time-averaged quantum occupation to classical degree centrality.
    Identifies low-degree accounts acting as topological resonant bridges.
    """
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
    """
    Amount-weighted directional flow asymmetry.

    WCFI(v) = Σ_u ŵ_{uv} * [P_q(u→v, t) - P_q(v→u, t)]

    Combines quantum chiral asymmetry with financial amount information.
    Positive WCFI → net quantum inflow (mule sink collecting funds).
    Negative WCFI → net quantum outflow (mule source dispersing funds).
    """
    N = len(eigenvalues)
    p_t = quantum_transition_matrix(eigenvalues, eigenvectors, t)
    log_adj = np.log1p(raw_adj)
    max_log = np.max(log_adj)
    if max_log > 0:
        log_adj /= max_log

    # p_t[v,u] = probability of reaching v starting from u
    # directional flux from u to v: p_t[v,u] - p_t[u,v]
    flux = p_t.T - p_t  # flux[u,v] = P(u→v) - P(v→u)
    combined = (log_adj + log_adj.T) / 2.0
    wcfi = np.sum(combined * flux, axis=0)
    return wcfi


def compute_von_neumann_entropy_deficit(laplacian: np.ndarray, beta: float = 1.0) -> float:
    """
    Normalized entropy deficit measuring structural rigidity.

    deficit = 1 - S(ρ) / log₂(N)

    High deficit → rigid low-entropy topology (mule ring).
    Low deficit → diverse high-entropy topology (legitimate commerce).
    """
    N = laplacian.shape[0]
    if N <= 1:
        return 0.0
    w = np.linalg.eigvalsh(laplacian)
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
    feature_potential_weight: float = 0.1
) -> np.ndarray:
    """
    Extracts the full HQW-AML v2 feature vector for all nodes.

    Each feature measures a distinct physical phenomenon:
    - QAR: quantum vs classical centrality divergence
    - ICR: interference contrast ratio (quantum / classical probability)
    - PQE: peak constructive interference excess
    - SPR: spectral localization (eigenvector participation)
    - CTV: coherent temporal variance (structural rigidity)
    - WCFI: amount-weighted chiral flow imbalance
    - Entropy Deficit: graph-level structural entropy loss
    - Wave features: unitary-propagated raw features (Dirichlet-preserving)
    """
    H, raw_adj = build_hamiltonian(
        num_nodes, edges, raw_node_features,
        gamma=gamma, feature_potential_weight=feature_potential_weight
    )
    eigenvalues, eigenvectors = spectral_decompose(H)
    times = adaptive_time_scales(eigenvalues, num_time_steps)

    # symmetrized Laplacian for classical diffusion baseline
    binary_adj = (raw_adj > 0).astype(float)
    sym_adj = 0.5 * (binary_adj + binary_adj.T)
    degrees = np.sum(sym_adj, axis=1)
    sym_laplacian = np.diag(degrees) - sym_adj

    # 1. Quantum Amplification Ratio (Faccin 2013)
    qar = compute_quantum_amplification_ratio(eigenvectors, raw_adj)

    # 2-3. Interference Contrast (the core innovation)
    icr, pqe, contrast_per_time = compute_interference_contrast(
        eigenvalues, eigenvectors, sym_laplacian, times
    )

    # 4. Spectral Participation Ratio (localization detector)
    spr = compute_spectral_participation_ratio(eigenvectors)

    # 5. Coherent Temporal Variance (structural rigidity)
    ctv = compute_coherent_temporal_variance(eigenvalues, eigenvectors, times)

    # 6. Weighted Chiral Flow Imbalance (amount-weighted direction)
    t_res = times[len(times) // 2]
    wcfi = compute_weighted_chiral_flow_imbalance(eigenvalues, eigenvectors, raw_adj, t_res)

    # 7. Von Neumann Entropy Deficit (graph-level, broadcast to all nodes)
    entropy_deficit = compute_von_neumann_entropy_deficit(sym_laplacian)
    entropy_col = np.full(num_nodes, entropy_deficit)

    # 8. Self-return and dispersion at resonance time
    p_res = quantum_transition_matrix(eigenvalues, eigenvectors, t_res)
    self_return = np.diag(p_res)
    p_safe = p_res + 1e-12
    dispersion_entropy = -np.sum(p_safe * np.log2(p_safe), axis=1)

    blocks = [
        qar[:, None],           # (N, 1)
        icr[:, None],           # (N, 1)
        pqe[:, None],           # (N, 1)
        spr[:, None],           # (N, 1)
        ctv[:, None],           # (N, 1)
        wcfi[:, None],          # (N, 1)
        entropy_col[:, None],   # (N, 1)
        self_return[:, None],   # (N, 1)
        dispersion_entropy[:, None],  # (N, 1)
        contrast_per_time,      # (N, num_time_steps)
    ]

    # 9. Unitary wave-propagated raw features (norm-preserving)
    if raw_node_features is not None:
        z_0 = raw_node_features.astype(np.complex128)
        blocks.append(raw_node_features)
        for t in (times[0], t_res, times[-1]):
            phase = np.exp(-1j * eigenvalues * t)
            U_t = eigenvectors @ (phase[:, None] * eigenvectors.conj().T)
            z_t = U_t @ z_0
            blocks.append(np.abs(z_t))        # magnitude (preserved norm)
            blocks.append(np.angle(z_t))       # interference phase

    return np.hstack(blocks)


class HQWAMLClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid Quantum-Walk Anomaly Detector for Financial Networks (v2).

    Extracts quantum interference invariants that have no classical analog,
    then trains a thin gradient-boosted calibration head.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        num_time_steps: int = 6,
        feature_potential_weight: float = 0.1,
        max_iter: int = 150,
        random_state: int = 42
    ):
        self.gamma = gamma
        self.num_time_steps = num_time_steps
        self.feature_potential_weight = feature_potential_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None

    def _build_features(self, num_nodes, edges, raw_node_features=None):
        return extract_features(
            num_nodes, edges, raw_node_features,
            gamma=self.gamma,
            num_time_steps=self.num_time_steps,
            feature_potential_weight=self.feature_potential_weight
        )

    def fit(self, num_nodes, edges, raw_node_features, labels, train_mask=None):
        """Extracts quantum invariants and trains the calibration head."""
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
        """Returns [P(licit), P(illicit)] for each node."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        X = self._build_features(num_nodes, edges, raw_node_features)
        if eval_mask is not None:
            X = X[eval_mask]
        return self.model.predict_proba(X)

    def predict(self, num_nodes, edges, raw_node_features=None, eval_mask=None):
        """Returns binary labels (0: licit, 1: illicit)."""
        return np.argmax(
            self.predict_proba(num_nodes, edges, raw_node_features, eval_mask),
            axis=1
        )

    def score_subgraph(self, subgraph_data: Dict) -> Dict:
        """
        High-level inference endpoint for backend services and SOC console.

        Input:
            {"nodes": [{"id": ..., "features": [...]}, ...],
             "edges": [{"source": ..., "target": ..., "amount": ...}, ...]}

        Output:
            {"graph_risk_score": float, "node_scores": [...],
             "quantum_diagnostics": {...}, "suspicious_paths": [...]}
        """
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

        # build Hamiltonian and compute core quantities
        H, raw_adj = build_hamiltonian(N, edge_tuples, raw_feats, gamma=self.gamma)
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

        # node risk scores
        if self.model is not None:
            probs = self.predict_proba(N, edge_tuples, raw_feats)
            node_risks = probs[:, 1]
        else:
            # unsupervised fallback using quantum invariants directly
            qar_norm = np.clip(qar / np.max(qar + 1e-8), 0, 1)
            wcfi_norm = np.clip(np.abs(wcfi) / (np.max(np.abs(wcfi)) + 1e-8), 0, 1)
            spr_inv = 1.0 - np.clip(spr / (np.max(spr) + 1e-8), 0, 1)
            node_risks = 0.4 * qar_norm + 0.3 * wcfi_norm + 0.3 * spr_inv

        graph_risk = float(np.max(node_risks))

        # edge-level quantum coherence flux for explainability
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

        # classify node roles
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
    print("HQW-AML v2 Self-Test\n")

    # mule ring: 0 → {1,2,3} → 4 (fan-out / fan-in)
    # legitimate: 5 ↔ 6 ↔ 7 (bilateral commerce)
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

    print("Node-level predictions:")
    for i, p in enumerate(probs):
        tag = "MULE" if labels[i] == 1 else "LEGIT"
        print(f"  Node {i} [{tag}]: P(Illicit)={p[1]:.3f}")

    payload = {
        "nodes": [{"id": f"ACC_{i}", "features": list(feats[i])} for i in range(N)],
        "edges": [{"source": f"ACC_{e[0]}", "target": f"ACC_{e[1]}", "amount": e[2]} for e in demo_edges]
    }
    result = clf.score_subgraph(payload)
    print(f"\nGraph Risk Score: {result['graph_risk_score']}")
    print(f"Entropy Deficit:  {result['quantum_diagnostics']['von_neumann_entropy_deficit']}")
    print(f"Spectral BW:      {result['quantum_diagnostics']['spectral_bandwidth']}")
    if result["suspicious_paths"]:
        print("Suspicious Paths:")
        for p in result["suspicious_paths"]:
            print(f"  {p['source']} → {p['target']}: flux={p['quantum_flux']}, {p['severity']}")