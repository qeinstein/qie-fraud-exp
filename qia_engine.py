"""
Hermitian Continuous-Time Quantum Walk Anomaly Engine (HQW-AML).

This engine implements a quantum-inspired graph anomaly detection pipeline
for directed transaction networks, specifically combating multi-hop money-mule
rings and transaction layering.

Key Theoretical Foundations:
- Farhi & Gutmann (1998): Unitary evolution U(t) = exp(-i*H*t)
- Childs et al. (2003): Constructive phase interference at reconvergence sinks
- Mülken & Blumen (2011): Continuous-time quantum walks on complex networks
- Zimborás et al. (2013) & Coutinho et al. (2022): Hermitian digraph Hamiltonian
- Faccin et al. (PRX 2013): Quantum Amplification Ratio (QAR) for low-degree mules
- Rossi et al. (2015): Von Neumann entropy of graph density matrices
- Kiani et al. (NeurIPS 2024): Unitary norm-preserving feature maps
"""

import numpy as np
import scipy.linalg
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


def build_hermitian_hamiltonian(
    num_nodes: int,
    edges: Union[List[Tuple[int, int]], List[Tuple[int, int, float]], np.ndarray],
    gamma: float = 1.0,
    directed: bool = True
) -> np.ndarray:
    """
    Constructs a complex Hermitian Hamiltonian matrix from directed transaction edges.
    
    Formula:
        H = A_s + i * gamma * (A - A^T)
    where A_s = 0.5 * (A + A^T) is the symmetric skeleton and (A - A^T) is the
    skew-symmetric directed flow operator.
    
    Args:
        num_nodes: Total number of vertices in the subgraph.
        edges: List of (u, v) or (u, v, weight) directed transaction edges.
        gamma: Directional chiral coupling parameter (default 1.0).
        directed: Whether to preserve edge direction via complex phase.
        
    Returns:
        H: (num_nodes, num_nodes) complex Hermitian matrix (H == H^dagger).
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    
    for edge in edges:
        if len(edge) == 2:
            u, v = int(edge[0]), int(edge[1])
            w = 1.0
        else:
            u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
            
        if 0 <= u < num_nodes and 0 <= v < num_nodes and u != v:
            A[u, v] += w

    if not directed:
        return 0.5 * (A + A.T).astype(np.complex128)

    a_sym = 0.5 * (A + A.T)
    a_skew = A - A.T
    h = a_sym + 1j * gamma * a_skew
    return h


def compute_unitary_spectrum(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes exact eigenvalues and eigenvectors of a complex Hermitian Hamiltonian.
    
    Returns:
        eigenvalues: (N,) real array
        eigenvectors: (N, N) complex unitary matrix V where H = V @ diag(w) @ V^dagger
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors


def compute_quantum_transition_probabilities(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    t: float
) -> np.ndarray:
    """
    Computes the transition probability matrix P(t) at evolution time t:
        P[v, u] = |<v| exp(-i * H * t) |u>|^2
    """
    phase_factors = np.exp(-1j * eigenvalues * t)
    u_t = eigenvectors @ (phase_factors[:, None] * eigenvectors.conj().T)
    p_t = np.abs(u_t) ** 2
    return p_t


def compute_quantum_amplification_ratio(
    eigenvectors: np.ndarray,
    adj_matrix: np.ndarray
) -> np.ndarray:
    """
    Computes Faccin et al. (PRX 2013) Quantum Amplification Ratio (QAR).
    
    Measures the ratio of time-averaged quantum occupation probability to
    classical degree centrality. Identifies low-degree accounts that act as
    topological resonant bridges (mules).
    """
    n = eigenvectors.shape[0]
    quantum_centrality = (1.0 / n) * np.sum(np.abs(eigenvectors) ** 4, axis=1)

    degrees = np.sum(adj_matrix > 0, axis=1) + np.sum(adj_matrix > 0, axis=0)
    total_degree = np.sum(degrees)
    if total_degree == 0:
        classical_centrality = np.ones(n) / n
    else:
        classical_centrality = degrees / total_degree

    qar = quantum_centrality / (classical_centrality + 1e-6)
    return qar


def compute_von_neumann_entropy(L: np.ndarray, beta: float = 1.0) -> float:
    """
    Computes the Von Neumann entropy of the graph thermal density matrix:
        rho = exp(-beta * L) / Tr(exp(-beta * L))
        S(rho) = -sum lambda_i * log2(lambda_i)
    """
    w = np.linalg.eigvalsh(L)
    w_shifted = w - np.min(w)
    exp_w = np.exp(-beta * w_shifted)
    trace = np.sum(exp_w)
    if trace <= 0:
        return 0.0
    probabilities = exp_w / trace
    probabilities = probabilities[probabilities > 1e-12]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


def extract_hqw_node_features(
    num_nodes: int,
    edges: Union[List[Tuple[int, int]], List[Tuple[int, int, float]], np.ndarray],
    raw_node_features: Optional[np.ndarray] = None,
    time_scales: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    gamma: float = 1.0
) -> np.ndarray:
    """
    Extracts multi-scale quantum topological invariants for all nodes in a subgraph.
    
    Extracted Features per node:
    - Self-return probability P_{uu}(t) for each time scale (coherent trapping)
    - Dispersion row entropy H_u(t) for each time scale (fan-out rate)
    - Peak exit transition coherence max_{v != u} P_{uv}(t)
    - Quantum Amplification Ratio (QAR)
    - Directional flow asymmetry (chiral sink bias)
    - Unitary wave-propagated features |U(t) X| preserving Dirichlet energy
    
    Returns:
        feature_matrix: (num_nodes, total_feature_dim) array of real float features.
    """
    h = build_hermitian_hamiltonian(num_nodes, edges, gamma=gamma, directed=True)
    eigenvalues, eigenvectors = compute_unitary_spectrum(h)
    
    raw_adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for edge in edges:
        u, v = int(edge[0]), int(edge[1])
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            raw_adj[u, v] = 1.0

    qar = compute_quantum_amplification_ratio(eigenvectors, raw_adj)
    
    feature_columns = [qar]
    
    for t in time_scales:
        p_t = compute_quantum_transition_probabilities(eigenvalues, eigenvectors, t)
        
        self_return = np.diag(p_t)
        feature_columns.append(self_return)
        
        p_safe = p_t + 1e-12
        row_entropy = -np.sum(p_safe * np.log2(p_safe), axis=1)
        feature_columns.append(row_entropy)
        
        p_offdiag = p_t.copy()
        np.fill_diagonal(p_offdiag, 0.0)
        peak_exit = np.max(p_offdiag, axis=1)
        feature_columns.append(peak_exit)
        
        incoming = np.sum(p_t, axis=1) - self_return
        outgoing = np.sum(p_t, axis=0) - self_return
        chiral_bias = incoming - outgoing
        feature_columns.append(chiral_bias)

    quantum_invariants = np.column_stack(feature_columns)

    if raw_node_features is not None:
        wave_features = [raw_node_features]
        z_0 = raw_node_features.astype(np.complex128)
        for t in (1.0, 2.0):
            phase_factors = np.exp(-1j * eigenvalues * t)
            u_t = eigenvectors @ (phase_factors[:, None] * eigenvectors.conj().T)
            z_t = u_t @ z_0
            wave_features.append(np.abs(z_t))
            wave_features.append(np.angle(z_t))
        
        all_features = np.hstack([quantum_invariants] + wave_features)
        return all_features

    return quantum_invariants


class HQWAMLClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid Quantum-Walk Anomaly Detector for Financial Networks.
    
    Combines Hermitian CTQW multi-scale spectral features with a gradient-boosted
    decision head. Immune to over-smoothing across multi-hop subgraphs.
    """
    
    def __init__(
        self,
        time_scales: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        gamma: float = 1.0,
        classifier_type: str = "hist_gbm",
        max_iter: int = 150,
        random_state: int = 42
    ):
        self.time_scales = time_scales
        self.gamma = gamma
        self.classifier_type = classifier_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.model: Optional[Union[HistGradientBoostingClassifier, RandomForestClassifier]] = None
        
    def _create_base_classifier(self):
        if self.classifier_type == "hist_gbm":
            return HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight="balanced"
            )
        elif self.classifier_type == "rf":
            return RandomForestClassifier(
                n_estimators=self.max_iter,
                random_state=self.random_state,
                class_weight="balanced_subsample"
            )
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")

    def fit(
        self,
        num_nodes: int,
        edges: Union[List[Tuple[int, int]], List[Tuple[int, int, float]], np.ndarray],
        raw_node_features: Optional[np.ndarray],
        labels: np.ndarray,
        train_mask: Optional[np.ndarray] = None
    ) -> "HQWAMLClassifier":
        """
        Extracts quantum invariants and trains the decision head.
        """
        x_quantum = extract_hqw_node_features(
            num_nodes=num_nodes,
            edges=edges,
            raw_node_features=raw_node_features,
            time_scales=self.time_scales,
            gamma=self.gamma
        )
        
        if train_mask is not None:
            x_train = x_quantum[train_mask]
            y_train = labels[train_mask]
        else:
            x_train = x_quantum
            y_train = labels

        self.model = self._create_base_classifier()
        self.model.fit(x_train, y_train)
        return self

    def predict_proba(
        self,
        num_nodes: int,
        edges: Union[List[Tuple[int, int]], List[Tuple[int, int, float]], np.ndarray],
        raw_node_features: Optional[np.ndarray] = None,
        eval_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Returns anomaly probabilities [P(licit), P(illicit)] for target nodes.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
            
        x_quantum = extract_hqw_node_features(
            num_nodes=num_nodes,
            edges=edges,
            raw_node_features=raw_node_features,
            time_scales=self.time_scales,
            gamma=self.gamma
        )
        
        if eval_mask is not None:
            x_eval = x_quantum[eval_mask]
        else:
            x_eval = x_quantum
            
        probabilities = self.model.predict_proba(x_eval)
        return probabilities

    def predict(
        self,
        num_nodes: int,
        edges: Union[List[Tuple[int, int]], List[Tuple[int, int, float]], np.ndarray],
        raw_node_features: Optional[np.ndarray] = None,
        eval_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Returns predicted binary labels (0: licit, 1: illicit).
        """
        probs = self.predict_proba(num_nodes, edges, raw_node_features, eval_mask)
        return np.argmax(probs, axis=1)

    def score_subgraph(self, subgraph_data: Dict) -> Dict:
        """
        High-level inference endpoint for backend microservices and SOC frontend.
        
        Accepts:
            subgraph_data = {
                "nodes": [{"id": "ACC_1", "features": [...]}, ...],
                "edges": [{"source": "ACC_1", "target": "ACC_2", "amount": 50000.0}, ...]
            }
            
        Returns:
            result = {
                "risk_score": float,
                "node_scores": [{"id": "ACC_1", "anomaly_score": 0.94, "qar": 3.42, "role": "MULE"}, ...],
                "quantum_entropy": float,
                "constructive_peak_detected": bool
            }
        """
        nodes = subgraph_data["nodes"]
        edges = subgraph_data["edges"]
        
        node_id_to_idx = {n["id"]: idx for idx, n in enumerate(nodes)}
        num_nodes = len(nodes)
        
        edge_tuples = []
        for e in edges:
            u = node_id_to_idx.get(e["source"])
            v = node_id_to_idx.get(e["target"])
            amt = float(e.get("amount", 1.0))
            if u is not None and v is not None:
                edge_tuples.append((u, v, amt))

        raw_feats = None
        if "features" in nodes[0]:
            feat_list = [n["features"] for n in nodes]
            raw_feats = np.array(feat_list, dtype=np.float64)

        h = build_hermitian_hamiltonian(num_nodes, edge_tuples, gamma=self.gamma, directed=True)
        w, v = compute_unitary_spectrum(h)
        
        raw_adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for u, v_idx, _ in edge_tuples:
            raw_adj[u, v_idx] = 1.0
        qar = compute_quantum_amplification_ratio(v, raw_adj)
        
        p_t1 = compute_quantum_transition_probabilities(w, v, t=1.0)
        p_offdiag = p_t1.copy()
        np.fill_diagonal(p_offdiag, 0.0)
        max_coherence = float(np.max(p_offdiag)) if num_nodes > 1 else 0.0
        
        laplacian = np.diag(np.sum(raw_adj, axis=1)) - raw_adj
        sym_lap = 0.5 * (laplacian + laplacian.T)
        entropy = compute_von_neumann_entropy(sym_lap)

        if self.model is not None:
            probs = self.predict_proba(num_nodes, edge_tuples, raw_feats)
            node_risks = probs[:, 1]
            graph_risk = float(np.max(node_risks))
        else:
            normalized_qar = np.clip(qar / 5.0, 0.0, 1.0)
            node_risks = normalized_qar
            graph_risk = float(np.max(node_risks))

        node_results = []
        for idx, n in enumerate(nodes):
            risk = float(node_risks[idx])
            q_val = float(qar[idx])
            if risk > 0.75:
                role = "MULE_CASHOUT_SINK" if q_val > 2.0 else "SUSPICIOUS_CONDUIT"
            elif risk > 0.4:
                role = "INTERMEDIARY"
            else:
                role = "NORMAL_ACCOUNT"
                
            node_results.append({
                "id": n["id"],
                "anomaly_score": round(risk, 4),
                "quantum_amplification_ratio": round(q_val, 4),
                "role": role
            })

        return {
            "graph_risk_score": round(graph_risk, 4),
            "node_scores": node_results,
            "von_neumann_entropy": round(entropy, 4),
            "constructive_peak_detected": bool(max_coherence > 0.4)
        }


if __name__ == "__main__":
    print("Testing HQW-AML standalone engine...")
    demo_edges = [
        (0, 1, 1000.0),
        (0, 2, 1000.0),
        (0, 3, 1000.0),
        (1, 4, 950.0),
        (2, 4, 950.0),
        (3, 4, 950.0),
        (5, 6, 20.0),
        (6, 7, 15.0)
    ]
    demo_num_nodes = 8
    dummy_feats = np.random.randn(demo_num_nodes, 4)
    dummy_labels = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    
    clf = HQWAMLClassifier(max_iter=50)
    clf.fit(demo_num_nodes, demo_edges, dummy_feats, dummy_labels)
    probs = clf.predict_proba(demo_num_nodes, demo_edges, dummy_feats)
    
    print("Inference probabilities (first 5 nodes = mule ring):")
    for i, p in enumerate(probs):
        print(f"Node {i}: P(Licit)={p[0]:.3f}, P(Illicit)={p[1]:.3f}")
        
    subgraph_payload = {
        "nodes": [{"id": f"ACC_{i}", "features": list(dummy_feats[i])} for i in range(demo_num_nodes)],
        "edges": [{"source": f"ACC_{u}", "target": f"ACC_{v}", "amount": amt} for u, v, amt in demo_edges]
    }
    score_res = clf.score_subgraph(subgraph_payload)
    print("\nSubgraph evaluation result:")
    print("Overall Graph Risk:", score_res["graph_risk_score"])
    print("Constructive Peak Detected:", score_res["constructive_peak_detected"])
    print("Von Neumann Entropy:", score_res["von_neumann_entropy"])