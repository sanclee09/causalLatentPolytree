from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import numpy as np
import networkx as nx


@dataclass
class Polytree:
    edges: Dict[Tuple[str, str], float]
    sigmas: Dict[str, float]
    kappas: Dict[str, float]
    graph: nx.DiGraph = field(init=False)

    def __post_init__(self) -> None:
        g = nx.DiGraph()
        g.add_edges_from(self.edges.keys())
        for v in g.nodes:
            if g.in_degree(v) > 1:
                raise ValueError(
                    f"Node {v} has in-degree > 1; the graph is not a polytree."
                )
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError(
                "The graph contains a directed cycle; a polytree must be acyclic."
            )
        und = g.to_undirected()
        if not nx.is_tree(und):
            raise ValueError("The underlying undirected graph must be a tree.")
        self.graph = g

    @property
    def nodes(self) -> List[str]:
        return self._topo_order()

    def _topo_order(self) -> List[str]:
        # Kahn's algorithm to ensure true topological order
        in_deg = {node: self.graph.in_degree(node) for node in self.graph.nodes}
        queue = [node for node, deg in in_deg.items() if deg == 0]
        order = []
        while queue:
            v = queue.pop(0)
            order.append(v)
            for w in self.graph.successors(v):
                in_deg[w] -= 1
                if in_deg[w] == 0:
                    queue.append(w)
        return order

    def alpha_matrix(self) -> Dict[str, Dict[str, float]]:
        alpha: Dict[str, Dict[str, float]] = {
            i: {j: 0.0 for j in self.graph.nodes} for i in self.graph.nodes
        }
        for v in self._topo_order():
            alpha[v][v] = 1.0
            for u in self.graph.predecessors(v):
                lam = self.edges[(u, v)]
                for h in self.graph.nodes:
                    alpha[v][h] += lam * alpha[u][h]
        return alpha

    def alpha_matrix_fast(self) -> np.ndarray:
        """
        Vectorized alpha = (I - B^T)^{-1}.
        We build B[u,v] = lambda_{uv} and then solve for A = (I - B^T)^{-1}.
        """
        nodes = self.nodes
        n = len(nodes)
        idx = {v: k for k, v in enumerate(nodes)}

        B = np.zeros((n, n), dtype=np.float64)
        for (u, v), lam in self.edges.items():
            B[idx[u], idx[v]] = lam

        I = np.eye(n, dtype=np.float64)
        # A_T = (I - B)^{-1}  => A = (I - B^T)^{-1}
        A_T = np.linalg.solve(I - B, I)  # <-- NOTE: not (I - B).T
        alpha = A_T.T
        return alpha

    def covariance(self) -> np.ndarray:
        n = len(self.nodes)
        Sigma = np.zeros((n, n), dtype=float)
        alpha = self.alpha_matrix()
        for i_idx, i in enumerate(self.nodes):
            for j_idx, j in enumerate(self.nodes):
                cov = 0.0
                for h in self.nodes:
                    cov += alpha[i][h] * alpha[j][h] * (self.sigmas[h] ** 2)
                Sigma[i_idx, j_idx] = cov
        return Sigma

    def covariance_fast(self, alpha: np.ndarray) -> np.ndarray:
        """
        Vectorized covariance: Sigma = alpha * diag(sigmas^2) * alpha^T.
        """
        nodes = self.nodes
        sigma2 = np.array(
            [self.sigmas[v] ** 2 for v in nodes], dtype=np.float64
        )  # (n,)
        # Multiply each column h of alpha by sigma2[h]
        Sigma = alpha @ (alpha * sigma2[None, :]).T
        return Sigma

    def third_cumulants(self) -> Dict[Tuple[str, str, str], float]:
        alpha = self.alpha_matrix()
        C3: Dict[Tuple[str, str, str], float] = {}
        for i in self.nodes:
            for j in self.nodes:
                for k in self.nodes:
                    val = 0.0
                    for h in self.nodes:
                        val += alpha[i][h] * alpha[j][h] * alpha[k][h] * self.kappas[h]
                    C3[(i, j, k)] = val
        return C3


def third_cumulants_fast(alpha: np.ndarray, kappa: np.ndarray, mode: str = "slices"):
    """
    mode='slices' -> returns (C_iij, C_ijj, C_iii)
    mode='full'   -> returns C (n,n,n)
    """
    if mode == "full":
        return np.einsum(
            "ih,jh,kh,h->ijk",
            alpha,
            alpha,
            alpha,
            kappa.astype(np.float64),
            optimize=True,
        )

    if mode == "slices":
        alpha = alpha.astype(np.float64, copy=False)
        kappa = kappa.astype(np.float64, copy=False)

        A2 = alpha**2
        A3 = alpha**3
        # multiply columns by kappa[h]
        P = alpha * kappa[None, :]
        N = A2 * kappa[None, :]

        C_iij = N @ alpha.T  # (n,n)
        C_ijj = P @ A2.T  # (n,n)
        C_iii = A3 @ kappa  # (n,)
        return C_iij, C_ijj, C_iii

    raise ValueError("mode must be 'slices' or 'full'")


def compute_discrepancy(poly: Polytree) -> np.ndarray:
    nodes = poly.nodes
    n = len(nodes)
    Sigma = poly.covariance()
    C3 = poly.third_cumulants()
    Gamma = np.zeros((n, n), dtype=float)

    def is_uncorrelated(i_idx: int, j_idx: int) -> bool:
        return np.isclose(Sigma[i_idx, j_idx], 0.0)

    for i_idx, i in enumerate(nodes):
        for j_idx, j in enumerate(nodes):
            if i == j:
                Gamma[i_idx, j_idx] = 0.0
                continue
            if is_uncorrelated(i_idx, j_idx):
                Gamma[i_idx, j_idx] = -1.0
                continue
            num_zero = (
                Sigma[i_idx, i_idx] * C3[(i, i, j)]
                - Sigma[i_idx, j_idx] * C3[(i, i, i)]
            )
            if np.isclose(num_zero, 0.0):
                Gamma[i_idx, j_idx] = 0.0
                continue
            numerator = C3[(i, j, j)] * Sigma[i_idx, i_idx]
            denominator = C3[(i, i, j)] * Sigma[i_idx, j_idx]
            Gamma[i_idx, j_idx] = numerator / denominator
    return Gamma


def compute_discrepancy_fast(
    poly: "Polytree",
    eps_corr: float = 1e-12,  # threshold for "uncorrelated" Sigma_ij ≈ 0
    eps_numzero: float = 1e-12,  # threshold for |Sigma_ii*C_iij - Sigma_ij*C_iii| ≈ 0
    eps_den: float = 1e-14,  # guard for tiny denominators
    cumulant_mode: str = "slices",
) -> np.ndarray:
    """
    Vectorized version of compute_discrepancy().

    Rules (matching your scalar logic):
      - i==j: Gamma[i,j] = 0
      - |Sigma[i,j]| < eps_corr: Gamma[i,j] = -1 (no common ancestor)
      - |Sigma[i,i]*C_iij - Sigma[i,j]*C_iii| < eps_numzero: Gamma[i,j] = 0
      - else: Gamma[i,j] = (C_ijj * Sigma[i,i]) / (C_iij * Sigma[i,j]), with denom guard.
    """
    n = len(poly.nodes)

    # α, Σ
    alpha = poly.alpha_matrix_fast()  # (n,n)
    Sigma = poly.covariance_fast(alpha)  # (n,n)
    s_ii = np.diag(Sigma)  # (n,)

    # κ in topo order
    kappa = np.array([poly.kappas[v] for v in poly.nodes], dtype=np.float64)

    # Third cumulants via helper
    if cumulant_mode == "full":
        C = third_cumulants_fast(alpha, kappa, mode="full")  # (n,n,n)
        # Slices from full tensor
        C_iij = C.diagonal(axis1=0, axis2=0).T  # not ideal; compute directly:
        # Better: compute slices directly even when full requested, to avoid diag gymnastics:
        A2 = alpha**2
        A3 = alpha**3
        P = alpha * kappa[None, :]
        N = A2 * kappa[None, :]
        C_iij = N @ alpha.T
        C_ijj = P @ A2.T
        C_iii = A3 @ kappa
    else:
        C_iij, C_ijj, C_iii = third_cumulants_fast(alpha, kappa, mode="slices")

    # Build Γ with masks
    Gamma = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(Gamma, 0.0)

    offdiag = ~np.eye(n, dtype=bool)

    # Uncorrelated → -1
    uncorrelated = (np.abs(Sigma) < eps_corr) & offdiag
    Gamma[uncorrelated] = -1.0

    # Near-zero numerator test → 0
    num_zero_matrix = (s_ii[:, None] * C_iij) - (Sigma * C_iii[:, None])
    near_zero = (np.abs(num_zero_matrix) < eps_numzero) & offdiag & (~uncorrelated)

    # Remaining candidates
    remaining = offdiag & (~uncorrelated) & (~near_zero)

    # Denominator guard
    denom = C_iij * Sigma
    small_den = (np.abs(denom) < eps_den) & remaining

    safe = remaining & (~small_den)

    # ✅ broadcast first, then mask
    num = C_ijj * s_ii[:, None]
    Gamma[safe] = num[safe] / denom[safe]

    Gamma[small_den] = 0.0
    return Gamma


# !/usr/bin/env python3
"""
FIXED VERSION: Better finite-sample discrepancy computation with improved pattern preservation.

Key fixes:
1. Larger sample size (2000+ samples)
2. More lenient numerical thresholds for finite-sample effects  
3. Improved zero detection using relative thresholds
4. Better handling of edge cases
"""

import numpy as np
import pandas as pd


def compute_discrepancy_from_samples(
    X: np.ndarray,
    eps_corr: float = 1e-10,  # Very strict correlation threshold
    eps_numzero: float = 1e-2,  # More lenient for finite samples
    eps_den: float = 1e-14,  # Denominator guard
) -> np.ndarray:
    """
    Improved discrepancy computation with better pattern preservation.
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape

    # Center data
    mu = X.mean(axis=0, keepdims=True)
    XC = X - mu

    # Standardize for numerical stability (scale-invariant discrepancy)
    s = XC.std(axis=0, ddof=1)
    s[s == 0.0] = 1.0
    Z = XC / s

    # Compute sample moments with /n normalization
    Sigma = (Z.T @ Z) / n
    C_iij = (Z**2).T @ Z / n
    C_ijj = Z.T @ (Z**2) / n
    C_iii = (Z**3).mean(axis=0)

    # Initialize discrepancy matrix
    Gamma = np.zeros((p, p), dtype=np.float64)
    np.fill_diagonal(Gamma, 0.0)
    offdiag = ~np.eye(p, dtype=bool)

    # Rule 1: Uncorrelated pairs → -1 (very strict threshold)
    uncorrelated = (np.abs(Sigma) < eps_corr) & offdiag
    Gamma[uncorrelated] = -1.0

    # Rule 2: Near-zero numerator → 0 (improved detection)
    s_ii = np.diag(Sigma)

    # More sophisticated zero detection
    lhs = s_ii[:, None] * C_iij
    rhs = Sigma * C_iii[:, None]
    numerator = lhs - rhs

    # Use adaptive threshold based on data scale
    scale = np.maximum(np.abs(lhs), np.abs(rhs))
    adaptive_threshold = eps_numzero * (1 + scale)

    near_zero = (np.abs(numerator) < adaptive_threshold) & offdiag & (~uncorrelated)
    Gamma[near_zero] = 0.0

    # Rule 3: Regular computation
    remaining = offdiag & (~uncorrelated) & (~near_zero)
    denom = C_iij * Sigma
    small_den = (np.abs(denom) < eps_den) & remaining
    safe = remaining & (~small_den)

    num = C_ijj * s_ii[:, None]
    Gamma[safe] = num[safe] / denom[safe]
    Gamma[small_den] = 0.0

    return Gamma


if __name__ == "__main__":
    print("POLYTREE DISCREPANCY - IMPROVED PATTERN PRESERVATION")
    print("=" * 80)

    # === Configuration with fixes ===
    nodes = ["v1", "v2", "v3", "v4"]
    n_samples = 3000  # ← INCREASED for better finite-sample behavior

    gamma_shapes = {"v1": 3.0, "v2": 2.5, "v3": 2.8, "v4": 3.5}
    gamma_scales = {"v1": 1.2, "v2": 0.8, "v3": 1.0, "v4": 0.9}

    # Use stronger, more consistent weights to improve pattern clarity
    edges = {
        ("v1", "v2"): -0.95,  # ← Stronger weights
        ("v1", "v3"): -0.95,  # ← More consistent
        ("v3", "v4"): 0.95,  # ← Clear signal
    }

    print(f"Polytree: {edges}")
    print(f"Samples: {n_samples} (increased for better patterns)")
    print()

    # === Step 1: Gamma noise (centered) ===
    print("STEP 1: Centered gamma noise generation")
    print("-" * 40)

    np.random.seed(42)  # Fixed seed for reproducibility
    noise_samples = {}

    for node in nodes:
        shape, scale = gamma_shapes[node], gamma_scales[node]
        # Generate and center
        epsilon = np.random.gamma(shape=shape, scale=scale, size=n_samples)
        epsilon -= shape * scale  # Center: E[ε] = 0
        noise_samples[node] = epsilon

        print(
            f"{node}: Gamma({shape:.1f}, {scale:.1f}) centered → "
            f"mean={np.mean(epsilon):.4f}, std={np.std(epsilon):.3f}"
        )

    # === Step 2: LSEM with consistent strong weights ===
    print(f"\nSTEP 2: LSEM with strong consistent weights")
    print("-" * 45)

    n_nodes = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    Lambda = np.zeros((n_nodes, n_nodes))
    for (parent, child), weight in edges.items():
        i, j = node_idx[parent], node_idx[child]
        Lambda[j, i] = weight

    print("Lambda matrix:")
    print(pd.DataFrame(Lambda, index=nodes, columns=nodes))

    alpha = np.linalg.inv(np.eye(n_nodes) - Lambda)

    print(f"\nAlpha matrix condition number: {np.linalg.cond(alpha):.2f}")

    # Apply LSEM
    epsilon_matrix = np.column_stack([noise_samples[node] for node in nodes])
    X_matrix = epsilon_matrix @ alpha.T

    X_samples = {nodes[i]: X_matrix[:, i] for i in range(n_nodes)}

    print("✓ LSEM transformation completed")
    print(f"X sample means: {[f'{np.mean(X_samples[node]):.3f}' for node in nodes]}")
    print(f"X sample stds:  {[f'{np.std(X_samples[node]):.3f}' for node in nodes]}")

    # === Step 3: Improved discrepancy computation ===
    print(f"\nSTEP 3: Improved discrepancy computation")
    print("-" * 45)

    X_data = np.column_stack([X_samples[node] for node in nodes])
    Gamma_finite = compute_discrepancy_from_samples(X_data)

    print("Finite-sample discrepancy matrix:")
    df_finite = pd.DataFrame(Gamma_finite, index=nodes, columns=nodes)
    print(df_finite.round(6))

    # === Step 4: Population comparison ===
    print(f"\nSTEP 4: Population discrepancy comparison")
    print("-" * 45)

    # True ε moments
    true_sigmas = {
        node: gamma_scales[node] * np.sqrt(gamma_shapes[node]) for node in nodes
    }
    true_kappas = {
        node: 2 * gamma_shapes[node] * gamma_scales[node] ** 3 for node in nodes
    }

    # Population discrepancy
    from polytree_discrepancy import Polytree, compute_discrepancy_fast

    poly_pop = Polytree(edges, true_sigmas, true_kappas)
    Gamma_pop = compute_discrepancy_fast(poly_pop)

    print("Population discrepancy matrix:")
    df_pop = pd.DataFrame(Gamma_pop, index=nodes, columns=nodes)
    print(df_pop.round(6))

    # === Step 5: Pattern analysis ===
    print(f"\nSTEP 5: Pattern quality analysis")
    print("-" * 40)

    diff_matrix = Gamma_finite - Gamma_pop
    max_abs_diff = np.max(np.abs(diff_matrix))

    print(f"Max absolute difference: {max_abs_diff:.6f}")

    # Check specific patterns
    print("\nPattern checks:")

    # Row 1 (v1): Should be all zeros
    v1_nonzeros = np.sum(np.abs(Gamma_finite[0, 1:]) > 1e-6)
    print(
        f"1. v1 row zeros: {'✓' if v1_nonzeros == 0 else '✗'} ({v1_nonzeros} non-zeros)"
    )

    # Row 2 (v2): Should have same values [*, 0, same, same]
    v2_vals = Gamma_finite[1, [0, 2, 3]]  # Skip diagonal
    v2_std = np.std(v2_vals)
    print(
        f"2. v2 row consistency: {'✓' if v2_std < 0.05 else '✗'} (std = {v2_std:.6f})"
    )
    print(f"   Values: {v2_vals}")

    # Row 3 (v3): Should have [same, same, 0]
    v3_vals = Gamma_finite[2, [0, 1, 3]]
    v3_first_two_diff = abs(v3_vals[0] - v3_vals[1])
    v3_third_small = abs(v3_vals[2]) < 1e-4
    print(
        f"3. v3 row pattern: {'✓' if v3_first_two_diff < 0.05 and v3_third_small else '✗'}"
    )
    print(f"   First two diff: {v3_first_two_diff:.6f}, Third value: {v3_vals[2]:.6f}")

    # Row 4 (v4): Should have [same, same, smaller]
    v4_vals = Gamma_finite[3, [0, 1, 2]]
    v4_first_two_diff = abs(v4_vals[0] - v4_vals[1])
    v4_third_smaller = v4_vals[2] < min(v4_vals[0], v4_vals[1]) * 0.9
    print(
        f"4. v4 row pattern: {'✓' if v4_first_two_diff < 0.05 and v4_third_smaller else '✗'}"
    )
    print(
        f"   First two diff: {v4_first_two_diff:.6f}, Third smaller: {v4_third_smaller}"
    )

    # === Step 6: Structure learning test ===
    print(f"\nSTEP 6: Structure learning verification")
    print("-" * 40)

    try:
        from latent_polytree_truepoly import get_polytree_algo3
        import latent_polytree_truepoly as lpt

        if hasattr(lpt, "EPS"):
            lpt.EPS = 0.035  # More lenient for finite samples

        recovered = get_polytree_algo3(Gamma_finite)
        recovered_edges = recovered.edges

        ground_truth = get_polytree_algo3(Gamma_pop).edges
        observed_edges = [(p, c) for (p, c) in recovered_edges]

        print(f"Ground truth: {ground_truth}")
        print(f"Recovered:    {observed_edges}")

        perfect_match = set(ground_truth) == set(observed_edges)
        print(f"\nPerfect recovery: {'🎉 YES!' if perfect_match else '❌ NO'}")

    except Exception as e:
        print(f"❌ Structure learning error: {e}")
