from __future__ import annotations
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats


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


def _center_columns(X: np.ndarray) -> np.ndarray:
    """Return column-centered copy of X."""
    return X - X.mean(axis=0, keepdims=True)


def _sample_moments_from_centered(
    XC: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Σ, C_iij, C_ijj, C_iii using /n normalization from centered data XC.
    Returns: (Sigma, C_iij, C_ijj, C_iii)
    """
    n, p = XC.shape
    Sigma = (XC.T @ XC) / n  # (p,p)
    C_iij = (XC**2).T @ XC / n  # (p,p)
    C_ijj = XC.T @ (XC**2) / n  # (p,p)
    C_iii = (XC**3).mean(axis=0)  # (p,)
    return Sigma, C_iij, C_ijj, C_iii


def _corr_from_cov(Sigma: np.ndarray, var_floor: float = 1e-18) -> np.ndarray:
    """
    Build the correlation matrix R from covariance Sigma, with a small variance floor.
    """
    s_ii = np.diag(Sigma).astype(np.float64)
    s_ii = np.maximum(s_ii, var_floor)
    denom = np.sqrt(s_ii[:, None] * s_ii[None, :])
    R = Sigma / denom
    # keep diagonal exactly 1.0 (avoid tiny drift)
    np.fill_diagonal(R, 1.0)
    return R


def _fisher_uncorrelated_mask(
    Sigma: np.ndarray, n: int, alpha: float = 0.01
) -> np.ndarray:
    """
    Off-diagonal mask for 'uncorrelated' using Fisher's r->z transform:
        z = atanh(r) ~ N(0, 1/(n-3)) under H0: rho=0.
    We declare 'uncorrelated' when |z| < z_(1-alpha/2)/sqrt(n-3).
    This threshold naturally decreases as n increases.
    """
    p = Sigma.shape[0]
    offdiag = ~np.eye(p, dtype=bool)

    # correlation
    R = _corr_from_cov(Sigma)

    # clip to avoid atanh(±1) → inf
    R_clipped = np.clip(R, -0.999999, 0.999999)
    Z = np.arctanh(R_clipped)

    # critical value
    denom = np.sqrt(max(n - 3, 1))  # guard n<4
    zcrit = stats.norm.ppf(1 - alpha / 2.0)
    thresh = zcrit / denom

    return (np.abs(Z) < thresh) & offdiag


def _near_zero_mask(
    Sigma: np.ndarray,
    C_iij: np.ndarray,
    C_iii: np.ndarray,
    s_ii: np.ndarray,
    eps_numzero: float,
    ban_mask: np.ndarray,
) -> np.ndarray:
    """
    Near-zero identity: |Σ_ii*C_iij - Σ_ij*C_iii| <= eps_numzero * (|lhs|+|rhs|).
    Only on off-diagonal & not banned by ban_mask (e.g., uncorrelated).
    """
    p = Sigma.shape[0]
    offdiag = ~np.eye(p, dtype=bool)
    lhs = s_ii[:, None] * C_iij
    rhs = Sigma * C_iii[:, None]
    diff = np.abs(lhs - rhs)
    scale = np.abs(lhs) + np.abs(rhs) + 1e-18
    return (diff <= eps_numzero * scale) & offdiag & (~ban_mask)


def _ratio_update_n_adaptive(
    Gamma: np.ndarray,
    Sigma: np.ndarray,
    C_iij: np.ndarray,
    C_ijj: np.ndarray,
    s_ii: np.ndarray,
    base_mask: np.ndarray,
    n: int,
    q: float = 0.1,  # quantile for typical magnitude
    c_rel: float = 3e-3,  # relative guard coefficient
) -> None:
    """
    Γ_ij = (C_ijj * Σ_ii) / (C_iij * Σ_ij) on safe entries.
    Small denominators (relative to a typical scale) are treated as 0.
    Guard shrinks like 1/sqrt(n).
    """
    denom = C_iij * Sigma  # same as before
    # typical magnitude from middle of distribution (ignore exact zeros)
    nz = np.abs(denom[denom != 0])
    scale = np.quantile(nz, q) if nz.size else 1.0
    # n-adaptive threshold: smaller with more data
    thresh = (c_rel / np.sqrt(n)) * scale

    small_den = (np.abs(denom) < thresh) & base_mask
    safe = base_mask & (~small_den)
    num = C_ijj * s_ii[:, None]

    Gamma[safe] = num[safe] / denom[safe]
    Gamma[small_den] = 0.0


# -------- main API (same outputs/variables) ------------------------


def compute_discrepancy_from_samples(
    X: np.ndarray,
    eps_numzero: float | None = None,
    *,
    alpha_corr: float | None = None,
) -> np.ndarray:
    X = np.asarray(X, np.float64)
    n, p = X.shape

    XC = X - X.mean(axis=0, keepdims=True)
    Sigma, C_iij, C_ijj, C_iii = _sample_moments_from_centered(XC)
    s_ii = np.diag(Sigma)

    # n-adaptive tolerances
    if eps_numzero is None:
        eps_numzero = max(5e-3, 0.8 / np.sqrt(n))
    if alpha_corr is None:
        alpha_corr = min(0.05, 2.0 / np.sqrt(n))

    Gamma = np.zeros((p, p), np.float64)
    np.fill_diagonal(Gamma, 0.0)

    # Fisher "uncorrelated"
    uncorrelated = _fisher_uncorrelated_mask(Sigma, n=n, alpha=alpha_corr)
    Gamma[uncorrelated] = -1.0

    # Near-zero identity (relative, n-adaptive)
    near_zero = _near_zero_mask(
        Sigma, C_iij, C_iii, s_ii, eps_numzero, ban_mask=uncorrelated
    )
    Gamma[near_zero] = 0.0

    # Ratio with n-adaptive small-denom guard (no ridge)
    off = ~np.eye(p, dtype=bool)
    remaining = off & (~uncorrelated) & (~near_zero)
    _ratio_update_n_adaptive(Gamma, Sigma, C_iij, C_ijj, s_ii, remaining, n=n)

    return Gamma


# ----------------------------- Helpers -------------------------------- #


def create_sample_configuration() -> Dict[str, Any]:
    """Exact same config that gave the good result."""
    nodes = ["v1", "v2", "v3", "v4"]
    edges = {("v1", "v2"): -0.95, ("v1", "v3"): -0.95, ("v3", "v4"): 0.95}
    gamma_shapes = {"v1": 3.0, "v2": 2.5, "v3": 2.8, "v4": 3.5}
    gamma_scales = {"v1": 1.2, "v2": 0.8, "v3": 1.0, "v4": 0.9}
    return {
        "nodes": nodes,
        "edges": edges,
        "gamma_shapes": gamma_shapes,
        "gamma_scales": gamma_scales,
        "n_samples": 15000,
        "seed": 42,
    }


def topo_order_from_edges(
    nodes: List[str], edges: Dict[Tuple[str, str], float]
) -> List[str]:
    """Respect DAG topological order (no alphabetical resorting)."""
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges.keys())
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("Edges must define a DAG.")
    return list(nx.topological_sort(g))


def generate_noise_samples(
    nodes: List[str],
    gamma_shapes: Dict[str, float],
    gamma_scales: Dict[str, float],
    n_samples: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Generate centered Gamma noise using analytic mean (shape*scale)."""
    np.random.seed(seed)
    noise_samples: Dict[str, np.ndarray] = {}

    print("STEP 1: Centered gamma noise generation")
    print("-" * 40)
    for node in nodes:
        shape, scale = gamma_shapes[node], gamma_scales[node]
        epsilon = np.random.gamma(shape=shape, scale=scale, size=n_samples)
        epsilon -= shape * scale  # analytic centering
        noise_samples[node] = epsilon
        print(
            f"{node}: Gamma({shape:.1f}, {scale:.1f}) centered → "
            f"mean={np.mean(epsilon):.4f}, std={np.std(epsilon):.3f}"
        )
    return noise_samples


def apply_lsem_transformation(
    noise_samples: Dict[str, np.ndarray],
    edges: Dict[Tuple[str, str], float],
    nodes: List[str],
) -> Dict[str, np.ndarray]:
    """X = (I - Λ)^(-1) ε with Λ[j,i] = weight for i→j."""
    print(f"\nSTEP 2: LSEM with strong consistent weights")
    print("-" * 45)
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    Lambda = np.zeros((n, n))
    for (parent, child), w in edges.items():
        Lambda[idx[child], idx[parent]] = w

    print("Lambda matrix:")
    print(pd.DataFrame(Lambda, index=nodes, columns=nodes))

    alpha = np.linalg.inv(np.eye(n) - Lambda)
    print(f"\nAlpha matrix condition number: {np.linalg.cond(alpha):.2f}")

    epsilon_matrix = np.column_stack([noise_samples[v] for v in nodes])
    X_matrix = epsilon_matrix @ alpha.T

    X_samples = {nodes[i]: X_matrix[:, i] for i in range(n)}
    print("✓ LSEM transformation completed")
    print(f"X sample means: {[f'{np.mean(X_samples[v]):.3f}' for v in nodes]}")
    print(f"X sample stds:  {[f'{np.std(X_samples[v]):.3f}' for v in nodes]}")
    return X_samples


def finite_sample_discrepancy(
    X_samples: Dict[str, np.ndarray], nodes: List[str]
) -> np.ndarray:
    print(f"\nSTEP 3: Improved discrepancy computation")
    print("-" * 45)
    X = np.column_stack([X_samples[v] for v in nodes])
    return compute_discrepancy_from_samples(X)


def population_discrepancy(
    edges: Dict[Tuple[str, str], float],
    gamma_shapes: Dict[str, float],
    gamma_scales: Dict[str, float],
) -> np.ndarray:
    print(f"\nSTEP 4: Population discrepancy comparison")
    print("-" * 45)
    sigmas = {v: gamma_scales[v] * np.sqrt(gamma_shapes[v]) for v in gamma_shapes}
    kappas = {v: 2 * gamma_shapes[v] * (gamma_scales[v] ** 3) for v in gamma_shapes}
    poly = Polytree(edges, sigmas, kappas)
    return compute_discrepancy_fast(poly)


def show_matrices(G_finite: np.ndarray, G_pop: np.ndarray, nodes: List[str]) -> None:
    print("Finite-sample discrepancy matrix:")
    print(pd.DataFrame(G_finite, index=nodes, columns=nodes).round(6))
    print("\nPopulation discrepancy matrix:")
    print(pd.DataFrame(G_pop, index=nodes, columns=nodes).round(6))


def analyze_patterns(G_finite: np.ndarray, G_pop: np.ndarray, nodes: List[str]) -> None:
    print(f"\nSTEP 5: Pattern quality analysis")
    print("-" * 40)
    diff = G_finite - G_pop
    print(f"Max absolute difference: {np.max(np.abs(diff)):.6f}")

    # Your checks
    v1_nonzeros = np.sum(np.abs(G_finite[0, 1:]) > 1e-6)
    print(f"\nPattern checks:")
    print(
        f"1. v1 row zeros: {'✓' if v1_nonzeros == 0 else '✗'} ({v1_nonzeros} non-zeros)"
    )
    v2_vals = G_finite[1, [0, 2, 3]]
    print(
        f"2. v2 row consistency: {'✓' if np.std(v2_vals) < 0.05 else '✗'} (std = {np.std(v2_vals):.6f})"
        f"\n   Values: {v2_vals}"
    )
    v3_vals = G_finite[2, [0, 1, 3]]
    print(
        f"3. v3 row pattern: {'✓' if abs(v3_vals[0]-v3_vals[1]) < 0.05 and abs(v3_vals[2]) < 1e-4 else '✗'}"
        f"\n   First two diff: {abs(v3_vals[0]-v3_vals[1]):.6f}, Third value: {v3_vals[2]:.6f}"
    )
    v4_vals = G_finite[3, [0, 1, 2]]
    smaller = v4_vals[2] < min(v4_vals[0], v4_vals[1]) * 0.9
    print(
        f"4. v4 row pattern: {'✓' if abs(v4_vals[0]-v4_vals[1]) < 0.05 and smaller else '✗'}"
        f"\n   First two diff: {abs(v4_vals[0]-v4_vals[1]):.6f}, Third smaller: {smaller}"
    )


def structure_learning_check(G_finite: np.ndarray, G_pop: np.ndarray, n: int) -> None:
    """
    Verify structure recovery. Uses an n-adaptive tolerance for the learner.
    """
    print(f"\nSTEP 6: Structure learning verification")
    print("-" * 40)
    try:
        from latent_polytree_truepoly import get_polytree_algo3
        import latent_polytree_truepoly as lpt

        # n-adaptive equality tolerance: shrinks like 1/sqrt(n)
        eps_n = max(0.005, 0.6 / np.sqrt(n))
        if hasattr(lpt, "EPS"):
            lpt.EPS = eps_n

        gt = get_polytree_algo3(G_pop).edges
        rec = get_polytree_algo3(G_finite).edges
        print(f"Using EPS={eps_n:.6f}")
        print(f"Ground truth: {gt}")
        print(f"Recovered:    {rec}")
        ok = set(gt) == set(rec)
        print(f"\nPerfect recovery: {'🎉 YES!' if ok else '❌ NO'}")
    except Exception as e:
        print(f"❌ Structure learning error: {e}")


# ----------------------------- Main ---------------------------------- #


def run_polytree_discrepancy_demo(
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if config is None:
        config = create_sample_configuration()

    print("POLYTREE DISCREPANCY - IMPROVED PATTERN PRESERVATION")
    print("=" * 80)

    nodes = config["nodes"]
    edges = config["edges"]
    gamma_shapes = config["gamma_shapes"]
    gamma_scales = config["gamma_scales"]
    n_samples = config["n_samples"]
    seed = config.get("seed", 42)

    # ensure topological order is respected
    nodes = topo_order_from_edges(nodes, edges)

    print(f"Polytree: {edges}")
    print(f"Samples: {n_samples} (increased for better patterns)")
    print()

    noise_samples = generate_noise_samples(
        nodes, gamma_shapes, gamma_scales, n_samples, seed
    )
    X_samples = apply_lsem_transformation(noise_samples, edges, nodes)
    G_finite = finite_sample_discrepancy(X_samples, nodes)
    G_pop = population_discrepancy(edges, gamma_shapes, gamma_scales)
    show_matrices(G_finite, G_pop, nodes)
    analyze_patterns(G_finite, G_pop, nodes)
    structure_learning_check(G_finite, G_pop, n=n_samples)

    return {
        "nodes": nodes,
        "edges": edges,
        "gamma_shapes": gamma_shapes,
        "gamma_scales": gamma_scales,
        "n_samples": n_samples,
        "Gamma_finite": G_finite,
        "Gamma_pop": G_pop,
    }


# ------------------------- Random polytree ---------------------------- #


def run_polytree_discrepancy_for_random_tree(
    polytree_data: Dict[str, Any], n_samples: int = 3000, seed: int = 42
):
    """
    Accepts a dict like:
      {
        "weights": {("v1","v2"): w12, ...},  # directed edges with weights
        "nodes": ["v1", ...],                # optional; will derive if absent
        "gamma_shapes": {...},               # optional
        "gamma_scales": {...},               # optional
      }
    """
    print("POLYTREE DISCREPANCY - RANDOM POLYTREE ANALYSIS")
    print("=" * 80)

    edges = polytree_data["weights"]
    nodes = polytree_data.get("nodes")
    if nodes is None:
        nodes = sorted(set(u for (u, _) in edges) | set(v for (_, v) in edges))
    nodes = topo_order_from_edges(nodes, edges)  # CRITICAL: topo order

    gamma_shapes = polytree_data.get("gamma_shapes") or {v: 2.5 for v in nodes}
    gamma_scales = polytree_data.get("gamma_scales") or {v: 1.0 for v in nodes}

    print(f"Random polytree: {len(nodes)} nodes, {len(edges)} edges")
    print(f"Directed edges: {sorted(edges.keys())}")

    noise_samples = generate_noise_samples(
        nodes, gamma_shapes, gamma_scales, n_samples, seed
    )
    X_samples = apply_lsem_transformation(noise_samples, edges, nodes)
    G_finite = finite_sample_discrepancy(X_samples, nodes)

    # population using the same gamma params
    G_pop = population_discrepancy(edges, gamma_shapes, gamma_scales)
    show_matrices(G_finite, G_pop, nodes)
    analyze_patterns(G_finite, G_pop, nodes)
    structure_learning_check(G_finite, G_pop)

    return {
        "nodes": nodes,
        "edges": edges,
        "Gamma_finite": G_finite,
        "Gamma_pop": G_pop,
        "gamma_shapes": gamma_shapes,
        "gamma_scales": gamma_scales,
        "n_samples": n_samples,
    }


if __name__ == "__main__":
    run_polytree_discrepancy_demo()
