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


if __name__ == "__main__":
    # Simple test run
    edges = {("v1", "v2"): 2.0, ("v1", "v3"): 3.0, ("v3", "v4"): 4.0}
    sigmas = {"v1": 1.0, "v2": 1.0, "v3": 1.0, "v4": 1.0}
    kappas = {"v1": 1.0, "v2": 1.0, "v3": 1.0, "v4": 1.0}
    poly = Polytree(edges, sigmas, kappas)
    Gamma = compute_discrepancy_fast(poly)
    import pandas as pd

    print(pd.DataFrame(Gamma, index=poly.nodes, columns=poly.nodes))
