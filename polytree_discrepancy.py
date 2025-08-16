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
                raise ValueError(f"Node {v} has in-degree > 1; the graph is not a polytree.")
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("The graph contains a directed cycle; a polytree must be acyclic.")
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
        alpha: Dict[str, Dict[str, float]] = {i: {j: 0.0 for j in self.graph.nodes} for i in self.graph.nodes}
        for v in self._topo_order():
            alpha[v][v] = 1.0
            for u in self.graph.predecessors(v):
                lam = self.edges[(u, v)]
                for h in self.graph.nodes:
                    alpha[v][h] += lam * alpha[u][h]
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
            num_zero = Sigma[i_idx, i_idx] * C3[(i, i, j)] - Sigma[i_idx, j_idx] * C3[(i, i, i)]
            if np.isclose(num_zero, 0.0):
                Gamma[i_idx, j_idx] = 0.0
                continue
            numerator = C3[(i, j, j)] * Sigma[i_idx, i_idx]
            denominator = C3[(i, i, j)] * Sigma[i_idx, j_idx]
            Gamma[i_idx, j_idx] = numerator / denominator
    return Gamma


if __name__ == "__main__":
    # Simple test run
    edges = {("v1", "v2"): 2.0, ("v1", "v3"): 3.0, ("v3", "v4"): 4.0}
    sigmas = {"v1": 1.0, "v2": 1.0, "v3": 1.0, "v4": 1.0}
    kappas = {"v1": 1.0, "v2": 1.0, "v3": 1.0, "v4": 1.0}
    poly = Polytree(edges, sigmas, kappas)
    Gamma = compute_discrepancy(poly)
    import pandas as pd
    print(pd.DataFrame(Gamma, index=poly.nodes, columns=poly.nodes))
