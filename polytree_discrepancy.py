"""Compute cumulant–based discrepancy matrices for polytrees.

This module provides a simple utility for representing a polytree with
arbitrary structural coefficients, noise variances and third cumulants,
and computing the covariance matrix, third–order cumulant tensor and
the cumulant discrepancy matrix defined in the thesis draft.

Example usage:

    from polytree_discrepancy import Polytree, compute_discrepancy

    # define a four–node polytree with edges and structural coefficients
    edges = {
        ("v1", "v2"): 2.0,
        ("v1", "v3"): 3.0,
        ("v3", "v4"): 4.0,
    }
    # noise variances (sigma^2) and third cumulants (kappa)
    sigmas = {"v1": 1.0, "v2": 1.0, "v3": 1.0, "v4": 1.0}
    kappas = {"v1": 1.0, "v2": 1.0, "v3": 1.0, "v4": 1.0}

    poly = Polytree(edges, sigmas, kappas)
    gamma = compute_discrepancy(poly)

    # print the discrepancy matrix in a readable format
    import pandas as pd
    df = pd.DataFrame(gamma, index=poly.nodes, columns=poly.nodes)
    print(df)

You can modify the ``edges``, ``sigmas`` and ``kappas`` dictionaries
to explore other structures.

The implementation uses ``networkx`` for basic graph utilities, but
does not depend on any external C extensions and should run with the
standard ``networkx`` package.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, List
import numpy as np
import networkx as nx

@dataclass
class Polytree:
    """Represents a directed polytree with associated parameters.

    Attributes
    ----------
    edges: Dict[Tuple[str, str], float]
        A mapping from parent–child edge (u, v) to its structural
        coefficient λ_{u,v}.  The underlying graph must be a directed
        tree (each node has at most one parent).

    sigmas: Dict[str, float]
        Variance (σ_i^2) of the noise variable ε_i at each node i.

    kappas: Dict[str, float]
        Third cumulant (κ_i) of the noise variable ε_i at each node i.
    """

    edges: Dict[Tuple[str, str], float]
    sigmas: Dict[str, float]
    kappas: Dict[str, float]
    graph: nx.DiGraph = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the underlying NetworkX graph and verify the polytree property."""
        g = nx.DiGraph()
        g.add_edges_from(self.edges.keys())
        # ensure each node has at most one parent
        for v in g.nodes:
            if g.in_degree(v) > 1:
                raise ValueError(f"Node {v} has in-degree > 1; the graph is not a polytree.")
        # ensure the graph is acyclic and connected (tree on the undirected level)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("The graph contains a directed cycle; a polytree must be acyclic.")
        und = g.to_undirected()
        if not nx.is_tree(und):
            raise ValueError("The underlying undirected graph must be a tree (connected and no cycles).")
        self.graph = g

    @property
    def nodes(self) -> List[str]:
        """Return the nodes in topological order."""
        return list(nx.topological_sort(self.graph))

    def alpha_matrix(self) -> Dict[str, Dict[str, float]]:
        """Compute the coefficients of each ε_j in each X_i.

        Returns a nested dictionary alpha[i][j] giving the coefficient
        of noise ε_j in the linear expression for X_i.  The computation
        follows the recursion X_i = sum_{p in pa(i)} λ_{p,i} X_p + ε_i.
        """
        # initialize alpha with zeros
        alpha: Dict[str, Dict[str, float]] = {i: {j: 0.0 for j in self.graph.nodes} for i in self.graph.nodes}
        # process nodes in topological order
        for v in self.nodes:
            # coefficient of its own noise
            alpha[v][v] = 1.0
            # propagate coefficients from parents
            for u in self.graph.predecessors(v):
                lam = self.edges[(u, v)]
                for h in self.graph.nodes:
                    alpha[v][h] += lam * alpha[u][h]
        return alpha

    def covariance(self) -> np.ndarray:
        """Compute the covariance matrix Σ of the observed variables X.

        Returns an |V|×|V| array ordered according to ``self.nodes``.
        """
        n = len(self.nodes)
        Sigma = np.zeros((n, n), dtype=float)
        alpha = self.alpha_matrix()
        for i_idx, i in enumerate(self.nodes):
            for j_idx, j in enumerate(self.nodes):
                cov = 0.0
                # sum over noise sources
                for h in self.nodes:
                    cov += alpha[i][h] * alpha[j][h] * (self.sigmas[h] ** 2)
                Sigma[i_idx, j_idx] = cov
        return Sigma

    def third_cumulants(self) -> Dict[Tuple[str, str, str], float]:
        """Compute the third–order cumulant tensor C^{(3)}.

        Returns a dictionary keyed by triples of node names.  Only
        combinations where the three indices share the same noise
        source yield nonzero entries.
        """
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
    """Compute the cumulant discrepancy matrix Γ for a given polytree.

    Parameters
    ----------
    poly : Polytree
        The polytree with structural coefficients and noise parameters.

    Returns
    -------
    numpy.ndarray
        An |V|×|V| array Γ with entries γ(i,j) as defined in the thesis
        (equation (4.1) in Nisrina’s thesis).  The nodes are ordered
        according to ``poly.nodes``.
    """
    nodes = poly.nodes
    n = len(nodes)
    Sigma = poly.covariance()
    C3 = poly.third_cumulants()
    Gamma = np.zeros((n, n), dtype=float)
    # helper to check correlation zero (Wright's formula): Σ_{i,j} = 0
    def is_uncorrelated(i_idx: int, j_idx: int) -> bool:
        return np.isclose(Sigma[i_idx, j_idx], 0.0)
    for i_idx, i in enumerate(nodes):
        for j_idx, j in enumerate(nodes):
            if i == j:
                Gamma[i_idx, j_idx] = 0.0
                continue
            # check if there is no trek (correlation zero)
            if is_uncorrelated(i_idx, j_idx):
                Gamma[i_idx, j_idx] = -1.0
                continue
            # compute terms for the 0-case: Σ_{ii} C^{(3)}_{i,i,j} - Σ_{i,j} C^{(3)}_{i,i,i}
            num_zero = Sigma[i_idx, i_idx] * C3[(i, i, j)] - Sigma[i_idx, j_idx] * C3[(i, i, i)]
            if np.isclose(num_zero, 0.0):
                Gamma[i_idx, j_idx] = 0.0
                continue
            # otherwise compute the ratio
            numerator = C3[(i, j, j)] * Sigma[i_idx, i_idx]
            denominator = C3[(i, i, j)] * Sigma[i_idx, j_idx]
            Gamma[i_idx, j_idx] = numerator / denominator
    return Gamma


def example_four_node() -> None:
    """Run the four–node example from the thesis draft and print results."""
    # define structural coefficients
    edges = {
        ("v1", "v2"): 2.0,
        ("v1", "v3"): 3.0,
        ("v3", "v4"): 4.0,
    }
    # noise variances and third cumulants
    sigmas = {"v1": 1.0, "v2": 1.0, "v3": 1.0, "v4": 1.0}
    kappas = {"v1": 1.0, "v2": 1.0, "v3": 1.0, "v4": 1.0}
    poly = Polytree(edges, sigmas, kappas)
    gamma = compute_discrepancy(poly)
    # pretty print
    import pandas as pd  # type: ignore
    df = pd.DataFrame(gamma, index=poly.nodes, columns=poly.nodes)
    print("Covariance matrix Σ:")
    Sigma = poly.covariance()
    df_sigma = pd.DataFrame(Sigma, index=poly.nodes, columns=poly.nodes)
    print(df_sigma.to_string())
    print("\nDiscrepancy matrix Γ:")
    print(df.to_string())


if __name__ == "__main__":
    example_four_node()