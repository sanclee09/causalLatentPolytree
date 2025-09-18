"""
learn_with_hidden.py

Utilities to (a) detect "learnable" (to-be-hidden) nodes from edge dicts,
(b) compute the observed-only cumulant discrepancy matrix Γ_obs,
and (c) recover a polytree using polytree_true on Γ_obs.
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Iterable, Optional, Set

import numpy as np

from numerical_stabilization import normalize_gamma
from polytree_discrepancy import Polytree, compute_discrepancy_fast
from latent_polytree_truepoly import get_polytree_algo3

# -------- Hidden-node detection --------


def detect_learnable_nodes(
    edges: Dict[Tuple[str, str], float], min_outdegree: int = 2
) -> Set[str]:
    """
    Return nodes that have out-degree >= min_outdegree.
    These are treated as "learnable" (to be hidden) when constructing Γ_obs.
    """
    outdeg: Dict[str, int] = {}
    for u, v in edges.keys():
        outdeg[u] = outdeg.get(u, 0) + 1
        outdeg.setdefault(v, outdeg.get(v, 0))
    return {u for u, d in outdeg.items() if d >= min_outdegree}


# -------- Observed Γ construction --------


def observed_gamma_from_params(
    edges: Dict[Tuple[str, str], float],
    sigmas: Dict[str, float],
    kappas: Dict[str, float],
    hidden: Optional[Iterable[str]] = None,
    auto_detect_hidden: bool = True,
    min_outdegree: int = 2,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build full Γ from (edges, sigmas, kappas), then restrict to observed nodes.

    Returns:
        (observed_discrepancy_matrix, observed_nodes, hidden_nodes)
    """
    poly = Polytree(edges, sigmas, kappas)
    full_discrepancy_matrix = compute_discrepancy_fast(poly)
    order = poly.nodes

    hidden_nodes: Set[str] = set(hidden or [])
    if auto_detect_hidden:
        hidden_nodes |= detect_learnable_nodes(edges, min_outdegree=min_outdegree)

    observed_nodes = [n for n in order if n not in hidden_nodes]
    obs_nodes_index = [order.index(n) for n in observed_nodes]
    observed_discrepancy_matrix = full_discrepancy_matrix[
        np.ix_(obs_nodes_index, obs_nodes_index)
    ]
    return observed_discrepancy_matrix, observed_nodes, sorted(hidden_nodes)


# -------- End-to-end learning --------


def learn_from_params_with_auto_hidden(
    edges: Dict[Tuple[str, str], float],
    sigmas: Dict[str, float],
    kappas: Dict[str, float],
    hidden: Optional[Iterable[str]] = None,
    auto_detect_hidden: bool = True,
    min_outdegree: int = 2,
):
    """
    Convenience wrapper:
      - detect hidden nodes (or use provided 'hidden')
      - compute Γ_obs
      - run polytree_true(Γ_obs)
      - map result indices back to observed node names

    Returns:
      observed_discrepancy_matrix (np.ndarray), observed_nodes (List[str]), hidden_nodes (List[str]), edges_named (List[Tuple[str,str]])
    """
    observed_discrepancy_matrix, observed_nodes, hidden_nodes = (
        observed_gamma_from_params(
            edges,
            sigmas,
            kappas,
            hidden=hidden,
            auto_detect_hidden=auto_detect_hidden,
            min_outdegree=min_outdegree,
        )
    )
    observed_polytree = get_polytree_algo3(observed_discrepancy_matrix)

    def name(x: str) -> str:
        if x.startswith("h"):
            return x
        return observed_nodes[int(x)]

    edges_named = [
        (name(parent), name(child)) for (parent, child) in observed_polytree.edges
    ]
    return observed_discrepancy_matrix, observed_nodes, hidden_nodes, edges_named


# ---- Demo ----
if __name__ == "__main__":
    # Four-node test where v1 is a branching node (auto-detected as hidden)
    edges = {("v1", "v2"): 2.0, ("v1", "v3"): 3.0, ("v3", "v4"): 4.0}
    sigmas = {"v1": 1.0, "v2": 1.0, "v3": 1.0, "v4": 1.0}
    kappas = {"v1": 1.0, "v2": 1.0, "v3": 1.0, "v4": 1.0}

    observed_discrepancy_matrix, observed_nodes, hidden_nodes, edges_named = (
        learn_from_params_with_auto_hidden(
            edges, sigmas, kappas, hidden=None, auto_detect_hidden=True, min_outdegree=2
        )
    )

    print("Observed nodes:", observed_nodes)
    print("Hidden (learnable) nodes:", hidden_nodes)
    print("Recovered edges:", sorted(edges_named))
