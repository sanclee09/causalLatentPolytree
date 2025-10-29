"""
validation.py

Utilities for validating latent polytree structures (minimality checks, etc.).
"""

from typing import List, Tuple, Set
from collections import deque


def is_minimal_latent_polytree_check(
        edges: List[Tuple[str, str]],
        hidden_nodes: Set[str]
) -> bool:
    """
    Check if latent polytree is minimal (no redundant degree-2 latent nodes).

    A latent node is redundant if it has exactly one parent and one child
    (undirected degree 2), as it can be bypassed.

    Args:
        edges: List of directed edges (parent, child)
        hidden_nodes: Set of hidden node names

    Returns:
        True if minimal (no redundant latent nodes), False otherwise
    """
    in_degree, out_degree = _compute_degrees(edges, hidden_nodes)

    for h in hidden_nodes:
        # Check if latent node has degree 2 (1 parent + 1 child)
        if in_degree.get(h, 0) == 1 and out_degree.get(h, 0) == 1:
            return False  # Redundant latent node found

    return True


def _compute_degrees(
        edges: List[Tuple[str, str]],
        nodes: Set[str]
) -> Tuple[dict, dict]:
    """Compute in-degree and out-degree for each node."""
    in_degree = {node: 0 for node in nodes}
    out_degree = {node: 0 for node in nodes}

    for parent, child in edges:
        if parent in nodes:
            out_degree[parent] = out_degree.get(parent, 0) + 1
        if child in nodes:
            in_degree[child] = in_degree.get(child, 0) + 1

    return in_degree, out_degree


def has_cycles(edges: List[Tuple[str, str]]) -> bool:
    """
    Check if graph has cycles using DFS.

    Args:
        edges: List of directed edges

    Returns:
        True if graph contains a cycle, False otherwise
    """
    # Build adjacency list
    graph = {}
    nodes = set()

    for parent, child in edges:
        nodes.add(parent)
        nodes.add(child)
        graph.setdefault(parent, []).append(child)

    visited = set()
    rec_stack = set()

    def dfs_has_cycle(node):
        visited.add(node)
        rec_stack.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs_has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node)
        return False

    for node in nodes:
        if node not in visited:
            if dfs_has_cycle(node):
                return True

    return False


def is_polytree(edges: List[Tuple[str, str]]) -> bool:
    """
    Check if graph is a polytree (DAG that is a tree when undirected).

    Args:
        edges: List of directed edges

    Returns:
        True if graph is a valid polytree
    """
    # Must be acyclic
    if has_cycles(edges):
        return False

    # Check connectivity and tree property on undirected version
    nodes = set()
    undirected_graph = {}

    for parent, child in edges:
        nodes.add(parent)
        nodes.add(child)
        undirected_graph.setdefault(parent, []).append(child)
        undirected_graph.setdefault(child, []).append(parent)

    if not nodes:
        return True

    # BFS to check connectivity
    start = next(iter(nodes))
    visited = {start}
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor in undirected_graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # Must be connected and have n-1 edges for n nodes (tree property)
    return len(visited) == len(nodes) and len(edges) == len(nodes) - 1


def get_latent_node_statistics(edges: List[Tuple[str, str]], hidden_nodes: Set[str]) -> dict:
    """
    Compute statistics about latent nodes in the polytree.

    Returns:
        Dictionary with latent node statistics
    """
    in_degree, out_degree = _compute_degrees(edges, hidden_nodes)

    stats = {
        "n_latent": len(hidden_nodes),
        "degree_distribution": [],
        "max_out_degree": 0,
        "latent_to_latent_edges": 0,
    }

    for h in hidden_nodes:
        in_deg = in_degree.get(h, 0)
        out_deg = out_degree.get(h, 0)
        total_deg = in_deg + out_deg

        stats["degree_distribution"].append(total_deg)
        stats["max_out_degree"] = max(stats["max_out_degree"], out_deg)

    # Count latent-to-latent edges
    stats["latent_to_latent_edges"] = sum(
        1 for p, c in edges if p in hidden_nodes and c in hidden_nodes
    )

    return stats