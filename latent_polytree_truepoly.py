"""
Latent polytree recovery (population-ready, deterministic, numerically stable).

Implements the three-phase algorithm:
1. Separation: Partition nodes into sibling groups
2. Tree: Learn directed tree structure within groups
3. Polytree: Merge trees to recover full latent polytree
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import itertools
from collections import deque

# Numerical tolerance
EPS = 1e-7


def _is_zero(x: float) -> bool:
    """Check if value is numerically zero."""
    return np.isclose(x, 0.0, atol=EPS)


def separation(gamma: np.ndarray) -> List[Set[int]]:
    """
    Algorithm 1: Partition nodes into sibling groups (maximal sets with non-negative discrepancies).

    Args:
        gamma: Discrepancy matrix

    Returns:
        List of maximal groups where nodes have non-negative pairwise discrepancies
    """
    n = gamma.shape[0]
    all_nodes = set(range(n))
    maximal_groups: List[Set[int]] = []

    for start_node in range(n):
        candidate = _grow_candidate_group(start_node, all_nodes, gamma)
        _add_if_maximal(candidate, maximal_groups)

    return _deduplicate_groups(maximal_groups)


def _grow_candidate_group(start_node: int, all_nodes: Set[int], gamma: np.ndarray) -> Set[int]:
    """Grow a candidate group starting from a node."""
    candidate: Set[int] = {start_node}
    added = True

    while added:
        added = False
        for node_u in all_nodes:
            if node_u in candidate:
                continue
            if _can_add_to_group(node_u, candidate, gamma):
                candidate.add(node_u)
                added = True

    return candidate


def _can_add_to_group(node: int, group: Set[int], gamma: np.ndarray) -> bool:
    """Check if node can be added to group (all discrepancies non-negative)."""
    for w in group:
        if gamma[node, w] < -EPS or gamma[w, node] < -EPS:
            return False
    return True


def _add_if_maximal(candidate: Set[int], maximal_groups: List[Set[int]]) -> None:
    """Add candidate to maximal groups if it's not a subset of existing group."""
    is_maximal = not any(candidate.issubset(existing) for existing in maximal_groups)

    if is_maximal:
        maximal_groups[:] = [g for g in maximal_groups if not g.issubset(candidate)]
        maximal_groups.append(candidate.copy())


def _deduplicate_groups(groups: List[Set[int]]) -> List[Set[int]]:
    """Remove duplicate groups."""
    seen = set()
    unique: List[Set[int]] = []

    for g in groups:
        fg = frozenset(g)
        if fg not in seen:
            seen.add(fg)
            unique.append(g)

    return unique


class LatentTree:
    """Directed tree structure with latent and observed nodes."""

    def __init__(self) -> None:
        self._children: Dict[str, Set[str]] = {}
        self._parent: Dict[str, str] = {}

    @property
    def nodes(self) -> List[str]:
        """Get all nodes in the tree."""
        all_nodes = set(self._children.keys())
        for children_set in self._children.values():
            all_nodes.update(children_set)
        return list(all_nodes)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """Get all directed edges (parent, child) in the tree."""
        return [
            (parent, child)
            for parent, kids in self._children.items()
            for child in kids
        ]

    @property
    def root(self) -> str:
        """Get the root node (node with no parent)."""
        all_nodes = set(self.nodes)
        children = set(self._parent.keys())
        roots = all_nodes - children

        if len(roots) != 1:
            raise ValueError(f"Tree should have exactly one root, found {len(roots)}: {roots}")

        return next(iter(roots))

    def add_edge(self, parent: str, child: str) -> None:
        """Add a directed edge from parent to child."""
        self._children.setdefault(parent, set()).add(child)

        if child in self._parent and self._parent[child] != parent:
            raise ValueError(f"Node {child} already has parent {self._parent[child]}")

        self._parent[child] = parent

    def add_node(self, node: str) -> None:
        """Add a node with no edges."""
        self._children.setdefault(node, set())

    def copy(self) -> "LatentTree":
        """Create a deep copy of the tree."""
        new_tree = LatentTree()
        for parent, children in self._children.items():
            new_tree.add_node(parent)
            for child in children:
                new_tree.add_node(child)
                new_tree.add_edge(parent, child)
        return new_tree


_latent_counter = itertools.count(1)


def _new_latent() -> str:
    """Generate a new unique latent node identifier."""
    return f"h{next(_latent_counter)}"


def tree(gamma: np.ndarray, _nodes: Optional[List[int]] = None) -> LatentTree:
    """
    Algorithm 2: Learn directed tree structure from discrepancy matrix.

    Args:
        gamma: Discrepancy matrix for nodes
        _nodes: Optional list of node indices (for recursive calls)

    Returns:
        LatentTree with recovered structure
    """
    nodes = list(range(gamma.shape[0])) if _nodes is None else list(_nodes)

    # Base cases
    if len(nodes) == 0:
        return LatentTree()

    if len(nodes) == 1:
        tree_obj = LatentTree()
        tree_obj.add_node(str(nodes[0]))
        return tree_obj

    # Build index mapping and B_v sets
    idx_of = {node_v: i for i, node_v in enumerate(nodes)}
    B = _compute_B_sets(nodes, gamma, idx_of)

    # Check for star structure
    if _is_star_structure(nodes, B):
        return _build_star_tree(nodes, gamma, idx_of)

    # Recursive case: decompose and merge
    return _decompose_and_merge(nodes, gamma, idx_of, B)


def _compute_B_sets(nodes: List[int], gamma: np.ndarray, idx_of: Dict[int, int]) -> Dict[int, Set[int]]:
    """Compute B_v sets: nodes with minimum discrepancy for each v."""
    B: Dict[int, Set[int]] = {}

    for node_v in nodes:
        v_index = idx_of[node_v]
        other_nodes = [n for n in nodes if n != node_v]

        min_discrepancy = min(gamma[v_index, idx_of[u]] for u in other_nodes)

        B[node_v] = {
            node_u
            for node_u in other_nodes
            if gamma[v_index, idx_of[node_u]] <= min_discrepancy + EPS
        }

    return B


def _is_star_structure(nodes: List[int], B: Dict[int, Set[int]]) -> bool:
    """Check if B_v = O \ {v} for all v (star structure indicator)."""
    if len(nodes) <= 1:
        return False

    return all(B[node_v] == set(nodes) - {node_v} for node_v in nodes)


def _build_star_tree(nodes: List[int], gamma: np.ndarray, idx_of: Dict[int, int]) -> LatentTree:
    """Build star tree (all nodes connected to a central root)."""
    tree_obj = LatentTree()

    # Check for observed root (has zero discrepancy to some other node)
    observed_root = _find_observed_root(nodes, gamma, idx_of)

    if observed_root is not None:
        root_name = str(observed_root)
        tree_obj.add_node(root_name)
        for node in nodes:
            if node != observed_root:
                tree_obj.add_node(str(node))
                tree_obj.add_edge(root_name, str(node))
    else:
        # Latent root case
        root_name = _new_latent()
        tree_obj.add_node(root_name)
        for node in nodes:
            tree_obj.add_node(str(node))
            tree_obj.add_edge(root_name, str(node))

    return tree_obj


def _find_observed_root(nodes: List[int], gamma: np.ndarray, idx_of: Dict[int, int]) -> Optional[int]:
    """Find observed root (node with zero discrepancy to another node)."""
    for node_w in nodes:
        w_index = idx_of[node_w]
        if any(_is_zero(gamma[w_index, idx_of[u]]) for u in nodes if u != node_w):
            return node_w
    return None


def _decompose_and_merge(
    nodes: List[int],
    gamma: np.ndarray,
    idx_of: Dict[int, int],
    B: Dict[int, Set[int]]
) -> LatentTree:
    """Decompose into two subtrees and merge."""
    # Select decomposition pivot w
    w = _select_decomposition_pivot(nodes, gamma, idx_of, B)

    # Partition nodes
    O1 = sorted(list(B[w] | {w}))
    O2 = sorted(list(set(nodes) - B[w]))

    # Extract submatrices and recurse
    gamma1 = _extract_submatrix(gamma, O1, idx_of)
    gamma2 = _extract_submatrix(gamma, O2, idx_of)

    T1 = tree(gamma1, _nodes=O1)
    T2 = tree(gamma2, _nodes=O2)

    return _merge_trees(T1, T2, str(w))


def _select_decomposition_pivot(
    nodes: List[int],
    gamma: np.ndarray,
    idx_of: Dict[int, int],
    B: Dict[int, Set[int]]
) -> int:
    """Select pivot node w for decomposition (deterministic selection)."""
    w_candidates = [node for node in nodes if B[node] != set(nodes) - {node}]

    if not w_candidates:
        raise ValueError("Cannot find suitable w for decomposition")

    def w_score(node: int) -> Tuple[float, int]:
        min_gamma = min(gamma[idx_of[node], idx_of[u]] for u in nodes if u != node)
        return (float(min_gamma), node)

    return min(w_candidates, key=w_score)


def _extract_submatrix(gamma: np.ndarray, sub_nodes: List[int], idx_of: Dict[int, int]) -> np.ndarray:
    """Extract submatrix for given nodes."""
    indices = [idx_of[node] for node in sub_nodes]
    return gamma[np.ix_(indices, indices)]


def _merge_trees(T1: LatentTree, T2: LatentTree, w_str: str) -> LatentTree:
    """
    Merge two trees by replacing w in T2 with a new latent h and attaching T1.

    Handles two cases:
    - w is a leaf in T2: Connect parent(w) -> root(T1)
    - w is root of T2: Connect root(T1) -> h
    """
    h = _new_latent()
    merged = LatentTree()

    parent_of_w = T2._parent.get(w_str)

    # Copy T2 with w replaced by h
    node_mapping = {n: (h if n == w_str else n) for n in T2.nodes}
    _copy_tree_with_mapping(T2, merged, node_mapping, skip_node=w_str)

    # Copy T1 unchanged
    _copy_tree_unchanged(T1, merged)

    # Connect the trees
    if parent_of_w is None:
        # w was root of T2: attach root(T1) -> h
        merged.add_node(h)
        if T1.nodes:
            merged.add_edge(T1.root, h)
    else:
        # w was leaf in T2: attach parent(w) -> root(T1)
        parent_mapped = node_mapping[parent_of_w]
        if T1.nodes:
            merged.add_node(parent_mapped)
            merged.add_node(T1.root)
            merged.add_edge(parent_mapped, T1.root)

    return merged


def _copy_tree_with_mapping(
    source: LatentTree,
    dest: LatentTree,
    mapping: Dict[str, str],
    skip_node: Optional[str] = None
) -> None:
    """Copy tree edges with node name mapping, optionally skipping edges to a node."""
    for parent, children in source._children.items():
        for child in children:
            if child == skip_node:
                continue
            parent_mapped = mapping[parent]
            child_mapped = mapping[child]
            dest.add_node(parent_mapped)
            dest.add_node(child_mapped)
            dest.add_edge(parent_mapped, child_mapped)


def _copy_tree_unchanged(source: LatentTree, dest: LatentTree) -> None:
    """Copy all edges from source to dest unchanged."""
    for parent, children in source._children.items():
        for child in children:
            dest.add_node(parent)
            dest.add_node(child)
            dest.add_edge(parent, child)


class PolyDAG:
    """Polytree structure (DAG that is a forest when edges are undirected)."""

    def __init__(self) -> None:
        self.children: Dict[str, Set[str]] = {}
        self.parents: Dict[str, Set[str]] = {}

    def add_node(self, node_v: str) -> None:
        """Add a node."""
        self.children.setdefault(node_v, set())
        self.parents.setdefault(node_v, set())

    def add_edge(self, parent: str, child: str) -> None:
        """Add a directed edge."""
        self.add_node(parent)
        self.add_node(child)
        self.children[parent].add(child)
        self.parents[child].add(parent)

    @property
    def nodes(self) -> List[str]:
        """Get all nodes."""
        nodes_set = set(self.children.keys())
        for children in self.children.values():
            nodes_set.update(children)
        return sorted(nodes_set)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """Get all directed edges."""
        return [
            (parent, child)
            for parent, children in self.children.items()
            for child in children
        ]


def get_polytree_algo3(gamma: np.ndarray) -> PolyDAG:
    """
    Algorithm 3: Learn minimal latent polytree from discrepancy matrix.

    Args:
        gamma: Discrepancy matrix for observed nodes

    Returns:
        PolyDAG representing the recovered latent polytree
    """
    # Partition into sibling groups
    sub_trees = separation(gamma)
    sub_trees = sorted(sub_trees, key=lambda g: (-len(g), sorted(g)))

    # Learn tree structure for each group and merge
    polytree = PolyDAG()
    for G in sub_trees:
        sub = sorted(G)
        T = tree(gamma[np.ix_(sub, sub)], _nodes=sub)
        for parent, child in T.edges:
            polytree.add_edge(parent, child)

    # Rename latent nodes in topological order
    polytree = rename_latent_nodes_by_topology(polytree)

    return polytree


def rename_latent_nodes_by_topology(polytree: PolyDAG) -> PolyDAG:
    """Rename latent nodes (h1, h2, ...) according to topological order."""
    latent_nodes = [n for n in polytree.nodes if n.startswith("h")]

    if not latent_nodes:
        return polytree

    # Get topological order
    topo_order = _topological_sort(polytree.nodes, polytree.edges)
    latent_in_topo = [n for n in topo_order if n.startswith("h")]

    # Create mapping
    latent_mapping = {old: f"h{i + 1}" for i, old in enumerate(latent_in_topo)}

    # Rebuild polytree with renamed nodes
    new_polytree = PolyDAG()
    for parent, child in polytree.edges:
        new_parent = latent_mapping.get(parent, parent)
        new_child = latent_mapping.get(child, child)
        new_polytree.add_edge(new_parent, new_child)

    return new_polytree


def _topological_sort(nodes: List[str], edges: List[Tuple[str, str]]) -> List[str]:
    """Perform topological sort on DAG."""
    in_degree = {node: 0 for node in nodes}
    adj = {node: [] for node in nodes}

    for parent, child in edges:
        adj[parent].append(child)
        in_degree[child] += 1

    queue = deque([node for node in nodes if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result