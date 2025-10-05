"""
Latent polytree recovery (population-ready, deterministic, numerically stable).

Key fixes vs earlier version:
1) Numerical tolerance in B_v construction and star/root tests.
2) Deterministic, stable selection of the decomposition pivot w.
3) Correct merger wiring when w is the root of T2: connect r1 -> h (not h -> r1).
"""

from __future__ import annotations
from typing import *
import numpy as np
import itertools

# Numerical tolerance
EPS = 1e-7


def _is_zero(x: float) -> bool:
    return np.isclose(x, 0.0, atol=EPS)


def separation(gamma: np.ndarray) -> List[Set[int]]:
    n = gamma.shape[0]
    all_nodes = set(range(n))
    maximal_groups: List[Set[int]] = []

    for start_node in range(n):
        candidate: Set[int] = {start_node}
        added = True
        while added:
            added = False
            for node_u in all_nodes:
                if node_u in candidate:
                    continue
                ok = True
                for w in candidate:
                    if gamma[node_u, w] < -EPS or gamma[w, node_u] < -EPS:
                        ok = False
                        break
                if ok:
                    candidate.add(node_u)
                    added = True

        is_maximal = True
        for existing_group in maximal_groups:
            if candidate.issubset(existing_group):
                is_maximal = False
                break
        if is_maximal:
            maximal_groups = [g for g in maximal_groups if not g.issubset(candidate)]
            maximal_groups.append(candidate.copy())

    # dedupe
    seen = set()
    uniq: List[Set[int]] = []
    for g in maximal_groups:
        fg = frozenset(g)
        if fg not in seen:
            seen.add(fg)
            uniq.append(g)
    return uniq


class LatentTree:
    def __init__(self) -> None:
        self._children: Dict[str, Set[str]] = {}
        self._parent: Dict[str, str] = {}

    @property
    def nodes(self) -> List[str]:
        all_nodes = set(self._children.keys())
        for children_set in self._children.values():
            all_nodes.update(children_set)
        return list(all_nodes)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return [
            (parent, child) for parent, kids in self._children.items() for child in kids
        ]

    @property
    def root(self) -> str:
        all_nodes = set(self.nodes)
        children = set(self._parent.keys())
        roots = all_nodes - children
        if len(roots) != 1:
            raise ValueError(
                f"Tree should have exactly one root, found {len(roots)}: {roots}"
            )
        return next(iter(roots))

    def add_edge(self, parent: str, child: str) -> None:
        self._children.setdefault(parent, set()).add(child)
        if child in self._parent and self._parent[child] != parent:
            raise ValueError(f"Node {child} already has parent {self._parent[child]}")
        self._parent[child] = parent

    def add_node(self, node: str) -> None:
        self._children.setdefault(node, set())

    def copy(self) -> "LatentTree":
        new_tree = LatentTree()
        for parent, children in self._children.items():
            new_tree.add_node(parent)
            for child in children:
                new_tree.add_node(child)
                new_tree.add_edge(parent, child)
        return new_tree


_latent_counter = itertools.count(1)


def _new_latent() -> str:
    return f"h{next(_latent_counter)}"


def tree(
    gamma: np.ndarray,
    _nodes: Optional[List[int]] = None,
) -> LatentTree:
    """Algorithm 2 (Tree) with deterministic, stable behavior."""
    if _nodes is None:
        nodes: List[int] = list(range(gamma.shape[0]))
    else:
        nodes = list(_nodes)

    if len(nodes) == 0:
        return LatentTree()

    if len(nodes) == 1:
        tree_obj = LatentTree()
        tree_obj.add_node(str(nodes[0]))
        return tree_obj

    idx_of: Dict[int, int] = {node_v: i for i, node_v in enumerate(nodes)}
    B: Dict[int, Set[int]] = {}

    # Build B_v : line 4 of ALgorithm 2
    for node_v in nodes:
        v_index = idx_of[node_v]
        discrepancies_of_v_th_row = [
            gamma[v_index, idx_of[node_u]] for node_u in nodes if node_u != node_v
        ]
        min_discrepancy_of_v_th_row = min(discrepancies_of_v_th_row)
        B[node_v] = {
            node_u
            for node_u in nodes
            if node_u != node_v
            and gamma[v_index, idx_of[node_u]] <= min_discrepancy_of_v_th_row + EPS
        }

    # Star case
    star_case = all(
        B[node_v] == set(nodes) - {node_v} for node_v in nodes if len(nodes) > 1
    )
    tree_obj = LatentTree()

    if star_case:  # line 5 of Algorithm 2
        root: Optional[int] = None
        # line 6 of Algorithm 2 : if this is true, the tree is a star graph with node_w as the root in the center.
        for node_w in nodes:
            w_index = idx_of[node_w]
            if any(
                _is_zero(gamma[w_index, idx_of[node_u]])
                for node_u in nodes
                if node_u != node_w
            ):
                root = node_w
                break

        if root is not None:
            # Observed root case - line 7 of Algorithm 2 : return a star graph with the observed root.
            root_name = str(root)
            tree_obj.add_node(root_name)
            for node in nodes:
                if node != root:
                    tree_obj.add_node(str(node))
                    tree_obj.add_edge(root_name, str(node))

        else:
            # Latent root case - line 8 of Algorithm 2 : return a star graph with a latent root.
            root_name = _new_latent()
            tree_obj.add_node(root_name)
            for node in nodes:
                tree_obj.add_node(str(node))
                tree_obj.add_edge(root_name, str(node))
        return tree_obj

    # Non-star: line 11 of Algorithm 2
    w_candidates = [node for node in nodes if B[node] != set(nodes) - {node}]
    if not w_candidates:
        raise ValueError("Cannot find suitable w for decomposition")

    # not sure if this is necessary.
    def w_score(node: int) -> Tuple[float, int]:
        min_gamma = min(
            gamma[idx_of[node], idx_of[node_u]] for node_u in nodes if node_u != node
        )
        return (float(min_gamma), node)  # tie-break by node id

    w = min(w_candidates, key=w_score)

    # Decompose - prepare for line 13 and 14 of Algorithm 2
    O1: List[int] = sorted(list(B[w] | {w}))
    O2: List[int] = sorted(list(set(nodes) - B[w]))

    def _get_sub_discrepancy_matrix(sub_nodes: List[int]) -> np.ndarray:
        sub_nodes_indexes = [idx_of[sub_node] for sub_node in sub_nodes]
        sub_gamma_matrix = gamma[np.ix_(sub_nodes_indexes, sub_nodes_indexes)]
        return sub_gamma_matrix

    gamma1 = _get_sub_discrepancy_matrix(O1)
    gamma2 = _get_sub_discrepancy_matrix(O2)

    T1 = tree(gamma1, _nodes=O1)
    T2 = tree(gamma2, _nodes=O2)

    return _get_connected_directed_trees(T1, T2, str(w))


def _get_connected_directed_trees(
    T1: LatentTree, T2: LatentTree, w_str: str
) -> LatentTree:
    """
    Merge T1 into T2 by replacing 'w' in T2 with a new latent 'h' and attaching T1.

    Fix: If w is the root of T2, attach as r1 -> h (not h -> r1).
    """
    h = _new_latent()
    connected_T1_and_T2 = LatentTree()

    parent_of_w: Optional[str] = T2._parent.get(w_str)

    # Copy T2 structure, with w replaced by h : line 15 of Algorithm 2
    substituted_T2_w_by_h = {
        node_in_T2: (h if node_in_T2 == w_str else node_in_T2)
        for node_in_T2 in T2.nodes
    }

    for parent, children in T2._children.items():
        for child in children:
            if child == w_str:
                continue  # skip edges into w
            parent_node_after_substitution = substituted_T2_w_by_h[parent]
            child_node_after_substitution = substituted_T2_w_by_h[child]
            connected_T1_and_T2.add_node(parent_node_after_substitution)
            connected_T1_and_T2.add_node(child_node_after_substitution)
            connected_T1_and_T2.add_edge(
                parent_node_after_substitution, child_node_after_substitution
            )

    # Copy T1 into connected_T1_and_T2
    for parent, children in T1._children.items():
        for child in children:
            connected_T1_and_T2.add_node(parent)
            connected_T1_and_T2.add_node(child)
            connected_T1_and_T2.add_edge(parent, child)

    # Connect
    if parent_of_w is None:
        # w was root of T2 -> attach T1.root -> h   (FIXED orientation)
        connected_T1_and_T2.add_node(h)
        if T1.nodes:
            r1 = T1.root
            connected_T1_and_T2.add_edge(r1, h)
    else:
        assert not T2._children.get(w_str, set()), "Expected w to be a leaf in T2"
        # Connect parent_of_w (mapped) -> T1.root
        parent_node_after_substitution = substituted_T2_w_by_h[parent_of_w]
        if T1.nodes:
            r1 = T1.root
            connected_T1_and_T2.add_node(parent_node_after_substitution)
            connected_T1_and_T2.add_node(r1)
            connected_T1_and_T2.add_edge(parent_node_after_substitution, r1)

    return connected_T1_and_T2


def rename_latent_nodes_by_topology(polytree: PolyDAG) -> PolyDAG:
    """
    Rename latent nodes (h1, h2, ...) according to topological order.
    """

    # Topological sort
    def topo_sort(nodes: List[str], edges: List[Tuple[str, str]]) -> List[str]:
        from collections import deque

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

    nodes = polytree.nodes
    edges = polytree.edges
    latent_nodes = [n for n in nodes if n.startswith("h")]

    if not latent_nodes:
        return polytree

    # Get topological order
    topo_order = topo_sort(nodes, edges)
    latent_in_topo = [n for n in topo_order if n.startswith("h")]

    # Create mapping: old latent name -> new enumerated name
    latent_mapping = {old: f"h{i + 1}" for i, old in enumerate(latent_in_topo)}

    # Rebuild polytree with renamed latent nodes
    new_polytree = PolyDAG()
    for parent, child in edges:
        new_parent = latent_mapping.get(parent, parent)
        new_child = latent_mapping.get(child, child)
        new_polytree.add_edge(new_parent, new_child)

    return new_polytree


class PolyDAG:
    def __init__(self) -> None:
        self.children: Dict[str, Set[str]] = {}
        self.parents: Dict[str, Set[str]] = {}

    def add_node(self, node_v: str) -> None:
        self.children.setdefault(node_v, set())
        self.parents.setdefault(node_v, set())

    def add_edge(self, parent: str, child: str) -> None:
        self.add_node(parent)
        self.add_node(child)
        self.children[parent].add(child)
        self.parents[child].add(parent)

    @property
    def nodes(self) -> List[str]:
        s = set(self.children.keys())
        for children in self.children.values():
            s.update(children)
        return sorted(s)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return [
            (parent, child)
            for parent, children in self.children.items()
            for child in children
        ]


def get_polytree_algo3(gamma: np.ndarray) -> PolyDAG:
    sub_trees = separation(gamma)
    sub_trees = sorted(sub_trees, key=lambda g: (-len(g), sorted(g)))
    polytree = PolyDAG()
    for G in sub_trees:
        sub = sorted(G)
        T = tree(gamma[np.ix_(sub, sub)], _nodes=sub)
        for parent, child in T.edges:
            polytree.add_edge(parent, child)
    polytree = rename_latent_nodes_by_topology(polytree)
    return polytree
