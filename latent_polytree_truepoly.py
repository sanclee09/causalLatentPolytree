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
    w_override: Optional[int] = None,
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

    # line 4 of ALgorithm 2 - Build B_v
    for node_v in nodes:
        v_index = idx_of[node_v]
        discrepancies_of_v_th_row = [
            gamma[v_index, idx_of[node_u]] for node_u in nodes if node_u != node_v
        ]
        if not discrepancies_of_v_th_row:
            B[node_v] = set()
            continue
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
        # Pick observed root if any row has a numerical zero entry
        root: Optional[int] = None
        for node_v in nodes:
            v_index = idx_of[node_v]
            # line 6 of Algorithm 2 : if this is true, the tree is a star graph with node_v as the root in the center.
            if any(
                _is_zero(gamma[v_index, idx_of[node_u]])
                for node_u in nodes
                if node_u != node_v
            ):
                root = node_v
                break

        if root is None:
            # Latent root case - line 8 of Algorithm 2
            root_name = _new_latent()
            tree_obj.add_node(root_name)
            for node_v in nodes:
                tree_obj.add_node(str(node_v))
                tree_obj.add_edge(root_name, str(node_v))
        else:
            # Observed root case - line 7 of Algorithm 2
            root_name = str(root)
            tree_obj.add_node(root_name)
            for node_v in nodes:
                if node_v != root:
                    tree_obj.add_node(str(node_v))
                    tree_obj.add_edge(root_name, str(node_v))
        return tree_obj

    # Non-star: choose w deterministically
    if w_override is not None:
        if w_override not in nodes:
            raise ValueError(f"w_override {w_override} not in current node set")
        w = w_override
    else:
        w_candidates = [
            node_v for node_v in nodes if B[node_v] != set(nodes) - {node_v}
        ]
        if not w_candidates:
            raise ValueError("Cannot find suitable w for decomposition")

        def w_score(node_v: int) -> Tuple[float, int]:
            min_gamma = min(
                gamma[idx_of[node_v], idx_of[node_u]]
                for node_u in nodes
                if node_u != node_v
            )
            return (float(min_gamma), node_v)  # tie-break by node id

        w = min(w_candidates, key=w_score)

    # Decompose
    O1: List[int] = sorted(list(B[w] | {w}))
    O2: List[int] = sorted(list(set(nodes) - B[w]))

    def restrict(nodes_sub: List[int]) -> np.ndarray:
        idx_sub = [idx_of[node_u] for node_u in nodes_sub]
        return gamma[np.ix_(idx_sub, idx_sub)]

    gamma1 = restrict(O1)
    gamma2 = restrict(O2)

    T1 = tree(gamma1, w_override=None, _nodes=O1)
    T2 = tree(gamma2, w_override=None, _nodes=O2)

    return _tree_merger(T1, T2, str(w))


def _tree_merger(T1: LatentTree, T2: LatentTree, w_str: str) -> LatentTree:
    """
    Merge T1 into T2 by replacing 'w' in T2 with a new latent 'h' and attaching T1.

    Fix: If w is the root of T2, attach as r1 -> h (not h -> r1).
    """
    h = _new_latent()
    new_tree = LatentTree()

    parent_of_w: Optional[str] = T2._parent.get(w_str)

    # Copy T2 structure, with w replaced by h
    mapping = {node_u: (h if node_u == w_str else node_u) for node_u in T2.nodes}

    for parent, children in T2._children.items():
        for child in children:
            if child == w_str:
                continue  # skip edges into w
            p_mapped = mapping[parent]
            c_mapped = mapping[child]
            new_tree.add_node(p_mapped)
            new_tree.add_node(c_mapped)
            new_tree.add_edge(p_mapped, c_mapped)

    # Copy T1 into new_tree
    for parent, children in T1._children.items():
        for child in children:
            new_tree.add_node(parent)
            new_tree.add_node(child)
            new_tree.add_edge(parent, child)

    # Connect
    if parent_of_w is None:
        # w was root of T2 -> attach T1.root -> h   (FIXED orientation)
        new_tree.add_node(h)
        if T1.nodes:
            r1 = T1.root
            new_tree.add_edge(r1, h)
    else:
        # Connect parent_of_w (mapped) -> T1.root
        p_mapped = mapping[parent_of_w]
        if T1.nodes:
            r1 = T1.root
            new_tree.add_node(p_mapped)
            new_tree.add_node(r1)
            new_tree.add_edge(p_mapped, r1)

    return new_tree


class PolyDAG:
    def __init__(self) -> None:
        self.children: Dict[str, Set[str]] = {}
        self.parents: Dict[str, Set[str]] = {}

    def add_node(self, node_v: str) -> None:
        self.children.setdefault(node_v, set())
        self.parents.setdefault(node_v, set())

    def add_edge(self, p: str, c: str) -> None:
        self.add_node(p)
        self.add_node(c)
        self.children[p].add(c)
        self.parents[c].add(p)

    @property
    def nodes(self) -> List[str]:
        s = set(self.children.keys())
        for cs in self.children.values():
            s.update(cs)
        return sorted(s)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return [(p, c) for p, cs in self.children.items() for c in cs]


def polytree_true(gamma: np.ndarray) -> PolyDAG:
    groups = separation(gamma)
    groups = sorted(groups, key=lambda g: (-len(g), sorted(g)))
    P = PolyDAG()
    for G in groups:
        sub = sorted(G)
        T = tree(gamma[np.ix_(sub, sub)], _nodes=sub)
        for p, c in T.edges:
            P.add_edge(p, c)
    return P
