from __future__ import annotations
from typing import *
import numpy as np
import itertools

def separation(gamma: np.ndarray) -> List[Set[int]]:
    n = gamma.shape[0]
    all_nodes = set(range(n))
    maximal_groups: List[Set[int]] = []

    for start_node in range(n):
        candidate: Set[int] = {start_node}
        added = True
        while added:
            added = False
            for u in all_nodes:
                if u in candidate:
                    continue
                ok = True
                for w in candidate:
                    if gamma[u, w] < 0 or gamma[w, u] < 0:
                        ok = False
                        break
                if ok:
                    candidate.add(u)
                    added = True

        is_maximal = True
        for existing_group in maximal_groups:
            if candidate.issubset(existing_group):
                is_maximal = False
                break
        if is_maximal:
            maximal_groups = [g for g in maximal_groups if not g.issubset(candidate)]
            maximal_groups.append(candidate.copy())

    return maximal_groups


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
        return [(parent, child) for parent, kids in self._children.items() for child in kids]

    @property
    def root(self) -> str:
        all_nodes = set(self.nodes)
        children = set(self._parent.keys())
        roots = all_nodes - children
        if len(roots) != 1:
            raise ValueError(f"Tree should have exactly one root, found {len(roots)}: {roots}")
        return next(iter(roots))

    def add_edge(self, parent: str, child: str) -> None:
        self._children.setdefault(parent, set()).add(child)
        if child in self._parent and self._parent[child] != parent:
            raise ValueError(f"Node {child} already has parent {self._parent[child]}")
        self._parent[child] = parent

    def add_node(self, node: str) -> None:
        self._children.setdefault(node, set())

    def copy(self) -> 'LatentTree':
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


def tree(gamma: np.ndarray, w_override: Optional[int] = None, _nodes: Optional[List[int]] = None) -> LatentTree:
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

    idx_of: Dict[int, int] = {v: i for i, v in enumerate(nodes)}
    B: Dict[int, Set[int]] = {}

    for v in nodes:
        i_v = idx_of[v]
        other_vals = [gamma[i_v, idx_of[u]] for u in nodes if u != v]
        if not other_vals:
            B[v] = set()
            continue
        min_val = min(other_vals)
        B[v] = {u for u in nodes if u != v and gamma[i_v, idx_of[u]] == min_val}

    star_case = all(B[v] == set(nodes) - {v} for v in nodes if len(nodes) > 1)
    tree_obj = LatentTree()

    if star_case:
        root: Optional[int] = None
        for v in nodes:
            i_v = idx_of[v]
            if any(gamma[i_v, idx_of[u]] == 0 for u in nodes if u != v):
                root = v
                break

        if root is None:
            root_name = _new_latent()
            tree_obj.add_node(root_name)
            for v in nodes:
                tree_obj.add_node(str(v))
                tree_obj.add_edge(root_name, str(v))
        else:
            root_name = str(root)
            tree_obj.add_node(root_name)
            for v in nodes:
                if v != root:
                    tree_obj.add_node(str(v))
                    tree_obj.add_edge(root_name, str(v))
        return tree_obj

    if w_override is not None:
        if w_override not in nodes:
            raise ValueError(f"w_override {w_override} not in current node set")
        w = w_override
    else:
        w_candidates = [v for v in nodes if B[v] != set(nodes) - {v}]
        if not w_candidates:
            raise ValueError("Cannot find suitable w for decomposition")
        w = w_candidates[0]

    O1: List[int] = sorted(list(B[w] | {w}))
    O2: List[int] = sorted(list(set(nodes) - B[w]))

    def restrict(nodes_sub: List[int]) -> np.ndarray:
        idx_sub = [idx_of[u] for u in nodes_sub]
        return gamma[np.ix_(idx_sub, idx_sub)]

    gamma1 = restrict(O1)
    gamma2 = restrict(O2)

    T1 = tree(gamma1, w_override=None, _nodes=O1)
    T2 = tree(gamma2, w_override=None, _nodes=O2)

    return _tree_merger(T1, T2, str(w))


def _tree_merger(T1: LatentTree, T2: LatentTree, w_str: str) -> LatentTree:
    h = _new_latent()
    new_tree = LatentTree()

    parent_of_w: Optional[str] = T2._parent.get(w_str)

    mapping = {u: (h if u == w_str else u) for u in T2.nodes}

    for parent, children in T2._children.items():
        for child in children:
            if child == w_str:
                continue
            p_mapped = mapping[parent]
            c_mapped = mapping[child]
            new_tree.add_node(p_mapped)
            new_tree.add_node(c_mapped)
            new_tree.add_edge(p_mapped, c_mapped)

    for parent, children in T1._children.items():
        for child in children:
            new_tree.add_node(parent)
            new_tree.add_node(child)
            new_tree.add_edge(parent, child)

    if parent_of_w is None:
        new_tree.add_node(h)
        if T1.nodes:
            r1 = T1.root
            new_tree.add_edge(h, r1)
    else:
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

    def add_node(self, v: str) -> None:
        self.children.setdefault(v, set())
        self.parents.setdefault(v, set())

    def add_edge(self, p: str, c: str) -> None:
        self.add_node(p); self.add_node(c)
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


def test_example_7():
    gamma_O = np.array([
        [0, 2, 3, 1, 4],
        [0, 0, -2, 0, 1],
        [1, -3, 0, 1, -3],
        [0, 1, 2, 0, 3],
        [0, 0, -1, 0, 0]
    ])
    print("=== EXAMPLE 7 (true-polytree merge) ===")
    groups = separation(gamma_O)
    print("Groups:", groups)
    for i, G in enumerate(groups):
        sub = sorted(G)
        T = tree(gamma_O[np.ix_(sub, sub)], _nodes=sub)
        print(f"T{i+1}:", T.edges)

    P = polytree_true(gamma_O)
    print("Merged edges:", sorted(P.edges))
    print("Nodes:", P.nodes)
    expected = [('4','1'), ('1','3'), ('3','0')]
    print("Expected chain present:", all(e in P.edges for e in expected))

if __name__ == "__main__":
    test_example_7()
