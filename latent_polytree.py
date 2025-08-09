"""
Implementation of latent polytree structure learning algorithms.
Based on Etesami, Kiyavash & Coleman (2016).
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional
import itertools
import numpy as np


def separation(gamma: np.ndarray) -> List[Set[int]]:
    """Find all maximal subsets with non-negative pairwise discrepancies."""
    n = gamma.shape[0]
    all_nodes = set(range(n))
    maximal_groups: List[Set[int]] = []

    for start_node in range(n):
        candidate: Set[int] = {start_node}

        # Greedily add nodes that maintain non-negative discrepancies
        added = True
        while added:
            added = False
            for u in all_nodes:
                if u in candidate:
                    continue

                # Check if adding u keeps gamma >= 0 for all pairs
                ok = True
                for w in candidate:
                    if gamma[u, w] < 0 or gamma[w, u] < 0:
                        ok = False
                        break

                if ok:
                    candidate.add(u)
                    added = True

        # Check if this is a new maximal group
        is_maximal = True
        for existing_group in maximal_groups:
            if candidate.issubset(existing_group):
                is_maximal = False
                break

        # Remove any existing groups that are subsets of the new candidate
        if is_maximal:
            maximal_groups = [g for g in maximal_groups if not g.issubset(candidate)]
            maximal_groups.append(candidate.copy())

    return maximal_groups


class LatentTree:
    """A simple directed tree structure with support for latent nodes."""

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
    """Recover a directed tree from a discrepancy matrix."""

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

    # Build B_v for each node: the set of nodes attaining the minimal discrepancy
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

    # Check if the graph is a star
    star_case = all(B[v] == set(nodes) - {v} for v in nodes if len(nodes) > 1)
    tree_obj = LatentTree()

    if star_case:
        # Star graph case
        root: Optional[int] = None
        for v in nodes:
            i_v = idx_of[v]
            if any(gamma[i_v, idx_of[u]] == 0 for u in nodes if u != v):
                root = v
                break

        if root is None:
            # Use latent root
            root_name = _new_latent()
            tree_obj.add_node(root_name)
            for v in nodes:
                tree_obj.add_node(str(v))
                tree_obj.add_edge(root_name, str(v))
        else:
            # Use observed root
            root_name = str(root)
            tree_obj.add_node(root_name)
            for v in nodes:
                if v != root:
                    tree_obj.add_node(str(v))
                    tree_obj.add_edge(root_name, str(v))
        return tree_obj

    # Non-star case: recursive decomposition
    if w_override is not None:
        if w_override not in nodes:
            raise ValueError(f"w_override {w_override} not in current node set")
        w = w_override
    else:
        w_candidates = [v for v in nodes if B[v] != set(nodes) - {v}]
        if not w_candidates:
            raise ValueError("Cannot find suitable w for decomposition")
        w = w_candidates[0]

    # Define O1 = B_w ∪ {w} and O2 = O \ B_w
    O1: List[int] = sorted(list(B[w] | {w}))
    O2: List[int] = sorted(list(set(nodes) - B[w]))

    # Build submatrices
    def restrict(nodes_sub: List[int]) -> np.ndarray:
        idx_sub = [idx_of[u] for u in nodes_sub]
        return gamma[np.ix_(idx_sub, idx_sub)]

    gamma1 = restrict(O1)
    gamma2 = restrict(O2)

    # Recursive calls
    T1 = tree(gamma1, w_override=None, _nodes=O1)
    T2 = tree(gamma2, w_override=None, _nodes=O2)

    # Tree merger operation
    return _tree_merger(T1, T2, str(w))


def _tree_merger(T1: LatentTree, T2: LatentTree, w_str: str) -> LatentTree:
    """Implement the tree merger operator T1 ⊕ T2(h)."""
    h = _new_latent()
    new_tree = LatentTree()

    # Determine parent of w in T2
    parent_of_w: Optional[str] = T2._parent.get(w_str)

    # Copy T2 structure, replacing w_str with h
    mapping = {u: (h if u == w_str else u) for u in T2.nodes}

    for parent, children in T2._children.items():
        for child in children:
            if child == w_str:
                continue  # Skip edge to w
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

    # Connect the trees
    if parent_of_w is None:
        # w was root of T2
        new_tree.add_node(h)
        if T1.nodes:
            r1 = T1.root
            new_tree.add_edge(h, r1)
    else:
        # Connect parent of w to root of T1
        p_mapped = mapping[parent_of_w]
        if T1.nodes:
            r1 = T1.root
            new_tree.add_node(p_mapped)
            new_tree.add_node(r1)
            new_tree.add_edge(p_mapped, r1)

    return new_tree


def polytree(gamma: np.ndarray) -> LatentTree:
    """Recover a minimal latent polytree from a discrepancy matrix."""

    # Phase 1: Separation
    sibling_groups = separation(gamma)

    if not sibling_groups:
        return LatentTree()

    if len(sibling_groups) == 1:
        group_nodes = list(sibling_groups[0])
        if len(group_nodes) == 0:
            return LatentTree()

        idx_mapping = {node: i for i, node in enumerate(group_nodes)}
        gamma_sub = np.zeros((len(group_nodes), len(group_nodes)))
        for i, node_i in enumerate(group_nodes):
            for j, node_j in enumerate(group_nodes):
                gamma_sub[i, j] = gamma[node_i, node_j]

        return tree(gamma_sub, _nodes=group_nodes)

    # Phase 2 & 3: Learn trees for each group and merge properly

    # Start with first group
    first_group = list(sibling_groups[0])
    gamma_sub = gamma[np.ix_(first_group, first_group)]
    result_tree = tree(gamma_sub, _nodes=first_group)
    processed_nodes = set(first_group)
    processed_groups = {0}

    # Iteratively merge remaining groups
    while len(processed_groups) < len(sibling_groups):
        # Find next group that intersects with processed nodes
        next_group_idx = None
        for i, group in enumerate(sibling_groups):
            if i not in processed_groups and group & processed_nodes:
                next_group_idx = i
                break

        if next_group_idx is None:
            # No intersecting group found, pick any remaining group
            for i, group in enumerate(sibling_groups):
                if i not in processed_groups:
                    next_group_idx = i
                    break

        if next_group_idx is None:
            break

        next_group = list(sibling_groups[next_group_idx])

        # Create tree for intersection (shared nodes)
        intersection = processed_nodes & set(next_group)

        # Create tree for the new group
        gamma_group = gamma[np.ix_(next_group, next_group)]
        T_i = tree(gamma_group, _nodes=next_group)

        # **KEY FIX**: Proper tree merger for polytree
        # We need to merge result_tree and T_i at their intersection
        if intersection:
            intersection_list = list(intersection)
            print(f"\nMerging trees at intersection: {intersection_list}")
            print(f"Tree 1 edges: {result_tree.edges}")
            print(f"Tree 2 edges: {T_i.edges}")

            # Merge the polytrees properly using the shared subtree
            result_tree = _merge_polytrees_at_intersection(result_tree, T_i, intersection_list)

            print(f"Merged result: {result_tree.edges}")
        else:
            # No intersection - this creates multiple roots (which is correct for polytrees)
            result_tree = _combine_disjoint_trees(result_tree, T_i)

        processed_nodes.update(next_group)
        processed_groups.add(next_group_idx)

    return result_tree


def _merge_polytrees_at_intersection(T1: LatentTree, T2: LatentTree, shared_nodes: List[int]) -> LatentTree:
    """Merge two trees at their intersection according to Algorithm 3."""

    # The shared nodes form a common subtree T_sub
    # We need to implement: T1 ∘ T2 | T_sub

    merged_tree = LatentTree()

    # Copy all edges from T1
    for parent, child in T1.edges:
        merged_tree.add_node(parent)
        merged_tree.add_node(child)
        merged_tree.add_edge(parent, child)

    # Copy edges from T2, but handle the intersection carefully
    for parent, child in T2.edges:
        merged_tree.add_node(parent)
        merged_tree.add_node(child)

        # Check if this edge already exists (part of common subtree)
        if (parent, child) not in T1.edges:
            try:
                merged_tree.add_edge(parent, child)
            except ValueError:
                # This means there's a conflict - child already has a different parent
                # This should not happen if the trees are properly formed
                print(f"Warning: Edge conflict ({parent}, {child})")
                pass

    return merged_tree


def _combine_disjoint_trees(T1: LatentTree, T2: LatentTree) -> LatentTree:
    """Combine two disjoint trees (multiple roots case)."""
    combined = T1.copy()

    # Add all nodes and edges from T2
    for node in T2.nodes:
        combined.add_node(node)

    for parent, child in T2.edges:
        combined.add_edge(parent, child)

    return combined


# Test function
def test_example_7():
    """Test with Example 7 from paper 1."""
    gamma_O = np.array([
        [0, 2, 3, 1, 4],
        [0, 0, -2, 0, 1],
        [1, -3, 0, 1, -3],
        [0, 1, 2, 0, 3],
        [0, 0, -1, 0, 0]
    ])

    print("=== EXAMPLE 7 FROM PAPER 1 ===")
    print("Ground truth: v6 → v2 → v4 → v1, v4 ← v5 → v3 (v5 latent)")

    print("\n--- SEPARATION ---")
    groups = separation(gamma_O)
    print(f"Groups: {groups}")

    print("\n--- INDIVIDUAL TREES ---")
    for i, group in enumerate(groups):
        group_list = list(group)
        gamma_sub = gamma_O[np.ix_(group_list, group_list)]
        T = tree(gamma_sub, _nodes=group_list)
        print(f"Tree {i+1} (nodes {group_list}): {T.edges}")

    print("\n--- POLYTREE RESULT ---")
    result = polytree(gamma_O)
    print(f"Edges: {result.edges}")
    print(f"Nodes: {result.nodes}")

    # Check if we have the expected structure
    # Should have: v6→v2→v4→v1 and h→v3, h→v4 (where h is latent)
    expected_edges = [
        ('4', '1'),  # v6 → v2
        ('1', '3'),  # v2 → v4
        ('3', '0'),  # v4 → v1
    ]

    print(f"\nExpected chain v6→v2→v4→v1: {all(edge in result.edges for edge in expected_edges)}")

    # Check for latent connections to v3 and v4
    latent_nodes = [node for node in result.nodes if node.startswith('h')]
    print(f"Latent nodes: {latent_nodes}")

    for h in latent_nodes:
        connected_to = [child for parent, child in result.edges if parent == h]
        print(f"{h} connects to: {connected_to}")


if __name__ == "__main__":
    test_example_7()