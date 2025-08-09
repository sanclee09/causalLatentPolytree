"""
Implementation of the structure–learning algorithms for latent polytrees.

This module follows the Separation–Tree–Merger pipeline described in
Definition 12, Algorithm 2 (Separation) and Algorithm 3 (Tree) of
Etesami's dissertation on learning latent polytrees【297200969759328†L707-L775】.
Given a discrepancy matrix between observed variables, the
`separation` function partitions nodes into sibling groups; `tree`
recovers a directed tree on a subset of nodes, inserting latent
ancestors when necessary; and `polytree` merges these directed trees
into a full latent polytree.

The user can provide an optional override for the pivot node `w` in
the `tree` function (corresponding to the line “Choose $w$ such that
$B_w \neq O \setminus \{w\}$” in the pseudocode【297200969759328†L737-L753】).  This allows
experimentation with different choices of the decomposition point.

Example usage (demonstration of Example 12 and Example 13 from
Etesami's dissertation):

>>> from latent_polytree import separation, tree
>>> import numpy as np
>>> # discrepancy matrix Γ_V from Example 12【572525137102189†screenshot】
>>> gamma_ex12 = np.array([
...     [0, 2, 3, 1, 3, 4],
...     [0, 0, -2, 0, -1, 1],
...     [1, -3, 0, 1, 1, -3],
...     [0, 1, 2, 0, 2, 3],
...     [0, -1, 0, 0, 0, -2],
...     [0, 0, -1, 0, -1, 0],
... ])
>>> nodes = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
>>> sibling_groups = separation(nodes, gamma_ex12)
>>> T = tree(nodes, gamma_ex12)
>>> T.edges

The returned object from `tree` is a ``LatentTree`` whose ``edges``
attribute returns a list of (parent, child) tuples.  Both observed and
latent nodes appear in the node set.

For Example 13【424822810520238†screenshot】, simply restrict the discrepancy matrix and call
`tree` on the observed node set.

"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional, Iterable
import itertools
# Note: We avoid using networkx because it is not installed in the runtime.
# Instead we implement a minimal directed tree structure using Python
# dictionaries.
import numpy as np


def separation(nodes: List[str], gamma: np.ndarray) -> List[Set[str]]:
    """Partition nodes into sibling groups based on non‑negative discrepancies.

    This implements Algorithm 2 (Separation) from the dissertation【297200969759328†L713-L729】.

    Parameters
    ----------
    nodes: list of str
        Names of the observed variables.  The index of each node in
        ``gamma`` must correspond to its position in this list.
    gamma: np.ndarray
        Square matrix where ``gamma[i, j]`` is the discrepancy γ(v_i, v_j).

    Returns
    -------
    List[Set[str]]
        A list of maximal subsets ``O_i`` such that all pairwise
        discrepancies within each subset are non‑negative.
    """
    n = len(nodes)
    # Map node name to index for convenience
    idx_of = {v: i for i, v in enumerate(nodes)}
    remaining: Set[str] = set(nodes)
    groups: List[Set[str]] = []
    while remaining:
        # choose an arbitrary node v from the yet unassigned set
        v = next(iter(remaining))
        # candidate group contains v and all nodes u such that gamma(u,w) >= 0
        # for all pairs (u,w) in the candidate set
        candidate: Set[str] = set([v])
        # iterate adding nodes that maintain non‑negative gamma with all in candidate
        # we greedily build the maximal set
        added = True
        while added:
            added = False
            for u in list(remaining):
                if u in candidate:
                    continue
                # check if adding u keeps gamma >= 0 for all pairs in candidate ∪ {u}
                ok = True
                for w in candidate:
                    i_u = idx_of[u]
                    i_w = idx_of[w]
                    if gamma[i_u, i_w] < 0 or gamma[i_w, i_u] < 0:
                        ok = False
                        break
                if ok:
                    candidate.add(u)
                    added = True
        groups.append(candidate.copy())
        remaining -= candidate
    return groups


class LatentTree:
    """A simple directed tree structure with support for latent nodes.

    The tree is represented as an adjacency list.  Each node has at
    most one parent (except the root, which has none), and cycles are
    not permitted.
    """

    def __init__(self) -> None:
        # adjacency: parent -> set of children
        self._children: Dict[str, Set[str]] = {}
        # store parents to facilitate root detection
        self._parent: Dict[str, str] = {}

    @property
    def nodes(self) -> List[str]:
        # union of keys and all children
        return list(set(self._children.keys()).union(*self._children.values()) if self._children else set())

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return [(parent, child) for parent, kids in self._children.items() for child in kids]

    @property
    def root(self) -> str:
        # root is a node that is not anyone's child
        all_nodes = set(self.nodes)
        children = set(self._parent.keys())
        roots = all_nodes - children
        if len(roots) != 1:
            raise ValueError("Tree should have exactly one root, found %d" % len(roots))
        return next(iter(roots))

    def add_edge(self, parent: str, child: str) -> None:
        # ensure adjacency lists
        self._children.setdefault(parent, set()).add(child)
        # record parent
        self._parent[child] = parent

    def add_node(self, node: str) -> None:
        # ensure node exists in adjacency dictionary
        self._children.setdefault(node, set())

    def merge(self, other: 'LatentTree', attach_at: str, new_child: str) -> 'LatentTree':
        """Attach another tree below ``attach_at`` using a new child label.

        .. warning::

           This method was originally designed to graft ``other`` beneath
           the node ``attach_at`` by simply connecting ``attach_at`` to a
           fresh label ``new_child`` and inserting all edges of ``other``
           with its root renamed to ``new_child``.  It is retained here
           for backwards compatibility but is no longer used in the core
           ``tree`` algorithm.  See ``_merge_into_leaf`` within
           :func:`tree` for the correct tree–merger operator.
        """
        # determine mapping from other.root to new_child
        root_other = other.root
        for parent, children in other._children.items():
            parent_mapped = new_child if parent == root_other else parent
            self.add_node(parent_mapped)
            for child in children:
                child_mapped = new_child if child == root_other else child
                self.add_node(child_mapped)
                self.add_edge(parent_mapped, child_mapped)
        self.add_node(attach_at)
        self.add_node(new_child)
        self.add_edge(attach_at, new_child)
        return self


_latent_counter = itertools.count(1)


def _new_latent() -> str:
    """Generate a fresh latent node label."""
    return f"h{next(_latent_counter)}"


def tree(nodes: List[str], gamma: np.ndarray, w_override: Optional[str] = None) -> LatentTree:
    """Recover a directed tree from a discrepancy matrix on a subset of nodes.

    This implements Algorithm 3 (Tree) from the dissertation【297200969759328†L733-L753】.  A latent
    node is represented by a synthetic name ``h1``, ``h2``, …  When the
    structure is a star graph, the function chooses an observed root
    whenever the minimum discrepancy of some node equals zero; otherwise
    it creates a latent root.

    Parameters
    ----------
    nodes: list of str
        The observed node names corresponding to the indices of ``gamma``.
    gamma: np.ndarray
        Square matrix of discrepancies for ``nodes``.
    w_override: str, optional
        If provided, forces the choice of ``w`` in the non‑trivial case
        where a node ``w`` satisfies ``B_w \neq O \setminus \{w\}``.  This
        allows exploration of whether different choices yield the same
        output, as suggested by the user.

    Returns
    -------
    LatentTree
        A directed tree (with latent nodes as needed) representing the
        recovered structure.
    """
    n = len(nodes)
    # build B_v for each node: indices of minimal discrepancies
    idx_of = {v: i for i, v in enumerate(nodes)}
    B: Dict[str, Set[str]] = {}
    for v in nodes:
        i_v = idx_of[v]
        # consider discrepancies to other nodes only
        other_vals = [gamma[i_v, idx_of[u]] for u in nodes if u != v]
        min_val = min(other_vals)
        B[v] = {u for u in nodes if u != v and gamma[i_v, idx_of[u]] == min_val}
    # check if all B_v equal O\{v}
    star_case = all(B[v] == set(nodes) - {v} for v in nodes)
    tree_obj = LatentTree()
    if star_case:
        # star graph: choose an observed root if some node has min discrepancy 0
        root: Optional[str] = None
        for v in nodes:
            i_v = idx_of[v]
            if any(gamma[i_v, idx_of[u]] == 0 for u in nodes if u != v):
                root = v
                break
        if root is None:
            root = _new_latent()
            # latent root with edges to all observed nodes
            tree_obj.add_node(root)
            for v in nodes:
                tree_obj.add_node(v)
                tree_obj.add_edge(root, v)
        else:
            # observed root
            for v in nodes:
                tree_obj.add_node(v)
            for v in nodes:
                if v != root:
                    tree_obj.add_edge(root, v)
        return tree_obj
    # non‑star case: choose w such that B_w != O\{w}
    if w_override is not None:
        if w_override not in nodes:
            raise ValueError(f"w_override {w_override} not in nodes")
        w = w_override
    else:
        w = next(v for v in nodes if B[v] != set(nodes) - {v})
    # define O1 = B_w ∪ {w}
    O1 = sorted(list(B[w] | {w}))
    # define O2 = O \ B_w  (note: O2 still contains w)
    O2 = sorted(list(set(nodes) - B[w]))
    # build submatrices for O1 and O2
    def restrict(nodes_sub: List[str]) -> np.ndarray:
        idx_sub = [idx_of[u] for u in nodes_sub]
        return gamma[np.ix_(idx_sub, idx_sub)]
    gamma1 = restrict(O1)
    gamma2 = restrict(O2)
    # recursive calls
    T1 = tree(O1, gamma1, w_override=None)
    T2 = tree(O2, gamma2, w_override=None)
    # In the original Tree algorithm, we replace the occurrence of w in T2
    # by a new latent node h, then merge T1 into T2 at that leaf.  To
    # implement the tree–merger operator correctly (Definition 14), we
    # graft T1 below the parent of h and remove h entirely.  See
    # explanation in the accompanying thesis draft【720472738237394†L733-L754】.
    # Create a fresh latent label to replace w in T2
    h = _new_latent()
    # Determine parent of w in T2; since w must be a leaf in T2
    parent_of_w: Optional[str] = None
    # Copy structure of T2, renaming w to h, but skip the edge to w
    new_tree = LatentTree()
    # Build mapping from nodes in T2 to new labels (w -> h)
    # Determine parent of w (if any)
    if w in T2._parent:
        parent_of_w = T2._parent[w]
    # Build T2 without the edge to w, renaming w to h
    mapping = {u: (h if u == w else u) for u in T2.nodes}
    for parent, children in T2._children.items():
        for child in children:
            if child == w:
                # skip the edge (parent -> w)
                continue
            # map endpoints
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
    # Now connect parent_of_w (renamed) to root of T1
    if parent_of_w is None:
        # If w had no parent in T2, then T2 was a single node (h).  In
        # that case, simply return T1 – there is nothing else to attach.
        return new_tree
    else:
        p_mapped = mapping[parent_of_w]
        r1 = T1.root
        new_tree.add_node(p_mapped)
        new_tree.add_node(r1)
        new_tree.add_edge(p_mapped, r1)
    return new_tree


def polytree(nodes: List[str], gamma: np.ndarray) -> LatentTree:
    """Recover a minimal latent polytree from a discrepancy matrix.

    This is the full three–phase algorithm described in Section 5.2 of
    Etesami's dissertation: separate nodes into sibling groups, learn
    directed trees on each group, and merge them via the tree–merger
    operator【297200969759328†L707-L775】.

    Parameters
    ----------
    nodes: list of str
        Names of the observed nodes.
    gamma: np.ndarray
        Discrepancy matrix for ``nodes``.

    Returns
    -------
    LatentTree
        Directed polytree containing observed and latent nodes.
    """
    groups = separation(nodes, gamma)
    # learn a tree on the first group
    T = tree(sorted(list(groups[0])), restrict_gamma(gamma, nodes, groups[0]))
    # maintain set S of nodes covered so far and indices visited
    S = set(groups[0])
    visited = {0}
    # iterate until all groups have been merged
    while len(visited) < len(groups):
        # find an index i whose group intersects S
        found = False
        for i, grp in enumerate(groups):
            if i in visited:
                continue
            if S.intersection(grp):
                found = True
                # learn tree on grp
                T_i = tree(sorted(list(grp)), restrict_gamma(gamma, nodes, grp))
                # intersection with S determines where to attach
                common = S.intersection(grp)
                # learn subtree on common nodes (should have single root)
                T_sub = tree(sorted(list(common)), restrict_gamma(gamma, nodes, common))
                attach_at = T_sub.root
                # new latent label for the root of T_i when merging
                h_new = _new_latent()
                # attach T_i below attach_at, renaming its root to h_new
                T.merge(T_i, attach_at=attach_at, new_child=h_new)
                # update S and visited
                S.update(grp)
                visited.add(i)
                break
        if not found:
            # should not happen – means groups are disjoint
            raise ValueError("Failed to merge all groups – groups may be disconnected")
    return T


def restrict_gamma(gamma: np.ndarray, nodes: List[str], subset: Iterable[str]) -> np.ndarray:
    """Return the submatrix of ``gamma`` for a given subset of node names."""
    idx_of = {v: i for i, v in enumerate(nodes)}
    idx = [idx_of[v] for v in subset]
    return gamma[np.ix_(idx, idx)]