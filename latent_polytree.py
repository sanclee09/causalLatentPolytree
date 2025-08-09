"""
Implementation of the structure–learning algorithms for latent polytrees.

This module follows the Separation–Tree–Merger pipeline described in
Definition 12, Algorithm 2 (Separation) and Algorithm 3 (Tree) of
Etesami's dissertation on learning latent polytrees【297200969759328†L707-L775】.
Given only a discrepancy matrix between observed variables, the
``separation`` function partitions the indices into sibling groups;
``tree`` recovers a directed tree from a discrepancy matrix, inserting
latent ancestors when necessary; and ``polytree`` merges these directed
trees into a full latent polytree.

The user can provide an optional override for the pivot node ``w`` in
the ``tree`` function via the ``w_override`` parameter (corresponding
to the line “Choose $w$ such that $B_w \neq O \setminus \{w\}$” in the
pseudocode【297200969759328†L737-L753】).  This allows experimentation with different
choices of the decomposition point.

Example usage (demonstration of Example 12 and Example 13 from
Etesami's dissertation):

>>> from latent_polytree import separation, tree, polytree
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
>>> sibling_groups = separation(gamma_ex12)
>>> T = tree(gamma_ex12)
>>> T.edges

The returned object from ``tree`` is a ``LatentTree`` whose ``edges``
attribute returns a list of (parent, child) tuples.  Both observed and
latent nodes appear in the node set.  To learn the full latent
polytree one can call ``polytree(gamma_ex12)``.

For Example 13【424822810520238†screenshot】, simply restrict the discrepancy matrix to the
observed variables of interest and call ``tree`` on the resulting
submatrix.

"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional, Iterable
import itertools
# Note: We avoid using networkx because it is not installed in the runtime.
# Instead we implement a minimal directed tree structure using Python
# dictionaries.
import numpy as np


def separation(gamma: np.ndarray) -> List[Set[int]]:
    """Partition the observed indices into sibling groups based on non‑negative discrepancies.

    This implements Algorithm 2 (Separation) from the dissertation【297200969759328†L713-L729】.  The
    function operates solely on the observed discrepancy matrix and does not require explicit
    node names.  Each row and column of ``gamma`` corresponds to an observed variable, and
    the indices ``0`` through ``n − 1`` are used as implicit node identifiers.

    Parameters
    ----------
    gamma: np.ndarray
        Square matrix where ``gamma[i, j]`` is the discrepancy γ(v_i, v_j).

    Returns
    -------
    List[Set[int]]
        A list of maximal subsets ``O_i`` of integer indices such that all pairwise
        discrepancies within each subset are non‑negative.  These index sets refer to
        the corresponding rows/columns of the input matrix.
    """
    n = gamma.shape[0]
    remaining: Set[int] = set(range(n))
    groups: List[Set[int]] = []
    while remaining:
        # choose an arbitrary index v from the yet unassigned set
        v = next(iter(remaining))
        # candidate group contains v and all indices u such that gamma(u,w) >= 0
        # for all pairs (u,w) in the candidate set
        candidate: Set[int] = set([v])
        # greedily build the maximal set by adding indices that maintain non‑negative discrepancies
        added = True
        while added:
            added = False
            for u in list(remaining):
                if u in candidate:
                    continue
                # check if adding u keeps gamma >= 0 for all pairs in candidate ∪ {u}
                ok = True
                for w in candidate:
                    if gamma[u, w] < 0 or gamma[w, u] < 0:
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


def tree(gamma: np.ndarray, w_override: Optional[int] = None, _nodes: Optional[List[int]] = None) -> LatentTree:
    """Recover a directed tree from a discrepancy matrix on a subset of nodes.

    This implements Algorithm 3 (Tree) from the dissertation【297200969759328†L733-L753】.  The function
    operates solely on the provided discrepancy matrix ``gamma`` and does not require
    explicit node names.  Instead, the indices of ``gamma`` (0 through ``n − 1``) are
    interpreted as observed node identifiers.  Latent nodes are represented by
    synthetic names ``h1``, ``h2``, …

    When the structure is a star graph, the function chooses an observed root
    whenever the minimum discrepancy of some node equals zero; otherwise it creates
    a latent root.  A user may optionally override the choice of the pivot node
    via ``w_override``, which specifies the index of the node ``w`` in the sense of
    Algorithm 3.  The internal parameter ``_nodes`` is used by recursive calls to
    track the mapping between rows/columns of submatrices and the corresponding
    global node indices; end‑users should not provide this argument.

    Parameters
    ----------
    gamma: np.ndarray
        Square matrix of discrepancies for a set of observed nodes.
    w_override: int, optional
        If provided, forces the choice of ``w`` in the non‑trivial case where a
        node ``w`` satisfies ``B_w \neq O \setminus \{w\}``.  The value should
        correspond to an index of the observed node in the original ``gamma``.
    _nodes: list of int, optional
        Internal list of global node indices corresponding to the rows/columns of
        ``gamma``.  This parameter is populated automatically during recursion.

    Returns
    -------
    LatentTree
        A directed tree (with latent nodes as needed) representing the recovered
        structure.  All observed node identifiers are converted to strings of their
        integer indices, and latent nodes are labelled ``h1``, ``h2``, etc.
    """
    # Determine the list of node identifiers corresponding to rows/columns of gamma.
    if _nodes is None:
        # top‑level call: use consecutive indices 0..n-1 as node identifiers
        nodes: List[int] = list(range(gamma.shape[0]))
    else:
        # recursive call: use provided mapping of global indices
        nodes = list(_nodes)

    # Build B_v for each node: the set of nodes attaining the minimal discrepancy
    idx_of: Dict[int, int] = {v: i for i, v in enumerate(nodes)}
    B: Dict[int, Set[int]] = {}
    for v in nodes:
        i_v = idx_of[v]
        # consider discrepancies to other nodes only
        other_vals = [gamma[i_v, idx_of[u]] for u in nodes if u != v]
        min_val = min(other_vals)
        B[v] = {u for u in nodes if u != v and gamma[i_v, idx_of[u]] == min_val}

    # Check if the graph is a star: all B_v equal O\{v}
    star_case = all(B[v] == set(nodes) - {v} for v in nodes)
    tree_obj = LatentTree()
    if star_case:
        # star graph: choose an observed root if some node has minimal discrepancy 0
        root: Optional[int] = None
        for v in nodes:
            i_v = idx_of[v]
            if any(gamma[i_v, idx_of[u]] == 0 for u in nodes if u != v):
                root = v
                break
        if root is None:
            # use a latent root
            root_name = _new_latent()
            tree_obj.add_node(root_name)
            # connect latent root to all observed nodes
            for v in nodes:
                tree_obj.add_node(str(v))
                tree_obj.add_edge(root_name, str(v))
        else:
            # use observed root
            root_name = str(root)
            for v in nodes:
                tree_obj.add_node(str(v))
            for v in nodes:
                if v != root:
                    tree_obj.add_edge(root_name, str(v))
        return tree_obj

    # non‑star case: choose w such that B_w != O\{w}
    if w_override is not None:
        if _nodes is None:
            # top‑level override refers to global indices
            if w_override not in nodes:
                raise ValueError(f"w_override {w_override} not in nodes")
            w = w_override
        else:
            # recursive override refers to global indices; ensure it exists in this subset
            if w_override not in nodes:
                raise ValueError(f"w_override {w_override} not in current node set")
            w = w_override
    else:
        # choose the first node whose B_v differs from O\{v}
        w = next(v for v in nodes if B[v] != set(nodes) - {v})

    # define O1 = B_w ∪ {w} and O2 = O \ B_w  (note: O2 still contains w)
    O1: List[int] = sorted(list(B[w] | {w}))
    O2: List[int] = sorted(list(set(nodes) - B[w]))

    # build submatrices for O1 and O2
    def restrict(nodes_sub: List[int]) -> np.ndarray:
        idx_sub = [idx_of[u] for u in nodes_sub]
        return gamma[np.ix_(idx_sub, idx_sub)]

    gamma1 = restrict(O1)
    gamma2 = restrict(O2)

    # recursive calls on each subset; pass along the global node identifiers
    T1 = tree(gamma1, w_override=None, _nodes=O1)
    T2 = tree(gamma2, w_override=None, _nodes=O2)

    # Following the Tree–Merger operator, replace the occurrence of w in T2
    # by a new latent node h, graft T1 below its parent, and remove w entirely.
    h = _new_latent()
    # Determine parent of w in T2; since w must be a leaf in T2
    parent_of_w: Optional[str] = None
    # Convert w to its string representation, because tree nodes are stored as strings
    w_str = str(w)
    # Determine parent of w (if any) in the existing tree
    if w_str in T2._parent:
        parent_of_w = T2._parent[w_str]

    # Build new_tree without the edge to w, renaming w to h
    new_tree = LatentTree()
    # Build mapping from nodes in T2 to new labels (w_str -> h)
    mapping = {u: (h if u == w_str else u) for u in T2.nodes}
    for parent, children in T2._children.items():
        for child in children:
            if child == w_str:
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
        # If w had no parent in T2, then T2 was a single node (which becomes h).
        # In that case, simply return the new tree containing T1 – there is nothing else to attach.
        return new_tree
    else:
        p_mapped = mapping[parent_of_w]
        r1 = T1.root
        new_tree.add_node(p_mapped)
        new_tree.add_node(r1)
        new_tree.add_edge(p_mapped, r1)
    return new_tree


def polytree(gamma: np.ndarray) -> LatentTree:
    """Recover a minimal latent polytree from a discrepancy matrix.

    This is the full three‑phase algorithm described in Section 5.2 of
    Etesami's dissertation: separate the observed variables into sibling groups,
    learn directed trees on each group, and merge them via the tree‑merger
    operator【297200969759328†L707-L775】.  Unlike the original implementation, this
    version requires only the discrepancy matrix ``gamma``; node identifiers are
    inferred from the matrix indices.

    Parameters
    ----------
    gamma: np.ndarray
        Square discrepancy matrix whose entry ``gamma[i, j]`` represents the
        discrepancy γ(v_i, v_j) between the i‑th and j‑th observed variables.

    Returns
    -------
    LatentTree
        Directed polytree containing both observed and latent nodes.  Observed
        nodes are labelled by the string representation of their zero‑based indices.
    """
    # Partition indices into sibling groups using the separation algorithm
    groups = separation(gamma)
    if not groups:
        raise ValueError("Separation produced no groups from the discrepancy matrix")

    # Learn a tree on the first group
    first_group = sorted(list(groups[0]))
    T = tree(restrict_gamma(gamma, first_group), _nodes=first_group)
    # Maintain set S of indices covered so far and a set of group indices visited
    S: Set[int] = set(first_group)
    visited: Set[int] = {0}
    # Iterate until all groups have been merged
    while len(visited) < len(groups):
        # find an index i whose group intersects S
        found = False
        for i, grp in enumerate(groups):
            if i in visited:
                continue
            if S.intersection(grp):
                found = True
                grp_sorted = sorted(list(grp))
                # learn tree on grp
                T_i = tree(restrict_gamma(gamma, grp_sorted), _nodes=grp_sorted)
                # intersection with S determines where to attach
                common = S.intersection(grp)
                common_sorted = sorted(list(common))
                # learn subtree on common nodes (should have single root)
                T_sub = tree(restrict_gamma(gamma, common_sorted), _nodes=common_sorted)
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


def restrict_gamma(gamma: np.ndarray, subset: Iterable[int]) -> np.ndarray:
    """Return the submatrix of ``gamma`` for a given subset of indices.

    This helper extracts the rows and columns of ``gamma`` corresponding to the
    provided integer indices, preserving the order of ``subset``.  It is used
    internally by :func:`tree` and :func:`polytree` to construct smaller
    discrepancy matrices for recursive calls.

    Parameters
    ----------
    gamma: np.ndarray
        Original discrepancy matrix.
    subset: Iterable[int]
        Iterable of integer indices whose rows/columns should be retained.

    Returns
    -------
    np.ndarray
        A square matrix formed by selecting the rows and columns of ``gamma``
        indexed by ``subset``.
    """
    idx = list(subset)
    return gamma[np.ix_(idx, idx)]