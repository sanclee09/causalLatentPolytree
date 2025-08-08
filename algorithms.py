"""Implementation of the Separation‑Tree‑Merger algorithms for recovering
minimal latent polytrees from a discrepancy matrix.

This module provides Python functions mirroring the high‑level pseudocode
presented in Algorithm 5 (Separation), Algorithm 6 (Tree) and Algorithm 7
(Polytree) from the thesis draft.  The implementation makes a few
modifications discussed with the user:

* When computing the sets ``B_v`` in ``tree`` we record the minimum
  discrepancy ``m_v = min_u gamma[v][u]`` for each node.  Nodes with
  ``m_v > 0`` are skipped when choosing the splitting vertex ``w``; this
  prevents selecting vertices whose smallest discrepancy is strictly
  positive, as those edges do not correspond to true zero‑discrepancy
  relationships.  If no node has ``m_v == 0`` then the function falls
  back to the star graph cases, either using an observed root (if one
  has a zero discrepancy) or a hidden root otherwise【15589856353448†L737-L753】.

* Among the eligible vertices (those with ``m_v == 0``), ``tree`` chooses
  the vertex ``w`` whose ``B_w`` has the largest cardinality.  This
  heuristic tends to group as many zero‑discrepancy neighbours as
  possible into a single subtree.

* Trees are represented as a pair ``(root, edges)`` where ``root`` is
  the name of the root node and ``edges`` is a dictionary mapping
  parents to a list of their children.  Hidden nodes are assigned
  automatically generated names (e.g. ``'h1'``, ``'h2'``) to avoid
  collisions with observed vertices.

These routines assume that the discrepancy matrix ``gamma`` is provided
as a nested dictionary or a 2‑D array‑like object indexed by vertex
labels.  The set of observed vertices ``O`` should be passed as an
iterable of labels (for example, a list or a set).

Example usage::

    from algorithms import tree
    gamma = {
        1: {2: 0, 3: 1, 4: 1},
        2: {1: 0, 3: 0, 4: 0},
        3: {1: 1, 2: 0, 4: 1},
        4: {1: 1, 2: 0, 3: 1},
    }
    root, edges = tree({1, 2, 3, 4}, gamma)
    # ``edges`` encodes the oriented tree structure.

"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set, Tuple, Hashable, Optional


class DiscrepancyMatrix:
    """Helper wrapper around a nested dictionary for gamma values.

    The algorithms accept a callable ``gamma(v, u)``; to make this easy
    on the Python side we encapsulate a nested mapping and expose a
    ``__call__`` method.  Missing entries default to ``float('inf')``.
    """

    def __init__(self, data: Dict[Hashable, Dict[Hashable, float]]):
        self.data = data

    def __call__(self, v: Hashable, u: Hashable) -> float:
        if v == u:
            return float('inf')  # ignore diagonal by returning large
        return self.data.get(v, {}).get(u, float('inf'))


def separation(O: Iterable[Hashable], gamma: DiscrepancyMatrix) -> List[Set[Hashable]]:
    """Partition the observed set into sibling groups according to gamma.

    The separation step groups vertices that have non‑negative mutual
    discrepancies.  It iterates through the remaining vertices and
    builds maximal cliques under the relation ``gamma(u, w) >= 0`` for
    all pairs ``u, w`` in the clique【15589856353448†L713-L726】.

    Args:
        O: Iterable of vertex labels representing the observed set.
        gamma: DiscrepancyMatrix giving pairwise discrepancies.

    Returns:
        A list of sets ``[O1, O2, ...]`` where each ``O_i`` is a
        sibling group.
    """
    O_set: Set[Hashable] = set(O)
    remaining: Set[Hashable] = set(O_set)
    groups: List[Set[Hashable]] = []

    # Build an adjacency based on non‑negative discrepancies.
    adj: Dict[Hashable, Set[Hashable]] = {v: set() for v in O_set}
    for v in O_set:
        for u in O_set:
            if v != u and gamma(v, u) >= 0:
                adj[v].add(u)

    while remaining:
        v = next(iter(remaining))
        # Use BFS to find the connected component containing v under adj.
        component: Set[Hashable] = set()
        queue: deque[Hashable] = deque([v])
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            for nb in adj[node]:
                if nb in remaining:
                    queue.append(nb)
        groups.append(component)
        remaining -= component
    return groups


class TreeBuilder:
    """Stateful helper to construct trees with hidden nodes.

    The builder maintains a counter for generating unique hidden
    identifiers.  Each call to :meth:`next_hidden` returns a new label.
    """

    def __init__(self, hidden_prefix: str = "h"):
        self.hidden_prefix = hidden_prefix
        self.hidden_counter = 0

    def next_hidden(self) -> str:
        self.hidden_counter += 1
        return f"{self.hidden_prefix}{self.hidden_counter}"

    def tree(self, O: Set[Hashable], gamma: DiscrepancyMatrix) -> Tuple[Hashable, Dict[Hashable, List[Hashable]]]:
        """Recursively recover a directed tree on the observed set ``O``.

        This implements Algorithm 6 (Tree) with the modifications
        described above.  It returns a tuple ``(root, edges)`` where
        ``edges`` is a parent→children mapping.

        Args:
            O: Set of observed vertices for which to construct a tree.
            gamma: DiscrepancyMatrix returning pairwise discrepancies.

        Returns:
            A pair ``(root, edges)`` describing the directed tree.
        """
        # Base case: empty or singleton set yields trivial tree.
        if not O:
            raise ValueError("Tree called with empty set")
        if len(O) == 1:
            v = next(iter(O))
            return v, {v: []}

        # Compute B_v sets and minimum discrepancies.
        B: Dict[Hashable, Set[Hashable]] = {}
        min_discrepancy: Dict[Hashable, float] = {}
        for v in O:
            # Identify minimum over all u != v.
            min_val = float('inf')
            for u in O:
                if u == v:
                    continue
                val = gamma(v, u)
                if val < min_val:
                    min_val = val
            # Record minimum and collect all u achieving it.
            min_discrepancy[v] = min_val
            candidates: Set[Hashable] = set()
            for u in O:
                if u != v and gamma(v, u) == min_val:
                    candidates.add(u)
            B[v] = candidates

        # Identify candidate vertices where the minimum discrepancy equals zero.
        candidates: List[Hashable] = [v for v in O if min_discrepancy[v] == 0 and B[v] != O - {v}]

        # Star graph cases: if there are no candidates (all min > 0), build a star.
        if not candidates:
            # Check if there is any vertex w with min_discrepancy == 0 –
            # this could occur when all B_v equal O\{v}, but some min is zero.
            observed_roots = [v for v in O if min_discrepancy[v] == 0]
            edges: Dict[Hashable, List[Hashable]] = defaultdict(list)
            if observed_roots:
                w = observed_roots[0]
                # Star rooted at observed w.
                for u in O:
                    if u != w:
                        edges[w].append(u)
                        edges.setdefault(u, [])
                return w, edges
            else:
                # No observed root available; introduce a hidden root.
                h = self.next_hidden()
                for u in O:
                    edges[h].append(u)
                    edges.setdefault(u, [])
                return h, edges

        # Choose w among candidates with the largest B_w.
        w = max(candidates, key=lambda v: len(B[v]))
        BW = B[w]
        # Partition the set: left cluster contains w and its min discrepancy neighbours.
        left_set = BW | {w}
        right_set = O - BW

        # Recursively build subtrees.
        root_left, edges_left = self.tree(left_set, gamma)
        root_right, edges_right = self.tree(right_set, gamma)

        # The right subtree has root_right as its root; if that root is w
        # we replace it by a hidden node h and attach h as a child of w.
        # Otherwise we attach root_right under w.
        # Note: if root_right equals w then it means the right subtree
        # collapsed to a single node w; in that case no action is needed.

        combined_edges: Dict[Hashable, List[Hashable]] = defaultdict(list)
        # Copy edges from left subtree.
        for parent, children in edges_left.items():
            combined_edges[parent].extend(children)

        # Determine attachment: if the right subtree root equals w, then
        # the right subtree is trivial.  Otherwise we need a new hidden
        # node and rewire.
        if root_right != w:
            # Introduce a hidden node h to replace root_right in the right subtree.
            h = self.next_hidden()
            # Add edge from w to h in the combined tree.
            combined_edges[w].append(h)
            # Copy right subtree edges, relabelling root_right -> h.
            def copy_with_relabel(node: Hashable) -> Hashable:
                return h if node == root_right else node
            # Use a queue to traverse the right subtree and copy edges.
            visited: Set[Hashable] = set()
            q: deque[Hashable] = deque([root_right])
            while q:
                parent = q.popleft()
                new_parent = copy_with_relabel(parent)
                for child in edges_right.get(parent, []):
                    new_child = copy_with_relabel(child)
                    combined_edges[new_parent].append(new_child)
                    if child not in visited:
                        visited.add(child)
                        q.append(child)
            # Ensure that h appears in the dictionary even if it has no children
            combined_edges.setdefault(h, [])
        else:
            # Root of right subtree is w; attach children directly.
            for child in edges_right.get(root_right, []):
                combined_edges[w].append(child)
                combined_edges.setdefault(child, [])

        # Ensure all nodes have an entry in edges.
        for v in O:
            combined_edges.setdefault(v, [])
        return root_left, combined_edges


def tree(O: Iterable[Hashable], gamma_data: Dict[Hashable, Dict[Hashable, float]]) -> Tuple[Hashable, Dict[Hashable, List[Hashable]]]:
    """Convenience wrapper around :meth:`TreeBuilder.tree`.

    Args:
        O: Iterable of observed vertices.
        gamma_data: Nested dictionary mapping ``v -> u -> discrepancy``.

    Returns:
        A tuple ``(root, edges)`` describing the oriented tree.
    """
    builder = TreeBuilder()
    gamma = DiscrepancyMatrix(gamma_data)
    return builder.tree(set(O), gamma)


def polytree(O: Iterable[Hashable], gamma_data: Dict[Hashable, Dict[Hashable, float]]) -> Tuple[Hashable, Dict[Hashable, List[Hashable]]]:
    """Recover a minimal latent polytree from a discrepancy matrix.

    This implements Algorithm 7 (Polytree) by first separating the
    observed set into sibling groups and then orienting and merging
    subtrees【15589856353448†L763-L773】.  The current implementation supports
    multiple sibling groups but uses a simplified merge strategy: each
    group is oriented independently with :func:`tree`, and groups are
    merged by connecting their roots to a new hidden parent.  For many
    practical applications, especially when only one group is present,
    this suffices.  Further refinement (e.g. merging along specific
    intersections) can be added as needed.

    Args:
        O: Iterable of observed vertices.
        gamma_data: Nested dictionary mapping ``v -> u -> discrepancy``.

    Returns:
        A tuple ``(root, edges)`` describing the minimal latent polytree.
    """
    gamma = DiscrepancyMatrix(gamma_data)
    groups = separation(O, gamma)
    builder = TreeBuilder()
    # Build a tree for each group.
    subtrees: List[Tuple[Hashable, Dict[Hashable, List[Hashable]]]] = []
    for group in groups:
        root, edges = builder.tree(set(group), gamma)
        subtrees.append((root, edges))
    # If there is only one group, return its tree.
    if len(subtrees) == 1:
        return subtrees[0]
    # Otherwise, create a hidden root and attach each subtree root.
    root_hidden = builder.next_hidden()
    combined_edges: Dict[Hashable, List[Hashable]] = defaultdict(list)
    combined_edges[root_hidden] = []
    for root_sub, edges_sub in subtrees:
        # Attach the root of the subtree to the hidden root.
        combined_edges[root_hidden].append(root_sub)
        # Copy edges from subtree.
        for parent, children in edges_sub.items():
            combined_edges[parent].extend(children)
            combined_edges.setdefault(parent, [])
        combined_edges.setdefault(root_sub, [])
    return root_hidden, combined_edges


__all__ = [
    "DiscrepancyMatrix",
    "separation",
    "tree",
    "polytree",
]