from __future__ import annotations
from typing import Dict, Tuple, List, Any, Optional, Set
import random
from collections import defaultdict, deque

import numpy as np

from learn_with_hidden import observed_gamma_from_params

# ---------- Prüfer utilities ----------


def print_discrepancy_comparison(Gamma_pop, Gamma_finite, observed_nodes, label=""):
    """Print side-by-side comparison of population and finite-sample discrepancy matrices."""
    print(f"\n  {label}Discrepancy Matrix Comparison:")
    print(f"  Observed nodes: {observed_nodes}")
    print(f"\n  Population Γ:")
    for i, row in enumerate(Gamma_pop):
        print(f"    {observed_nodes[i]}: " + " ".join(f"{val:8.4f}" for val in row))

    print(f"\n  Finite-sample Γ:")
    for i, row in enumerate(Gamma_finite):
        print(f"    {observed_nodes[i]}: " + " ".join(f"{val:8.4f}" for val in row))

    print(f"\n  Difference (|Finite - Population|):")
    diff = np.abs(Gamma_finite - Gamma_pop)
    for i, row in enumerate(diff):
        print(f"    {observed_nodes[i]}: " + " ".join(f"{val:8.4f}" for val in row))
    print(f"  Max difference: {np.max(diff):.6f}")


def random_pruefer_sequence(n: int, rng: random.Random) -> List[int]:
    """Draw a random Prüfer sequence of length n-2 with labels 1..n."""
    return [rng.randint(1, n) for _ in range(n - 2)]


def pruefer_to_tree(seq: List[int]) -> List[Tuple[int, int]]:
    """
    Decode a Prüfer sequence (labels 1..n) to an undirected tree edge list.
    Returns edges as pairs of ints in [1..n].
    """
    n = len(seq) + 2
    degree = [1] * (n + 1)  # degree[0] unused
    for x in seq:
        degree[x] += 1
    # min-heap of leaves
    import heapq

    leaves = [i for i in range(1, n + 1) if degree[i] == 1]
    heapq.heapify(leaves)

    edges: List[Tuple[int, int]] = []
    for x in seq:
        u = heapq.heappop(leaves)  # smallest label leaf
        edges.append((u, x))
        degree[u] -= 1
        degree[x] -= 1
        if degree[x] == 1:
            heapq.heappush(leaves, x)
    # last two leaves
    u = heapq.heappop(leaves)
    v = heapq.heappop(leaves)
    edges.append((u, v))
    return edges


# ---------- Orientation ----------


def orient_tree_from_root(
    edges_undirected: List[Tuple[int, int]],
    root: int,
) -> List[Tuple[int, int]]:
    """Orient the undirected tree by directing all edges away from the chosen root."""
    nodes = sorted({u for e in edges_undirected for u in e})
    adj = {u: set() for u in nodes}
    for u, v in edges_undirected:
        adj[u].add(v)
        adj[v].add(u)

    parent = {root: None}
    order = [root]
    q = [root]
    while q:
        x = q.pop(0)
        for y in adj[x]:
            if y not in parent:
                parent[y] = x
                order.append(y)
                q.append(y)

    directed: List[Tuple[int, int]] = []
    for v in order:
        p = parent[v]
        if p is not None:
            directed.append((p, v))
    return directed


def orient_tree_random_topo(
    edges_undirected: List[Tuple[int, int]],
    rng: random.Random,
    force_hidden_root: bool = False,
) -> List[Tuple[int, int]]:
    """
    Orient the undirected tree into a directed tree (each node has <= 1 parent).
    Pick a root and direct edges away from the root (BFS orientation).

    If force_hidden_root=True, choose the root among nodes with undirected degree >= 2
    so the root has at least two children after orientation (guaranteeing at least one
    latent node under branching-based rules).
    """
    nodes = sorted({u for e in edges_undirected for u in e})
    # build undirected adjacency + degrees
    adj = {u: set() for u in nodes}
    for u, v in edges_undirected:
        adj[u].add(v)
        adj[v].add(u)

    if force_hidden_root:
        candidates = [u for u in nodes if len(adj[u]) >= 2]
        root = rng.choice(candidates) if candidates else rng.choice(nodes)
    else:
        root = rng.choice(nodes)

    # BFS orientation away from root
    parent = {root: None}
    order = [root]
    q = [root]
    while q:
        x = q.pop(0)
        for y in adj[x]:
            if y not in parent:
                parent[y] = x
                order.append(y)
                q.append(y)

    directed: List[Tuple[int, int]] = []
    for v in order:
        p = parent[v]
        if p is not None:
            directed.append((p, v))
    return directed


# ---------- Hidden/Observed rules ----------


def outdeg_map(edges_dir: List[Tuple[int, int]]) -> Dict[int, int]:
    nodes = {x for e in edges_dir for x in e}
    od = {u: 0 for u in nodes}
    for u, v in edges_dir:
        od[u] += 1
    return od


def branching_hidden_nodes(edges_dir: List[Tuple[int, int]]) -> Set[str]:
    """New rule: any node with out-degree >= 2 is hidden (learnable)."""
    od = outdeg_map(edges_dir)
    return {f"v{u}" for u, d in od.items() if d >= 2}


# ---------- Weighting & parameters ----------


def random_weights(
    edges_dir: List[Tuple[int, int]],
    rng: random.Random,
    low: float = -1.0,
    high: float = 1.0,
    avoid_small: float = 0.8,
) -> Dict[Tuple[str, str], float]:
    """Assign random edge weights in [low, high], avoiding magnitudes < avoid_small. Labels renamed to ('v{i}', 'v{j}')."""
    weights: Dict[Tuple[str, str], float] = {}
    for u, v in edges_dir:
        while True:
            w = rng.uniform(low, high)
            if abs(w) >= avoid_small:
                break
        weights[(f"v{u}", f"v{v}")] = w
    return weights


def unit_sigmas_kappas(nodes: List[int]):
    sigmas = {f"v{i}": 1.0 for i in nodes}
    kappas = {f"v{i}": 1.0 for i in nodes}
    return sigmas, kappas


def sample_sigmas_kappas(
    nodes: List[int],
    rng: random.Random,
    family: str = "gamma",  # "exp" | "gamma" | "lognormal"
    gamma_k_range: Tuple[float, float] = (1.2, 9.0),  # shape range if family="gamma"
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Return per-node standard deviations (sigmas) and third cumulants (kappas)
    drawn from a non-Gaussian noise family. We standardize to Var=1 so that
    edge-weight scale carries the signal, while κ₃ controls skew/asymmetry.

    families:
      - "exp"  : centered exponential (standardized) => σ=1, |κ₃|=2
      - "gamma": centered gamma with random shape k  => σ=1, |κ₃|=2/√k
      - "lognormal": centered lognormal (standardized) => σ=1, |κ₃| depends on μ, s (heavy tails!)
    """
    sigmas: Dict[str, float] = {}
    kappas: Dict[str, float] = {}

    for i in nodes:
        name = f"v{i}"

        if family == "exp":
            # Y = Exp(λ) - 1/λ has Var=1/λ² and κ₃=2/λ³. After standardizing X=λY:
            # Var(X)=1 and κ₃(X)=2. Randomize sign to diversify.
            sigmas[name] = 1.0
            kappa_mag = 2.0
            kappas[name] = kappa_mag if rng.random() < 0.5 else -kappa_mag

        elif family == "gamma":
            # Centered Gamma(k,θ) with Var=kθ², κ₃=2kθ³. Standardize to Var=1:
            # choose shape k in range, set θ=1/√k -> Var=1, κ₃=2/√k
            k = rng.uniform(*gamma_k_range)
            sigmas[name] = 1.0
            kappa_mag = 2.0 / (k**0.5)
            kappas[name] = kappa_mag if rng.random() < 0.5 else -kappa_mag

        elif family == "lognormal":
            # Let Z~N(0,s²), L=exp(Z). Choose s in [0.3, 0.8] for moderate tails.
            # Center: Y=L - E[L], then standardize X=Y/SD(Y) -> Var=1.
            # κ₃/σ³ (skewness) grows quickly with s; clamp to avoid extreme dynamic range.
            s = rng.uniform(0.3, 0.8)
            # For lognormal with μ=0:
            import math

            m1 = math.exp(0.5 * s * s)
            var = (math.exp(s * s) - 1.0) * math.exp(s * s)
            sd = var**0.5
            # third central moment of lognormal with μ=0:
            mu3 = (
                math.exp(3 * s * s)
                - 3 * math.exp(2 * s * s)
                + 2 * math.exp(1.5 * s * s)
            ) * math.exp(1.5 * s * s)
            # After centering & standardizing, σ=1 and κ₃ = mu3 / sd^3
            sigmas[name] = 1.0
            kappa_mag = mu3 / (sd**3)
            # cap very large values to keep Γ well-conditioned
            kappa_mag = max(min(kappa_mag, 6.0), 0.4)
            kappas[name] = kappa_mag if rng.random() < 0.5 else -kappa_mag

        else:
            raise ValueError(f"Unknown family '{family}'")

    return sigmas, kappas


# ---------- End-to-end ----------
def is_minimal_latent_polytree_check(edges_labeled, hidden_nodes):
    """
    Check if the latent polytree is minimal.
    A latent node h is redundant if removing it keeps the graph a forest.

    Specifically checks for latent chains: if h has exactly one latent parent
    and the parent could directly reach all of h's children, then h is redundant.
    """
    hidden_set = set(hidden_nodes)

    # Build adjacency
    children_map = defaultdict(set)
    parent_map = {}
    for u, v in edges_labeled:
        children_map[u].add(v)
        parent_map[v] = u

    for h in hidden_nodes:
        # Check if this latent has a latent parent
        if h in parent_map and parent_map[h] in hidden_set:
            # This is a latent chain: latent_parent → h → children
            # h is redundant because parent can directly connect to h's children
            return False

    return True


def compute_max_latent_nodes(n_total: int) -> int:
    """
    Compute theoretical maximum number of latent nodes for a minimal latent polytree.

    In a tree with n nodes:
    - Each latent node needs out-degree >= 2
    - This limits the maximum number of latent nodes to roughly n/2

    Returns:
        Maximum number of latent nodes possible
    """
    if n_total <= 2:
        return 0  # Can't have latent nodes with out-degree >= 2

    # Conservative upper bound: floor((n-1)/2)
    # This ensures we have enough nodes to be children
    return (n_total - 1) // 2


def validate_latent_node_request(n_total: int, n_latent: int) -> None:
    """
    Validate that the requested number of latent nodes is feasible.

    Args:
        n_total: Total number of nodes
        n_latent: Requested number of latent nodes

    Raises:
        ValueError: If n_latent exceeds theoretical maximum
    """
    max_latent = compute_max_latent_nodes(n_total)

    if n_latent > max_latent:
        raise ValueError(
            f"Requested {n_latent} latent nodes for tree with {n_total} total nodes. "
            f"Theoretical maximum for minimal latent polytree is ~{max_latent}. "
            f"Each latent node requires out-degree >= 2, limiting the maximum."
        )


def get_candidate_latent_nodes(edges_dir: List[Tuple[int, int]]) -> Set[int]:
    """
    Return nodes that CAN be latent (out-degree >= 2).
    This is the necessary condition from Etesami's paper.
    Not all of these need to be latent, but latent nodes must come from this set.
    """
    od = outdeg_map(edges_dir)
    return {u for u, d in od.items() if d >= 2}


def select_latent_nodes(
    edges_dir: List[Tuple[int, int]],
    n_latent: Optional[int],
    rng: random.Random,
    n_total: int,  # NEW: total nodes for validation
) -> Set[str]:
    """
    Select which nodes should be latent.

    Args:
        edges_dir: Directed edges as integer pairs
        n_latent: Number of latent nodes to select. If None, use all candidates.
        rng: Random number generator
        n_total: Total number of nodes (for validation)

    Returns:
        Set of latent node labels (e.g., {'v1', 'v3'})
    """
    # Validate theoretical upper bound
    if n_latent is not None:
        validate_latent_node_request(n_total, n_latent)

    candidates = get_candidate_latent_nodes(edges_dir)

    if not candidates:
        return set()

    if n_latent is None:
        # Default: all candidates become latent
        return {f"v{u}" for u in candidates}

    if n_latent > len(candidates):
        raise ValueError(
            f"Requested {n_latent} latent nodes but only {len(candidates)} "
            f"candidates with out-degree >= 2 available"
        )

    # Randomly select n_latent nodes from candidates
    selected = rng.sample(sorted(candidates), n_latent)
    return {f"v{u}" for u in selected}


def get_random_polytree_via_pruefer(
    n: int,
    seed: Optional[int] = None,
    weights_range: Tuple[float, float] = (-1.0, 1.0),
    avoid_small: float = 0.8,
    ensure_at_least_one_hidden: bool = True,
    n_latent: Optional[int] = None,  # NEW parameter
):
    """
    Pipeline:
      - generate undirected tree via random Prüfer sequence
      - orient via BFS from a (possibly constrained) root to ensure in-degree <= 1
      - assign random weights
      - set unit sigmas/kappas
      - choose hidden nodes: if n_latent is specified, randomly select n_latent
        nodes from those with out-degree >= 2; otherwise use all such nodes
      - VERIFY minimality: reject if any latent node is redundant
      - compute Γ_full and then Γ_obs by removing hidden nodes
      - run learner on Γ_obs

    Args:
        n: Total number of nodes
        seed: Random seed
        weights_range: Range for edge weights
        avoid_small: Minimum absolute edge weight
        ensure_at_least_one_hidden: If True, retry until at least one latent node
        n_latent: Number of latent nodes to select. If None, all nodes with
                  out-degree >= 2 become latent (default behavior)

    Returns:
      dict with keys:
        'edges_undirected', 'edges_directed', 'weights', 'sigmas', 'kappas',
        'hidden_nodes', 'observed_nodes',
        'Gamma_obs', 'recovered_edges'
    """
    rng = random.Random(seed)

    max_attempts = 1000
    attempts = 0

    while attempts < max_attempts:
        attempts += 1

        seq = random_pruefer_sequence(n, rng)
        undirected = pruefer_to_tree(seq)
        directed = orient_tree_random_topo(undirected, rng, force_hidden_root=True)

        # parameters
        weights = random_weights(
            directed,
            rng,
            low=weights_range[0],
            high=weights_range[1],
            avoid_small=avoid_small,
        )
        nodes = sorted({x for e in directed for x in e})
        sigmas, kappas = sample_sigmas_kappas(nodes, rng, family="gamma")

        # Select latent nodes (with validation)
        try:
            hidden = select_latent_nodes(directed, n_latent, rng, n)  # Pass n
        except ValueError as e:
            if "theoretical maximum" in str(e):
                raise  # Re-raise validation errors immediately
            # Not enough candidates in this specific tree, retry
            continue

        if ensure_at_least_one_hidden and not hidden:
            continue

        # Check minimality
        edges_labeled = [(f"v{u}", f"v{v}") for u, v in directed]
        if not is_minimal_latent_polytree_check(edges_labeled, hidden):
            continue  # Reject non-minimal, keep trying

        # Valid minimal latent polytree found
        break

    if attempts >= max_attempts:
        raise ValueError(
            f"Could not generate minimal latent polytree after {max_attempts} attempts"
        )

    # Build Γ_obs and run learner
    Gamma_obs, observed_nodes, hidden_nodes = observed_gamma_from_params(
        weights, sigmas, kappas, hidden=hidden, auto_detect_hidden=False
    )
    from latent_polytree_truepoly import get_polytree_algo3

    recovered_polytree = get_polytree_algo3(Gamma_obs)

    def name(x: str) -> str:
        return x if x.startswith("h") else observed_nodes[int(x)]

    edges_named = [(name(p), name(c)) for (p, c) in recovered_polytree.edges]
    return {
        "edges_undirected": undirected,
        "edges_directed": [(f"v{u}", f"v{v}") for (u, v) in directed],
        "weights": weights,
        "sigmas": sigmas,
        "kappas": kappas,
        "hidden_nodes": sorted(hidden),
        "observed_nodes": observed_nodes,
        "Gamma_obs": Gamma_obs,
        "recovered_edges": edges_named,
    }


if __name__ == "__main__":
    # Test 1: Default behavior (all candidates become latent)
    print("=== Test 1: Default behavior ===")
    out = get_random_polytree_via_pruefer(n=10, seed=1234)
    print("Directed edges:", out["edges_directed"])
    print("Hidden nodes:", out["hidden_nodes"])
    print("Observed nodes:", out["observed_nodes"])

    # Test 2: Specify exact number of latent nodes
    print("\n=== Test 2: Exactly 2 latent nodes ===")
    out2 = get_random_polytree_via_pruefer(n=10, seed=456, n_latent=2)
    print("Directed edges:", out2["edges_directed"])
    print("Hidden nodes:", out2["hidden_nodes"])
    print("Observed nodes:", out2["observed_nodes"])
    print("Number of latent nodes:", len(out2["hidden_nodes"]))
