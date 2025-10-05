from __future__ import annotations
from typing import Dict, Tuple, List, Any, Optional, Set
import random
from collections import defaultdict, deque
from learn_with_hidden import observed_gamma_from_params

# ---------- Prüfer utilities ----------


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


def get_random_polytree_via_pruefer(
    n: int,
    seed: Optional[int] = None,
    weights_range: Tuple[float, float] = (-1.0, 1.0),
    avoid_small: float = 0.8,
    ensure_at_least_one_hidden: bool = True,
):
    """
    Pipeline:
      - generate undirected tree via random Prüfer sequence
      - orient via BFS from a (possibly constrained) root to ensure in-degree <= 1
      - assign random weights
      - set unit sigmas/kappas
      - choose hidden nodes per rule (default: any node with out-degree >= 2)
      - compute Γ_full and then Γ_obs by removing hidden nodes
      - run learner on Γ_obs

    Returns:
      dict with keys:
        'edges_undirected', 'edges_directed', 'weights', 'sigmas', 'kappas',
        'hidden_nodes', 'observed_nodes',
        'Gamma_obs', 'recovered_edges'
    """
    rng = random.Random(seed)

    while True:
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

        # hidden per rule
        hidden = branching_hidden_nodes(directed)

        if (not ensure_at_least_one_hidden) or hidden:
            break
        # Otherwise resample (rare for n >= 3 with branching rule)

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


def is_minimal_latent_polytree(
    edges: List[Tuple[int, int]], latent_nodes: Set[int]
) -> bool:
    """
    Check if latent nodes form a minimal latent polytree.
    A latent polytree is minimal if no latent node is redundant.

    A latent node is redundant if:
    - It has out-degree >= 2 (branching)
    - BUT all its children are observed leaf nodes (no descendants)

    In such cases, the latent node doesn't add identifiability and can be removed.
    """
    if not latent_nodes:
        return True

    children_map = defaultdict(set)
    for u, v in edges:
        children_map[u].add(v)

    for latent in latent_nodes:
        children = children_map[latent]

        # Check if this latent has at least one child that:
        # 1. Is also latent (forms a latent chain), OR
        # 2. Is observed but has its own children (not a leaf)
        has_nonleaf_child = False
        for child in children:
            if child in latent_nodes:
                # Child is latent - this is good
                has_nonleaf_child = True
                break
            elif len(children_map[child]) > 0:
                # Child is observed but has descendants - this is good
                has_nonleaf_child = True
                break

        # If all children are observed leaves, this latent is redundant
        if not has_nonleaf_child:
            return False

    return True


def generate_random_latent_polytree(
    n: int,
    seed: Optional[int] = None,
    weights_range: Tuple[float, float] = (-1.0, 1.0),
    avoid_small: float = 0.8,
    ensure_at_least_one_hidden: bool = True,
    max_attempts: int = 1000,
) -> Dict[str, Any]:
    """
    Generate a random MINIMAL latent polytree structure.
    Only nodes with undirected degree >= 3 can be latent (non-redundant).
    """
    rng = random.Random(seed)

    attempts = 0
    while attempts < max_attempts:
        attempts += 1

        # Generate undirected tree
        seq = random_pruefer_sequence(n, rng)
        undirected = pruefer_to_tree(seq)

        # Compute undirected degrees
        from collections import defaultdict

        deg = defaultdict(int)
        for u, v in undirected:
            deg[u] += 1
            deg[v] += 1

        # Only nodes with degree >= 3 can be latent (branching points)
        candidates = [u for u, d in deg.items() if d >= 3]

        if ensure_at_least_one_hidden and not candidates:
            continue

        # Pick latent nodes from candidates
        if candidates:
            k_hidden = (
                rng.randint(1, len(candidates))
                if ensure_at_least_one_hidden
                else rng.randint(0, len(candidates))
            )
            latent_integer_nodes = (
                set(rng.sample(candidates, k_hidden)) if k_hidden > 0 else set()
            )
        else:
            latent_integer_nodes = set()

        if ensure_at_least_one_hidden and not latent_integer_nodes:
            continue

        # Orient from a latent root to avoid observed→latent edges
        if latent_integer_nodes:
            root = rng.choice(list(latent_integer_nodes))
            directed = orient_tree_from_root(undirected, root)
        else:
            # No latents - use random orientation
            directed = orient_tree_random_topo(undirected, rng, force_hidden_root=False)

        # Valid minimal latent polytree found
        break

    if attempts >= max_attempts:
        raise ValueError(
            f"Could not generate minimal latent polytree after {max_attempts} attempts"
        )

    # Rest of the code remains the same (weight assignment, renaming, etc.)
    weights_integer = random_weights(
        directed,
        rng,
        low=weights_range[0],
        high=weights_range[1],
        avoid_small=avoid_small,
    )

    # Topological ordering for consistent latent node naming
    all_integer_nodes = sorted({x for e in directed for x in e})

    from collections import deque

    in_degree = {node: 0 for node in all_integer_nodes}
    adj = {node: [] for node in all_integer_nodes}
    for u, v in directed:
        adj[u].append(v)
        in_degree[v] += 1

    queue = deque([node for node in all_integer_nodes if in_degree[node] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Assign names: latent nodes by topo order, observed nodes by integer order
    latent_in_topo = [node for node in topo_order if node in latent_integer_nodes]
    observed_in_order = sorted(
        [node for node in all_integer_nodes if node not in latent_integer_nodes]
    )

    # Create mapping from integer to named nodes
    node_mapping = {}
    for i, node in enumerate(latent_in_topo):
        node_mapping[node] = f"h{i + 1}"
    for i, node in enumerate(observed_in_order):
        node_mapping[node] = f"v{i + 1}"

    # Rebuild edges with new names
    renamed_edges = {}
    for (u_str, v_str), weight in weights_integer.items():
        # Extract integer from 'v3' -> 3
        u_int = int(u_str[1:])
        v_int = int(v_str[1:])

        # Map to new names
        u_name = node_mapping[u_int]
        v_name = node_mapping[v_int]
        renamed_edges[(u_name, v_name)] = weight

    # Sort nodes: latent first (h1, h2, ...), then observed (v1, v2, ...)
    all_nodes = sorted(
        node_mapping.values(), key=lambda x: (0 if x.startswith("h") else 1, int(x[1:]))
    )
    observed_nodes = [node_mapping[n] for n in observed_in_order]
    latent_nodes = [node_mapping[n] for n in latent_in_topo]

    return {
        "edges": renamed_edges,
        "all_nodes": all_nodes,
        "observed_nodes": observed_nodes,
        "latent_nodes": latent_nodes,
    }


if __name__ == "__main__":
    # Tiny smoke test
    out = get_random_polytree_via_pruefer(n=7, seed=123)
    print("Directed edges:", out["edges_directed"])
    print("Hidden nodes:", out["hidden_nodes"])
    print("Observed nodes:", out["observed_nodes"])
    print("Recovered edges:", sorted(out["recovered_edges"]))
