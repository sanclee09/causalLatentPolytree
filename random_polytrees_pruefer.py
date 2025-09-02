from __future__ import annotations
from typing import List, Tuple, Dict, Set, Optional
import random

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


def thesis_hidden_nodes_root_only(edges_dir: List[Tuple[int, int]]) -> Set[str]:
    """Old rule: only roots (in-degree 0) with >=2 children are hidden."""
    nodes = {x for e in edges_dir for x in e}
    indeg = {u: 0 for u in nodes}
    ch: Dict[int, set] = {u: set() for u in nodes}
    for u, v in edges_dir:
        indeg[v] += 1
        ch[u].add(v)
    return {f"v{u}" for u in nodes if indeg[u] == 0 and len(ch[u]) >= 2}


def branching_hidden_nodes(edges_dir: List[Tuple[int, int]]) -> Set[str]:
    """New rule: any node with out-degree >= 2 is hidden (learnable)."""
    od = outdeg_map(edges_dir)
    return {f"v{u}" for u, d in od.items() if d >= 2}


def choose_hidden_nodes(
    edges_dir: List[Tuple[int, int]], rule: str = "branching"
) -> Set[str]:
    if rule == "branching":
        return branching_hidden_nodes(edges_dir)
    elif rule == "root":
        return thesis_hidden_nodes_root_only(edges_dir)
    else:
        raise ValueError(f"Unknown hidden rule: {rule}")


# ---------- Weighting & parameters ----------


def random_weights(
    edges_dir: List[Tuple[int, int]],
    rng: random.Random,
    low: float = -1.0,
    high: float = 1.0,
    avoid_small: float = 0.1,
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


# ---------- End-to-end ----------


def sample_random_polytree_via_pruefer(
    n: int,
    seed: Optional[int] = None,
    weights_range: Tuple[float, float] = (-1.0, 1.0),
    avoid_small: float = 0.1,
    ensure_at_least_one_hidden: bool = True,
    hidden_rule: str = "branching",  # 'branching' or 'root'
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
        sigmas, kappas = unit_sigmas_kappas(nodes)

        # hidden per rule
        hidden = choose_hidden_nodes(directed, rule=hidden_rule)

        if (not ensure_at_least_one_hidden) or hidden:
            break
        # Otherwise resample (rare for n >= 3 with branching rule)

    # Build Γ_obs and run learner
    Gamma_obs, observed_nodes, hidden_nodes = observed_gamma_from_params(
        weights, sigmas, kappas, hidden=hidden, auto_detect_hidden=False
    )
    from latent_polytree_truepoly import get_polytree_algo3

    observed_polytree = get_polytree_algo3(Gamma_obs)

    def name(x: str) -> str:
        return x if x.startswith("h") else observed_nodes[int(x)]

    edges_named = [(name(p), name(c)) for (p, c) in observed_polytree.edges]
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
    # Tiny smoke test
    out = sample_random_polytree_via_pruefer(n=7, seed=123, hidden_rule="branching")
    print("Directed edges:", out["edges_directed"])
    print("Hidden nodes:", out["hidden_nodes"])
    print("Observed nodes:", out["observed_nodes"])
    print("Recovered edges:", sorted(out["recovered_edges"]))
