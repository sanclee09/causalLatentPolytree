#!/usr/bin/env python3
"""
topology_stratified_evaluation.py

Topology-specific polytree generation and evaluation framework.
Implements three canonical topologies: Chain, Balanced, and Star structures.

Author: TUM Master's Thesis - Latent Polytree Structure Learning
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set, Any
import random
from collections import defaultdict, deque
import numpy as np

# Import from existing codebase
from polytree_discrepancy import (
    Polytree,
    compute_discrepancy_fast,
    generate_noise_samples,
    apply_lsem_transformation,
    finite_sample_discrepancy,
    compute_gamma_parameters_from_moments,
)
from latent_polytree_truepoly import get_polytree_algo3
from learn_with_hidden import observed_gamma_from_params
from random_polytrees_pruefer import is_minimal_latent_polytree_check


def build_adjacency(edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    """Build adjacency list from directed edges."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
    return adj


def compute_outdegrees(edges: List[Tuple[int, int]]) -> Dict[int, int]:
    """Compute out-degrees for all nodes."""
    outdeg = defaultdict(int)
    for u, v in edges:
        outdeg[u] += 1
    return outdeg


def is_properly_balanced(edges: List[Tuple[int, int]], latent_nodes: Set[int], n: int) -> bool:
    """
    Check if topology is properly balanced (hierarchical, not star-like or chain-like).

    Criteria:
    - Multiple nodes should have children (not just root)
    - Max out-degree should be 2 or 3 (not too centralized)
    - At least 2 nodes with out-degree >= 2
    """
    outdeg = compute_outdegrees(edges)

    # Count nodes with different out-degrees
    nodes_with_children = sum(1 for d in outdeg.values() if d >= 1)
    nodes_with_multi_children = sum(1 for d in outdeg.values() if d >= 2)

    total_nodes = len(set(u for e in edges for u in e))

    # Key constraint: Max out-degree should be at most 3 for balanced
    # (Prevents one node from dominating)
    max_deg = max(outdeg.values()) if outdeg else 0
    if max_deg > 3:
        return False

    # For n=5: Need at least 1 non-latent with children AND not too chain-like
    if n == 5:
        non_latent_parents = sum(1 for node, deg in outdeg.items()
                                 if deg >= 1 and node not in latent_nodes)
        # Need at least 1 non-latent parent
        if non_latent_parents < 1:
            return False
        # Should have at least 2 nodes with out-degree >= 2 (latent + 1 other)
        if nodes_with_multi_children < 2:
            return False
        return True

    if n < 8:
        # For small graphs (5-7 nodes), require:
        # - At least 1 non-latent node with children
        # - At least 2 nodes total with out-degree >= 2
        non_latent_parents = sum(1 for node, deg in outdeg.items()
                                 if deg >= 1 and node not in latent_nodes)
        return non_latent_parents >= 1 and nodes_with_multi_children >= 2

    # For larger graphs, more stringent requirements
    # Criterion 1: At least 25% of nodes should have children
    has_enough_parents = nodes_with_children >= 0.25 * total_nodes

    # Criterion 2: At least 2 nodes should have 2+ children (creates branching)
    has_enough_branching = nodes_with_multi_children >= 2

    # Criterion 3: Maximum degree should not be too high (not too star-like)
    # Already checked max_deg <= 3 above

    return has_enough_parents and has_enough_branching


def is_star_structure(edges: List[Tuple[int, int]], latent_nodes: Set[int]) -> bool:
    """
    Check if topology is a star structure (one latent, all children are leaves).
    """
    if len(latent_nodes) != 1:
        return False

    outdeg = compute_outdegrees(edges)
    latent = list(latent_nodes)[0]

    # Check if latent has high degree and all children are leaves
    latent_children = [v for u, v in edges if u == latent]

    # Star: all children of latent are leaves (out-degree 0)
    all_children_are_leaves = all(outdeg.get(child, 0) == 0 for child in latent_children)

    return all_children_are_leaves and len(latent_children) >= 3


def is_chain_structure(edges: List[Tuple[int, int]], latent_nodes: Set[int]) -> bool:
    """
    Check if topology is a chain structure.

    Clear pattern for chain:
    - Single latent root with out-degree = 2
    - Most other nodes have out-degree 0 or 1 (leaves or linear path)
    - Specifically: at least one node with out-degree = 1 (the chain)
    """
    if len(latent_nodes) != 1:
        return False

    outdeg = compute_outdegrees(edges)
    latent = list(latent_nodes)[0]

    # Root latent must have exactly 2 children for chain
    if outdeg.get(latent, 0) != 2:
        return False

    # Get out-degrees of all non-latent nodes
    non_latent_degrees = [outdeg.get(node, 0) for node in outdeg.keys()
                          if node not in latent_nodes]

    # Chain pattern: most nodes should have degree 0 or 1
    # Count nodes with degree > 1 (excluding latent)
    nodes_with_high_degree = sum(1 for d in non_latent_degrees if d > 1)

    # If there are multiple nodes with degree > 1, it's not a simple chain
    # Allow at most 1 such node for flexibility
    return nodes_with_high_degree <= 1


# ============================================================================
# CHAIN TOPOLOGY GENERATOR
# ============================================================================

def generate_chain_topology(
        n: int,
        n_latent: int,
        rng: random.Random,
) -> Tuple[List[Tuple[int, int]], Set[int]]:
    """
    Generate a chain topology: latent root with one immediate child and continuing chain.

    Structure for k=1: h1 -> v1
                           -> v2 -> v3 -> v4 -> ... -> vn

    For k>1, additional latent nodes are placed in chain positions where they can have
    2+ children by adding branching.

    Args:
        n: Total number of nodes
        n_latent: Number of latent nodes (typically 1 for pure chain)
        rng: Random number generator

    Returns:
        (edges, latent_nodes) where edges are directed and latent_nodes is a set
    """
    if n < 4:
        raise ValueError("Chain topology requires n >= 4")
    if n_latent < 1:
        raise ValueError("Chain topology requires at least 1 latent node")

    # For now, only support n_latent=1 (pure chain)
    # Multiple latent nodes in chain require more complex structure
    if n_latent > 1:
        raise ValueError("Chain topology currently supports only n_latent=1")

    nodes = list(range(1, n + 1))
    edges = []

    # Node 1 is the latent root
    latent_nodes = {1}

    # Root has two children: node 2 (observed leaf) and node 3 (chain start)
    edges.append((1, 2))  # immediate observed child
    edges.append((1, 3))  # chain start

    # Build the chain: 3 -> 4 -> 5 -> ... -> n
    for i in range(3, n):
        edges.append((i, i + 1))

    return edges, latent_nodes


# ============================================================================
# BALANCED TOPOLOGY GENERATOR
# ============================================================================

def generate_balanced_topology(
        n: int,
        n_latent: int,
        rng: random.Random,
        max_attempts: int = 100,
) -> Tuple[List[Tuple[int, int]], Set[int]]:
    """
    Generate a balanced branching topology with distributed latent nodes.

    Structure: Hierarchical tree with multiple nodes having children (not star-like).
    Requires n >= 5 to create a structure distinct from chain.

    Args:
        n: Total number of nodes (minimum 5)
        n_latent: Number of latent nodes
        rng: Random number generator
        max_attempts: Maximum retry attempts

    Returns:
        (edges, latent_nodes) where edges are directed and latent_nodes is a set
    """
    if n < 5:
        raise ValueError("Balanced topology requires n >= 5 (need enough nodes for distinct structure)")
    if n_latent < 1:
        raise ValueError("Balanced topology requires at least 1 latent node")

    for attempt in range(max_attempts):
        nodes = list(range(1, n + 1))
        rng.shuffle(nodes)

        # Select latent nodes
        latent_nodes = set(nodes[:n_latent])
        observed_nodes = list(nodes[n_latent:])

        edges = []

        # Root is first latent node
        root = nodes[0]

        # Scale branching requirements with n
        if n == 5:
            # Minimum case: need exactly 1 observed branching node
            n_branching_observed = 1
        elif n < 8:
            # Small graphs: 1-2 observed branching nodes
            n_branching_observed = min(2, max(1, len(observed_nodes) // 3))
        else:
            # Larger graphs: more branching
            n_branching_observed = max(2, min(3, len(observed_nodes) // 3))

        # Select random observed nodes to be branching nodes
        if len(observed_nodes) >= n_branching_observed + 1:  # Need leaves too
            branching_observed = rng.sample(
                observed_nodes,
                min(n_branching_observed, len(observed_nodes) - 1)  # Leave at least 1 for leaf
            )
        else:
            branching_observed = []

        # All latent nodes + selected observed nodes are "parents"
        all_parents = list(latent_nodes) + branching_observed
        rng.shuffle(all_parents)

        # Non-branching observed nodes (will be distributed as children)
        leaf_candidates = [n for n in observed_nodes if n not in branching_observed]
        remaining_children = list(nodes[1:])  # Everyone except root
        rng.shuffle(remaining_children)

        # Build tree: each parent gets children
        assigned = set([root])  # Track assigned nodes
        child_pool = [c for c in remaining_children if c not in assigned]

        # First, give each branching parent children
        for parent in all_parents:
            if parent == root:
                continue  # Handle root separately

            if parent not in assigned:
                # This parent needs to be connected first
                if assigned:
                    connector = rng.choice(list(assigned))
                    edges.append((connector, parent))
                    assigned.add(parent)

            # Give this parent children (scale with graph size)
            # IMPORTANT: Max 2 children per node for balanced structure
            if parent in latent_nodes:
                n_children = 2  # Latent nodes get exactly 2 children
            else:
                # Branching observed: 1-2 children
                if n == 5:
                    n_children = 1  # For n=5, 1 child is enough for branching
                elif n < 8:
                    n_children = rng.randint(1, 2)  # 1-2 children for small graphs
                else:
                    n_children = 2  # 2 children for larger graphs

            for _ in range(n_children):
                if not child_pool:
                    break
                child = child_pool.pop(0)
                edges.append((parent, child))
                assigned.add(child)

        # Connect root to remaining unassigned branching nodes
        for parent in all_parents:
            if parent != root and parent not in assigned:
                edges.append((root, parent))
                assigned.add(parent)

        # Assign remaining children to random parents
        while child_pool:
            parent = rng.choice(all_parents)
            child = child_pool.pop(0)
            edges.append((parent, child))
            assigned.add(child)

        # Verify constraints
        outdeg = compute_outdegrees(edges)

        # 1. All latent nodes have out-degree >= 2
        all_latent_minimal = all(outdeg.get(lat, 0) >= 2 for lat in latent_nodes)
        if not all_latent_minimal:
            continue

        # 2. Check if properly balanced (not star-like)
        if not is_properly_balanced(edges, latent_nodes, n):
            continue

        # 3. Not a chain
        if is_chain_structure(edges, latent_nodes):
            continue

        # Success!
        return edges, latent_nodes

    # Failed after max attempts
    raise ValueError(
        f"Could not generate properly balanced topology after {max_attempts} attempts. "
        f"Try with different n or n_latent."
    )


# ============================================================================
# STAR TOPOLOGY GENERATOR
# ============================================================================

def generate_star_topology(
        n: int,
        n_latent: int,
        rng: random.Random,
) -> Tuple[List[Tuple[int, int]], Set[int]]:
    """
    Generate a star topology: single latent root with many observed children.

    Structure: h1 -> v1, v2, v3, ..., v_{n-1}

    This is the hardest topology due to symmetric discrepancy patterns.

    Args:
        n: Total number of nodes
        n_latent: Number of latent nodes (typically 1 for pure star, 2 for star+secondary)
        rng: Random number generator

    Returns:
        (edges, latent_nodes) where edges are directed and latent_nodes is a set
    """
    if n < 4:
        raise ValueError("Star topology requires n >= 4")
    if n_latent < 1:
        raise ValueError("Star topology requires at least 1 latent node")

    nodes = list(range(1, n + 1))

    # Node 1 is the central latent hub
    latent_nodes = {1}
    root = 1

    edges = []

    if n_latent == 1:
        # Pure star: root connects to all other nodes
        for i in range(2, n + 1):
            edges.append((root, i))

    elif n_latent == 2:
        # Star with one secondary latent node
        # Root connects to most nodes + one secondary latent
        # CRITICAL: Secondary latent must have at least 2 children to be minimal
        secondary_latent = 2
        latent_nodes.add(secondary_latent)

        # Root to secondary latent
        edges.append((root, secondary_latent))

        # Ensure secondary latent has at least 2 children
        # Distribute remaining nodes: give at least 2 to secondary, rest to root
        remaining_nodes = list(range(3, n + 1))
        if len(remaining_nodes) < 2:
            raise ValueError(f"Star with k=2 requires n >= 5 (need 2+ children for secondary latent)")

        # Give first 2 nodes to secondary latent (ensures minimality)
        for i in range(2):
            edges.append((secondary_latent, remaining_nodes[i]))

        # Distribute rest to root
        for i in range(2, len(remaining_nodes)):
            edges.append((root, remaining_nodes[i]))

    else:
        raise ValueError("Star topology supports n_latent=1 or 2 only")

    return edges, latent_nodes


# ============================================================================
# UNIFIED TOPOLOGY GENERATOR
# ============================================================================

def generate_topology(
        topology_type: str,
        n: int,
        n_latent: int,
        seed: int,
) -> Tuple[List[Tuple[int, int]], Set[int]]:
    """
    Unified interface for generating specific topology types.

    Args:
        topology_type: One of 'chain', 'balanced', 'star'
        n: Total number of nodes
        n_latent: Number of latent nodes
        seed: Random seed

    Returns:
        (edges, latent_nodes) where edges are directed and latent_nodes is a set
    """
    rng = random.Random(seed)

    if topology_type == "chain":
        return generate_chain_topology(n, n_latent, rng)
    elif topology_type == "balanced":
        return generate_balanced_topology(n, n_latent, rng)
    elif topology_type == "star":
        return generate_star_topology(n, n_latent, rng)
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")


# ============================================================================
# TESTING
# ============================================================================

def create_polytree_from_topology(
        topology_type: str,
        n: int,
        n_latent: int,
        seed: int,
        weights_range: Tuple[float, float] = (-1.0, 1.0),
        avoid_small: float = 0.8,
        verify_minimality: bool = True,
) -> Dict:
    """
    Create a complete polytree structure from a topology specification.
    Returns same format as get_random_polytree_via_pruefer for compatibility.

    Args:
        topology_type: One of 'chain', 'balanced', 'star'
        n: Total number of nodes
        n_latent: Number of latent nodes
        seed: Random seed
        weights_range: Range for edge weights
        avoid_small: Minimum absolute edge weight
        verify_minimality: Whether to verify the polytree is minimal

    Returns:
        Dictionary with polytree structure compatible with existing codebase
    """
    rng = random.Random(seed)

    # Generate topology
    edges_directed, latent_set = generate_topology(topology_type, n, n_latent, seed)

    # Create undirected edges
    edges_undirected = [(min(u, v), max(u, v)) for u, v in edges_directed]

    # Assign weights
    weights = {}
    for u, v in edges_directed:
        while True:
            w = rng.uniform(weights_range[0], weights_range[1])
            if abs(w) >= avoid_small:
                weights[(u, v)] = w
                break

    # Sample RANDOM noise parameters (matching get_random_polytree_via_pruefer behavior!)
    # This is CRITICAL for fair comparison
    nodes_int = list(range(1, n + 1))
    from random_polytrees_pruefer import sample_sigmas_kappas
    sigmas, kappas = sample_sigmas_kappas(nodes_int, rng, family="gamma")

    # Create node name mappings
    latent_nodes = sorted([f"v{i}" for i in latent_set])
    all_nodes_int = set(range(1, n + 1))
    observed_int = all_nodes_int - latent_set
    observed_nodes = sorted([f"v{i}" for i in observed_int])

    # Convert edges to named format
    edges_directed_named = [(f"v{u}", f"v{v}") for u, v in edges_directed]
    edges_undirected_named = [(f"v{u}", f"v{v}") for u, v in edges_undirected]

    # Verify minimality if requested
    if verify_minimality:
        is_minimal = is_minimal_latent_polytree_check(
            edges_directed_named, latent_nodes
        )
        if not is_minimal:
            raise ValueError(
                f"Generated {topology_type} topology is NOT minimal! "
                f"This indicates a bug in the topology generator."
            )

    # Convert weights to named format
    weights_named = {(f"v{u}", f"v{v}"): w for (u, v), w in weights.items()}

    return {
        "edges_undirected": edges_undirected_named,
        "edges_directed": edges_directed_named,
        "weights": weights_named,
        "sigmas": sigmas,  # String keys: {'v1': 1.0, 'v2': 1.0, ...}
        "kappas": kappas,  # String keys: {'v1': 1.0, 'v2': 1.0, ...}
        "hidden_nodes": latent_nodes,
        "observed_nodes": observed_nodes,
        "topology_type": topology_type,
        "is_minimal": True,  # Verified above
    }


def compute_population_discrepancy_from_topology(
        topology_structure: Dict,
) -> np.ndarray:

    edges = topology_structure["edges_directed"]
    weights = topology_structure["weights"]
    sigmas = topology_structure["sigmas"]  # Already string keys now!
    kappas = topology_structure["kappas"]  # Already string keys now!

    # Convert to edge weight dictionary with string keys
    edge_weights = {}
    for (u, v), w in weights.items():
        edge_weights[(u, v)] = w

    # Create Polytree object
    poly = Polytree(edge_weights, sigmas, kappas)

    # Compute population discrepancy
    Gamma_pop = compute_discrepancy_fast(poly)

    return Gamma_pop


def compute_observed_discrepancy_from_topology(
topology_structure: Dict,
Gamma_full: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:

    # sigmas keys are already strings like 'v1', 'v2', etc.
    all_nodes = sorted(topology_structure["sigmas"].keys())
    latent_nodes = set(topology_structure["hidden_nodes"])
    observed_nodes = [n for n in all_nodes if n not in latent_nodes]

    # Find indices of observed nodes
    obs_indices = [all_nodes.index(n) for n in observed_nodes]

    # Extract observed submatrix
    Gamma_obs = Gamma_full[np.ix_(obs_indices, obs_indices)]

    return Gamma_obs, observed_nodes


if __name__ == "__main__":
    print("=" * 70)
    print("TOPOLOGY GENERATOR TESTS")
    print("=" * 70)

    test_configs = [
        ("chain", 10, 1),
        ("balanced", 10, 2),
        ("star", 10, 1),
    ]

    for topology, n, k in test_configs:
        print(f"\n{'=' * 70}")
        print(f"Testing: {topology.upper()} topology (n={n}, k={k})")
        print(f"{'=' * 70}")

        edges, latent = generate_topology(topology, n, k, seed=42)
        observed = set(range(1, n + 1)) - latent

        print(f"Latent nodes: {sorted(latent)}")
        print(f"Observed nodes: {sorted(observed)}")
        print(f"Edges ({len(edges)}):")
        for u, v in sorted(edges):
            u_type = "h" if u in latent else "v"
            v_type = "h" if v in latent else "v"
            print(f"  {u}({u_type}) -> {v}({v_type})")

        # Verify it's a valid polytree
        outdeg = compute_outdegrees(edges)
        print(f"\nOut-degrees:")
        for node in sorted(range(1, n + 1)):
            deg = outdeg[node]
            node_type = "LATENT" if node in latent else "observed"
            print(f"  Node {node} ({node_type}): out-degree = {deg}")

        # Check minimality for latent nodes (manual check)
        print(f"\nMinimality check (manual):")
        for node in sorted(latent):
            out_deg = outdeg[node]
            if out_deg < 2:
                print(f"  WARNING: Latent node {node} has out-degree {out_deg} < 2")
            else:
                print(f"  ✓ Latent node {node} is minimal (out-degree={out_deg})")

        # Verify using the official minimality checker
        edges_labeled = [(f"v{u}", f"v{v}") for u, v in edges]
        latent_labeled = {f"v{node}" for node in latent}
        is_minimal = is_minimal_latent_polytree_check(edges_labeled, latent_labeled)
        print(f"\nOfficial minimality check: {'✅ MINIMAL' if is_minimal else '❌ NOT MINIMAL'}")

        # Check structure type
        is_star = is_star_structure(edges, latent)
        is_chain = is_chain_structure(edges, latent)
        is_balanced = is_properly_balanced(edges, latent, n)

        print(f"\nStructure classification:")
        print(f"  Is star: {'⭐ YES' if is_star else '✗ No'}")
        print(f"  Is chain: {'🔗 YES' if is_chain else '✗ No'}")
        print(f"  Is balanced: {'🌳 YES' if is_balanced else '✗ No'}")

        # Show branching statistics for balanced topologies
        if topology == "balanced":
            nodes_with_deg_2plus = sum(1 for node in range(1, n + 1) if outdeg.get(node, 0) >= 2)
            nodes_with_deg_1plus = sum(1 for node in range(1, n + 1) if outdeg.get(node, 0) >= 1)
            max_degree = max(outdeg.values()) if outdeg else 0
            print(f"\n  Branching statistics:")
            print(f"    Nodes with out-degree >= 2: {nodes_with_deg_2plus}")
            print(f"    Nodes with out-degree >= 1: {nodes_with_deg_1plus}")
            print(f"    Maximum out-degree: {max_degree}")
            print(f"    Latent out-degrees: {[outdeg.get(lat, 0) for lat in sorted(latent)]}")

        if not is_star and not is_chain and is_balanced:
            print(f"  → ✅ Properly balanced/hierarchical structure")

    # Test full pipeline with discrepancy computation
    print(f"\n{'=' * 70}")
    print("FULL PIPELINE TEST: Chain topology with discrepancy")
    print(f"{'=' * 70}")

    # Create polytree structure
    chain_poly = create_polytree_from_topology('chain', n=6, n_latent=1, seed=42)

    print(f"\nPolytree structure:")
    print(f"  Topology: {chain_poly['topology_type']}")
    print(f"  Total nodes: 6")
    print(f"  Latent nodes: {chain_poly['hidden_nodes']}")
    print(f"  Observed nodes: {chain_poly['observed_nodes']}")
    print(f"  Edges: {chain_poly['edges_directed']}")

    # Compute population discrepancy (full)
    print(f"\n{'-' * 70}")
    print("POPULATION DISCREPANCY COMPUTATION")
    print(f"{'-' * 70}")

    Gamma_full = compute_population_discrepancy_from_topology(chain_poly)
    print(f"Full discrepancy matrix shape: {Gamma_full.shape}")

    # Get all nodes in order
    all_nodes = sorted([f"v{i}" for i in range(1, 7)])
    latent_set = set(chain_poly['hidden_nodes'])

    print(f"\nFull Γ matrix (including latent node v1):")
    print(f"Nodes: {all_nodes}")
    print(f"       " + "  ".join(f"{n:>8s}" for n in all_nodes))
    for i, node in enumerate(all_nodes):
        node_type = "LATENT" if node in latent_set else "obs"
        print(f"{node}({node_type}): " + " ".join(f"{Gamma_full[i, j]:8.4f}" for j in range(len(all_nodes))))

    # Extract observed discrepancy
    print(f"\n{'-' * 70}")
    print("OBSERVED DISCREPANCY MATRIX (marginalizing out latent)")
    print(f"{'-' * 70}")

    Gamma_obs, obs_nodes = compute_observed_discrepancy_from_topology(
        chain_poly, Gamma_full
    )

    print(f"\nObserved Γ matrix ({len(obs_nodes)}x{len(obs_nodes)}):")
    print(f"Nodes: {obs_nodes}")
    print(f"     " + "  ".join(f"{n:>8s}" for n in obs_nodes))
    for i, node in enumerate(obs_nodes):
        print(f"{node}: " + " ".join(f"{Gamma_obs[i, j]:8.4f}" for j in range(len(obs_nodes))))

    # Analyze discrepancy patterns
    print(f"\n{'-' * 70}")
    print("DISCREPANCY PATTERN ANALYSIS")
    print(f"{'-' * 70}")

    # Check key properties
    print(f"\n1. Diagonal zeros:")
    diag_zeros = np.allclose(np.diag(Gamma_obs), 0.0)
    print(f"   {'✓' if diag_zeros else '✗'} All diagonal entries are zero")

    print(f"\n2. Root children pattern (v2, v3):")
    print(f"   v2 is leaf (immediate child of h1)")
    print(f"   v3 starts chain (child of h1, parent of v4)")
    v2_idx = obs_nodes.index('v2')
    v3_idx = obs_nodes.index('v3')
    print(f"   Γ(v2, v3) = {Gamma_obs[v2_idx, v3_idx]:.4f}")
    print(f"   Γ(v3, v2) = {Gamma_obs[v3_idx, v2_idx]:.4f}")

    print(f"\n3. Chain structure (v3->v4->v5->v6):")
    for i in range(len(obs_nodes) - 1):
        if obs_nodes[i] in ['v3', 'v4', 'v5']:
            node_i = obs_nodes[i]
            node_j = obs_nodes[i + 1]
            idx_i = obs_nodes.index(node_i)
            idx_j = obs_nodes.index(node_j)
            gamma_ij = Gamma_obs[idx_i, idx_j]
            gamma_ji = Gamma_obs[idx_j, idx_i]
            print(f"   Γ({node_i}, {node_j}) = {gamma_ij:.4f},  Γ({node_j}, {node_i}) = {gamma_ji:.4f}")

    print(f"\n4. Separation patterns:")
    print(f"   Nodes in chain should have decreasing discrepancy to root children")
    v2_to_chain = [Gamma_obs[v2_idx, obs_nodes.index(f'v{j}')] for j in [3, 4, 5, 6]]
    print(f"   v2 to chain nodes: {[f'{x:.4f}' for x in v2_to_chain]}")

    print(f"\n{'=' * 70}")
    print("✅ FULL PIPELINE TEST COMPLETE")
    print(f"{'=' * 70}")