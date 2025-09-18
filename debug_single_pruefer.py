#!/usr/bin/env python3
"""
debug_single_pruefer.py

Refactored debugging utility for polytree structure recovery using Prüfer sequences.
This module provides comprehensive validation and analysis of the polytree learning pipeline.

Key functionality:
- Validates polytree generation via Prüfer sequences
- Verifies discrepancy matrix computation
- Analyzes structure recovery accuracy
- Provides detailed diagnostic output

Author: Based on original implementation for TUM thesis
"""

from __future__ import annotations
from typing import Dict, Set, Tuple, List, Optional, Any
import numpy as np

from random_polytrees_pruefer import get_random_polytree_via_pruefer
from polytree_discrepancy import Polytree, compute_discrepancy_fast


class PolytreeDebugger:
    """
    Comprehensive debugging and validation class for polytree structure learning.

    This class encapsulates all validation logic and provides structured output
    for analyzing the performance of the polytree recovery algorithm.
    """

    def __init__(self, n: int = 20, seed: int = 2025081314):
        """
        Initialize debugger with specified parameters.

        Args:
            n: Number of nodes in the polytree
            seed: Random seed for reproducible results
        """
        self.n = n
        self.seed = seed
        self.polytree_sample = get_random_polytree_via_pruefer(n=n, seed=seed)

    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Execute complete debugging analysis.

        Returns:
            Dictionary containing all analysis results
        """
        print("=" * 60)
        print(f"POLYTREE DEBUGGING ANALYSIS (n={self.n}, seed={self.seed})")
        print("=" * 60)

        results = {}

        # Step 1: Display ground truth structure
        results["ground_truth"] = self._analyze_ground_truth()

        # Step 2: Validate discrepancy matrix computation
        results["discrepancy_validation"] = self._validate_discrepancy_matrix()

        # Step 3: Analyze structure recovery performance
        results["recovery_analysis"] = self._analyze_structure_recovery()

        # Step 4: Validate latent-to-observed mapping
        results["latent_mapping"] = self._analyze_latent_mapping()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

        return results

    def _analyze_ground_truth(self) -> Dict[str, Any]:
        """Analyze and display ground truth polytree structure."""
        print("\n=== GROUND TRUTH POLYTREE STRUCTURE ===")

        directed_edges = self.polytree_sample["edges_directed"]
        hidden_nodes = self.polytree_sample["hidden_nodes"]
        observed_nodes = self.polytree_sample["observed_nodes"]

        print(f"Total nodes: {self.n}")
        print(f"Hidden nodes ({len(hidden_nodes)}): {sorted(hidden_nodes)}")
        print(f"Observed nodes ({len(observed_nodes)}): {sorted(observed_nodes)}")
        print(f"Directed edges ({len(directed_edges)}): {sorted(directed_edges)}")

        return {
            "total_nodes": self.n,
            "hidden_nodes": hidden_nodes,
            "observed_nodes": observed_nodes,
            "directed_edges": directed_edges,
            "n_hidden": len(hidden_nodes),
            "n_observed": len(observed_nodes),
        }

    def _validate_discrepancy_matrix(self) -> Dict[str, Any]:
        """Validate discrepancy matrix computation consistency."""
        print("\n=== DISCREPANCY MATRIX VALIDATION ===")

        # Display observed discrepancy matrix
        gamma_obs = self.polytree_sample["Gamma_obs"]
        print(f"Γ_obs shape: {gamma_obs.shape}")
        print("Γ_obs (population discrepancy matrix):")
        self._print_matrix(gamma_obs)

        # Recompute discrepancy matrix for validation
        polytree = Polytree(
            self.polytree_sample["weights"],
            self.polytree_sample["sigmas"],
            self.polytree_sample["kappas"],
        )

        full_gamma = compute_discrepancy_fast(polytree)
        observed_nodes = self.polytree_sample["observed_nodes"]
        observed_indices = [polytree.nodes.index(v) for v in observed_nodes]
        gamma_obs_recomputed = full_gamma[np.ix_(observed_indices, observed_indices)]

        # Check consistency
        max_diff = np.max(np.abs(gamma_obs_recomputed - gamma_obs))
        is_consistent = max_diff < 1e-9

        print(f"\nConsistency check: {'PASSED' if is_consistent else 'FAILED'}")
        print(f"Maximum difference: {max_diff:.2e}")

        return {
            "gamma_obs_shape": gamma_obs.shape,
            "gamma_obs": gamma_obs,
            "consistency_check": is_consistent,
            "max_difference": float(max_diff),
        }

    def _analyze_structure_recovery(self) -> Dict[str, Any]:
        """Analyze structure recovery performance."""
        print("\n=== STRUCTURE RECOVERY ANALYSIS ===")

        recovered_edges = self.polytree_sample["recovered_edges"]
        print(f"Recovered edges ({len(recovered_edges)}): {sorted(recovered_edges)}")

        # Analyze observed-to-observed edge recovery
        obs_recovery = self._analyze_observed_edge_recovery()

        return {
            "recovered_edges": recovered_edges,
            "n_recovered": len(recovered_edges),
            **obs_recovery,
        }

    def _analyze_observed_edge_recovery(self) -> Dict[str, Any]:
        """Analyze recovery performance for observed-to-observed edges."""
        observed_nodes = set(self.polytree_sample["observed_nodes"])

        # Extract true observed-to-observed edges
        true_obs_edges = {
            (u, v)
            for (u, v) in self.polytree_sample["edges_directed"]
            if u in observed_nodes and v in observed_nodes
        }

        # Extract recovered observed-to-observed edges
        recovered_obs_edges = {
            (u, v)
            for (u, v) in self.polytree_sample["recovered_edges"]
            if not u.startswith("h") and not v.startswith("h")
        }

        # Compute metrics
        is_superset = true_obs_edges.issubset(recovered_obs_edges)
        missing_edges = true_obs_edges - recovered_obs_edges
        extra_edges = recovered_obs_edges - true_obs_edges

        print(f"\nObserved-to-observed edge analysis:")
        print(f"  True observed edges: {len(true_obs_edges)}")
        print(f"  Recovered observed edges: {len(recovered_obs_edges)}")
        print(f"  Recovery is superset of truth: {'YES' if is_superset else 'NO'}")

        if missing_edges:
            print(f"  Missing edges ({len(missing_edges)}): {sorted(missing_edges)}")
        if extra_edges:
            print(f"  Extra edges ({len(extra_edges)}): {sorted(extra_edges)}")

        return {
            "true_observed_edges": true_obs_edges,
            "recovered_observed_edges": recovered_obs_edges,
            "is_superset": is_superset,
            "missing_edges": list(missing_edges),
            "extra_edges": list(extra_edges),
            "n_true_obs": len(true_obs_edges),
            "n_recovered_obs": len(recovered_obs_edges),
        }

    def _analyze_latent_mapping(self) -> Dict[str, Any]:
        """Analyze latent variable to observed variable mapping."""
        print("\n=== LATENT-TO-OBSERVED MAPPING ANALYSIS ===")

        observed_nodes = set(self.polytree_sample["observed_nodes"])
        true_hidden = set(self.polytree_sample["hidden_nodes"])

        # Build true latent-to-children mapping
        true_latent_children = self._build_latent_children_mapping(
            self.polytree_sample["edges_directed"], true_hidden, observed_nodes
        )

        # Build recovered latent-to-children mapping
        recovered_latent_children = self._build_recovered_latent_children_mapping(
            self.polytree_sample["recovered_edges"]
        )

        # Perform greedy matching
        mapping = self._compute_greedy_mapping(
            true_latent_children, recovered_latent_children
        )

        # Display results
        print(f"True latent children sets:")
        for latent, children in true_latent_children.items():
            print(f"  {latent}: {sorted(children)}")

        print(f"\nRecovered latent children sets:")
        for latent, children in recovered_latent_children.items():
            print(f"  {latent}: {sorted(children)}")

        print(f"\nGreedy mapping (true → recovered):")
        for true_latent, recovered_latent in mapping.items():
            if recovered_latent is not None:
                print(f"  {true_latent} → {recovered_latent}")
            else:
                print(f"  {true_latent} → [UNMATCHED]")

        return {
            "true_latent_children": true_latent_children,
            "recovered_latent_children": recovered_latent_children,
            "mapping": mapping,
            "n_true_latents": len(true_latent_children),
            "n_recovered_latents": len(recovered_latent_children),
        }

    def _build_latent_children_mapping(
        self,
        edges: List[Tuple[str, str]],
        hidden_nodes: Set[str],
        observed_nodes: Set[str],
    ) -> Dict[str, Set[str]]:
        """Build mapping from latent nodes to their observed children."""
        latent_children = {}
        for u, v in edges:
            if u in hidden_nodes and v in observed_nodes:
                latent_children.setdefault(u, set()).add(v)
        return latent_children

    def _build_recovered_latent_children_mapping(
        self, recovered_edges: List[Tuple[str, str]]
    ) -> Dict[str, Set[str]]:
        """Build mapping from recovered latent nodes to their children."""
        recovered_latent_children = {}
        for u, v in recovered_edges:
            if u.startswith("h") and not v.startswith("h"):
                recovered_latent_children.setdefault(u, set()).add(v)
        return recovered_latent_children

    def _compute_greedy_mapping(
        self,
        true_children: Dict[str, Set[str]],
        recovered_children: Dict[str, Set[str]],
    ) -> Dict[str, Optional[str]]:
        """Compute greedy mapping between true and recovered latents using Jaccard similarity."""

        def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
            """Compute Jaccard similarity coefficient."""
            if not set_a and not set_b:
                return 1.0
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / len(set_a | set_b)

        # Generate all possible pairings with similarity scores
        candidates = []
        for true_latent, true_set in true_children.items():
            for recovered_latent, recovered_set in recovered_children.items():
                similarity = jaccard_similarity(true_set, recovered_set)
                candidates.append((similarity, true_latent, recovered_latent))

        # Sort by similarity (descending)
        candidates.sort(reverse=True)

        # Greedy matching
        mapped_true = set()
        mapped_recovered = set()
        mapping = {true_latent: None for true_latent in true_children}

        for similarity, true_latent, recovered_latent in candidates:
            if true_latent in mapped_true or recovered_latent in mapped_recovered:
                continue

            # Accept mapping if similarity is reasonable or both sets are empty
            if similarity >= 0.5 or (
                not true_children[true_latent]
                and not recovered_children[recovered_latent]
            ):
                mapping[true_latent] = recovered_latent
                mapped_true.add(true_latent)
                mapped_recovered.add(recovered_latent)

        return mapping

    def _print_matrix(self, matrix: np.ndarray, precision: int = 3) -> None:
        """Pretty print a matrix with specified precision."""
        for row in matrix:
            formatted_row = " ".join(f"{x:6.{precision}f}" for x in row)
            print(f"  {formatted_row}")


def main(n: int = 20, seed: int = 2025081314) -> Dict[str, Any]:
    """
    Main debugging function.

    Args:
        n: Number of nodes in the polytree
        seed: Random seed for reproducible results

    Returns:
        Dictionary containing complete analysis results
    """
    debugger = PolytreeDebugger(n=n, seed=seed)
    return debugger.run_complete_analysis()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Debug and validate polytree structure recovery for a single n"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of nodes in the polytree (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025081314,
        help="Random seed for reproducible results (default: 2025081314)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output, only show summary",
    )

    args = parser.parse_args()

    if args.quiet:
        # Suppress print statements in quiet mode
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            results = main(n=args.n, seed=args.seed)
            sys.stdout = old_stdout
            print(f"Analysis completed for n={args.n}, seed={args.seed}")
            print(
                f"Recovery is superset: {results['recovery_analysis']['is_superset']}"
            )
            print(
                f"Consistency check: {results['discrepancy_validation']['consistency_check']}"
            )
        finally:
            sys.stdout = old_stdout
    else:
        main(n=args.n, seed=args.seed)
