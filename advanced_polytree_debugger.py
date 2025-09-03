#!/usr/bin/env python3
"""
advanced_polytree_debugger.py

Advanced debugging tools for polytree structure recovery with detailed failure analysis.
This extends the basic debugger with sophisticated diagnostic capabilities for larger graphs.

Key features:
- Path analysis for edge recovery failures
- Latent hierarchy reconstruction validation
- Numerical stability analysis
- Performance scaling metrics
- Detailed error categorization

Author: For TUM thesis - Linear non-Gaussian models with latent polytree structure
"""

from __future__ import annotations
from typing import Dict, Set, Tuple, List, Optional, Any, Union
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

from debug_single_pruefer import PolytreeDebugger
from random_polytrees_pruefer import get_random_polytree_via_pruefer


class AdvancedPolytreeDebugger(PolytreeDebugger):
    """
    Advanced debugging class with sophisticated failure analysis capabilities.

    Extends the basic debugger with:
    - Path analysis for understanding edge recovery failures
    - Latent hierarchy validation
    - Numerical stability diagnostics
    - Performance scaling analysis
    """

    def __init__(self, n: int = 20, seed: int = 2025081314):
        super().__init__(n, seed)
        self.true_graph = None
        self.recovered_graph = None
        self._build_graphs()

    def _build_graphs(self):
        """Build NetworkX graphs for advanced analysis."""
        # True graph
        self.true_graph = nx.DiGraph()
        for u, v in self.polytree_sample["edges_directed"]:
            self.true_graph.add_edge(u, v)

        # Recovered graph
        self.recovered_graph = nx.DiGraph()
        for u, v in self.polytree_sample["recovered_edges"]:
            self.recovered_graph.add_edge(u, v)

    def run_advanced_analysis(self) -> Dict[str, Any]:
        """Execute comprehensive advanced debugging analysis."""
        print("=" * 80)
        print(f"ADVANCED POLYTREE DEBUGGING ANALYSIS (n={self.n}, seed={self.seed})")
        print("=" * 80)

        # Run basic analysis first
        basic_results = super().run_complete_analysis()

        # Advanced analyses
        advanced_results = {}

        print("\n" + "=" * 60)
        print("ADVANCED FAILURE ANALYSIS")
        print("=" * 60)

        # 1. Path Analysis
        advanced_results["path_analysis"] = self._analyze_path_failures()

        # 2. Latent Hierarchy Analysis
        advanced_results["hierarchy_analysis"] = self._analyze_latent_hierarchy()

        # 3. Numerical Stability Analysis
        advanced_results["numerical_analysis"] = self._analyze_numerical_stability()

        # 4. Graph Structure Metrics
        advanced_results["structure_metrics"] = self._compute_structure_metrics()

        # 5. Error Categorization
        advanced_results["error_categories"] = self._categorize_errors()

        # 6. Recovery Quality Assessment
        advanced_results["recovery_quality"] = self._assess_recovery_quality()

        return {**basic_results, **advanced_results}

    def _analyze_path_failures(self) -> Dict[str, Any]:
        """Analyze path-related failures in edge recovery."""
        print("\n--- PATH FAILURE ANALYSIS ---")

        observed_nodes = set(self.polytree_sample["observed_nodes"])

        # Get missing and extra edges from basic analysis
        true_obs_edges = {
            (u, v)
            for (u, v) in self.polytree_sample["edges_directed"]
            if u in observed_nodes and v in observed_nodes
        }
        recovered_obs_edges = {
            (u, v)
            for (u, v) in self.polytree_sample["recovered_edges"]
            if not u.startswith("h") and not v.startswith("h")
        }

        missing_edges = true_obs_edges - recovered_obs_edges
        extra_edges = recovered_obs_edges - true_obs_edges

        path_analysis = {
            "missing_edges": list(missing_edges),
            "extra_edges": list(extra_edges),
            "path_explanations": {},
        }

        # Analyze each missing edge
        for u, v in missing_edges:
            explanation = self._explain_missing_edge(u, v)
            path_analysis["path_explanations"][f"missing_{u}_{v}"] = explanation
            print(f"Missing edge ({u} → {v}): {explanation['summary']}")

        # Analyze each extra edge
        for u, v in extra_edges:
            explanation = self._explain_extra_edge(u, v)
            path_analysis["path_explanations"][f"extra_{u}_{v}"] = explanation
            print(f"Extra edge ({u} → {v}): {explanation['summary']}")

        return path_analysis

    def _explain_missing_edge(self, u: str, v: str) -> Dict[str, Any]:
        """Explain why a true edge is missing from recovery."""
        # Check if there's an alternative path in recovered graph
        try:
            if nx.has_path(self.recovered_graph, u, v):
                path = nx.shortest_path(self.recovered_graph, u, v)
                return {
                    "summary": f"Replaced by indirect path: {' → '.join(path)}",
                    "alternative_path": path,
                    "path_length": len(path) - 1,
                    "type": "indirect_path",
                }
            else:
                return {
                    "summary": "No alternative path found",
                    "alternative_path": None,
                    "path_length": float("inf"),
                    "type": "completely_missing",
                }
        except nx.NetworkXNoPath:
            return {
                "summary": "No path exists in recovered graph",
                "alternative_path": None,
                "path_length": float("inf"),
                "type": "no_path",
            }

    def _explain_extra_edge(self, u: str, v: str) -> Dict[str, Any]:
        """Explain why a recovered edge is not in the true graph."""
        # Check if this edge shortcuts a longer path in true graph
        try:
            if nx.has_path(self.true_graph, u, v):
                true_path = nx.shortest_path(self.true_graph, u, v)
                if len(true_path) > 2:  # Shortcut detected
                    return {
                        "summary": f"Shortcuts true path: {' → '.join(true_path)}",
                        "true_path": true_path,
                        "path_length": len(true_path) - 1,
                        "type": "shortcut",
                    }
                else:
                    return {
                        "summary": "Direct edge exists in true graph (unexpected)",
                        "true_path": true_path,
                        "path_length": 1,
                        "type": "unexpected_match",
                    }
            else:
                return {
                    "summary": "No corresponding path in true graph",
                    "true_path": None,
                    "path_length": float("inf"),
                    "type": "spurious",
                }
        except nx.NetworkXNoPath:
            return {
                "summary": "No path in true graph",
                "true_path": None,
                "path_length": float("inf"),
                "type": "spurious",
            }

    def _analyze_latent_hierarchy(self) -> Dict[str, Any]:
        """Analyze latent variable hierarchy reconstruction."""
        print("\n--- LATENT HIERARCHY ANALYSIS ---")

        # Build latent hierarchy from true graph
        true_latents = set(self.polytree_sample["hidden_nodes"])
        recovered_latents = {
            node for node in self.recovered_graph.nodes() if node.startswith("h")
        }

        # Analyze latent-to-latent connections
        true_latent_edges = {
            (u, v)
            for (u, v) in self.polytree_sample["edges_directed"]
            if u in true_latents and v in true_latents
        }

        recovered_latent_edges = {
            (u, v)
            for (u, v) in self.polytree_sample["recovered_edges"]
            if u.startswith("h") and v.startswith("h")
        }

        # Compute hierarchy metrics
        true_hierarchy_depth = self._compute_hierarchy_depth(
            self.true_graph, true_latents
        )
        recovered_hierarchy_depth = self._compute_hierarchy_depth(
            self.recovered_graph, recovered_latents
        )

        hierarchy_analysis = {
            "true_latent_count": len(true_latents),
            "recovered_latent_count": len(recovered_latents),
            "true_latent_edges": list(true_latent_edges),
            "recovered_latent_edges": list(recovered_latent_edges),
            "true_hierarchy_depth": true_hierarchy_depth,
            "recovered_hierarchy_depth": recovered_hierarchy_depth,
            "latent_edge_accuracy": self._compute_latent_edge_accuracy(
                true_latent_edges, recovered_latent_edges
            ),
        }

        print(f"True latent hierarchy depth: {true_hierarchy_depth}")
        print(f"Recovered latent hierarchy depth: {recovered_hierarchy_depth}")
        print(
            f"Latent-latent edges - True: {len(true_latent_edges)}, Recovered: {len(recovered_latent_edges)}"
        )

        return hierarchy_analysis

    def _compute_hierarchy_depth(
        self, graph: nx.DiGraph, latent_nodes: Set[str]
    ) -> int:
        """Compute the maximum depth of the latent variable hierarchy."""
        if not latent_nodes:
            return 0

        # Create subgraph with only latent nodes
        latent_subgraph = graph.subgraph(latent_nodes)

        if latent_subgraph.number_of_edges() == 0:
            return 1

        # Find longest path in latent hierarchy
        max_depth = 0
        for node in latent_subgraph.nodes():
            if latent_subgraph.in_degree(node) == 0:  # Root of hierarchy
                try:
                    depths = nx.single_source_shortest_path_length(
                        latent_subgraph, node
                    )
                    max_depth = max(max_depth, max(depths.values()) + 1)
                except:
                    max_depth = max(max_depth, 1)

        return max_depth if max_depth > 0 else 1

    def _compute_latent_edge_accuracy(
        self, true_edges: Set, recovered_edges: Set
    ) -> Dict[str, float]:
        """Compute precision, recall, F1 for latent-to-latent edges."""
        if not true_edges and not recovered_edges:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

        tp = len(true_edges & recovered_edges)
        fp = len(recovered_edges - true_edges)
        fn = len(true_edges - recovered_edges)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    def _analyze_numerical_stability(self) -> Dict[str, Any]:
        """Analyze numerical properties of the discrepancy matrix."""
        print("\n--- NUMERICAL STABILITY ANALYSIS ---")

        gamma_obs = self.polytree_sample["Gamma_obs"]

        # Compute numerical properties
        eigenvals = np.linalg.eigvals(gamma_obs)
        condition_number = np.linalg.cond(gamma_obs)
        matrix_rank = np.linalg.matrix_rank(gamma_obs)

        # Analyze value distribution
        gamma_flat = gamma_obs.flatten()
        gamma_nonzero = gamma_flat[np.abs(gamma_flat) > 1e-10]

        numerical_analysis = {
            "matrix_shape": gamma_obs.shape,
            "condition_number": float(condition_number),
            "matrix_rank": int(matrix_rank),
            "is_full_rank": matrix_rank == min(gamma_obs.shape),
            "min_eigenvalue": float(np.min(eigenvals.real)),
            "max_eigenvalue": float(np.max(eigenvals.real)),
            "eigenvalue_ratio": (
                float(np.max(eigenvals.real) / np.min(eigenvals.real))
                if np.min(eigenvals.real) > 1e-10
                else float("inf")
            ),
            "value_statistics": {
                "min_abs_nonzero": (
                    float(np.min(np.abs(gamma_nonzero)))
                    if len(gamma_nonzero) > 0
                    else 0.0
                ),
                "max_abs": float(np.max(np.abs(gamma_flat))),
                "mean_abs": float(np.mean(np.abs(gamma_flat))),
                "std_abs": float(np.std(np.abs(gamma_flat))),
                "dynamic_range": (
                    float(np.max(np.abs(gamma_flat)) / np.min(np.abs(gamma_nonzero)))
                    if len(gamma_nonzero) > 0
                    else float("inf")
                ),
            },
        }

        print(f"Matrix condition number: {condition_number:.2e}")
        print(
            f"Matrix rank: {matrix_rank}/{min(gamma_obs.shape)} ({'full rank' if numerical_analysis['is_full_rank'] else 'rank deficient'})"
        )
        print(
            f"Dynamic range: {numerical_analysis['value_statistics']['dynamic_range']:.2e}"
        )
        print(f"Eigenvalue ratio: {numerical_analysis['eigenvalue_ratio']:.2e}")

        # Identify potential numerical issues
        issues = []
        if condition_number > 1e12:
            issues.append("High condition number (ill-conditioned matrix)")
        if not numerical_analysis["is_full_rank"]:
            issues.append("Matrix is rank deficient")
        if numerical_analysis["value_statistics"]["dynamic_range"] > 1e10:
            issues.append("Extreme dynamic range in values")
        if numerical_analysis["eigenvalue_ratio"] > 1e10:
            issues.append("Large eigenvalue spread")

        numerical_analysis["potential_issues"] = issues

        if issues:
            print("Potential numerical issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No significant numerical issues detected")

        return numerical_analysis

    def _compute_structure_metrics(self) -> Dict[str, Any]:
        """Compute graph structure complexity metrics."""
        print("\n--- STRUCTURE COMPLEXITY METRICS ---")

        metrics = {
            "true_graph": self._graph_metrics(self.true_graph),
            "recovered_graph": self._graph_metrics(self.recovered_graph),
        }

        # Compare metrics
        print("Graph Complexity Comparison:")
        for metric in ["nodes", "edges", "avg_degree", "max_degree", "diameter"]:
            true_val = metrics["true_graph"].get(metric, "N/A")
            rec_val = metrics["recovered_graph"].get(metric, "N/A")
            print(f"  {metric}: True={true_val}, Recovered={rec_val}")

        return metrics

    def _graph_metrics(self, graph: nx.DiGraph) -> Dict[str, Union[int, float]]:
        """Compute metrics for a single graph."""
        if graph.number_of_nodes() == 0:
            return {}

        try:
            # Convert to undirected for some metrics
            undirected = graph.to_undirected()

            metrics = {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "avg_degree": float(np.mean([d for n, d in graph.degree()])),
                "max_degree": (
                    max([d for n, d in graph.degree()])
                    if graph.number_of_nodes() > 0
                    else 0
                ),
            }

            # Diameter (only if connected)
            try:
                if nx.is_connected(undirected):
                    metrics["diameter"] = nx.diameter(undirected)
                else:
                    # Use largest connected component
                    largest_cc = max(nx.connected_components(undirected), key=len)
                    if len(largest_cc) > 1:
                        subgraph = undirected.subgraph(largest_cc)
                        metrics["diameter"] = nx.diameter(subgraph)
                    else:
                        metrics["diameter"] = 0
            except:
                metrics["diameter"] = float("inf")

            return metrics
        except:
            return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}

    def _categorize_errors(self) -> Dict[str, Any]:
        """Categorize different types of errors."""
        print("\n--- ERROR CATEGORIZATION ---")

        # This would be expanded based on the specific error patterns observed
        error_categories = {
            "edge_recovery_errors": {
                "missing_direct_edges": 0,
                "spurious_edges": 0,
                "path_shortcuts": 0,
                "indirect_replacements": 0,
            },
            "latent_mapping_errors": {
                "unmatched_latents": 0,
                "incorrect_children_sets": 0,
                "hierarchy_depth_mismatch": 0,
            },
            "severity": "moderate",  # Could be 'low', 'moderate', 'high', 'critical'
        }

        # Count error types (simplified for now)
        observed_nodes = set(self.polytree_sample["observed_nodes"])
        true_obs_edges = {
            (u, v)
            for (u, v) in self.polytree_sample["edges_directed"]
            if u in observed_nodes and v in observed_nodes
        }
        recovered_obs_edges = {
            (u, v)
            for (u, v) in self.polytree_sample["recovered_edges"]
            if not u.startswith("h") and not v.startswith("h")
        }

        error_categories["edge_recovery_errors"]["missing_direct_edges"] = len(
            true_obs_edges - recovered_obs_edges
        )
        error_categories["edge_recovery_errors"]["spurious_edges"] = len(
            recovered_obs_edges - true_obs_edges
        )

        print(
            f"Missing direct edges: {error_categories['edge_recovery_errors']['missing_direct_edges']}"
        )
        print(
            f"Spurious edges: {error_categories['edge_recovery_errors']['spurious_edges']}"
        )

        return error_categories

    def _assess_recovery_quality(self) -> Dict[str, Any]:
        """Overall assessment of recovery quality."""
        print("\n--- RECOVERY QUALITY ASSESSMENT ---")

        # Compute overall quality score
        basic_success = len(
            set(self.polytree_sample["edges_directed"])
            & set(
                [
                    e
                    for e in self.polytree_sample["recovered_edges"]
                    if not e[0].startswith("h") or not e[1].startswith("h")
                ]
            )
        ) / max(len(self.polytree_sample["edges_directed"]), 1)

        quality_assessment = {
            "overall_score": basic_success,
            "quality_grade": self._compute_quality_grade(basic_success),
            "main_issues": [],
            "recommendations": [],
        }

        # Identify main issues and recommendations
        if basic_success < 0.9:
            quality_assessment["main_issues"].append("Significant edge recovery errors")
            quality_assessment["recommendations"].append(
                "Investigate discrepancy matrix numerical properties"
            )

        if self.n > 50 and basic_success < 0.95:
            quality_assessment["main_issues"].append(
                "Scalability challenges at large n"
            )
            quality_assessment["recommendations"].append(
                "Consider regularization or numerical stabilization"
            )

        print(
            f"Overall recovery quality: {quality_assessment['quality_grade']} ({basic_success:.1%})"
        )

        return quality_assessment

    def _compute_quality_grade(self, score: float) -> str:
        """Convert numerical score to quality grade."""
        if score >= 0.95:
            return "Excellent"
        elif score >= 0.85:
            return "Good"
        elif score >= 0.70:
            return "Fair"
        else:
            return "Poor"


def compare_scaling_performance(
    n_values: List[int] = [20, 50, 100], seed: int = 2025081314
) -> Dict[int, Dict[str, Any]]:
    """
    Compare performance across different scales to identify breakdown points.

    Args:
        n_values: List of polytree sizes to test
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping n values to their detailed analysis results
    """
    print("=" * 80)
    print("SCALING PERFORMANCE COMPARISON")
    print("=" * 80)

    results = {}

    for n in n_values:
        print(f"\n{'=' * 20} ANALYZING n={n} {'=' * 20}")

        try:
            debugger = AdvancedPolytreeDebugger(n=n, seed=seed)
            # Run analysis with minimal output for batch processing
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            analysis_results = debugger.run_advanced_analysis()

            sys.stdout = old_stdout

            # Extract key metrics
            summary = {
                "n": n,
                "success": analysis_results["recovery_analysis"]["is_superset"],
                "n_hidden": analysis_results["ground_truth"]["n_hidden"],
                "n_observed": analysis_results["ground_truth"]["n_observed"],
                "missing_edges": len(
                    analysis_results["recovery_analysis"]["missing_edges"]
                ),
                "extra_edges": len(
                    analysis_results["recovery_analysis"]["extra_edges"]
                ),
                "condition_number": analysis_results["numerical_analysis"][
                    "condition_number"
                ],
                "hierarchy_depth_true": analysis_results["hierarchy_analysis"][
                    "true_hierarchy_depth"
                ],
                "hierarchy_depth_recovered": analysis_results["hierarchy_analysis"][
                    "recovered_hierarchy_depth"
                ],
                "quality_score": analysis_results["recovery_quality"]["overall_score"],
                "quality_grade": analysis_results["recovery_quality"]["quality_grade"],
            }

            results[n] = {**analysis_results, "summary": summary}

            # Print summary
            status = "✓" if summary["success"] else "✗"
            print(
                f"n={n}: {status} | Quality: {summary['quality_grade']} | "
                f"Missing: {summary['missing_edges']} | Extra: {summary['extra_edges']} | "
                f"Condition: {summary['condition_number']:.1e}"
            )

        except Exception as e:
            print(f"n={n}: ERROR - {str(e)}")
            results[n] = {"error": str(e)}

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced polytree debugging with failure analysis"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of nodes in the polytree (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025081314,
        help="Random seed for reproducible results",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Run scaling comparison across multiple n values",
    )
    parser.add_argument(
        "--scaling-sizes",
        nargs="+",
        type=int,
        default=[20, 50, 100, 150],
        help="List of n values for scaling analysis",
    )

    args = parser.parse_args()

    if args.scaling:
        results = compare_scaling_performance(
            n_values=args.scaling_sizes, seed=args.seed
        )
        print("\n" + "=" * 80)
        print("SCALING ANALYSIS SUMMARY")
        print("=" * 80)
        for n in sorted(results.keys()):
            if "error" not in results[n]:
                summary = results[n]["summary"]
                print(
                    f"n={summary['n']:3d}: {summary['quality_grade']:>9} | "
                    f"Errors: {summary['missing_edges'] + summary['extra_edges']:2d} | "
                    f"Condition: {summary['condition_number']:.1e}"
                )
    else:
        debugger = AdvancedPolytreeDebugger(n=args.n, seed=args.seed)
        results = debugger.run_advanced_analysis()
