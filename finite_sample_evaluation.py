#!/usr/bin/env python3
"""
finite_sample_evaluation.py

Comprehensive finite-sample evaluation for the 4-node polytree example.
Compares finite-sample moment estimates and discrepancy matrices against population values
across different sample sizes.

Author: For TUM thesis on Linear non-Gaussian models with latent polytree structure
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polytree_discrepancy import (
    create_sample_configuration,
    generate_noise_samples,
    apply_lsem_transformation,
    finite_sample_discrepancy,
    population_discrepancy,
    topo_order_from_edges,
    compare_sample_vs_population_moments,
)


class FiniteSampleEvaluator:
    """
    Evaluates finite-sample performance for polytree discrepancy estimation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with polytree configuration."""
        if config is None:
            config = create_sample_configuration()

        self.nodes = config["nodes"]
        self.edges = config["edges"]
        self.gamma_shapes = config["gamma_shapes"]
        self.gamma_scales = config["gamma_scales"]
        self.seed = config.get("seed", 42)

        # Ensure topological ordering
        self.nodes = topo_order_from_edges(self.nodes, self.edges)

        # Compute population benchmarks
        self._compute_population_benchmarks()

    def _compute_population_benchmarks(self):
        """Compute population discrepancy matrix."""
        print("Computing population benchmarks...")

        # Population discrepancy matrix
        self.pop_discrepancy = population_discrepancy(
            self.edges, self.gamma_shapes, self.gamma_scales
        )

        print("✓ Population benchmarks computed")

    def evaluate_sample_size(self, n_samples: int, n_trials: int = 5) -> Dict[str, Any]:
        """
        Evaluate finite-sample performance for a given sample size.

        Args:
            n_samples: Number of samples to generate
            n_trials: Number of independent trials for statistical stability

        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\nEvaluating n_samples = {n_samples:,} over {n_trials} trials")

        results = {
            "n_samples": n_samples,
            "moment_deviations": [],
            "discrepancy_deviations": [],
            "variance_errors": [],
            "third_cumulant_errors": [],
            "max_discrepancy_errors": [],
        }

        for trial in range(n_trials):
            # Generate finite samples
            noise_samples = generate_noise_samples(
                self.nodes,
                self.gamma_shapes,
                self.gamma_scales,
                n_samples,
                self.seed + trial,  # Different seed per trial
            )

            X_samples = apply_lsem_transformation(noise_samples, self.edges, self.nodes)

            # Compute finite-sample moments and compare with population
            moment_comparison = compare_sample_vs_population_moments(
                X_samples, self.edges, self.gamma_shapes, self.gamma_scales, self.nodes
            )

            # Compute finite-sample discrepancy
            finite_discrepancy = finite_sample_discrepancy(X_samples, self.nodes)

            # Evaluate discrepancy deviations
            discrepancy_dev = np.max(np.abs(finite_discrepancy - self.pop_discrepancy))
            results["discrepancy_deviations"].append(discrepancy_dev)
            results["max_discrepancy_errors"].append(discrepancy_dev)

            # Store moment comparison results
            results["moment_deviations"].append(moment_comparison)
            results["variance_errors"].append(moment_comparison["variance_mae"])
            results["third_cumulant_errors"].append(
                moment_comparison["third_cumulant_mae"]
            )

        # Aggregate statistics
        for key in [
            "discrepancy_deviations",
            "variance_errors",
            "third_cumulant_errors",
            "max_discrepancy_errors",
        ]:
            values = results[key]
            results[f"{key}_mean"] = np.mean(values)
            results[f"{key}_std"] = np.std(values)
            results[f"{key}_min"] = np.min(values)
            results[f"{key}_max"] = np.max(values)

        return results

    def run_sample_size_analysis(
        self, sample_sizes: List[int], n_trials: int = 5
    ) -> pd.DataFrame:
        """
        Run comprehensive analysis across multiple sample sizes.

        Args:
            sample_sizes: List of sample sizes to evaluate
            n_trials: Number of trials per sample size

        Returns:
            DataFrame with results for all sample sizes
        """
        print("=" * 70)
        print("FINITE-SAMPLE EVALUATION: 4-NODE POLYTREE")
        print("=" * 70)
        print(f"Configuration: {self.edges}")
        print(f"Sample sizes: {sample_sizes}")
        print(f"Trials per size: {n_trials}")

        all_results = []

        for n_samples in sample_sizes:
            result = self.evaluate_sample_size(n_samples, n_trials)
            all_results.append(result)

        # Convert to DataFrame for easy analysis
        df_results = pd.DataFrame(all_results)

        return df_results

    def plot_convergence_analysis(
        self, results_df: pd.DataFrame, save_path: str = None
    ):
        """Plot convergence analysis across sample sizes."""

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Finite-Sample Convergence Analysis: 4-Node Polytree", fontsize=14)

        sample_sizes = results_df["n_samples"].values

        # Plot 1: Variance estimation error
        axes[0, 0].errorbar(
            sample_sizes,
            results_df["variance_errors_mean"],
            yerr=results_df["variance_errors_std"],
            marker="o",
            capsize=5,
        )
        axes[0, 0].set_xlabel("Sample Size")
        axes[0, 0].set_ylabel("Mean Absolute Error")
        axes[0, 0].set_title("Variance Estimation Error")
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_yscale("log")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Third cumulant estimation error
        axes[0, 1].errorbar(
            sample_sizes,
            results_df["third_cumulant_errors_mean"],
            yerr=results_df["third_cumulant_errors_std"],
            marker="s",
            capsize=5,
        )
        axes[0, 1].set_xlabel("Sample Size")
        axes[0, 1].set_ylabel("Mean Absolute Error")
        axes[0, 1].set_title("Third Cumulant Estimation Error")
        axes[0, 1].set_xscale("log")
        axes[0, 1].set_yscale("log")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Max discrepancy matrix error
        axes[1, 0].errorbar(
            sample_sizes,
            results_df["max_discrepancy_errors_mean"],
            yerr=results_df["max_discrepancy_errors_std"],
            marker="^",
            capsize=5,
        )
        axes[1, 0].set_xlabel("Sample Size")
        axes[1, 0].set_ylabel("Max Absolute Difference")
        axes[1, 0].set_title("Discrepancy Matrix Error")
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Theoretical convergence rate comparison
        axes[1, 1].loglog(
            sample_sizes,
            results_df["max_discrepancy_errors_mean"],
            "o-",
            label="Observed Error",
        )
        # Add theoretical n^(-1/2) line for comparison
        theoretical_rate = results_df["max_discrepancy_errors_mean"].iloc[0] * np.sqrt(
            sample_sizes[0] / sample_sizes
        )
        axes[1, 1].loglog(sample_sizes, theoretical_rate, "--", label="n^(-1/2) rate")
        axes[1, 1].set_xlabel("Sample Size")
        axes[1, 1].set_ylabel("Error")
        axes[1, 1].set_title("Convergence Rate Analysis")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def print_detailed_summary(self, results_df: pd.DataFrame):
        """Print detailed summary of results."""
        print("\n" + "=" * 70)
        print("DETAILED RESULTS SUMMARY")
        print("=" * 70)

        for _, row in results_df.iterrows():
            n = int(row["n_samples"])
            print(f"\nSample Size: {n:,}")
            print(
                f"  Variance Error:      {row['variance_errors_mean']:.6f} ± {row['variance_errors_std']:.6f}"
            )
            print(
                f"  Third Cumulant Error: {row['third_cumulant_errors_mean']:.6f} ± {row['third_cumulant_errors_std']:.6f}"
            )
            print(
                f"  Max Discrepancy Error: {row['max_discrepancy_errors_mean']:.6f} ± {row['max_discrepancy_errors_std']:.6f}"
            )

        # Convergence analysis
        if len(results_df) >= 2:
            print(f"\nConvergence Analysis:")
            initial_error = results_df["max_discrepancy_errors_mean"].iloc[0]
            final_error = results_df["max_discrepancy_errors_mean"].iloc[-1]
            initial_n = results_df["n_samples"].iloc[0]
            final_n = results_df["n_samples"].iloc[-1]

            improvement_ratio = initial_error / final_error
            sample_ratio = final_n / initial_n
            theoretical_improvement = np.sqrt(sample_ratio)

            print(f"  Error improvement: {improvement_ratio:.2f}x")
            print(f"  Sample size increase: {sample_ratio:.2f}x")
            print(f"  Theoretical (n^(-1/2)): {theoretical_improvement:.2f}x")
            print(f"  Efficiency: {improvement_ratio / theoretical_improvement:.2f}")


def main():
    """Main function to run finite-sample evaluation."""

    # Initialize evaluator
    evaluator = FiniteSampleEvaluator()

    # Define sample sizes to test
    sample_sizes = [100, 1000, 10000, 100000, 1000000, 10000000]

    # Run analysis
    results = evaluator.run_sample_size_analysis(sample_sizes, n_trials=10)

    # Print summary
    evaluator.print_detailed_summary(results)

    # Generate plots
    evaluator.plot_convergence_analysis(results, "finite_sample_convergence.png")

    # Save results
    results.to_csv("finite_sample_results.csv", index=False)
    print(f"\nResults saved to finite_sample_results.csv")

    return results


if __name__ == "__main__":
    results = main()
