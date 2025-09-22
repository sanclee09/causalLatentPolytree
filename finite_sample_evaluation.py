from __future__ import annotations
from typing import Dict, List, Any
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
from learn_with_hidden import detect_learnable_nodes
from latent_polytree_truepoly import get_polytree_algo3


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

        # Identify latent vs observed nodes using existing infrastructure
        self.hidden_nodes = detect_learnable_nodes(self.edges, min_outdegree=2)
        self.observed_nodes = [n for n in self.nodes if n not in self.hidden_nodes]

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
            "hidden_nodes": list(self.hidden_nodes),
            "observed_nodes": self.observed_nodes,
            "moment_deviations": [],
            "discrepancy_deviations": [],
            "variance_errors": [],
            "third_cumulant_errors": [],
            "max_discrepancy_errors": [],
            "structure_recovery_results": [],
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

            # Structure recovery validation using existing infrastructure
            structure_result = self._validate_structure_recovery(X_samples, trial == 0)
            results["structure_recovery_results"].append(structure_result)

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

        # Structure recovery success rate
        success_count = sum(
            1
            for r in results["structure_recovery_results"]
            if r.get("recovery_success", False)
        )
        results["structure_recovery_success_rate"] = success_count / n_trials

        return results

    def _validate_structure_recovery(
        self, X_samples: Dict[str, np.ndarray], verbose: bool = True
    ) -> Dict[str, Any]:
        """Validate structure recovery using existing learn_with_hidden infrastructure."""

        try:
            # Step 1: Compute finite-sample discrepancy matrix (all nodes)
            full_discrepancy = finite_sample_discrepancy(X_samples, self.nodes)

            # Step 2: Extract observed-only discrepancy matrix using existing method
            # Note: We need sigmas/kappas for the existing method, but we only need the extraction logic
            node_to_index = {node: i for i, node in enumerate(self.nodes)}
            observed_indices = [node_to_index[node] for node in self.observed_nodes]
            observed_discrepancy = full_discrepancy[
                np.ix_(observed_indices, observed_indices)
            ]

            if verbose:
                print(f"Structure Recovery Validation:")
                print(f"  Hidden nodes: {list(self.hidden_nodes)}")
                print(f"  Observed nodes: {self.observed_nodes}")
                print(f"  Observed discrepancy shape: {observed_discrepancy.shape}")

            # Step 3: Run structure recovery on observed discrepancy
            recovered_structure = get_polytree_algo3(observed_discrepancy)

            # Step 4: Extract recovered edges and map back to node names
            def map_name(x: str) -> str:
                if x.startswith("h"):  # latent node from algorithm
                    return x
                return self.observed_nodes[int(x)]  # observed node

            recovered_edges = [
                (map_name(parent), map_name(child))
                for (parent, child) in recovered_structure.edges
            ]

            # Step 5: Determine ground truth observed edges
            true_observed_edges = []
            for (parent, child), weight in self.edges.items():
                if parent in self.observed_nodes and child in self.observed_nodes:
                    true_observed_edges.append((parent, child))

            if verbose:
                print(f"  True observed edges: {true_observed_edges}")
                print(f"  Recovered edges: {recovered_edges}")

            # Check if recovery was successful
            recovery_success = set(true_observed_edges) == set(
                edge
                for edge in recovered_edges
                if not (edge[0].startswith("h") or edge[1].startswith("h"))
            )

            return {
                "recovery_success": recovery_success,
                "recovered_edges": recovered_edges,
                "true_observed_edges": true_observed_edges,
                "observed_discrepancy": observed_discrepancy,
                "error": None,
            }

        except Exception as e:
            if verbose:
                print(f"  Structure recovery failed: {e}")
            return {
                "recovery_success": False,
                "recovered_edges": [],
                "true_observed_edges": [],
                "observed_discrepancy": None,
                "error": str(e),
            }

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
        """Plot moment and discrepancy convergence analysis across sample sizes."""

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            "Finite-Sample Moment and Discrepancy Convergence: 4-Node Polytree",
            fontsize=14,
        )

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
            print(f"Moment convergence plot saved to {save_path}")

        plt.show()

    def plot_structure_recovery_analysis(
        self, results_df: pd.DataFrame, save_path: str = None
    ):
        """Plot structure recovery success rate analysis."""

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            "Structure Recovery Performance Analysis: 4-Node Polytree", fontsize=14
        )

        sample_sizes = results_df["n_samples"].values
        success_rates = results_df["structure_recovery_success_rate"] * 100

        # Plot 1: Structure Recovery Success Rate
        axes[0].plot(sample_sizes, success_rates, "ro-", linewidth=2, markersize=8)
        axes[0].set_xlabel("Sample Size")
        axes[0].set_ylabel("Success Rate (%)")
        axes[0].set_title("Structure Recovery Success Rate")
        axes[0].set_xscale("log")
        axes[0].set_ylim(-5, 105)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(
            y=100, color="green", linestyle="--", alpha=0.7, label="Perfect Recovery"
        )
        axes[0].axhline(
            y=90, color="orange", linestyle="--", alpha=0.7, label="90% Threshold"
        )
        axes[0].axhline(
            y=50, color="red", linestyle="--", alpha=0.7, label="50% Threshold"
        )
        axes[0].legend()

        # Add annotations for key transition points
        for i, (n, rate) in enumerate(zip(sample_sizes, success_rates)):
            if rate >= 90 and i > 0 and success_rates[i - 1] < 90:
                axes[0].annotate(
                    f"90% at n={n:,}",
                    xy=(n, rate),
                    xytext=(n * 0.3, rate + 10),
                    arrowprops=dict(arrowstyle="->", color="orange", alpha=0.7),
                    fontsize=10,
                    ha="center",
                )
            elif rate >= 100 and i > 0 and success_rates[i - 1] < 100:
                axes[0].annotate(
                    f"100% at n={n:,}",
                    xy=(n, rate),
                    xytext=(n * 0.3, rate - 15),
                    arrowprops=dict(arrowstyle="->", color="green", alpha=0.7),
                    fontsize=10,
                    ha="center",
                )

        # Plot 2: Combined Success Rate vs Discrepancy Error
        ax2 = axes[1]
        ax2_twin = ax2.twinx()

        # Success rate (left axis)
        line1 = ax2.plot(
            sample_sizes,
            success_rates,
            "ro-",
            linewidth=2,
            markersize=6,
            label="Success Rate (%)",
        )
        ax2.set_xlabel("Sample Size")
        ax2.set_ylabel("Success Rate (%)", color="red")
        ax2.set_xscale("log")
        ax2.set_ylim(-5, 105)
        ax2.tick_params(axis="y", labelcolor="red")

        # Discrepancy error (right axis)
        line2 = ax2_twin.loglog(
            sample_sizes,
            results_df["max_discrepancy_errors_mean"],
            "b^-",
            linewidth=2,
            markersize=6,
            label="Discrepancy Error",
        )
        ax2_twin.set_ylabel("Max Discrepancy Error", color="blue")
        ax2_twin.tick_params(axis="y", labelcolor="blue")

        ax2.set_title("Success Rate vs Discrepancy Error")
        ax2.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="center right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Structure recovery plot saved to {save_path}")

        plt.show()

    def print_detailed_summary(self, results_df: pd.DataFrame, n_trials: int = 5):
        """Print detailed summary of results."""
        print("\n" + "=" * 70)
        print("DETAILED RESULTS SUMMARY")
        print("=" * 70)
        print(
            f"Node classification: {results_df.iloc[0]['hidden_nodes']} (hidden), {results_df.iloc[0]['observed_nodes']} (observed)"
        )
        print(f"Trials per sample size: {n_trials}")

        for _, row in results_df.iterrows():
            n = int(row["n_samples"])
            success_rate = row.get("structure_recovery_success_rate", 0)
            print(f"\nSample Size: {n:,}")
            print(
                f"  Variance Error:       {row['variance_errors_mean']:.6f} ± {row['variance_errors_std']:.6f}"
            )
            print(
                f"  Third Cumulant Error: {row['third_cumulant_errors_mean']:.6f} ± {row['third_cumulant_errors_std']:.6f}"
            )
            print(
                f"  Max Discrepancy Error: {row['max_discrepancy_errors_mean']:.6f} ± {row['max_discrepancy_errors_std']:.6f}"
            )
            print(
                f"  Structure Recovery:   {success_rate:.1%} success rate ({int(success_rate * n_trials)}/{n_trials} trials)"
            )

        # Structure recovery transition analysis
        success_rates = results_df["structure_recovery_success_rate"].values
        transition_info = []
        for i, rate in enumerate(success_rates):
            if rate >= 1.0:
                transition_info.append(
                    f"n≥{results_df.iloc[i]['n_samples']:,}: Perfect recovery"
                )
            elif rate >= 0.9:
                transition_info.append(
                    f"n≥{results_df.iloc[i]['n_samples']:,}: High reliability (≥90%)"
                )
            elif rate >= 0.5:
                transition_info.append(
                    f"n≥{results_df.iloc[i]['n_samples']:,}: Moderate reliability (≥50%)"
                )

        if transition_info:
            print(f"\nStructure Recovery Transition Points:")
            for info in transition_info[:3]:  # Show first 3 transition points
                print(f"  {info}")

        # Find sample size for reliable recovery (≥90%)
        reliable_samples = results_df[
            results_df["structure_recovery_success_rate"] >= 0.9
        ]
        if not reliable_samples.empty:
            min_reliable = reliable_samples["n_samples"].min()
            print(
                f"\nPractical Recommendation: n ≥ {min_reliable:,} for reliable structure recovery (≥90% success)"
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
    sample_sizes = [100, 1000, 10000, 100000, 500000, 1000000, 10000000]
    n_trials = 20  # Number of trials per sample size

    # Run analysis
    results = evaluator.run_sample_size_analysis(sample_sizes, n_trials)

    # Print summary
    evaluator.print_detailed_summary(results, n_trials)

    # Generate separate plots
    evaluator.plot_convergence_analysis(results, "finite_sample_moment_convergence.png")
    evaluator.plot_structure_recovery_analysis(
        results, "finite_sample_structure_recovery.png"
    )

    # Save results
    results.to_csv("finite_sample_results.csv", index=False)
    print(f"\nResults saved to finite_sample_results.csv")

    return results


if __name__ == "__main__":
    results = main()
