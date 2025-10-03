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

    def _generate_trial_samples(self, n_samples: int, trial: int) -> np.ndarray:
        """Generate samples for a single trial."""
        # Generate finite samples
        noise_samples = generate_noise_samples(
            self.nodes,
            self.gamma_shapes,
            self.gamma_scales,
            n_samples,
            self.seed + trial,  # Different seed per trial
        )

        X_samples = apply_lsem_transformation(noise_samples, self.edges, self.nodes)
        return X_samples

    def _compute_trial_metrics(self, X_samples: np.ndarray) -> Dict[str, Any]:
        """Compute all metrics for a single trial."""
        # Compute finite-sample moments and compare with population
        moment_comparison = compare_sample_vs_population_moments(
            X_samples, self.edges, self.gamma_shapes, self.gamma_scales, self.nodes
        )

        # Compute finite-sample discrepancy
        finite_discrepancy = finite_sample_discrepancy(X_samples, self.nodes)

        # Evaluate discrepancy deviations
        discrepancy_dev = np.max(np.abs(finite_discrepancy - self.pop_discrepancy))

        return {
            "moment_comparison": moment_comparison,
            "finite_discrepancy": finite_discrepancy,
            "discrepancy_deviation": discrepancy_dev,
        }

    def _aggregate_trial_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate statistics across all trials."""
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
        results["structure_recovery_success_rate"] = success_count / len(
            results["structure_recovery_results"]
        )

        return results

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
            # Generate samples for this trial
            X_samples = self._generate_trial_samples(n_samples, trial)

            # Compute trial metrics
            trial_metrics = self._compute_trial_metrics(X_samples)

            # Store results
            results["discrepancy_deviations"].append(
                trial_metrics["discrepancy_deviation"]
            )
            results["max_discrepancy_errors"].append(
                trial_metrics["discrepancy_deviation"]
            )
            results["moment_deviations"].append(trial_metrics["moment_comparison"])
            results["variance_errors"].append(
                trial_metrics["moment_comparison"]["variance_mae"]
            )
            results["third_cumulant_errors"].append(
                trial_metrics["moment_comparison"]["third_cumulant_mae"]
            )

            # Structure recovery validation
            structure_result = self._validate_structure_recovery(X_samples, trial == 0)
            results["structure_recovery_results"].append(structure_result)

        # Aggregate statistics
        results = self._aggregate_trial_results(results)

        return results

    def _extract_observed_discrepancy(self, full_discrepancy: np.ndarray) -> np.ndarray:
        """Extract observed-only discrepancy matrix from full discrepancy matrix."""
        node_to_index = {node: i for i, node in enumerate(self.nodes)}
        observed_indices = [node_to_index[node] for node in self.observed_nodes]
        observed_discrepancy = full_discrepancy[
            np.ix_(observed_indices, observed_indices)
        ]
        return observed_discrepancy

    def _get_ground_truth_edges(self) -> List[tuple]:
        """Get ground truth observed edges."""
        true_observed_edges = []
        for (parent, child), weight in self.edges.items():
            if parent in self.observed_nodes and child in self.observed_nodes:
                true_observed_edges.append((parent, child))
        return true_observed_edges

    def _map_recovered_edges(self, recovered_structure) -> List[tuple]:
        """Map recovered edge indices back to node names."""

        def map_name(x: str) -> str:
            if x.startswith("h"):  # latent node from algorithm
                return x
            return self.observed_nodes[int(x)]  # observed node

        recovered_edges = [
            (map_name(parent), map_name(child))
            for (parent, child) in recovered_structure.edges
        ]
        return recovered_edges

    def _check_recovery_success(
        self, recovered_edges: List[tuple], true_edges: List[tuple]
    ) -> bool:
        """Check if structure recovery was successful."""
        observed_recovered_edges = [
            edge
            for edge in recovered_edges
            if not (edge[0].startswith("h") or edge[1].startswith("h"))
        ]
        return set(true_edges) == set(observed_recovered_edges)

    def _validate_structure_recovery(
        self, X_samples: Dict[str, np.ndarray], verbose: bool = True
    ) -> Dict[str, Any]:
        """Validate structure recovery using existing learn_with_hidden infrastructure."""

        try:
            # Step 1: Compute finite-sample discrepancy matrix (all nodes)
            full_discrepancy = finite_sample_discrepancy(X_samples, self.nodes)

            # Step 2: Extract observed-only discrepancy matrix
            observed_discrepancy = self._extract_observed_discrepancy(full_discrepancy)

            if verbose:
                print(f"Structure Recovery Validation:")
                print(f"  Hidden nodes: {list(self.hidden_nodes)}")
                print(f"  Observed nodes: {self.observed_nodes}")
                print(f"  Observed discrepancy shape: {observed_discrepancy.shape}")

            # Step 3: Run structure recovery on observed discrepancy
            recovered_structure = get_polytree_algo3(observed_discrepancy)

            # Step 4: Map recovered edges back to node names
            recovered_edges = self._map_recovered_edges(recovered_structure)

            # Step 5: Get ground truth observed edges
            true_observed_edges = self._get_ground_truth_edges()

            if verbose:
                print(f"  True observed edges: {true_observed_edges}")
                print(f"  Recovered edges: {recovered_edges}")

            # Check if recovery was successful
            recovery_success = self._check_recovery_success(
                recovered_edges, true_observed_edges
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
        self._print_analysis_header(sample_sizes, n_trials)

        all_results = []
        for n_samples in sample_sizes:
            result = self.evaluate_sample_size(n_samples, n_trials)
            all_results.append(result)

        # Convert to DataFrame for easy analysis
        df_results = pd.DataFrame(all_results)
        return df_results

    def _print_analysis_header(self, sample_sizes: List[int], n_trials: int):
        """Print analysis header information."""
        print("=" * 70)
        print("FINITE-SAMPLE EVALUATION: 4-NODE POLYTREE")
        print("=" * 70)
        print(f"Configuration: {self.edges}")
        print(f"Sample sizes: {sample_sizes}")
        print(f"Trials per size: {n_trials}")

    def _plot_variance_error(
        self, ax, sample_sizes: np.ndarray, results_df: pd.DataFrame
    ):
        """Plot variance estimation error."""
        ax.errorbar(
            sample_sizes,
            results_df["variance_errors_mean"],
            yerr=results_df["variance_errors_std"],
            marker="o",
            capsize=5,
        )
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Mean Absolute Error")
        ax.set_title("Variance Estimation Error")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    def _plot_cumulant_error(
        self, ax, sample_sizes: np.ndarray, results_df: pd.DataFrame
    ):
        """Plot third cumulant estimation error."""
        ax.errorbar(
            sample_sizes,
            results_df["third_cumulant_errors_mean"],
            yerr=results_df["third_cumulant_errors_std"],
            marker="s",
            capsize=5,
        )
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Mean Absolute Error")
        ax.set_title("Third Cumulant Estimation Error")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    def _plot_discrepancy_error(
        self, ax, sample_sizes: np.ndarray, results_df: pd.DataFrame
    ):
        """Plot max discrepancy matrix error."""
        ax.errorbar(
            sample_sizes,
            results_df["max_discrepancy_errors_mean"],
            yerr=results_df["max_discrepancy_errors_std"],
            marker="^",
            capsize=5,
        )
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Max Absolute Difference")
        ax.set_title("Discrepancy Matrix Error")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    def _plot_convergence_rate(
        self, ax, sample_sizes: np.ndarray, results_df: pd.DataFrame
    ):
        """Plot theoretical convergence rate comparison."""
        ax.loglog(
            sample_sizes,
            results_df["max_discrepancy_errors_mean"],
            "o-",
            label="Observed Error",
        )
        # Add theoretical n^(-1/2) line for comparison
        theoretical_rate = results_df["max_discrepancy_errors_mean"].iloc[0] * np.sqrt(
            sample_sizes[0] / sample_sizes
        )
        ax.loglog(sample_sizes, theoretical_rate, "--", label="n^(-1/2) rate")
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Error")
        ax.set_title("Convergence Rate Analysis")
        ax.legend()
        ax.grid(True, alpha=0.3)

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

        # Plot individual error components
        self._plot_variance_error(axes[0, 0], sample_sizes, results_df)
        self._plot_cumulant_error(axes[0, 1], sample_sizes, results_df)
        self._plot_discrepancy_error(axes[1, 0], sample_sizes, results_df)
        self._plot_convergence_rate(axes[1, 1], sample_sizes, results_df)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Moment convergence plot saved to {save_path}")

        plt.show()

    def _plot_success_rate(
        self, ax, sample_sizes: np.ndarray, success_rates: np.ndarray
    ):
        """Plot structure recovery success rate."""
        ax.plot(sample_sizes, success_rates, "ro-", linewidth=2, markersize=8)
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Structure Recovery Success Rate")
        ax.set_xscale("log")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)

        # Add reference lines
        ax.axhline(
            y=100, color="green", linestyle="--", alpha=0.7, label="Perfect Recovery"
        )
        ax.axhline(
            y=90, color="orange", linestyle="--", alpha=0.7, label="90% Threshold"
        )
        ax.axhline(y=50, color="red", linestyle="--", alpha=0.7, label="50% Threshold")
        ax.legend()

    def _add_transition_annotations(
        self, ax, sample_sizes: np.ndarray, success_rates: np.ndarray
    ):
        """Add annotations for key transition points."""
        for i, (n, rate) in enumerate(zip(sample_sizes, success_rates)):
            if rate >= 90 and i > 0 and success_rates[i - 1] < 90:
                ax.annotate(
                    f"90% at n={n:,}",
                    xy=(n, rate),
                    xytext=(n * 0.3, rate + 10),
                    arrowprops=dict(arrowstyle="->", color="orange", alpha=0.7),
                    fontsize=10,
                    ha="center",
                )
            elif rate >= 100 and i > 0 and success_rates[i - 1] < 100:
                ax.annotate(
                    f"100% at n={n:,}",
                    xy=(n, rate),
                    xytext=(n * 0.3, rate - 15),
                    arrowprops=dict(arrowstyle="->", color="green", alpha=0.7),
                    fontsize=10,
                    ha="center",
                )

    def _plot_combined_metrics(
        self,
        ax,
        ax_twin,
        sample_sizes: np.ndarray,
        success_rates: np.ndarray,
        results_df: pd.DataFrame,
    ):
        """Plot combined success rate vs discrepancy error."""
        # Success rate (left axis)
        line1 = ax.plot(
            sample_sizes,
            success_rates,
            "ro-",
            linewidth=2,
            markersize=6,
            label="Success Rate (%)",
        )
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Success Rate (%)", color="red")
        ax.set_xscale("log")
        ax.set_ylim(-5, 105)
        ax.tick_params(axis="y", labelcolor="red")

        # Discrepancy error (right axis)
        line2 = ax_twin.loglog(
            sample_sizes,
            results_df["max_discrepancy_errors_mean"],
            "b^-",
            linewidth=2,
            markersize=6,
            label="Discrepancy Error",
        )
        ax_twin.set_ylabel("Max Discrepancy Error", color="blue")
        ax_twin.tick_params(axis="y", labelcolor="blue")

        ax.set_title("Success Rate vs Discrepancy Error")
        ax.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="center right")

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
        self._plot_success_rate(axes[0], sample_sizes, success_rates)
        self._add_transition_annotations(axes[0], sample_sizes, success_rates)

        # Plot 2: Combined Success Rate vs Discrepancy Error
        ax2_twin = axes[1].twinx()
        self._plot_combined_metrics(
            axes[1], ax2_twin, sample_sizes, success_rates, results_df
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Structure recovery plot saved to {save_path}")

        plt.show()

    def _print_summary_header(self, results_df: pd.DataFrame, n_trials: int):
        """Print summary header information."""
        print("\n" + "=" * 70)
        print("DETAILED RESULTS SUMMARY")
        print("=" * 70)
        print(
            f"Node classification: {results_df.iloc[0]['hidden_nodes']} (hidden), {results_df.iloc[0]['observed_nodes']} (observed)"
        )
        print(f"Trials per sample size: {n_trials}")

    def _print_sample_size_results(self, results_df: pd.DataFrame, n_trials: int):
        """Print results for each sample size."""
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

    def _analyze_transition_points(self, results_df: pd.DataFrame):
        """Analyze and print structure recovery transition points."""
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

    def _print_practical_recommendation(self, results_df: pd.DataFrame):
        """Print practical recommendation for sample size."""
        reliable_samples = results_df[
            results_df["structure_recovery_success_rate"] >= 0.9
        ]
        if not reliable_samples.empty:
            min_reliable = reliable_samples["n_samples"].min()
            print(
                f"\nPractical Recommendation: n ≥ {min_reliable:,} for reliable structure recovery (≥90% success)"
            )

    def _analyze_convergence(self, results_df: pd.DataFrame):
        """Analyze and print convergence information."""
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

    def print_detailed_summary(self, results_df: pd.DataFrame, n_trials: int = 5):
        """Print detailed summary of results."""
        self._print_summary_header(results_df, n_trials)
        self._print_sample_size_results(results_df, n_trials)
        self._analyze_transition_points(results_df)
        self._print_practical_recommendation(results_df)
        self._analyze_convergence(results_df)


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
