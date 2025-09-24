#!/usr/bin/env python3
"""
working_extended_finite_sample_evaluation.py

Working version that bypasses the structure_learning_check issue
and extracts the discrepancy matrix directly.
"""

from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random_polytrees_pruefer import get_random_polytree_via_pruefer
from polytree_discrepancy import (
    generate_noise_samples,
    apply_lsem_transformation,
    finite_sample_discrepancy,
    population_discrepancy,
    topo_order_from_edges,
)
from latent_polytree_truepoly import get_polytree_algo3


def run_finite_sample_for_random_polytree_working(
    n_nodes: int, sample_sizes: List[int], n_trials: int = 3, seed: int = 42
) -> Dict[str, Any]:
    """
    Working version that bypasses the structure_learning_check issue.
    """
    print(f"\n{'=' * 60}")
    print(f"FINITE-SAMPLE EVALUATION: RANDOM POLYTREE (n={n_nodes}) - WORKING")
    print(f"{'=' * 60}")

    # Generate random polytree
    polytree_sample = get_random_polytree_via_pruefer(
        n=n_nodes,
        seed=seed,
        weights_range=(-1.0, 1.0),
        avoid_small=0.8,
        ensure_at_least_one_hidden=True,
    )

    print(f"Generated polytree:")
    print(f"  Total nodes: {n_nodes}")
    print(
        f"  Hidden nodes ({len(polytree_sample['hidden_nodes'])}): {polytree_sample['hidden_nodes']}"
    )
    print(
        f"  Observed nodes ({len(polytree_sample['observed_nodes'])}): {polytree_sample['observed_nodes']}"
    )

    # Population discrepancy matrix (ground truth) - this is for observed nodes only
    Gamma_population_obs = polytree_sample["Gamma_obs"]
    print(f"  Population discrepancy shape: {Gamma_population_obs.shape}")

    # Prepare data for existing functions
    edges_dict = polytree_sample["weights"]
    all_nodes = sorted(set(sum(polytree_sample["edges_directed"], ())))
    all_nodes = topo_order_from_edges(all_nodes, edges_dict)
    observed_nodes = polytree_sample["observed_nodes"]

    # Create gamma parameters (using same as your function)
    gamma_shapes = {node: 2.5 for node in all_nodes}
    gamma_scales = {node: 1.0 for node in all_nodes}

    results = []

    for n_samples in sample_sizes:
        print(f"\nTesting n_samples = {n_samples:,}...")

        sample_results = {
            "n_nodes": n_nodes,
            "n_samples": n_samples,
            "n_hidden": len(polytree_sample["hidden_nodes"]),
            "n_observed": len(polytree_sample["observed_nodes"]),
            "discrepancy_errors": [],
            "structure_recovery_success": [],
        }

        for trial in range(n_trials):
            try:
                # Use your existing functions directly (bypassing run_polytree_discrepancy_for_random_tree)
                noise_samples = generate_noise_samples(
                    all_nodes,
                    gamma_shapes,
                    gamma_scales,
                    n_samples,
                    seed + trial * 1000,
                )

                X_samples = apply_lsem_transformation(
                    noise_samples, edges_dict, all_nodes
                )

                # Compute finite-sample discrepancy using your function
                Gamma_finite_full = finite_sample_discrepancy(X_samples, all_nodes)

                # Extract observed-only discrepancy matrix
                observed_indices = [
                    all_nodes.index(node)
                    for node in observed_nodes
                    if node in all_nodes
                ]
                Gamma_finite_obs = Gamma_finite_full[
                    np.ix_(observed_indices, observed_indices)
                ]

                # Compare with population discrepancy (observed nodes only)
                discrepancy_error = np.max(
                    np.abs(Gamma_finite_obs - Gamma_population_obs)
                )
                sample_results["discrepancy_errors"].append(discrepancy_error)

                # Test structure recovery on observed discrepancy matrix
                try:
                    recovered_polytree = get_polytree_algo3(Gamma_finite_obs)
                    recovered_edges = set(recovered_polytree.edges)
                    expected_edges = set(polytree_sample["recovered_edges"])

                    success = (
                        len(recovered_edges.symmetric_difference(expected_edges)) <= 1
                    )
                    sample_results["structure_recovery_success"].append(success)

                    if trial == 0:  # Print details for first trial
                        print(
                            f"    Trial {trial}: error={discrepancy_error:.6f}, success={success}"
                        )

                except Exception as e:
                    print(f"    Structure recovery failed in trial {trial}: {e}")
                    sample_results["structure_recovery_success"].append(False)

            except Exception as e:
                print(f"    Trial {trial} failed: {e}")
                sample_results["discrepancy_errors"].append(np.inf)
                sample_results["structure_recovery_success"].append(False)

        # Compute summary statistics
        valid_errors = [
            e for e in sample_results["discrepancy_errors"] if np.isfinite(e)
        ]

        if valid_errors:
            sample_results["discrepancy_error_mean"] = np.mean(valid_errors)
            sample_results["discrepancy_error_std"] = np.std(valid_errors)
        else:
            sample_results["discrepancy_error_mean"] = np.inf
            sample_results["discrepancy_error_std"] = np.inf

        sample_results["structure_success_rate"] = np.mean(
            sample_results["structure_recovery_success"]
        )

        print(
            f"  Discrepancy error: {sample_results['discrepancy_error_mean']:.6f} ± {sample_results['discrepancy_error_std']:.6f}"
        )
        print(
            f"  Structure recovery rate: {sample_results['structure_success_rate']:.3f}"
        )

        results.append(sample_results)

    return {
        "polytree_info": {
            "n_nodes": n_nodes,
            "n_hidden": len(polytree_sample["hidden_nodes"]),
            "n_observed": len(polytree_sample["observed_nodes"]),
            "edges_directed": polytree_sample["edges_directed"],
            "hidden_nodes": polytree_sample["hidden_nodes"],
            "observed_nodes": polytree_sample["observed_nodes"],
        },
        "results": results,
    }


def plot_convergence_analysis(all_results: Dict[int, Dict[str, Any]]):
    """Plot convergence analysis showing n^(-1/2) behavior."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Finite-Sample Analysis: Random Polytrees (Working Version)", fontsize=14
    )

    colors = ["blue", "red", "green", "orange", "purple"]

    for i, (n_nodes, data) in enumerate(all_results.items()):
        color = colors[i % len(colors)]
        info = data["polytree_info"]
        results = data["results"]

        label = f"n={n_nodes} ({info['n_hidden']}H, {info['n_observed']}O)"

        sample_sizes = [r["n_samples"] for r in results]
        error_means = [r["discrepancy_error_mean"] for r in results]
        error_stds = [r["discrepancy_error_std"] for r in results]
        success_rates = [r["structure_success_rate"] for r in results]

        # Filter out infinite errors
        finite_mask = [np.isfinite(e) for e in error_means]
        plot_sample_sizes = [s for s, mask in zip(sample_sizes, finite_mask) if mask]
        plot_error_means = [e for e, mask in zip(error_means, finite_mask) if mask]
        plot_error_stds = [e for e, mask in zip(error_stds, finite_mask) if mask]

        if plot_sample_sizes:
            # Error convergence
            axes[0, 0].loglog(
                plot_sample_sizes, plot_error_means, "o-", color=color, label=label
            )
            axes[0, 0].fill_between(
                plot_sample_sizes,
                np.array(plot_error_means) - np.array(plot_error_stds),
                np.array(plot_error_means) + np.array(plot_error_stds),
                alpha=0.2,
                color=color,
            )

            # Theoretical n^(-1/2) comparison
            if len(plot_sample_sizes) > 1:
                theoretical = plot_error_means[0] * np.sqrt(
                    plot_sample_sizes[0] / np.array(plot_sample_sizes)
                )
                axes[1, 0].loglog(
                    plot_sample_sizes,
                    theoretical,
                    "--",
                    color=color,
                    alpha=0.5,
                    label=f"{label} Theory",
                )
                axes[1, 0].loglog(
                    plot_sample_sizes,
                    plot_error_means,
                    "o-",
                    color=color,
                    label=f"{label} Observed",
                )

        # Structure recovery
        axes[0, 1].semilogx(sample_sizes, success_rates, "o-", color=color, label=label)

    # Convergence efficiency analysis
    for n_nodes, data in all_results.items():
        results = data["results"]
        valid_results = [r for r in results if np.isfinite(r["discrepancy_error_mean"])]

        if len(valid_results) >= 2:
            first_n = valid_results[0]["n_samples"]
            first_error = valid_results[0]["discrepancy_error_mean"]
            last_n = valid_results[-1]["n_samples"]
            last_error = valid_results[-1]["discrepancy_error_mean"]

            if last_error > 0:
                sample_ratio = last_n / first_n
                error_ratio = first_error / last_error
                theoretical_ratio = np.sqrt(sample_ratio)
                efficiency = error_ratio / theoretical_ratio

                axes[1, 1].scatter([n_nodes], [efficiency], s=100, alpha=0.7)
                axes[1, 1].annotate(
                    f"n={n_nodes}\nEff={efficiency:.2f}",
                    (n_nodes, efficiency),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    # Formatting
    axes[0, 0].set_xlabel("Sample Size")
    axes[0, 0].set_ylabel("Max Discrepancy Error")
    axes[0, 0].set_title("Error Convergence")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Sample Size")
    axes[0, 1].set_ylabel("Structure Recovery Success Rate")
    axes[0, 1].set_title("Structure Recovery Performance")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.1)

    axes[1, 0].set_xlabel("Sample Size")
    axes[1, 0].set_ylabel("Discrepancy Error")
    axes[1, 0].set_title("Observed vs Theoretical n^(-1/2) Convergence")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Polytree Size")
    axes[1, 1].set_ylabel("Convergence Efficiency")
    axes[1, 1].set_title("n^(-1/2) Efficiency")
    axes[1, 1].axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Perfect")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("working_random_polytree_analysis.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("working_random_polytree_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_convergence_summary(all_results: Dict[int, Dict[str, Any]]):
    """Print detailed convergence analysis."""
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("=" * 70)

    for n_nodes, data in all_results.items():
        info = data["polytree_info"]
        results = data["results"]

        print(
            f"\nPolytree n={n_nodes} ({info['n_hidden']} hidden, {info['n_observed']} observed):"
        )
        print("-" * 50)

        valid_results = []
        for result in results:
            error_mean = result["discrepancy_error_mean"]
            error_std = result["discrepancy_error_std"]
            success_rate = result["structure_success_rate"]

            if np.isfinite(error_mean):
                print(
                    f"  n_samples={result['n_samples']:,}: error={error_mean:.6f}±{error_std:.6f}, recovery={success_rate:.3f}"
                )
                valid_results.append((result["n_samples"], error_mean))
            else:
                print(
                    f"  n_samples={result['n_samples']:,}: error=FAILED, recovery={success_rate:.3f}"
                )

        # Convergence analysis
        if len(valid_results) >= 2:
            print(f"\n  Convergence Analysis:")
            first_n, first_error = valid_results[0]
            last_n, last_error = valid_results[-1]

            sample_increase = last_n / first_n
            error_decrease = first_error / last_error
            theoretical_decrease = np.sqrt(sample_increase)
            efficiency = error_decrease / theoretical_decrease

            print(
                f"    Sample size: {first_n:,} → {last_n:,} ({sample_increase:.1f}x increase)"
            )
            print(
                f"    Error: {first_error:.6f} → {last_error:.6f} ({error_decrease:.2f}x decrease)"
            )
            print(
                f"    Theoretical n^(-1/2): {theoretical_decrease:.2f}x decrease expected"
            )
            print(f"    Convergence efficiency: {efficiency:.3f}")

            if efficiency >= 0.8:
                print(f"    → Excellent convergence! Very close to n^(-1/2)")
            elif efficiency >= 0.5:
                print(f"    → Good convergence behavior")
            elif efficiency >= 0.2:
                print(f"    → Moderate convergence")
            else:
                print(f"    → Poor convergence - may need investigation")


def main():
    """Main function."""
    print("Working Extended Finite-Sample Evaluation for Random Polytrees")
    print("=" * 70)

    # Test with single polytree size first
    polytree_sizes = [10]
    sample_sizes = [100, 1000, 10000, 100000, 1000000, 10000000]
    n_trials = 20
    base_seed = 42

    print(f"Configuration:")
    print(f"  Polytree sizes: {polytree_sizes}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Trials per configuration: {n_trials}")

    all_results = {}

    for n_nodes in polytree_sizes:
        results = run_finite_sample_for_random_polytree_working(
            n_nodes=n_nodes,
            sample_sizes=sample_sizes,
            n_trials=n_trials,
            seed=base_seed + n_nodes,
        )
        all_results[n_nodes] = results

    # Print summary
    print_convergence_summary(all_results)

    # Create plots
    plot_convergence_analysis(all_results)

    # Save results
    summary_data = []
    for n_nodes, data in all_results.items():
        info = data["polytree_info"]
        for result in data["results"]:
            summary_data.append(
                {
                    "polytree_size": n_nodes,
                    "n_hidden": info["n_hidden"],
                    "n_observed": info["n_observed"],
                    "n_samples": result["n_samples"],
                    "discrepancy_error_mean": result["discrepancy_error_mean"],
                    "discrepancy_error_std": result["discrepancy_error_std"],
                    "structure_success_rate": result["structure_success_rate"],
                }
            )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("working_random_polytree_results.csv", index=False)
    print(f"\nResults saved to 'working_random_polytree_results.csv'")

    return all_results, summary_df


if __name__ == "__main__":
    results, summary = main()
