"""
visualization.py

Shared visualization utilities for polytree structure learning experiments.
"""

from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from config import FIGURE_DPI, FIGURE_FORMAT_PDF, FIGURE_FORMAT_PNG


def plot_convergence_analysis(
        results_df,
        output_path: str = "convergence_analysis",
        show_plot: bool = True
) -> None:
    """
    Create convergence analysis plots for finite-sample experiments.

    Args:
        results_df: DataFrame with columns: n_samples, error metrics, success rates
        output_path: Base path for output files (without extension)
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sample_sizes = results_df['n_samples'].values

    # Variance error
    axes[0, 0].errorbar(
        sample_sizes,
        results_df['variance_errors_mean'],
        yerr=results_df['variance_errors_std'],
        marker='o', capsize=5
    )
    axes[0, 0].set_xlabel('Sample Size')
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].set_title('Variance Estimation Error')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulant error
    axes[0, 1].errorbar(
        sample_sizes,
        results_df['third_cumulant_errors_mean'],
        yerr=results_df['third_cumulant_errors_std'],
        marker='s', capsize=5
    )
    axes[0, 1].set_xlabel('Sample Size')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].set_title('Third Cumulant Estimation Error')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    # Discrepancy error
    axes[1, 0].errorbar(
        sample_sizes,
        results_df['max_discrepancy_errors_mean'],
        yerr=results_df['max_discrepancy_errors_std'],
        marker='^', capsize=5, label='Observed Error'
    )

    # Add theoretical n^(-1/2) line
    n0 = sample_sizes[0]
    error0 = results_df['max_discrepancy_errors_mean'].iloc[0]
    theoretical = error0 * np.sqrt(n0 / sample_sizes)
    axes[1, 0].plot(sample_sizes, theoretical, '--', label='n^(-1/2) rate', alpha=0.7)

    axes[1, 0].set_xlabel('Sample Size')
    axes[1, 0].set_ylabel('Max Absolute Difference')
    axes[1, 0].set_title('Discrepancy Matrix Error')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Success rate
    axes[1, 1].plot(sample_sizes, results_df['structure_recovery_success_rate'],
                    'o-', linewidth=2)
    axes[1, 1].axhline(y=0.9, color='r', linestyle='--', label='90% threshold', alpha=0.5)
    axes[1, 1].set_xlabel('Sample Size')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title('Structure Recovery Success Rate')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    for fmt in [FIGURE_FORMAT_PDF, FIGURE_FORMAT_PNG]:
        plt.savefig(f"{output_path}.{fmt}", dpi=FIGURE_DPI, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_breakdown_curves(
        results: Dict[int, Dict[str, float]],
        output_path: str = "breakdown_analysis",
        show_plot: bool = True
) -> None:
    """
    Plot performance degradation curves as function of graph size.

    Args:
        results: Dictionary mapping n -> performance metrics
        output_path: Base path for output files
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    n_vals = sorted([n for n in results.keys() if "error" not in results[n]])

    if not n_vals:
        print("No valid results to plot")
        return

    # Extract metrics
    f1_means = [results[n]["f1"] for n in n_vals]
    f1_stds = [results[n]["f1_std"] for n in n_vals]
    prec_means = [results[n]["precision"] for n in n_vals]
    rec_means = [results[n]["recall"] for n in n_vals]

    # F1 score
    axes[0, 0].errorbar(n_vals, f1_means, yerr=f1_stds, marker='o', capsize=5)
    axes[0, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Excellent (0.9)')
    axes[0, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.5)')
    axes[0, 0].set_xlabel('Graph Size (n)')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_title('Overall Performance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Precision vs Recall
    axes[0, 1].plot(n_vals, prec_means, 'o-', label='Precision')
    axes[0, 1].plot(n_vals, rec_means, 's-', label='Recall')
    axes[0, 1].set_xlabel('Graph Size (n)')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision vs Recall')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 on log scale
    axes[1, 0].semilogy(n_vals, f1_means, 'o-')
    axes[1, 0].set_xlabel('Graph Size (n)')
    axes[1, 0].set_ylabel('F1 Score (log scale)')
    axes[1, 0].set_title('Performance Decay')
    axes[1, 0].grid(True, alpha=0.3)

    # Latent complexity
    if all("n_latent" in results[n] for n in n_vals):
        n_latents = [results[n]["n_latent"] for n in n_vals]
        axes[1, 1].scatter(n_latents, f1_means, alpha=0.7)
        axes[1, 1].set_xlabel('Average Number of Latent Variables')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Performance vs Latent Complexity')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    for fmt in [FIGURE_FORMAT_PDF, FIGURE_FORMAT_PNG]:
        plt.savefig(f"{output_path}.{fmt}", dpi=FIGURE_DPI, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_threshold_comparison(
        results_df,
        output_path: str = "threshold_comparison",
        show_plot: bool = True
) -> None:
    """
    Plot performance comparison across edge weight thresholds.

    Args:
        results_df: DataFrame with threshold and performance columns
        output_path: Base path for output files
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    thresholds = results_df['threshold'].values

    # F1 Score
    axes[0, 0].errorbar(
        thresholds,
        results_df['f1'],
        yerr=results_df['f1_std'],
        marker='o', capsize=5, linewidth=2
    )
    axes[0, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Minimum Edge Weight (η)')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_title('F1 Score vs Edge Weight Threshold')
    axes[0, 0].grid(True, alpha=0.3)

    # Precision
    axes[0, 1].errorbar(
        thresholds,
        results_df['precision'],
        yerr=results_df.get('precision_std', 0),
        marker='s', capsize=5, linewidth=2, color='green'
    )
    axes[0, 1].set_xlabel('Minimum Edge Weight (η)')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Threshold')
    axes[0, 1].grid(True, alpha=0.3)

    # Recall
    axes[1, 0].errorbar(
        thresholds,
        results_df['recall'],
        yerr=results_df.get('recall_std', 0),
        marker='^', capsize=5, linewidth=2, color='orange'
    )
    axes[1, 0].set_xlabel('Minimum Edge Weight (η)')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall vs Threshold')
    axes[1, 0].grid(True, alpha=0.3)

    # Combined view
    axes[1, 1].plot(thresholds, results_df['f1'], 'o-', label='F1', linewidth=2)
    axes[1, 1].plot(thresholds, results_df['precision'], 's-', label='Precision', linewidth=2)
    axes[1, 1].plot(thresholds, results_df['recall'], '^-', label='Recall', linewidth=2)
    axes[1, 1].set_xlabel('Minimum Edge Weight (η)')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('All Metrics vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    for fmt in [FIGURE_FORMAT_PDF, FIGURE_FORMAT_PNG]:
        plt.savefig(f"{output_path}.{fmt}", dpi=FIGURE_DPI, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_random_polytree_analysis(
        all_results: Dict[int, Dict[str, Any]],
        output_path: str = "random_polytree_analysis",
        show_plot: bool = True
) -> None:
    """
    Create comprehensive plots for random polytree experiments.

    Args:
        all_results: Dictionary mapping polytree size -> results
        output_path: Base path for output files
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for n_nodes, data in sorted(all_results.items()):
        results = data["results"]
        if not results:
            continue

        sample_sizes = [r["n_samples"] for r in results]
        f1_means = [r["f1_mean"] for r in results]
        f1_stds = [r["f1_std"] for r in results]
        disc_errors = [r["discrepancy_error_mean"] for r in results]
        perfect_rates = [r["perfect_recovery_rate"] for r in results]

        label = f"n={n_nodes}"

        # Discrepancy error convergence
        axes[0, 0].errorbar(
            sample_sizes, disc_errors, yerr=[0] * len(sample_sizes),
            marker='o', label=label, capsize=3
        )

        # F1 score convergence
        axes[0, 1].errorbar(
            sample_sizes, f1_means, yerr=f1_stds,
            marker='s', label=label, capsize=3
        )

        # Perfect recovery rate
        axes[1, 0].plot(sample_sizes, perfect_rates, 'o-', label=label, linewidth=2)

        # Convergence rate check
        if len(sample_sizes) >= 2:
            theoretical = disc_errors[0] * np.sqrt(sample_sizes[0] / np.array(sample_sizes))
            axes[1, 1].plot(sample_sizes, disc_errors, 'o-', label=label)
            axes[1, 1].plot(
                sample_sizes, theoretical, '--',
                alpha=0.3, color=axes[1, 1].lines[-1].get_color()
            )

    # Format axes
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('Sample Size')
    axes[0, 0].set_ylabel('Discrepancy Error')
    axes[0, 0].set_title('Discrepancy Error Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Sample Size')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Structure Recovery F1 Score')
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('Sample Size')
    axes[1, 0].set_ylabel('Perfect Recovery Rate')
    axes[1, 0].set_title('Perfect Structure Recovery Rate')
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel('Sample Size')
    axes[1, 1].set_ylabel('Discrepancy Error')
    axes[1, 1].set_title('Observed vs Theoretical n^(-1/2) Convergence')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    for fmt in [FIGURE_FORMAT_PDF, FIGURE_FORMAT_PNG]:
        plt.savefig(f"{output_path}.{fmt}", dpi=FIGURE_DPI, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()