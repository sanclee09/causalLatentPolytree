#!/usr/bin/env python3
"""
breakdown_analysis.py

Systematic experiment to find where the cumulant-based polytree recovery breaks down
as a function of graph size n.
"""

import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from eval_runner_pruefer import run_experiments


def breakdown_analysis(
    n_values: List[int] = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100],
    K: int = 30,  # Reduced for faster execution
    seed: int = 42,
    out_dir: str = "breakdown_analysis",
) -> Dict[int, Dict[str, float]]:
    """
    Run experiments across different graph sizes to identify breakdown point.

    Args:
        n_values: List of graph sizes to test
        K: Number of trials per size
        seed: Random seed for reproducibility
        out_dir: Output directory for results

    Returns:
        Dictionary mapping n -> summary statistics
    """
    results = {}
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Running breakdown analysis: n in {n_values}, K={K} trials each")
    print("=" * 60)

    for n in n_values:
        print(f"\nTesting n={n}...")
        start_time = time.time()

        try:
            summary, runs = run_experiments(K=K, n=n, seed=seed)
            elapsed = time.time() - start_time

            # Extract key metrics
            results[n] = {
                "precision": summary["avg_prec_lat"],
                "recall": summary["avg_rec_lat"],
                "f1": summary["avg_f1_lat"],
                "precision_std": summary["std_prec_lat"],
                "recall_std": summary["std_rec_lat"],
                "f1_std": summary["std_f1_lat"],
                "n_obs": summary["avg_n_obs"],
                "n_true_lat": summary["avg_n_true_lat"],
                "unmatched_latents": summary["avg_unmatched_rec_latents"],
                "runtime": elapsed,
            }

            print(f"  F1: {results[n]['f1']:.3f} ± {results[n]['f1_std']:.3f}")
            print(
                f"  Precision: {results[n]['precision']:.3f} ± {results[n]['precision_std']:.3f}"
            )
            print(
                f"  Recall: {results[n]['recall']:.3f} ± {results[n]['recall_std']:.3f}"
            )
            print(f"  Runtime: {elapsed:.1f}s")

            # Save individual results
            (out_path / f"n{n}_summary.json").write_text(json.dumps(summary, indent=2))

            # Early stopping if performance becomes terrible
            if results[n]["f1"] < 0.1 and n > 30:
                print(f"  Performance collapsed at n={n}, stopping early")
                break

        except Exception as e:
            print(f"  ERROR at n={n}: {e}")
            results[n] = {"error": str(e)}

    return results


def plot_breakdown_curves(
    results: Dict[int, Dict[str, float]], out_dir: str = "breakdown_analysis"
):
    """Create plots showing performance degradation with graph size."""

    # Extract data for plotting
    n_vals = []
    f1_means = []
    f1_stds = []
    prec_means = []
    prec_stds = []
    rec_means = []
    rec_stds = []

    for n in sorted(results.keys()):
        if "error" not in results[n]:
            n_vals.append(n)
            f1_means.append(results[n]["f1"])
            f1_stds.append(results[n]["f1_std"])
            prec_means.append(results[n]["precision"])
            prec_stds.append(results[n]["precision_std"])
            rec_means.append(results[n]["recall"])
            rec_stds.append(results[n]["recall_std"])

    n_vals = np.array(n_vals)
    f1_means = np.array(f1_means)
    f1_stds = np.array(f1_stds)
    prec_means = np.array(prec_means)
    prec_stds = np.array(prec_stds)
    rec_means = np.array(rec_means)
    rec_stds = np.array(rec_stds)

    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Cumulant-Based Polytree Recovery: Performance vs Graph Size", fontsize=14
    )

    # F1 Score
    axes[0, 0].errorbar(n_vals, f1_means, yerr=f1_stds, marker="o", capsize=5)
    axes[0, 0].set_xlabel("Graph Size (n)")
    axes[0, 0].set_ylabel("F1 Score")
    axes[0, 0].set_title("F1 Score vs Graph Size")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)

    # Precision vs Recall
    axes[0, 1].errorbar(
        n_vals, prec_means, yerr=prec_stds, marker="s", label="Precision", capsize=5
    )
    axes[0, 1].errorbar(
        n_vals, rec_means, yerr=rec_stds, marker="^", label="Recall", capsize=5
    )
    axes[0, 1].set_xlabel("Graph Size (n)")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Precision vs Recall")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.1)

    # Performance degradation rate
    if len(n_vals) > 3:
        # Fit exponential decay to F1 scores
        try:
            log_f1 = np.log(np.maximum(f1_means, 1e-6))
            poly_coeffs = np.polyfit(n_vals, log_f1, 1)
            decay_rate = -poly_coeffs[0]

            axes[1, 0].semilogy(n_vals, f1_means, "o-", label="Observed F1")
            axes[1, 0].semilogy(
                n_vals,
                np.exp(np.polyval(poly_coeffs, n_vals)),
                "--",
                label=f"Exp fit (decay={decay_rate:.4f})",
            )
            axes[1, 0].set_xlabel("Graph Size (n)")
            axes[1, 0].set_ylabel("F1 Score (log scale)")
            axes[1, 0].set_title("Performance Decay Rate")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        except:
            axes[1, 0].plot(n_vals, f1_means, "o-")
            axes[1, 0].set_title("F1 Score (linear)")

    # Number of latent variables vs performance
    n_latents = [results[n]["n_true_lat"] for n in n_vals]
    axes[1, 1].scatter(n_latents, f1_means, alpha=0.7)
    axes[1, 1].set_xlabel("Average Number of Latent Variables")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].set_title("Performance vs Latent Complexity")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/breakdown_analysis.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out_dir}/breakdown_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def find_breakdown_threshold(
    results: Dict[int, Dict[str, float]], threshold: float = 0.5
) -> int:
    """Find the graph size where F1 score drops below threshold."""
    for n in sorted(results.keys()):
        if "error" not in results[n] and results[n]["f1"] < threshold:
            return n
    return max(results.keys())  # Never broke down in tested range


if __name__ == "__main__":
    # Run the breakdown analysis
    print("Starting breakdown point analysis...")

    # Test smaller range first to see the pattern
    n_values = [
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
        120,
        140,
    ]

    results = breakdown_analysis(
        n_values=n_values,
        K=30,  # Reduced for faster execution
        seed=42,
        out_dir="breakdown_analysis",
    )

    # Save comprehensive results
    out_path = Path("breakdown_analysis")
    with open(out_path / "full_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create plots
    plot_breakdown_curves(results)

    # Find breakdown threshold
    breakdown_n = find_breakdown_threshold(results, threshold=0.5)
    print(f"\nBreakdown analysis complete!")
    print(f"F1 score drops below 0.5 at n = {breakdown_n}")

    # Summary table
    print("\nSummary Table:")
    print("n\tF1\tPrec\tRec\tLatents")
    print("-" * 40)
    for n in sorted(results.keys()):
        if "error" not in results[n]:
            r = results[n]
            print(
                f"{n}\t{r['f1']:.3f}\t{r['precision']:.3f}\t{r['recall']:.3f}\t{r['n_true_lat']:.1f}"
            )
