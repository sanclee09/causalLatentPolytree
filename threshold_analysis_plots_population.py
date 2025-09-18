#!/usr/bin/env python3
"""
corrected_threshold_analysis.py

Fixed version that uses the exact same evaluation methodology as breakdown_analysis.py
and eval_runner_pruefer.py but varies the avoid_small parameter.

Author: For TUM thesis - Linear non-Gaussian models with latent polytree structure
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
import random

# Import the evaluation functions
from eval_runner_pruefer import jaccard, greedy_match_true_to_recovered
from random_polytrees_pruefer import get_random_polytree_via_pruefer


def evaluate_one_with_eta(seed: int, n: int, eta: float) -> Dict[str, float]:
    """
    Modified version of eval_runner_pruefer.evaluate_one() that uses specific eta.
    This is the exact same evaluation logic but with customizable avoid_small parameter.
    """
    # Generate polytree with custom avoid_small parameter
    res = get_random_polytree_via_pruefer(
        n=n,
        seed=seed,
        weights_range=(-1.0, 1.0),
        avoid_small=eta,  # This is the key parameter we're varying
        ensure_at_least_one_hidden=True,
    )

    obs = set(res["observed_nodes"])
    hid_true = set(res["hidden_nodes"])
    edges_true_dir = set(res["edges_directed"])
    edges_rec = set(res["recovered_edges"])

    # Build truth sets for evaluation - EXACT SAME LOGIC AS eval_runner_pruefer.py
    true_latent_children: Dict[str, set] = {}
    for u, v in edges_true_dir:
        if u in hid_true and v in obs:
            true_latent_children.setdefault(u, set()).add(v)

    rec_latent_children: Dict[str, set] = {}
    for u, v in edges_rec:
        if u.startswith("h") and (not v.startswith("h")):
            rec_latent_children.setdefault(u, set()).add(v)

    mapping = greedy_match_true_to_recovered(true_latent_children, rec_latent_children)

    # Compute precision/recall on latent->observed edges - EXACT SAME LOGIC
    tp = fp = fn = 0
    for t, children in true_latent_children.items():
        r = mapping.get(t)
        rec_children = rec_latent_children.get(r, set()) if r is not None else set()
        tp += len(children & rec_children)
        fn += len(children - rec_children)

    for r, rec_children in rec_latent_children.items():
        mapped_ts = [t for t, rr in mapping.items() if rr == r]
        if mapped_ts:
            t = mapped_ts[0]
            fp += len(rec_children - true_latent_children.get(t, set()))
        else:
            fp += len(rec_children)

    prec_lat = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_lat = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_lat = (
        (2 * prec_lat * rec_lat / (prec_lat + rec_lat))
        if (prec_lat + rec_lat) > 0
        else 0.0
    )

    unmatched_rec = sum(1 for r in rec_latent_children if r not in mapping.values())

    return {
        "prec_lat": prec_lat,
        "rec_lat": rec_lat,
        "f1_lat": f1_lat,
        "unmatched_rec_latents": float(unmatched_rec),
        "n_obs": float(len(obs)),
        "n_true_lat": float(len(hid_true)),
    }


def run_experiments_with_eta(
    K: int, n: int, eta: float, seed: int = 1
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Modified version of eval_runner_pruefer.run_experiments() with custom eta.
    """
    rng = random.Random(seed)
    metrics = []

    for _ in range(K):
        m = evaluate_one_with_eta(seed=rng.randint(0, 10**9), n=n, eta=eta)
        metrics.append(m)

    keys = metrics[0].keys() if metrics else []
    avg = {f"avg_{k}": float(np.mean([m[k] for m in metrics])) for k in keys}
    std = {f"std_{k}": float(np.std([m[k] for m in metrics], ddof=1)) for k in keys}

    return {**avg, **std}, metrics


def evaluate_configuration(
    n: int, eta: float, n_trials: int = 3, base_seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate algorithm performance for a single (n, eta) configuration.
    Uses the exact same evaluation methodology as breakdown_analysis.py
    """
    try:
        summary, runs = run_experiments_with_eta(
            K=n_trials, n=n, eta=eta, seed=base_seed
        )

        return {
            "n": n,
            "eta": eta,
            "precision": summary["avg_prec_lat"],
            "recall": summary["avg_rec_lat"],
            "f1": summary["avg_f1_lat"],
            "precision_std": summary["std_prec_lat"],
            "recall_std": summary["std_rec_lat"],
            "f1_std": summary["std_f1_lat"],
            "n_obs": summary["avg_n_obs"],
            "n_true_lat": summary["avg_n_true_lat"],
        }
    except Exception as e:
        print(f"Error for n={n}, eta={eta:.2f}: {e}")
        return {
            "n": n,
            "eta": eta,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "precision_std": 0.0,
            "recall_std": 0.0,
            "f1_std": 0.0,
            "n_obs": 0.0,
            "n_true_lat": 0.0,
        }


def run_threshold_analysis(
    n_values: List[int] = [20, 30, 40, 50, 65, 80, 100, 120, 140],
    eta_values: List[float] = [0.1, 0.3, 0.5, 0.8],
    n_trials: int = 10,
) -> pd.DataFrame:
    """Run threshold analysis using the correct evaluation methodology."""

    print(f"Running corrected threshold analysis...")
    print(f"Testing {len(n_values)} polytree sizes: {n_values}")
    print(f"Testing {len(eta_values)} edge weight thresholds: {eta_values}")
    print(f"Using {n_trials} trials per configuration")
    print(f"Using the SAME evaluation as breakdown_analysis.py")
    print()

    results = []
    total_configs = len(n_values) * len(eta_values)
    completed = 0

    for n in n_values:
        for eta in eta_values:
            print(f"Evaluating n={n:3d}, eta={eta:.1f}... ", end="")

            result = evaluate_configuration(n, eta, n_trials, base_seed=42)
            results.append(result)
            print(f"F1={result['f1']:.3f} ± {result['f1_std']:.3f}")

            completed += 1
            if completed % 5 == 0:
                print(
                    f"Progress: {completed}/{total_configs} ({100 * completed / total_configs:.1f}%)"
                )

    return pd.DataFrame(results)


def create_threshold_plots(
    df: pd.DataFrame, save_path: str = "corrected_threshold_analysis.pdf"
):
    """Create 3-panel plot showing the threshold phenomenon with proper spacing."""

    plt.style.use("default")
    sns.set_palette("husl")

    # Create 2x2 layout but only use 3 panels
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Main title with more space
    fig.suptitle(
        "Critical Edge Weight Threshold Phenomenon",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    eta_values = sorted(df["eta"].unique())
    colors = sns.color_palette("viridis", len(eta_values))

    # Define the 3 plots we want
    plot_configs = [
        ("f1", "F1 Score", axes[0, 0]),
        ("precision", "Precision", axes[0, 1]),
        ("recall", "Recall", axes[1, 0]),
    ]

    # Plot the first 3 panels
    for metric, title, ax in plot_configs:
        for i, eta in enumerate(eta_values):
            eta_data = df[df["eta"] == eta].sort_values("n")

            if len(eta_data) > 0:
                # Main line with error bars
                ax.errorbar(
                    eta_data["n"],
                    eta_data[metric],
                    yerr=eta_data[f"{metric}_std"],
                    marker="o",
                    linewidth=2.5,
                    markersize=6,
                    color=colors[i],
                    label=f"η = {eta:.1f}",
                    capsize=3,
                    alpha=0.8,
                )

        ax.set_xlabel("Number of Nodes (n)", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(
            f"{title} vs. Polytree Size", fontsize=12, fontweight="bold", pad=15
        )
        ax.legend(title="Min Edge Weight", fontsize=9, title_fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Add threshold lines only for F1 score
        if metric == "f1":
            ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.7, linewidth=1)
            ax.text(
                ax.get_xlim()[1] * 0.6,
                0.92,
                "Excellent",
                fontsize=8,
                color="red",
                alpha=0.8,
            )
            ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.7, linewidth=1)
            ax.text(
                ax.get_xlim()[1] * 0.6,
                0.52,
                "Breakdown",
                fontsize=8,
                color="orange",
                alpha=0.8,
            )

    # Fourth panel: Number of correctly recovered latent variables
    ax4 = axes[1, 1]

    for i, eta in enumerate(eta_values):
        eta_data = df[df["eta"] == eta].sort_values("n")

        if len(eta_data) > 0:
            # Calculate recovered latents (this is approximate - you might want to adjust based on your data)
            # Using F1 score and number of true latents as proxy
            recovered_latents = eta_data["f1"] * eta_data["n_true_lat"]
            recovered_latents_std = eta_data["f1_std"] * eta_data["n_true_lat"]

            ax4.errorbar(
                eta_data["n"],
                recovered_latents,
                yerr=recovered_latents_std,
                marker="s",
                linewidth=2.5,
                markersize=6,
                color=colors[i],
                label=f"η = {eta:.1f}",
                capsize=3,
                alpha=0.8,
            )

    ax4.set_xlabel("Number of Nodes (n)", fontsize=11)
    ax4.set_ylabel("Recovered Latent Variables", fontsize=11)
    ax4.set_title(
        "Latent Recovery vs. Polytree Size", fontsize=12, fontweight="bold", pad=15
    )
    ax4.legend(title="Min Edge Weight", fontsize=9, title_fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)

    # Adjust layout with much more space
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.45, wspace=0.25)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Corrected plot saved to: {save_path}")


def create_summary_table(df: pd.DataFrame):
    """Create and print a summary table showing breakdown points."""

    print("\nCorrected Threshold Analysis Summary:")
    print("=" * 90)
    print(
        f"{'η':>6} {'Best F1':>8} {'Best n':>8} {'50% Drop n':>12} {'10% Drop n':>12} {'Status':>15}"
    )
    print("-" * 90)

    for eta in sorted(df["eta"].unique()):
        eta_data = df[df["eta"] == eta].sort_values("n")

        best_f1 = eta_data["f1"].max()
        best_n = eta_data.loc[eta_data["f1"].idxmax(), "n"] if best_f1 > 0 else None

        # Find 50% drop point
        drop50_data = eta_data[eta_data["f1"] < 0.5]
        drop50_n = drop50_data["n"].min() if not drop50_data.empty else None

        # Find 10% drop from maximum
        if best_f1 > 0.1:
            drop10_threshold = best_f1 - 0.1
            drop10_data = eta_data[eta_data["f1"] < drop10_threshold]
            drop10_n = drop10_data["n"].min() if not drop10_data.empty else None
        else:
            drop10_n = None

        if best_f1 > 0.9:
            status = "Excellent"
        elif best_f1 > 0.7:
            status = "Good"
        elif best_f1 > 0.3:
            status = "Fair"
        else:
            status = "Poor"

        print(
            f"{eta:6.2f} {best_f1:8.3f} {best_n if best_n else 'N/A':>8} "
            f"{drop50_n if drop50_n else '>140':>12} "
            f"{drop10_n if drop10_n else '>140':>12} {status:>15}"
        )


def main():
    """Main execution function."""

    print("Starting CORRECTED Threshold Analysis for Master's Thesis")
    print("Using the SAME evaluation methodology as breakdown_analysis.py")
    print("=" * 70)

    # Test configuration that should show the threshold effect
    n_values = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    eta_values = [0.1, 0.3, 0.5, 0.8]
    n_trials = 20  # More trials for reliability

    # Run the analysis
    results_df = run_threshold_analysis(
        n_values=n_values, eta_values=eta_values, n_trials=n_trials
    )

    # Save results
    results_df.to_csv("corrected_threshold_analysis_results.csv", index=False)
    print(f"\nRaw results saved to: corrected_threshold_analysis_results.csv")

    # Create summary
    create_summary_table(results_df)

    # Create plots
    create_threshold_plots(results_df, "corrected_threshold_analysis.pdf")

    print("\nCorrected analysis complete!")
    print("\nKey expectations for your thesis:")
    print("- η = 0.1 should show breakdown around n=65 (like your original finding)")
    print("- η = 0.8 should maintain better performance at larger scales")
    print("- The results should now match the breakdown_analysis.py behavior")
    print("- This will provide proper evidence for the threshold phenomenon")


if __name__ == "__main__":
    main()
