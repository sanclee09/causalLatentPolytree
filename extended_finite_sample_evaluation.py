from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random_polytrees_pruefer import generate_random_latent_polytree
from polytree_discrepancy import (
    generate_noise_samples,
    apply_lsem_transformation,
    finite_sample_discrepancy,
)
from latent_polytree_truepoly import get_polytree_algo3
from learn_with_hidden import observed_gamma_from_params

from collections import defaultdict, deque


def _suppress_degree2_latents(edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Remove redundant degree-2 latent nodes from edge list.
    A latent node with undirected degree 2 (1 parent + 1 child) can be bypassed.
    """
    from collections import defaultdict

    E = set(edges)

    def build_maps(E):
        children = defaultdict(set)
        parent = {}
        nodes = set()
        for u, v in E:
            children[u].add(v)
            parent[v] = u
            nodes.add(u)
            nodes.add(v)
        for x in nodes:
            children.setdefault(x, set())
        return nodes, children, parent

    changed = True
    while changed:
        changed = False
        nodes, children, parent = build_maps(E)

        for h in list(nodes):
            if not (isinstance(h, str) and h.startswith("h")):
                continue

            indeg = 1 if h in parent else 0
            outdeg = len(children[h])
            undeg = indeg + outdeg

            if undeg != 2:
                continue

            # Case A: Chain latent (1 parent, 1 child) -> bypass
            if indeg == 1 and outdeg == 1:
                p = parent[h]
                c = next(iter(children[h]))
                E.discard((p, h))
                E.discard((h, c))
                E.add((p, c))
                changed = True
                continue

            # Case B: Root latent with 2 children (1 latent, 1 observed) -> connect latent→observed
            if indeg == 0 and outdeg == 2:
                ch = list(children[h])
                lat = [x for x in ch if isinstance(x, str) and x.startswith("h")]
                obs = [x for x in ch if not (isinstance(x, str) and x.startswith("h"))]
                if len(lat) == 1 and len(obs) == 1:
                    L, o = lat[0], obs[0]
                    E.discard((h, L))
                    E.discard((h, o))
                    E.add((L, o))
                    changed = True
                    continue

    return list(E)


def _topo_order(edges: list[tuple[str, str]]):
    nodes = set(u for u, v in edges) | set(v for u, v in edges)
    children = defaultdict(set)
    indeg = defaultdict(int)
    for u, v in edges:
        children[u].add(v)
        indeg[v] += 1
    q = deque(sorted([u for u in nodes if indeg[u] == 0]))
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for w in sorted(children.get(u, ())):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    return topo


def _relabel_latents_by_topo(edges: list[tuple[str, str]]):
    topo = _topo_order(edges)
    latents = [u for u in topo if isinstance(u, str) and u.startswith("h")]
    ren = {old: f"h{i+1}" for i, old in enumerate(latents)}

    def m(x):
        return ren.get(x, x)

    return [(m(u), m(v)) for (u, v) in edges]


def run_finite_sample_for_random_polytree(
    n_nodes: int, sample_sizes: List[int], n_trials: int = 3, seed: int = 42
) -> Dict[str, Any]:
    """
    Finite-sample evaluation for random polytrees with FULL structure comparison.
    Each trial uses a DIFFERENT random polytree structure.
    """
    print(f"\n{'=' * 60}")
    print(f"FINITE-SAMPLE EVALUATION: RANDOM POLYTREE (n={n_nodes})")
    print(f"{'=' * 60}")

    results = []

    for n_samples in sample_sizes:
        print(f"\nTesting n_samples = {n_samples:,}...")

        sample_results = {
            "n_nodes": n_nodes,
            "n_samples": n_samples,
            "trials": [],  # Store each trial's complete info
        }

        for trial in range(n_trials):
            print(f"\n  === Trial {trial + 1}/{n_trials} ===")

            # Generate a NEW random polytree for each trial
            ground_truth = generate_random_latent_polytree(
                n=n_nodes,
                seed=seed + trial + n_samples,
                weights_range=(-1.0, 1.0),
                avoid_small=0.8,
                ensure_at_least_one_hidden=True,
            )

            edges_dict = ground_truth["edges"]
            all_nodes = ground_truth["all_nodes"]
            observed_nodes = ground_truth["observed_nodes"]
            latent_nodes = ground_truth["latent_nodes"]

            print(
                f"  Ground truth: {len(latent_nodes)} latent, {len(observed_nodes)} observed, {len(edges_dict)} edges"
            )
            print(f"    Latent: {latent_nodes}")
            print(f"    Edges: {sorted(edges_dict.keys())}")

            # Fixed gamma parameters
            gamma_shapes = {node: 2.5 for node in all_nodes}
            gamma_scales = {node: 1.0 for node in all_nodes}

            # Compute population observed discrepancy
            sigmas = {v: gamma_scales[v] * np.sqrt(gamma_shapes[v]) for v in all_nodes}
            kappas = {
                v: 2 * gamma_shapes[v] * (gamma_scales[v] ** 3) for v in all_nodes
            }

            Gamma_population_obs, _, _ = observed_gamma_from_params(
                edges_dict,
                sigmas,
                kappas,
                hidden=set(latent_nodes),
                auto_detect_hidden=False,
            )

            trial_result = {
                "trial_id": trial + 1,
                "ground_truth_edges": sorted(edges_dict.keys()),
                "n_latent": len(latent_nodes),
                "n_observed": len(observed_nodes),
                "latent_nodes": latent_nodes,
            }

            try:
                # Suppress verbose output for all trials
                verbose = False

                # Generate finite-sample data
                noise_samples = generate_noise_samples(
                    all_nodes,
                    gamma_shapes,
                    gamma_scales,
                    n_samples,
                    seed + trial * 1000,
                    verbose=verbose,
                )

                X_samples = apply_lsem_transformation(
                    noise_samples, edges_dict, all_nodes, verbose=verbose
                )

                Gamma_finite_full = finite_sample_discrepancy(
                    X_samples, all_nodes, verbose=verbose
                )

                # Extract observed-only discrepancy matrix
                observed_indices = [all_nodes.index(node) for node in observed_nodes]
                Gamma_finite_obs = Gamma_finite_full[
                    np.ix_(observed_indices, observed_indices)
                ]

                # Discrepancy error
                discrepancy_error = np.max(
                    np.abs(Gamma_finite_obs - Gamma_population_obs)
                )
                trial_result["discrepancy_error"] = discrepancy_error
                print(f"  Discrepancy error: {discrepancy_error:.6f}")

                # Recover structure
                recovered_polytree = get_polytree_algo3(Gamma_finite_obs)

                # Map recovered edges to actual node names
                # Map recovered edges to actual node names
                def map_recovered_edge_names(edge_tuple, observed_nodes):
                    parent, child = edge_tuple

                    def map_one(x):
                        if isinstance(x, str) and x.startswith("h"):
                            return x
                        idx = int(x) if isinstance(x, str) else x
                        return observed_nodes[idx]

                    return (map_one(parent), map_one(child))

                recovered_edges_raw = [
                    map_recovered_edge_names(edge, observed_nodes)
                    for edge in recovered_polytree.edges
                ]

                # Remove redundant degree-2 latents
                recovered_edges_min = _suppress_degree2_latents(recovered_edges_raw)

                # Apply deterministic latent relabeling to BOTH graphs
                pred_edges_list = _relabel_latents_by_topo(sorted(recovered_edges_min))
                true_edges_list = _relabel_latents_by_topo(sorted(edges_dict.keys()))

                print(f"  Recovered edges: {pred_edges_list}")

                pred_set = set(pred_edges_list)
                true_set = set(true_edges_list)

                # Compute metrics
                correct_edges = len(true_set & pred_set)
                total_recovered = len(pred_set)
                total_true = len(true_set)

                precision = (
                    correct_edges / total_recovered if total_recovered > 0 else 0
                )
                recall = correct_edges / total_true if total_true > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
                perfect = true_set == pred_set

                trial_result["recovered_edges"] = pred_edges_list
                trial_result["precision"] = precision
                trial_result["recall"] = recall
                trial_result["f1"] = f1
                trial_result["perfect"] = perfect

                # Print results
                print(
                    f"  Results: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Perfect={perfect}"
                )

                if not perfect:
                    missing = sorted(true_set - pred_set)
                    extra = sorted(pred_set - true_set)
                    if missing:
                        print(f"    Missing ({len(missing)}): {missing}")
                    if extra:
                        print(f"    Extra ({len(extra)}): {extra}")

                trial_result["success"] = True

            except Exception as e:
                print(f"  Trial {trial + 1} FAILED: {e}")
                trial_result["success"] = False
                trial_result["error"] = str(e)

            sample_results["trials"].append(trial_result)

        # Aggregate successful trials
        successful_trials = [
            t for t in sample_results["trials"] if t.get("success", False)
        ]

        if successful_trials:
            aggregated = {
                "n_nodes": n_nodes,
                "n_samples": n_samples,
                "n_trials_total": n_trials,
                "n_trials_successful": len(successful_trials),
                "n_latent_mean": np.mean([t["n_latent"] for t in successful_trials]),
                "n_latent_std": np.std([t["n_latent"] for t in successful_trials]),
                "n_observed_mean": np.mean(
                    [t["n_observed"] for t in successful_trials]
                ),
                "n_observed_std": np.std([t["n_observed"] for t in successful_trials]),
                "discrepancy_error_mean": np.mean(
                    [t["discrepancy_error"] for t in successful_trials]
                ),
                "discrepancy_error_std": np.std(
                    [t["discrepancy_error"] for t in successful_trials]
                ),
                "precision_mean": np.mean([t["precision"] for t in successful_trials]),
                "precision_std": np.std([t["precision"] for t in successful_trials]),
                "recall_mean": np.mean([t["recall"] for t in successful_trials]),
                "recall_std": np.std([t["recall"] for t in successful_trials]),
                "f1_mean": np.mean([t["f1"] for t in successful_trials]),
                "f1_std": np.std([t["f1"] for t in successful_trials]),
                "perfect_recovery_rate": np.mean(
                    [t["perfect"] for t in successful_trials]
                ),
            }

            print(
                f"\n  === Summary for n={n_samples:,} ({len(successful_trials)}/{n_trials} successful) ==="
            )
            print(
                f"  Latent nodes: {aggregated['n_latent_mean']:.1f} ± {aggregated['n_latent_std']:.1f}"
            )
            print(
                f"  Discrepancy error: {aggregated['discrepancy_error_mean']:.6f} ± {aggregated['discrepancy_error_std']:.6f}"
            )
            print(
                f"  F1 Score: {aggregated['f1_mean']:.3f} ± {aggregated['f1_std']:.3f}"
            )
            print(f"  Perfect recovery: {aggregated['perfect_recovery_rate']:.1%}")

            results.append(aggregated)

    return {
        "n_nodes": n_nodes,
        "results": results,
        "all_trials": sample_results["trials"],  # Include all trial details
    }


def plot_convergence_analysis(all_results: Dict[int, Dict[str, Any]]):
    """
    Create comprehensive convergence analysis plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(all_results)))

    for idx, (n_nodes, data) in enumerate(sorted(all_results.items())):
        color = colors[idx]
        label = f"n={n_nodes}"

        results = data["results"]
        if not results:
            continue

        plot_sample_sizes = [r["n_samples"] for r in results]
        plot_error_means = [r["discrepancy_error_mean"] for r in results]
        plot_error_stds = [r["discrepancy_error_std"] for r in results]
        plot_f1_means = [r["f1_mean"] for r in results]
        plot_f1_stds = [r["f1_std"] for r in results]
        plot_perfect_rates = [r["perfect_recovery_rate"] for r in results]

        # Discrepancy error convergence
        axes[0, 0].fill_between(
            plot_sample_sizes,
            np.array(plot_error_means) - np.array(plot_error_stds),
            np.array(plot_error_means) + np.array(plot_error_stds),
            alpha=0.3,
            color=color,
        )
        axes[0, 0].loglog(
            plot_sample_sizes, plot_error_means, "o-", color=color, label=label
        )

        # F1 Score
        axes[0, 1].fill_between(
            plot_sample_sizes,
            np.array(plot_f1_means) - np.array(plot_f1_stds),
            np.array(plot_f1_means) + np.array(plot_f1_stds),
            alpha=0.3,
            color=color,
        )
        axes[0, 1].semilogx(
            plot_sample_sizes, plot_f1_means, "o-", color=color, label=label
        )

        # Perfect recovery rate
        axes[1, 0].semilogx(
            plot_sample_sizes, plot_perfect_rates, "o-", color=color, label=label
        )

        # Theoretical n^(-1/2) comparison
        if len(plot_sample_sizes) > 1:
            theoretical = plot_error_means[0] * np.sqrt(
                plot_sample_sizes[0] / np.array(plot_sample_sizes)
            )
            axes[1, 1].loglog(
                plot_sample_sizes,
                theoretical,
                "--",
                color=color,
                alpha=0.5,
                label=f"{label} Theory",
            )
            axes[1, 1].loglog(
                plot_sample_sizes,
                plot_error_means,
                "o-",
                color=color,
                label=f"{label} Observed",
            )

    # Formatting
    axes[0, 0].set_xlabel("Sample Size")
    axes[0, 0].set_ylabel("Max Discrepancy Error")
    axes[0, 0].set_title("Discrepancy Error Convergence")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Sample Size")
    axes[0, 1].set_ylabel("F1 Score")
    axes[0, 1].set_title("Structure Recovery F1 Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.1)

    axes[1, 0].set_xlabel("Sample Size")
    axes[1, 0].set_ylabel("Perfect Recovery Rate")
    axes[1, 0].set_title("Perfect Structure Recovery Rate")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.1)

    axes[1, 1].set_xlabel("Sample Size")
    axes[1, 1].set_ylabel("Discrepancy Error")
    axes[1, 1].set_title("Observed vs Theoretical n^(-1/2) Convergence")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("random_polytree_analysis.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("random_polytree_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
    handles, labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        axes[1, 1].legend()


def print_convergence_summary(all_results: Dict[int, Dict[str, Any]]):
    """Print detailed convergence analysis summary."""
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("=" * 70)

    for n_nodes, data in sorted(all_results.items()):
        print(f"\nPolytree size: {n_nodes} nodes")

        results = data["results"]
        if not results:
            print("  No successful results")
            continue

        # Get stats from aggregated results
        first_result = results[0]
        print(
            f"  Avg latent nodes: {first_result['n_latent_mean']:.1f} ± {first_result['n_latent_std']:.1f}"
        )
        print(
            f"  Avg observed nodes: {first_result['n_observed_mean']:.1f} ± {first_result['n_observed_std']:.1f}"
        )

        print("\n  Sample Size | F1 Score | Perfect Recovery | Trials")
        print("  " + "-" * 60)
        for r in results:
            print(
                f"  {r['n_samples']:>11,} | {r['f1_mean']:>4.2f} ± {r['f1_std']:>4.2f} | "
                f"{r['perfect_recovery_rate']:>6.1%} | {r['n_trials_successful']}/{r['n_trials_total']}"
            )


def main():
    """Main evaluation function."""
    print("Extended Finite-Sample Evaluation for Random Polytrees")
    print("=" * 70)

    # Configuration
    polytree_sizes = [5]
    sample_sizes = [10000000]
    n_trials = 10
    base_seed = 42

    print(f"Configuration:")
    print(f"  Polytree sizes: {polytree_sizes}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Trials per configuration: {n_trials}")

    all_results = {}

    for n_nodes in polytree_sizes:
        results = run_finite_sample_for_random_polytree(
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
        for result in data["results"]:
            summary_data.append(
                {
                    "polytree_size": n_nodes,
                    "n_samples": result["n_samples"],
                    "n_latent_mean": result["n_latent_mean"],
                    "n_latent_std": result["n_latent_std"],
                    "n_observed_mean": result["n_observed_mean"],
                    "n_observed_std": result["n_observed_std"],
                    "discrepancy_error_mean": result["discrepancy_error_mean"],
                    "precision_mean": result["precision_mean"],
                    "recall_mean": result["recall_mean"],
                    "f1_mean": result["f1_mean"],
                    "perfect_recovery_rate": result["perfect_recovery_rate"],
                }
            )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("random_polytree_results.csv", index=False)
    print(f"\nResults saved to 'random_polytree_results.csv'")

    return all_results, summary_df


if __name__ == "__main__":
    results, summary = main()
