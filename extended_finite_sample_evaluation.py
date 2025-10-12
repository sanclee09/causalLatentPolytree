from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random_polytrees_pruefer import get_random_polytree_via_pruefer
from polytree_discrepancy import (
    generate_noise_samples,
    apply_lsem_transformation,
    finite_sample_discrepancy,
    compute_gamma_parameters_from_moments,
)
from latent_polytree_truepoly import get_polytree_algo3
from learn_with_hidden import observed_gamma_from_params

from collections import defaultdict, deque


def print_discrepancy_comparison(Gamma_pop, Gamma_finite, observed_nodes, label=""):
    """Print side-by-side comparison of population and finite-sample discrepancy matrices."""
    print(f"\n  {label}DISCREPANCY MATRIX DIAGNOSTICS")
    print(f"  Observed nodes: {observed_nodes}")
    print(f"\n  Population Γ_obs:")
    for i, row in enumerate(Gamma_pop):
        print(f"    {observed_nodes[i]}: " + " ".join(f"{val:8.4f}" for val in row))

    print(f"\n  Finite-sample Γ_obs:")
    for i, row in enumerate(Gamma_finite):
        print(f"    {observed_nodes[i]}: " + " ".join(f"{val:8.4f}" for val in row))

    print(f"\n  Difference (|Finite - Population|):")
    diff = np.abs(Gamma_finite - Gamma_pop)
    for i, row in enumerate(diff):
        print(f"    {observed_nodes[i]}: " + " ".join(f"{val:8.4f}" for val in row))
    print(f"  Max difference: {np.max(diff):.6f}\n")


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
    n_nodes: int,
    sample_sizes: List[int],
    n_trials: int = 3,
    seed: int = 42,
    n_latent: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate finite-sample recovery performance on random polytrees.
    Each trial generates ONE polytree tested across ALL sample sizes.

    Args:
        n_nodes: Number of nodes in generated polytrees
        sample_sizes: List of sample sizes to test
        n_trials: Number of polytrees to generate
        seed: Base random seed
        n_latent: Number of latent nodes (None = all candidates)

    Returns:
        Dictionary with aggregated results and trial details
    """
    print(f"\n{'=' * 60}")
    print(f"FINITE-SAMPLE EVALUATION: RANDOM POLYTREE (n={n_nodes})")
    print(f"{'=' * 60}")

    # Store all trial results grouped by sample size
    all_results_by_sample_size = {n_samples: [] for n_samples in sample_sizes}

    for trial in range(n_trials):
        print(f"\n{'=' * 60}")
        print(f"TRIAL {trial + 1}/{n_trials} - Generating new polytree")
        print(f"{'=' * 60}")

        # Generate ONE polytree for this trial
        ground_truth = get_random_polytree_via_pruefer(
            n=n_nodes,
            seed=seed + trial,  # Same seed for this trial
            weights_range=(-1.0, 1.0),
            avoid_small=0.8,
            ensure_at_least_one_hidden=True,
            n_latent=n_latent,
        )

        # Extract and rename (once per trial)
        edges_dict, latent_nodes, observed_nodes, all_nodes, sigmas, kappas = (
            _extract_and_rename_ground_truth(ground_truth)
        )

        print(
            f"Ground truth: {len(latent_nodes)} latent, {len(observed_nodes)} observed, {len(edges_dict)} edges"
        )
        print(f"  Latent: {latent_nodes}")
        print(f"  Edges: {sorted(edges_dict.keys())}")

        # Convert moments to Gamma parameters
        gamma_shapes, gamma_scales = compute_gamma_parameters_from_moments(
            sigmas, kappas
        )

        print(f"  Noise parameters:")
        for node in all_nodes:
            print(
                f"    {node}: σ={sigmas[node]:.4f}, κ³={kappas[node]:.4f}, "
                f"k={gamma_shapes[node]:.4f}, θ={gamma_scales[node]:.4f}"
            )

        # Compute population discrepancy (once per trial)
        Gamma_population_obs = _compute_population_discrepancy(
            edges_dict, sigmas, kappas, latent_nodes
        )

        # Test this SAME polytree with DIFFERENT sample sizes
        for n_samples in sample_sizes:
            print(f"\n  --- Testing with n_samples = {n_samples:,} ---")

            trial_result = _run_single_sample_size(
                all_nodes,
                observed_nodes,
                edges_dict,
                latent_nodes,
                gamma_shapes,
                gamma_scales,
                Gamma_population_obs,
                n_samples,
                seed,
                trial,
            )

            trial_result["trial_id"] = trial + 1
            trial_result["n_latent"] = len(latent_nodes)
            trial_result["n_observed"] = len(observed_nodes)
            trial_result["latent_nodes"] = latent_nodes
            trial_result["ground_truth_edges"] = sorted(edges_dict.keys())

            all_results_by_sample_size[n_samples].append(trial_result)

    # Aggregate results for each sample size
    aggregated_results = []
    for n_samples in sample_sizes:
        trials_for_this_size = all_results_by_sample_size[n_samples]
        successful_trials = [t for t in trials_for_this_size if t.get("success", False)]

        if successful_trials:
            aggregated = _aggregate_trial_results(
                successful_trials, n_nodes, n_samples, n_trials
            )
            aggregated_results.append(aggregated)

    return {
        "n_nodes": n_nodes,
        "results": aggregated_results,
        "all_trials_by_sample_size": all_results_by_sample_size,
    }


def _run_single_sample_size(
    all_nodes: List[str],
    observed_nodes: List[str],
    edges_dict: Dict,
    latent_nodes: List[str],
    gamma_shapes: Dict[str, float],
    gamma_scales: Dict[str, float],
    Gamma_population_obs: np.ndarray,
    n_samples: int,
    seed: int,
    trial: int,
) -> Dict[str, Any]:
    """Test one polytree with one sample size."""
    try:
        # Generate finite-sample data
        Gamma_finite_obs = _generate_and_compute_finite_discrepancy(
            all_nodes,
            observed_nodes,
            gamma_shapes,
            gamma_scales,
            edges_dict,
            n_samples,
            seed,
            trial,
        )

        # Compute discrepancy error
        discrepancy_error = np.max(np.abs(Gamma_finite_obs - Gamma_population_obs))
        print(f"    Discrepancy error: {discrepancy_error:.6f}")

        # Recover structure
        pred_edges_list, true_edges_list = _recover_and_compare_structure(
            Gamma_finite_obs, observed_nodes, edges_dict
        )

        # Compute metrics
        metrics = _compute_recovery_metrics(pred_edges_list, true_edges_list)

        # Print results
        print(
            f"    Results: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
            f"F1={metrics['f1']:.3f}, Perfect={metrics['perfect']}"
        )

        if not metrics["perfect"]:
            print_discrepancy_comparison(
                Gamma_population_obs,
                Gamma_finite_obs,
                observed_nodes,
                label="    *** IMPERFECT *** ",
            )

            pred_set = set(pred_edges_list)
            true_set = set(true_edges_list)
            missing = sorted(true_set - pred_set)
            extra = sorted(pred_set - true_set)

            if missing:
                print(f"      Missing ({len(missing)}): {missing}")
            if extra:
                print(f"      Extra ({len(extra)}): {extra}")

        return {
            "discrepancy_error": discrepancy_error,
            "recovered_edges": pred_edges_list,
            **metrics,
            "success": True,
        }

    except Exception as e:
        print(f"    FAILED: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def _run_single_trial(
    n_nodes: int, n_samples: int, trial: int, seed: int, n_trials: int
) -> Dict[str, Any]:
    """Run a single trial: generate polytree, sample data, recover structure."""
    print(f"\n  === Trial {trial + 1}/{n_trials} ===")

    try:
        # Generate ground truth
        ground_truth = get_random_polytree_via_pruefer(
            n=n_nodes,
            seed=seed + trial + n_samples,
            weights_range=(-1.0, 1.0),
            avoid_small=0.8,
            ensure_at_least_one_hidden=True,
        )

        # Extract and rename nodes
        edges_dict, latent_nodes, observed_nodes, all_nodes, sigmas, kappas = (
            _extract_and_rename_ground_truth(ground_truth)
        )

        _print_trial_info(latent_nodes, observed_nodes, edges_dict)

        # Convert moments to Gamma parameters
        gamma_shapes, gamma_scales = compute_gamma_parameters_from_moments(
            sigmas, kappas
        )

        # Compute population discrepancy
        Gamma_population_obs = _compute_population_discrepancy(
            edges_dict, sigmas, kappas, latent_nodes
        )

        # Generate finite-sample data and compute discrepancy
        Gamma_finite_obs = _generate_and_compute_finite_discrepancy(
            all_nodes,
            observed_nodes,
            gamma_shapes,
            gamma_scales,
            edges_dict,
            n_samples,
            seed,
            trial,
        )

        # Compute discrepancy error
        discrepancy_error = np.max(np.abs(Gamma_finite_obs - Gamma_population_obs))
        print(f"  Discrepancy error: {discrepancy_error:.6f}")

        # Recover structure
        pred_edges_list, true_edges_list = _recover_and_compare_structure(
            Gamma_finite_obs, observed_nodes, edges_dict
        )

        # Compute metrics
        metrics = _compute_recovery_metrics(pred_edges_list, true_edges_list)

        # Print results and diagnostics
        _print_trial_results(
            metrics,
            Gamma_population_obs,
            Gamma_finite_obs,
            observed_nodes,
            pred_edges_list,
            true_edges_list,
        )

        return {
            "trial_id": trial + 1,
            "ground_truth_edges": sorted(edges_dict.keys()),
            "n_latent": len(latent_nodes),
            "n_observed": len(observed_nodes),
            "latent_nodes": latent_nodes,
            "discrepancy_error": discrepancy_error,
            "recovered_edges": pred_edges_list,
            **metrics,
            "success": True,
        }

    except Exception as e:
        print(f"  Trial {trial + 1} FAILED: {e}")
        return {
            "trial_id": trial + 1,
            "success": False,
            "error": str(e),
        }


def _extract_and_rename_ground_truth(
    ground_truth: Dict[str, Any],
) -> Tuple[Dict, List[str], List[str], List[str], Dict[str, float], Dict[str, float]]:
    """Extract and rename latent nodes from v* to h* format."""
    edges_dict = ground_truth["weights"]
    hidden_nodes_set = set(ground_truth["hidden_nodes"])
    observed_nodes = ground_truth["observed_nodes"]

    # Sort and rename latent nodes
    latent_nodes_old = sorted(hidden_nodes_set, key=lambda x: int(x[1:]))
    node_mapping = {node: f"h{i + 1}" for i, node in enumerate(latent_nodes_old)}

    # Rebuild edges with renamed latent nodes
    renamed_edges_dict = {}
    for (u, v), weight in edges_dict.items():
        u_new = node_mapping.get(u, u)
        v_new = node_mapping.get(v, v)
        renamed_edges_dict[(u_new, v_new)] = weight

    # Rename sigmas and kappas
    sigmas = {}
    kappas = {}
    for old_node in ground_truth["sigmas"]:
        new_node = node_mapping.get(old_node, old_node)
        sigmas[new_node] = ground_truth["sigmas"][old_node]
        kappas[new_node] = ground_truth["kappas"][old_node]

    latent_nodes = [node_mapping[node] for node in latent_nodes_old]
    all_nodes = latent_nodes + observed_nodes

    return renamed_edges_dict, latent_nodes, observed_nodes, all_nodes, sigmas, kappas


def _print_trial_info(
    latent_nodes: List[str], observed_nodes: List[str], edges_dict: Dict
):
    """Print trial ground truth information."""
    print(
        f"  Ground truth: {len(latent_nodes)} latent, {len(observed_nodes)} observed, {len(edges_dict)} edges"
    )
    print(f"    Latent: {latent_nodes}")
    print(f"    Edges: {sorted(edges_dict.keys())}")


def _compute_population_discrepancy(
    edges_dict: Dict,
    sigmas: Dict[str, float],
    kappas: Dict[str, float],
    latent_nodes: List[str],
) -> np.ndarray:
    """Compute population observed discrepancy matrix."""
    Gamma_population_obs, _, _ = observed_gamma_from_params(
        edges_dict,
        sigmas,
        kappas,
        hidden=set(latent_nodes),
        auto_detect_hidden=False,
    )
    return Gamma_population_obs


def _generate_and_compute_finite_discrepancy(
    all_nodes: List[str],
    observed_nodes: List[str],
    gamma_shapes: Dict[str, float],
    gamma_scales: Dict[str, float],
    edges_dict: Dict,
    n_samples: int,
    seed: int,
    trial: int,
) -> np.ndarray:
    """Generate finite-sample data and compute observed discrepancy."""
    # Generate finite-sample data
    trial_seed = seed + hash((trial, n_samples)) % 1000000
    noise_samples = generate_noise_samples(
        all_nodes,
        gamma_shapes,
        gamma_scales,
        n_samples,
        trial_seed,
        verbose=False,
    )

    X_samples = apply_lsem_transformation(
        noise_samples, edges_dict, all_nodes, verbose=False
    )

    Gamma_finite_full = finite_sample_discrepancy(X_samples, all_nodes, verbose=False)

    # Extract observed-only discrepancy matrix
    observed_indices = [all_nodes.index(node) for node in observed_nodes]
    Gamma_finite_obs = Gamma_finite_full[np.ix_(observed_indices, observed_indices)]

    return Gamma_finite_obs


def _recover_and_compare_structure(
    Gamma_finite_obs: np.ndarray, observed_nodes: List[str], edges_dict: Dict
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Recover structure from discrepancy and prepare for comparison."""
    recovered_polytree = get_polytree_algo3(Gamma_finite_obs)

    # Map recovered edges to actual node names
    def map_edge(edge_tuple):
        parent, child = edge_tuple

        def map_node(x):
            if isinstance(x, str) and x.startswith("h"):
                return x
            idx = int(x) if isinstance(x, str) else x
            return observed_nodes[idx]

        return (map_node(parent), map_node(child))

    recovered_edges_raw = [map_edge(edge) for edge in recovered_polytree.edges]

    # Remove redundant degree-2 latents
    recovered_edges_min = _suppress_degree2_latents(recovered_edges_raw)

    # Apply deterministic latent relabeling to BOTH graphs
    pred_edges_list = _relabel_latents_by_topo(sorted(recovered_edges_min))
    true_edges_list = _relabel_latents_by_topo(sorted(edges_dict.keys()))

    print(f"  Recovered edges: {pred_edges_list}")

    return pred_edges_list, true_edges_list


def _compute_recovery_metrics(
    pred_edges: List[Tuple[str, str]], true_edges: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """Compute precision, recall, F1, and perfect recovery."""
    pred_set = set(pred_edges)
    true_set = set(true_edges)

    correct_edges = len(true_set & pred_set)
    total_recovered = len(pred_set)
    total_true = len(true_set)

    precision = correct_edges / total_recovered if total_recovered > 0 else 0
    recall = correct_edges / total_true if total_true > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    perfect = true_set == pred_set

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "perfect": perfect,
    }


def _print_trial_results(
    metrics: Dict,
    Gamma_population_obs: np.ndarray,
    Gamma_finite_obs: np.ndarray,
    observed_nodes: List[str],
    pred_edges: List[Tuple[str, str]],
    true_edges: List[Tuple[str, str]],
):
    """Print trial results and diagnostics if imperfect recovery."""
    print(
        f"  Results: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
        f"F1={metrics['f1']:.3f}, Perfect={metrics['perfect']}"
    )

    if not metrics["perfect"]:
        print_discrepancy_comparison(
            Gamma_population_obs,
            Gamma_finite_obs,
            observed_nodes,
            label="*** IMPERFECT RECOVERY *** ",
        )

        pred_set = set(pred_edges)
        true_set = set(true_edges)
        missing = sorted(true_set - pred_set)
        extra = sorted(pred_set - true_set)

        if missing:
            print(f"    Missing ({len(missing)}): {missing}")
        if extra:
            print(f"    Extra ({len(extra)}): {extra}")


def _aggregate_trial_results(
    successful_trials: List[Dict], n_nodes: int, n_samples: int, n_trials: int
) -> Dict[str, Any]:
    """Aggregate statistics across successful trials."""
    aggregated = {
        "n_nodes": n_nodes,
        "n_samples": n_samples,
        "n_trials_total": n_trials,
        "n_trials_successful": len(successful_trials),
        "n_latent_mean": np.mean([t["n_latent"] for t in successful_trials]),
        "n_latent_std": np.std([t["n_latent"] for t in successful_trials]),
        "n_observed_mean": np.mean([t["n_observed"] for t in successful_trials]),
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
        "perfect_recovery_rate": np.mean([t["perfect"] for t in successful_trials]),
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
    print(f"  F1 Score: {aggregated['f1_mean']:.3f} ± {aggregated['f1_std']:.3f}")
    print(f"  Perfect recovery: {aggregated['perfect_recovery_rate']:.1%}")

    return aggregated


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
    polytree_sizes = [4, 5, 6, 7, 8]
    sample_sizes = [15000000]
    n_trials = 10
    base_seed = 42
    n_latent = 1

    print(f"Configuration:")
    print(f"  Polytree sizes: {polytree_sizes}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Trials per configuration: {n_trials}")
    print(f"  Number of latent nodes: {n_latent}")

    all_results = {}

    for n_nodes in polytree_sizes:
        results = run_finite_sample_for_random_polytree(
            n_nodes=n_nodes,
            sample_sizes=sample_sizes,
            n_trials=n_trials,
            seed=base_seed + n_nodes,
            n_latent=n_latent,
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
