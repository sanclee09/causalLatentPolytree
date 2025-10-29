"""
metrics.py

Shared metrics computation for polytree structure recovery evaluation.
"""

from typing import Set, Tuple, List, Dict, Any


def compute_precision_recall_f1(
        predicted_edges: Set[Tuple[str, str]],
        true_edges: Set[Tuple[str, str]]
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score for edge recovery.

    Args:
        predicted_edges: Set of predicted (parent, child) edges
        true_edges: Set of true (parent, child) edges

    Returns:
        Dictionary with 'precision', 'recall', 'f1', and 'perfect' keys
    """
    if not predicted_edges:
        precision = 0.0
        recall = 0.0 if true_edges else 1.0
        f1 = 0.0
    elif not true_edges:
        precision = 0.0
        recall = 1.0
        f1 = 0.0
    else:
        true_positives = len(predicted_edges & true_edges)
        precision = true_positives / len(predicted_edges)
        recall = true_positives / len(true_edges)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    perfect = (predicted_edges == true_edges)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "perfect": perfect
    }


def compute_structure_distance(
        predicted_edges: Set[Tuple[str, str]],
        true_edges: Set[Tuple[str, str]]
) -> Dict[str, Any]:
    """
    Compute structural distance metrics between predicted and true graphs.

    Returns:
        Dictionary with distance metrics including missing, extra, and hamming distance
    """
    missing_edges = true_edges - predicted_edges
    extra_edges = predicted_edges - true_edges
    hamming_distance = len(missing_edges) + len(extra_edges)

    return {
        "missing_edges": list(missing_edges),
        "extra_edges": list(extra_edges),
        "n_missing": len(missing_edges),
        "n_extra": len(extra_edges),
        "hamming_distance": hamming_distance
    }


def aggregate_metrics_across_trials(trial_results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple trials.

    Args:
        trial_results: List of metric dictionaries from individual trials

    Returns:
        Dictionary with mean and std for each metric
    """
    import numpy as np

    if not trial_results:
        return {}

    metrics = {}

    for key in ['precision', 'recall', 'f1']:
        values = [r[key] for r in trial_results if key in r]
        if values:
            metrics[f"{key}_mean"] = float(np.mean(values))
            metrics[f"{key}_std"] = float(np.std(values))

    # Perfect recovery rate
    if 'perfect' in trial_results[0]:
        perfect_count = sum(1 for r in trial_results if r.get('perfect', False))
        metrics['perfect_recovery_rate'] = perfect_count / len(trial_results)

    return metrics


def print_metrics_summary(metrics: Dict[str, float], prefix: str = "") -> None:
    """Print formatted metrics summary."""
    print(f"{prefix}Precision: {metrics.get('precision', 0):.3f}")
    print(f"{prefix}Recall:    {metrics.get('recall', 0):.3f}")
    print(f"{prefix}F1 Score:  {metrics.get('f1', 0):.3f}")

    if 'perfect' in metrics:
        print(f"{prefix}Perfect:   {metrics['perfect']}")