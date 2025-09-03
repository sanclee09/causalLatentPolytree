#!/usr/bin/env python3
"""
numerical_stabilization.py

Order-preserving reparameterization for numerical stability in polytree learning.
Implements the log-normalization approach for stabilizing discrepancy matrices.

Theoretical foundation:
- Preserves all order relations and equality patterns in Γ
- Reduces dynamic range from exponential to logarithmic scale
- Maintains structural zeros exactly
- No hyperparameter tuning required

For TUM thesis: "Linear non-Gaussian models with latent polytree structure"
"""

import numpy as np
from typing import Tuple, Dict, Any


def normalize_gamma(G, *, ridge=0.0, eps=None, return_meta=False):
    G = np.asarray(G, dtype=float)
    # robust single global scale: median of nonzeros by magnitude
    nz = np.abs(G[np.nonzero(G)])
    s = float(np.median(nz)) if nz.size else 1.0
    if eps is None:
        eps = float(np.finfo(G.dtype).tiny)

    # strictly monotone, sign-preserving
    def phi(x):
        return np.sign(x) * np.log1p(np.abs(x) / (s + eps))

    def inv(y):
        return np.sign(y) * (s * np.expm1(np.abs(y)))

    Gn = phi(G)
    if ridge:
        Gn = Gn + float(ridge) * np.eye(G.shape[0])

    if return_meta:
        return Gn, {"scale": s, "eps": eps, "ridge": float(ridge), "inverse": inv}
    return Gn


def compute_stability_metrics(
    G_before: np.ndarray, G_after: np.ndarray
) -> Dict[str, float]:
    """
    Compute numerical stability metrics before/after normalization.

    Args:
        G_before: Original discrepancy matrix
        G_after: Normalized discrepancy matrix

    Returns:
        Dictionary of stability metrics
    """

    def matrix_metrics(G):
        pos_vals = G[G > 0]
        if len(pos_vals) == 0:
            return {
                "condition_number": 1.0,
                "dynamic_range": 1.0,
                "tie_rate": 0.0,
                "min_positive": 0.0,
                "max_value": 0.0,
            }

        # Condition number (for symmetric case, use eigenvalues)
        try:
            if G.shape[0] == G.shape[1]:
                eigenvals = np.linalg.eigvals(G + G.T)  # Symmetrize for stability
                eigenvals = eigenvals[eigenvals > 1e-15]  # Filter near-zero
                cond_num = (
                    np.max(eigenvals) / np.min(eigenvals)
                    if len(eigenvals) > 0
                    else np.inf
                )
            else:
                cond_num = np.linalg.cond(G)
        except:
            cond_num = np.inf

        # Dynamic range
        min_pos = np.min(pos_vals)
        max_val = np.max(pos_vals)
        dynamic_range = max_val / min_pos if min_pos > 0 else np.inf

        # Tie rate: fraction of near-equal pairs
        n = G.shape[0]
        tie_count = 0
        total_pairs = 0

        for i in range(n):
            row_vals = G[i, G[i] > 0]
            if len(row_vals) > 1:
                for j in range(len(row_vals)):
                    for k in range(j + 1, len(row_vals)):
                        diff = abs(row_vals[j] - row_vals[k])
                        # Tie if difference is within machine precision
                        tie_count += diff <= 10 * np.finfo(float).eps * max(
                            row_vals[j], row_vals[k]
                        )
                        total_pairs += 1

        tie_rate = tie_count / total_pairs if total_pairs > 0 else 0.0

        return {
            "condition_number": float(cond_num),
            "dynamic_range": float(dynamic_range),
            "tie_rate": float(tie_rate),
            "min_positive": float(min_pos),
            "max_value": float(max_val),
        }

    before_metrics = matrix_metrics(G_before)
    after_metrics = matrix_metrics(G_after)

    return {
        "before_condition_number": before_metrics["condition_number"],
        "after_condition_number": after_metrics["condition_number"],
        "before_dynamic_range": before_metrics["dynamic_range"],
        "after_dynamic_range": after_metrics["dynamic_range"],
        "before_tie_rate": before_metrics["tie_rate"],
        "after_tie_rate": after_metrics["tie_rate"],
        "condition_improvement": before_metrics["condition_number"]
        / after_metrics["condition_number"],
        "range_compression": before_metrics["dynamic_range"]
        / after_metrics["dynamic_range"],
        "tie_reduction": before_metrics["tie_rate"] - after_metrics["tie_rate"],
    }


def validate_order_preservation(
    G_before: np.ndarray, G_after: np.ndarray, n_samples: int = 1000
) -> Dict[str, bool]:
    """
    Validate that order relations are preserved after normalization.

    Tests the fundamental invariance property:
    G[i,j] ≤ G[i,k] ⟺ φ(G)[i,j] ≤ φ(G)[i,k]

    Args:
        G_before: Original matrix
        G_after: Normalized matrix
        n_samples: Number of random triplets (i,j,k) to test

    Returns:
        Dictionary with validation results
    """
    n = G_before.shape[0]
    if n < 3:
        return {"order_preserved": True, "equality_preserved": True, "tests_run": 0}

    rng = np.random.RandomState(42)  # Deterministic for reproducibility

    order_violations = 0
    equality_violations = 0
    zero_violations = 0

    for _ in range(n_samples):
        # Random triplet
        i = rng.randint(0, n)
        j, k = rng.choice([x for x in range(n) if x != i], size=2, replace=False)

        # Test order preservation
        before_order = G_before[i, j] <= G_before[i, k]
        after_order = G_after[i, j] <= G_after[i, k]
        if before_order != after_order:
            order_violations += 1

        # Test equality preservation (with tolerance)
        eps = 1e-12
        before_equal = abs(G_before[i, j] - G_before[i, k]) <= eps
        after_equal = abs(G_after[i, j] - G_after[i, k]) <= eps
        if before_equal != after_equal:
            equality_violations += 1

        # Test zero preservation
        if (G_before[i, j] == 0) != (G_after[i, j] == 0):
            zero_violations += 1

    return {
        "order_preserved": order_violations == 0,
        "equality_preserved": equality_violations == 0,
        "zeros_preserved": zero_violations == 0,
        "order_violations": order_violations,
        "equality_violations": equality_violations,
        "zero_violations": zero_violations,
        "tests_run": n_samples,
    }


def demonstrate_stabilization(G_raw: np.ndarray) -> None:
    """
    Demonstrate the stabilization effect with before/after analysis.

    Args:
        G_raw: Raw discrepancy matrix
    """
    print("NUMERICAL STABILIZATION ANALYSIS")
    print("=" * 50)

    # Apply normalization
    G_normalized = normalize_gamma(G_raw)

    # Compute metrics
    stability_metrics = compute_stability_metrics(G_raw, G_normalized)
    validation_results = validate_order_preservation(G_raw, G_normalized)

    # Display results
    print(f"Matrix shape: {G_raw.shape}")
    print(f"Non-zero entries: {np.sum(G_raw > 0)}")
    print()

    print("NUMERICAL STABILITY IMPROVEMENTS:")
    print(
        f"  Condition number: {stability_metrics['before_condition_number']:.2e} → {stability_metrics['after_condition_number']:.2e}"
    )
    print(f"  Improvement factor: {stability_metrics['condition_improvement']:.2e}")
    print(
        f"  Dynamic range: {stability_metrics['before_dynamic_range']:.2e} → {stability_metrics['after_dynamic_range']:.2e}"
    )
    print(f"  Compression factor: {stability_metrics['range_compression']:.2e}")
    print(
        f"  Tie rate: {stability_metrics['before_tie_rate']:.4f} → {stability_metrics['after_tie_rate']:.4f}"
    )
    print(f"  Tie reduction: {stability_metrics['tie_reduction']:.4f}")
    print()

    print("INVARIANCE VALIDATION:")
    print(
        f"  Order relations preserved: {'✓' if validation_results['order_preserved'] else '✗'}"
    )
    print(
        f"  Equality patterns preserved: {'✓' if validation_results['equality_preserved'] else '✗'}"
    )
    print(
        f"  Structural zeros preserved: {'✓' if validation_results['zeros_preserved'] else '✗'}"
    )
    print(f"  Tests performed: {validation_results['tests_run']}")
    print()

    if not all(
        [
            validation_results["order_preserved"],
            validation_results["equality_preserved"],
            validation_results["zeros_preserved"],
        ]
    ):
        print("⚠️  WARNING: Invariance violations detected!")
        print(f"    Order violations: {validation_results['order_violations']}")
        print(f"    Equality violations: {validation_results['equality_violations']}")
        print(f"    Zero violations: {validation_results['zero_violations']}")
    else:
        print("✅ All invariance properties verified")


# Integration helper for existing codebase
def apply_stabilization(polytree_sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply stabilization to an existing polytree sample.

    Args:
        polytree_sample: Dictionary from get_random_polytree_via_pruefer

    Returns:
        Modified sample with stabilized Γ_obs and metrics
    """
    G_raw = polytree_sample["Gamma_obs"]
    G_stabilized = normalize_gamma(G_raw)

    # Update the sample
    result = polytree_sample.copy()
    result["Gamma_obs"] = G_stabilized
    result["Gamma_raw"] = G_raw  # Keep original for comparison

    # Add stabilization metrics
    result["stabilization_metrics"] = compute_stability_metrics(G_raw, G_stabilized)
    result["invariance_validation"] = validate_order_preservation(G_raw, G_stabilized)

    return result


if __name__ == "__main__":
    # Demonstration with synthetic ill-conditioned matrix
    print("Testing with synthetic ill-conditioned matrix...")

    # Create a matrix with extreme dynamic range (similar to your n=100 case)
    n = 100
    G = np.zeros((n, n))

    # Fill with exponentially growing values to simulate the real problem
    for i in range(n):
        for j in range(n):
            if i != j:
                # Exponential growth with some structure
                G[i, j] = np.exp(abs(i - j) * 2.0) * (1 + 0.1 * np.sin(i + j))

    # Add some near-zeros and exact zeros
    G[G < 0.01] = 0
    G[::3, ::3] = 0  # Structural zeros

    print(f"Synthetic matrix dynamic range: {np.max(G) / np.min(G[G > 0]):.2e}")

    demonstrate_stabilization(G)
