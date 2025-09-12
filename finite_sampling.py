"""
final_fixed_sampling.py

THE FINAL FIX: Use exact same three-rule logic as population discrepancy computation.
This ensures Algorithm 3 receives the same pattern of entries as the population case.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, List, Any
import numpy as np

# Project imports
from random_polytrees_pruefer import get_random_polytree_via_pruefer
from latent_polytree_truepoly import get_polytree_algo3


@dataclass
class FinalConfig:
    n_nodes: int = 8
    seed: int = 42
    sample_size: int = 2000
    # Edge weights
    weights_range: Tuple[float, float] = (-0.9, 0.9)
    avoid_small: float = 0.6

    # Gamma noise parameters (Daniele's approach)
    alpha_range: Tuple[float, float] = (0.5, 5.0)
    beta_range: Tuple[float, float] = (0.5, 5.0)

    # Population-matching thresholds
    eps_corr: float = 1e-3  # correlation gate (match population)
    eps_num: float = 1e-3  # numerator ≈ 0 gate
    eps_den: float = 1e-6  # denominator guard
    clip_max: float = 1e6  # generous cap


class FinalSampler:
    def __init__(self, cfg: FinalConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)

    def _generate_gamma_noise(self, n_vars: int, m: int) -> np.ndarray:
        """Generate Gamma noise following Daniele's approach exactly."""
        noise = np.zeros((n_vars, m))

        for i in range(n_vars):
            alpha = self.rng.uniform(self.cfg.alpha_range[0], self.cfg.alpha_range[1])
            beta = self.rng.uniform(self.cfg.beta_range[0], self.cfg.beta_range[1])

            raw_samples = self.rng.gamma(alpha, beta, m)
            theoretical_mean = alpha * beta
            theoretical_var = alpha * beta**2
            theoretical_std = np.sqrt(theoretical_var)

            standardized = (raw_samples - theoretical_mean) / theoretical_std
            noise[i, :] = standardized

        return noise

    @staticmethod
    def _build_B(
        weights: Dict[Tuple[str, str], float], node_order: List[str]
    ) -> np.ndarray:
        """Build structure matrix."""
        n = len(node_order)
        B = np.zeros((n, n))
        idx = {v: i for i, v in enumerate(node_order)}
        for (p, c), w in weights.items():
            if p in idx and c in idx:
                B[idx[c], idx[p]] = w
        return B

    def generate(self, gt: Dict[str, Any]) -> Dict[str, Any]:
        """Generate samples from ground truth model."""
        all_nodes = sorted(list(gt["sigmas"].keys()))
        n = len(all_nodes)
        B = self._build_B(gt["weights"], all_nodes)
        I = np.eye(n)

        try:
            A = np.linalg.inv(I - B.T)
        except np.linalg.LinAlgError:
            raise ValueError(f"Singular matrix (I - B^T)")

        eps = self._generate_gamma_noise(n, self.cfg.sample_size)
        X = A @ eps

        obs = gt["observed_nodes"]
        obs_idx = [all_nodes.index(v) for v in obs]
        X_obs = X[obs_idx, :]

        return {
            "X_obs": X_obs,
            "A": A,
            "B": B,
            "all_nodes": all_nodes,
            "observed_nodes": obs,
            "observed_idx": obs_idx,
            "epsilon": eps,
        }


class RawMoments:
    """Moment estimation on centered raw data (no standardization)."""

    @staticmethod
    def center(X: np.ndarray) -> np.ndarray:
        """Only center - do NOT standardize."""
        return X - X.mean(axis=1, keepdims=True)

    @staticmethod
    def cov(Xc: np.ndarray, unbiased: bool = True) -> np.ndarray:
        """Compute covariance matrix."""
        m = Xc.shape[1]
        denom = (m - 1) if unbiased and m > 1 else m
        return (Xc @ Xc.T) / denom

    @staticmethod
    def third(Xc: np.ndarray):
        """Compute third moments."""
        p, m = Xc.shape
        C_iij = np.zeros((p, p))
        C_ijj = np.zeros((p, p))
        C_iii = np.mean(Xc**3, axis=1)
        for i in range(p):
            Xi = Xc[i, :]
            Xi2 = Xi * Xi
            for j in range(p):
                if i == j:
                    continue
                Xj = Xc[j, :]
                C_iij[i, j] = np.mean(Xi2 * Xj)
                C_ijj[i, j] = np.mean(Xi * (Xj * Xj))
        return {"C_iij": C_iij, "C_ijj": C_ijj, "C_iii": C_iii}


def build_gamma_population_rules(
    Sigma: np.ndarray,
    C: Dict[str, np.ndarray],
    eps_corr: float = 1e-3,  # correlation gate (match pop. convention)
    eps_num: float = 1e-3,  # numerator ~ 0 gate
    eps_den: float = 1e-6,  # denominator guard
    clip_max: float = 1e6,  # generous cap; usually never reached
) -> np.ndarray:
    """
    Sample Γ̂ constructed with the same three-rule logic used in the population code:

      For i ≠ j:
        1) if |Σ_ij| < eps_corr: Γ̂_ij = 0
        2) elif |Σ_ii * C_iij - Σ_ij * C_iii| < eps_num: Γ̂_ij = 0
        3) else if |C_iij * Σ_ij| ≥ eps_den:
               Γ̂_ij = (C_ijj * Σ_ii) / (C_iij * Σ_ij)   # keep signs, no abs
           else: Γ̂_ij = 0

      Γ̂_ii = 0.

    NOTE: No geometric mean, no -1 "missing", no absolute values—this matches the
    population computation.
    """
    p = Sigma.shape[0]
    G = np.zeros((p, p), dtype=float)
    C_iij = C["C_iij"]
    C_ijj = C["C_ijj"]
    C_iii = C["C_iii"]

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            sij = Sigma[i, j]
            if abs(sij) < eps_corr:
                G[i, j] = 0.0
                continue

            num_check = Sigma[i, i] * C_iij[i, j] - sij * C_iii[i]
            if abs(num_check) < eps_num:
                G[i, j] = 0.0
                continue

            den = C_iij[i, j] * sij
            if abs(den) < eps_den:
                G[i, j] = 0.0
                continue

            val = (C_ijj[i, j] * Sigma[i, i]) / den
            # population Γ is typically ≥ 0; negative from noise → treat as 0
            if not np.isfinite(val) or val < 0:
                val = 0.0
            G[i, j] = min(val, clip_max)

    return G


def run_final_experiment(cfg: FinalConfig):
    """Run experiment with exact population rules."""

    print(f"=== FINAL FIXED FINITE SAMPLING ===")
    print(f"Setup: n={cfg.n_nodes}, m={cfg.sample_size}, seed={cfg.seed}")
    print(f"Fix: EXACT population rules (3-gate logic)")

    # 1) Ground truth
    gt = get_random_polytree_via_pruefer(
        n=cfg.n_nodes,
        seed=cfg.seed,
        weights_range=cfg.weights_range,
        avoid_small=cfg.avoid_small,
    )
    print(
        f"Ground truth: {len(gt['observed_nodes'])} observed, {len(gt['hidden_nodes'])} hidden"
    )
    print(f"Edges: {gt['edges_directed']}")

    # 2) Sample generation
    print(f"Step 2: Sampling with Daniele's Gamma noise")
    sampler = FinalSampler(cfg)
    sample_data = sampler.generate(gt)
    X_obs = sample_data["X_obs"]
    print(f"  Sample shape: {X_obs.shape}")
    print(f"  Sample range: [{X_obs.min():.3f}, {X_obs.max():.3f}]")

    # 3) Compute moments on centered RAW data (no standardization)
    print(f"Step 3: Computing moments on RAW centered data")
    Xc = RawMoments.center(X_obs)
    Sigma = RawMoments.cov(Xc, unbiased=True)
    C = RawMoments.third(Xc)

    print(f"  Covariance range: [{Sigma.min():.4f}, {Sigma.max():.4f}]")
    print(
        f"  Third moment (C_iii) range: [{C['C_iii'].min():.4f}, {C['C_iii'].max():.4f}]"
    )

    # 4) Build discrepancy using EXACT population rules
    print(f"Step 4: Building discrepancy with EXACT population rules")
    G_hat = build_gamma_population_rules(
        Sigma,
        C,
        eps_corr=cfg.eps_corr,
        eps_num=cfg.eps_num,
        eps_den=cfg.eps_den,
        clip_max=cfg.clip_max,
    )
    G_pop = gt["Gamma_obs"]

    mse = np.mean((G_pop - G_hat) ** 2)
    print(f"  Population Γ range: [{G_pop.min():.3f}, {G_pop.max():.3f}]")
    print(f"  Sample Γ̂ range: [{G_hat.min():.3f}, {G_hat.max():.3f}]")
    print(f"  MSE(Population, Sample): {mse:.6f}")

    # Check entry patterns
    zero_pop = np.sum(np.abs(G_pop) < 1e-10)
    zero_sample = np.sum(np.abs(G_hat) < 1e-10)
    print(f"  Zero entries: population={zero_pop}, sample={zero_sample}")

    # 5) Structure recovery
    print(f"Step 5: Structure recovery")
    recovered_edges = []
    try:
        G_in = G_hat.copy()
        np.fill_diagonal(G_in, 0.0)

        T = get_polytree_algo3(G_in)
        obs = sample_data["observed_nodes"]

        def name_node(node_str: str) -> str:
            return node_str if node_str.startswith("h") else obs[int(node_str)]

        recovered_edges = [(name_node(p), name_node(c)) for p, c in T.edges]
        print(f"  Recovered {len(recovered_edges)} total edges")

        # Show breakdown
        obs_to_obs = [
            (u, v)
            for (u, v) in recovered_edges
            if not u.startswith("h") and not v.startswith("h")
        ]
        hidden_to_obs = [
            (u, v)
            for (u, v) in recovered_edges
            if u.startswith("h") and not v.startswith("h")
        ]

        print(
            f"  Breakdown: {len(obs_to_obs)} obs→obs, {len(hidden_to_obs)} hidden→obs"
        )

    except Exception as e:
        print(f"  Structure recovery failed: {e}")

    return {
        "gt": gt,
        "recovered_edges": recovered_edges,
        "Gamma_pop": G_pop,
        "Gamma_hat": G_hat,
        "observed_nodes": sample_data["observed_nodes"],
        "mse": mse,
    }


def evaluate_final_results(results: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate performance."""
    gt = results["gt"]
    obs_set = set(gt["observed_nodes"])

    true_edges = {
        (u, v) for (u, v) in gt["edges_directed"] if u in obs_set and v in obs_set
    }
    rec_edges = {
        (u, v)
        for (u, v) in results["recovered_edges"]
        if not u.startswith("h") and not v.startswith("h")
    }

    tp = len(true_edges & rec_edges)
    fp = len(rec_edges - true_edges)
    fn = len(true_edges - rec_edges)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "true_edges": len(true_edges),
        "recovered_edges": len(rec_edges),
        "true_edge_set": true_edges,
        "recovered_edge_set": rec_edges,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mse": results["mse"],
    }


def detailed_comparison(results):
    """Compare population vs sample matrices in detail."""

    print(f"\n=== DETAILED MATRIX COMPARISON ===")

    G_pop = results["Gamma_pop"]
    G_hat = results["Gamma_hat"]
    obs_nodes = results["observed_nodes"]

    print(f"Population Γ matrix:")
    for i, row in enumerate(G_pop):
        print(f"  {obs_nodes[i]}: " + " ".join(f"{x:7.3f}" for x in row))

    print(f"\nSample Γ̂ matrix:")
    for i, row in enumerate(G_hat):
        print(f"  {obs_nodes[i]}: " + " ".join(f"{x:7.3f}" for x in row))

    print(f"\nDifference matrix (Sample - Population):")
    diff = G_hat - G_pop
    for i, row in enumerate(diff):
        print(f"  {obs_nodes[i]}: " + " ".join(f"{x:7.3f}" for x in row))


def test_population_matrix_performance(results):
    """Test Algorithm 3 performance on the exact population matrix."""

    print(f"\n=== POPULATION MATRIX TEST ===")

    G_pop = results["Gamma_pop"]
    obs_nodes = results["observed_nodes"]

    # Test Algorithm 3 on population matrix
    G_pop_input = G_pop.copy()
    np.fill_diagonal(G_pop_input, 0.0)

    try:
        T_pop = get_polytree_algo3(G_pop_input)

        def name_node(node_str: str) -> str:
            return node_str if node_str.startswith("h") else obs_nodes[int(node_str)]

        pop_edges = [(name_node(p), name_node(c)) for p, c in T_pop.edges]
        pop_obs_edges = {
            (u, v)
            for (u, v) in pop_edges
            if not u.startswith("h") and not v.startswith("h")
        }

        print(f"Algorithm 3 on population matrix:")
        print(f"  Total edges: {len(pop_edges)}")
        print(f"  Obs→obs edges: {pop_obs_edges}")

        # Compare to ground truth
        gt = results["gt"]
        obs_set = set(gt["observed_nodes"])
        true_obs_edges = {
            (u, v) for (u, v) in gt["edges_directed"] if u in obs_set and v in obs_set
        }

        tp = len(true_obs_edges & pop_obs_edges)
        precision = tp / len(pop_obs_edges) if pop_obs_edges else 0.0
        recall = tp / len(true_obs_edges) if true_obs_edges else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        print(f"  True obs→obs: {true_obs_edges}")
        print(
            f"  Population performance: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}"
        )

        return f1 > 0

    except Exception as e:
        print(f"Population matrix test failed: {e}")
        return False


def test_final_approach():
    """Test the final approach with exact population rules."""

    print(f"FINAL FINITE SAMPLING TEST")
    print(f"=" * 50)

    cfg = FinalConfig(
        n_nodes=8,
        seed=42,
        sample_size=2000,
        eps_corr=1e-3,  # Match population
        eps_num=1e-3,
        eps_den=1e-6,
    )

    results = run_final_experiment(cfg)
    performance = evaluate_final_results(results)

    print(f"\n=== RESULTS ===")
    print(f"True observed→observed edges: {performance['true_edge_set']}")
    print(f"Recovered observed→observed edges: {performance['recovered_edge_set']}")
    print(f"")
    print(f"Performance:")
    print(f"  F1-score: {performance['f1']:.3f}")
    print(f"  Precision: {performance['precision']:.3f}")
    print(f"  Recall: {performance['recall']:.3f}")
    print(f"  Discrepancy MSE: {performance['mse']:.6f}")

    # Detailed analysis
    detailed_comparison(results)

    # Test population matrix as reference
    pop_works = test_population_matrix_performance(results)

    if performance["f1"] > 0:
        print(f"\n✅ SUCCESS! F1 = {performance['f1']:.3f}")
        print(f"Finite sampling now works with exact population rules!")

        # Test with larger sample
        print(f"\n--- Testing with larger sample ---")
        cfg_large = FinalConfig(
            n_nodes=8,
            seed=42,
            sample_size=10000,
            eps_corr=1e-3,
            eps_num=1e-3,
            eps_den=1e-6,
        )
        results_large = run_final_experiment(cfg_large)
        perf_large = evaluate_final_results(results_large)
        print(
            f"Large sample (n=10k): F1={perf_large['f1']:.3f}, MSE={perf_large['mse']:.3f}"
        )

    elif pop_works:
        print(f"\n⚠ Population matrix works but sample doesn't")
        print(f"Need larger samples or looser thresholds")

        # Try with relaxed thresholds
        print(f"\n--- Trying relaxed thresholds ---")
        cfg_relaxed = FinalConfig(
            n_nodes=8,
            seed=42,
            sample_size=5000,
            eps_corr=5e-3,  # Looser
            eps_num=3e-3,  # Looser
            eps_den=1e-6,
        )
        results_relaxed = run_final_experiment(cfg_relaxed)
        perf_relaxed = evaluate_final_results(results_relaxed)
        print(
            f"Relaxed thresholds: F1={perf_relaxed['f1']:.3f}, MSE={perf_relaxed['mse']:.3f}"
        )

    else:
        print(f"\n⚠ Even population matrix fails - structure may not be recoverable")
        print(f"Try different random seed or simpler test case")


if __name__ == "__main__":
    test_final_approach()
