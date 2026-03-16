# Balanced Topology — Multi-Latent Experiment Plan

## Motivation

The existing balanced-topology finite-sample evaluation fixes the number of latent nodes at $k=1$.
This extension investigates how the algorithm scales when the latent structure is richer:
$k \in \{2, 3\}$ latent nodes within balanced topologies of total size $n \in \{10, 20\}$.

The core question: **does additional latent complexity degrade sample complexity, and if so, at what rate?**

---

## How to Run

The experiment uses the existing `run_experiment.py` + `experiment_config.yaml` infrastructure.
New entries `balanced_k2` and `balanced_k3` have been added to the config.

```bash
cd /Users/SancLee/Projects/causalLatentPolytree/causalLatentPolytree

# k=2 latent nodes
python run_experiment.py balanced_k2

# k=3 latent nodes
python run_experiment.py balanced_k3

# k=1 baseline (already done, but re-run for direct comparison if needed)
python run_experiment.py balanced
```

Each run creates a timestamped folder under `experiments/` containing:
- `run.log` — full console output
- `config_snapshot.yaml` — exact parameters used
- `balanced_k{k}_topology_results.csv` — F1 scores per (n, sample size, trial)
- `balanced_k{k}_polytree_analysis.png/.pdf` — convergence plots

---

## Configuration (experiment_config.yaml)

| Parameter | balanced_k2 | balanced_k3 |
|-----------|------------|------------|
| `n_nodes` | [10, 20] | [10, 20] |
| `n_latent` | 2 | 3 |
| `n_trials` | 10 | 10 |
| `base_seed` | 42 | 42 |
| `sample_sizes` | $10^2$–$10^7$ | $10^2$–$10^7$ |

Observed nodes: $n - k$, so 8 and 18 for $k=2$; 7 and 17 for $k=3$.

---

## What to Look For

| Expected pattern | Interpretation |
|-----------------|----------------|
| F1 drops as $k$ increases (fixed $n$, fixed samples) | Richer latent structure is harder to identify |
| Convergence requires more samples for larger $k$ | Sample complexity increases with $k$ |
| F1 drops as $n$ increases (fixed $k$, fixed samples) | Larger graphs need more data |

Also check: does the $n^{-1/2}$ convergence rate (established for $k=1$) persist for $k=2,3$?

---

## Potential Issues

1. **Topology generation failures:** With $k>1$, `generate_balanced_topology` may retry more.
   If generation is slow, reduce `n_trials` to 5 temporarily.

2. **Memory:** $10^7$ samples × $n=20$ nodes can be heavy. If needed, cap `sample_sizes`
   at `1000000` for $n=20$ runs by editing the config.

---

## Thesis Integration

After collecting results, update:

### New subsection in Section 6 (Experiments)
Add: **"Effect of Latent Node Count on Sample Complexity"**
- F1 tables for $(n, k) \in \{10, 20\} \times \{1, 2, 3\}$
- Convergence plot comparing $k=1,2,3$ side by side

### Section 7 (Summary) and Section 8 (Conclusion)
- Mention the multi-latent experiment as an extension of the $k=1$ results
- State whether convergence rate appears robust to $k$
- Flag degradation (if any) as an open problem

### Table format (one per $k$)

| $n_{\text{samples}}$ | $n=10, k$ | $n=20, k$ |
|----------------------|-----------|-----------|
| $10^2$ | F1 ± std | F1 ± std |
| $\vdots$ | | |
| $10^7$ | F1 ± std | F1 ± std |
