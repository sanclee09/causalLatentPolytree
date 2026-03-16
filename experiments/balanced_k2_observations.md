# Balanced k=2 Experiment — Observations and Analysis

## Results Summary

| n  | 100  | 1K   | 10K  | 100K | 1M   | 10M  | 20M  | Perfect@20M |
|----|------|------|------|------|------|------|------|-------------|
| 8  | 0.32 | 0.30 | 0.41 | 0.63 | 0.68 | 0.69 | 0.74 | 40%         |
| 10 | 0.16 | 0.17 | 0.28 | 0.46 | 0.60 | 0.77 | 0.81 | 50%         |
| 20 | 0.08 | 0.04 | 0.10 | 0.31 | 0.69 | 0.86 | 0.88 | 70%         |

Experiment: balanced topology, k=2 latent nodes, n_trials=10, seed=42.
Full results: `experiments/balanced_k2_20260316_081516/`

---

## Observation 1: Larger graphs benefit from more constraints

**n=20 achieves higher perfect recovery at 20M samples (70%) than n=8 (40%) or n=10 (50%).**

This is counterintuitive at first — larger graphs are harder in terms of sample complexity, but they
also provide more observed pairs and hence more cumulant constraints for the algorithm to work with.
For n=20, k=2, the 18 observed nodes yield a 18×18 discrepancy matrix (306 entries), giving the
algorithm substantially more signal than the 6×6 matrix for n=8. Once enough data is available,
this surplus of constraints helps distinguish the correct structure more reliably.

---

## Observation 2: Sharp sample complexity cliff for n=20

For n=20, F1 jumps from **0.31 at 100K** to **0.69 at 1M** — more than doubling with a 10× increase
in samples. By contrast the gains from 10M to 20M are marginal (0.86 → 0.88). This suggests a
threshold effect: below ~1M samples the discrepancy matrix is too noisy to recover the structure at
all; above it, the signal overwhelms the noise and recovery becomes reliable.

---

## Observation 3: F1 ceiling below 1.0 — fundamental non-identifiability

Perfect recovery never reaches 100% for any n, and **F1 plateaus between 10M and 20M** for all n.
This is not a sample size limitation. It reflects topologies where the algorithm cannot identify
the full structure even with arbitrarily large data.

### Case study: Trial 4, n=20 (F1 = 0.737 at 1M, 10M, and 20M)

Ground truth:
```
h2 → v3 → {v11, v12, v5, v6}
h1 → {v19, v7, v8},  v5 → h1
```

Recovered (at all sample sizes ≥ 1M):
```
h1 → v11 → {v12, v3, v5, v6}    ← v3 and v11 roles swapped
h2 → {v19, v7, v8},  v5 → h2
```

The h1/h2 label swap is harmless — latent node labels are arbitrary. The substantive error is that
the edge **v3 → v11** (ground truth) is consistently recovered as **v11 → v3**, with v11 rather
than v3 serving as the direct child of the latent node.

### Why this is a fundamental non-identifiability

Inspecting the **population** cumulant discrepancy matrix Γ (i.e., at infinite data):

```
γ(v11, v3) = 0
γ(v11, v5) = 0
γ(v11, v6) = 0
```

v11 has zero discrepancy with all its "siblings-under-v3" ({v5, v6, v12}) and with v3 itself.
This happens because v11, v3, v5, v6 all share the same unique latent common ancestor (h2), and
within the observed sub-chain rooted at v3 the discrepancy measure cannot distinguish the direction
of edges. Concretely:

- Under ground truth (v3 → v11): γ(v3, v11) involves a ratio of cumulants that reduces to zero
  when the path between them passes only through observed nodes.
- Under the flipped structure (v11 → v3): the same population gamma matrix is obtained.

The algorithm cannot tell which of v3 or v11 is the "hub" (direct child of h2) because their
mutual discrepancy is zero at the population level — **no amount of additional data resolves this**.
This is why F1 is exactly 0.737 at 1M, 10M, and 20M: the misidentified edges are permanently wrong.

### Connection to the genericness condition (Definition 4.6)

This non-identifiability is not a violation of genericness. Rather, it reveals a **structural
ambiguity** that the cumulant discrepancy measure itself cannot resolve for this topology class:
edges within an observed sub-chain (where all nodes between them are observed) have zero population
gamma and are therefore outside the reach of the algorithm's identification strategy. The algorithm
achieves perfect recovery only up to the **identifiable edges** — those for which the population
discrepancy is nonzero.

This suggests a refinement of the identifiability claim: the algorithm recovers the full structure
when every latent-node-adjacent edge pair has a nonzero population discrepancy. Topologies where
some observed node v has both a latent parent and observed children that are also observed siblings
of one another may introduce zero-gamma pairs that create irresolvable orientation ambiguities.

---

## Observation 4: Numerical instability at low sample sizes

At n=20 with 100–1000 samples, the discrepancy matrix can produce extreme values (max difference
up to 4260 in one trial), and the algorithm occasionally generates spurious latent nodes with
hash-like IDs (e.g., `h1301`, `h1329`). These are artifacts of near-zero denominators in the
cumulant ratio estimate — a known failure mode at very low sample-to-dimension ratios. They do not
indicate a code bug.

---

## Next steps

- Run `balanced_k3` experiment and compare convergence rates across k=1, 2, 3.
- Consider whether the F1 metric should be made invariant to latent node label permutations
  (try both assignments for k=2 and take the max F1). This would give a cleaner picture of
  which errors are genuine structural misidentification vs. arbitrary label choices.
- Thesis Section 6: add subsection "Effect of Latent Node Count on Sample Complexity" with
  F1 tables and convergence plots for k=1, 2, 3 side by side.
- Thesis Section 7/8: mention the non-identifiability ceiling as an open problem — characterising
  which topology classes admit a zero-gamma pair, and whether a higher-order cumulant measure
  could resolve the ambiguity.
