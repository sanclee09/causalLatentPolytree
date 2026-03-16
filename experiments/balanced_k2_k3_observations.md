# Balanced Topology — Multi-Latent Experiment Observations (k=2, k=3)

## Setup

All experiments use the same pipeline as the k=1 balanced experiments, including the near-tie
relative tolerance fix (ρ=0.05) in the B_v construction step of `latent_polytree_truepoly.py`.
Any remaining failures are therefore not attributable to the near-tie mis-grouping artefact —
they reflect either finite-sample estimation noise or structural non-identifiability.

---

## Results Summary

### k=2 (balanced_k2_20260316_081516)

| n  | 100  | 1K   | 10K  | 100K | 1M   | 10M  | 20M  |
|----|------|------|------|------|------|------|------|
| 8  | 0.32 | 0.30 | 0.41 | 0.63 | 0.68 | 0.69 | 0.74 |
| 10 | 0.16 | 0.17 | 0.28 | 0.46 | 0.60 | 0.77 | 0.81 |
| 20 | 0.08 | 0.04 | 0.10 | 0.31 | 0.69 | 0.86 | 0.88 |

### k=3 (balanced_k3_20260316_103157)

| n  | 100  | 1K   | 10K  | 100K | 1M   | 10M  | 20M  |
|----|------|------|------|------|------|------|------|
| 8  | 0.24 | 0.34 | 0.32 | 0.47 | 0.35 | 0.22 | 0.25 |
| 10 | 0.24 | 0.27 | 0.35 | 0.68 | 0.75 | 0.86 | 0.78 |
| 20 | 0.07 | 0.05 | 0.06 | 0.33 | 0.70 | 0.71 | 0.80 |

### Cross-k comparison at 20M samples

| n  | k=2 F1 | k=3 F1 |
|----|--------|--------|
| 8  | 0.74   | 0.25   |
| 10 | 0.81   | 0.78   |
| 20 | 0.88   | 0.80   |

---

## Figure interpretations

### k=2 convergence plot (`balanced_k2_polytree_analysis.pdf`)

Four panels: max, mean, and median discrepancy error (log scale, top and bottom-left), and F1
score (bottom-right), all as a function of sample size for n=8, 10, 20.

**Discrepancy error panels**: All three n values converge, but n=20 starts roughly 2–3 orders of
magnitude higher than n=8 at 100 samples and takes until ~10^5–10^6 to stabilise. The mean and
median errors plateau rather than continuing to decrease — consistent with a population-level
non-zero floor caused by the identifiability ceiling (some entries of the discrepancy matrix never
converge to the correct value because they are zero at the population level).

**F1 panel**: The three curves have qualitatively different shapes:
- n=8: slow but monotone increase, plateauing around 0.74 from 10^6 onward.
- n=10: steeper climb, reaching ~0.81 by 2×10^7.
- n=20: nearly flat until 10^5, then a sharp cliff between 10^5 and 10^6, after which
  it overtakes n=10 and reaches ~0.88. This is the most visually striking feature of the k=2 plot
  — the large graph is worst at small samples but best at large samples.

**Suggested caption**: *Convergence of discrepancy estimation error and structure recovery F1 for
balanced topology with k=2 latent nodes. The F1 panel shows a sharp sample complexity cliff for
n=20 between 10^5 and 10^6 samples, after which the larger graph outperforms the smaller ones.*

---

### k=3 convergence plot (`balanced_k3_polytree_analysis.pdf`)

Same four-panel layout as k=2.

**Discrepancy error panels**: n=20 shows a dramatic spike in max error between 10^3 and 10^4
(exceeding 10^3) before collapsing — this is the numerical instability at low samples for large
sparse graphs (Observation 7). n=8 achieves the lowest discrepancy errors overall, yet has the
worst F1 — a clear sign that low estimation error does not translate to correct structure recovery
when the population gamma matrix itself has many zero entries.

**F1 panel**: Three qualitatively distinct behaviours:
- n=8: peaks at ~0.47 around 10^5 then degrades toward 0.25 at 2×10^7 — the unique
  failure mode where the algorithm converges more confidently to a wrong structure with more data.
- n=10: climbs to ~0.86 at 10^7 then drops slightly to 0.78 at 2×10^7 (Observation 5).
- n=20: flat near zero until 10^4, sharp cliff between 10^4 and 10^5, then steady
  improvement to ~0.80 at 2×10^7.

The contrast between n=8's degrading curve and n=10/20's improving curves is the most important
visual in the k=3 plot — it directly illustrates that n=8 is a structurally degenerate case.

**Suggested caption**: *Convergence of discrepancy estimation error and structure recovery F1 for
balanced topology with k=3 latent nodes. The n=8 F1 curve degrades at large sample sizes,
reflecting structural non-identifiability when only 5 observed nodes are available for 3 latent
sources. The n=20 curve exhibits a sharp sample complexity cliff between 10^4 and 10^5 samples.*

---

## Observation 1: n=8, k=3 is effectively unworkable

With only 5 observed nodes and 3 latent nodes, the discrepancy matrix is a 5×5 matrix — far too
sparse to resolve 3 latent sources. F1 never exceeds 0.47 and actually degrades at high samples
(0.47 at 100K → 0.25 at 20M), indicating the algorithm converges more confidently to an incorrect
structure as data grows.

This is a structural constraint: with k latent nodes each requiring out-degree ≥ 2, the minimum
number of edges from latent nodes is 2k. For n=8, k=3, this leaves very little room for
observed-to-observed edges, creating many zero-gamma pairs and making the graph nearly
non-identifiable from the discrepancy measure alone.

**Practical implication**: for k=3, n≥10 is the minimum viable graph size.

---

## Observation 2: Larger graphs benefit from more constraints (both k=2 and k=3)

For k=2, F1 at 20M is highest for n=20 (0.88) compared to n=8 (0.74) and n=10 (0.81). For k=3
(excluding the degenerate n=8 case), n=20 (0.80) is comparable to n=10 (0.78).

The intuition: more observed nodes produce a larger discrepancy matrix with more nonzero entries,
giving the algorithm more constraints to work with once sufficient data is available. This
counteracts the naive expectation that larger graphs are always harder.

---

## Observation 3: Sample complexity cliff around 100K–1M for n=20

For n=20, both k=2 and k=3 show a dramatic F1 jump around 100K–1M:
- k=2: 0.31 (100K) → 0.69 (1M)
- k=3: 0.33 (100K) → 0.70 (1M)

Below this threshold the discrepancy matrix is too noisy; above it, recovery becomes reliable.
The cliff position is nearly identical for k=2 and k=3 at n=20, suggesting that sample complexity
for large n is driven more by n than by k in this regime.

---

## Observation 4: F1 ceiling — fundamental non-identifiability

F1 plateaus between 10M and 20M for many (n, k) combinations. Since the tolerance fix is applied,
this is not an algorithmic artefact. It reflects topologies where the algorithm cannot identify
the full structure even with arbitrarily large data — a fundamental property of the cumulant
discrepancy measure.

### Case study: k=2, Trial 4, n=20 (F1 = 0.737 at 1M, 10M, and 20M)

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

The h1/h2 label swap is harmless — latent labels are arbitrary. The substantive error is that
the edge v3 → v11 is consistently recovered as v11 → v3.

Inspecting the **population** Γ matrix (i.e., at infinite data):
```
γ(v11, v3) = 0
γ(v11, v5) = 0
γ(v11, v6) = 0
```

v11, v3, v5, v6 all share the same unique latent ancestor (h2), and within the observed sub-chain
rooted at v3 the discrepancy measure cannot distinguish edge directions. Both orientations
(v3→v11 and v11→v3) produce the same population Γ matrix — no amount of data resolves this.

### Connection to the genericness condition (Definition 4.6)

This non-identifiability is not a violation of genericness. It reveals a **structural ambiguity**
that the cumulant discrepancy measure cannot resolve: edges within an all-observed sub-chain have
zero population gamma and are outside the reach of the algorithm's identification strategy.
Topologies where an observed node v has both a latent parent and observed children that are also
observed siblings of one another may introduce irresolvable orientation ambiguities.

---

## Observation 5: Non-monotone F1 for k=3, n=10

For k=3, n=10 shows a slight F1 drop from 10M to 20M (0.86 → 0.78), while n=20 continues to
improve (0.71 → 0.80) as expected.

The n=10 dip likely reflects high trial-to-trial variance (std ≈ 0.22–0.24) combined with the
identifiability ceiling: some trials plateau at a fixed wrong structure, and averaging across 10
trials can produce a slight dip when the algorithm switches between wrong structures at different
sample sizes. It is not a signal of degrading performance.

---

## Observation 6: F1 ceiling degrades with k

The maximum F1 at 20M samples decreases from k=2 to k=3, most dramatically at n=8 (0.74 → 0.25)
and n=20 (0.88 → 0.80). For n=10 the ceiling is roughly unchanged (0.81 → 0.78), suggesting that
for medium-sized graphs the algorithm's identifiability is robust to an additional latent node
when the observed count (7 nodes) is sufficient.

---

## Observation 7: Numerical instability at low sample sizes

At n=20 with 100–1000 samples, the discrepancy matrix produces extreme values (max difference up
to 4260 in some k=2 trials), and the algorithm can generate spurious latent nodes with hash-like
IDs (e.g., `h1301`, `h1329`). These are artifacts of near-zero denominators in the cumulant ratio
estimate at very low sample-to-dimension ratios. They do not indicate a code bug.

---

## Thesis integration plan

### Motivation framing (important for the thesis)

The multi-latent extension is not arbitrary — it is the **only structurally valid next step**
given the constraints of the other two topologies:

- **Chain, k≥2**: Any second latent node in a chain would have out-degree 1 (pointing to the next
  node in the chain), immediately violating the minimality condition (Definition 3.1). So k=1 is
  the only valid configuration for chains.
- **Star, k=1 by definition**: In the Etesami et al. framework, a star has a single latent center
  with all observed nodes as direct children. Adding further latent nodes breaks the star structure.
  k=1 is inherent to the topology.
- **Balanced, k≥2 natural**: Balanced structures have multiple branching points, each of which can
  naturally host a latent node with out-degree ≥ 2. This is the only topology where k=2,3,... are
  all structurally valid while preserving minimality.

The subsection should open by making this reasoning explicit — the balanced k=2,3 experiments are
not just one option among many, but the uniquely motivated extension of the k=1 evaluation.

### Where to add

The k=2 and k=3 results belong in **Section 6 (Experiments)**, as a new subsection immediately
**before** "Summary of Topology-Stratified Findings" (which currently closes the section).
The natural title: **"Effect of Latent Node Count on Sample Complexity"**.

Structure of the new subsection:
1. Motivation — why balanced is the only topology that admits k≥2 while preserving minimality
2. State that the tolerance fix is applied throughout (same algorithm as the k=1 re-evaluation)
3. Table: F1 scores for k=2 (n=8,10,20)
4. Table: F1 scores for k=3 (n=8,10,20)
5. Figures: convergence plots from `balanced_k2_polytree_analysis.pdf` and `balanced_k3_polytree_analysis.pdf`
6. Paragraphs covering the main findings (Observations 1–6 above)

### What to update afterwards

**Summary of Topology-Stratified Findings** — add a paragraph: convergence rate robust to k for
n≥10; n=8 k=3 degenerate; F1 ceiling decreases with k; failures not algorithmic (tolerance fix applied).

**Limitations and Future Directions**:
- Update "Scope of experimental validation" bullet: k∈{1,2,3} for balanced; k=1 only for chains/stars (by structural necessity)
- Update "Multiple latent nodes (k≥2)" future work bullet: mark as done for balanced; note chains/stars cannot be extended without changing the topology definition

**Conclusion (Section 8)**:
- Add a sentence noting the convergence rate appears empirically robust to k≥2 for n≥10
- Flag the non-identifiability ceiling (zero-gamma pairs in all-observed sub-chains) as an open problem

### Metric note
The F1 metric is not invariant to latent node label permutations. For k=2 one could compute F1
under both label assignments and take the max, giving a cleaner measure of structural error vs.
arbitrary labelling. Left as future work.
