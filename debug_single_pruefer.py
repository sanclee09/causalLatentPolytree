from random_polytrees_pruefer import sample_random_polytree_via_pruefer
from polytree_discrepancy import compute_discrepancy, Polytree, compute_discrepancy_fast
import numpy as np


def main(n=8, seed=2025081314):
    out = sample_random_polytree_via_pruefer(n=n, seed=seed)

    print("=== Ground truth (from Prüfer) ===")
    print("Directed edges:", out["edges_directed"])
    print("Hidden (thesis rule):", out["hidden_nodes"])
    print("Observed:", out["observed_nodes"])

    print("\n=== Γ_obs (population, exact) ===")
    G = out["Gamma_obs"]
    for row in G:
        print("  ", " ".join(f"{x:6.3f}" for x in row))

    print("\n=== Learned polytree from Γ_obs ===")
    print("Recovered edges:", sorted(out["recovered_edges"]))

    # Sanity: recompute population Γ from params and confirm it matches
    poly = Polytree(out["weights"], out["sigmas"], out["kappas"])
    Gamma_full = compute_discrepancy_fast(poly)
    obs = out["observed_nodes"]
    idx = [poly.nodes.index(v) for v in obs]
    Gamma_obs_check = Gamma_full[np.ix_(idx, idx)]
    same = (abs(Gamma_obs_check - G) < 1e-9).all()
    print("\nPopulation Γ re-check equals returned Γ_obs:", same)

    # Quick correctness check for this configuration
    true_obs_edges = {
        (u, v) for (u, v) in out["edges_directed"] if u in obs and v in obs
    }
    rec_obs_edges = {
        (u, v)
        for (u, v) in out["recovered_edges"]
        if (not u.startswith("h")) and (not v.startswith("h"))
    }

    expected_chain_ok = true_obs_edges.issubset(rec_obs_edges)
    print("\nObserved-edge recovery is a superset of truth:", expected_chain_ok)
    if not expected_chain_ok:
        print("  Missing observed edges:", sorted(true_obs_edges - rec_obs_edges))
        print("  Extra observed edges:", sorted(rec_obs_edges - true_obs_edges))

    # Latent-to-observed check: true hidden parents should map to a recovered latent with the same children set
    true_hidden = set(out["hidden_nodes"])
    true_latent_children = {}
    for u, v in out["edges_directed"]:
        if u in true_hidden and v in obs:
            true_latent_children.setdefault(u, set()).add(v)
    rec_latent_children = {}
    for u, v in out["recovered_edges"]:
        if u.startswith("h") and (not v.startswith("h")):
            rec_latent_children.setdefault(u, set()).add(v)

    # Greedy match by Jaccard
    def jacc(a, b):
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    pairs = []
    for t, tc in true_latent_children.items():
        for r, rc in rec_latent_children.items():
            pairs.append((jacc(tc, rc), t, r))
    pairs.sort(reverse=True)
    mapped_t = set()
    mapped_r = set()
    mapping = {t: None for t in true_latent_children}
    for score, t, r in pairs:
        if t in mapped_t or r in mapped_r:
            continue
        if score >= 0.5 or (not true_latent_children[t] and not rec_latent_children[r]):
            mapping[t] = r
            mapped_t.add(t)
            mapped_r.add(r)

    print(
        "\nTrue latent children sets:",
        {k: sorted(v) for k, v in true_latent_children.items()},
    )
    print(
        "Recovered latent children sets:",
        {k: sorted(v) for k, v in rec_latent_children.items()},
    )
    print("Greedy mapping:", mapping)


if __name__ == "__main__":
    main()
