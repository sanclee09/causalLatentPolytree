from __future__ import annotations
from typing import Dict, Tuple, List, Set, Optional
import random
import numpy as np

from random_polytrees_pruefer import get_random_polytree_via_pruefer


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def greedy_match_true_to_recovered(
    true_children: Dict[str, Set[str]],
    rec_children: Dict[str, Set[str]],
) -> Dict[str, Optional[str]]:
    """
    Greedy bipartite matching between true hidden nodes and recovered latent nodes
    maximizing Jaccard similarity of observed-child sets.
    Returns mapping true_hidden -> recovered_h (or None).
    """
    pairs = []
    for t, tc in true_children.items():
        for r, rc in rec_children.items():
            pairs.append((jaccard(tc, rc), t, r))
    pairs.sort(reverse=True)  # highest score first

    mapped_true = set()
    mapped_rec = set()
    mapping: Dict[str, Optional[str]] = {t: None for t in true_children}
    for score, t, r in pairs:
        if t in mapped_true or r in mapped_rec:
            continue
        if score >= 0.5 or (not true_children[t] and not rec_children[r]):
            mapping[t] = r
            mapped_true.add(t)
            mapped_rec.add(r)
    return mapping


def evaluate_one(seed: int, n: int = 8) -> Dict[str, float]:
    res = get_random_polytree_via_pruefer(n=n, seed=seed)
    obs = set(res["observed_nodes"])
    hid_true = set(res["hidden_nodes"])
    edges_true_dir = set(res["edges_directed"])
    edges_rec = set(res["recovered_edges"])

    # ---- Build truth sets for evaluation ----
    true_latent_children: Dict[str, Set[str]] = {}
    for u, v in edges_true_dir:
        if u in hid_true and v in obs:
            true_latent_children.setdefault(u, set()).add(v)

    rec_latent_children: Dict[str, Set[str]] = {}
    for u, v in edges_rec:
        if u.startswith("h") and (not v.startswith("h")):
            rec_latent_children.setdefault(u, set()).add(v)

    mapping = greedy_match_true_to_recovered(true_latent_children, rec_latent_children)

    # Compute precision/recall on latent->observed edges
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


def run_experiments(K: int = 100, n: int = 8, seed: int = 1) -> Dict[str, float]:
    rng = random.Random(seed)
    metrics = []
    for _ in range(K):
        m = evaluate_one(seed=rng.randint(0, 10**9), n=n)
        metrics.append(m)

    keys = metrics[0].keys() if metrics else []
    avg = {f"avg_{k}": float(np.mean([m[k] for m in metrics])) for k in keys}
    std = {f"std_{k}": float(np.std([m[k] for m in metrics], ddof=1)) for k in keys}
    return {**avg, **std}, metrics


if __name__ == "__main__":
    import csv, json, time
    from pathlib import Path

    n = 100  # number of nodes
    K = 50  # number of trials
    seed = 42  # random seed
    out_dir = Path("pruefer_eval")  # output folder

    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    summary, runs = run_experiments(K=K, n=n, seed=seed)
    end = time.time()
    elapsed = end - start

    # pretty print summary
    print("\n=== Summary (latent only) ===")
    for k in sorted(summary.keys()):
        print(f"{k:>24}: {summary[k]:.4f}")

    print(f"\nRuntime: {elapsed:.2f} seconds (n={n}, K={K})")

    # save summary.json
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # save runs.csv
    if runs:
        keys = sorted(runs[0].keys())
        with (out_dir / "runs.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in runs:
                w.writerow(r)
        print(f"\nWrote: {out_dir/'summary.json'} and {out_dir/'runs.csv'}")
    else:
        print("\nNo runs produced (K=0?).")
