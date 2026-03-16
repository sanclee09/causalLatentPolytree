"""
run_experiment.py — unified experiment runner.

Usage:
    python run_experiment.py <topology> [--config experiment_config.yaml]

    topology: chain | balanced | star

Each run creates:
    experiments/<topology>_<YYYYMMDD_HHMMSS>/
        config_snapshot.yaml
        run.log
        <topology>_topology_results.csv
        <topology>_polytree_analysis.pdf
        <topology>_polytree_analysis.png
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml


# ── Tee: write to both terminal and log file simultaneously ──────────────────

class _Tee:
    """Mirrors writes to multiple file-like objects (e.g. stdout + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    def fileno(self):
        return self._streams[0].fileno()


# ── Output directory setup ───────────────────────────────────────────────────

def make_output_dir(topology: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("experiments") / f"{topology}_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── Config helpers ───────────────────────────────────────────────────────────

def load_config(config_path: str, topology: str) -> dict:
    with open(config_path) as f:
        full = yaml.safe_load(f)
    if topology not in full.get("topologies", {}):
        raise ValueError(
            f"Topology '{topology}' not found in {config_path}. "
            f"Available: {list(full['topologies'].keys())}"
        )
    return full["topologies"][topology]


def save_config_snapshot(config_path: str, out_dir: Path, topology: str, cfg: dict) -> None:
    snapshot = {"topology": topology, "parameters": cfg}
    with open(out_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
    shutil.copy(config_path, out_dir / "experiment_config.yaml")


# ── Topology generator factory ───────────────────────────────────────────────

def make_generator(topology: str, n_latent: int):
    """Return a get_polytree function compatible with extended_finite_sample_evaluation."""
    import topology_stratified_evaluation as tse
    importlib.reload(tse)

    # Strip variant suffixes (_k2, _k3) to get the base topology name
    base_topology = topology.split("_")[0]

    def _generator(n, seed, **kwargs):
        return tse.create_polytree_from_topology(
            topology_type=base_topology,
            n=n,
            n_latent=n_latent,
            seed=seed,
        )

    return _generator


# ── Main runner ──────────────────────────────────────────────────────────────

def run(topology: str, config_path: str) -> None:
    cfg = load_config(config_path, topology)

    n_nodes_list: list[int] = cfg["n_nodes"]
    n_latent: int           = cfg["n_latent"]
    n_trials: int           = cfg["n_trials"]
    base_seed: int          = cfg["base_seed"]
    sample_sizes: list[int] = cfg["sample_sizes"]

    out_dir = make_output_dir(topology)
    save_config_snapshot(config_path, out_dir, topology, cfg)

    log_file = open(out_dir / "run.log", "w", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)

    try:
        import extended_finite_sample_evaluation as efe

        # Patch the polytree generator to produce the requested topology
        efe.get_random_polytree_via_pruefer = make_generator(topology, n_latent)

        label = topology.upper()
        print("=" * 70)
        print(f"{label} TOPOLOGY FINITE-SAMPLE EVALUATION")
        print("=" * 70)
        print(f"\nOutput directory : {out_dir.resolve()}")
        print(f"\nConfiguration:")
        print(f"  Topology     : {topology}")
        print(f"  Node counts  : {n_nodes_list}")
        print(f"  n_latent     : {n_latent}")
        print(f"  Sample sizes : {sample_sizes}")
        print(f"  Trials       : {n_trials}")
        print(f"  Base seed    : {base_seed}")

        all_results: dict = {}
        for n_nodes in n_nodes_list:
            results = efe.run_finite_sample_for_random_polytree(
                n_nodes=n_nodes,
                sample_sizes=sample_sizes,
                n_trials=n_trials,
                seed=base_seed + n_nodes,
                n_latent=n_latent,
            )
            all_results[n_nodes] = results

        efe.print_convergence_summary(all_results)

        output_prefix = str(out_dir / f"{topology}_polytree_analysis")
        efe.plot_convergence_analysis(all_results, output_prefix=output_prefix)

        csv_path = str(out_dir / f"{topology}_topology_results.csv")
        efe.save_results_to_csv(all_results, topology=topology, filename=csv_path)

        print(f"\n{'=' * 70}")
        print(f"All outputs saved to: {out_dir.resolve()}")
        print(f"  run.log")
        print(f"  config_snapshot.yaml")
        print(f"  experiment_config.yaml")
        print(f"  {topology}_topology_results.csv")
        print(f"  {topology}_polytree_analysis.pdf")
        print(f"  {topology}_polytree_analysis.png")

    finally:
        sys.stdout = original_stdout
        log_file.close()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a topology-stratified finite-sample evaluation experiment."
    )
    parser.add_argument(
        "topology",
        choices=["chain", "balanced", "star", "balanced_k2", "balanced_k3"],
        help="Topology to evaluate.",
    )
    parser.add_argument(
        "--config",
        default="experiment_config.yaml",
        help="Path to the YAML config file (default: experiment_config.yaml).",
    )
    args = parser.parse_args()
    run(args.topology, args.config)


if __name__ == "__main__":
    main()
