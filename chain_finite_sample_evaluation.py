import importlib
import extended_finite_sample_evaluation
import topology_stratified_evaluation

importlib.reload(topology_stratified_evaluation)
from topology_stratified_evaluation import create_polytree_from_topology


# NOW patch the function IN THE MODULE where it's used
def get_chain_polytree(n, seed, **kwargs):
    """Generate chain polytree instead of random."""
    return create_polytree_from_topology(
        topology_type='chain',
        n=n,
        n_latent=1,
        seed=seed,
    )


# Patch it in the extended_finite_sample_evaluation module namespace
extended_finite_sample_evaluation.get_random_polytree_via_pruefer = get_chain_polytree


def main():
    """Run the exact same pipeline as extended_finite_sample_evaluation.py."""
    print("=" * 70)
    print("CHAIN TOPOLOGY FINITE-SAMPLE EVALUATION")
    print("=" * 70)

    polytree_sizes = [6, 8, 10]
    sample_sizes = [100, 1000, 10000, 100000, 1000000, 10000000, 20000000]
    n_trials = 10
    base_seed = 42

    print(f"\nConfiguration:")
    print(f"  Topology: Chain")
    print(f"  Polytree sizes: {polytree_sizes}")
    print(f"  Sample sizes: {sample_sizes}")
    print(f"  Trials per size: {n_trials}")

    all_results = {}

    for n_nodes in polytree_sizes:
        # Just call the existing function!
        results = extended_finite_sample_evaluation.run_finite_sample_for_random_polytree(
            n_nodes=n_nodes,
            sample_sizes=sample_sizes,
            n_trials=n_trials,
            seed=base_seed + n_nodes,
            n_latent=1,  # Chain always has 1 latent
        )
        all_results[n_nodes] = results

    # Use existing summary function
    extended_finite_sample_evaluation.print_convergence_summary(all_results)

    # Create COMBINED plot with all n values (like thesis Figure 7)
    extended_finite_sample_evaluation.plot_convergence_analysis(all_results)

    # Save results
    import pandas as pd
    summary_data = []
    for n_nodes, data in all_results.items():
        for result in data['results']:
            summary_data.append({
                'topology': 'chain',
                'n_nodes': n_nodes,
                'n_samples': result['n_samples'],
                'f1_mean': result['f1_mean'],
                'f1_std': result['f1_std'],
                'precision_mean': result['precision_mean'],
                'recall_mean': result['recall_mean'],
                'perfect_recovery_rate': result['perfect_recovery_rate'],
                'discrepancy_error_mean': result.get('discrepancy_error_mean', float('nan')),
                'n_trials_successful': result['n_trials_successful'],
                'n_trials_total': result['n_trials_total'],
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("chain_topology_results.csv", index=False)
    print(f"\n✅ Results saved to 'chain_topology_results.csv'")
    print(f"✅ Combined plot saved to 'random_polytree_analysis.png'")

    return all_results, summary_df


if __name__ == "__main__":
    results, summary = main()