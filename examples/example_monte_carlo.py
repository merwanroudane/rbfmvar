"""
Example 2: Monte Carlo Simulation Study
========================================

This example replicates the Monte Carlo simulation study from Section 5
of Chang (2000), comparing OLS-VAR and RBFM-VAR estimators.

The simulation considers three cases:
- Case A: Both variables are I(2), no cointegration, no Granger causality
- Case B: y₁ is I(1), y₂ is I(2), no Granger causality  
- Case C: y₁ is I(1), y₂ is I(2), y₂ Granger-causes y₁

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com

Reference: Chang (2000), Econometric Theory 16(6), 905-926, Tables 1-2
"""

import numpy as np
import matplotlib.pyplot as plt

from rbfmvar import (
    monte_carlo_simulation,
    generate_dgp,
    RBFMVAR
)


def main():
    """Run Monte Carlo simulation example."""
    
    print("=" * 80)
    print("RBFM-VAR: Monte Carlo Simulation Study")
    print("Replicating Chang (2000), Section 5")
    print("=" * 80)
    print()
    
    # Simulation parameters
    T = 150           # Sample size (as in the paper)
    n_reps = 500      # Number of replications (use 10000 for full replication)
    seed = 42         # Random seed
    
    print(f"Simulation Parameters:")
    print(f"  Sample Size (T):        {T}")
    print(f"  Number of Replications: {n_reps}")
    print(f"  Random Seed:            {seed}")
    print()
    
    # Run simulations for all three cases
    all_results = {}
    
    for case in ['case_a', 'case_b', 'case_c']:
        print("\n" + "=" * 80)
        print(f"Running {case.upper()}")
        print("=" * 80)
        
        # Get DGP info
        _, info = generate_dgp(case, T=10)
        print(f"\nDGP: {info['case']}")
        print(f"  ρ₁ = {info['rho1']:.2f}")
        print(f"  ρ₂ = {info['rho2']:.2f}")
        print(f"  Granger causality: {'Yes' if info['has_causality'] else 'No'}")
        print()
        
        # Run simulation
        results = monte_carlo_simulation(
            dgp=case,
            T=T,
            n_reps=n_reps,
            bandwidth='auto',
            kernel='bartlett',
            test_causality=True,
            seed=seed,
            verbose=True
        )
        
        all_results[case] = results
        
        # Print summary
        print(results.summary())
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    cases = ['case_a', 'case_b', 'case_c']
    case_labels = ['Case A', 'Case B', 'Case C']
    
    # Plot bias comparisons
    for idx, case in enumerate(cases):
        results = all_results[case]
        
        # Get biases for Π₁₁₂ and Π₂₁₂ (the causality coefficients)
        ols_bias = results.compute_bias('ols').flatten()
        rbfm_bias = results.compute_bias('rbfm').flatten()
        
        x = np.arange(len(ols_bias))
        width = 0.35
        
        axes[0, idx].bar(x - width/2, np.abs(ols_bias) * np.sqrt(T), width, 
                        label='OLS-VAR', alpha=0.7)
        axes[0, idx].bar(x + width/2, np.abs(rbfm_bias) * np.sqrt(T), width,
                        label='RBFM-VAR', alpha=0.7)
        axes[0, idx].set_title(f'{case_labels[idx]}: |Bias| × √T')
        axes[0, idx].set_xlabel('Coefficient')
        axes[0, idx].legend()
        axes[0, idx].set_xticks(x)
        
        # Plot RMSE comparisons
        ols_rmse = results.compute_rmse('ols').flatten()
        rbfm_rmse = results.compute_rmse('rbfm').flatten()
        
        axes[1, idx].bar(x - width/2, ols_rmse * np.sqrt(T), width,
                        label='OLS-VAR', alpha=0.7)
        axes[1, idx].bar(x + width/2, rbfm_rmse * np.sqrt(T), width,
                        label='RBFM-VAR', alpha=0.7)
        axes[1, idx].set_title(f'{case_labels[idx]}: RMSE × √T')
        axes[1, idx].set_xlabel('Coefficient')
        axes[1, idx].legend()
        axes[1, idx].set_xticks(x)
    
    plt.tight_layout()
    plt.savefig('example2_mc_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to 'example2_mc_comparison.png'")
    plt.close()
    
    # Create Table 2 (test sizes and power)
    print("\n" + "=" * 80)
    print("Table 2: Finite Sample Sizes and Rejection Probabilities")
    print("=" * 80)
    print()
    
    print(f"{'Case':<10} {'Test':<15} {'1% test':<12} {'5% test':<12} {'10% test':<12}")
    print("-" * 60)
    
    for case in cases:
        results = all_results[case]
        
        # OLS Wald test
        print(f"{case.upper():<10} {'W_F (OLS)':<15} "
              f"{results.compute_rejection_rate('ols', 0.01):<12.3f} "
              f"{results.compute_rejection_rate('ols', 0.05):<12.3f} "
              f"{results.compute_rejection_rate('ols', 0.10):<12.3f}")
        
        # Modified Wald test
        print(f"{'':<10} {'W_F⁺ (RBFM)':<15} "
              f"{results.compute_rejection_rate('rbfm', 0.01):<12.3f} "
              f"{results.compute_rejection_rate('rbfm', 0.05):<12.3f} "
              f"{results.compute_rejection_rate('rbfm', 0.10):<12.3f}")
        print()
    
    print("\nNotes:")
    print("- Cases A and B: These are SIZE results (H₀ is true)")
    print("  - Rejection rates should be close to nominal levels")
    print("  - OLS Wald test tends to over-reject (size distortion)")
    print("  - Modified Wald test maintains better size control")
    print()
    print("- Case C: These are POWER results (H₀ is false)")
    print("  - Higher rejection rates indicate better power")
    print()
    
    # Export LaTeX table
    print("=" * 80)
    print("LaTeX Table Export")
    print("=" * 80)
    
    for case in cases:
        results = all_results[case]
        latex = results.to_latex_table()
        filename = f'example2_table1_{case}.tex'
        with open(filename, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to '{filename}'")
    
    print("\n" + "=" * 80)
    print("Simulation Study Complete")
    print("=" * 80)
    print()
    print("Key findings (consistent with Chang 2000):")
    print("1. The OLS Wald test has serious size distortions in Cases A and B")
    print("2. The modified Wald test based on RBFM-VAR has better size control")
    print("3. Both tests have power to detect causality in Case C")
    print("4. RBFM-VAR generally has smaller bias and variance than OLS-VAR")


if __name__ == '__main__':
    main()
