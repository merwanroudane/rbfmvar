"""
Example 1: Basic RBFM-VAR Estimation
====================================

This example demonstrates basic usage of the RBFM-VAR package for
estimating VAR models with unknown mixtures of I(0), I(1), and I(2)
components.

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com

Reference: Chang (2000), Econometric Theory 16(6), 905-926
"""

import numpy as np
import matplotlib.pyplot as plt

# Import the package
from rbfmvar import RBFMVAR, generate_dgp


def main():
    """Run basic RBFM-VAR estimation example."""
    
    print("=" * 70)
    print("RBFM-VAR: Basic Estimation Example")
    print("=" * 70)
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate data using the DGP from Chang (2000), Case C
    # This case has:
    # - y1 is I(1)
    # - y2 is I(2)  
    # - y2 Granger-causes y1
    print("Generating data from Case C DGP...")
    print("- y1 is I(1)")
    print("- y2 is I(2)")
    print("- y2 Granger-causes y1")
    print()
    
    T = 200  # Sample size
    y, dgp_info = generate_dgp('case_c', T=T, seed=42)
    
    # Print true parameter values
    print("True Parameter Values:")
    print(f"  ρ₁ = {dgp_info['rho1']}")
    print(f"  ρ₂ = {dgp_info['rho2']}")
    print(f"\nTrue Π₁ (coefficient for Δy_{{t-1}}):")
    print(dgp_info['Pi1'])
    print(f"\nTrue Π₂ (coefficient for y_{{t-1}}):")
    print(dgp_info['Pi2'])
    print()
    
    # Plot the data
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Levels
    axes[0, 0].plot(y[:, 0], label='y₁')
    axes[0, 0].plot(y[:, 1], label='y₂')
    axes[0, 0].set_title('Levels')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Time')
    
    # First differences
    dy = np.diff(y, axis=0)
    axes[0, 1].plot(dy[:, 0], label='Δy₁')
    axes[0, 1].plot(dy[:, 1], label='Δy₂')
    axes[0, 1].set_title('First Differences')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Time')
    
    # Second differences
    d2y = np.diff(y, n=2, axis=0)
    axes[1, 0].plot(d2y[:, 0], label='Δ²y₁')
    axes[1, 0].plot(d2y[:, 1], label='Δ²y₂')
    axes[1, 0].set_title('Second Differences')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Time')
    
    # y1 vs y2 scatter
    axes[1, 1].scatter(y[:, 1], y[:, 0], alpha=0.5)
    axes[1, 1].set_xlabel('y₂')
    axes[1, 1].set_ylabel('y₁')
    axes[1, 1].set_title('y₁ vs y₂')
    
    plt.tight_layout()
    plt.savefig('example1_data.png', dpi=150, bbox_inches='tight')
    print("Data plot saved to 'example1_data.png'")
    plt.close()
    
    # Fit RBFM-VAR model
    print("\n" + "=" * 70)
    print("Fitting RBFM-VAR Model")
    print("=" * 70)
    
    model = RBFMVAR(
        lag_order=1,
        bandwidth='auto',  # Automatic bandwidth selection
        kernel='bartlett'  # Bartlett kernel (most common)
    )
    
    results = model.fit(y)
    
    # Print summary
    print(results.summary())
    
    # Compare OLS-VAR and RBFM-VAR estimates
    print("\n" + "=" * 70)
    print("Comparison: OLS-VAR vs RBFM-VAR")
    print("=" * 70)
    print(results.compare_with_ols())
    
    # Granger causality test
    print("\n" + "=" * 70)
    print("Granger Causality Test")
    print("=" * 70)
    print("\nH₀: y₂ does not Granger-cause y₁")
    print()
    
    # Modified Wald test (conservative, valid for nonstationary systems)
    gc_modified = results.granger_causality_test(
        caused_variable=0,
        causing_variables=[1],
        test_type='modified'
    )
    
    print("Modified Wald Test (RBFM-VAR):")
    print(f"  Test Statistic: {gc_modified['statistic']:.4f}")
    print(f"  P-value:        {gc_modified['p_value']:.4f}")
    print(f"  Degrees of Freedom: {gc_modified['df']}")
    print(f"  Conservative:   {gc_modified['is_conservative']}")
    print()
    
    # Standard Wald test (for comparison - may be oversized)
    gc_standard = results.granger_causality_test(
        caused_variable=0,
        causing_variables=[1],
        test_type='standard'
    )
    
    print("Standard Wald Test (OLS-VAR):")
    print(f"  Test Statistic: {gc_standard['statistic']:.4f}")
    print(f"  P-value:        {gc_standard['p_value']:.4f}")
    print()
    
    print("Note: The modified Wald test is asymptotically conservative,")
    print("meaning the actual size is bounded by the nominal size.")
    print("The standard Wald test may be oversized in nonstationary settings.")
    
    # Forecasting
    print("\n" + "=" * 70)
    print("Forecasting")
    print("=" * 70)
    
    forecasts = results.forecast(steps=5)
    print("\nOut-of-sample forecasts (5 steps ahead):")
    print(f"{'Step':<6} {'y₁ Forecast':<15} {'y₂ Forecast':<15}")
    print("-" * 40)
    for i, fc in enumerate(forecasts):
        print(f"{i+1:<6} {fc[0]:<15.4f} {fc[1]:<15.4f}")
    
    # Export to LaTeX
    print("\n" + "=" * 70)
    print("LaTeX Export")
    print("=" * 70)
    
    latex_table = results.to_latex()
    with open('example1_results.tex', 'w') as f:
        f.write(latex_table)
    print("\nLaTeX table saved to 'example1_results.tex'")
    
    print("\n" + "=" * 70)
    print("Example Complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
