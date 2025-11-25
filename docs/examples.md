---
layout: default
title: Examples
---

# Examples

[← Back to Home](.)

Detailed examples demonstrating RBFM-VAR capabilities.

---

## Table of Contents

1. [Basic Estimation](#example-1-basic-estimation)
2. [Real Data Analysis](#example-2-real-data-analysis)
3. [Granger Causality Testing](#example-3-granger-causality-testing)
4. [Monte Carlo Simulation](#example-4-monte-carlo-simulation)
5. [Comparison: RBFM-VAR vs OLS-VAR](#example-5-comparison-study)
6. [Multiple Variable System](#example-6-multiple-variables)
7. [Publication-Ready Output](#example-7-publication-ready-output)

---

## Example 1: Basic Estimation

A complete example from data generation to results.

```python
import numpy as np
import matplotlib.pyplot as plt
from rbfmvar import RBFMVAR, generate_dgp

# Set seed for reproducibility
np.random.seed(42)

# Generate data from Case C DGP
# - y₁ is I(1)
# - y₂ is I(2)
# - y₂ Granger-causes y₁
y, info = generate_dgp('case_c', T=200)

print(f"DGP: {info['case']}")
print(f"True ρ₁ = {info['rho1']}")
print(f"True ρ₂ = {info['rho2']}")
print(f"Data shape: {y.shape}")

# Plot the data
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Levels
axes[0, 0].plot(y[:, 0], label='y₁', color='blue')
axes[0, 0].plot(y[:, 1], label='y₂', color='red')
axes[0, 0].set_title('Levels')
axes[0, 0].legend()
axes[0, 0].set_xlabel('Time')

# First differences
dy = np.diff(y, axis=0)
axes[0, 1].plot(dy[:, 0], label='Δy₁', color='blue')
axes[0, 1].plot(dy[:, 1], label='Δy₂', color='red')
axes[0, 1].set_title('First Differences')
axes[0, 1].legend()

# Second differences
d2y = np.diff(y, n=2, axis=0)
axes[1, 0].plot(d2y[:, 0], label='Δ²y₁', color='blue')
axes[1, 0].plot(d2y[:, 1], label='Δ²y₂', color='red')
axes[1, 0].set_title('Second Differences')
axes[1, 0].legend()

# Scatter plot
axes[1, 1].scatter(y[:, 1], y[:, 0], alpha=0.5)
axes[1, 1].set_xlabel('y₂')
axes[1, 1].set_ylabel('y₁')
axes[1, 1].set_title('y₁ vs y₂')

plt.tight_layout()
plt.savefig('data_plot.png', dpi=150)
plt.show()

# Fit RBFM-VAR model
model = RBFMVAR(lag_order=1, bandwidth='auto', kernel='bartlett')
results = model.fit(y)

# Print summary
print(results.summary())

# Compare with true values
print("\nTrue Π₁:")
print(info['Pi1'])
print("\nEstimated Π₁ (RBFM-VAR):")
print(results.Pi1)
```

---

## Example 2: Real Data Analysis

Analyzing real macroeconomic data.

```python
import numpy as np
import pandas as pd
from rbfmvar import RBFMVAR

# Load your data (example with CSV)
# df = pd.read_csv('macro_data.csv', parse_dates=['date'], index_col='date')

# For this example, create synthetic macro data
np.random.seed(123)
T = 200

# Simulate GDP (I(1)) and Interest Rate (I(1))
gdp = np.cumsum(np.random.randn(T) * 0.5 + 0.1)
interest = np.cumsum(np.random.randn(T) * 0.2)

# Add some relationship
for t in range(1, T):
    gdp[t] += -0.1 * interest[t-1]

# Combine into array
y = np.column_stack([gdp, interest])
var_names = ['GDP', 'Interest_Rate']

print(f"Data shape: {y.shape}")
print(f"Variables: {var_names}")

# Fit model
model = RBFMVAR(lag_order=2)  # VAR(2)
results = model.fit(y)

# Summary with variable names
print("=" * 70)
print("RBFM-VAR Results")
print("=" * 70)
print(f"\nVariables: {var_names}")
print(f"Sample size: {y.shape[0]}")
print(f"Lag order: 2")

print("\nCoefficient Estimates (RBFM-VAR):")
print(results.coefficients)

print(f"\nR-squared: {results.r_squared}")
print(f"AIC: {results.aic:.2f}")
print(f"BIC: {results.bic:.2f}")

# Test if Interest Rate Granger-causes GDP
gc = results.granger_causality_test(
    caused_variable=0,     # GDP
    causing_variables=[1], # Interest Rate
    test_type='modified'
)

print("\n" + "=" * 70)
print("Granger Causality Test")
print("=" * 70)
print(f"H₀: Interest Rate does not Granger-cause GDP")
print(f"Statistic: {gc['statistic']:.4f}")
print(f"P-value: {gc['p_value']:.4f}")

if gc['p_value'] < 0.05:
    print("→ Reject H₀ at 5% level: Interest Rate Granger-causes GDP")
else:
    print("→ Cannot reject H₀ at 5% level")
```

---

## Example 3: Granger Causality Testing

Comprehensive causality analysis.

```python
import numpy as np
from rbfmvar import RBFMVAR, generate_dgp

np.random.seed(42)

# Test all three DGP cases
cases = ['case_a', 'case_b', 'case_c']
results_dict = {}

print("=" * 70)
print("Granger Causality Analysis Across DGPs")
print("=" * 70)

for case in cases:
    # Generate data
    y, info = generate_dgp(case, T=300)
    
    # Fit model
    model = RBFMVAR(lag_order=1)
    results = model.fit(y)
    
    # Test causality: y₂ → y₁
    gc_modified = results.granger_causality_test(0, [1], test_type='modified')
    gc_standard = results.granger_causality_test(0, [1], test_type='standard')
    
    results_dict[case] = {
        'info': info,
        'gc_modified': gc_modified,
        'gc_standard': gc_standard
    }
    
    print(f"\n{case.upper()}: {info['case']}")
    print(f"  True causality: {'Yes' if info['has_causality'] else 'No'}")
    print(f"  Modified Wald p-value: {gc_modified['p_value']:.4f}")
    print(f"  Standard Wald p-value: {gc_standard['p_value']:.4f}")

# Summary table
print("\n" + "=" * 70)
print("Summary: Rejection at 5% Level")
print("=" * 70)
print(f"{'Case':<10} {'True':<10} {'Modified':<12} {'Standard':<12}")
print("-" * 50)

for case in cases:
    r = results_dict[case]
    true = 'Causality' if r['info']['has_causality'] else 'No'
    mod_reject = 'Reject' if r['gc_modified']['p_value'] < 0.05 else 'No reject'
    std_reject = 'Reject' if r['gc_standard']['p_value'] < 0.05 else 'No reject'
    print(f"{case:<10} {true:<10} {mod_reject:<12} {std_reject:<12}")

print("\nNote: Cases A and B have no true causality (these are SIZE tests)")
print("      Case C has true causality (this is a POWER test)")
```

---

## Example 4: Monte Carlo Simulation

Replicate the simulation study from Chang (2000).

```python
import numpy as np
from rbfmvar import monte_carlo_simulation, generate_dgp

print("=" * 70)
print("Monte Carlo Simulation Study")
print("Replicating Chang (2000), Section 5")
print("=" * 70)

# Parameters
T = 150           # Sample size
n_reps = 500      # Use 10000 for paper replication
seed = 42

# Run simulation for Case C (has causality)
print("\nRunning simulation for Case C...")
print(f"T = {T}, Replications = {n_reps}")

results = monte_carlo_simulation(
    dgp='case_c',
    T=T,
    n_reps=n_reps,
    bandwidth='auto',
    kernel='bartlett',
    test_causality=True,
    seed=seed,
    verbose=True
)

# Print summary
print(results.summary())

# Detailed results
print("\n" + "=" * 70)
print("Detailed Results")
print("=" * 70)

# Bias
bias_ols = results.compute_bias('ols')
bias_rbfm = results.compute_bias('rbfm')
print(f"\nBias (×√T):")
print(f"  OLS-VAR: {np.abs(bias_ols).mean() * np.sqrt(T):.4f}")
print(f"  RBFM-VAR: {np.abs(bias_rbfm).mean() * np.sqrt(T):.4f}")

# RMSE
rmse_ols = results.compute_rmse('ols')
rmse_rbfm = results.compute_rmse('rbfm')
print(f"\nRMSE (×√T):")
print(f"  OLS-VAR: {rmse_ols.mean() * np.sqrt(T):.4f}")
print(f"  RBFM-VAR: {rmse_rbfm.mean() * np.sqrt(T):.4f}")

# Rejection rates
print(f"\nRejection Rates (Power):")
for alpha in [0.01, 0.05, 0.10]:
    rej_ols = results.compute_rejection_rate('ols', alpha)
    rej_rbfm = results.compute_rejection_rate('rbfm', alpha)
    print(f"  {int(alpha*100)}% level: OLS={rej_ols:.3f}, RBFM={rej_rbfm:.3f}")

# Export to LaTeX
latex = results.to_latex_table()
with open('simulation_table.tex', 'w') as f:
    f.write(latex)
print("\nLaTeX table saved to 'simulation_table.tex'")
```

---

## Example 5: Comparison Study

Detailed comparison of RBFM-VAR and OLS-VAR.

```python
import numpy as np
import matplotlib.pyplot as plt
from rbfmvar import RBFMVAR, generate_dgp

np.random.seed(42)

# Generate data
y, info = generate_dgp('case_c', T=300)

# Fit model
model = RBFMVAR(lag_order=1)
results = model.fit(y)

# Print comparison
print(results.compare_with_ols())

# Extract coefficients
coef_rbfm = results.coefficients.flatten()
coef_ols = results.ols_coefficients.flatten()
true_coef = info['true_coefficients'].flatten()

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Bar plot
x = np.arange(len(coef_rbfm))
width = 0.25

axes[0].bar(x - width, true_coef, width, label='True', color='green', alpha=0.7)
axes[0].bar(x, coef_ols, width, label='OLS-VAR', color='blue', alpha=0.7)
axes[0].bar(x + width, coef_rbfm, width, label='RBFM-VAR', color='red', alpha=0.7)
axes[0].set_xlabel('Coefficient Index')
axes[0].set_ylabel('Value')
axes[0].set_title('Coefficient Estimates')
axes[0].legend()
axes[0].set_xticks(x)

# Bias plot
bias_ols = coef_ols - true_coef
bias_rbfm = coef_rbfm - true_coef

axes[1].bar(x - width/2, np.abs(bias_ols), width, label='OLS-VAR', color='blue', alpha=0.7)
axes[1].bar(x + width/2, np.abs(bias_rbfm), width, label='RBFM-VAR', color='red', alpha=0.7)
axes[1].set_xlabel('Coefficient Index')
axes[1].set_ylabel('|Bias|')
axes[1].set_title('Absolute Bias')
axes[1].legend()
axes[1].set_xticks(x)

# Scatter plot: OLS vs RBFM
axes[2].scatter(coef_ols, coef_rbfm, s=100, alpha=0.7)
axes[2].plot([min(coef_ols), max(coef_ols)], 
             [min(coef_ols), max(coef_ols)], 
             'k--', label='45° line')
axes[2].set_xlabel('OLS-VAR Estimates')
axes[2].set_ylabel('RBFM-VAR Estimates')
axes[2].set_title('OLS vs RBFM-VAR')
axes[2].legend()

plt.tight_layout()
plt.savefig('comparison_plot.png', dpi=150)
plt.show()

# Summary statistics
print("\n" + "=" * 70)
print("Summary Statistics")
print("=" * 70)
print(f"Mean Absolute Bias (OLS):   {np.abs(bias_ols).mean():.4f}")
print(f"Mean Absolute Bias (RBFM):  {np.abs(bias_rbfm).mean():.4f}")
print(f"Reduction: {(1 - np.abs(bias_rbfm).mean()/np.abs(bias_ols).mean())*100:.1f}%")
```

---

## Example 6: Multiple Variables

Working with systems of 3+ variables.

```python
import numpy as np
from rbfmvar import RBFMVAR

np.random.seed(42)

# Generate 3-variable system
T = 200
n = 3

# Create I(1) and I(2) processes
y1 = np.cumsum(np.random.randn(T))  # I(1)
y2 = np.cumsum(np.cumsum(np.random.randn(T)))  # I(2)
y3 = np.cumsum(np.random.randn(T) + 0.5 * np.diff(np.r_[0, y2]))  # I(1) with influence from y2

y = np.column_stack([y1, y2, y3])
var_names = ['y₁', 'y₂', 'y₃']

print(f"Data shape: {y.shape}")
print(f"Variables: {var_names}")

# Fit model
model = RBFMVAR(lag_order=1)
results = model.fit(y)

print(results.summary())

# Test all pairwise Granger causality relationships
print("\n" + "=" * 70)
print("Pairwise Granger Causality Tests")
print("=" * 70)
print(f"\n{'Caused':<10} {'Causing':<10} {'Statistic':<12} {'P-value':<10} {'Result':<15}")
print("-" * 60)

for i in range(n):
    for j in range(n):
        if i != j:
            gc = results.granger_causality_test(i, [j], test_type='modified')
            result = "Reject H₀" if gc['p_value'] < 0.05 else "Cannot reject"
            print(f"{var_names[i]:<10} {var_names[j]:<10} {gc['statistic']:<12.4f} {gc['p_value']:<10.4f} {result:<15}")

# Test joint causality: y₂ and y₃ → y₁
print("\n" + "=" * 70)
print("Joint Causality Test")
print("=" * 70)

gc_joint = results.granger_causality_test(
    caused_variable=0,
    causing_variables=[1, 2],  # Both y₂ and y₃
    test_type='modified'
)

print(f"H₀: y₂ and y₃ do not jointly Granger-cause y₁")
print(f"Statistic: {gc_joint['statistic']:.4f}")
print(f"P-value: {gc_joint['p_value']:.4f}")
print(f"Degrees of freedom: {gc_joint['df']}")
```

---

## Example 7: Publication-Ready Output

Creating tables for academic papers.

```python
import numpy as np
from rbfmvar import RBFMVAR, generate_dgp

np.random.seed(42)

# Generate and fit
y, info = generate_dgp('case_c', T=200)
model = RBFMVAR(lag_order=1)
results = model.fit(y)

# LaTeX table
latex = results.to_latex(float_format="%.3f", include_std_errors=True)

print("=" * 70)
print("LaTeX Table Output")
print("=" * 70)
print(latex)

# Save to file
with open('results_table.tex', 'w') as f:
    f.write(latex)
print("\nTable saved to 'results_table.tex'")

# Create custom table for Granger causality
gc_table = """
\\begin{table}[htbp]
\\centering
\\caption{Granger Causality Tests}
\\label{tab:granger}
\\begin{tabular}{lccc}
\\toprule
Null Hypothesis & Statistic & P-value & Decision (5\\%) \\\\
\\midrule
"""

# Test both directions
gc_21 = results.granger_causality_test(0, [1], test_type='modified')
gc_12 = results.granger_causality_test(1, [0], test_type='modified')

gc_table += f"$y_2 \\\\not\\\\rightarrow y_1$ & {gc_21['statistic']:.3f} & {gc_21['p_value']:.4f} & {'Reject' if gc_21['p_value'] < 0.05 else 'Cannot reject'} \\\\\n"
gc_table += f"$y_1 \\\\not\\\\rightarrow y_2$ & {gc_12['statistic']:.3f} & {gc_12['p_value']:.4f} & {'Reject' if gc_12['p_value'] < 0.05 else 'Cannot reject'} \\\\\n"

gc_table += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Notes: Modified Wald test based on RBFM-VAR. The test is asymptotically conservative.
\\end{tablenotes}
\\end{table}
"""

print("\n" + "=" * 70)
print("Granger Causality LaTeX Table")
print("=" * 70)
print(gc_table)

with open('granger_table.tex', 'w') as f:
    f.write(gc_table)
print("\nTable saved to 'granger_table.tex'")
```

---

## Download Examples

All examples are available on GitHub:

- [example_basic.py](https://github.com/merwanroudane/rbfmvar/blob/main/examples/example_basic.py)
- [example_monte_carlo.py](https://github.com/merwanroudane/rbfmvar/blob/main/examples/example_monte_carlo.py)

---

[← Back to Home](.)
