---
layout: default
title: Simulation
---

# Monte Carlo Simulation

[← Back to Home](.)

Guide to running Monte Carlo simulations to evaluate RBFM-VAR performance.

---

## Overview

The simulation module allows you to:

1. Replicate the results from Chang (2000), Section 5
2. Evaluate finite-sample properties of estimators
3. Compare OLS-VAR and RBFM-VAR
4. Assess test size and power

---

## Data Generating Processes

### Available DGPs

The package includes three DGPs from Chang (2000):

```python
from rbfmvar import generate_dgp

# Case A: Both I(2), no cointegration, no causality
y_a, info_a = generate_dgp('case_a', T=150)

# Case B: y₁ I(1), y₂ I(2), no causality
y_b, info_b = generate_dgp('case_b', T=150)

# Case C: y₁ I(1), y₂ I(2), y₂ causes y₁
y_c, info_c = generate_dgp('case_c', T=150)
```

### DGP Specification

The DGP is:

```
Δy₁ₜ = ρ₁Δy₁,ₜ₋₁ + ρ₂(y₁,ₜ₋₁ - Δy₂,ₜ₋₁) + ε₁ₜ
Δ²y₂ₜ = ε₂ₜ
```

| Case | ρ₁ | ρ₂ | y₁ Order | y₂ Order | Granger Causality |
|------|-----|-----|----------|----------|-------------------|
| A | 1.0 | 0.0 | I(2) | I(2) | No |
| B | 0.5 | 0.0 | I(1) | I(2) | No |
| C | -0.3 | -0.15 | I(1) | I(2) | Yes (y₂ → y₁) |

### Error Covariance

Default:
```
Σ = [[1.0, 0.5],
     [0.5, 1.0]]
```

Custom:
```python
Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
y, info = generate_dgp('case_c', T=150, Sigma=Sigma)
```

---

## Running Simulations

### Basic Simulation

```python
from rbfmvar import monte_carlo_simulation

results = monte_carlo_simulation(
    dgp='case_c',
    T=150,
    n_reps=1000,
    seed=42
)

print(results.summary())
```

### Full Parameters

```python
results = monte_carlo_simulation(
    dgp='case_c',        # DGP case: 'case_a', 'case_b', 'case_c'
    T=150,               # Sample size
    n_reps=10000,        # Number of replications
    bandwidth='auto',    # Bandwidth: 'auto' or integer
    kernel='bartlett',   # Kernel: 'bartlett', 'parzen', 'qs'
    test_causality=True, # Compute Granger causality tests
    seed=42,             # Random seed for reproducibility
    verbose=True         # Print progress
)
```

---

## Analyzing Results

### Bias

```python
# Compute bias for each coefficient
bias_ols = results.compute_bias('ols')
bias_rbfm = results.compute_bias('rbfm')

print("Bias (OLS):")
print(bias_ols)
print("\nBias (RBFM-VAR):")
print(bias_rbfm)

# Scaled by √T (as in the paper)
import numpy as np
T = 150
print(f"\nBias × √T (OLS): {bias_ols.flatten() * np.sqrt(T)}")
print(f"Bias × √T (RBFM): {bias_rbfm.flatten() * np.sqrt(T)}")
```

### Standard Deviation

```python
std_ols = results.compute_std('ols')
std_rbfm = results.compute_std('rbfm')

print(f"Std × √T (OLS): {std_ols.flatten() * np.sqrt(T)}")
print(f"Std × √T (RBFM): {std_rbfm.flatten() * np.sqrt(T)}")
```

### Root Mean Squared Error

```python
rmse_ols = results.compute_rmse('ols')
rmse_rbfm = results.compute_rmse('rbfm')

print(f"RMSE × √T (OLS): {rmse_ols.flatten() * np.sqrt(T)}")
print(f"RMSE × √T (RBFM): {rmse_rbfm.flatten() * np.sqrt(T)}")
```

### Rejection Rates

```python
# Test size/power at different significance levels
for alpha in [0.01, 0.05, 0.10]:
    rej_ols = results.compute_rejection_rate('ols', alpha)
    rej_rbfm = results.compute_rejection_rate('rbfm', alpha)
    
    print(f"{int(alpha*100)}% level: OLS={rej_ols:.3f}, RBFM={rej_rbfm:.3f}")
```

---

## Replicating Tables from the Paper

### Table 1: Bias and Standard Deviation

```python
import numpy as np
from rbfmvar import monte_carlo_simulation

# Paper parameters
T = 150
n_reps = 10000

# Run all cases
all_results = {}
for case in ['case_a', 'case_b', 'case_c']:
    print(f"Running {case}...")
    all_results[case] = monte_carlo_simulation(
        dgp=case, T=T, n_reps=n_reps, seed=42, verbose=False
    )

# Print Table 1 format
print("\n" + "=" * 80)
print("Table 1: Bias and Standard Deviation (× √T)")
print("=" * 80)

for case in ['case_a', 'case_b', 'case_c']:
    res = all_results[case]
    bias_ols = res.compute_bias('ols').flatten() * np.sqrt(T)
    bias_rbfm = res.compute_bias('rbfm').flatten() * np.sqrt(T)
    std_ols = res.compute_std('ols').flatten() * np.sqrt(T)
    std_rbfm = res.compute_std('rbfm').flatten() * np.sqrt(T)
    
    print(f"\n{case.upper()}:")
    print(f"{'Coef':<8} {'Bias(OLS)':<12} {'Bias(RBFM)':<12} {'Std(OLS)':<12} {'Std(RBFM)':<12}")
    print("-" * 60)
    
    coef_names = ['π₁₁₁', 'π₁₁₂', 'π₁₂₁', 'π₁₂₂', 'π₂₁₁', 'π₂₁₂', 'π₂₂₁', 'π₂₂₂']
    for i, name in enumerate(coef_names):
        print(f"{name:<8} {bias_ols[i]:<12.4f} {bias_rbfm[i]:<12.4f} {std_ols[i]:<12.4f} {std_rbfm[i]:<12.4f}")
```

### Table 2: Test Size and Power

```python
print("\n" + "=" * 80)
print("Table 2: Rejection Probabilities")
print("=" * 80)
print(f"\n{'Case':<10} {'Test':<15} {'1%':<10} {'5%':<10} {'10%':<10}")
print("-" * 55)

for case in ['case_a', 'case_b', 'case_c']:
    res = all_results[case]
    
    # OLS Wald
    rej_ols = [res.compute_rejection_rate('ols', a) for a in [0.01, 0.05, 0.10]]
    print(f"{case.upper():<10} {'W_F (OLS)':<15} {rej_ols[0]:<10.3f} {rej_ols[1]:<10.3f} {rej_ols[2]:<10.3f}")
    
    # Modified Wald
    rej_rbfm = [res.compute_rejection_rate('rbfm', a) for a in [0.01, 0.05, 0.10]]
    print(f"{'':<10} {'W_F+ (RBFM)':<15} {rej_rbfm[0]:<10.3f} {rej_rbfm[1]:<10.3f} {rej_rbfm[2]:<10.3f}")

print("\nNotes:")
print("- Cases A, B: SIZE (H₀ true). Ideal rejection = nominal level.")
print("- Case C: POWER (H₀ false). Higher = better.")
```

---

## Export Results

### Summary Table

```python
print(results.summary())
```

### LaTeX Export

```python
# Generate LaTeX table
latex = results.to_latex_table()

# Save to file
with open('simulation_results.tex', 'w') as f:
    f.write(latex)

print("LaTeX table saved!")
```

### Custom LaTeX Table

```python
def create_custom_latex(all_results, T=150):
    """Create custom LaTeX table for simulation results."""
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Monte Carlo Simulation Results (T=%d)}
\label{tab:mc}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{3}{c}{Bias $\times \sqrt{T}$} & \multicolumn{3}{c}{Std $\times \sqrt{T}$} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
Coefficient & Case A & Case B & Case C & Case A & Case B & Case C \\
\midrule
""" % T
    
    coef_names = [r'$\pi_{11}^{(1)}$', r'$\pi_{12}^{(1)}$', 
                  r'$\pi_{21}^{(1)}$', r'$\pi_{22}^{(1)}$',
                  r'$\pi_{11}^{(2)}$', r'$\pi_{12}^{(2)}$',
                  r'$\pi_{21}^{(2)}$', r'$\pi_{22}^{(2)}$']
    
    for i, name in enumerate(coef_names):
        bias_vals = []
        std_vals = []
        for case in ['case_a', 'case_b', 'case_c']:
            res = all_results[case]
            bias = res.compute_bias('rbfm').flatten()[i] * np.sqrt(T)
            std = res.compute_std('rbfm').flatten()[i] * np.sqrt(T)
            bias_vals.append(f"{bias:.3f}")
            std_vals.append(f"{std:.3f}")
        
        latex += f"{name} & {' & '.join(bias_vals)} & {' & '.join(std_vals)} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Notes: Results based on 10,000 replications. RBFM-VAR estimates shown.
\end{tablenotes}
\end{table}
"""
    return latex

# Generate and save
latex = create_custom_latex(all_results, T=150)
with open('custom_table.tex', 'w') as f:
    f.write(latex)
```

---

## Sensitivity Analysis

### Sample Size Comparison

```python
sample_sizes = [100, 150, 200, 500]

print("Effect of Sample Size on Bias")
print("=" * 60)
print(f"{'T':<10} {'Mean |Bias| (OLS)':<20} {'Mean |Bias| (RBFM)':<20}")
print("-" * 50)

for T in sample_sizes:
    res = monte_carlo_simulation('case_c', T=T, n_reps=500, seed=42, verbose=False)
    
    bias_ols = np.abs(res.compute_bias('ols')).mean()
    bias_rbfm = np.abs(res.compute_bias('rbfm')).mean()
    
    print(f"{T:<10} {bias_ols:<20.4f} {bias_rbfm:<20.4f}")
```

### Kernel Comparison

```python
kernels = ['bartlett', 'parzen', 'qs']

print("\nEffect of Kernel Choice")
print("=" * 60)
print(f"{'Kernel':<15} {'Size (5%)':<15} {'RMSE':<15}")
print("-" * 45)

for kernel in kernels:
    res = monte_carlo_simulation(
        'case_a', T=150, n_reps=500, 
        kernel=kernel, seed=42, verbose=False
    )
    
    size = res.compute_rejection_rate('rbfm', 0.05)
    rmse = res.compute_rmse('rbfm').mean()
    
    print(f"{kernel:<15} {size:<15.3f} {rmse:<15.4f}")
```

---

## Interpretation Guide

### Size Results (Cases A, B)

- **Good size control**: Rejection rate ≈ nominal level (e.g., ≈5% at 5%)
- **Size distortion**: Rejection rate > nominal level
- **OLS problem**: Standard Wald test often shows 20-40% rejection at 5%
- **RBFM advantage**: Modified Wald maintains size ≤ nominal level

### Power Results (Case C)

- **Good power**: High rejection rate when H₀ is false
- **Trade-off**: Conservative tests may have lower power
- **Both tests**: Generally have good power in Case C

### Bias and Variance

- **Bias**: Should decrease with T (consistency)
- **Variance**: Should decrease with T
- **RMSE = √(Bias² + Variance²)**: Overall measure of accuracy

---

## Computational Notes

### Time Estimates

| n_reps | Approximate Time |
|--------|------------------|
| 100 | ~10 seconds |
| 1,000 | ~2 minutes |
| 10,000 | ~20 minutes |

### Memory Usage

Each replication stores:
- Coefficient estimates (n × k matrix)
- Test statistics (2 values)

For 10,000 replications with n=2, k=4: ~1 MB

### Parallelization

For large-scale simulations, consider parallelization:

```python
# Note: This is a conceptual example
from joblib import Parallel, delayed

def run_single_rep(seed):
    y, _ = generate_dgp('case_c', T=150, seed=seed)
    model = RBFMVAR(lag_order=1)
    results = model.fit(y)
    return results.coefficients

# Run in parallel
coefs = Parallel(n_jobs=-1)(
    delayed(run_single_rep)(i) for i in range(1000)
)
```

---

[← Back to Home](.)
