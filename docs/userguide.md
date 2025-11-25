---
layout: default
title: User Guide
---

# User Guide

[← Back to Home](.)

This guide covers all features of the RBFM-VAR package in detail.

---

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Model Specification](#model-specification)
3. [Estimation](#estimation)
4. [Hypothesis Testing](#hypothesis-testing)
5. [Results and Output](#results-and-output)
6. [Forecasting](#forecasting)
7. [Comparison with OLS](#comparison-with-ols)

---

## Data Preparation

### Input Format

Your data must be a numpy array with shape `(T, n)`:

```python
import numpy as np

# T = 200 observations, n = 3 variables
y = np.random.randn(200, 3)
print(f"Shape: {y.shape}")  # (200, 3)
```

### From Pandas DataFrame

```python
import pandas as pd

df = pd.read_csv('macro_data.csv')

# Select columns
y = df[['GDP', 'Inflation', 'Interest_Rate']].values

# Or use .to_numpy()
y = df[['GDP', 'Inflation', 'Interest_Rate']].to_numpy()
```

### Handling Missing Values

Remove or interpolate missing values before estimation:

```python
# Check for NaN
print(f"Missing values: {np.isnan(y).sum()}")

# Remove rows with NaN
y = y[~np.isnan(y).any(axis=1)]

# Or interpolate (using pandas)
df = df.interpolate(method='linear')
y = df[['var1', 'var2']].values
```

### Data Transformations

The package expects **levels** (not differences). It computes differences internally:

```python
# DON'T do this - the package handles differencing
# y_diff = np.diff(y, axis=0)  # Wrong!

# DO this - pass levels directly
model = RBFMVAR(lag_order=1)
results = model.fit(y)  # Pass levels
```

---

## Model Specification

### The RBFMVAR Class

```python
from rbfmvar import RBFMVAR

model = RBFMVAR(
    lag_order=1,         # Number of lags (p)
    bandwidth='auto',    # Bandwidth for kernel estimation
    kernel='bartlett',   # Kernel function
    trend=None           # Trend specification (future)
)
```

### Parameters

#### `lag_order` (int)

Number of lags in the VAR model. Default: 1

```python
# VAR(1)
model = RBFMVAR(lag_order=1)

# VAR(2)
model = RBFMVAR(lag_order=2)

# VAR(4) for quarterly data
model = RBFMVAR(lag_order=4)
```

**How to choose:**
- Use information criteria (AIC, BIC)
- Economic theory
- Standard VAR lag selection procedures

#### `bandwidth` (int or 'auto')

Bandwidth (K) for kernel covariance estimation. Default: `'auto'`

```python
# Automatic selection (Andrews 1991)
model = RBFMVAR(bandwidth='auto')

# Fixed bandwidth
model = RBFMVAR(bandwidth=5)
model = RBFMVAR(bandwidth=10)
```

**Guidelines:**
- `'auto'` is recommended for most cases
- Larger K → more smoothing, less variance, more bias
- Smaller K → less smoothing, more variance, less bias
- Rule of thumb: K ≈ T^(1/3)

#### `kernel` (str)

Kernel function for covariance estimation. Default: `'bartlett'`

| Kernel | Description | Properties |
|--------|-------------|------------|
| `'bartlett'` | Newey-West kernel | Most common, linear decay |
| `'parzen'` | Parzen kernel | Smoother, cubic decay |
| `'qs'` | Quadratic Spectral | Optimal, no finite truncation |

```python
# Bartlett (Newey-West)
model = RBFMVAR(kernel='bartlett')

# Parzen
model = RBFMVAR(kernel='parzen')

# Quadratic Spectral
model = RBFMVAR(kernel='qs')
```

---

## Estimation

### Basic Estimation

```python
from rbfmvar import RBFMVAR

model = RBFMVAR(lag_order=1)
results = model.fit(y)
```

### Accessing Estimates

```python
# RBFM-VAR coefficients (bias-corrected)
F_plus = model.coefficients
print(f"RBFM-VAR shape: {F_plus.shape}")

# OLS-VAR coefficients (for comparison)
F_ols = model.ols_coefficients
print(f"OLS-VAR shape: {F_ols.shape}")

# Error covariance matrix
Sigma = model.sigma_epsilon
print(f"Σ_εε:\n{Sigma}")

# Residuals
residuals = model.residuals
print(f"Residuals shape: {residuals.shape}")
```

### ECM Coefficient Matrices

Extract Π₁ (coefficient for Δy_{t-1}) and Π₂ (coefficient for y_{t-1}):

```python
# Using RBFM-VAR estimates
Pi1_rbfm, Pi2_rbfm = model.get_Pi_matrices(use_rbfm=True)

# Using OLS estimates
Pi1_ols, Pi2_ols = model.get_Pi_matrices(use_rbfm=False)

print(f"Π₁ (RBFM-VAR):\n{Pi1_rbfm}")
print(f"Π₂ (RBFM-VAR):\n{Pi2_rbfm}")
```

---

## Hypothesis Testing

### Granger Causality Test

Test whether one variable Granger-causes another:

```python
# Test: Does y₂ (index 1) Granger-cause y₁ (index 0)?
gc = results.granger_causality_test(
    caused_variable=0,
    causing_variables=[1],
    test_type='modified'  # or 'standard'
)

print(f"Statistic: {gc['statistic']:.4f}")
print(f"P-value: {gc['p_value']:.4f}")
print(f"Degrees of freedom: {gc['df']}")
print(f"Conservative: {gc['is_conservative']}")
```

**Test Types:**
- `'modified'`: Modified Wald test (RBFM-VAR) - **recommended**
  - Conservative: actual size ≤ nominal size
  - Valid for I(0)/I(1)/I(2) mixtures
- `'standard'`: Standard Wald test (OLS-VAR)
  - May have size distortions with nonstationary data

### Multiple Causing Variables

```python
# Test: Do y₂ AND y₃ jointly Granger-cause y₁?
gc = results.granger_causality_test(
    caused_variable=0,
    causing_variables=[1, 2],  # Multiple variables
    test_type='modified'
)
```

### General Wald Test

For custom hypotheses:

```python
from rbfmvar import wald_test, modified_wald_test
import numpy as np

# Test H₀: F[0,0] = 0 and F[0,1] = 0
n, k = model.coefficients.shape
R = np.zeros((2, n * k))
R[0, 0] = 1  # First restriction: F[0,0] = 0
R[1, 1] = 1  # Second restriction: F[0,1] = 0
r = np.zeros(2)

result = results.wald_test(R, r, test_type='modified')
print(f"Statistic: {result['statistic']:.4f}")
print(f"P-value: {result['p_value']:.4f}")
```

---

## Results and Output

### Summary

```python
# Full summary
print(results.summary())

# With custom significance level
print(results.summary(alpha=0.01))
```

### Individual Components

```python
# Coefficients
print(results.coefficients)

# Standard errors
print(results.std_errors)

# t-statistics
print(results.t_statistics)

# P-values
print(results.p_values)

# R-squared
print(results.r_squared)

# Adjusted R-squared
print(results.adj_r_squared)

# Information criteria
print(f"AIC: {results.aic}")
print(f"BIC: {results.bic}")
print(f"HQC: {results.hqc}")

# Log-likelihood
print(f"Log-L: {results.log_likelihood}")
```

### ECM Matrices

```python
# Π₁: Coefficient for Δy_{t-1}
print(f"Π₁:\n{results.Pi1}")

# Π₂: Coefficient for y_{t-1}
print(f"Π₂:\n{results.Pi2}")
```

### Export to Dictionary

```python
d = results.to_dict()

# Access all results programmatically
print(d.keys())
# ['coefficients', 'ols_coefficients', 'std_errors', 
#  't_statistics', 'p_values', 'sigma_epsilon', ...]
```

### Export to LaTeX

```python
# Generate LaTeX table
latex = results.to_latex()
print(latex)

# With options
latex = results.to_latex(
    float_format="%.4f",
    include_std_errors=True
)

# Save to file
with open('table.tex', 'w') as f:
    f.write(latex)
```

---

## Forecasting

Generate out-of-sample forecasts:

```python
# Forecast 5 steps ahead
forecasts = results.forecast(steps=5)

print("Forecasts:")
print(f"{'Step':<6} {'y₁':<15} {'y₂':<15}")
print("-" * 40)
for i, fc in enumerate(forecasts):
    print(f"{i+1:<6} {fc[0]:<15.4f} {fc[1]:<15.4f}")
```

---

## Comparison with OLS

Compare RBFM-VAR and OLS-VAR estimates:

```python
print(results.compare_with_ols())
```

Output:
```
======================================================================
              Comparison: RBFM-VAR vs OLS-VAR Estimates               
======================================================================

Equation 1:
Coef      RBFM-VAR    OLS-VAR    Difference  % Diff
------  ----------  ---------  ------------  --------
[1,1]     -0.2891    -0.3005       0.0114    -3.79%
[1,2]     -0.1523    -0.1834       0.0311   -16.96%
...
```

---

## Advanced Usage

### Custom Covariance Estimation

```python
from rbfmvar import kernel_covariance, optimal_bandwidth

# Get residuals
residuals = model.residuals

# Compute optimal bandwidth
K = optimal_bandwidth(residuals, kernel='bartlett')
print(f"Optimal bandwidth: {K}")

# Compute long-run covariance
Omega = kernel_covariance(residuals, K=K, kernel='bartlett')
print(f"Long-run covariance:\n{Omega}")
```

### Accessing Internal Matrices

```python
# After fitting
model.fit(y)

# ECM matrices
Y = model._Y   # Dependent variable matrix
Z = model._Z   # Lagged Δ² terms
W = model._W   # (Δy_{t-1}, y_{t-1})
X = model._X   # Full regressor matrix

# v̂ process
v_hat = model._v_hat

# Covariance matrices
Omega_ev = model._Omega_ev
Omega_vv = model._Omega_vv
```

---

## Tips and Best Practices

### 1. Check Data Quality

```python
# Summary statistics
print(f"Mean: {y.mean(axis=0)}")
print(f"Std: {y.std(axis=0)}")
print(f"Min: {y.min(axis=0)}")
print(f"Max: {y.max(axis=0)}")
print(f"Missing: {np.isnan(y).sum()}")
```

### 2. Use Appropriate Sample Size

- Minimum: T > 50 for reliable inference
- Recommended: T > 100
- For Monte Carlo: T = 150 or T = 500 (as in paper)

### 3. Compare Results

Always compare RBFM-VAR with OLS-VAR to understand the bias correction:

```python
print(results.compare_with_ols())
```

### 4. Report Both Tests

For robustness, report both modified and standard Wald tests:

```python
gc_mod = results.granger_causality_test(0, [1], test_type='modified')
gc_std = results.granger_causality_test(0, [1], test_type='standard')

print(f"Modified Wald: {gc_mod['p_value']:.4f}")
print(f"Standard Wald: {gc_std['p_value']:.4f}")
```

---

[← Back to Home](.)
