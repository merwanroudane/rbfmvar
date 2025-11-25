---
layout: default
title: Quick Start
---

# Quick Start Guide

[← Back to Home](.)

Get started with RBFM-VAR in 5 minutes!

---

## Step 1: Import the Package

```python
import numpy as np
from rbfmvar import RBFMVAR, generate_dgp
```

---

## Step 2: Prepare Your Data

### Option A: Use Your Own Data

Your data should be a 2D numpy array with shape `(T, n)` where:
- `T` = number of time periods (rows)
- `n` = number of variables (columns)

```python
import pandas as pd

# Load from CSV
df = pd.read_csv('your_data.csv')
y = df[['var1', 'var2']].values

print(f"Data shape: {y.shape}")  # Should be (T, n)
```

### Option B: Generate Simulated Data

Use the built-in DGP generator:

```python
# Generate data from Case C: y₂ Granger-causes y₁
y, info = generate_dgp('case_c', T=200, seed=42)

print(f"Data shape: {y.shape}")
print(f"DGP: {info['case']}")
```

Available DGPs:
- `'case_a'`: Both I(2), no cointegration, no causality
- `'case_b'`: y₁ I(1), y₂ I(2), no causality
- `'case_c'`: y₁ I(1), y₂ I(2), y₂ causes y₁

---

## Step 3: Fit the Model

```python
# Create RBFM-VAR model
model = RBFMVAR(
    lag_order=1,        # Number of lags (p)
    bandwidth='auto',   # Automatic bandwidth selection
    kernel='bartlett'   # Kernel function
)

# Fit the model
results = model.fit(y)
```

---

## Step 4: View Results

### Summary Table

```python
print(results.summary())
```

Output:
```
================================================================================
                          RBFM-VAR Estimation Results                           
================================================================================

Model Information:
----------------------------------------
  Number of equations:       2
  Lag order (p):            1
  Observations used:        198
  Bandwidth (K):            4

RBFM-VAR Coefficient Estimates:
--------------------------------------------------------------------------------

Equation 1 (y1):
Variable      Coefficient    Std. Error    t-statistic    P-value  Signif.
----------  -------------  ------------  -------------  ---------  ---------
Δy1(t-1)         -0.2891        0.0641        -4.5102     0.0000  ***
Δy2(t-1)         -0.1523        0.0365        -4.1726     0.0000  ***
y1(t-1)           0.9876        0.0400        24.6900     0.0000  ***
y2(t-1)           0.0001        0.0001         1.0000     0.3173
...
```

### Coefficients Only

```python
print("RBFM-VAR coefficients:")
print(results.coefficients)

print("\nOLS-VAR coefficients (for comparison):")
print(results.ols_coefficients)
```

---

## Step 5: Test Granger Causality

Test whether variable 1 Granger-causes variable 0:

```python
gc = results.granger_causality_test(
    caused_variable=0,      # y₁ (index 0)
    causing_variables=[1],  # y₂ (index 1)
    test_type='modified'    # Use modified Wald test
)

print(f"H₀: y₂ does not Granger-cause y₁")
print(f"Test Statistic: {gc['statistic']:.4f}")
print(f"P-value: {gc['p_value']:.4f}")
print(f"Degrees of Freedom: {gc['df']}")

if gc['p_value'] < 0.05:
    print("→ Reject H₀: y₂ Granger-causes y₁")
else:
    print("→ Cannot reject H₀")
```

---

## Step 6: Export Results

### To LaTeX

```python
latex_table = results.to_latex()
print(latex_table)

# Save to file
with open('results_table.tex', 'w') as f:
    f.write(latex_table)
```

### To Dictionary

```python
d = results.to_dict()
print(d.keys())
# dict_keys(['coefficients', 'ols_coefficients', 'std_errors', ...])
```

---

## Complete Example

```python
import numpy as np
from rbfmvar import RBFMVAR, generate_dgp

# Set seed for reproducibility
np.random.seed(42)

# Generate data
y, info = generate_dgp('case_c', T=200)
print(f"Generated: {info['case']}")

# Fit model
model = RBFMVAR(lag_order=1)
results = model.fit(y)

# Print summary
print(results.summary())

# Test Granger causality
gc = results.granger_causality_test(0, [1])
print(f"\nGranger Causality Test:")
print(f"  Statistic: {gc['statistic']:.4f}")
print(f"  P-value: {gc['p_value']:.4f}")

# Compare with OLS
print("\n" + results.compare_with_ols())
```

---

## Next Steps

Now that you've run your first RBFM-VAR model:

1. **[User Guide](userguide)** - Learn about all options and features
2. **[API Reference](api)** - Complete function documentation
3. **[Examples](examples)** - More detailed examples
4. **[Theory](theory)** - Understand the methodology

---

## Common Questions

### How do I choose the lag order?

Use information criteria (AIC, BIC) or standard VAR lag selection methods. The package reports these:

```python
print(f"AIC: {results.aic}")
print(f"BIC: {results.bic}")
```

### What bandwidth should I use?

The default `'auto'` uses Andrews (1991) optimal bandwidth selection. This is recommended for most cases.

### Which kernel is best?

- `'bartlett'` (default): Newey-West, most common
- `'parzen'`: Smoother, slightly more efficient
- `'qs'`: Quadratic spectral, optimal but no finite truncation

---

[← Back to Home](.)
