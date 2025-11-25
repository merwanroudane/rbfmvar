---
layout: default
title: API Reference
---

# API Reference

[← Back to Home](.)

Complete API documentation for the RBFM-VAR package.

---

## Table of Contents

- [Core Classes](#core-classes)
  - [RBFMVAR](#rbfmvar)
  - [OLSVAREstimator](#olsvarestimator)
  - [RBFMVARResults](#rbfmvarresults)
- [Testing Functions](#testing-functions)
- [Covariance Functions](#covariance-functions)
- [Simulation Functions](#simulation-functions)
- [Utility Functions](#utility-functions)

---

## Core Classes

### RBFMVAR

Main class for RBFM-VAR estimation.

```python
from rbfmvar import RBFMVAR
```

#### Constructor

```python
RBFMVAR(lag_order=1, bandwidth='auto', kernel='bartlett', trend=None)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lag_order` | int | 1 | Number of lags (p) in the VAR model |
| `bandwidth` | int or 'auto' | 'auto' | Bandwidth for kernel covariance estimation |
| `kernel` | str | 'bartlett' | Kernel function: 'bartlett', 'parzen', or 'qs' |
| `trend` | str or None | None | Trend specification (reserved for future use) |

#### Methods

##### `fit(y)`

Fit the RBFM-VAR model.

```python
results = model.fit(y)
```

**Parameters:**
- `y` (np.ndarray): Time series data of shape (T, n)

**Returns:**
- `RBFMVARResults`: Results object

##### `get_phi_and_A(use_rbfm=True)`

Get Φ and A coefficient matrices separately.

```python
Phi, A = model.get_phi_and_A(use_rbfm=True)
```

**Parameters:**
- `use_rbfm` (bool): If True, return RBFM-VAR estimates; else OLS

**Returns:**
- Tuple of (Φ, A) matrices

##### `get_Pi_matrices(use_rbfm=True)`

Extract Π₁ and Π₂ from ECM representation.

```python
Pi1, Pi2 = model.get_Pi_matrices(use_rbfm=True)
```

**Parameters:**
- `use_rbfm` (bool): If True, return RBFM-VAR estimates; else OLS

**Returns:**
- Tuple of (Π₁, Π₂) matrices

##### `predict(steps=1)`

Generate forecasts.

```python
forecasts = model.predict(steps=5)
```

**Parameters:**
- `steps` (int): Number of steps ahead to forecast

**Returns:**
- np.ndarray: Forecast values of shape (steps, n)

#### Attributes (after fitting)

| Attribute | Type | Description |
|-----------|------|-------------|
| `coefficients` | np.ndarray | RBFM-VAR coefficient estimates F̂⁺ |
| `ols_coefficients` | np.ndarray | OLS-VAR coefficient estimates F̂ |
| `residuals` | np.ndarray | Model residuals |
| `sigma_epsilon` | np.ndarray | Error covariance matrix Σ̂_εε |

---

### OLSVAREstimator

Standard OLS-VAR estimator.

```python
from rbfmvar import OLSVAREstimator
```

#### Constructor

```python
OLSVAREstimator(lag_order=1)
```

#### Methods

##### `fit(y)`

Fit OLS-VAR model.

```python
estimator = OLSVAREstimator(lag_order=1)
estimator.fit(y)
```

##### `predict(y, steps=1)`

Generate forecasts.

---

### RBFMVARResults

Results container returned by `RBFMVAR.fit()`.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `coefficients` | np.ndarray | RBFM-VAR estimates |
| `ols_coefficients` | np.ndarray | OLS-VAR estimates |
| `std_errors` | np.ndarray | Standard errors |
| `t_statistics` | np.ndarray | t-statistics |
| `p_values` | np.ndarray | P-values |
| `sigma_epsilon` | np.ndarray | Error covariance |
| `residuals` | np.ndarray | Residuals |
| `Pi1` | np.ndarray | Π₁ matrix |
| `Pi2` | np.ndarray | Π₂ matrix |
| `r_squared` | np.ndarray | R² for each equation |
| `adj_r_squared` | np.ndarray | Adjusted R² |
| `log_likelihood` | float | Log-likelihood |
| `aic` | float | Akaike Information Criterion |
| `bic` | float | Bayesian Information Criterion |
| `hqc` | float | Hannan-Quinn Criterion |

#### Methods

##### `summary(alpha=0.05)`

Generate formatted summary.

```python
print(results.summary())
print(results.summary(alpha=0.01))  # 1% significance
```

##### `granger_causality_test(caused_variable, causing_variables, test_type='modified')`

Test Granger causality.

```python
gc = results.granger_causality_test(
    caused_variable=0,
    causing_variables=[1],
    test_type='modified'
)
```

**Parameters:**
- `caused_variable` (int): Index of caused variable
- `causing_variables` (list): Indices of causing variables
- `test_type` (str): 'modified' or 'standard'

**Returns:**
- dict with keys: 'statistic', 'p_value', 'df', 'is_conservative'

##### `wald_test(R, r, test_type='modified')`

General Wald test.

```python
result = results.wald_test(R, r, test_type='modified')
```

##### `to_latex(float_format='%.4f', include_std_errors=True)`

Export to LaTeX.

```python
latex = results.to_latex()
```

##### `to_dict()`

Export to dictionary.

```python
d = results.to_dict()
```

##### `compare_with_ols()`

Compare RBFM-VAR and OLS-VAR.

```python
print(results.compare_with_ols())
```

##### `forecast(steps=1)`

Generate forecasts.

```python
fc = results.forecast(steps=5)
```

---

## Testing Functions

### wald_test

```python
from rbfmvar import wald_test

result = wald_test(F, R, r, Sigma_F)
```

Standard Wald test.

**Parameters:**
- `F` (np.ndarray): Coefficient matrix
- `R` (np.ndarray): Restriction matrix
- `r` (np.ndarray): Restriction values
- `Sigma_F` (np.ndarray): Variance of vec(F)

**Returns:**
- dict with 'statistic', 'p_value', 'df'

### modified_wald_test

```python
from rbfmvar import modified_wald_test

result = modified_wald_test(F_plus, R, r, Sigma_epsilon, XtX, T)
```

Modified Wald test (conservative).

**Parameters:**
- `F_plus` (np.ndarray): RBFM-VAR coefficients
- `R` (np.ndarray): Restriction matrix
- `r` (np.ndarray): Restriction values
- `Sigma_epsilon` (np.ndarray): Error covariance
- `XtX` (np.ndarray): X'X matrix
- `T` (int): Sample size

**Returns:**
- dict with 'statistic', 'p_value', 'df', 'is_conservative'

### granger_causality_test

```python
from rbfmvar import granger_causality_test

result = granger_causality_test(results, caused_var, causing_vars, test_type)
```

---

## Covariance Functions

### kernel_covariance

```python
from rbfmvar import kernel_covariance

Omega = kernel_covariance(u, K=10, kernel='bartlett')
```

Estimate long-run covariance matrix.

**Parameters:**
- `u` (np.ndarray): Residuals (T, n)
- `K` (int): Bandwidth
- `kernel` (str): 'bartlett', 'parzen', or 'qs'

**Returns:**
- np.ndarray: Long-run covariance Ω̂

### one_sided_kernel_covariance

```python
from rbfmvar import one_sided_kernel_covariance

Delta = one_sided_kernel_covariance(u, K=10, kernel='bartlett')
```

One-sided long-run covariance.

### optimal_bandwidth

```python
from rbfmvar import optimal_bandwidth

K = optimal_bandwidth(u, kernel='bartlett', method='andrews')
```

Select optimal bandwidth (Andrews 1991).

**Parameters:**
- `u` (np.ndarray): Residuals
- `kernel` (str): Kernel function
- `method` (str): 'andrews' or 'newey_west'

**Returns:**
- int: Optimal bandwidth

### Kernel Functions

```python
from rbfmvar import bartlett_kernel, parzen_kernel, quadratic_spectral_kernel

w = bartlett_kernel(0.5)  # = 0.5
w = parzen_kernel(0.5)    # ≈ 0.5625
w = quadratic_spectral_kernel(0.5)  # ≈ 0.82
```

---

## Simulation Functions

### generate_dgp

```python
from rbfmvar import generate_dgp

y, info = generate_dgp(case, T=150, Sigma=None, seed=None)
```

Generate data from DGPs in Chang (2000).

**Parameters:**
- `case` (str): 'case_a', 'case_b', or 'case_c'
- `T` (int): Sample size
- `Sigma` (np.ndarray): Error covariance (default: [[1, 0.5], [0.5, 1]])
- `seed` (int): Random seed

**Returns:**
- Tuple of (y, info) where info contains DGP details

### monte_carlo_simulation

```python
from rbfmvar import monte_carlo_simulation

results = monte_carlo_simulation(
    dgp='case_c',
    T=150,
    n_reps=1000,
    bandwidth='auto',
    kernel='bartlett',
    test_causality=True,
    seed=42,
    verbose=True
)
```

Run Monte Carlo simulation.

**Parameters:**
- `dgp` (str): DGP case
- `T` (int): Sample size
- `n_reps` (int): Number of replications
- `bandwidth`: Bandwidth specification
- `kernel` (str): Kernel function
- `test_causality` (bool): Compute Granger tests
- `seed` (int): Random seed
- `verbose` (bool): Print progress

**Returns:**
- `SimulationResults` object

### SimulationResults

```python
# Methods
results.compute_bias('ols')      # or 'rbfm'
results.compute_std('ols')
results.compute_rmse('ols')
results.compute_rejection_rate('ols', alpha=0.05)
results.summary()
results.to_latex_table()
```

---

## Utility Functions

### vec

```python
from rbfmvar import vec

v = vec(A)  # Vectorize matrix (column-major)
```

### unvec

```python
from rbfmvar import unvec

A = unvec(v, m, n)  # Reshape to m×n matrix
```

### difference

```python
from rbfmvar import difference

dy = difference(y, d=1)   # First difference
d2y = difference(y, d=2)  # Second difference
```

### lag_matrix

```python
from rbfmvar import lag_matrix

X = lag_matrix(y, lags=2)  # Create lagged values
```

### kronecker_product

```python
from rbfmvar import kronecker_product

C = kronecker_product(A, B)  # A ⊗ B
```

---

## Complete Import List

```python
from rbfmvar import (
    # Core classes
    RBFMVAR,
    OLSVAREstimator,
    RBFMVARResults,
    
    # Testing
    wald_test,
    modified_wald_test,
    granger_causality_test,
    
    # Covariance
    kernel_covariance,
    one_sided_kernel_covariance,
    cross_kernel_covariance,
    optimal_bandwidth,
    bartlett_kernel,
    parzen_kernel,
    quadratic_spectral_kernel,
    
    # Simulation
    generate_dgp,
    monte_carlo_simulation,
    SimulationResults,
    
    # Utilities
    vec,
    unvec,
    difference,
    lag_matrix,
    kronecker_product
)
```

---

[← Back to Home](.)
