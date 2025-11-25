# RBFM-VAR: Residual-Based Fully Modified Vector Autoregression

[![Python Version](https://img.shields.io/pypi/pyversions/rbfmvar.svg)](https://pypi.org/project/rbfmvar/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/merwanroudane/rbfmvar)

A Python implementation of the Residual-Based Fully Modified Vector Autoregression (RBFM-VAR) methodology for nonstationary vector autoregressions with unknown mixtures of I(0), I(1), and I(2) components.

## Overview

This package implements the methodology developed by **Yoosoon Chang (2000)**: *"Vector Autoregressions with Unknown Mixtures of I(0), I(1), and I(2) Components"*, published in *Econometric Theory*, Vol. 16, No. 6, pp. 905-926.

### Key Features

- **No Prior Knowledge Required**: Estimates VAR models without requiring prior knowledge about the exact number and location of unit roots in the system
- **Mixed Integration Orders**: Handles any mixture of I(0), I(1), and I(2) variables that may be cointegrated in any form
- **Optimal Estimation**: Yields an estimator that is optimal in the sense of Phillips (1991)
- **Mixed Normal Limit Theory**: The nonstationary component has mixed normal limit distribution without unit root distributions
- **Conservative Wald Tests**: Provides asymptotically valid tests using conventional chi-square critical values
- **Granger Causality Testing**: Direct application for causality testing in nonstationary VARs

## Installation

```bash
pip install rbfmvar
```

Or install from source:

```bash
git clone https://github.com/merwanroudane/rbfmvar.git
cd rbfmvar
pip install -e .
```

## Quick Start

```python
import numpy as np
from rbfmvar import RBFMVAR

# Generate or load your data (T x n matrix)
# Example: bivariate system with 200 observations
T, n = 200, 2
np.random.seed(42)

# Simulate data (replace with your actual data)
data = np.cumsum(np.cumsum(np.random.randn(T, n), axis=0), axis=0)

# Fit RBFM-VAR model
model = RBFMVAR(lag_order=2, bandwidth='auto')
results = model.fit(data)

# Print summary
print(results.summary())

# Test for Granger causality
causality_test = results.granger_causality_test(
    caused_variable=0,
    causing_variables=[1]
)
print(causality_test)
```

## Mathematical Background

### The Model

The RBFM-VAR procedure estimates a p-th order VAR:

$$y_t = A_1 y_{t-1} + \cdots + A_p y_{t-p} + \varepsilon_t$$

which can be written in Error Correction Model (ECM) format:

$$\Delta^2 y_t = \Phi(L)\Delta^2 y_{t-1} + \Pi_1 \Delta y_{t-1} + \Pi_2 y_{t-1} + \varepsilon_t$$

### RBFM-VAR Estimator

The estimator is defined as:

$$\hat{F}^+ = (Y'Z, Y^{+'} W + T\hat{A}^+)(X'X)^{-1}$$

where:

$$Y^{+'} = Y' - \hat{\Omega}_{\varepsilon\hat{v}} \hat{\Omega}_{\hat{v}\hat{v}}^{-1} \hat{V}'$$

$$\hat{A}^+ = \hat{\Omega}_{\varepsilon\hat{v}} \hat{\Omega}_{\hat{v}\hat{v}}^{-1} \hat{\Delta}_{\hat{v}\Delta w}$$

### Key Results (Theorem 1)

**(a)** For the stationary component:
$$\sqrt{T}(\hat{F}^+ - F)G^1 \xrightarrow{d} N(0, \Sigma_{\varepsilon\varepsilon} \otimes \Sigma_{x11}^{-1})$$

**(b)** For the nonstationary component:
$$(\hat{F}^+ - F)G^b D_T \xrightarrow{d} \text{MN}(0, \Omega_{\varepsilon\varepsilon \cdot 2} \otimes (\int_0^1 \bar{B}_b \bar{B}_b')^{-1})$$

### Modified Wald Test (Theorem 2)

The modified Wald statistic has a limit distribution that is a mixture of chi-square variates:

$$W_F^+ \xrightarrow{d} \chi^2_{q_1(q_\Phi + q_{A1})} + \sum_{i=1}^{q_1} d_i \chi^2_{q_{Ab}}(i)$$

This is bounded above by a $\chi^2_q$ distribution, allowing the use of conventional critical values.

## API Reference

### Main Classes

#### `RBFMVAR`

The main class for RBFM-VAR estimation.

```python
RBFMVAR(
    lag_order: int = 1,
    bandwidth: Union[int, str] = 'auto',
    kernel: str = 'bartlett',
    trend: str = 'c'
)
```

**Parameters:**
- `lag_order`: Number of lags in the VAR model (p)
- `bandwidth`: Bandwidth for kernel estimation ('auto' or integer)
- `kernel`: Kernel function ('bartlett', 'parzen', 'qs')
- `trend`: Trend specification ('n' = none, 'c' = constant, 'ct' = constant + trend)

**Methods:**
- `fit(data)`: Estimate the RBFM-VAR model
- `predict(steps)`: Generate forecasts

#### `RBFMVARResults`

Results class containing estimation output.

**Attributes:**
- `coefficients`: Estimated coefficient matrices
- `residuals`: Model residuals
- `sigma_epsilon`: Estimated error covariance matrix
- `wald_statistic`: Modified Wald test statistic

**Methods:**
- `summary()`: Print formatted summary
- `granger_causality_test()`: Test for Granger causality
- `wald_test(R, r)`: General linear hypothesis test

### Utility Functions

```python
from rbfmvar import (
    kernel_covariance,      # Long-run covariance estimation
    bartlett_kernel,        # Bartlett kernel function
    parzen_kernel,          # Parzen kernel function
    qs_kernel,              # Quadratic Spectral kernel
    optimal_bandwidth       # Automatic bandwidth selection
)
```

## Examples

### Example 1: Basic Estimation

```python
import numpy as np
import pandas as pd
from rbfmvar import RBFMVAR

# Load data
data = pd.read_csv('your_data.csv').values

# Estimate model
model = RBFMVAR(lag_order=2)
results = model.fit(data)

# Get coefficient estimates
print("OLS-VAR Coefficients:")
print(results.ols_coefficients)

print("\nRBFM-VAR Coefficients:")
print(results.coefficients)
```

### Example 2: Granger Causality Testing

```python
from rbfmvar import RBFMVAR

# Estimate model
model = RBFMVAR(lag_order=2)
results = model.fit(data)

# Test if variable 1 Granger-causes variable 0
causality = results.granger_causality_test(
    caused_variable=0,
    causing_variables=[1]
)

print(f"Modified Wald Statistic: {causality['statistic']:.4f}")
print(f"P-value (conservative): {causality['p_value']:.4f}")
print(f"Degrees of Freedom: {causality['df']}")
```

### Example 3: Monte Carlo Simulation

```python
from rbfmvar import monte_carlo_simulation

# Run simulation study
results = monte_carlo_simulation(
    dgp='case_a',      # Data generating process
    T=150,             # Sample size
    n_reps=1000,       # Number of replications
    seed=42
)

# Display results
print(results.summary())
```

## Data Generating Processes

The package includes three DGPs from the paper:

- **Case A** (ρ₁=1, ρ₂=0): Both y₁ and y₂ are I(2) with no cointegration
- **Case B** (ρ₁=0.5, ρ₂=0): y₁ is I(1), y₂ is I(2), no Granger causality
- **Case C** (ρ₁=-0.3, ρ₂=-0.15): y₁ is I(1), y₂ is I(2), y₂ Granger-causes y₁

## Citation

If you use this package in your research, please cite:

```bibtex
@article{chang2000var,
  title={Vector Autoregressions with Unknown Mixtures of {I}(0), {I}(1), and {I}(2) Components},
  author={Chang, Yoosoon},
  journal={Econometric Theory},
  volume={16},
  number={6},
  pages={905--926},
  year={2000},
  publisher={Cambridge University Press}
}

@software{roudane2024rbfmvar,
  title={RBFM-VAR: Python Implementation of Residual-Based Fully Modified VAR},
  author={Roudane, Merwan},
  year={2024},
  url={https://github.com/merwanroudane/rbfmvar}
}
```

## References

- Chang, Y. (2000). Vector Autoregressions with Unknown Mixtures of I(0), I(1), and I(2) Components. *Econometric Theory*, 16(6), 905-926.
- Chang, Y. & Phillips, P.C.B. (1995). Time series regression with mixtures of integrated processes. *Econometric Theory*, 11, 1033-1094.
- Phillips, P.C.B. (1991). Optimal inference in cointegrated systems. *Econometrica*, 59, 283-306.
- Phillips, P.C.B. (1995). Fully modified least squares and vector autoregression. *Econometrica*, 63, 1023-1078.
- Johansen, S. (1995). A statistical analysis of cointegration for I(2) variables. *Econometric Theory*, 11, 25-59.
- Toda, H. & Phillips, P.C.B. (1993). Vector autoregressions and causality. *Econometrica*, 61, 1367-1393.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Dr. Merwan Roudane**
- Email: merwanroudane920@gmail.com
- GitHub: https://github.com/merwanroudane

## Changelog

### Version 2.0.0 (2025)
- Complete implementation of RBFM-VAR methodology
- Long-run covariance estimation with multiple kernel options
- Modified Wald tests for hypothesis testing
- Granger causality testing
- Monte Carlo simulation tools
- Publication-ready output formatting
