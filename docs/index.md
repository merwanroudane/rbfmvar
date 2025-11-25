---
layout: default
title: RBFM-VAR
---

# RBFM-VAR

## Residual-Based Fully Modified Vector Autoregression

[![PyPI version](https://badge.fury.io/py/rbfmvar.svg)](https://badge.fury.io/py/rbfmvar)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package implementing the RBFM-VAR methodology for estimating Vector Autoregressive models with **unknown mixtures of I(0), I(1), and I(2) components**.

---

## Overview

The RBFM-VAR package provides:

- **Bias-corrected estimation** for VAR models with nonstationary variables
- **Modified Wald tests** with conservative inference (bounded size)
- **Granger causality testing** valid for I(0)/I(1)/I(2) mixtures
- **Monte Carlo simulation** tools replicating Chang (2000)
- **Publication-ready output** with LaTeX table export

---

## Why RBFM-VAR?

Traditional VAR estimation (OLS) suffers from **serious bias** when dealing with integrated time series, especially I(2) processes. The standard Wald test has **severe size distortions** in nonstationary settings.

**RBFM-VAR solves these problems by:**

1. Applying kernel-based corrections to eliminate asymptotic bias
2. Providing modified Wald tests with proper size control
3. Working without requiring prior knowledge of integration orders

---

## Quick Installation

```bash
pip install rbfmvar
```

---

## Simple Example

```python
from rbfmvar import RBFMVAR, generate_dgp

# Generate data with I(2) components
y, info = generate_dgp('case_c', T=200)

# Fit RBFM-VAR model
model = RBFMVAR(lag_order=1)
results = model.fit(y)

# Print results
print(results.summary())

# Test Granger causality
gc = results.granger_causality_test(
    caused_variable=0,
    causing_variables=[1]
)
print(f"P-value: {gc['p_value']:.4f}")
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **RBFM-VAR Estimator** | Bias-corrected estimates for mixed I(0)/I(1)/I(2) systems |
| **OLS-VAR Estimator** | Standard estimation for comparison |
| **Kernel Covariance** | Bartlett, Parzen, Quadratic Spectral kernels |
| **Automatic Bandwidth** | Andrews (1991) optimal selection |
| **Modified Wald Test** | Conservative test with bounded size (Theorem 2) |
| **Granger Causality** | Valid inference for nonstationary systems |
| **Monte Carlo** | Replicate Tables 1-2 from Chang (2000) |
| **LaTeX Export** | Publication-ready coefficient tables |

---

## Documentation

- [Installation Guide](installation) - How to install the package
- [Quick Start](quickstart) - Get started in 5 minutes
- [User Guide](userguide) - Complete usage guide
- [API Reference](api) - Full API documentation
- [Examples](examples) - Detailed examples with code
- [Theory](theory) - Mathematical background
- [Simulation](simulation) - Monte Carlo studies
- [Changelog](changelog) - Version history

---

## Reference

This package implements the methodology from:

> **Chang, Y. (2000)**. "Vector Autoregressions with Unknown Mixtures of I(0), I(1), and I(2) Components". *Econometric Theory*, Vol. 16, No. 6, pp. 905-926.
> 
> [Paper on JSTOR](http://www.jstor.org/stable/3533260)

---

## Citation

If you use this package in your research, please cite:

```bibtex
@software{rbfmvar,
  author = {Roudane, Merwan},
  title = {RBFM-VAR: Residual-Based Fully Modified Vector Autoregression},
  year = {2025},
  url = {https://github.com/merwanroudane/rbfmvar}
}

@article{chang2000,
  author = {Chang, Yoosoon},
  title = {Vector Autoregressions with Unknown Mixtures of {I(0)}, {I(1)}, and {I(2)} Components},
  journal = {Econometric Theory},
  volume = {16},
  number = {6},
  pages = {905--926},
  year = {2000}
}
```

---

## Author

**Dr. Merwan Roudane**

- Email: merwanroudane920@gmail.com
- GitHub: [merwanroudane](https://github.com/merwanroudane)

---

## License

MIT License - see [LICENSE](https://github.com/merwanroudane/rbfmvar/blob/main/LICENSE) for details.
