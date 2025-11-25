---
layout: default
title: Changelog
---

# Changelog

[← Back to Home](.)

All notable changes to the RBFM-VAR package.

---

## [2.0.0] - 2025-11-25

### 🎉 Major Release - Complete Rewrite

This is a complete rewrite of the RBFM-VAR package with full implementation of the Chang (2000) methodology.

### ✨ New Features

#### Core Estimation
- **RBFMVAR class**: Main estimator with bias correction
- **OLSVAREstimator class**: Standard OLS-VAR for comparison
- **RBFMVARResults class**: Comprehensive results container
- Automatic bandwidth selection (Andrews 1991)
- Multiple kernel functions (Bartlett, Parzen, Quadratic Spectral)

#### Hypothesis Testing
- **Modified Wald test**: Conservative inference valid for I(0)/I(1)/I(2)
- **Standard Wald test**: For comparison
- **Granger causality testing**: With proper size control
- General linear hypothesis testing

#### Covariance Estimation
- Long-run covariance (symmetric)
- One-sided covariance
- Cross-covariance
- Automatic bandwidth selection
- Multiple kernel options

#### Monte Carlo Simulation
- Three DGPs from Chang (2000) Section 5
- Bias and RMSE computation
- Test size and power analysis
- Replication of Tables 1-2

#### Output and Export
- Publication-ready summary tables
- LaTeX table export
- Comparison tables (RBFM-VAR vs OLS-VAR)
- Dictionary export for programmatic access

### 📚 Documentation
- Comprehensive README
- Installation guide
- Quick start tutorial
- User guide
- API reference
- Examples with code
- Theoretical background
- Simulation guide

### 🧪 Testing
- Unit tests for all modules
- Test coverage for core functionality
- Validation against paper results

### 📦 Package Structure
```
rbfmvar/
├── __init__.py      # Package exports
├── utils.py         # Utility functions
├── covariance.py    # Kernel covariance estimation
├── estimation.py    # RBFMVAR and OLSVAREstimator
├── testing.py       # Hypothesis tests
├── results.py       # Results class
└── simulation.py    # Monte Carlo simulation
```

---

## [1.0.2] - 2025-11-12

### ⚠️ Deprecated

This version had implementation issues and has been superseded by v2.0.0.

---

## [1.0.1] - 2025-11-XX

### ⚠️ Deprecated

Initial release with bugs. Superseded by v2.0.0.

---

## [1.0.0] - 2025-11-XX

### ⚠️ Deprecated

Initial release. Superseded by v2.0.0.

---

## Upgrade Guide

### From v1.x to v2.0.0

Version 2.0.0 is a complete rewrite. The API has changed significantly.

**Old (v1.x):**
```python
# Not working properly
from rbfmvar import rbfmvar_estimate
results = rbfmvar_estimate(y, p=1)
```

**New (v2.0.0):**
```python
# Working correctly
from rbfmvar import RBFMVAR
model = RBFMVAR(lag_order=1)
results = model.fit(y)
print(results.summary())
```

### Key Changes

| Feature | v1.x | v2.0.0 |
|---------|------|--------|
| Main class | N/A | `RBFMVAR` |
| Results | Basic | `RBFMVARResults` with full diagnostics |
| Granger test | Not working | `granger_causality_test()` |
| LaTeX export | No | `to_latex()` |
| Simulation | No | Full Monte Carlo module |
| Documentation | Minimal | Comprehensive |

---

## Roadmap

### Planned for v2.1.0
- [ ] Structural break detection
- [ ] Cointegration rank testing
- [ ] Impulse response functions
- [ ] Variance decomposition

### Planned for v2.2.0
- [ ] Panel RBFM-VAR
- [ ] Bootstrap inference
- [ ] Bayesian extension

### Under Consideration
- [ ] Seasonal adjustment
- [ ] Mixed frequency data
- [ ] Large VAR methods

---

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/merwanroudane/rbfmvar) for:

- Bug reports
- Feature requests
- Pull requests

---

## Reference

Chang, Y. (2000). "Vector Autoregressions with Unknown Mixtures of I(0), I(1), and I(2) Components". *Econometric Theory*, 16(6), 905-926.

---

[← Back to Home](.)
