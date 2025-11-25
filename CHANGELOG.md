# Changelog

All notable changes to the RBFM-VAR package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-11-25

### Major Release - Complete Rewrite
- Complete implementation of RBFM-VAR estimator following Chang (2000)
- OLS-VAR estimator for comparison
- Kernel-based long-run covariance estimation:
  - Bartlett (Newey-West) kernel
  - Parzen kernel
  - Quadratic Spectral kernel
- Automatic bandwidth selection (Andrews 1991)
- Modified Wald test for conservative inference
- Standard Wald test for hypothesis testing
- Granger causality testing framework
- Monte Carlo simulation module:
  - Three DGPs from Chang (2000) Section 5
  - Bias and RMSE computation
  - Test size and power analysis
- Publication-ready output:
  - Formatted summary tables
  - LaTeX table export
  - Comparison with OLS-VAR
- Comprehensive documentation and examples

### Core Features
- ECM representation of VAR models
- v̂ process construction (eq. 11)
- RBFM-VAR correction terms (eq. 12-13)
- Π₁ and Π₂ matrix extraction
- Forecasting capabilities

### Testing
- Unit tests for all modules
- Validation against paper results

## [0.0.1] - 2024-XX-XX

### Added
- Initial project structure
- Basic package skeleton

---

## References

Chang, Y. (2000). "Vector Autoregressions with Unknown Mixtures of I(0), I(1),
and I(2) Components". Econometric Theory, Vol. 16, No. 6, pp. 905-926.
