---
layout: default
title: Theory
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Theoretical Background

[← Back to Home](.)

Mathematical foundations of the RBFM-VAR methodology.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Model Specification](#model-specification)
3. [ECM Representation](#ecm-representation)
4. [The v̂ Process](#the-v-process)
5. [RBFM-VAR Estimator](#rbfm-var-estimator)
6. [Asymptotic Theory](#asymptotic-theory)
7. [Modified Wald Test](#modified-wald-test)
8. [Kernel Covariance Estimation](#kernel-covariance-estimation)

---

## The Problem

### Traditional VAR Limitations

Standard OLS estimation of VAR models faces serious problems when dealing with **integrated time series**:

1. **Bias**: OLS estimates are biased in finite samples with I(1) or I(2) components
2. **Nonstandard distributions**: t-statistics don't follow t-distributions
3. **Size distortions**: Wald tests reject too often under the null
4. **Unknown integration orders**: We often don't know if series are I(0), I(1), or I(2)

### The RBFM-VAR Solution

Chang (2000) proposes the **Residual-Based Fully Modified VAR (RBFM-VAR)** estimator that:

- Works without knowing which components are I(0), I(1), or I(2)
- Removes asymptotic bias from OLS estimates
- Provides a modified Wald test with proper size control
- Achieves optimal efficiency in the class of FM estimators

---

## Model Specification

Consider the VAR(p) model:

$$
y_t = A_1 y_{t-1} + A_2 y_{t-2} + \cdots + A_p y_{t-p} + \varepsilon_t
$$

where:
- \\( y_t \\) is an \\( n \times 1 \\) vector of time series
- \\( A_i \\) are \\( n \times n \\) coefficient matrices
- \\( \varepsilon_t \\) is an i.i.d. error with \\( E[\varepsilon_t] = 0 \\) and \\( E[\varepsilon_t \varepsilon_t'] = \Sigma_{\varepsilon\varepsilon} \\)

### Component Structure

The variables \\( y_t \\) may contain unknown mixtures of:

- **I(0) components**: Stationary
- **I(1) components**: Unit root (integrated of order 1)
- **I(2) components**: Double unit root (integrated of order 2)

The key insight is that we can estimate consistently **without knowing** which components are which.

---

## ECM Representation

### Error Correction Form

The VAR can be rewritten in Error Correction Model (ECM) form:

$$
y_t = \Phi z_t + A w_t + \varepsilon_t
$$

where:

$$
z_t = (\Delta^2 y_{t-1}', \ldots, \Delta^2 y_{t-p+2}')' \quad \text{(lagged second differences)}
$$

$$
w_t = (\Delta y_{t-1}', y_{t-1}')' \quad \text{(first difference and level)}
$$

The coefficient matrix is:

$$
F = (\Phi, A) = (\Phi_1, \ldots, \Phi_{p-2}, \Pi_1, \Pi_2)
$$

### ECM Coefficients

The matrices \\( \Pi_1 \\) and \\( \Pi_2 \\) are:

$$
\Pi_1 = -\sum_{k=2}^{p} (k-1) A_k \quad \text{(coefficient for } \Delta y_{t-1} \text{)}
$$

$$
\Pi_2 = \sum_{k=1}^{p} A_k \quad \text{(coefficient for } y_{t-1} \text{)}
$$

---

## The v̂ Process

### Definition

A key innovation in RBFM-VAR is the construction of the \\( \hat{v}_t \\) process (eq. 11):

$$
\hat{v}_t = \begin{pmatrix} \hat{v}_{1t} \\ \hat{v}_{2t} \end{pmatrix} = \begin{pmatrix} \Delta^2 y_{t-1} \\ \Delta y_{t-1} - \hat{N} \Delta y_{t-2} \end{pmatrix}
$$

where \\( \hat{N} \\) is the OLS coefficient from regressing \\( \Delta y_{t-1} \\) on \\( \Delta y_{t-2} \\).

### Purpose

The \\( \hat{v}_t \\) process is used to:
1. Estimate the long-run covariance structure
2. Construct the bias correction terms
3. Account for correlation between regressors and errors

---

## RBFM-VAR Estimator

### The Correction

The RBFM-VAR estimator applies a correction to the OLS estimator (eq. 12-13):

$$
\hat{F}^+ = (Y'Z, Y^{+'}W + T\hat{A}^+)(X'X)^{-1}
$$

where the corrected dependent variable is:

$$
Y^{+'} = Y' - \hat{\Omega}_{\hat{\varepsilon}\hat{v}} \hat{\Omega}_{\hat{v}\hat{v}}^{-1} \hat{V}'
$$

and the coefficient correction is:

$$
\hat{A}^+ = \hat{\Omega}_{\hat{\varepsilon}\hat{v}} \hat{\Omega}_{\hat{v}\hat{v}}^{-1} \hat{\Delta}_{\hat{v}\Delta w}
$$

### Covariance Matrices

- \\( \hat{\Omega}_{\hat{\varepsilon}\hat{v}} \\): Long-run cross-covariance of residuals and \\( \hat{v}_t \\)
- \\( \hat{\Omega}_{\hat{v}\hat{v}} \\): Long-run variance of \\( \hat{v}_t \\)
- \\( \hat{\Delta}_{\hat{v}\Delta w} \\): One-sided cross-covariance

---

## Asymptotic Theory

### Theorem 1: Limit Distribution

**Part (a) - Stationary Component:**

For the stationary part of the regressor:

$$
\sqrt{T}(\hat{F}^+ - F)G^1 \xrightarrow{d} N(0, \Sigma_{\varepsilon\varepsilon} \otimes \Sigma_{x11}^{-1})
$$

This is a standard normal limit with the usual \\( \sqrt{T} \\) rate.

**Part (b) - Nonstationary Component:**

For the nonstationary part:

$$
(\hat{F}^+ - F)G^b D_T \xrightarrow{d} MN\left(0, \Omega_{\varepsilon\varepsilon \cdot 2} \otimes \left(\int_0^1 \bar{B}_b \bar{B}_b' \right)^{-1}\right)
$$

This is a mixed normal limit, where:
- \\( D_T \\) is a diagonal normalization matrix
- \\( \bar{B}_b \\) is a demeaned Brownian motion
- \\( \Omega_{\varepsilon\varepsilon \cdot 2} \\) is the conditional long-run variance

### Bandwidth Requirements

For part (a): bandwidth expansion rate \\( k \in (1/4, 1/2) \\)

For part (b): bandwidth expansion rate \\( k \in (0, 1/3) \\)

---

## Modified Wald Test

### Theorem 2: Test Distribution

The modified Wald test statistic is:

$$
W_{F^+} = T(\text{vec}(R\hat{F}^+) - r)'[R(\hat{\Sigma}_{\varepsilon\varepsilon} \otimes T(X'X)^{-1})R']^{-1}(\text{vec}(R\hat{F}^+) - r)
$$

Under the null hypothesis \\( H_0: RF = r \\):

$$
W_{F^+} \xrightarrow{d} \chi^2_{q_1}(q_\Phi + q_{A_1}) + \sum_{i=1}^{q^1} d_i \chi^2_{q_{Ab}(i)}
$$

### Key Properties

1. **Mixture of chi-squares**: The limit is a weighted sum of chi-square distributions
2. **Eigenvalue weights**: The weights \\( d_i \\) satisfy \\( 0 \leq d_i \leq 1 \\)
3. **Conservative bound**: \\( W_{F^+} \\) is bounded above by \\( \chi^2_q \\)

### Practical Implication

Use standard \\( \chi^2_q \\) critical values for a **conservative test**:
- Actual size ≤ nominal size
- Valid regardless of I(0)/I(1)/I(2) composition

---

## Kernel Covariance Estimation

### Long-Run Covariance

The long-run covariance is estimated by:

$$
\hat{\Omega} = \sum_{j=-K}^{K} w(j/K) \hat{\Gamma}(j)
$$

where \\( \hat{\Gamma}(j) \\) is the sample autocovariance at lag \\( j \\).

### Kernel Functions

**Bartlett (Newey-West):**

$$
w(x) = \begin{cases} 1 - |x| & \text{if } |x| \leq 1 \\ 0 & \text{otherwise} \end{cases}
$$

**Parzen:**

$$
w(x) = \begin{cases} 
1 - 6x^2 + 6|x|^3 & \text{if } |x| \leq 0.5 \\
2(1-|x|)^3 & \text{if } 0.5 < |x| \leq 1 \\
0 & \text{otherwise}
\end{cases}
$$

**Quadratic Spectral:**

$$
w(x) = \frac{25}{12\pi^2 x^2}\left(\frac{\sin(6\pi x/5)}{6\pi x/5} - \cos(6\pi x/5)\right)
$$

### Optimal Bandwidth

Andrews (1991) optimal bandwidth for the Bartlett kernel:

$$
K = \lfloor 1.1447 (\hat{\alpha} T)^{1/3} \rfloor
$$

where \\( \hat{\alpha} \\) is estimated from an AR(1) approximation.

---

## Granger Causality

### Definition

Variable \\( y_j \\) does **not** Granger-cause variable \\( y_i \\) if:

$$
\Pi_1[i,j] = 0 \quad \text{and} \quad \Pi_2[i,j] = 0
$$

### Testing

The null hypothesis for Granger non-causality can be written as:

$$
H_0: R \text{vec}(F) = 0
$$

where \\( R \\) selects the appropriate coefficients.

### Using Modified Wald Test

The modified Wald test provides **conservative inference**:
- Under the null (no causality): P(reject) ≤ α
- Under the alternative (causality exists): Test has power

---

## Monte Carlo Evidence

### DGP from Chang (2000)

$$
\Delta y_{1t} = \rho_1 \Delta y_{1,t-1} + \rho_2 (y_{1,t-1} - \Delta y_{2,t-1}) + \varepsilon_{1t}
$$

$$
\Delta^2 y_{2t} = \varepsilon_{2t}
$$

| Case | ρ₁ | ρ₂ | Description |
|------|-----|-----|-------------|
| A | 1 | 0 | Both I(2), no causality |
| B | 0.5 | 0 | y₁ I(1), y₂ I(2), no causality |
| C | -0.3 | -0.15 | y₁ I(1), y₂ I(2), causality |

### Key Results (T=150, 10,000 replications)

**Size (Cases A, B):**
- OLS Wald test: Severely oversized (25-40% rejection at 5% level)
- Modified Wald test: Well-controlled size (≤5%)

**Power (Case C):**
- Both tests have good power to detect causality
- Modified test slightly less powerful (trade-off for size control)

---

## References

**Primary Reference:**

Chang, Y. (2000). "Vector Autoregressions with Unknown Mixtures of I(0), I(1), and I(2) Components". *Econometric Theory*, 16(6), 905-926.

**Related Literature:**

- Andrews, D.W.K. (1991). "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation". *Econometrica*, 59(3), 817-858.

- Newey, W.K. and West, K.D. (1994). "Automatic Lag Selection in Covariance Matrix Estimation". *Review of Economic Studies*, 61(4), 631-653.

- Phillips, P.C.B. and Hansen, B.E. (1990). "Statistical Inference in Instrumental Variables Regression with I(1) Processes". *Review of Economic Studies*, 57(1), 99-125.

---

[← Back to Home](.)
