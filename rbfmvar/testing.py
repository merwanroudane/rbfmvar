"""
Hypothesis Testing Module for RBFM-VAR.

This module implements the modified Wald tests for hypothesis testing in
RBFM-VAR regressions, including Granger causality tests.

Key results from Chang (2000):
- Theorem 2: The modified Wald statistic has a limit distribution that is
  a mixture of chi-square variates, bounded above by χ² distribution
- The standard Wald test has nonstandard distributions and nuisance parameter
  dependence when rank condition (19) fails

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
from numpy.linalg import inv, pinv, eigvals
from scipy import stats
from typing import Optional, Union, Tuple, List, Dict, Any
import warnings

from .utils import vec, unvec, kronecker_product, moore_penrose_inverse


def wald_test(F: np.ndarray, 
              R: np.ndarray, 
              r: np.ndarray,
              Sigma_F: np.ndarray) -> Dict[str, Any]:
    """
    Compute Wald test statistic for linear restrictions.
    
    Tests H₀: R vec(F) = r
    
    W = (R vec(F) - r)' [R Var(vec(F)) R']⁻¹ (R vec(F) - r)
    
    Parameters
    ----------
    F : np.ndarray
        Coefficient matrix (n x k)
    R : np.ndarray
        Restriction matrix (q x nk)
    r : np.ndarray
        Restriction values (q,)
    Sigma_F : np.ndarray
        Covariance matrix of vec(F) (nk x nk)
        
    Returns
    -------
    Dict with keys:
        - 'statistic': Wald test statistic
        - 'p_value': p-value from chi-square distribution
        - 'df': Degrees of freedom
        
    References
    ----------
    Chang (2000), eq. (18)
    """
    # Vectorize F
    vec_F = vec(F)
    
    # Compute restriction deviation
    deviation = R @ vec_F - r
    
    # Variance of the restriction
    var_restriction = R @ Sigma_F @ R.T
    
    # Invert (using pseudo-inverse if singular)
    try:
        var_inv = inv(var_restriction)
    except np.linalg.LinAlgError:
        var_inv = pinv(var_restriction)
    
    # Wald statistic
    W = deviation @ var_inv @ deviation
    
    # Degrees of freedom
    q = len(r)
    
    # P-value from chi-square
    p_value = 1 - stats.chi2.cdf(W, q)
    
    return {
        'statistic': float(W),
        'p_value': float(p_value),
        'df': q,
        'deviation': deviation,
    }


def modified_wald_test(F_plus: np.ndarray,
                       R: np.ndarray,
                       r: np.ndarray,
                       Sigma_epsilon: np.ndarray,
                       XtX: np.ndarray,
                       T: int) -> Dict[str, Any]:
    """
    Compute the modified Wald test statistic for RBFM-VAR.
    
    W_F⁺ = T(R vec(F̂⁺) - r)'(R(Σ̂_εε ⊗ T(X'X)⁻¹)R')⁻¹(R vec(F̂⁺) - r)
    
    Under rank condition failure, this statistic has a limit distribution
    that is a mixture of chi-square variates bounded above by χ²_q.
    
    Parameters
    ----------
    F_plus : np.ndarray
        RBFM-VAR coefficient estimates (n x k)
    R : np.ndarray
        Restriction matrix (q x nk)
    r : np.ndarray
        Restriction values (q,)
    Sigma_epsilon : np.ndarray
        Estimated error covariance (n x n)
    XtX : np.ndarray
        X'X matrix (k x k)
    T : int
        Sample size
        
    Returns
    -------
    Dict with keys:
        - 'statistic': Modified Wald test statistic
        - 'p_value': Conservative p-value from χ² distribution
        - 'df': Degrees of freedom (upper bound)
        - 'is_conservative': True (always for this test)
        
    Notes
    -----
    The p-value is conservative because the true limit distribution is
    bounded above by χ²_q (Theorem 2, Remark (a)).
    
    References
    ----------
    Chang (2000), eq. (20), Theorem 2
    """
    n, k = F_plus.shape
    
    # Vectorize F⁺
    vec_F = vec(F_plus)
    
    # Compute restriction deviation
    deviation = R @ vec_F - r
    
    # Compute the variance matrix: Σ̂_εε ⊗ T(X'X)⁻¹
    # Note: We need (X'X)⁻¹, not T(X'X)⁻¹
    try:
        XtX_inv = inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = pinv(XtX)
    
    # Variance of vec(F̂⁺): Σ_εε ⊗ (X'X)⁻¹
    Var_F = kronecker_product(Sigma_epsilon, XtX_inv)
    
    # Variance of restriction
    var_restriction = R @ Var_F @ R.T
    
    # Scale by T for the test statistic
    var_restriction_scaled = var_restriction
    
    # Invert
    try:
        var_inv = inv(var_restriction_scaled)
    except np.linalg.LinAlgError:
        var_inv = pinv(var_restriction_scaled)
    
    # Modified Wald statistic
    W_F_plus = T * (deviation @ var_inv @ deviation)
    
    # Degrees of freedom (conservative upper bound)
    q = len(r)
    
    # Conservative p-value
    p_value = 1 - stats.chi2.cdf(W_F_plus, q)
    
    return {
        'statistic': float(W_F_plus),
        'p_value': float(p_value),
        'df': q,
        'is_conservative': True,
        'deviation': deviation,
    }


def compute_eigenvalue_weights(R1: np.ndarray,
                               Omega_ee: np.ndarray,
                               Sigma_ee: np.ndarray) -> np.ndarray:
    """
    Compute the eigenvalue weights d_i for the mixture distribution.
    
    The weights are eigenvalues of:
    (R₁ Ω_{εε·2} R₁')^{1/2} (R₁ Σ_εε R₁')⁻¹ (R₁ Ω_{εε·2} R₁')^{1/2}
    
    Parameters
    ----------
    R1 : np.ndarray
        Left restriction matrix from Kronecker decomposition
    Omega_ee : np.ndarray
        Long-run covariance Ω_{εε·2}
    Sigma_ee : np.ndarray
        Contemporaneous covariance Σ_εε
        
    Returns
    -------
    np.ndarray
        Eigenvalue weights d_1, ..., d_{q_1}
        
    Notes
    -----
    The weights satisfy 0 ≤ d_i ≤ 1 (Theorem 2, Remark (a)).
    
    References
    ----------
    Chang (2000), Theorem 2
    """
    # Compute R₁ Ω_{εε·2} R₁'
    R_Omega_R = R1 @ Omega_ee @ R1.T
    
    # Compute R₁ Σ_εε R₁'
    R_Sigma_R = R1 @ Sigma_ee @ R1.T
    
    # Compute square roots
    try:
        # Eigendecomposition of R_Omega_R
        eigvals_O, eigvecs_O = np.linalg.eigh(R_Omega_R)
        eigvals_O = np.maximum(eigvals_O, 0)  # Ensure non-negative
        sqrt_R_Omega_R = eigvecs_O @ np.diag(np.sqrt(eigvals_O)) @ eigvecs_O.T
        
        # Inverse of R_Sigma_R
        R_Sigma_R_inv = inv(R_Sigma_R)
        
        # Matrix product
        M = sqrt_R_Omega_R @ R_Sigma_R_inv @ sqrt_R_Omega_R
        
        # Eigenvalues are the weights
        weights = np.real(eigvals(M))
        
        # Ensure weights are in [0, 1]
        weights = np.clip(weights, 0, 1)
        
    except np.linalg.LinAlgError:
        # Fallback: return ones (most conservative)
        weights = np.ones(R1.shape[0])
    
    return np.sort(weights)[::-1]  # Sort descending


def granger_causality_test(results, 
                           caused_variable: int,
                           causing_variables: Union[int, List[int]],
                           test_type: str = 'modified') -> Dict[str, Any]:
    """
    Test for Granger causality in RBFM-VAR model.
    
    Tests whether variable(s) in causing_variables Granger-cause the
    variable specified by caused_variable.
    
    The null hypothesis is:
    H₀: The causing variable(s) do not Granger-cause the caused variable
    
    This corresponds to testing that the relevant coefficients in Π₁ and Π₂
    are zero, as formulated in eq. (25) of Chang (2000).
    
    Parameters
    ----------
    results : RBFMVARResults
        Fitted RBFM-VAR results
    caused_variable : int
        Index of the variable being caused (0-indexed)
    causing_variables : int or list of int
        Index(es) of the causing variable(s)
    test_type : str
        Type of test: 'modified' (RBFM-VAR) or 'standard' (OLS-VAR)
        
    Returns
    -------
    Dict with keys:
        - 'statistic': Wald test statistic
        - 'p_value': P-value (conservative for modified test)
        - 'df': Degrees of freedom
        - 'null_hypothesis': Description of null hypothesis
        - 'test_type': Type of test used
        
    Notes
    -----
    For the modified Wald test based on RBFM-VAR estimates, the limiting
    distribution is bounded above by χ²_q, so the test is asymptotically
    conservative (Theorem 2).
    
    For the standard Wald test based on OLS-VAR, the limiting distribution
    may be nonstandard and nuisance parameter dependent when I(1) or I(2)
    components are involved (see Toda and Phillips, 1993, 1994).
    
    Examples
    --------
    >>> # Test if variable 1 Granger-causes variable 0
    >>> causality = granger_causality_test(results, 
    ...                                    caused_variable=0, 
    ...                                    causing_variables=[1])
    >>> print(f"Test statistic: {causality['statistic']:.4f}")
    >>> print(f"P-value: {causality['p_value']:.4f}")
    
    References
    ----------
    Chang (2000), Section 5, eq. (25); Toda & Phillips (1993, 1994)
    """
    if isinstance(causing_variables, int):
        causing_variables = [causing_variables]
    
    model = results._model
    n = model._n
    p = model.lag_order
    
    # Get coefficient estimates
    if test_type.lower() == 'modified':
        F = model.coefficients
        is_modified = True
    else:
        F = model.ols_coefficients
        is_modified = False
    
    # Dimension of coefficient matrix: n x k where k = n(p-1) + 2n
    k = F.shape[1]
    
    # Build restriction matrix R for testing that the causing variables
    # have zero coefficients in the equation for the caused variable
    
    # The coefficient matrix F = (Φ, A) where:
    # - Φ is n x n(p-1) for lagged Δ²y
    # - A is n x 2n for (Δy_{t-1}, y_{t-1})
    
    # For Granger causality, we test that in the caused_variable equation,
    # the coefficients on the causing_variables are zero in both Π₁ and Π₂
    
    # Number of restrictions: 2 per causing variable (one in Π₁, one in Π₂)
    # Plus (p-1) restrictions for each lagged Δ² term if p > 1
    
    restrictions = []
    phi_cols = n * (p - 1) if p > 1 else 0
    
    for j in causing_variables:
        # Coefficient in Π₁ (at column phi_cols + j)
        col_Pi1 = phi_cols + j
        
        # Coefficient in Π₂ (at column phi_cols + n + j)
        col_Pi2 = phi_cols + n + j
        
        # Create restriction vectors
        # For vec(F), we select the (caused_variable, col) element
        # vec(F) stacks columns, so position is: col * n + caused_variable
        
        # Restriction on Π₁ coefficient
        r1 = np.zeros(n * k)
        r1[col_Pi1 * n + caused_variable] = 1
        restrictions.append(r1)
        
        # Restriction on Π₂ coefficient
        r2 = np.zeros(n * k)
        r2[col_Pi2 * n + caused_variable] = 1
        restrictions.append(r2)
        
        # Also test lagged Δ² coefficients if p > 1
        if p > 1:
            for lag in range(p - 1):
                col_phi = lag * n + j
                r_phi = np.zeros(n * k)
                r_phi[col_phi * n + caused_variable] = 1
                restrictions.append(r_phi)
    
    R = np.array(restrictions)
    r = np.zeros(len(restrictions))
    
    # Get required matrices for the test
    Sigma_epsilon = model.sigma_epsilon
    XtX = model._X.T @ model._X
    T = model._T
    
    # Perform the test
    if is_modified:
        test_result = modified_wald_test(F, R, r, Sigma_epsilon, XtX, T)
    else:
        # Standard Wald test using OLS variance estimate
        try:
            XtX_inv = inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = pinv(XtX)
        
        Var_F = kronecker_product(Sigma_epsilon, XtX_inv)
        test_result = wald_test(F, R, r, Var_F)
    
    # Format output
    causing_str = ', '.join([f'y{j+1}' for j in causing_variables])
    caused_str = f'y{caused_variable + 1}'
    
    return {
        'statistic': test_result['statistic'],
        'p_value': test_result['p_value'],
        'df': test_result['df'],
        'null_hypothesis': f'{causing_str} does not Granger-cause {caused_str}',
        'test_type': 'Modified Wald (RBFM-VAR)' if is_modified else 'Standard Wald (OLS-VAR)',
        'is_conservative': test_result.get('is_conservative', False),
        'caused_variable': caused_variable,
        'causing_variables': causing_variables,
    }


def joint_significance_test(results,
                           equation: int,
                           variables: List[int],
                           test_type: str = 'modified') -> Dict[str, Any]:
    """
    Test joint significance of multiple variables in an equation.
    
    Parameters
    ----------
    results : RBFMVARResults
        Fitted RBFM-VAR results
    equation : int
        Index of the equation (0-indexed)
    variables : list of int
        Indices of variables to test
    test_type : str
        Type of test: 'modified' or 'standard'
        
    Returns
    -------
    Dict with test results
    """
    return granger_causality_test(results, equation, variables, test_type)


def rank_test(Pi2: np.ndarray, 
              r: int,
              method: str = 'trace') -> Dict[str, Any]:
    """
    Test the rank of Π₂ = αβ' matrix.
    
    Tests H₀: rank(Π₂) ≤ r vs H₁: rank(Π₂) > r
    
    Parameters
    ----------
    Pi2 : np.ndarray
        The Π₂ matrix (n x n)
    r : int
        Hypothesized maximum rank
    method : str
        Test method: 'trace' or 'max_eigenvalue'
        
    Returns
    -------
    Dict with test results
        
    Notes
    -----
    This is a simplified implementation. For full reduced-rank testing,
    see Johansen (1991, 1995).
    
    References
    ----------
    Chang (2000), Theorem 2, Remark (c); Johansen (1995)
    """
    n = Pi2.shape[0]
    
    # Compute singular values
    singular_values = np.linalg.svd(Pi2, compute_uv=False)
    
    # Estimate rank
    tol = np.max(Pi2.shape) * np.finfo(float).eps * singular_values[0]
    estimated_rank = np.sum(singular_values > tol)
    
    # Simple test based on singular values
    if method == 'trace':
        # Sum of squared singular values beyond rank r
        if r < n:
            statistic = np.sum(singular_values[r:]**2)
        else:
            statistic = 0.0
    else:
        # Maximum singular value beyond rank r
        if r < n:
            statistic = singular_values[r]**2
        else:
            statistic = 0.0
    
    return {
        'statistic': float(statistic),
        'estimated_rank': int(estimated_rank),
        'singular_values': singular_values,
        'method': method,
        'null_hypothesis': f'rank(Π₂) ≤ {r}',
    }


def coefficient_test(results,
                    equation: int,
                    coefficient_idx: int,
                    hypothesized_value: float = 0.0,
                    test_type: str = 'modified') -> Dict[str, Any]:
    """
    Test a single coefficient restriction.
    
    Tests H₀: F[equation, coefficient_idx] = hypothesized_value
    
    Parameters
    ----------
    results : RBFMVARResults
        Fitted RBFM-VAR results
    equation : int
        Row index in F (0-indexed)
    coefficient_idx : int  
        Column index in F (0-indexed)
    hypothesized_value : float
        Value under null hypothesis
    test_type : str
        Type of test: 'modified' or 'standard'
        
    Returns
    -------
    Dict with test results
    """
    model = results._model
    n = model._n
    k = model.coefficients.shape[1]
    
    # Get coefficient estimates
    if test_type.lower() == 'modified':
        F = model.coefficients
        is_modified = True
    else:
        F = model.ols_coefficients
        is_modified = False
    
    # Create single restriction
    R = np.zeros((1, n * k))
    # Position in vec(F): coefficient_idx * n + equation
    R[0, coefficient_idx * n + equation] = 1
    r = np.array([hypothesized_value])
    
    # Get required matrices
    Sigma_epsilon = model.sigma_epsilon
    XtX = model._X.T @ model._X
    T = model._T
    
    if is_modified:
        test_result = modified_wald_test(F, R, r, Sigma_epsilon, XtX, T)
    else:
        try:
            XtX_inv = inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = pinv(XtX)
        Var_F = kronecker_product(Sigma_epsilon, XtX_inv)
        test_result = wald_test(F, R, r, Var_F)
    
    # Compute t-statistic equivalent
    coefficient_value = F[equation, coefficient_idx]
    std_error = np.sqrt(test_result['statistic']) / np.abs(coefficient_value - hypothesized_value) if np.abs(coefficient_value - hypothesized_value) > 1e-10 else np.nan
    t_stat = (coefficient_value - hypothesized_value) / std_error if not np.isnan(std_error) else np.nan
    
    return {
        'statistic': test_result['statistic'],
        'p_value': test_result['p_value'],
        'df': 1,
        'coefficient': float(coefficient_value),
        'hypothesized_value': hypothesized_value,
        't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
        'equation': equation,
        'coefficient_idx': coefficient_idx,
        'test_type': 'Modified Wald' if is_modified else 'Standard Wald',
    }
