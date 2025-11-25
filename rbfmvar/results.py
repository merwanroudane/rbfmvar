"""
Results Module for RBFM-VAR.

This module provides a comprehensive results class for RBFM-VAR estimation
with publication-ready output formatting.

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
from numpy.linalg import inv, pinv
from typing import Optional, Union, List, Dict, Any
from tabulate import tabulate
import warnings


class RBFMVARResults:
    """
    Results class for RBFM-VAR estimation.
    
    Contains estimation results, diagnostic information, and methods for
    hypothesis testing and output formatting.
    
    Attributes
    ----------
    coefficients : np.ndarray
        RBFM-VAR coefficient estimates F̂⁺
    ols_coefficients : np.ndarray
        OLS-VAR coefficient estimates F̂
    residuals : np.ndarray
        Model residuals
    sigma_epsilon : np.ndarray
        Estimated error covariance matrix Σ̂_εε
    n_obs : int
        Number of observations used in estimation
    n_vars : int
        Number of variables
    lag_order : int
        Lag order of the VAR
    bandwidth : int
        Bandwidth used for kernel estimation
    kernel : str
        Kernel function used
        
    Methods
    -------
    summary()
        Print comprehensive model summary
    granger_causality_test()
        Test for Granger causality
    wald_test()
        General linear hypothesis test
    to_latex()
        Export results as LaTeX tables
    to_dict()
        Export results as dictionary
        
    References
    ----------
    Chang, Y. (2000). Vector Autoregressions with Unknown Mixtures of I(0),
    I(1), and I(2) Components. Econometric Theory, 16(6), 905-926.
    """
    
    def __init__(self, model):
        """
        Initialize results from fitted RBFM-VAR model.
        
        Parameters
        ----------
        model : RBFMVAR
            Fitted RBFM-VAR model instance
        """
        self._model = model
        
        # Core results
        self.coefficients = model.coefficients
        self.ols_coefficients = model.ols_coefficients
        self.residuals = model.residuals
        self.sigma_epsilon = model.sigma_epsilon
        
        # Model information
        self.n_obs = model._T
        self.n_vars = model._n
        self.lag_order = model.lag_order
        self.bandwidth = model._K if hasattr(model, '_K') else None
        self.kernel = model.kernel
        
        # ECM matrices
        self.Pi1, self.Pi2 = model.get_Pi_matrices(use_rbfm=True)
        self.Pi1_ols, self.Pi2_ols = model.get_Pi_matrices(use_rbfm=False)
        
        # Covariance matrices for correction terms
        self._Omega_ev = model._Omega_ev if hasattr(model, '_Omega_ev') else None
        self._Omega_vv = model._Omega_vv if hasattr(model, '_Omega_vv') else None
        self._Delta_vdw = model._Delta_vdw if hasattr(model, '_Delta_vdw') else None
        
        # Compute standard errors
        self._compute_standard_errors()
        
        # Compute model diagnostics
        self._compute_diagnostics()
    
    def _compute_standard_errors(self):
        """Compute standard errors for coefficient estimates."""
        model = self._model
        n = self.n_vars
        k = self.coefficients.shape[1]
        T = self.n_obs
        
        # X'X matrix
        XtX = model._X.T @ model._X
        
        try:
            XtX_inv = inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = pinv(XtX)
        
        # Variance of vec(F): Σ_εε ⊗ (X'X)⁻¹
        # Standard errors are sqrt of diagonal elements
        
        # For each coefficient F[i,j], variance is Σ_εε[i,i] * (X'X)⁻¹[j,j]
        self.std_errors = np.zeros_like(self.coefficients)
        self.std_errors_ols = np.zeros_like(self.ols_coefficients)
        
        for i in range(n):
            for j in range(k):
                var_ij = self.sigma_epsilon[i, i] * XtX_inv[j, j]
                self.std_errors[i, j] = np.sqrt(max(0, var_ij))
                self.std_errors_ols[i, j] = np.sqrt(max(0, var_ij))
        
        # Compute t-statistics
        with np.errstate(divide='ignore', invalid='ignore'):
            self.t_statistics = self.coefficients / self.std_errors
            self.t_statistics_ols = self.ols_coefficients / self.std_errors_ols
        
        # P-values (two-sided)
        from scipy import stats
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_statistics), T - k))
        self.p_values_ols = 2 * (1 - stats.t.cdf(np.abs(self.t_statistics_ols), T - k))
    
    def _compute_diagnostics(self):
        """Compute model diagnostics."""
        T = self.n_obs
        n = self.n_vars
        k = self.coefficients.shape[1]
        
        # Residual sum of squares
        self.rss = np.sum(self.residuals**2, axis=0)
        
        # Total sum of squares (using mean)
        Y = self._model._Y[:self.residuals.shape[0]]
        Y_mean = Y.mean(axis=0)
        self.tss = np.sum((Y - Y_mean)**2, axis=0)
        
        # R-squared
        with np.errstate(divide='ignore', invalid='ignore'):
            self.r_squared = 1 - self.rss / self.tss
            self.r_squared = np.clip(self.r_squared, 0, 1)
        
        # Adjusted R-squared
        self.adj_r_squared = 1 - (1 - self.r_squared) * (T - 1) / (T - k - 1)
        
        # Log-likelihood (assuming Gaussian errors)
        log_det_sigma = np.log(np.linalg.det(self.sigma_epsilon) + 1e-10)
        self.log_likelihood = -0.5 * T * (n * np.log(2 * np.pi) + log_det_sigma + n)
        
        # Information criteria
        n_params = n * k + n * (n + 1) / 2  # Coefficients + covariance parameters
        self.aic = -2 * self.log_likelihood + 2 * n_params
        self.bic = -2 * self.log_likelihood + np.log(T) * n_params
        self.hqc = -2 * self.log_likelihood + 2 * np.log(np.log(T)) * n_params
    
    def summary(self, alpha: float = 0.05) -> str:
        """
        Generate comprehensive model summary.
        
        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals
            
        Returns
        -------
        str
            Formatted summary string
        """
        n = self.n_vars
        k = self.coefficients.shape[1]
        p = self.lag_order
        
        lines = []
        lines.append("=" * 80)
        lines.append("RBFM-VAR Estimation Results".center(80))
        lines.append("Residual-Based Fully Modified Vector Autoregression".center(80))
        lines.append("=" * 80)
        lines.append("")
        
        # Model information
        lines.append("Model Information:")
        lines.append("-" * 40)
        lines.append(f"  Number of equations:       {n}")
        lines.append(f"  Lag order (p):            {p}")
        lines.append(f"  Observations used:        {self.n_obs}")
        lines.append(f"  Parameters per equation:  {k}")
        lines.append(f"  Kernel function:          {self.kernel}")
        if self.bandwidth:
            lines.append(f"  Bandwidth (K):            {self.bandwidth}")
        lines.append("")
        
        # Goodness of fit
        lines.append("Goodness of Fit:")
        lines.append("-" * 40)
        for i in range(n):
            lines.append(f"  Equation {i+1}:")
            lines.append(f"    R-squared:          {self.r_squared[i]:.6f}")
            lines.append(f"    Adj. R-squared:     {self.adj_r_squared[i]:.6f}")
        lines.append("")
        
        lines.append(f"  Log-Likelihood:       {self.log_likelihood:.4f}")
        lines.append(f"  AIC:                  {self.aic:.4f}")
        lines.append(f"  BIC:                  {self.bic:.4f}")
        lines.append(f"  HQC:                  {self.hqc:.4f}")
        lines.append("")
        
        # Coefficient estimates
        lines.append("RBFM-VAR Coefficient Estimates:")
        lines.append("-" * 80)
        
        # Create coefficient table
        headers = ["Variable", "Coefficient", "Std. Error", "t-statistic", "P-value", "Signif."]
        
        phi_cols = n * (p - 1) if p > 1 else 0
        
        for eq in range(n):
            lines.append(f"\nEquation {eq+1} (y{eq+1}):")
            rows = []
            
            col_idx = 0
            
            # Lagged Δ² coefficients
            if p > 1:
                for lag in range(1, p):
                    for var in range(n):
                        coef = self.coefficients[eq, col_idx]
                        se = self.std_errors[eq, col_idx]
                        t_stat = self.t_statistics[eq, col_idx]
                        p_val = self.p_values[eq, col_idx]
                        
                        signif = ""
                        if p_val < 0.01:
                            signif = "***"
                        elif p_val < 0.05:
                            signif = "**"
                        elif p_val < 0.10:
                            signif = "*"
                        
                        var_name = f"Δ²y{var+1}(t-{lag})"
                        rows.append([var_name, f"{coef:.6f}", f"{se:.6f}", 
                                   f"{t_stat:.4f}", f"{p_val:.4f}", signif])
                        col_idx += 1
            
            # Π₁ coefficients (Δy_{t-1})
            for var in range(n):
                coef = self.coefficients[eq, col_idx]
                se = self.std_errors[eq, col_idx]
                t_stat = self.t_statistics[eq, col_idx]
                p_val = self.p_values[eq, col_idx]
                
                signif = ""
                if p_val < 0.01:
                    signif = "***"
                elif p_val < 0.05:
                    signif = "**"
                elif p_val < 0.10:
                    signif = "*"
                
                var_name = f"Δy{var+1}(t-1)"
                rows.append([var_name, f"{coef:.6f}", f"{se:.6f}",
                           f"{t_stat:.4f}", f"{p_val:.4f}", signif])
                col_idx += 1
            
            # Π₂ coefficients (y_{t-1})
            for var in range(n):
                coef = self.coefficients[eq, col_idx]
                se = self.std_errors[eq, col_idx]
                t_stat = self.t_statistics[eq, col_idx]
                p_val = self.p_values[eq, col_idx]
                
                signif = ""
                if p_val < 0.01:
                    signif = "***"
                elif p_val < 0.05:
                    signif = "**"
                elif p_val < 0.10:
                    signif = "*"
                
                var_name = f"y{var+1}(t-1)"
                rows.append([var_name, f"{coef:.6f}", f"{se:.6f}",
                           f"{t_stat:.4f}", f"{p_val:.4f}", signif])
                col_idx += 1
            
            lines.append(tabulate(rows, headers=headers, tablefmt="simple"))
        
        lines.append("")
        lines.append("Significance codes: *** p<0.01, ** p<0.05, * p<0.10")
        lines.append("")
        
        # Error covariance matrix
        lines.append("Error Covariance Matrix (Σ̂_εε):")
        lines.append("-" * 40)
        sigma_str = np.array2string(self.sigma_epsilon, precision=6, 
                                    suppress_small=True)
        lines.append(sigma_str)
        lines.append("")
        
        # Π₁ and Π₂ matrices
        lines.append("ECM Coefficient Matrices:")
        lines.append("-" * 40)
        lines.append("\nΠ₁ (coefficient for Δy_{t-1}):")
        lines.append(np.array2string(self.Pi1, precision=6, suppress_small=True))
        lines.append("\nΠ₂ (coefficient for y_{t-1}):")
        lines.append(np.array2string(self.Pi2, precision=6, suppress_small=True))
        lines.append("")
        
        # Notes
        lines.append("=" * 80)
        lines.append("Notes:")
        lines.append("- RBFM-VAR estimates are consistent with mixed normal limit theory")
        lines.append("- P-values are based on standard asymptotic theory")
        lines.append("- For Wald tests, use granger_causality_test() for conservative inference")
        lines.append("")
        lines.append("Reference: Chang (2000), Econometric Theory 16(6), 905-926")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """String representation."""
        return self.summary()
    
    def __repr__(self) -> str:
        """Repr representation."""
        return f"RBFMVARResults(n_vars={self.n_vars}, lag_order={self.lag_order}, n_obs={self.n_obs})"
    
    def granger_causality_test(self, 
                               caused_variable: int,
                               causing_variables: Union[int, List[int]],
                               test_type: str = 'modified') -> Dict[str, Any]:
        """
        Test for Granger causality.
        
        Parameters
        ----------
        caused_variable : int
            Index of the variable being caused (0-indexed)
        causing_variables : int or list of int
            Index(es) of the causing variable(s)
        test_type : str
            Type of test: 'modified' (default) or 'standard'
            
        Returns
        -------
        Dict with test results
            
        Examples
        --------
        >>> # Test if variable 1 Granger-causes variable 0
        >>> result = results.granger_causality_test(0, [1])
        >>> print(f"Statistic: {result['statistic']:.4f}")
        >>> print(f"P-value: {result['p_value']:.4f}")
        """
        from .testing import granger_causality_test
        return granger_causality_test(self, caused_variable, causing_variables, test_type)
    
    def wald_test(self, 
                  R: np.ndarray, 
                  r: np.ndarray,
                  test_type: str = 'modified') -> Dict[str, Any]:
        """
        Perform general linear hypothesis test.
        
        Tests H₀: R vec(F) = r
        
        Parameters
        ----------
        R : np.ndarray
            Restriction matrix (q x nk)
        r : np.ndarray
            Restriction values (q,)
        test_type : str
            Type of test: 'modified' or 'standard'
            
        Returns
        -------
        Dict with test results
        """
        from .testing import modified_wald_test, wald_test
        
        model = self._model
        
        if test_type.lower() == 'modified':
            F = model.coefficients
            return modified_wald_test(
                F, R, r, 
                model.sigma_epsilon,
                model._X.T @ model._X,
                model._T
            )
        else:
            F = model.ols_coefficients
            XtX = model._X.T @ model._X
            try:
                XtX_inv = inv(XtX)
            except np.linalg.LinAlgError:
                XtX_inv = pinv(XtX)
            
            from .utils import kronecker_product
            Var_F = kronecker_product(model.sigma_epsilon, XtX_inv)
            return wald_test(F, R, r, Var_F)
    
    def to_latex(self, 
                 float_format: str = "%.4f",
                 include_std_errors: bool = True) -> str:
        """
        Export coefficient estimates as LaTeX table.
        
        Parameters
        ----------
        float_format : str
            Format string for floating point numbers
        include_std_errors : bool
            Whether to include standard errors in parentheses
            
        Returns
        -------
        str
            LaTeX table code
        """
        n = self.n_vars
        k = self.coefficients.shape[1]
        p = self.lag_order
        
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{RBFM-VAR Estimation Results}")
        lines.append("\\label{tab:rbfmvar}")
        
        # Column specification
        col_spec = "l" + "c" * n
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        
        # Header
        header = "Variable"
        for i in range(n):
            header += f" & Equation {i+1}"
        header += " \\\\"
        lines.append(header)
        lines.append("\\midrule")
        
        phi_cols = n * (p - 1) if p > 1 else 0
        
        # Coefficients
        col_idx = 0
        
        # Lagged Δ² coefficients
        if p > 1:
            for lag in range(1, p):
                for var in range(n):
                    row = f"$\\Delta^2 y_{{{var+1},t-{lag}}}$"
                    for eq in range(n):
                        coef = self.coefficients[eq, col_idx]
                        se = self.std_errors[eq, col_idx]
                        
                        if include_std_errors:
                            row += f" & {coef:{float_format[1:]}} ({se:{float_format[1:]}})"
                        else:
                            row += f" & {coef:{float_format[1:]}}"
                    
                    row += " \\\\"
                    lines.append(row)
                    col_idx += 1
        
        # Π₁ coefficients
        for var in range(n):
            row = f"$\\Delta y_{{{var+1},t-1}}$"
            for eq in range(n):
                coef = self.coefficients[eq, col_idx + var]
                se = self.std_errors[eq, col_idx + var]
                
                if include_std_errors:
                    row += f" & {coef:{float_format[1:]}} ({se:{float_format[1:]}})"
                else:
                    row += f" & {coef:{float_format[1:]}}"
            
            row += " \\\\"
            lines.append(row)
        
        col_idx += n
        
        # Π₂ coefficients
        for var in range(n):
            row = f"$y_{{{var+1},t-1}}$"
            for eq in range(n):
                coef = self.coefficients[eq, col_idx + var]
                se = self.std_errors[eq, col_idx + var]
                
                if include_std_errors:
                    row += f" & {coef:{float_format[1:]}} ({se:{float_format[1:]}})"
                else:
                    row += f" & {coef:{float_format[1:]}}"
            
            row += " \\\\"
            lines.append(row)
        
        lines.append("\\midrule")
        
        # Diagnostics
        r2_row = "$R^2$"
        for i in range(n):
            r2_row += f" & {self.r_squared[i]:{float_format[1:]}}"
        r2_row += " \\\\"
        lines.append(r2_row)
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        
        # Notes
        lines.append("\\begin{tablenotes}")
        lines.append("\\small")
        if include_std_errors:
            lines.append("\\item Standard errors in parentheses.")
        lines.append(f"\\item Observations: {self.n_obs}. Lag order: {p}.")
        lines.append("\\item RBFM-VAR estimates following Chang (2000).")
        lines.append("\\end{tablenotes}")
        
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export results as dictionary.
        
        Returns
        -------
        Dict
            Dictionary containing all results
        """
        return {
            'coefficients': self.coefficients.tolist(),
            'ols_coefficients': self.ols_coefficients.tolist(),
            'std_errors': self.std_errors.tolist(),
            't_statistics': self.t_statistics.tolist(),
            'p_values': self.p_values.tolist(),
            'sigma_epsilon': self.sigma_epsilon.tolist(),
            'residuals': self.residuals.tolist(),
            'Pi1': self.Pi1.tolist(),
            'Pi2': self.Pi2.tolist(),
            'n_obs': self.n_obs,
            'n_vars': self.n_vars,
            'lag_order': self.lag_order,
            'bandwidth': self.bandwidth,
            'kernel': self.kernel,
            'r_squared': self.r_squared.tolist(),
            'adj_r_squared': self.adj_r_squared.tolist(),
            'log_likelihood': self.log_likelihood,
            'aic': self.aic,
            'bic': self.bic,
            'hqc': self.hqc,
        }
    
    def compare_with_ols(self) -> str:
        """
        Generate comparison between RBFM-VAR and OLS-VAR estimates.
        
        Returns
        -------
        str
            Formatted comparison table
        """
        n = self.n_vars
        k = self.coefficients.shape[1]
        
        lines = []
        lines.append("=" * 70)
        lines.append("Comparison: RBFM-VAR vs OLS-VAR Estimates".center(70))
        lines.append("=" * 70)
        lines.append("")
        
        headers = ["Coef", "RBFM-VAR", "OLS-VAR", "Difference", "% Diff"]
        
        for eq in range(n):
            lines.append(f"\nEquation {eq+1}:")
            rows = []
            
            for j in range(k):
                rbfm = self.coefficients[eq, j]
                ols = self.ols_coefficients[eq, j]
                diff = rbfm - ols
                pct_diff = 100 * diff / ols if abs(ols) > 1e-10 else np.nan
                
                rows.append([
                    f"[{eq+1},{j+1}]",
                    f"{rbfm:.6f}",
                    f"{ols:.6f}",
                    f"{diff:.6f}",
                    f"{pct_diff:.2f}%" if not np.isnan(pct_diff) else "N/A"
                ])
            
            lines.append(tabulate(rows, headers=headers, tablefmt="simple"))
        
        lines.append("")
        lines.append("Note: RBFM-VAR provides bias-corrected estimates for")
        lines.append("systems with unknown mixtures of I(0), I(1), I(2) components.")
        
        return "\n".join(lines)
    
    def forecast(self, steps: int = 1) -> np.ndarray:
        """
        Generate forecasts.
        
        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast
            
        Returns
        -------
        np.ndarray
            Forecasts of shape (steps, n_vars)
        """
        return self._model.predict(steps)
