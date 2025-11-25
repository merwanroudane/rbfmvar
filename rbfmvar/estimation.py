"""
RBFM-VAR Estimation Module.

This module implements the Residual-Based Fully Modified Vector Autoregression
(RBFM-VAR) estimator developed by Yoosoon Chang (2000).

The RBFM-VAR estimator is defined as (eq. 12):
    F̂⁺ = (Φ̂⁺, Â⁺) = (Y'Z, Y⁺'W + TÂ⁺)(X'X)⁻¹

with correction terms (eq. 13):
    Y⁺' = Y' - Ω̂_{ε̂v̂} Ω̂⁻¹_{v̂v̂} V̂'
    Â⁺ = Ω̂_{ε̂v̂} Ω̂⁻¹_{v̂v̂} Δ̂_{v̂Δw}

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
from numpy.linalg import inv, pinv
from typing import Optional, Union, Tuple, List, Dict, Any
import warnings

from .utils import (
    create_ecm_matrices,
    create_full_regressor_matrix,
    difference,
    vec,
    unvec,
    kronecker_product,
    moore_penrose_inverse,
)
from .covariance import (
    kernel_covariance,
    one_sided_kernel_covariance,
    cross_kernel_covariance,
    one_sided_cross_covariance,
    optimal_bandwidth,
    safe_inverse,
    estimate_contemporaneous_covariance,
)


class OLSVAREstimator:
    """
    Ordinary Least Squares VAR Estimator.
    
    Estimates the VAR model:
        y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + ε_t
    
    reformulated as:
        y_t = Φz_t + Aw_t + ε_t
    
    where z_t contains lagged second differences and w_t contains
    first differences and levels.
    
    Parameters
    ----------
    lag_order : int
        Number of lags p in the VAR
        
    Attributes
    ----------
    coefficients : np.ndarray
        Estimated coefficient matrix F = (Φ, A)
    residuals : np.ndarray
        OLS residuals ε̂
    sigma_epsilon : np.ndarray
        Estimated error covariance Σ̂_εε
        
    References
    ----------
    Chang (2000), eq. (1)-(3), (10)
    """
    
    def __init__(self, lag_order: int = 1):
        self.lag_order = lag_order
        self.coefficients = None
        self.residuals = None
        self.sigma_epsilon = None
        self._Y = None
        self._Z = None
        self._W = None
        self._X = None
        self._T = None
        self._n = None
        self._is_fitted = False
    
    def fit(self, y: np.ndarray) -> 'OLSVAREstimator':
        """
        Fit the OLS-VAR model.
        
        Parameters
        ----------
        y : np.ndarray
            Time series data of shape (T, n)
            
        Returns
        -------
        OLSVAREstimator
            Fitted estimator
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        T_original, n = y.shape
        self._n = n
        
        # Create matrices for ECM representation
        Y, Z, W = create_ecm_matrices(y, self.lag_order)
        self._Y = Y
        self._Z = Z
        self._W = W
        self._T = Y.shape[0]
        
        # Full regressor matrix X = (Z, W)
        if Z.shape[1] > 0:
            X = np.hstack([Z, W])
        else:
            X = W
        self._X = X
        
        # OLS estimation: F̂ = (Y'X)(X'X)⁻¹
        XtX = X.T @ X
        XtY = X.T @ Y
        
        # Solve for coefficients
        self.coefficients = np.linalg.solve(XtX.T, XtY.T).T  # F is n x k
        
        # Compute residuals
        self.residuals = Y - X @ self.coefficients.T
        
        # Estimate error covariance
        self.sigma_epsilon = self.residuals.T @ self.residuals / self._T
        
        self._is_fitted = True
        return self
    
    def get_phi_and_A(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the Φ and A coefficient matrices separately.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Phi : Coefficients for lagged second differences
            A : Coefficients for (Δy_{t-1}, y_{t-1})
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n = self._n
        p = self.lag_order
        
        # Φ is n x n(p-1) for lagged second differences
        phi_cols = n * (p - 1) if p > 1 else 0
        
        if phi_cols > 0:
            Phi = self.coefficients[:, :phi_cols]
        else:
            Phi = np.zeros((n, 0))
        
        # A is n x 2n for (Δy_{t-1}, y_{t-1})
        A = self.coefficients[:, phi_cols:]
        
        return Phi, A
    
    def predict(self, y: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Generate out-of-sample predictions.
        
        Parameters
        ----------
        y : np.ndarray
            Historical data
        steps : int
            Number of steps to forecast
            
        Returns
        -------
        np.ndarray
            Forecasts
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Implementation of forecasting
        forecasts = []
        y_extended = y.copy()
        
        for _ in range(steps):
            # Get last observations for regressors
            Y_new, Z_new, W_new = create_ecm_matrices(y_extended, self.lag_order)
            if Z_new.shape[1] > 0:
                X_new = np.hstack([Z_new[-1:], W_new[-1:]])
            else:
                X_new = W_new[-1:]
            
            # Predict
            y_pred = X_new @ self.coefficients.T
            forecasts.append(y_pred.flatten())
            
            # Extend y for next prediction
            y_extended = np.vstack([y_extended, y_pred])
        
        return np.array(forecasts)


class RBFMVAR:
    """
    Residual-Based Fully Modified Vector Autoregression (RBFM-VAR) Estimator.
    
    Implements the RBFM-VAR methodology from Chang (2000) for estimation of
    VAR models with unknown mixtures of I(0), I(1), and I(2) components.
    
    The RBFM-VAR estimator is defined as (eq. 12):
        F̂⁺ = (Φ̂⁺, Â⁺) = (Y'Z, Y⁺'W + TÂ⁺)(X'X)⁻¹
    
    with correction terms (eq. 13):
        Y⁺' = Y' - Ω̂_{ε̂v̂} Ω̂⁻¹_{v̂v̂} V̂'
        Â⁺ = Ω̂_{ε̂v̂} Ω̂⁻¹_{v̂v̂} Δ̂_{v̂Δw}
    
    Parameters
    ----------
    lag_order : int
        Number of lags p in the VAR model
    bandwidth : int or str
        Bandwidth for kernel estimation. If 'auto', uses automatic selection.
    kernel : str
        Kernel function: 'bartlett', 'parzen', or 'qs'
    trend : str
        Trend specification: 'n' (none), 'c' (constant), 'ct' (constant + trend)
        
    Attributes
    ----------
    ols_coefficients : np.ndarray
        OLS-VAR coefficient estimates
    coefficients : np.ndarray
        RBFM-VAR coefficient estimates F̂⁺
    residuals : np.ndarray
        Model residuals
    sigma_epsilon : np.ndarray
        Estimated error covariance
        
    Examples
    --------
    >>> import numpy as np
    >>> from rbfmvar import RBFMVAR
    >>> 
    >>> # Generate data
    >>> np.random.seed(42)
    >>> T, n = 200, 2
    >>> data = np.cumsum(np.cumsum(np.random.randn(T, n), axis=0), axis=0)
    >>> 
    >>> # Fit model
    >>> model = RBFMVAR(lag_order=2)
    >>> results = model.fit(data)
    >>> print(results.summary())
    
    References
    ----------
    Chang, Y. (2000). Vector Autoregressions with Unknown Mixtures of I(0),
    I(1), and I(2) Components. Econometric Theory, 16(6), 905-926.
    """
    
    def __init__(self, 
                 lag_order: int = 1,
                 bandwidth: Union[int, str] = 'auto',
                 kernel: str = 'bartlett',
                 trend: str = 'c'):
        
        self.lag_order = lag_order
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.trend = trend
        
        # Estimation results
        self.ols_coefficients = None
        self.coefficients = None
        self.residuals = None
        self.sigma_epsilon = None
        
        # Internal storage
        self._Y = None
        self._Z = None
        self._W = None
        self._X = None
        self._V = None
        self._v_hat = None
        self._N_hat = None
        self._T = None
        self._n = None
        self._y = None
        self._ols_residuals = None
        self._Omega_ev = None
        self._Omega_vv = None
        self._Delta_vdw = None
        self._K = None
        self._is_fitted = False
    
    def _compute_v_process(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the v̂_t process defined in eq. (11).
        
        v̂_t = (v̂'_{1t}, v̂'_{2t})' = (Δ²y_{t-1}, Δy_{t-1} - N̂Δy_{t-2})'
        
        where N̂ is the OLS coefficient from regression of Δy_{t-1} on Δy_{t-2}.
        
        Parameters
        ----------
        y : np.ndarray
            Time series data
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            v_hat : The v̂_t process
            V : Matrix form V̂
            N_hat : Estimated N̂ coefficient
            
        References
        ----------
        Chang (2000), eq. (11) and eq. (14)-(17)
        """
        T_orig, n = y.shape
        p = self.lag_order
        
        # Compute differences
        delta_y = difference(y, 1)      # Δy_t: (T_orig-1, n)
        delta2_y = difference(y, 2)     # Δ²y_t: (T_orig-2, n)
        
        # Effective sample size (matching the main regression)
        T_eff = self._T
        
        # For the v-process, we need Δy_{t-1} and Δy_{t-2}
        # The main regression uses observations from t = p+2 to T_orig
        # So we need Δy from t = p+1 to T_orig-1 for Δy_{t-1}
        # and Δy from t = p to T_orig-2 for Δy_{t-2}
        
        # Δ²y_{t-1}: starts at index p (which gives Δ²y_{p+1})
        # and has T_eff observations
        v1_start = max(0, p - 1)
        v1_end = v1_start + T_eff
        if v1_end > len(delta2_y):
            v1_end = len(delta2_y)
            T_eff = v1_end - v1_start
        v1 = delta2_y[v1_start:v1_end]  # (T_eff, n)
        
        # For the N̂ regression: Δy_{t-1} on Δy_{t-2}
        # We use the full sample of first differences
        T_delta = len(delta_y)
        
        # Δy_{t-1} and Δy_{t-2} for regression
        delta_y_t1 = delta_y[1:]        # Δy_{t-1}, t from 2 to T_orig-1
        delta_y_t2 = delta_y[:-1]       # Δy_{t-2}, t from 2 to T_orig-1
        
        # OLS: N̂ = (Δy_{t-2}' Δy_{t-2})⁻¹ (Δy_{t-2}' Δy_{t-1})
        # Note: We regress Δy_{t-1} on Δy_{t-2}
        XtX = delta_y_t2.T @ delta_y_t2
        XtY = delta_y_t2.T @ delta_y_t1
        
        try:
            N_hat = np.linalg.solve(XtX, XtY).T  # (n, n)
        except np.linalg.LinAlgError:
            N_hat = pinv(XtX) @ XtY
            N_hat = N_hat.T
        
        # Construct v̂_{2t} = Δy_{t-1} - N̂Δy_{t-2}
        # Align with the main regression sample
        # We need T_eff observations starting appropriately
        
        # Indices for Δy_{t-1} aligned with the main regression
        dy_t1_start = p
        dy_t1_end = dy_t1_start + T_eff
        if dy_t1_end > len(delta_y):
            dy_t1_end = len(delta_y)
            T_eff = min(T_eff, dy_t1_end - dy_t1_start, len(v1))
        
        # Indices for Δy_{t-2}
        dy_t2_start = dy_t1_start - 1
        dy_t2_end = dy_t2_start + T_eff
        
        delta_y_t1_aligned = delta_y[dy_t1_start:dy_t1_start + T_eff]
        delta_y_t2_aligned = delta_y[dy_t2_start:dy_t2_start + T_eff]
        
        # Compute v̂_{2t}
        v2 = delta_y_t1_aligned - delta_y_t2_aligned @ N_hat.T  # (T_eff, n)
        
        # Ensure v1 and v2 have same length
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        
        # Stack to form v̂_t = (v̂'_{1t}, v̂'_{2t})'
        v_hat = np.hstack([v1, v2])  # (min_len, 2n)
        
        # Also compute V̂ for matrix form
        V = v_hat
        
        return v_hat, V, N_hat
    
    def _compute_correction_terms(self, 
                                  epsilon_hat: np.ndarray,
                                  v_hat: np.ndarray,
                                  W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute the RBFM-VAR correction terms from eq. (13).
        
        Y⁺' = Y' - Ω̂_{ε̂v̂} Ω̂⁻¹_{v̂v̂} V̂'
        Â⁺ = Ω̂_{ε̂v̂} Ω̂⁻¹_{v̂v̂} Δ̂_{v̂Δw}
        
        Parameters
        ----------
        epsilon_hat : np.ndarray
            OLS residuals
        v_hat : np.ndarray
            The v̂_t process
        W : np.ndarray
            The W matrix (Δy_{t-1}, y_{t-1})
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, int]
            Y_correction : Correction for Y matrix
            A_correction : Correction for A coefficient
            min_T : The effective sample size used
            
        References
        ----------
        Chang (2000), eq. (13)
        """
        T = epsilon_hat.shape[0]
        
        # Compute Δw_t = Δ(Δy_{t-1}, y_{t-1})' = (Δ²y_{t-1}, Δy_{t-1})'
        delta_W = difference(W, 1)
        
        # Align with v_hat - use the minimum of all lengths
        min_T = min(len(epsilon_hat), len(v_hat), len(delta_W))
        epsilon_aligned = epsilon_hat[:min_T]
        v_aligned = v_hat[:min_T]
        delta_W_aligned = delta_W[:min_T]
        
        # Determine bandwidth
        if self.bandwidth == 'auto':
            combined = np.hstack([epsilon_aligned, v_aligned])
            K = optimal_bandwidth(combined, self.kernel)
        else:
            K = self.bandwidth
        
        self._K = K
        
        # Compute covariance matrices
        # Ω̂_{ε̂v̂}: Long-run cross-covariance
        Omega_ev = cross_kernel_covariance(epsilon_aligned, v_aligned, K, self.kernel)
        
        # Ω̂_{v̂v̂}: Long-run variance of v̂
        Omega_vv = kernel_covariance(v_aligned, K, self.kernel)
        
        # Δ̂_{v̂Δw}: One-sided cross-covariance
        Delta_vdw = one_sided_cross_covariance(v_aligned, delta_W_aligned, K, self.kernel)
        
        # Store for later use
        self._Omega_ev = Omega_ev
        self._Omega_vv = Omega_vv
        self._Delta_vdw = Delta_vdw
        
        # Compute Ω̂⁻¹_{v̂v̂} using Moore-Penrose inverse (may be singular)
        Omega_vv_inv = safe_inverse(Omega_vv)
        
        # Y correction: Ω̂_{ε̂v̂} Ω̂⁻¹_{v̂v̂} V̂'
        Y_correction = Omega_ev @ Omega_vv_inv @ v_aligned.T  # (n, min_T)
        
        # A correction: Ω̂_{ε̂v̂} Ω̂⁻¹_{v̂v̂} Δ̂_{v̂Δw}
        A_correction = Omega_ev @ Omega_vv_inv @ Delta_vdw  # (n, 2n)
        
        return Y_correction.T, A_correction, min_T
    
    def fit(self, y: np.ndarray) -> 'RBFMVARResults':
        """
        Fit the RBFM-VAR model.
        
        Parameters
        ----------
        y : np.ndarray
            Time series data of shape (T, n)
            
        Returns
        -------
        RBFMVARResults
            Object containing estimation results
            
        Notes
        -----
        The estimation follows the procedure in Chang (2000):
        1. Compute OLS-VAR estimates
        2. Construct the v̂_t process (eq. 11)
        3. Estimate covariance matrices using kernel methods
        4. Apply correction terms to obtain RBFM-VAR estimates
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self._y = y.copy()
        T_original, n = y.shape
        self._n = n
        
        # Step 1: Create matrices for regression
        Y, Z, W = create_ecm_matrices(y, self.lag_order)
        self._Y = Y
        self._Z = Z  
        self._W = W
        self._T = Y.shape[0]
        
        # Full regressor matrix X = (Z, W)
        if Z.shape[1] > 0:
            X = np.hstack([Z, W])
        else:
            X = W
        self._X = X
        
        # Step 2: Compute OLS estimates
        XtX = X.T @ X
        XtY = X.T @ Y
        
        # OLS: F̂ = Y'X(X'X)⁻¹
        try:
            ols_coef_T = np.linalg.solve(XtX, XtY)  # This is F'
        except np.linalg.LinAlgError:
            ols_coef_T = pinv(XtX) @ XtY
        
        self.ols_coefficients = ols_coef_T.T  # F = (Φ, A) is n x k
        
        # OLS residuals
        self._ols_residuals = Y - X @ ols_coef_T
        
        # Estimate error covariance
        self.sigma_epsilon = self._ols_residuals.T @ self._ols_residuals / self._T
        
        # Step 3: Compute v̂_t process (eq. 11)
        v_hat, V, N_hat = self._compute_v_process(y)
        self._v_hat = v_hat
        self._V = V
        self._N_hat = N_hat
        
        # Align dimensions
        min_T = min(self._ols_residuals.shape[0], v_hat.shape[0], W.shape[0])
        epsilon_aligned = self._ols_residuals[:min_T]
        v_aligned = v_hat[:min_T]
        W_aligned = W[:min_T]
        
        # Step 4: Compute correction terms
        Y_correction, A_correction, min_T = self._compute_correction_terms(
            epsilon_aligned, v_aligned, W_aligned
        )
        
        # Step 5: Compute RBFM-VAR estimates (eq. 12)
        # F̂⁺ = (Y'Z, Y⁺'W + TÂ⁺)(X'X)⁻¹
        
        # Y⁺ = Y - correction (use min_T from correction computation)
        Y_plus = Y[:min_T] - Y_correction
        
        # Reconstruct the corrected products
        Z_aligned = Z[:min_T] if Z.shape[1] > 0 else None
        W_final = W[:min_T]
        
        # Compute (Y'Z, Y⁺'W + T*Â⁺)
        if Z_aligned is not None and Z_aligned.shape[1] > 0:
            YtZ = Y[:min_T].T @ Z_aligned  # n x n(p-1)
            Yplus_tW = Y_plus.T @ W_final + self._T * A_correction
            numerator = np.hstack([YtZ, Yplus_tW])
        else:
            Yplus_tW = Y_plus.T @ W_final + self._T * A_correction
            numerator = Yplus_tW
        
        # (X'X)⁻¹
        X_aligned = X[:min_T]
        XtX_aligned = X_aligned.T @ X_aligned
        
        try:
            XtX_inv = inv(XtX_aligned)
        except np.linalg.LinAlgError:
            XtX_inv = pinv(XtX_aligned)
        
        # F̂⁺ = numerator * (X'X)⁻¹
        self.coefficients = numerator @ XtX_inv  # n x k
        
        # Compute residuals with RBFM-VAR coefficients
        self.residuals = Y[:min_T] - X_aligned @ self.coefficients.T
        
        self._is_fitted = True
        
        # Return results object
        from .results import RBFMVARResults
        return RBFMVARResults(self)
    
    def get_phi_and_A(self, use_rbfm: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the Φ and A coefficient matrices separately.
        
        Parameters
        ----------
        use_rbfm : bool
            If True, return RBFM-VAR estimates; else OLS estimates
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Phi : Coefficients for lagged second differences
            A : Coefficients for (Δy_{t-1}, y_{t-1})
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        coef = self.coefficients if use_rbfm else self.ols_coefficients
        n = self._n
        p = self.lag_order
        
        # Φ is n x n(p-1) for lagged second differences
        phi_cols = n * (p - 1) if p > 1 else 0
        
        if phi_cols > 0:
            Phi = coef[:, :phi_cols]
        else:
            Phi = np.zeros((n, 0))
        
        # A is n x 2n for (Δy_{t-1}, y_{t-1})
        A = coef[:, phi_cols:]
        
        return Phi, A
    
    def get_Pi_matrices(self, use_rbfm: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the Π₁ and Π₂ matrices from ECM representation.
        
        For the ECM: Δ²y_t = Φ(L)Δ²y_{t-1} + Π₁Δy_{t-1} + Π₂y_{t-1} + ε_t
        
        Parameters
        ----------
        use_rbfm : bool
            If True, return RBFM-VAR estimates; else OLS estimates
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Pi1 : n x n matrix for Δy_{t-1}
            Pi2 : n x n matrix for y_{t-1}
        """
        Phi, A = self.get_phi_and_A(use_rbfm)
        n = self._n
        
        # A = (A_delta, A_level) where A_delta is for Δy_{t-1}, A_level for y_{t-1}
        Pi1 = A[:, :n]   # Coefficient for Δy_{t-1}
        Pi2 = A[:, n:]   # Coefficient for y_{t-1}
        
        return Pi1, Pi2
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Generate out-of-sample forecasts.
        
        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast
            
        Returns
        -------
        np.ndarray
            Forecasts of shape (steps, n)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        y = self._y
        forecasts = []
        y_extended = y.copy()
        
        for _ in range(steps):
            # Get last observations for regressors
            Y_new, Z_new, W_new = create_ecm_matrices(y_extended, self.lag_order)
            if Z_new.shape[1] > 0:
                X_new = np.hstack([Z_new[-1:], W_new[-1:]])
            else:
                X_new = W_new[-1:]
            
            # Predict using RBFM-VAR coefficients
            y_pred = X_new @ self.coefficients.T
            forecasts.append(y_pred.flatten())
            
            # Extend y for next prediction
            y_extended = np.vstack([y_extended, y_pred])
        
        return np.array(forecasts)


def fit_rbfm_var(y: np.ndarray, 
                 lag_order: int = 1,
                 bandwidth: Union[int, str] = 'auto',
                 kernel: str = 'bartlett') -> 'RBFMVARResults':
    """
    Convenience function to fit RBFM-VAR model.
    
    Parameters
    ----------
    y : np.ndarray
        Time series data
    lag_order : int
        Number of lags
    bandwidth : int or str
        Bandwidth for kernel estimation
    kernel : str
        Kernel function
        
    Returns
    -------
    RBFMVARResults
        Estimation results
        
    Examples
    --------
    >>> from rbfmvar import fit_rbfm_var
    >>> results = fit_rbfm_var(data, lag_order=2)
    >>> print(results.summary())
    """
    model = RBFMVAR(lag_order=lag_order, bandwidth=bandwidth, kernel=kernel)
    return model.fit(y)
