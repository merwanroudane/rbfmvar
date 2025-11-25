"""
Utility functions for RBFM-VAR estimation.

This module provides essential utility functions for matrix operations,
data transformations, and lag generation used throughout the package.

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
from numpy.linalg import inv, pinv, matrix_rank
from typing import Union, Tuple, Optional, List


def vec(A: np.ndarray) -> np.ndarray:
    """
    Vectorize a matrix by stacking columns.
    
    Converts an (m x n) matrix A into an (mn x 1) column vector by
    stacking the columns of A.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (m, n)
        
    Returns
    -------
    np.ndarray
        Column vector of shape (m*n,)
        
    Notes
    -----
    This follows the standard vec operator convention used in econometrics.
    For row-major stacking (as in the paper), we use Fortran order.
    
    References
    ----------
    Chang (2000), notation section, p. 906
    """
    return A.flatten(order='F')


def unvec(v: np.ndarray, m: int, n: int) -> np.ndarray:
    """
    Reshape a vector back into a matrix.
    
    Inverse operation of vec().
    
    Parameters
    ----------
    v : np.ndarray
        Input vector of length m*n
    m : int
        Number of rows in output matrix
    n : int
        Number of columns in output matrix
        
    Returns
    -------
    np.ndarray
        Matrix of shape (m, n)
    """
    return v.reshape((m, n), order='F')


def kronecker_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute Kronecker product of two matrices.
    
    Parameters
    ----------
    A : np.ndarray
        First matrix
    B : np.ndarray
        Second matrix
        
    Returns
    -------
    np.ndarray
        Kronecker product A ⊗ B
        
    References
    ----------
    Used extensively in the Wald test formulations, see eq. (20) in Chang (2000)
    """
    return np.kron(A, B)


def difference(y: np.ndarray, d: int = 1, axis: int = 0) -> np.ndarray:
    """
    Compute the d-th difference of a time series.
    
    Parameters
    ----------
    y : np.ndarray
        Input array, typically (T x n) for multivariate series
    d : int, optional
        Order of differencing, default is 1
    axis : int, optional
        Axis along which to difference, default is 0 (time axis)
        
    Returns
    -------
    np.ndarray
        Differenced series of shape (T-d, n) for d-th difference
        
    Notes
    -----
    Δy_t = y_t - y_{t-1}
    Δ²y_t = Δy_t - Δy_{t-1} = y_t - 2y_{t-1} + y_{t-2}
    
    References
    ----------
    Chang (2000), eq. (1)-(2)
    """
    result = y.copy()
    for _ in range(d):
        result = np.diff(result, n=1, axis=axis)
    return result


def lag_matrix(y: np.ndarray, lags: Union[int, List[int]], 
               include_original: bool = False) -> np.ndarray:
    """
    Create a matrix of lagged values.
    
    Parameters
    ----------
    y : np.ndarray
        Input time series of shape (T, n)
    lags : int or list of int
        If int, create lags 1 to lags
        If list, create specific lags
    include_original : bool, optional
        If True, include lag 0 (original series)
        
    Returns
    -------
    np.ndarray
        Matrix of lagged values
        
    Examples
    --------
    >>> y = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> lag_matrix(y, 2)  # Creates lags 1 and 2
    """
    T, n = y.shape if y.ndim == 2 else (len(y), 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    if isinstance(lags, int):
        lag_list = list(range(1, lags + 1))
    else:
        lag_list = list(lags)
    
    max_lag = max(lag_list)
    T_eff = T - max_lag
    
    # Determine columns
    start_col = 1 if not include_original else 0
    n_lags = len(lag_list) + (1 if include_original else 0)
    
    result = np.zeros((T_eff, n * n_lags))
    
    col_idx = 0
    if include_original:
        result[:, :n] = y[max_lag:]
        col_idx = n
    
    for lag in lag_list:
        start_idx = max_lag - lag
        end_idx = T - lag
        result[:, col_idx:col_idx + n] = y[start_idx:end_idx]
        col_idx += n
    
    return result


def create_var_matrices(y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create matrices for VAR estimation.
    
    Creates the Y and X matrices for the regression:
    y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + ε_t
    
    Parameters
    ----------
    y : np.ndarray
        Time series data of shape (T, n)
    p : int
        Lag order
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Y : (T-p, n) matrix of dependent variables
        X : (T-p, n*p) matrix of lagged regressors
        
    References
    ----------
    Chang (2000), eq. (1)
    """
    T, n = y.shape
    Y = y[p:]  # (T-p, n)
    X = lag_matrix(y, p)  # (T-p, n*p)
    return Y, X


def create_ecm_matrices(y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create matrices for ECM representation of VAR.
    
    Creates the matrices for the ECM format:
    Δ²y_t = Φ(L)Δ²y_{t-1} + Π_1 Δy_{t-1} + Π_2 y_{t-1} + ε_t
    
    Which corresponds to regression (3):
    y_t = Φz_t + Aw_t + ε_t
    
    where:
    z_t = (Δ²y'_{t-1}, ..., Δ²y'_{t-p+2})'
    w_t = (Δy'_{t-1}, y'_{t-1})'
    
    Parameters
    ----------
    y : np.ndarray
        Time series data of shape (T, n)
    p : int
        Lag order
        
    Returns
    -------
    Tuple of arrays
        Y : Dependent variable (y_t)
        Z : Matrix of lagged second differences (z_t) 
        W : Matrix of first differences and levels (w_t)
        
    References
    ----------
    Chang (2000), eq. (2) and (3), p. 907
    """
    T, n = y.shape
    
    # Compute differences
    delta_y = difference(y, d=1)       # Δy_t: (T-1, n)
    delta2_y = difference(y, d=2)      # Δ²y_t: (T-2, n)
    
    # Determine effective sample size
    # We need p+1 lags for Δ²y and 1 lag for Δy and y
    T_eff = T - p - 1
    
    # Create Z matrix: lagged second differences Δ²y_{t-1}, ..., Δ²y_{t-p+2}
    # Number of Δ² lags is p-2 (if p >= 2)
    if p >= 2:
        Z_lags = p - 2 + 1  # From Δ²y_{t-1} to Δ²y_{t-p+2}
        Z = np.zeros((T_eff, n * (p - 1)))
        for lag in range(1, p):
            start_idx = p + 1 - lag - 2
            Z[:, (lag-1)*n:lag*n] = delta2_y[start_idx:start_idx + T_eff]
    else:
        Z = np.zeros((T_eff, 0))
    
    # Create W matrix: (Δy_{t-1}, y_{t-1})'
    # W is (T_eff, 2n)
    W = np.zeros((T_eff, 2 * n))
    
    # Δy_{t-1}: one lag of first differences
    W[:, :n] = delta_y[p-1:p-1+T_eff]
    
    # y_{t-1}: one lag of levels
    W[:, n:] = y[p:p+T_eff]
    
    # Y: dependent variable y_t
    Y = y[p+1:p+1+T_eff]
    
    return Y, Z, W


def create_full_regressor_matrix(y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create full regressor matrix X = (Z, W) for the model y_t = FX_t + ε_t.
    
    Parameters
    ----------
    y : np.ndarray
        Time series data of shape (T, n)
    p : int
        Lag order
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Y : Dependent variable matrix
        X : Full regressor matrix (Z, W)
        
    References
    ----------
    Chang (2000), eq. (10)
    """
    Y, Z, W = create_ecm_matrices(y, p)
    if Z.shape[1] > 0:
        X = np.hstack([Z, W])
    else:
        X = W
    return Y, X


def compute_phi_matrices(A_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute Φ matrices from A matrices.
    
    Φ = (Φ_1, ..., Φ_{p-2}) where Φ_i = Σ_{k=i}^p (k-i+1)A_k
    
    Parameters
    ----------
    A_list : list of np.ndarray
        List of coefficient matrices [A_1, A_2, ..., A_p]
        
    Returns
    -------
    list of np.ndarray
        List of Φ matrices [Φ_1, ..., Φ_{p-2}]
        
    References
    ----------
    Chang (2000), p. 907-908
    """
    p = len(A_list)
    n = A_list[0].shape[0]
    
    if p < 2:
        return []
    
    Phi_list = []
    for i in range(1, p - 1):  # i = 1, ..., p-2
        Phi_i = np.zeros((n, n))
        for k in range(i, p + 1):  # k = i, ..., p
            Phi_i += (k - i + 1) * A_list[k - 1]
        Phi_list.append(Phi_i)
    
    return Phi_list


def compute_A_from_original(A_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the A matrix (coefficient for w_t) from original VAR coefficients.
    
    A = (-Σ_{k=2}^p (k-1)A_k, Σ_{k=1}^p A_k)
    
    Parameters
    ----------
    A_list : list of np.ndarray
        List of coefficient matrices [A_1, A_2, ..., A_p]
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A_delta : Coefficient for Δy_{t-1}
        A_level : Coefficient for y_{t-1}
        
    References
    ----------
    Chang (2000), p. 908
    """
    p = len(A_list)
    n = A_list[0].shape[0]
    
    # A_delta = -Σ_{k=2}^p (k-1)A_k
    A_delta = np.zeros((n, n))
    for k in range(2, p + 1):
        A_delta -= (k - 1) * A_list[k - 1]
    
    # A_level = Σ_{k=1}^p A_k
    A_level = sum(A_list)
    
    return A_delta, A_level


def moore_penrose_inverse(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute the Moore-Penrose pseudo-inverse.
    
    Used for singular covariance matrices as discussed in the paper.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix
    tol : float, optional
        Tolerance for singular values
        
    Returns
    -------
    np.ndarray
        Pseudo-inverse of A
        
    References
    ----------
    Chang (2000), Appendix, discussion after eq. (A.3)
    """
    return pinv(A, rcond=tol)


def check_stationarity(A_list: List[np.ndarray]) -> bool:
    """
    Check if VAR system satisfies stationarity conditions.
    
    Checks if roots of |I - A(L)L| = 0 are on or outside unit circle.
    
    Parameters
    ----------
    A_list : list of np.ndarray
        List of coefficient matrices [A_1, A_2, ..., A_p]
        
    Returns
    -------
    bool
        True if all roots are on or outside unit circle
        
    References
    ----------
    Chang (2000), Assumption 1(b)
    """
    p = len(A_list)
    n = A_list[0].shape[0]
    
    # Construct companion matrix
    companion = np.zeros((n * p, n * p))
    for i, A in enumerate(A_list):
        companion[:n, i*n:(i+1)*n] = A
    if p > 1:
        companion[n:, :n*(p-1)] = np.eye(n * (p - 1))
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(companion)
    
    # Check if all eigenvalues have modulus >= 1 or <= 1 for invertibility
    # For VAR stability, we want eigenvalues inside unit circle
    # But the paper allows unit roots (I(1), I(2))
    return np.all(np.abs(eigenvalues) >= 1 - 1e-10)


def estimate_rank(matrix: np.ndarray, tol: float = 1e-6) -> int:
    """
    Estimate the rank of a matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix
    tol : float, optional
        Tolerance for rank determination
        
    Returns
    -------
    int
        Estimated rank
    """
    return matrix_rank(matrix, tol=tol)


def orthogonal_complement(A: np.ndarray) -> np.ndarray:
    """
    Compute the orthogonal complement of a matrix.
    
    For a full column rank matrix A (n x r), returns A_⊥ (n x (n-r))
    such that A'_⊥ A = 0.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (n, r) with full column rank
        
    Returns
    -------
    np.ndarray
        Orthogonal complement A_⊥ of shape (n, n-r)
        
    References
    ----------
    Chang (2000), notation before Assumption 1
    """
    n, r = A.shape
    Q, _ = np.linalg.qr(A, mode='complete')
    return Q[:, r:]


def normalize_matrix(gamma: np.ndarray) -> np.ndarray:
    """
    Compute the normalized matrix γ̄ = γ(γ'γ)^{-1}.
    
    Parameters
    ----------
    gamma : np.ndarray
        Input matrix of shape (n, r)
        
    Returns
    -------
    np.ndarray
        Normalized matrix γ̄
        
    References
    ----------
    Chang (2000), notation before Assumption 1
    """
    return gamma @ inv(gamma.T @ gamma)


def compute_sample_variance(X: np.ndarray) -> np.ndarray:
    """
    Compute sample variance matrix.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (T, n)
        
    Returns
    -------
    np.ndarray
        Sample variance matrix Σ = X'X/T
    """
    T = X.shape[0]
    return X.T @ X / T


def compute_sample_covariance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute sample covariance matrix.
    
    Parameters
    ----------
    X : np.ndarray
        First data matrix of shape (T, n1)
    Y : np.ndarray
        Second data matrix of shape (T, n2)
        
    Returns
    -------
    np.ndarray
        Sample covariance matrix Σ_XY = X'Y/T
    """
    T = X.shape[0]
    return X.T @ Y / T


def spectral_density_at_zero(u: np.ndarray, K: int, kernel: str = 'bartlett') -> np.ndarray:
    """
    Estimate the spectral density at frequency zero (long-run variance).
    
    This is a convenience wrapper around kernel_covariance from covariance.py.
    
    Parameters
    ----------
    u : np.ndarray
        Stationary time series of shape (T, n)
    K : int
        Bandwidth parameter
    kernel : str, optional
        Kernel function ('bartlett', 'parzen', 'qs')
        
    Returns
    -------
    np.ndarray
        Estimated long-run variance Ω = Σ_{j=-∞}^{∞} E[u_t u'_{t-j}]
    """
    from .covariance import kernel_covariance
    return kernel_covariance(u, K, kernel)


def is_positive_definite(A: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is positive definite.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix
    tol : float, optional
        Tolerance for eigenvalue check
        
    Returns
    -------
    bool
        True if A is positive definite
    """
    if A.shape[0] != A.shape[1]:
        return False
    try:
        eigenvalues = np.linalg.eigvalsh(A)
        return np.all(eigenvalues > -tol)
    except np.linalg.LinAlgError:
        return False


def make_positive_definite(A: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Adjust a matrix to be positive definite.
    
    Adds a small positive value to the diagonal if needed.
    
    Parameters
    ----------
    A : np.ndarray
        Square symmetric matrix
    epsilon : float, optional
        Small positive value to add to diagonal
        
    Returns
    -------
    np.ndarray
        Positive definite matrix
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def format_matrix_latex(A: np.ndarray, name: str = "A", 
                        precision: int = 4) -> str:
    """
    Format a matrix as LaTeX code.
    
    Parameters
    ----------
    A : np.ndarray
        Matrix to format
    name : str, optional
        Matrix name
    precision : int, optional
        Decimal precision
        
    Returns
    -------
    str
        LaTeX representation
    """
    rows = []
    for row in A:
        row_str = " & ".join([f"{val:.{precision}f}" for val in row])
        rows.append(row_str)
    
    matrix_body = " \\\\\n".join(rows)
    
    return f"""\\begin{{equation}}
{name} = \\begin{{pmatrix}}
{matrix_body}
\\end{{pmatrix}}
\\end{{equation}}"""
