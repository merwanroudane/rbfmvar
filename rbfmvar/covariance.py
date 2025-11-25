"""
Long-run covariance matrix estimation for RBFM-VAR.

This module implements kernel-based estimation of long-run variance and covariance
matrices, which are essential components of the RBFM-VAR correction terms.

The kernel estimation follows the methodology in:
- Phillips, P.C.B. (1995). Fully modified least squares and vector autoregression.
- Chang, Y. & Phillips, P.C.B. (1995). Time series regression with mixtures of 
  integrated processes.

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
from numpy.linalg import inv, pinv
from typing import Union, Tuple, Optional, Callable
import warnings


def bartlett_kernel(x: float) -> float:
    """
    Bartlett (triangular) kernel function.
    
    w(x) = 1 - |x|  for |x| <= 1
         = 0        otherwise
    
    Parameters
    ----------
    x : float
        Input value
        
    Returns
    -------
    float
        Kernel weight
        
    Notes
    -----
    Also known as the Newey-West kernel. This is the most commonly used
    kernel for long-run variance estimation.
    
    References
    ----------
    Chang (2000), Section 3; Phillips (1995)
    """
    if abs(x) <= 1:
        return 1 - abs(x)
    return 0.0


def parzen_kernel(x: float) -> float:
    """
    Parzen kernel function.
    
    w(x) = 1 - 6x² + 6|x|³     for |x| <= 0.5
         = 2(1 - |x|)³          for 0.5 < |x| <= 1
         = 0                     otherwise
    
    Parameters
    ----------
    x : float
        Input value
        
    Returns
    -------
    float
        Kernel weight
        
    Notes
    -----
    The Parzen kernel provides smoother weights than the Bartlett kernel.
    """
    abs_x = abs(x)
    if abs_x <= 0.5:
        return 1 - 6 * x**2 + 6 * abs_x**3
    elif abs_x <= 1:
        return 2 * (1 - abs_x)**3
    return 0.0


def quadratic_spectral_kernel(x: float) -> float:
    """
    Quadratic Spectral (QS) kernel function.
    
    w(x) = (25/(12π²x²)) * (sin(6πx/5)/(6πx/5) - cos(6πx/5))
    
    For x = 0: w(0) = 1
    
    Parameters
    ----------
    x : float
        Input value
        
    Returns
    -------
    float
        Kernel weight
        
    Notes
    -----
    The QS kernel is optimal in a certain asymptotic sense and does not
    truncate to zero at finite lags.
    """
    if abs(x) < 1e-10:
        return 1.0
    
    term = 6 * np.pi * x / 5
    return (25 / (12 * np.pi**2 * x**2)) * (np.sin(term) / term - np.cos(term))


def get_kernel_function(kernel: str) -> Callable[[float], float]:
    """
    Get kernel function by name.
    
    Parameters
    ----------
    kernel : str
        Kernel name: 'bartlett', 'parzen', or 'qs'
        
    Returns
    -------
    Callable
        Kernel function
    """
    kernels = {
        'bartlett': bartlett_kernel,
        'newey_west': bartlett_kernel,
        'parzen': parzen_kernel,
        'qs': quadratic_spectral_kernel,
        'quadratic_spectral': quadratic_spectral_kernel,
    }
    
    kernel_lower = kernel.lower()
    if kernel_lower not in kernels:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose from: {list(kernels.keys())}")
    
    return kernels[kernel_lower]


def optimal_bandwidth(u: np.ndarray, kernel: str = 'bartlett', 
                     method: str = 'andrews') -> int:
    """
    Compute optimal bandwidth for kernel estimation.
    
    Implements automatic bandwidth selection following Andrews (1991) and
    Newey-West (1994).
    
    Parameters
    ----------
    u : np.ndarray
        Stationary time series of shape (T, n)
    kernel : str, optional
        Kernel function name
    method : str, optional
        Method for bandwidth selection: 'andrews' or 'newey_west'
        
    Returns
    -------
    int
        Optimal bandwidth K
        
    Notes
    -----
    For the Bartlett kernel, the optimal rate is K = O(T^(1/3)).
    The paper uses K = O_e(T^k) with k ∈ (1/4, 1/2) for different results.
    
    References
    ----------
    Chang (2000), p. 911; Andrews (1991); Newey & West (1994)
    """
    T = u.shape[0]
    n = u.shape[1] if u.ndim > 1 else 1
    
    if method == 'newey_west':
        # Newey-West automatic bandwidth
        # K = floor(4 * (T/100)^(2/9))
        return max(1, int(4 * (T / 100) ** (2/9)))
    
    elif method == 'andrews':
        # Andrews (1991) automatic bandwidth
        # Based on AR(1) approximation
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        
        # Estimate AR(1) coefficient for each series
        rho_estimates = []
        sigma_estimates = []
        
        for j in range(n):
            u_j = u[:, j]
            if np.std(u_j) < 1e-10:
                rho_estimates.append(0)
                sigma_estimates.append(1)
                continue
                
            # AR(1) regression
            y = u_j[1:]
            x = u_j[:-1]
            rho = np.sum(x * y) / np.sum(x**2) if np.sum(x**2) > 1e-10 else 0
            rho = max(min(rho, 0.97), -0.97)  # Truncate to avoid explosion
            
            resid = y - rho * x
            sigma2 = np.mean(resid**2)
            
            rho_estimates.append(rho)
            sigma_estimates.append(sigma2)
        
        # Compute optimal bandwidth using average parameters
        rho_avg = np.mean(rho_estimates)
        
        if kernel.lower() in ['bartlett', 'newey_west']:
            # For Bartlett kernel
            alpha_hat = 4 * rho_avg**2 / ((1 - rho_avg)**4)
            K = max(1, int(1.1447 * (alpha_hat * T) ** (1/3)))
        elif kernel.lower() == 'parzen':
            # For Parzen kernel
            alpha_hat = 4 * rho_avg**2 / ((1 - rho_avg)**4)
            K = max(1, int(2.6614 * (alpha_hat * T) ** (1/5)))
        else:
            # Default: Bartlett-like
            K = max(1, int(4 * (T / 100) ** (1/3)))
        
        return K
    
    else:
        # Default: simple rule of thumb
        return max(1, int(np.floor(T ** (1/3))))


def sample_autocovariance(u: np.ndarray, lag: int = 0) -> np.ndarray:
    """
    Compute sample autocovariance matrix at given lag.
    
    Γ(k) = (1/T) Σ_{t=k+1}^T u_t u'_{t-k}
    
    Parameters
    ----------
    u : np.ndarray
        Stationary time series of shape (T, n)
    lag : int
        Lag order k
        
    Returns
    -------
    np.ndarray
        Autocovariance matrix Γ(k) of shape (n, n)
        
    Notes
    -----
    For lag k, returns E[u_t u'_{t-k}].
    """
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    
    T, n = u.shape
    
    if abs(lag) >= T:
        return np.zeros((n, n))
    
    if lag >= 0:
        result = u[lag:].T @ u[:T-lag] / T
    else:
        result = u[:T+lag].T @ u[-lag:] / T
    
    return result


def kernel_covariance(u: np.ndarray, K: Optional[int] = None, 
                      kernel: str = 'bartlett') -> np.ndarray:
    """
    Estimate long-run covariance matrix using kernel estimation.
    
    Ω̂ = Σ_{j=-K}^K w(j/K) Γ̂(j)
    
    where w(.) is the kernel function and Γ̂(j) is the sample autocovariance.
    
    Parameters
    ----------
    u : np.ndarray
        Stationary time series of shape (T, n)
    K : int, optional
        Bandwidth parameter. If None, uses automatic selection.
    kernel : str, optional
        Kernel function: 'bartlett', 'parzen', or 'qs'
        
    Returns
    -------
    np.ndarray
        Estimated long-run covariance matrix Ω of shape (n, n)
        
    Notes
    -----
    This is the symmetric long-run covariance Ω = Σ_{j=-∞}^∞ E[u_j u'_0].
    
    References
    ----------
    Chang (2000), p. 910-911; Phillips (1995)
    """
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    
    T, n = u.shape
    
    # Automatic bandwidth selection if not provided
    if K is None:
        K = optimal_bandwidth(u, kernel)
    
    # Get kernel function
    kernel_func = get_kernel_function(kernel)
    
    # Initialize with lag-0 covariance (weight = 1)
    Omega = sample_autocovariance(u, 0)
    
    # Add weighted autocovariances for j = 1, ..., K
    for j in range(1, min(K + 1, T)):
        weight = kernel_func(j / K)
        if weight > 0:
            Gamma_j = sample_autocovariance(u, j)
            # Ω = Σ w(j/K)[Γ(j) + Γ(j)'] for j > 0
            Omega = Omega + weight * (Gamma_j + Gamma_j.T)
    
    return Omega


def one_sided_kernel_covariance(u: np.ndarray, K: Optional[int] = None,
                                kernel: str = 'bartlett') -> np.ndarray:
    """
    Estimate one-sided long-run covariance matrix.
    
    Δ̂ = Σ_{j=0}^K w(j/K) Γ̂(j)
    
    Parameters
    ----------
    u : np.ndarray
        Stationary time series of shape (T, n)
    K : int, optional
        Bandwidth parameter
    kernel : str, optional
        Kernel function
        
    Returns
    -------
    np.ndarray
        Estimated one-sided covariance matrix Δ of shape (n, n)
        
    Notes
    -----
    This is the one-sided covariance Δ = Σ_{j=0}^∞ E[u_j u'_0].
    Related to symmetric covariance by: Ω = Δ + Δ' - Γ(0)
    
    References
    ----------
    Chang (2000), eq. (13), p. 910
    """
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    
    T, n = u.shape
    
    if K is None:
        K = optimal_bandwidth(u, kernel)
    
    kernel_func = get_kernel_function(kernel)
    
    # Initialize with lag-0 covariance
    Delta = sample_autocovariance(u, 0)
    
    # Add weighted positive lag autocovariances
    for j in range(1, min(K + 1, T)):
        weight = kernel_func(j / K)
        if weight > 0:
            Gamma_j = sample_autocovariance(u, j)
            Delta = Delta + weight * Gamma_j
    
    return Delta


def cross_kernel_covariance(u: np.ndarray, v: np.ndarray, 
                            K: Optional[int] = None,
                            kernel: str = 'bartlett') -> np.ndarray:
    """
    Estimate long-run cross-covariance matrix between two series.
    
    Ω̂_uv = Σ_{j=-K}^K w(j/K) Γ̂_uv(j)
    
    Parameters
    ----------
    u : np.ndarray
        First stationary time series of shape (T, n1)
    v : np.ndarray
        Second stationary time series of shape (T, n2)
    K : int, optional
        Bandwidth parameter
    kernel : str, optional
        Kernel function
        
    Returns
    -------
    np.ndarray
        Estimated cross-covariance matrix Ω_uv of shape (n1, n2)
        
    References
    ----------
    Chang (2000), eq. (13), Ω̂_{ε̂v̂}
    """
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    
    T = min(u.shape[0], v.shape[0])
    n1, n2 = u.shape[1], v.shape[1]
    
    # Truncate to same length
    u = u[:T]
    v = v[:T]
    
    if K is None:
        # Use combined series for bandwidth
        combined = np.hstack([u, v])
        K = optimal_bandwidth(combined, kernel)
    
    kernel_func = get_kernel_function(kernel)
    
    # Lag-0 cross-covariance
    Omega = u.T @ v / T
    
    # Add weighted cross-autocovariances
    for j in range(1, min(K + 1, T)):
        weight = kernel_func(j / K)
        if weight > 0:
            # Cross-covariance at lag j and -j
            Gamma_j = u[j:].T @ v[:T-j] / T     # Γ_uv(j)
            Gamma_mj = u[:T-j].T @ v[j:] / T    # Γ_uv(-j)
            Omega = Omega + weight * (Gamma_j + Gamma_mj)
    
    return Omega


def one_sided_cross_covariance(u: np.ndarray, v: np.ndarray,
                               K: Optional[int] = None,
                               kernel: str = 'bartlett') -> np.ndarray:
    """
    Estimate one-sided cross-covariance Δ̂_uv = Σ_{j=0}^K w(j/K) Γ̂_uv(j).
    
    Parameters
    ----------
    u : np.ndarray
        First time series of shape (T, n1)
    v : np.ndarray
        Second time series of shape (T, n2)
    K : int, optional
        Bandwidth parameter
    kernel : str, optional
        Kernel function
        
    Returns
    -------
    np.ndarray
        One-sided cross-covariance matrix of shape (n1, n2)
        
    References
    ----------
    Chang (2000), eq. (13), Δ̂_{v̂Δw}
    """
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    
    T = min(u.shape[0], v.shape[0])
    u = u[:T]
    v = v[:T]
    
    if K is None:
        combined = np.hstack([u, v])
        K = optimal_bandwidth(combined, kernel)
    
    kernel_func = get_kernel_function(kernel)
    
    # Lag-0
    Delta = u.T @ v / T
    
    # Positive lags only
    for j in range(1, min(K + 1, T)):
        weight = kernel_func(j / K)
        if weight > 0:
            Gamma_j = u[j:].T @ v[:T-j] / T
            Delta = Delta + weight * Gamma_j
    
    return Delta


def compute_correction_covariances(epsilon_hat: np.ndarray, 
                                   v_hat: np.ndarray,
                                   delta_w: np.ndarray,
                                   K: Optional[int] = None,
                                   kernel: str = 'bartlett'
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the covariance matrices needed for RBFM-VAR correction terms.
    
    Computes:
    - Ω̂_{ε̂v̂}: Long-run cross-covariance of residuals and v-process
    - Ω̂_{v̂v̂}: Long-run variance of v-process  
    - Δ̂_{v̂Δw}: One-sided covariance of v-process and Δw
    
    Parameters
    ----------
    epsilon_hat : np.ndarray
        OLS residuals of shape (T, n)
    v_hat : np.ndarray
        v-process of shape (T, 2n) defined in eq. (11)
    delta_w : np.ndarray
        First difference of w_t of shape (T, 2n)
    K : int, optional
        Bandwidth parameter
    kernel : str, optional
        Kernel function
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Omega_ev: Cross-covariance Ω̂_{ε̂v̂}
        Omega_vv: Variance Ω̂_{v̂v̂}
        Delta_vdw: One-sided covariance Δ̂_{v̂Δw}
        
    References
    ----------
    Chang (2000), eq. (13)
    """
    # Determine bandwidth
    if K is None:
        combined = np.hstack([epsilon_hat, v_hat])
        K = optimal_bandwidth(combined, kernel)
    
    # Compute covariances
    Omega_ev = cross_kernel_covariance(epsilon_hat, v_hat, K, kernel)
    Omega_vv = kernel_covariance(v_hat, K, kernel)
    Delta_vdw = one_sided_cross_covariance(v_hat, delta_w, K, kernel)
    
    return Omega_ev, Omega_vv, Delta_vdw


def estimate_contemporaneous_covariance(u: np.ndarray) -> np.ndarray:
    """
    Estimate the contemporaneous covariance matrix Σ = E[u_t u'_t].
    
    Parameters
    ----------
    u : np.ndarray
        Time series of shape (T, n)
        
    Returns
    -------
    np.ndarray
        Sample covariance matrix of shape (n, n)
    """
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    
    T = u.shape[0]
    return u.T @ u / T


def regularize_covariance(Omega: np.ndarray, 
                          method: str = 'eigenvalue',
                          epsilon: float = 1e-8) -> np.ndarray:
    """
    Regularize a potentially singular covariance matrix.
    
    Parameters
    ----------
    Omega : np.ndarray
        Covariance matrix
    method : str
        Regularization method: 'eigenvalue', 'ridge', or 'pinv'
    epsilon : float
        Regularization parameter
        
    Returns
    -------
    np.ndarray
        Regularized covariance matrix
        
    Notes
    -----
    The long-run variance matrix may be singular in certain directions
    as discussed in Remark (b) following Theorem 1 in Chang (2000).
    """
    if method == 'eigenvalue':
        # Ensure positive semi-definiteness by adjusting eigenvalues
        eigvals, eigvecs = np.linalg.eigh(Omega)
        eigvals = np.maximum(eigvals, epsilon)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    elif method == 'ridge':
        # Add small ridge to diagonal
        n = Omega.shape[0]
        return Omega + epsilon * np.eye(n)
    
    elif method == 'pinv':
        # Return original and use pseudo-inverse for inversion
        return Omega
    
    else:
        raise ValueError(f"Unknown regularization method: {method}")


def safe_inverse(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Safely invert a matrix, using pseudo-inverse if singular.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix to invert
    tol : float
        Tolerance for singularity detection
        
    Returns
    -------
    np.ndarray
        (Pseudo-)inverse of A
        
    Notes
    -----
    Uses Moore-Penrose inverse when the matrix is singular or near-singular,
    as required for the singular covariance matrices discussed in Chang (2000).
    """
    try:
        # Check condition number
        cond = np.linalg.cond(A)
        if cond > 1 / tol:
            # Matrix is ill-conditioned, use pseudo-inverse
            return pinv(A, rcond=tol)
        else:
            return inv(A)
    except np.linalg.LinAlgError:
        # Singular matrix, use pseudo-inverse
        return pinv(A, rcond=tol)
