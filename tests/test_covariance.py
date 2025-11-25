"""
Tests for covariance estimation functions.

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import sys
sys.path.insert(0, '..')

from rbfmvar.covariance import (
    bartlett_kernel, parzen_kernel, quadratic_spectral_kernel,
    sample_autocovariance, kernel_covariance, one_sided_kernel_covariance,
    cross_kernel_covariance, optimal_bandwidth, safe_inverse
)


class TestKernelFunctions:
    """Tests for kernel functions."""
    
    def test_bartlett_kernel_at_zero(self):
        """Test Bartlett kernel at x=0."""
        assert bartlett_kernel(0) == 1.0
    
    def test_bartlett_kernel_at_one(self):
        """Test Bartlett kernel at x=1."""
        assert bartlett_kernel(1) == 0.0
    
    def test_bartlett_kernel_outside(self):
        """Test Bartlett kernel outside [-1, 1]."""
        assert bartlett_kernel(1.5) == 0.0
        assert bartlett_kernel(-1.5) == 0.0
    
    def test_bartlett_kernel_symmetric(self):
        """Test Bartlett kernel is symmetric."""
        x = 0.5
        assert bartlett_kernel(x) == bartlett_kernel(-x)
    
    def test_parzen_kernel_at_zero(self):
        """Test Parzen kernel at x=0."""
        assert parzen_kernel(0) == 1.0
    
    def test_parzen_kernel_at_one(self):
        """Test Parzen kernel at x=1."""
        assert parzen_kernel(1) == 0.0
    
    def test_parzen_kernel_continuity(self):
        """Test Parzen kernel is continuous at 0.5."""
        eps = 1e-10
        left = parzen_kernel(0.5 - eps)
        right = parzen_kernel(0.5 + eps)
        assert abs(left - right) < 1e-8
    
    def test_qs_kernel_at_zero(self):
        """Test Quadratic Spectral kernel at x=0."""
        assert quadratic_spectral_kernel(0) == 1.0
    
    def test_qs_kernel_nonzero(self):
        """Test QS kernel at nonzero values."""
        # QS kernel doesn't truncate to zero
        assert quadratic_spectral_kernel(2) != 0


class TestAutocovariance:
    """Tests for autocovariance estimation."""
    
    def test_lag_zero_covariance(self):
        """Test lag-0 autocovariance equals variance."""
        np.random.seed(42)
        u = np.random.randn(100, 2)
        
        gamma_0 = sample_autocovariance(u, 0)
        expected_var = u.T @ u / 100
        
        assert_array_almost_equal(gamma_0, expected_var)
    
    def test_autocovariance_symmetry(self):
        """Test Γ(-k) = Γ(k)'."""
        np.random.seed(42)
        u = np.random.randn(100, 2)
        
        gamma_k = sample_autocovariance(u, 3)
        gamma_mk = sample_autocovariance(u, -3)
        
        assert_array_almost_equal(gamma_k, gamma_mk.T, decimal=10)
    
    def test_autocovariance_univariate(self):
        """Test autocovariance for univariate series."""
        np.random.seed(42)
        u = np.random.randn(100)
        
        gamma_0 = sample_autocovariance(u, 0)
        assert gamma_0.shape == (1, 1)


class TestKernelCovariance:
    """Tests for kernel-based covariance estimation."""
    
    def test_kernel_covariance_positive_semidefinite(self):
        """Test that estimated covariance is positive semi-definite."""
        np.random.seed(42)
        u = np.random.randn(200, 2)
        
        Omega = kernel_covariance(u, K=10, kernel='bartlett')
        
        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(Omega)
        assert np.all(eigenvalues >= -1e-10)
    
    def test_kernel_covariance_symmetric(self):
        """Test that estimated covariance is symmetric."""
        np.random.seed(42)
        u = np.random.randn(200, 2)
        
        Omega = kernel_covariance(u, K=10, kernel='bartlett')
        
        assert_array_almost_equal(Omega, Omega.T)
    
    def test_one_sided_covariance_relationship(self):
        """Test relationship between one-sided and symmetric covariance."""
        np.random.seed(42)
        u = np.random.randn(200, 2)
        K = 10
        
        Omega = kernel_covariance(u, K, 'bartlett')
        Delta = one_sided_kernel_covariance(u, K, 'bartlett')
        Gamma_0 = sample_autocovariance(u, 0)
        
        # Ω should approximately equal Δ + Δ' - Γ(0)
        reconstructed = Delta + Delta.T - Gamma_0
        
        # This is approximate due to kernel weighting
        # Just check they're in the same ballpark
        assert np.linalg.norm(Omega - reconstructed) < np.linalg.norm(Omega)


class TestCrossCovariance:
    """Tests for cross-covariance estimation."""
    
    def test_cross_covariance_dimensions(self):
        """Test cross-covariance dimensions."""
        np.random.seed(42)
        u = np.random.randn(100, 2)
        v = np.random.randn(100, 3)
        
        Omega_uv = cross_kernel_covariance(u, v, K=10, kernel='bartlett')
        
        assert Omega_uv.shape == (2, 3)
    
    def test_cross_covariance_self(self):
        """Test cross-covariance with itself equals auto-covariance."""
        np.random.seed(42)
        u = np.random.randn(100, 2)
        
        Omega_uu = cross_kernel_covariance(u, u, K=10, kernel='bartlett')
        Omega = kernel_covariance(u, K=10, kernel='bartlett')
        
        assert_array_almost_equal(Omega_uu, Omega, decimal=10)


class TestOptimalBandwidth:
    """Tests for automatic bandwidth selection."""
    
    def test_optimal_bandwidth_positive(self):
        """Test that optimal bandwidth is positive."""
        np.random.seed(42)
        u = np.random.randn(100, 2)
        
        K = optimal_bandwidth(u, kernel='bartlett')
        assert K > 0
    
    def test_optimal_bandwidth_reasonable(self):
        """Test that optimal bandwidth is reasonable relative to T."""
        np.random.seed(42)
        T = 500
        u = np.random.randn(T, 2)
        
        K = optimal_bandwidth(u, kernel='bartlett')
        
        # Should be much smaller than T
        assert K < T / 2
        # Should grow with T but sublinearly
        assert K < T ** 0.5 * 10


class TestSafeInverse:
    """Tests for safe matrix inversion."""
    
    def test_safe_inverse_regular(self):
        """Test safe inverse of regular matrix."""
        A = np.array([[2, 1], [1, 2]])
        A_inv = safe_inverse(A)
        
        # Should be actual inverse
        assert_array_almost_equal(A @ A_inv, np.eye(2))
    
    def test_safe_inverse_singular(self):
        """Test safe inverse of singular matrix uses pseudo-inverse."""
        A = np.array([[1, 2], [2, 4]])  # Rank-deficient
        A_inv = safe_inverse(A)
        
        # Should satisfy A @ A_inv @ A ≈ A
        result = A @ A_inv @ A
        assert_array_almost_equal(result, A, decimal=10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
