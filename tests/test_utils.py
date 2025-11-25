"""
Tests for utility functions.

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import sys
sys.path.insert(0, '..')

from rbfmvar.utils import (
    vec, unvec, kronecker_product, difference, lag_matrix,
    create_var_matrices, create_ecm_matrices, moore_penrose_inverse,
    orthogonal_complement, normalize_matrix, is_positive_definite
)


class TestVecOperations:
    """Tests for vec and unvec operations."""
    
    def test_vec_basic(self):
        """Test basic vec operation."""
        A = np.array([[1, 2], [3, 4]])
        expected = np.array([1, 3, 2, 4])  # Column-major (Fortran) order
        result = vec(A)
        assert_array_equal(result, expected)
    
    def test_unvec_basic(self):
        """Test basic unvec operation."""
        v = np.array([1, 3, 2, 4])
        expected = np.array([[1, 2], [3, 4]])
        result = unvec(v, 2, 2)
        assert_array_equal(result, expected)
    
    def test_vec_unvec_roundtrip(self):
        """Test that vec and unvec are inverses."""
        A = np.random.randn(3, 4)
        result = unvec(vec(A), 3, 4)
        assert_array_almost_equal(result, A)


class TestKronecker:
    """Tests for Kronecker product."""
    
    def test_kronecker_basic(self):
        """Test basic Kronecker product."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[0, 5], [6, 7]])
        result = kronecker_product(A, B)
        expected = np.kron(A, B)
        assert_array_equal(result, expected)
    
    def test_kronecker_dimensions(self):
        """Test Kronecker product dimensions."""
        A = np.random.randn(2, 3)
        B = np.random.randn(4, 5)
        result = kronecker_product(A, B)
        assert result.shape == (8, 15)


class TestDifference:
    """Tests for differencing function."""
    
    def test_first_difference(self):
        """Test first difference."""
        y = np.array([1, 3, 6, 10, 15])
        expected = np.array([2, 3, 4, 5])
        result = difference(y, d=1)
        assert_array_equal(result, expected)
    
    def test_second_difference(self):
        """Test second difference."""
        y = np.array([1, 3, 6, 10, 15])
        expected = np.array([1, 1, 1])
        result = difference(y, d=2)
        assert_array_equal(result, expected)
    
    def test_multivariate_difference(self):
        """Test differencing multivariate series."""
        y = np.array([[1, 10], [2, 20], [4, 30], [7, 40]])
        result = difference(y, d=1)
        expected = np.array([[1, 10], [2, 10], [3, 10]])
        assert_array_equal(result, expected)


class TestLagMatrix:
    """Tests for lag matrix creation."""
    
    def test_lag_matrix_basic(self):
        """Test basic lag matrix creation."""
        y = np.array([[1], [2], [3], [4], [5]])
        result = lag_matrix(y, 2)
        # With 2 lags, we lose 2 observations
        assert result.shape[0] == 3
        assert result.shape[1] == 2  # 2 lags * 1 variable
    
    def test_lag_matrix_multivariate(self):
        """Test lag matrix for multivariate series."""
        y = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
        result = lag_matrix(y, 2)
        assert result.shape == (3, 4)  # 2 lags * 2 variables


class TestECMMatrices:
    """Tests for ECM matrix creation."""
    
    def test_ecm_matrices_dimensions(self):
        """Test ECM matrix dimensions."""
        np.random.seed(42)
        T, n, p = 100, 2, 2
        y = np.random.randn(T, n)
        
        Y, Z, W = create_ecm_matrices(y, p)
        
        # Check dimensions
        T_eff = Y.shape[0]
        assert Y.shape == (T_eff, n)
        assert W.shape == (T_eff, 2 * n)  # (Δy_{t-1}, y_{t-1})
    
    def test_ecm_matrices_lag1(self):
        """Test ECM matrices with lag order 1."""
        np.random.seed(42)
        T, n = 50, 2
        y = np.random.randn(T, n)
        
        Y, Z, W = create_ecm_matrices(y, 1)
        
        # With p=1, Z should be empty (no lagged Δ² terms)
        assert W.shape[1] == 2 * n


class TestMatrixUtilities:
    """Tests for matrix utility functions."""
    
    def test_moore_penrose_inverse(self):
        """Test Moore-Penrose inverse."""
        A = np.array([[1, 2], [2, 4]])  # Rank-deficient
        A_pinv = moore_penrose_inverse(A)
        
        # Should satisfy A @ A_pinv @ A ≈ A
        result = A @ A_pinv @ A
        assert_array_almost_equal(result, A, decimal=10)
    
    def test_orthogonal_complement(self):
        """Test orthogonal complement."""
        A = np.array([[1, 0], [0, 1], [0, 0]])  # 3x2 matrix
        A_perp = orthogonal_complement(A)
        
        # A'_⊥ @ A should be zero
        product = A_perp.T @ A
        assert_array_almost_equal(product, np.zeros((1, 2)), decimal=10)
    
    def test_is_positive_definite(self):
        """Test positive definiteness check."""
        # Positive definite matrix
        A_pd = np.array([[2, 1], [1, 2]])
        assert is_positive_definite(A_pd)
        
        # Not positive definite
        A_not_pd = np.array([[1, 2], [2, 1]])
        assert not is_positive_definite(A_not_pd)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
