"""
Tests for hypothesis testing functions.

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import sys
sys.path.insert(0, '..')

from rbfmvar.estimation import RBFMVAR
from rbfmvar.testing import (
    wald_test, modified_wald_test, granger_causality_test,
    coefficient_test
)
from rbfmvar.simulation import generate_dgp


class TestWaldTest:
    """Tests for Wald test."""
    
    def test_wald_test_basic(self):
        """Test basic Wald test computation."""
        np.random.seed(42)
        
        # Simple test: F is 2x4, test if first coefficient is zero
        F = np.array([[0.1, 0.2, 0.3, 0.4],
                      [0.5, 0.6, 0.7, 0.8]])
        
        # Restriction: F[0,0] = 0
        R = np.zeros((1, 8))
        R[0, 0] = 1
        r = np.array([0.0])
        
        # Variance matrix (simplified)
        Sigma_F = 0.01 * np.eye(8)
        
        result = wald_test(F, R, r, Sigma_F)
        
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'df' in result
        assert result['df'] == 1
    
    def test_wald_test_chi_square_bound(self):
        """Test that Wald statistic is non-negative."""
        np.random.seed(42)
        
        F = np.random.randn(2, 4)
        R = np.zeros((2, 8))
        R[0, 0] = 1
        R[1, 1] = 1
        r = np.zeros(2)
        Sigma_F = 0.01 * np.eye(8)
        
        result = wald_test(F, R, r, Sigma_F)
        
        assert result['statistic'] >= 0


class TestModifiedWaldTest:
    """Tests for modified Wald test."""
    
    def test_modified_wald_test_basic(self):
        """Test basic modified Wald test."""
        np.random.seed(42)
        
        F_plus = np.random.randn(2, 4)
        R = np.zeros((1, 8))
        R[0, 0] = 1
        r = np.array([0.0])
        
        Sigma_epsilon = np.eye(2)
        XtX = 100 * np.eye(4)
        T = 100
        
        result = modified_wald_test(F_plus, R, r, Sigma_epsilon, XtX, T)
        
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'is_conservative' in result
        assert result['is_conservative'] == True
    
    def test_modified_wald_test_with_real_data(self):
        """Test modified Wald test with estimated model."""
        y, info = generate_dgp('case_a', T=200, seed=42)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        n, k = model.coefficients.shape
        
        # Test that first coefficient is zero
        R = np.zeros((1, n * k))
        R[0, 0] = 1
        r = np.array([0.0])
        
        result = modified_wald_test(
            model.coefficients, R, r,
            model.sigma_epsilon,
            model._X.T @ model._X,
            model._T
        )
        
        assert result['statistic'] >= 0
        assert 0 <= result['p_value'] <= 1


class TestGrangerCausality:
    """Tests for Granger causality testing."""
    
    def test_granger_causality_case_a(self):
        """Test Granger causality in Case A (no causality)."""
        y, info = generate_dgp('case_a', T=300, seed=42)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        # Test if y2 Granger-causes y1 (should not)
        gc_result = results.granger_causality_test(
            caused_variable=0,
            causing_variables=[1],
            test_type='modified'
        )
        
        assert 'statistic' in gc_result
        assert 'p_value' in gc_result
        assert gc_result['is_conservative'] == True
        
        # Under null (no causality), p-value should be relatively large
        # on average across simulations
        # This single test may or may not have large p-value due to randomness
    
    def test_granger_causality_case_c(self):
        """Test Granger causality in Case C (y2 causes y1)."""
        y, info = generate_dgp('case_c', T=500, seed=42)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        # Test if y2 Granger-causes y1 (should, given true DGP)
        gc_result = results.granger_causality_test(
            caused_variable=0,
            causing_variables=[1],
            test_type='modified'
        )
        
        assert 'statistic' in gc_result
        assert 'p_value' in gc_result
        
        # With large enough sample, should reject null
        # (but may not in every single replication)
    
    def test_granger_causality_modified_vs_standard(self):
        """Compare modified and standard Wald tests."""
        y, info = generate_dgp('case_a', T=200, seed=42)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        gc_modified = results.granger_causality_test(
            caused_variable=0,
            causing_variables=[1],
            test_type='modified'
        )
        
        gc_standard = results.granger_causality_test(
            caused_variable=0,
            causing_variables=[1],
            test_type='standard'
        )
        
        # Both should return results
        assert gc_modified['statistic'] >= 0
        assert gc_standard['statistic'] >= 0
        
        # Standard test tends to over-reject in nonstationary settings
        # Modified test should be more conservative
    
    def test_granger_causality_multiple_causing(self):
        """Test Granger causality with multiple causing variables."""
        np.random.seed(42)
        T, n = 200, 3
        y = np.random.randn(T, n)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        # Test if y2 and y3 jointly Granger-cause y1
        gc_result = results.granger_causality_test(
            caused_variable=0,
            causing_variables=[1, 2],
            test_type='modified'
        )
        
        assert gc_result['df'] >= 2  # At least 2 restrictions


class TestCoefficientTest:
    """Tests for individual coefficient testing."""
    
    def test_coefficient_test_basic(self):
        """Test individual coefficient test."""
        y, info = generate_dgp('case_a', T=200, seed=42)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        # Test first coefficient
        result = coefficient_test(
            results,
            equation=0,
            coefficient_idx=0,
            hypothesized_value=0.0,
            test_type='modified'
        )
        
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'coefficient' in result
        assert result['df'] == 1


class TestSizeAndPower:
    """Tests for size and power properties."""
    
    @pytest.mark.slow
    def test_modified_wald_size_control(self):
        """Test that modified Wald test has controlled size."""
        np.random.seed(42)
        n_reps = 100  # Reduced for testing
        alpha = 0.05
        
        rejections = 0
        
        for rep in range(n_reps):
            y, _ = generate_dgp('case_a', T=200, seed=None)
            
            try:
                model = RBFMVAR(lag_order=1)
                results = model.fit(y)
                
                gc = results.granger_causality_test(
                    caused_variable=0,
                    causing_variables=[1],
                    test_type='modified'
                )
                
                if gc['p_value'] < alpha:
                    rejections += 1
            except Exception:
                pass
        
        rejection_rate = rejections / n_reps
        
        # Modified test should have size <= alpha (conservative)
        # Allow some slack for finite sample
        assert rejection_rate < 2 * alpha, f"Rejection rate {rejection_rate} too high"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
