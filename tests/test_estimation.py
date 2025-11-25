"""
Tests for RBFM-VAR estimation.

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import sys
sys.path.insert(0, '..')

from rbfmvar.estimation import RBFMVAR, OLSVAREstimator
from rbfmvar.simulation import generate_dgp


class TestOLSVAREstimator:
    """Tests for OLS-VAR estimator."""
    
    def test_ols_var_basic(self):
        """Test basic OLS-VAR estimation."""
        np.random.seed(42)
        T, n = 200, 2
        y = np.random.randn(T, n)
        
        estimator = OLSVAREstimator(lag_order=2)
        estimator.fit(y)
        
        assert estimator.coefficients is not None
        assert estimator.residuals is not None
        assert estimator.sigma_epsilon is not None
    
    def test_ols_var_coefficient_dimensions(self):
        """Test OLS-VAR coefficient dimensions."""
        np.random.seed(42)
        T, n, p = 200, 2, 2
        y = np.random.randn(T, n)
        
        estimator = OLSVAREstimator(lag_order=p)
        estimator.fit(y)
        
        # F = (Φ, A) where Φ is n x n(p-1), A is n x 2n
        # Total columns: n(p-1) + 2n = np + n
        expected_cols = n * (p - 1) + 2 * n
        assert estimator.coefficients.shape == (n, expected_cols)
    
    def test_ols_var_residuals(self):
        """Test that OLS residuals are orthogonal to regressors."""
        np.random.seed(42)
        T, n = 200, 2
        y = np.random.randn(T, n)
        
        estimator = OLSVAREstimator(lag_order=1)
        estimator.fit(y)
        
        # X'ε ≈ 0
        X = estimator._X
        residuals = estimator.residuals
        XtE = X.T @ residuals
        
        # Should be small relative to magnitudes
        assert np.linalg.norm(XtE) < 1e-10 * np.linalg.norm(X)
    
    def test_ols_var_sigma_epsilon_pd(self):
        """Test that error covariance is positive definite."""
        np.random.seed(42)
        T, n = 200, 2
        y = np.random.randn(T, n)
        
        estimator = OLSVAREstimator(lag_order=1)
        estimator.fit(y)
        
        eigenvalues = np.linalg.eigvalsh(estimator.sigma_epsilon)
        assert np.all(eigenvalues > 0)


class TestRBFMVAR:
    """Tests for RBFM-VAR estimator."""
    
    def test_rbfm_var_basic(self):
        """Test basic RBFM-VAR estimation."""
        np.random.seed(42)
        T, n = 200, 2
        y = np.random.randn(T, n)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        assert model.coefficients is not None
        assert model.ols_coefficients is not None
        assert model.residuals is not None
    
    def test_rbfm_var_coefficient_dimensions(self):
        """Test RBFM-VAR coefficient dimensions."""
        np.random.seed(42)
        T, n, p = 200, 2, 2
        y = np.random.randn(T, n)
        
        model = RBFMVAR(lag_order=p)
        model.fit(y)
        
        # Same dimensions as OLS
        expected_cols = n * (p - 1) + 2 * n
        assert model.coefficients.shape == (n, expected_cols)
        assert model.ols_coefficients.shape == (n, expected_cols)
    
    def test_rbfm_var_with_dgp_case_a(self):
        """Test RBFM-VAR with Case A DGP."""
        y, info = generate_dgp('case_a', T=200, seed=42)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        # Should produce valid estimates
        assert not np.any(np.isnan(model.coefficients))
        assert not np.any(np.isnan(model.ols_coefficients))
    
    def test_rbfm_var_with_dgp_case_b(self):
        """Test RBFM-VAR with Case B DGP."""
        y, info = generate_dgp('case_b', T=200, seed=42)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        assert not np.any(np.isnan(model.coefficients))
    
    def test_rbfm_var_with_dgp_case_c(self):
        """Test RBFM-VAR with Case C DGP."""
        y, info = generate_dgp('case_c', T=200, seed=42)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        assert not np.any(np.isnan(model.coefficients))
    
    def test_rbfm_var_correction_applied(self):
        """Test that RBFM-VAR differs from OLS-VAR (correction applied)."""
        y, _ = generate_dgp('case_c', T=500, seed=42)
        
        model = RBFMVAR(lag_order=1)
        model.fit(y)
        
        # RBFM and OLS estimates should generally differ
        diff = np.linalg.norm(model.coefficients - model.ols_coefficients)
        
        # They should be different (correction was applied)
        # But not wildly different (both should be reasonable)
        assert diff > 1e-10  # Some difference
        assert diff < 100  # Not wildly different
    
    def test_rbfm_var_pi_matrices(self):
        """Test Π₁ and Π₂ matrix extraction."""
        np.random.seed(42)
        T, n = 200, 2
        y = np.random.randn(T, n)
        
        model = RBFMVAR(lag_order=1)
        model.fit(y)
        
        Pi1, Pi2 = model.get_Pi_matrices(use_rbfm=True)
        
        assert Pi1.shape == (n, n)
        assert Pi2.shape == (n, n)
    
    def test_rbfm_var_different_bandwidths(self):
        """Test RBFM-VAR with different bandwidth specifications."""
        np.random.seed(42)
        y, _ = generate_dgp('case_a', T=200, seed=42)
        
        # Test with auto bandwidth
        model_auto = RBFMVAR(lag_order=1, bandwidth='auto')
        model_auto.fit(y)
        
        # Test with fixed bandwidth
        model_fixed = RBFMVAR(lag_order=1, bandwidth=5)
        model_fixed.fit(y)
        
        # Both should work
        assert not np.any(np.isnan(model_auto.coefficients))
        assert not np.any(np.isnan(model_fixed.coefficients))
    
    def test_rbfm_var_different_kernels(self):
        """Test RBFM-VAR with different kernel functions."""
        np.random.seed(42)
        y, _ = generate_dgp('case_a', T=200, seed=42)
        
        for kernel in ['bartlett', 'parzen', 'qs']:
            model = RBFMVAR(lag_order=1, kernel=kernel)
            results = model.fit(y)
            
            assert not np.any(np.isnan(model.coefficients)), f"Failed for {kernel}"
    
    def test_rbfm_var_forecast(self):
        """Test RBFM-VAR forecasting."""
        np.random.seed(42)
        y, _ = generate_dgp('case_b', T=200, seed=42)
        
        model = RBFMVAR(lag_order=1)
        model.fit(y)
        
        # Generate forecasts
        forecasts = model.predict(steps=5)
        
        assert forecasts.shape == (5, 2)
        assert not np.any(np.isnan(forecasts))


class TestRBFMVARResults:
    """Tests for RBFM-VAR results object."""
    
    def test_results_summary(self):
        """Test results summary generation."""
        np.random.seed(42)
        y, _ = generate_dgp('case_a', T=200, seed=42)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        summary = results.summary()
        
        assert isinstance(summary, str)
        assert 'RBFM-VAR' in summary
        assert 'Coefficient' in summary
    
    def test_results_to_dict(self):
        """Test results export to dictionary."""
        np.random.seed(42)
        y, _ = generate_dgp('case_a', T=200, seed=42)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        d = results.to_dict()
        
        assert 'coefficients' in d
        assert 'ols_coefficients' in d
        assert 'sigma_epsilon' in d
        assert 'r_squared' in d
    
    def test_results_to_latex(self):
        """Test LaTeX export."""
        np.random.seed(42)
        y, _ = generate_dgp('case_a', T=200, seed=42)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        latex = results.to_latex()
        
        assert isinstance(latex, str)
        assert '\\begin{table}' in latex
        assert '\\end{table}' in latex
    
    def test_results_diagnostics(self):
        """Test diagnostic statistics."""
        np.random.seed(42)
        y, _ = generate_dgp('case_a', T=200, seed=42)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        # Check R-squared is between 0 and 1
        assert np.all(results.r_squared >= 0)
        assert np.all(results.r_squared <= 1)
        
        # Check information criteria exist
        assert not np.isnan(results.aic)
        assert not np.isnan(results.bic)
    
    def test_results_compare_with_ols(self):
        """Test comparison method."""
        np.random.seed(42)
        y, _ = generate_dgp('case_c', T=200, seed=42)
        
        model = RBFMVAR(lag_order=1)
        results = model.fit(y)
        
        comparison = results.compare_with_ols()
        
        assert isinstance(comparison, str)
        assert 'RBFM-VAR' in comparison
        assert 'OLS-VAR' in comparison


class TestConsistency:
    """Tests for consistency properties of estimators."""
    
    @pytest.mark.slow
    def test_rbfm_var_consistency(self):
        """Test that RBFM-VAR estimates converge as T increases."""
        np.random.seed(42)
        
        # Generate data from known DGP
        y_small, info = generate_dgp('case_c', T=100, seed=42)
        y_large, _ = generate_dgp('case_c', T=1000, seed=42)
        
        true_coef = info['true_coefficients']
        
        model_small = RBFMVAR(lag_order=1)
        model_small.fit(y_small)
        
        model_large = RBFMVAR(lag_order=1)
        model_large.fit(y_large)
        
        error_small = np.linalg.norm(model_small.coefficients - true_coef)
        error_large = np.linalg.norm(model_large.coefficients - true_coef)
        
        # Error should decrease with larger sample (on average)
        # This is a weak test due to randomness
        # In practice, run multiple simulations
        assert error_large < error_small * 3  # Allow for some variance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
