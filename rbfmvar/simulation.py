"""
Monte Carlo Simulation Module for RBFM-VAR.

This module implements the Monte Carlo simulation study from Section 5 of
Chang (2000), allowing comparison of OLS-VAR and RBFM-VAR estimators.

The simulation considers three cases:
- Case A: (ρ₁, ρ₂) = (1, 0) - Both y₁ and y₂ are I(2) with no cointegration
- Case B: (ρ₁, ρ₂) = (0.5, 0) - y₁ is I(1), y₂ is I(2), no Granger causality
- Case C: (ρ₁, ρ₂) = (-0.3, -0.15) - y₁ is I(1), y₂ is I(2), y₂ Granger-causes y₁

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
"""

import numpy as np
from numpy.linalg import inv
from scipy import stats
from typing import Optional, Union, Tuple, List, Dict, Any
from tabulate import tabulate
import warnings


class SimulationResults:
    """
    Container for Monte Carlo simulation results.
    
    Stores and summarizes simulation outcomes including biases, 
    standard deviations, and test size/power results.
    
    Attributes
    ----------
    n_reps : int
        Number of replications
    sample_size : int
        Sample size used
    dgp : str
        Data generating process name
    ols_estimates : np.ndarray
        OLS-VAR estimates from all replications
    rbfm_estimates : np.ndarray
        RBFM-VAR estimates from all replications
    ols_wald : np.ndarray
        OLS Wald test statistics
    rbfm_wald : np.ndarray
        Modified Wald test statistics
    true_coefficients : np.ndarray
        True coefficient values
    """
    
    def __init__(self, 
                 n_reps: int,
                 sample_size: int,
                 dgp: str,
                 true_coefficients: np.ndarray):
        self.n_reps = n_reps
        self.sample_size = sample_size
        self.dgp = dgp
        self.true_coefficients = true_coefficients
        
        # Storage for results
        self.ols_estimates = None
        self.rbfm_estimates = None
        self.ols_wald = None
        self.rbfm_wald = None
        self._is_complete = False
    
    def compute_bias(self, estimator: str = 'rbfm') -> np.ndarray:
        """
        Compute bias of estimator.
        
        Bias = E[θ̂] - θ₀
        
        Parameters
        ----------
        estimator : str
            'rbfm' or 'ols'
            
        Returns
        -------
        np.ndarray
            Bias for each coefficient
        """
        estimates = self.rbfm_estimates if estimator == 'rbfm' else self.ols_estimates
        mean_estimate = np.mean(estimates, axis=0)
        return mean_estimate - self.true_coefficients
    
    def compute_std(self, estimator: str = 'rbfm') -> np.ndarray:
        """
        Compute standard deviation of estimator.
        
        Parameters
        ----------
        estimator : str
            'rbfm' or 'ols'
            
        Returns
        -------
        np.ndarray
            Standard deviation for each coefficient
        """
        estimates = self.rbfm_estimates if estimator == 'rbfm' else self.ols_estimates
        return np.std(estimates, axis=0, ddof=1)
    
    def compute_rmse(self, estimator: str = 'rbfm') -> np.ndarray:
        """
        Compute root mean squared error.
        
        RMSE = sqrt(E[(θ̂ - θ₀)²])
        
        Parameters
        ----------
        estimator : str
            'rbfm' or 'ols'
            
        Returns
        -------
        np.ndarray
            RMSE for each coefficient
        """
        estimates = self.rbfm_estimates if estimator == 'rbfm' else self.ols_estimates
        errors = estimates - self.true_coefficients
        return np.sqrt(np.mean(errors**2, axis=0))
    
    def compute_rejection_rate(self, 
                               estimator: str = 'rbfm',
                               alpha: float = 0.05) -> float:
        """
        Compute rejection rate of Wald test.
        
        Parameters
        ----------
        estimator : str
            'rbfm' or 'ols'
        alpha : float
            Significance level
            
        Returns
        -------
        float
            Rejection rate (proportion of tests rejecting H₀)
        """
        wald_stats = self.rbfm_wald if estimator == 'rbfm' else self.ols_wald
        
        if wald_stats is None:
            return np.nan
        
        # Get critical value from chi-square distribution
        # Degrees of freedom is 2 for the causality test (Π₁ and Π₂ coefficients)
        df = 2  # As in the paper's simulation
        critical_value = stats.chi2.ppf(1 - alpha, df)
        
        return np.mean(wald_stats > critical_value)
    
    def summary(self) -> str:
        """
        Generate summary of simulation results.
        
        Returns
        -------
        str
            Formatted summary
        """
        lines = []
        lines.append("=" * 80)
        lines.append("Monte Carlo Simulation Results".center(80))
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"DGP:              {self.dgp}")
        lines.append(f"Sample Size (T):  {self.sample_size}")
        lines.append(f"Replications:     {self.n_reps}")
        lines.append("")
        
        # Bias comparison (scaled by sqrt(T) as in Table 1)
        lines.append("Bias (scaled by √T):")
        lines.append("-" * 60)
        
        T = self.sample_size
        sqrt_T = np.sqrt(T)
        
        ols_bias = self.compute_bias('ols') * sqrt_T
        rbfm_bias = self.compute_bias('rbfm') * sqrt_T
        
        headers = ["Coefficient", "OLS-VAR", "RBFM-VAR"]
        rows = []
        
        flat_ols = ols_bias.flatten()
        flat_rbfm = rbfm_bias.flatten()
        flat_true = self.true_coefficients.flatten()
        
        coef_names = self._get_coefficient_names()
        
        for i, name in enumerate(coef_names):
            rows.append([name, f"{flat_ols[i]:.5f}", f"{flat_rbfm[i]:.5f}"])
        
        lines.append(tabulate(rows, headers=headers, tablefmt="simple"))
        lines.append("")
        
        # Standard deviation comparison (scaled by sqrt(T))
        lines.append("Standard Deviation (scaled by √T):")
        lines.append("-" * 60)
        
        ols_std = self.compute_std('ols') * sqrt_T
        rbfm_std = self.compute_std('rbfm') * sqrt_T
        
        rows = []
        flat_ols_std = ols_std.flatten()
        flat_rbfm_std = rbfm_std.flatten()
        
        for i, name in enumerate(coef_names):
            rows.append([name, f"{flat_ols_std[i]:.5f}", f"{flat_rbfm_std[i]:.5f}"])
        
        lines.append(tabulate(rows, headers=headers, tablefmt="simple"))
        lines.append("")
        
        # Test size/power
        if self.ols_wald is not None and self.rbfm_wald is not None:
            lines.append("Test Rejection Rates:")
            lines.append("-" * 60)
            
            headers = ["Test", "1% level", "5% level", "10% level"]
            rows = [
                ["OLS Wald (W_F)", 
                 f"{self.compute_rejection_rate('ols', 0.01):.3f}",
                 f"{self.compute_rejection_rate('ols', 0.05):.3f}",
                 f"{self.compute_rejection_rate('ols', 0.10):.3f}"],
                ["Modified Wald (W_F^+)",
                 f"{self.compute_rejection_rate('rbfm', 0.01):.3f}",
                 f"{self.compute_rejection_rate('rbfm', 0.05):.3f}",
                 f"{self.compute_rejection_rate('rbfm', 0.10):.3f}"],
            ]
            
            lines.append(tabulate(rows, headers=headers, tablefmt="simple"))
            lines.append("")
            
            if 'case_a' in self.dgp.lower() or 'case_b' in self.dgp.lower():
                lines.append("Note: These are SIZE results (H₀ is true)")
            else:
                lines.append("Note: These are POWER results (H₀ is false)")
        
        lines.append("")
        lines.append("Reference: Chang (2000), Econometric Theory 16(6), 905-926")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _get_coefficient_names(self) -> List[str]:
        """Generate coefficient names."""
        n = self.true_coefficients.shape[0]
        k = self.true_coefficients.shape[1]
        
        # For bivariate case with lag 1
        if n == 2 and k == 4:
            return [
                "₁π₁₁", "₁π₁₂", "₁π₂₁", "₁π₂₂",
                "₂π₁₁", "₂π₁₂", "₂π₂₁", "₂π₂₂"
            ]
        else:
            names = []
            for i in range(n):
                for j in range(k):
                    names.append(f"F[{i+1},{j+1}]")
            return names
    
    def to_latex_table(self) -> str:
        """
        Export results as LaTeX table (matching Table 1 in the paper).
        
        Returns
        -------
        str
            LaTeX table code
        """
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Finite Sample Biases and Standard Deviations}")
        lines.append("\\label{tab:mc_results}")
        lines.append("\\begin{tabular}{lcccccccc}")
        lines.append("\\toprule")
        lines.append(" & ${}_{1}\\pi_{11}$ & ${}_{1}\\pi_{12}$ & ${}_{1}\\pi_{21}$ & ${}_{1}\\pi_{22}$ & ${}_{2}\\pi_{11}$ & ${}_{2}\\pi_{12}$ & ${}_{2}\\pi_{21}$ & ${}_{2}\\pi_{22}$ \\\\")
        lines.append("\\midrule")
        
        sqrt_T = np.sqrt(self.sample_size)
        
        # OLS-VAR estimates
        lines.append("\\multicolumn{9}{c}{OLS-VAR estimators} \\\\")
        
        ols_bias = (self.compute_bias('ols') * sqrt_T).flatten()
        bias_str = "Bias"
        for b in ols_bias:
            bias_str += f" & {b:.5f}"
        bias_str += " \\\\"
        lines.append(bias_str)
        
        ols_std = (self.compute_std('ols') * sqrt_T).flatten()
        std_str = "s.d."
        for s in ols_std:
            std_str += f" & {s:.5f}"
        std_str += " \\\\"
        lines.append(std_str)
        
        lines.append("\\midrule")
        
        # RBFM-VAR estimates
        lines.append("\\multicolumn{9}{c}{RBFM-VAR estimators} \\\\")
        
        rbfm_bias = (self.compute_bias('rbfm') * sqrt_T).flatten()
        bias_str = "Bias"
        for b in rbfm_bias:
            bias_str += f" & {b:.5f}"
        bias_str += " \\\\"
        lines.append(bias_str)
        
        rbfm_std = (self.compute_std('rbfm') * sqrt_T).flatten()
        std_str = "s.d."
        for s in rbfm_std:
            std_str += f" & {s:.5f}"
        std_str += " \\\\"
        lines.append(std_str)
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\begin{tablenotes}")
        lines.append(f"\\small \\item Note: Sample size $T = {self.sample_size}$. "
                    f"Values are scaled by $\\sqrt{{T}}$. DGP: {self.dgp}.")
        lines.append("\\end{tablenotes}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)


def generate_dgp(case: str, 
                 T: int, 
                 Sigma: Optional[np.ndarray] = None,
                 seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate data according to the DGPs in Chang (2000), Section 5.
    
    The DGP is:
        Δy₁ₜ = ρ₁Δy₁,ₜ₋₁ + ρ₂(y₁,ₜ₋₁ - Δy₂,ₜ₋₁) + ε₁ₜ
        Δ²y₂ₜ = ε₂ₜ
    
    Parameters
    ----------
    case : str
        DGP case: 'case_a', 'case_b', or 'case_c'
    T : int
        Sample size
    Sigma : np.ndarray, optional
        Error covariance matrix. Default is [[1, 0.5], [0.5, 1]].
    seed : int, optional
        Random seed
        
    Returns
    -------
    Tuple[np.ndarray, Dict]
        y : Generated data (T x 2)
        info : Dictionary with DGP parameters and true coefficients
        
    References
    ----------
    Chang (2000), Section 5, eq. (24)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if Sigma is None:
        Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    
    # Set parameters based on case
    case_lower = case.lower().replace(' ', '_')
    
    if 'case_a' in case_lower or case_lower == 'a':
        rho1, rho2 = 1.0, 0.0
        case_name = "Case A (ρ₁=1, ρ₂=0): Both I(2), no cointegration"
    elif 'case_b' in case_lower or case_lower == 'b':
        rho1, rho2 = 0.5, 0.0
        case_name = "Case B (ρ₁=0.5, ρ₂=0): y₁ I(1), y₂ I(2), no causality"
    elif 'case_c' in case_lower or case_lower == 'c':
        rho1, rho2 = -0.3, -0.15
        case_name = "Case C (ρ₁=-0.3, ρ₂=-0.15): y₁ I(1), y₂ I(2), y₂ causes y₁"
    else:
        raise ValueError(f"Unknown case: {case}. Use 'case_a', 'case_b', or 'case_c'.")
    
    # Generate innovations
    epsilon = np.random.multivariate_normal(np.zeros(2), Sigma, size=T + 10)
    
    # Initialize
    y = np.zeros((T + 10, 2))
    delta_y = np.zeros((T + 10, 2))
    delta2_y = np.zeros((T + 10, 2))
    
    # Generate y₂ as I(2): Δ²y₂ₜ = ε₂ₜ
    delta2_y[:, 1] = epsilon[:, 1]
    delta_y[:, 1] = np.cumsum(delta2_y[:, 1])
    y[:, 1] = np.cumsum(delta_y[:, 1])
    
    # Generate y₁ according to the model
    # Δy₁ₜ = ρ₁Δy₁,ₜ₋₁ + ρ₂(y₁,ₜ₋₁ - Δy₂,ₜ₋₁) + ε₁ₜ
    for t in range(2, T + 10):
        delta_y[t, 0] = (rho1 * delta_y[t-1, 0] + 
                        rho2 * (y[t-1, 0] - delta_y[t-1, 1]) + 
                        epsilon[t, 0])
        y[t, 0] = y[t-1, 0] + delta_y[t, 0]
    
    # Discard initial observations
    y = y[10:]
    
    # Compute true coefficient matrices for ECM representation
    # Δ²y_t = Π₁ Δy_{t-1} + Π₂ y_{t-1} + ε_t
    # 
    # From the DGP:
    # Δy₁ₜ = ρ₁Δy₁,ₜ₋₁ + ρ₂y₁,ₜ₋₁ - ρ₂Δy₂,ₜ₋₁ + ε₁ₜ
    # Δ²y₂ₜ = ε₂ₜ
    #
    # Π₁ = [[ρ₁-1, -ρ₂], [0, 0]]
    # Π₂ = [[ρ₂, 0], [0, 0]]
    
    Pi1 = np.array([[rho1 - 1, -rho2], [0.0, 0.0]])
    Pi2 = np.array([[rho2, 0.0], [0.0, 0.0]])
    
    # True coefficient matrix F = (Π₁, Π₂) for lag order 1
    true_coefficients = np.hstack([Pi1, Pi2])
    
    info = {
        'case': case_name,
        'rho1': rho1,
        'rho2': rho2,
        'Sigma': Sigma,
        'Pi1': Pi1,
        'Pi2': Pi2,
        'true_coefficients': true_coefficients,
        'has_causality': rho2 != 0,
    }
    
    return y, info


def monte_carlo_simulation(dgp: str = 'case_a',
                          T: int = 150,
                          n_reps: int = 1000,
                          bandwidth: Union[int, str] = 'auto',
                          kernel: str = 'bartlett',
                          test_causality: bool = True,
                          seed: Optional[int] = None,
                          verbose: bool = True) -> SimulationResults:
    """
    Run Monte Carlo simulation study.
    
    Replicates the simulation in Section 5 of Chang (2000) to compare
    OLS-VAR and RBFM-VAR estimators.
    
    Parameters
    ----------
    dgp : str
        Data generating process: 'case_a', 'case_b', or 'case_c'
    T : int
        Sample size
    n_reps : int
        Number of replications
    bandwidth : int or str
        Bandwidth for kernel estimation
    kernel : str
        Kernel function
    test_causality : bool
        Whether to compute Granger causality test statistics
    seed : int, optional
        Random seed for reproducibility
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    SimulationResults
        Object containing simulation results
        
    Examples
    --------
    >>> results = monte_carlo_simulation(dgp='case_a', T=150, n_reps=1000)
    >>> print(results.summary())
    
    References
    ----------
    Chang (2000), Section 5
    """
    from .estimation import RBFMVAR
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate one sample to get dimensions
    y_sample, dgp_info = generate_dgp(dgp, T)
    true_coef = dgp_info['true_coefficients']
    n, k = true_coef.shape
    
    # Initialize results storage
    results = SimulationResults(
        n_reps=n_reps,
        sample_size=T,
        dgp=dgp_info['case'],
        true_coefficients=true_coef
    )
    
    results.ols_estimates = np.zeros((n_reps, n, k))
    results.rbfm_estimates = np.zeros((n_reps, n, k))
    
    if test_causality:
        results.ols_wald = np.zeros(n_reps)
        results.rbfm_wald = np.zeros(n_reps)
    
    # Run replications
    for rep in range(n_reps):
        if verbose and (rep + 1) % 100 == 0:
            print(f"Replication {rep + 1}/{n_reps}")
        
        # Generate data
        y, _ = generate_dgp(dgp, T, seed=None)  # Don't reset seed each time
        
        try:
            # Fit RBFM-VAR model
            model = RBFMVAR(lag_order=1, bandwidth=bandwidth, kernel=kernel)
            fit_results = model.fit(y)
            
            # Store estimates
            results.ols_estimates[rep] = model.ols_coefficients
            results.rbfm_estimates[rep] = model.coefficients
            
            # Compute Wald tests for Granger causality
            # H₀: y₂ does not Granger-cause y₁ (i.e., ₁π₁₂ = 0 and ₂π₁₂ = 0)
            if test_causality:
                try:
                    # Modified Wald test
                    gc_modified = fit_results.granger_causality_test(
                        caused_variable=0,
                        causing_variables=[1],
                        test_type='modified'
                    )
                    results.rbfm_wald[rep] = gc_modified['statistic']
                    
                    # Standard Wald test
                    gc_standard = fit_results.granger_causality_test(
                        caused_variable=0,
                        causing_variables=[1],
                        test_type='standard'
                    )
                    results.ols_wald[rep] = gc_standard['statistic']
                    
                except Exception as e:
                    results.rbfm_wald[rep] = np.nan
                    results.ols_wald[rep] = np.nan
                    
        except Exception as e:
            if verbose:
                warnings.warn(f"Error in replication {rep + 1}: {e}")
            results.ols_estimates[rep] = np.nan
            results.rbfm_estimates[rep] = np.nan
            if test_causality:
                results.ols_wald[rep] = np.nan
                results.rbfm_wald[rep] = np.nan
    
    results._is_complete = True
    
    if verbose:
        print("\nSimulation complete.")
    
    return results


def replicate_table1(T: int = 150, 
                     n_reps: int = 10000,
                     seed: int = 42) -> Dict[str, SimulationResults]:
    """
    Replicate Table 1 from Chang (2000).
    
    Runs Monte Carlo simulations for Cases A, B, and C.
    
    Parameters
    ----------
    T : int
        Sample size (150 or 500 in the paper)
    n_reps : int
        Number of replications (10,000 in the paper)
    seed : int
        Random seed
        
    Returns
    -------
    Dict[str, SimulationResults]
        Results for each case
    """
    all_results = {}
    
    for case in ['case_a', 'case_b', 'case_c']:
        print(f"\n{'='*60}")
        print(f"Running {case.upper()}...")
        print('='*60)
        
        results = monte_carlo_simulation(
            dgp=case,
            T=T,
            n_reps=n_reps,
            seed=seed,
            verbose=True
        )
        
        all_results[case] = results
        print(results.summary())
    
    return all_results


def replicate_table2(T: int = 150,
                     n_reps: int = 10000,
                     seed: int = 42) -> str:
    """
    Replicate Table 2 from Chang (2000).
    
    Reports finite sample sizes and rejection probabilities of Wald tests.
    
    Parameters
    ----------
    T : int
        Sample size
    n_reps : int
        Number of replications
    seed : int
        Random seed
        
    Returns
    -------
    str
        Formatted table
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Table 2: Finite Sample Sizes and Rejection Probabilities".center(70))
    lines.append(f"T = {T}".center(70))
    lines.append("=" * 70)
    
    headers = ["Case", "Test", "1% test", "5% test", "10% test"]
    rows = []
    
    for case in ['case_a', 'case_b', 'case_c']:
        print(f"Running {case}...")
        results = monte_carlo_simulation(
            dgp=case,
            T=T,
            n_reps=n_reps,
            test_causality=True,
            seed=seed,
            verbose=False
        )
        
        rows.append([
            case.upper().replace('_', ' '),
            "W_F",
            f"{results.compute_rejection_rate('ols', 0.01):.3f}",
            f"{results.compute_rejection_rate('ols', 0.05):.3f}",
            f"{results.compute_rejection_rate('ols', 0.10):.3f}",
        ])
        rows.append([
            "",
            "W_F^+",
            f"{results.compute_rejection_rate('rbfm', 0.01):.3f}",
            f"{results.compute_rejection_rate('rbfm', 0.05):.3f}",
            f"{results.compute_rejection_rate('rbfm', 0.10):.3f}",
        ])
    
    lines.append(tabulate(rows, headers=headers, tablefmt="simple"))
    lines.append("")
    lines.append("Notes:")
    lines.append("- W_F: Standard Wald test based on OLS-VAR")
    lines.append("- W_F^+: Modified Wald test based on RBFM-VAR")
    lines.append("- Cases A and B: Size (H₀ true)")
    lines.append("- Case C: Power (H₀ false)")
    lines.append("=" * 70)
    
    return "\n".join(lines)
