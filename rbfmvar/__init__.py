"""
RBFM-VAR: Residual-Based Fully Modified Vector Autoregression
==============================================================

A Python implementation of the Residual-Based Fully Modified Vector Autoregression
(RBFM-VAR) methodology for nonstationary vector autoregressions with unknown
mixtures of I(0), I(1), and I(2) components.

Based on: Chang, Y. (2000). "Vector Autoregressions with Unknown Mixtures of I(0),
I(1), and I(2) Components", Econometric Theory, 16(6), 905-926.

Author: Dr. Merwan Roudane
Email: merwanroudane920@gmail.com
GitHub: https://github.com/merwanroudane/rbfmvar

Version: 0.0.2
"""

__version__ = "2.0.0"
__author__ = "Dr. Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"
__url__ = "https://github.com/merwanroudane/rbfmvar"

# Core classes
from .estimation import RBFMVAR, OLSVAREstimator
from .results import RBFMVARResults

# Covariance estimation
from .covariance import (
    kernel_covariance,
    one_sided_kernel_covariance,
    bartlett_kernel,
    parzen_kernel,
    quadratic_spectral_kernel,
    optimal_bandwidth,
)

# Hypothesis testing
from .testing import (
    wald_test,
    modified_wald_test,
    granger_causality_test,
)

# Utility functions
from .utils import (
    lag_matrix,
    difference,
    vec,
    unvec,
    kronecker_product,
)

# Simulation tools
from .simulation import (
    generate_dgp,
    monte_carlo_simulation,
    SimulationResults,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__url__",
    # Core classes
    "RBFMVAR",
    "OLSVAREstimator",
    "RBFMVARResults",
    # Covariance
    "kernel_covariance",
    "one_sided_kernel_covariance",
    "bartlett_kernel",
    "parzen_kernel",
    "quadratic_spectral_kernel",
    "optimal_bandwidth",
    # Testing
    "wald_test",
    "modified_wald_test",
    "granger_causality_test",
    # Utilities
    "lag_matrix",
    "difference",
    "vec",
    "unvec",
    "kronecker_product",
    # Simulation
    "generate_dgp",
    "monte_carlo_simulation",
    "SimulationResults",
]
