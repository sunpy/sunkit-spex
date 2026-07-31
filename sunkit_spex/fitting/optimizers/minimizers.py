"""
This module contains functions to wrap around minimizer tools.
"""

from scipy.optimize import minimize

__all__ = ["MINIMIZERS"]

MINIMIZERS = {"scipy_minimize":minimize}
