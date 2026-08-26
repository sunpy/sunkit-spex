"""
This module contains functions that compute a fit statistic between two data-sets.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = ["chi_squared"]


def chi_squared(data_y: NDArray[np.float64], model_y: NDArray[np.float64]) -> np.floating:
    """
    The form to optimise while fitting.

    * No error included here. *

    Parameters
    ----------
    data_y : `ndarray`
        The data to be fitted.

    model_y : `ndarray`
        The model values being fitted.

    Returns
    -------
    `float`
        The value to be optimized that compares the model to the data.
    """
    return np.sum((data_y - model_y) ** 2)
