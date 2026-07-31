"""
This module contains functions that compute a fit statistic between two data-sets.
"""

from astropy import units as u
import numpy as np

from sunkit_spex.fitting.metrics import decorators

__all__ = ["chi_squared"]


@decorators.check_metric_inputs
def chi_squared(data_ys, model_ys, data_y_weights=None, **kwargs):
    """
    The value to optimise while fitting.

    Parameters
    ----------
    data_ys : `tuple[ndarray]`
        The data to be fitted.

    model_ys : `tuple[ndarray]`
        The model values being fitted.

    data_y_weights : `tuple[ndarray]` or `NoneType`
        The associated weights for `data_ys`. Weights are 1/error**2. If
        given `None`, weights will be ignored and the statistic will be 
        the square of the difference between data and model only.

    Returns
    -------
    `float`
        The value to be optimized that compares the model to the data.
    """

    stat_value = 0
    for i, data_y in enumerate(data_ys):
        if data_y_weights is None:
            _stat_value = np.sum(((data_y - model_ys[i]) ** 2))
        else:
            _stat_value = np.sum(((data_y - model_ys[i]) ** 2)*data_y_weights[i])
        stat_value += _stat_value.value if isinstance(_stat_value, u.Quantity) else _stat_value

    return stat_value
