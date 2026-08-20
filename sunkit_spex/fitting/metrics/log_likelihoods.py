"""
This module contains functions that compute a log-likelihood between two data-sets.
"""

import numpy as np

from astropy import units as u

from sunkit_spex.fitting.metrics.statistics import chi_squared
from sunkit_spex.fitting.metrics.util import weights_to_error

__all__ = ["gaussian"]


def gaussian(data_ys, model_ys, data_y_weights, **kwargs):
    r"""Gaussian log-likelihood (to be maximized).

    .. math::
        ln(L_{Gauss}) = -\frac{N}{2} ln(2\pi \sigma^{2}) - \frac{1}{2}\Chi^{2}

    where N is the number of observed bins, sigma is the data error, and
    chi-squared is its usual, minimise version.

    Parameters
    ----------
    data_ys : `tuple[ndarray]`
        The data to be fitted.

    model_ys : `tuple[ndarray]`
        The model values being fitted.

    data_y_weights : `tuple[ndarray]`
        The associated weights for `data_ys`. Weights are 1/error**2.

    Returns
    -------
    `float`
        A float, the gaussian log-likelihood.
    """

    log_likelihood_value = 0
    for i, data_y in enumerate(data_ys):
        log_likelihoods = gaussian_maximum(data_y, data_y_weights[i]) - (1 / 2) * chi_squared(
            (data_y,), (model_ys[i],), data_y_weights=(data_y_weights[i],)
        )

        log_likelihood_value += log_likelihoods.value if isinstance(log_likelihoods, u.Quantity) else log_likelihoods

    # best value is first whole term, if the chi squared section has any value then it is always subtracted
    return log_likelihood_value  # =ln(L)


def gaussian_maximum(data_y, data_y_weight):
    r"""The maximum value the Gaussian log-likelihood can ever achieve.

    .. math::
        ln(L_{Gauss Maximum}) = -\frac{N}{2} ln(2\pi \sigma^{2})

    where N is the number of observed bins and sigma is the data error.

    Parameters
    ----------
    data_y : `ndarray`
        The data to be fitted.

    data_y_weight : `ndarray`
        The associated weights for `data_ys`. Weights are 1/error**2.

    Returns
    -------
    `float`
    """

    return np.sum(-(len(data_y) / 2) * np.log(2 * np.pi * weights_to_error(data_y_weight) ** 2))
