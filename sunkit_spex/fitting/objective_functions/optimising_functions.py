"""
This module contains functions that can evaluate models and return a fit statistic.
"""

__all__ = ["minimize_func"]


def minimize_func(params, obs_spec, model_func, statistic_func):
    """
    Minimization function.

    Parameters
    ----------
    params : `ndarray`
        Guesses of the independent variables.

    data_y : `ndarray`
        The data to be fitted.

    model_x : `ndarray`
        The values at which to evaluate `model_func` at with `params`.

    model_func : `astropy.modeling.core._ModelMeta`
        The model being fitted to the data. Crucially will have an
        `evaluate` method.

    statistic_func : `function`
        The chosen function to compare the data and the model.

    Returns
    -------
    `float`
        The value to be optimized that compares the model to the data.
    """
    model_y = model_func.evaluate(obs_spec._spectral_axis.value, *params)
    return statistic_func(obs_spec.data, model_y)
