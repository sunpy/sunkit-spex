from scipy.optimize import minimize

__all__ = ["scipy_minimize"]


def scipy_minimize(objective_func, param_guesses, objective_func_args, **kwargs):
    """ A function to optimize fitted parameters to data.

    Parameters
    ----------
    objective_func : `function`
        The function to be optimized.

    param_guesses : `ndarray`
        Initial guesses of the independent variables.

    objective_func_args : `tuple`
        Any arguments required to be passed to the objective function
        after the param_guesses.
        E.g., `objective_func(param_guesses, *objective_func_args)`.

    kwargs :
        Passed to `scipy.optimize.minimize`.
        A default value for the method is chosen to be "Nelder-Mead".

    Returns
    -------
    `scipy.optimize.OptimizeResult`
        The optimized result after comparing the model to the data.
    """

    method = kwargs.pop("method", "Nelder-Mead")

    return minimize(objective_func,
                    param_guesses,
                    args=objective_func_args,
                    method=method,
                    **kwargs)
