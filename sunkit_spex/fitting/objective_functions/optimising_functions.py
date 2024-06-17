def minimize_func(params, data_y, model_x, model_func, statistic_func):
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
    model_y = model_func.evaluate(model_x, *params)
    return statistic_func(data_y, model_y)
