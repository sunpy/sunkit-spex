"""Decorators for the fit metrics module."""


def check_metric_inputs(func):
    """Metrics should contain the same base inputs so need to check them.

    Want to check that the length of data and model are the same, then
    again with the weights if they are given.
    """

    def wrapper(data_ys, model_ys, **kwargs):

        if len(data_ys) != len(model_ys):
            raise ValueError("Inputs `data_ys_tuple` and `model_ys_tuple` must be tuples of same length.")

        if "data_y_weights" in kwargs:
            if (kwargs["data_y_weights"] is not None) and (len(data_ys) != len(kwargs["data_y_weights"])):
                raise ValueError(
                    "Inputs `data_ys_tuple` and `data_y_weights` must be tuples of of same length if `data_y_weights` is not `None`."
                )
            return func(data_ys, model_ys, **kwargs)
        return func(data_ys, model_ys, data_y_weights=None, **kwargs)

    return wrapper
