import numpy as np

from astropy import units as u
from astropy.modeling.fitting import Fitter, _validate_model, model_to_fit_params


class JointFitter(Fitter):
    def __init__(self, optimizer, statistic):
        super().__init__(optimizer=optimizer, statistic=statistic)
        self.fit_info = {}

        self._use_min_max_bounds = True

    def joint_model_to_fit_params(self, models):
        """
        jmodel_params (N_fparams,)
        jfit_param_indices (N_models, N_fparams_in_model)
        jmodel_bounds ((N_fparams,), (N_fparams,))
        """
        jmodel_params = []
        jfit_param_indices = []
        jmodel_bounds_lower = []
        jmodel_bounds_upper = []
        for _, model in enumerate(models):
            model_params, fit_param_indices, model_bounds = model_to_fit_params(model)
            jmodel_params.extend(model_params)
            jfit_param_indices.append(fit_param_indices)
            jmodel_bounds_lower.extend(model_bounds[0])
            jmodel_bounds_upper.extend(model_bounds[1])

        # return
        jmodel_bounds = (tuple(jmodel_bounds_lower), tuple(jmodel_bounds_upper))
        return jmodel_params, jfit_param_indices, jmodel_bounds

    def _update_model_params(self, models, fps, jfit_param_indices):
        for mod_num, model in enumerate(models):
            # The param list is rebuilt with those being fitted and those not
            free_inds_in_mod = jfit_param_indices[mod_num]
            mod_fps = fps[: len(free_inds_in_mod)]

            # get the reconstructed parameter array for the given model
            mod_params = fitter_to_model_params_array(
                model, mod_fps, self._use_min_max_bounds, fit_param_indices=free_inds_in_mod, model_list=models
            )
            # update the model so tied parameters stay up-to-date
            model.parameters = mod_params

            # avoid awkward indexing by removing the params we've used and accounted for
            remove = np.arange(len(free_inds_in_mod))
            fps = np.delete(fps, remove)
            del mod_fps, free_inds_in_mod

    def objective_function(self, fps, *args, weights=None, jfit_param_indices=None, parameter_units=None):
        """
        Function to minimize.

        Parameters
        ----------
        fps : list
            the fitted parameters - result of an one iteration of the
            fitting algorithm
        args : dict
            tuple of measured and input coordinates
            args is always passed as a tuple from optimize.leastsq
        fit_param_indices : list, optional
            The ``fit_param_indices`` as returned by ``self.model_to_fit_params``.
            This is a list of the parameter indices being fit, so excluding any
            tied or fixed parameters.  This can be passed in to the objective
            function to prevent it having to be computed on every call.
            This must be optional as not all fitters support passing kwargs to
            the objective function.

        Job is to call `self._stat_method`
        """

        models = args[0]
        xs = args[1][0]
        ys = args[1][1]

        fitted = []

        # double call in case earlier model relies on new parameters in later model
        self._update_model_params(models, fps, jfit_param_indices)
        self._update_model_params(models, fps, jfit_param_indices)

        for mod_num, model in enumerate(models):
            # units, make sure we have units
            if parameter_units is not None:
                pu = parameter_units[mod_num]
                mod_params = [
                    round(mp, 15) if unit is None else round(mp, 15) * unit for mp, unit in zip(model.parameters, pu)
                ]

            # actually evaluate the model with the constructed units and get residuals
            res = model.evaluate(xs[mod_num], *mod_params) - ys[mod_num]
            value = res if weights is None else weights[mod_num] * res
            value = value.value if isinstance(value, u.Quantity) else value

            fitted.extend(value)

        return np.sum(np.ravel(fitted) ** 2)

    def _verify_input(self, args):
        if len(args) % 3 != 0:
            raise ValueError(
                f"Expected list of ``model1, x1, y1, model2, ...`` in args "
                f"but {len(args)} provided is not divisible by 3."
            )

    def _run_fitter(self, models, farg, fkwarg=None):
        """
        models are the models
        farg are the xs and ys for the models
        fkwargs are anything else

        Job is to call `self._opt_method` and pass in `self.objective_function`
        """
        fkwarg = {} if fkwarg is None else fkwarg

        jmodel_params, jfit_param_indices, jmodel_bounds = self.joint_model_to_fit_params(models)

        param_units = self._get_param_units(models)

        fkwarg |= {"jfit_param_indices": jfit_param_indices, "parameter_units": param_units}

        from scipy.optimize import minimize

        ## should really call self._opt_method()
        # optimize.least_squares just minimises the square of what comes out of self.objective_function
        fun = lambda x: self.objective_function(x, *(models, farg), **fkwarg)
        result = minimize(
            fun,
            jmodel_params,
            bounds=tuple(zip(jmodel_bounds[0], jmodel_bounds[1])),
            method="Nelder-Mead",
            tol=1e-8,
        )

        self.fit_info["params"] = result.x

        # update the models with the fitted values
        for mod_num, model in enumerate(models):
            free_inds_in_mod = jfit_param_indices[mod_num]
            fps = result.x[: len(free_inds_in_mod)]
            parameters = fitter_to_model_params_array(
                model, fps, self._use_min_max_bounds, fit_param_indices=jfit_param_indices[mod_num], model_list=models
            )
            model.parameters = parameters
            remove = np.arange(len(free_inds_in_mod))
            result.x = np.delete(result.x, remove)

        return models

    @staticmethod
    def _get_param_units(models):
        param_units = []
        for model in models:
            if model._supports_unit_fitting:
                units = [getattr(model, name).unit for name in model.param_names]
                units = [u.dimensionless_unscaled if unit is None else unit for unit in units]
                param_units.append(units)
            else:
                param_units.append([None for _ in model.param_names])

        if len(param_units) > 0:
            return param_units

    def __call__(self, *args, fkwarg=None, inplace=False):
        """
        Fit data to these models keeping some of the parameters common to the
        two models.

        Setup, call `self._run_fitter`, and sort results
        """
        self._verify_input(args)

        models = list(args[::3])
        xs = list(args[1::3])
        ys = list(args[2::3])

        # verify the models and change the ones in the fitters to copies of the original
        for model_num, model in enumerate(models):
            models[model_num] = _validate_model(
                model,
                ["fixed", "tied", "bounds", "eqcons", "ineqcons"],
                copy=not inplace,
            )

        self.fitted_models = self._run_fitter(
            models,
            (xs, ys),
            fkwarg=fkwarg,
        )

        return self.fitted_models


## KRIS: All I did was add ``model_list``
def fitter_to_model_params_array(model, fps, use_min_max_bounds=True, *, fit_param_indices=None, model_list=None):
    """
    Constructs the full list of model parameters from the fitted and
    constrained parameters.

    Parameters
    ----------
    model :
        The model being fit
    fps :
        The fit parameter values to be assigned
    use_min_max_bounds: bool
        If set, the parameter bounds for the model will be enforced on each
        parameter with bounds.
        Default: True
    """
    has_tied = model.has_tied
    has_bound = use_min_max_bounds and model.has_bounds
    if not (has_tied or model.has_fixed or has_bound):
        return fps

    bounds = model.bounds
    param_metrics = model._param_metrics
    parameters = np.empty(sum(m["size"] for m in param_metrics.values()), dtype=float)

    if fit_param_indices is None:
        _, fit_param_indices, _ = model_to_fit_params(model)

    offset = 0
    for idx, name in enumerate(model.param_names):
        metrics = param_metrics[name]
        slice_ = metrics["slice"]
        if idx not in fit_param_indices:
            parameters[slice_] = getattr(model, name).value
            continue

        shape = metrics["shape"]
        size = metrics["size"]

        values = fps[offset : offset + size]

        # Check bounds constraints
        bound = bounds[name]
        if has_bound and bound != (None, None):
            _min, _max = bound
            if _min is not None:
                values = np.fmax(values, _min)
            if _max is not None:
                values = np.fmin(values, _max)

        parameters[slice_] = values
        offset += size

    # This has to be done in a separate loop due to how tied parameters are
    # currently evaluated (the fitted parameters need to actually be *set* on
    # the model first, for use in evaluating the "tied" expression--it might be
    # better to change this at some point
    if has_tied:
        # Update model parameters before calling ``tied`` constraints.
        model.parameters = parameters

        for name in model.param_names:
            if model.tied[name]:
                value = model.tied[name](model) if model_list is None else model.tied[name](model_list)
                slice_ = param_metrics[name]["slice"]

                # To handle multiple tied constraints, model parameters
                # need to be updated after each iteration.
                parameters[slice_] = value
                model._array_to_parameters()

    return parameters


def fitter_to_model_params(model, fps, use_min_max_bounds=True, model_list=None):
    """
    Constructs the full list of model parameters from the fitted and
    constrained parameters.

    Parameters
    ----------
    model :
        The model being fit
    fps :
        The fit parameter values to be assigned
    use_min_max_bounds: bool
        If set, the parameter bounds for the model will be enforced on each
        parameter with bounds.
        Default: True
    """
    _, fit_param_indices, _ = model_to_fit_params(model)
    parameters = fitter_to_model_params_array(
        model, fps, use_min_max_bounds, fit_param_indices=fit_param_indices, model_list=model_list
    )
    model.parameters = parameters
