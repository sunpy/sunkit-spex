import numpy as np

from astropy import units as u
from astropy.modeling.fitting import Fitter, _validate_model, model_to_fit_params

__all__ = ["JointFitter"]


class JointFitter(Fitter):
    """
    Base class for all joint fitters.

    Parameters
    ==========
    optimizer : callable
        A callable implementing an optimization algorithm.

    statistic : callable
        Statistic function.
    """

    def __init__(self, optimizer, statistic):
        super().__init__(optimizer=optimizer, statistic=statistic)
        self.fit_info = {}

        self._use_min_max_bounds = True

    def joint_model_to_fit_params(self, models):
        """Obtains the parameter values, indices, and bounds to be fitted.

        Parameters
        ==========
        models : `list[~astropy.modeling.FittableModel]`
            A list of the model(s).

        Returns
        =======
        `list[float]`, `list[list[int]]`, `tuple[tuple[float]]` :
            First output (1) is a list of the values being fitted. Second
            output (2) is a list of index lists tracking where each
            value from (1) comes from in each model. Output three (3) is
            a tuple of two tuples, the first and second being the lower
            and upper bounds, respectively, of fittable parameters in (1).

        Examples
        ========
        >>> from astropy.modeling.functional_models import Gaussian1D
        >>> from astropy.modeling.optimizers import SLSQP
        >>> from astropy.modeling.statistic import leastsquare
        >>>
        >>> from sunkit_spex.fitting import fitters
        >>>
        >>> # define models and give bounds
        >>> # fix one parameter so that won't be fittable
        >>> gjf1 = Gaussian1D(amplitude=3, mean=5.5, stddev=0.4, bounds={"mean": (0.5, 10)}, fixed={"stddev": True})
        >>> gjf2 = Gaussian1D(amplitude=1, mean=6, stddev=0.1)
        >>>
        >>> # set-up ``JointFitter`` to use the method
        >>> fit_joint = fitters.JointFitter(optimizer=SLSQP, statistic=leastsquare)
        >>>
        >>> def print_output(output):
        ...     print(f'List of fitting parameter values:\\n\\t{output[0]}')
        ...     print(f'List of fitting parameter indices per model:\\n\\t{output[1]}')
        ...     print(f'Tuple of fitting parameter lower and upper bounds:\\n\\t{output[2]}')
        >>>
        >>> print_output(fit_joint.joint_model_to_fit_params([gjf1, gjf2]))
        List of fitting parameter values:
            [np.float64(3.0), np.float64(5.5), np.float64(1.0), np.float64(6.0), np.float64(0.1)]
        List of fitting parameter indices per model:
            [[0, 1], [0, 1, 2]]
        Tuple of fitting parameter lower and upper bounds:
            ((-inf, 0.5, -inf, -inf, 1.1754943508222875e-38), (inf, 10.0, inf, inf, inf))
        >>>
        >>> # tie one model parameter to another so another won't be fittable
        >>> gjf2.mean.tied = lambda models: models[0].mean.value
        >>> print_output(fit_joint.joint_model_to_fit_params([gjf1, gjf2]))
        List of fitting parameter values:
            [np.float64(3.0), np.float64(5.5), np.float64(1.0), np.float64(0.1)]
        List of fitting parameter indices per model:
            [[0, 1], [0, 2]]
        Tuple of fitting parameter lower and upper bounds:
            ((-inf, 0.5, -inf, 1.1754943508222875e-38), (inf, 10.0, inf, inf))
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
        """Updates model parameter values inplace to the ones given."""
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

        Job is to call ``self._stat_method`` which should calculate the
        value between the model output and data.

        Parameters
        ==========
        fps : `list[float]`
            the fitted parameters - result of an one iteration of the
            fitting algorithm

        *args : `tuple`
            A tuple of measured and input coordinates

        weights : `list[list[float]]`
            A list of the weights associated with the given datasets

        fit_param_indices : `list[list[int]]`
            The ``fit_param_indices`` as returned by ``self.model_to_fit_params``.
            This is a list of the parameter indices being fit, so excluding any
            tied or fixed parameters.  This can be passed in to the objective
            function to prevent it having to be computed on every call.
            This must be optional as not all fitters support passing kwargs to
            the objective function.

        parameter_units : `list[~astropy.Quantity]` or `NoneType`
            A list of parameter units for the fittable parameters if they
            exist.
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
        """Verify the arguments given come in ``model, x, y`` groups."""
        if len(args) % 3 != 0:
            raise ValueError(
                f"Expected list of ``model1, x1, y1, model2, ...`` in args "
                f"but {len(args)} provided is not divisible by 3."
            )

    def _run_fitter(self, models, farg, fkwarg=None):
        """
        Function to set-up and run the optimization method.

        Job is to call `self._opt_method` and pass in `self.objective_function`

        - farg are the xs and ys for the models
        - fkwargs are anything else needed to be passed to the objective
        function
        """
        fkwarg = {} if fkwarg is None else fkwarg

        jmodel_params, jfit_param_indices, jmodel_bounds = self.joint_model_to_fit_params(models)

        param_units = self._get_param_units(models)

        fkwarg |= {"jfit_param_indices": jfit_param_indices, "parameter_units": param_units}

        from scipy.optimize import minimize  # noqa

        ## should really call self._opt_method()
        # optimize.least_squares just minimises the square of what comes out of self.objective_function
        # need the lambda function for now since `minimize` won't accept other kwargs
        fun = lambda x: self.objective_function(x, *(models, farg), **fkwarg)  # noqa
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
        """Obtains the parameter units if the model supports units."""
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
        return None

    def __call__(self, *args, fkwarg=None, inplace=False):
        """
        Fit data to these models keeping some of the parameters common to the
        two models.

        Purpose is to setup, call `self._run_fitter`, and sort results.

        Parameters
        ==========
        *args :
            The ``model``, ``x``, and ``y`` groups given.

        fkwarg :
            Fitting keyword arguments.
            Default: None

        inplace : `bool`
            Defines whether the models should be updated in-place or if
            updates should happen to copies of the models.
            Default: False

        Returns
        =======
        `list[~astropy.modeling.FittableModel]` :
            A list of the models (copies if ``inplace`` is False) with
            their parameters updated with the fitted result.
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
    ==========
    model :
        The model being fit
    fps :
        The fit parameter values to be assigned
    use_min_max_bounds: bool
        If set, the parameter bounds for the model will be enforced on each
        parameter with bounds.
        Default: True
    fit_param_indices :
        Index locations of the fit parameters located in the model.
    model_list :
        Is checked when parameters are tied and allows one model's
        parameter to be tied to another, separate model's parameter in
        the list.
        Default: None
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
    ==========
    model :
        The model being fit
    fps :
        The fit parameter values to be assigned
    use_min_max_bounds: bool
        If set, the parameter bounds for the model will be enforced on each
        parameter with bounds.
        Default: True
    model_list :
        Is checked when parameters are tied and allows one model's
        parameter to be tied to another, separate model's parameter in
        the list.
        Default: None
    """
    _, fit_param_indices, _ = model_to_fit_params(model)
    parameters = fitter_to_model_params_array(
        model, fps, use_min_max_bounds, fit_param_indices=fit_param_indices, model_list=model_list
    )
    model.parameters = parameters
