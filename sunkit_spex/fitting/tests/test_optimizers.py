"""
This module contains package tests for the optimizer functions.
"""

import numpy as np
from numpy.testing import assert_allclose

from sunkit_spex.fitting.metrics import statistics
from sunkit_spex.fitting.optimizers.minimizers import MINIMIZERS
from sunkit_spex.models.models import StraightLineModel


def minimize_func(params, data, x, model, stat):
    """Simple objective function to use to test optimizers."""
    model_output = model.evaluate(x, *params)
    return stat(data, model_output)


def test_scipy_minimize():
    """Test the `scipy_minimize` function against known outputs."""
    sim_x0 = np.arange(3)
    model_params0 = {"slope": 1, "intercept": 0}
    model_param_values0 = tuple(model_params0.values())
    sim_model0 = StraightLineModel(edges=False, **model_params0)
    sim_data0 = sim_model0.evaluate(sim_x0, **model_params0)
    opt_res0 = MINIMIZERS["scipy_minimize"](
        minimize_func, model_param_values0, (sim_data0, sim_x0, sim_model0, statistics.chi_squared)
    )

    sim_x1 = np.arange(3)
    model_params1 = {"slope": 8, "intercept": 5}
    model_param_values1 = tuple(model_params1.values())
    sim_model1 = StraightLineModel(edges=False, **model_params1)
    sim_data1 = sim_model1.evaluate(sim_x1, **model_params1)
    opt_res1 = MINIMIZERS["scipy_minimize"](
        minimize_func, model_param_values1, (sim_data1, sim_x1, sim_model1, statistics.chi_squared)
    )

    assert_allclose(opt_res0.x, model_param_values0, rtol=1e-3)
    assert_allclose(opt_res1.x, model_param_values1, rtol=1e-3)
