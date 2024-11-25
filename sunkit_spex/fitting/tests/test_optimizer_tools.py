"""
This module contains package tests for the optimizer functions.
"""

import numpy as np
from numpy.testing import assert_allclose

import astropy.units as u

from sunkit_spex.fitting.objective_functions.optimising_functions import minimize_func
from sunkit_spex.fitting.optimizer_tools.minimizer_tools import scipy_minimize
from sunkit_spex.fitting.statistics.gaussian import chi_squared
from sunkit_spex.models.models import StraightLineModel
from sunkit_spex.spectrum import Spectrum


def test_scipy_minimize():
    """Test the `scipy_minimize` function against known outputs."""
    sim_x0 = np.arange(3)
    model_params0 = {"slope": 1, "intercept": 0}
    model_param_values0 = tuple(model_params0.values())
    sim_model0 = StraightLineModel(**model_params0)
    sim_data0 = sim_model0.evaluate(sim_x0, **model_params0)
    opt_res0 = scipy_minimize(
        minimize_func,
        model_param_values0,
        (Spectrum(sim_data0 * u.dimensionless_unscaled, spectral_axis=sim_x0 * u.keV), sim_model0, chi_squared),
    )

    sim_x1 = np.arange(3)
    model_params1 = {"slope": 8, "intercept": 5}
    model_param_values1 = tuple(model_params1.values())
    sim_model1 = StraightLineModel(**model_params1)
    sim_data1 = sim_model1.evaluate(sim_x1, **model_params1)
    opt_res1 = scipy_minimize(
        minimize_func,
        model_param_values1,
        (Spectrum(sim_data1 * u.dimensionless_unscaled, spectral_axis=sim_x1 * u.keV), sim_model1, chi_squared),
    )

    assert_allclose(opt_res0.x, model_param_values0, rtol=1e-3)
    assert_allclose(opt_res1.x, model_param_values1, rtol=1e-3)
