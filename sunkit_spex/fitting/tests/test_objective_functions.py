"""
This module contains package tests for the objective functions.
"""

import numpy as np

import astropy.units as u

from sunkit_spex.fitting.objective_functions.optimising_functions import minimize_func
from sunkit_spex.fitting.statistics.gaussian import chi_squared
from sunkit_spex.models.models import StraightLineModel
from sunkit_spex.spectrum import Spectrum


def test_minimize_func():
    """Test the `minimize_func` function against known outputs."""
    sim_x0 = np.arange(3) * u.keV
    model_params0 = {"slope": 1, "intercept": 0}
    sim_model0 = StraightLineModel(**model_params0)
    sim_data0 = sim_model0.evaluate(sim_x0, **model_params0)
    res0 = minimize_func(
        params=tuple(model_params0.values()),
        obs_spec=Spectrum(sim_data0, spectral_axis=sim_x0),
        model_func=sim_model0,
        statistic_func=chi_squared,
    )

    sim_x1 = np.arange(3) * u.keV
    model_params1 = {"slope": 1, "intercept": 0}
    sim_model1 = StraightLineModel(**model_params1)
    sim_data1 = sim_model1.evaluate(sim_x1, **model_params1)[::-1]
    res1 = minimize_func(
        params=tuple(model_params1.values()),
        obs_spec=Spectrum(sim_data1, spectral_axis=sim_x1),
        model_func=sim_model1,
        statistic_func=chi_squared,
    )

    assert res0 == 0
    assert res1 == 8
