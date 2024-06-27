# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module contains package tests for the optimising_functions functions.
"""

import numpy as np

from sunkit_spex.fitting.objective_functions.optimising_functions import minimize_func
from sunkit_spex.fitting.statistics.gaussian import chi_squared
from sunkit_spex.models.models import StraightLineModel


def test_minimize_func():
    sim_x0 = np.arange(3)
    model_params0 = {"slope": 1, "intercept": 0}
    sim_model0 = StraightLineModel(**model_params0)
    sim_data0 = sim_model0.evaluate(sim_x0, **model_params0)
    res0 = minimize_func(
        params=tuple(model_params0.values()),
        data_y=sim_data0,
        model_x=sim_x0,
        model_func=sim_model0,
        statistic_func=chi_squared,
    )

    assert res0 == 0
