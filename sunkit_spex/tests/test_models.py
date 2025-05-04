"""
This module contains package tests for package models.
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sunkit_spex.data.simulated_data import simulate_square_response_matrix
from sunkit_spex.models.instrument_response import MatrixModel
from sunkit_spex.models.models import GaussianModel, StraightLineModel


def test_StraightLineModel():
    """Test the straight line model evaluation methods to a known output."""
    sim_x0 = np.arange(3)
    model_params0_init = {"edges": False, "slope": 1, "intercept": 0}
    model_params0_eval = {"slope": 1, "intercept": 0}
    sim_model0 = StraightLineModel(**model_params0_init)
    exp_res0 = [0, 1, 2]
    ans0_0 = sim_model0(sim_x0)
    ans0_1 = sim_model0.evaluate(sim_x0, *tuple(model_params0_eval.values()))

    assert_allclose(exp_res0, ans0_0, rtol=1e-3)
    assert_allclose(ans0_0, ans0_1, rtol=1e-3)


def test_GaussianModel():
    """Test the Gaussian model evaluation methods to a known output."""
    sim_x0 = np.arange(-1, 2) * np.sqrt(2 * np.log(2))
    model_params0_init = {"edges": False, "amplitude": 10, "mean": 0, "stddev": 1}
    model_params0_eval = {"amplitude": 10, "mean": 0, "stddev": 1}
    sim_model0 = GaussianModel(**model_params0_init)
    exp_res0 = [5, 10, 5]
    ans0_0 = sim_model0(sim_x0)
    ans0_1 = sim_model0.evaluate(sim_x0, *tuple(model_params0_eval.values()))

    assert_allclose(exp_res0, ans0_0, rtol=1e-3)
    assert_allclose(ans0_0, ans0_1, rtol=1e-3)


def test_StraightLineModel_edges():
    """Test the straight line model evaluation methods to a known output."""
    sim_x0 = np.arange(3)
    model_params0 = {"slope": 1, "intercept": 0}
    sim_model0 = StraightLineModel(**model_params0)
    exp_res0 = [0.5, 1.5]
    ans0_0 = sim_model0(sim_x0)
    ans0_1 = sim_model0.evaluate(sim_x0, *tuple(model_params0.values()))

    assert_allclose(exp_res0, ans0_0, rtol=1e-3)
    assert_allclose(ans0_0, ans0_1, rtol=1e-3)


def test_GaussianModel_edges():
    """Test the Gaussian model evaluation methods to a known output."""
    sim_x0 = np.arange(-1, 2) * np.sqrt(2 * np.log(2))
    model_params0 = {"amplitude": 10, "mean": 0, "stddev": 1}
    sim_model0 = GaussianModel(**model_params0)
    exp_res0 = [8.40896415, 8.40896415]
    ans0_0 = sim_model0(sim_x0)
    ans0_1 = sim_model0.evaluate(sim_x0, *tuple(model_params0.values()))

    assert_allclose(exp_res0, ans0_0, rtol=1e-3)
    assert_allclose(ans0_0, ans0_1, rtol=1e-3)


def test_MatrixModel():
    """Test the matrix model contents and compound model behaviour."""
    size0 = 3
    srm0 = simulate_square_response_matrix(size0)
    srm_model0 = MatrixModel(matrix=srm0)

    assert_array_equal(srm_model0.matrix, srm0)

    sim_x0 = np.arange(size0)
    model_params0_init = {"edges": False, "slope": 1, "intercept": 0}
    sim_model0 = StraightLineModel(**model_params0_init)
    comp_model0 = sim_model0 | srm_model0
    comp_res0 = comp_model0(sim_x0)
    exp_res0 = [0.00682338, 1.00348448, 1.98969213]

    assert_allclose(comp_res0, exp_res0, rtol=1e-6)
