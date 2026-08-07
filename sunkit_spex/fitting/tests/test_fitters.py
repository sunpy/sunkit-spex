"""
This module contains package tests for the fitters functions.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from astropy import units as u
from astropy.modeling import models
from astropy.modeling.optimizers import SLSQP
from astropy.modeling.statistic import leastsquare

from sunkit_spex.data.simulated_data import simulate_gaussian_data_source, simulate_thermal_data_source
from sunkit_spex.fitting.fitters import JointFitter, ScipyMinimizeJointFitter, fitter_to_model_params_array
from sunkit_spex.models.physical.thermal import ThermalEmission


def test_JointFitter_joint_model_to_fit_params():
    """Test the `JointFitter` class method: ``joint_model_to_fit_params``.

    This tests the correct initial values are extracted from the models,
    the correct fittable parameter indices, and the correct parameter bounds.
    """

    # define models
    guess_amplitude = 5
    guess_mean = 6
    guess_stddev1 = 0.3
    guess_stddev2 = 0.5
    g1 = models.Gaussian1D(amplitude=guess_amplitude, mean=guess_mean, stddev=guess_stddev1)
    g2 = models.Gaussian1D(stddev=guess_stddev2)

    # tie relevant parameters
    g2.amplitude.tied = lambda models: models[0].amplitude
    g2.mean.tied = lambda models: models[0].mean

    # initialise the fitter, need to pass optimizer and statistic to avoid error
    fit_joint = JointFitter(optimizer=SLSQP, statistic=leastsquare)

    # check if the correct parameters are extracted for fitting
    extracted_param_info = fit_joint.joint_model_to_fit_params([g1, g2])
    # check fittable parameter values are good
    assert_allclose(extracted_param_info[0], [g1.amplitude.value, g1.mean.value, g1.stddev.value, g2.stddev.value])
    # check fittable parameter indices are good
    assert_allclose(extracted_param_info[1][0], [0, 1, 2])
    assert_allclose(extracted_param_info[1][1], [2])
    # check fittable parameter bounds are good
    assert_allclose(extracted_param_info[2][0], (-np.inf, -np.inf, 1.1754943508222875e-38, 1.1754943508222875e-38))
    assert_allclose(extracted_param_info[2][1], (np.inf, np.inf, np.inf, np.inf))


def test_JointFitter__update_model_params():
    """Test the `JointFitter` class method: ``_update_model_params``.

    This tests the model parameters are update with new parameters.
    These are done in order of the model list.

    This test tests model parameters being tied and the back-tying parameters
    work (tying a parameter from a model earlier in the list to a parameter
    from a later model), but that forward tying (earlier model parameter tied
    to a later model parameter does not work).
    """
    # define models
    guess_amplitude = 5
    guess_mean = 6
    guess_stddev1 = 0.3
    guess_stddev2 = 0.5
    g1 = models.Gaussian1D(amplitude=guess_amplitude, mean=guess_mean, stddev=guess_stddev1)
    g2 = models.Gaussian1D(stddev=guess_stddev2)

    # tie relevant parameters
    g2.amplitude.tied = lambda models: models[0].amplitude
    g2.mean.tied = lambda models: models[0].mean

    # initialise the fitter, need to pass optimizer and statistic to avoid error
    fit_joint = JointFitter(optimizer=SLSQP, statistic=leastsquare)

    # confirm models have coprrect parameters to begin with
    assert_allclose(
        [*g1.parameters, g2.parameters[2]], [g1.amplitude.value, g1.mean.value, g1.stddev.value, g2.stddev.value]
    )

    # check if the correct parameters are extracted for fitting
    # they set up is that we're fitting all three parameters from model g1 and only the final parameter from g2
    updated_params = [20, 42, 137, 3.14]
    value2param_indices = [[0, 1, 2], [2]]
    fit_joint._update_model_params([g1, g2], updated_params, value2param_indices)
    # show the fit param indices updated everything correctly
    assert_allclose([*g1.parameters, g2.parameters[2]], updated_params)
    # show tying parameters (back-tying at least) works too
    assert_allclose(g2.parameters[:2], g1.parameters[:2])
    assert_allclose(g2.parameters[:2], updated_params[:2])

    # show forward tying fails
    g2.mean.tied = False
    g1.mean.tied = lambda models: models[1].mean
    original_g2_mean = g2.mean.value
    value2param_indices = [[0, 2], [1, 2]]
    fit_joint._update_model_params([g1, g2], updated_params, value2param_indices)
    # check update of fittable params still works to make sure
    assert_allclose([g1.parameters[0], g1.parameters[2], g2.parameters[1], g2.parameters[2]], updated_params)
    # now show the mean from g1 still has the old value of the mean from g2
    assert_allclose([g1.parameters[1]], [original_g2_mean])
    # but the following should fail if forward tying worked
    with pytest.raises(AssertionError):
        assert g1.parameters[1] == g2.parameters[1]
    with pytest.raises(AssertionError):
        assert g1.parameters[1] == updated_params[2]


def test_JointFitter__update_model_params_twice():
    """Test the `JointFitter` class method: ``_update_model_params_twice``.

    This tests the model parameters being updated twice and shows forward
    tying parameters is now posisble.
    """
    # define models
    guess_amplitude = 5
    guess_mean = 6
    guess_stddev1 = 0.3
    guess_stddev2 = 0.5
    g1 = models.Gaussian1D(amplitude=guess_amplitude, mean=guess_mean, stddev=guess_stddev1)
    g2 = models.Gaussian1D(stddev=guess_stddev2)

    # tie relevant parameters
    g1.amplitude.tied = lambda models: models[1].amplitude
    g1.mean.tied = lambda models: models[1].mean

    # initialise the fitter, need to pass optimizer and statistic to avoid error
    fit_joint = JointFitter(optimizer=SLSQP, statistic=leastsquare)

    # check the values we're going to tie to make sure they start off as we expect
    assert_allclose(g1.parameters[:2], [guess_amplitude, guess_mean])

    # show forward tying works
    updated_params = [20, 42, 137, 3.14]
    value2param_indices = [[2], [0, 1, 2]]
    fit_joint._update_model_params_twice([g1, g2], updated_params, value2param_indices)
    # check update of fittable params still works to make sure
    assert_allclose([g1.parameters[2], *g2.parameters], updated_params)
    # now show the amplitude and mean from g1 no longer has it's original values
    with pytest.raises(AssertionError):
        assert_allclose(g1.parameters[:2], [guess_amplitude, guess_mean])
    # now check the first model parameters were update with the updated parameters from the second model
    assert_allclose(g1.parameters[:2], g2.parameters[:2])
    assert_allclose(g1.parameters[:2], updated_params[1:3])


def test_JointFitter__verify_input():
    """Test the `JointFitter` class method: ``_verify_input``."""
    # initialise the fitter, need to pass optimizer and statistic to avoid error
    fit_joint = JointFitter(optimizer=SLSQP, statistic=leastsquare)
    with pytest.raises(ValueError, match=r"Expected*"):
        fit_joint._verify_input((0,))
    with pytest.raises(ValueError, match=r"Expected*"):
        fit_joint._verify_input((0, 1))

    fit_joint._verify_input((0, 1, 2))

    with pytest.raises(ValueError, match=r"Expected*"):
        fit_joint._verify_input((0, 1, 2, 3))
    with pytest.raises(ValueError, match=r"Expected*"):
        fit_joint._verify_input((0, 1, 2, 3, 4))

    fit_joint._verify_input((0, 1, 2, 3, 4, 5))


def test_JointFitter__get_param_units():
    """Test the `JointFitter` class method: ``_get_param_units``."""
    gu1 = ThermalEmission(temperature=10 << u.MK, emission_measure=1 << u.cm**-3)

    # for one model
    param_units = JointFitter._get_param_units([gu1])
    assert param_units[0][0] == u.MK
    assert param_units[0][1] == u.cm**-3
    assert param_units[0][2] == u.dimensionless_unscaled
    assert param_units[0][3] == u.dimensionless_unscaled
    assert param_units[0][4] == u.dimensionless_unscaled
    assert param_units[0][5] == u.dimensionless_unscaled
    assert param_units[0][5] == u.dimensionless_unscaled
    assert param_units[0][6] == u.dimensionless_unscaled
    assert param_units[0][7] == u.dimensionless_unscaled

    # for two models
    param_units = JointFitter._get_param_units([gu1, gu1])
    assert param_units[0][0] == u.MK
    assert param_units[0][1] == u.cm**-3
    assert param_units[0][2] == u.dimensionless_unscaled
    assert param_units[0][3] == u.dimensionless_unscaled
    assert param_units[0][4] == u.dimensionless_unscaled
    assert param_units[0][5] == u.dimensionless_unscaled
    assert param_units[0][5] == u.dimensionless_unscaled
    assert param_units[0][6] == u.dimensionless_unscaled
    assert param_units[0][7] == u.dimensionless_unscaled
    assert param_units[1][0] == u.MK
    assert param_units[1][1] == u.cm**-3
    assert param_units[1][2] == u.dimensionless_unscaled
    assert param_units[1][3] == u.dimensionless_unscaled
    assert param_units[1][4] == u.dimensionless_unscaled
    assert param_units[1][5] == u.dimensionless_unscaled
    assert param_units[1][5] == u.dimensionless_unscaled
    assert param_units[1][6] == u.dimensionless_unscaled
    assert param_units[1][7] == u.dimensionless_unscaled


def test_JointFitter__assign_param_units():
    """Test the `JointFitter` class method: ``_assign_param_units``."""
    gu1 = ThermalEmission(temperature=10 << u.MK, emission_measure=1 << u.cm**-3)

    # for one model
    units = [[u.MK, u.cm**-3, None, None, None, None, None, None, None]]

    param = JointFitter._assign_param_units(units, 0, gu1)

    assert param[0] == gu1.temperature.value << gu1.temperature.unit
    assert param[1] == gu1.emission_measure.value << gu1.emission_measure.unit
    assert param[2] == gu1.mg.value
    assert param[3] == gu1.al.value
    assert param[4] == gu1.si.value
    assert param[5] == gu1.s.value
    assert param[6] == gu1.ar.value
    assert param[7] == gu1.ca.value
    assert param[8] == gu1.fe.value


def test_JointFitter__evaluate_models():
    """Test the `JointFitter` class method: ``_evaluate_models``."""
    gu1 = ThermalEmission(temperature=10 << u.MK, emission_measure=1 << u.cm**-3)
    units = [u.MK, u.cm**-3, None, None, None, None, None, None, None]
    x1 = np.arange(3, 9, 0.1)

    fit_joint = JointFitter(optimizer=SLSQP, statistic=leastsquare)
    ev = fit_joint._evaluate_models([gu1], [x1], parameter_units=[units])

    np.allclose(ev[0], gu1(x1))

    evs = fit_joint._evaluate_models([gu1, gu1], [x1, x1], parameter_units=[units, units])
    np.allclose(evs[0], gu1(x1))
    np.allclose(evs[1], gu1(x1))


def test_fitter_to_model_params_array():
    "Test the ``fitter_to_model_params_array`` function."
    guess_amplitude = 5
    guess_mean = 6
    guess_stddev = 0.3
    g1 = models.Gaussian1D(amplitude=guess_amplitude, mean=guess_mean, stddev=guess_stddev)
    # make sure parameters can be reconstructed
    parameters = fitter_to_model_params_array(g1, [3.14, 1.67], True, fit_param_indices=[0, 2], model_list=None)
    np.allclose(parameters, [3.14, g1.parameters[1], 1.67])

    # now try with a model list so we can tie parameters from one model onto another
    g2 = models.Gaussian1D()
    model_list = [g1, g2]
    g2.amplitude.tied = lambda models: models[0].amplitude
    g2_parameters = fitter_to_model_params_array(g2, [137], True, fit_param_indices=[2], model_list=model_list)
    np.allclose(g2_parameters, [g2.amplitude.value, g1.mean.value, 137])


def test_ScipyMinimizeJointFitter_objective_function():
    """Test `ScipyMinimizeJointFitter.objective_function`."""

    fit_joint = ScipyMinimizeJointFitter()

    # one model and data source
    fps = [3.14, 1.67]
    data = np.array([1, 5, 1])
    args = ([models.Gaussian1D()], ([[0, 1, 2]], [data]))
    weights = [1 / data]
    jfit_param_indices = [[1, 2]]
    parameter_units = [[None, None, None]]
    val = fit_joint.objective_function(
        fps, *args, weights=weights, jfit_param_indices=jfit_param_indices, parameter_units=parameter_units
    )
    assert_allclose([val], [4.889649292435376])

    # now try multiple models and data sources
    fps2 = [3.14, 1.67, 3.14, 1.67]
    args2 = ([models.Gaussian1D(), models.Gaussian1D()], ([[0, 1, 2], [0, 1, 2]], [data, data]))
    weights2 = [1 / data, 1 / data]
    jfit_param_indices2 = [[1, 2], [1, 2]]
    parameter_units2 = [[None, None, None], [None, None, None]]
    val2 = fit_joint.objective_function(
        fps2, *args2, weights=weights2, jfit_param_indices=jfit_param_indices2, parameter_units=parameter_units2
    )
    assert_allclose([val2, val2], [9.779298584870752, 2 * val])


def test_ScipyMinimizeJointFitter_unitless():
    """Test the `ScipyMinimizeJointFitter` class with unitless models."""
    # define some synthetic data
    x1 = np.linspace(1.0, 6.0, 200)
    x2 = np.linspace(4.0, 10.0, 200)
    data_amplitude = 5.6
    data_mean = 5.1
    data_stddev1 = 0.4
    data_stddev2 = 0.2
    y1 = simulate_gaussian_data_source(x1, data_amplitude, data_mean, data_stddev1)
    y2 = simulate_gaussian_data_source(x2, data_amplitude, data_mean, data_stddev2)

    # define models
    guess_amplitude = 5
    guess_mean = 6
    guess_stddev1 = 0.3
    guess_stddev2 = 0.5
    g1 = models.Gaussian1D(amplitude=guess_amplitude, mean=guess_mean, stddev=guess_stddev1)
    g2 = models.Gaussian1D(stddev=guess_stddev2)

    # tie relevant parameters
    g2.amplitude.tied = lambda models: models[0].amplitude
    g2.mean.tied = lambda models: models[0].mean

    # initialise the fitter, need to pass optimizer and statistic to avoid error
    fit_joint = ScipyMinimizeJointFitter()

    # check if the correct parameters are extracted for fitting
    extracted_param_info = fit_joint.joint_model_to_fit_params([g1, g2])
    # check fittable parameter values are good
    assert_allclose(extracted_param_info[0], [g1.amplitude.value, g1.mean.value, g1.stddev.value, g2.stddev.value])
    # check fittable parameter indices are good
    assert_allclose(extracted_param_info[1][0], [0, 1, 2])
    assert_allclose(extracted_param_info[1][1], [2])
    # check fittable parameter bounds are good
    assert_allclose(extracted_param_info[2][0], (-np.inf, -np.inf, 1.1754943508222875e-38, 1.1754943508222875e-38))
    assert_allclose(extracted_param_info[2][1], (np.inf, np.inf, np.inf, np.inf))

    # run the fits to the data and get a copy of the resulting models
    g12 = fit_joint(g1, x1, y1, g2, x2, y2)

    # check fitted solution is working
    assert_allclose(g12[0].parameters, [data_amplitude, data_mean, data_stddev1], atol=1e-1)
    assert_allclose(g12[1].parameters, [data_amplitude, data_mean, data_stddev2], atol=1e-1)


def test_ScipyMinimizeJointFitter_forward_tying():
    """Test the `ScipyMinimizeJointFitter` class by forward tying parameters."""
    # define some synthetic data
    x1 = np.linspace(1.0, 6.0, 200)
    x2 = np.linspace(4.0, 10.0, 200)
    data_amplitude = 5.6
    data_mean = 5.1
    data_stddev1 = 0.4
    data_stddev2 = 0.2
    y1 = simulate_gaussian_data_source(x1, data_amplitude, data_mean, data_stddev1)
    y2 = simulate_gaussian_data_source(x2, data_amplitude, data_mean, data_stddev2)

    # define models
    guess_amplitude = 5
    guess_mean = 6
    guess_stddev1 = 0.3
    guess_stddev2 = 0.5
    g1 = models.Gaussian1D(amplitude=guess_amplitude, mean=guess_mean, stddev=guess_stddev1)
    g2 = models.Gaussian1D(stddev=guess_stddev2)

    # forward tie relevant parameters
    g1.amplitude.tied = lambda models: models[1].amplitude
    g2.mean.tied = lambda models: models[0].mean

    # initialise the fitter, need to pass optimizer and statistic to avoid error

    # initialise the fitter, need to pass optimizer and statistic to avoid error
    fit_joint = ScipyMinimizeJointFitter()

    # check if the correct parameters are extracted for fitting after forward tying
    extracted_param_info = fit_joint.joint_model_to_fit_params([g1, g2])
    # check fittable parameter values are good
    assert_allclose(extracted_param_info[0], [g1.mean.value, g1.stddev.value, g2.amplitude.value, g2.stddev.value])
    # check fittable parameter indices are good
    assert_allclose(extracted_param_info[1][0], [1, 2])
    assert_allclose(extracted_param_info[1][1], [0, 2])
    # check fittable parameter bounds are good
    assert_allclose(extracted_param_info[2][0], (-np.inf, 1.1754943508222875e-38, -np.inf, 1.1754943508222875e-38))
    assert_allclose(extracted_param_info[2][1], (np.inf, np.inf, np.inf, np.inf))

    # run the fits to the data and get a copy of the resulting models
    g12 = fit_joint(g1, x1, y1, g2, x2, y2)

    # check fitted solution is working
    assert_allclose(g12[0].parameters, [data_amplitude, data_mean, data_stddev1], atol=1e-1)
    assert_allclose(g12[1].parameters, [data_amplitude, data_mean, data_stddev2], atol=1e-1)


def test_ScipyMinimizeJointFitter_units_bounds_fixing_weights():
    """Test the `ScipyMinimizeJointFitter` class with models containing units.

    This also tests:
    - Bounds enforcement since temperature needs bounds.
    - Fixing parameters since we don't want to fit all the abundances too.
    - Adding weights to the fitting since the data-model values are so large.

    """
    energy_edges1 = np.arange(1.6, 30, 0.2) << u.keV
    data_temp = 15 << u.MK
    data_em1 = 5 << u.cm**-3  # measured in 1e49
    data_y1 = simulate_thermal_data_source(energy_edges1, data_temp, data_em1)

    energy_edges2 = np.arange(4, 15, 0.1) << u.keV
    data_em2 = 0.3 << u.cm**-3  # measured in 1e49
    data_y2 = simulate_thermal_data_source(energy_edges2, data_temp, data_em2)

    # get models
    gjf1 = ThermalEmission(
        temperature=10 << u.MK,
        emission_measure=1 << u.cm**-3,
        bounds={"temperature": (5 << u.MK, 20 << u.MK)},
        fixed={"mg": True, "al": True, "si": True, "s": True, "ar": True, "ca": True, "fe": True},
    )

    gjf2 = ThermalEmission(
        emission_measure=0.1 << u.cm**-3,
        fixed={"mg": True, "al": True, "si": True, "s": True, "ar": True, "ca": True, "fe": True},
    )

    # tie temperatures together
    gjf2.temperature.tied = lambda models: models[0].temperature

    # set up the base joint fitter
    fit_joint = ScipyMinimizeJointFitter()

    g12 = fit_joint(
        gjf1,
        energy_edges1,
        data_y1,
        gjf2,
        energy_edges2,
        data_y2,
        fkwarg={
            "weights": [
                1 / data_y1**2,
                1 / data_y2**2,
            ]
        },
    )

    # check fitted solution is working
    assert_allclose(g12[0].parameters, [data_temp.value, data_em1.value, *gjf1.parameters[2:]], atol=5e-1)
    assert_allclose(g12[1].parameters, [data_temp.value, data_em2.value, *gjf2.parameters[2:]], atol=5e-1)
