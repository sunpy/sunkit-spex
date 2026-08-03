"""
This module contains package tests for the fitters functions.
"""

import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u
from astropy.modeling import models
from astropy.modeling.optimizers import SLSQP
from astropy.modeling.statistic import leastsquare

from sunkit_spex.data.simulated_data import simulate_gaussian_data_source, simulate_thermal_data_source
from sunkit_spex.fitting.fitters import JointFitter, ScipyMinimizeJointFitter
from sunkit_spex.models.physical.thermal import ThermalEmission


def test_JointFitter_initial_values():
    """Test the `JointFitter` class obtains the correct initial values."""

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
