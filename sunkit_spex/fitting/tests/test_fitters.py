"""
This module contains package tests for the fitters functions.
"""

import numpy as np
from numpy.testing import assert_allclose

from astropy.modeling import models
from astropy.modeling.optimizers import SLSQP
from astropy.modeling.statistic import leastsquare

from sunkit_spex.data.simulated_data import simulate_gaussian_data_source
from sunkit_spex.fitting.fitters import JointFitter


def test_JointFitter():
    """Test the `JointFitter` class."""
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
    fit_joint = JointFitter(optimizer=SLSQP, statistic=leastsquare)

    # check if the correct parameters are extracted for fitting
    extracted_param_info = fit_joint.joint_model_to_fit_params([g1, g2])
    # check fittable parameter values are good
    assert_allclose(extracted_param_info[0], [guess_amplitude, guess_mean, guess_stddev1, guess_stddev2])
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
