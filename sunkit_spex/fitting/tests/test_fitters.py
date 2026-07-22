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
    amplitude = 5.6
    mean = 5.1
    stddev1 = 0.4
    stddev2 = 0.2
    y1 = simulate_gaussian_data_source(x1, amplitude, mean, stddev1)
    y2 = simulate_gaussian_data_source(x2, amplitude, mean, stddev2)

    # define models
    g1 = models.Gaussian1D(amplitude=5, mean=6, stddev=0.3)
    g2 = models.Gaussian1D(stddev=0.5)

    # tie relevant parameters
    g2.amplitude.tied = lambda models: models[0].amplitude
    g2.mean.tied = lambda models: models[0].mean

    # initialise the fitter, need to pass optimizer and statistic to avoid error
    fit_joint = JointFitter(optimizer=SLSQP, statistic=leastsquare)

    # run the fits to the data and get a copy of the resulting models
    g12 = fit_joint(g1, x1, y1, g2, x2, y2)

    # check fitted solution is working
    assert_allclose(g12[0].parameters, [amplitude, mean, stddev1], atol=1e-1)
    assert_allclose(g12[1].parameters, [amplitude, mean, stddev2], atol=1e-1)
