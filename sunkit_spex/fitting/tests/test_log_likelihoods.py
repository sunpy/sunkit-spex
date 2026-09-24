"""
This module contains package tests for the log-likelihood functions.
"""

import numpy as np
import pytest

from sunkit_spex.fitting.metrics.log_likelihoods import gaussian, gaussian_maximum


def test_gaussian_maximum():
    sim_data0 = np.array([0])
    sim_data_weights0 = sim_data0
    # expect a division by zero here since error will be zero, resulting in -inf
    with pytest.warns(RuntimeWarning):
        gmax_s0 = gaussian_maximum(sim_data0, sim_data_weights0)

    sim_data1 = np.array([1])
    sim_data_weights1 = sim_data1
    gmax_s1 = gaussian_maximum(sim_data1, sim_data_weights1)

    sim_data2 = np.array([1, 2, 3])
    sim_data_weights2 = sim_data2
    gmax_s2 = gaussian_maximum(sim_data2, sim_data_weights2)

    assert gmax_s0 == -np.inf
    assert np.allclose([gmax_s1], [-0.9189385332046727])
    assert np.allclose([gmax_s2], [-5.582807594999972])


def test_gaussian():
    sim_data0 = (np.array([0]),)
    sim_data_weights0 = (sim_data0[0],)
    sim_model0 = (sim_data0[0],)
    # expect a division by zero here since error will be zero, resulting in -inf
    with pytest.warns(RuntimeWarning):
        gaussian_s0 = gaussian(sim_data0, sim_model0, sim_data_weights0)

    sim_data1 = (np.array([1, 2, 3]),)
    sim_data_weights1 = (sim_data1[0],)
    sim_model1 = (sim_data1[0],)
    gaussian_s1 = gaussian(sim_data1, sim_model1, sim_data_weights1)

    sim_data2 = (np.array([1, 2, 3]),)
    sim_data_weights2 = (sim_data2[0],)
    sim_model2 = (sim_data2[0][::-1],)
    gaussian_s2 = gaussian(sim_data2, sim_model2, sim_data_weights2)

    assert gaussian_s0 == gaussian_maximum(sim_data0[0], sim_data_weights0[0])
    assert gaussian_s1 == gaussian_maximum(sim_data1[0], sim_data_weights1[0])
    assert gaussian_s2 == gaussian_maximum(sim_data2[0], sim_data_weights2[0]) - 8
