"""
This module contains package tests for the statistics functions.
"""

import numpy as np

from sunkit_spex.fitting.statistics.gaussian import chi_squared


def test_chi_squared():
    sim_data0 = np.array([0])
    sim_model0 = sim_data0
    chi_s0 = chi_squared(sim_data0, sim_model0)

    sim_data1 = np.array([1])
    sim_model1 = sim_data1
    chi_s1 = chi_squared(sim_data1, sim_model1)

    sim_data2 = np.array([1, 2, 3])
    sim_model2 = sim_data2
    chi_s2 = chi_squared(sim_data2, sim_model2)

    sim_data3 = np.array([1, 2, 3])
    sim_model3 = sim_data3[::-1]
    chi_s3 = chi_squared(sim_data3, sim_model3)

    assert chi_s0 == 0
    assert chi_s1 == 0
    assert chi_s2 == 0
    assert chi_s3 == 8
