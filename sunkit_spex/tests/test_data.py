"""
This module contains package tests for the data functions.
"""

import numpy as np
from numpy.testing import assert_allclose

from sunkit_spex.data.simulated_data import simulate_square_response_matrix


def test_simulate_square_response_matrix():
    """Ensure `simulate_square_response_matrix` behaviour does not change."""
    array0 = simulate_square_response_matrix(0)
    exp_res0 = np.identity(0)

    array1 = simulate_square_response_matrix(1)
    exp_res1 = [[1]]

    array2 = simulate_square_response_matrix(2)
    exp_res2 = [[1, 0], [0.00475727, 0.99524273]]

    array3 = simulate_square_response_matrix(3)
    exp_res3 = [[1, 0, 0.0], [0.00475727, 0.99524273, 0.0], [0.00103306, 0.00412088, 0.99484607]]

    assert_allclose(array0, exp_res0, rtol=1e-3)
    assert_allclose(array1, exp_res1, rtol=1e-3)
    assert_allclose(array2, exp_res2, rtol=1e-3)
    assert_allclose(array3, exp_res3, rtol=1e-3)
