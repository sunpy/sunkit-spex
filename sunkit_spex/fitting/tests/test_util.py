"""
This module contains package tests for the util functions.
"""

import numpy as np

from sunkit_spex.fitting.metrics import util


def test_weights_to_error():
    data0 = np.arange(1, 4)
    np.testing.assert_allclose(util.weights_to_error(data0), 1/np.sqrt(data0))

def test_error_to_weights():
    data0 = np.arange(1, 4)
    np.testing.assert_allclose(util.error_to_weights(data0), 1/(data0**2))
