import numpy as np
from numpy.testing import assert_array_equal

import astropy.units as u

from sunkit_spex.spectrum.spectrum import Spectrum


def test_spectrum_bin_edges():
    spec = Spectrum(np.arange(1, 11)*u.watt, spectral_axis=np.arange(1, 12)*u.keV)
    assert_array_equal(spec._spectral_axis, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5] * u.keV)


def test_spectrum_bin_centers():
    spec = Spectrum(np.arange(1, 11)*u.watt, spectral_axis=(np.arange(1, 11) - 0.5) * u.keV)
    assert_array_equal(spec._spectral_axis, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5] * u.keV)
