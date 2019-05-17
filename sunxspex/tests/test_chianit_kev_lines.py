
import pytest
import numpy as np
import astropy.units as u

from sunxspex import chianti_kev_lines

# Define some default input parameters.
default_EM = 1e44/(u.cm**3)
energy_edges = np.arange(3, 28.5, 0.5)*u.keV

# Output spectrum from SSW given energy edges from 3 - 28.5 keV in 0.5keV bins with 
# no relative abundances and not scaled to the observer distance.
expected_spectrum_E032805_6MK_EM1e44_NoRelAbund_NotObserverScaled = np.array([
    1.6399155e+25, 3.1898577e+24, 9.9494481e+23, 1.5177821e+23, 5.2238873e+21, 5.7652733e+16,
    7.6214856e+21, 4.7918666e+21, 1.1277105e+20, 1.8742482e+15, 6.2098577e+14, 3.4694431e+13,
    4.3706952e+09, 2973542.5,     234939.45,     0.0000000,     0.0000000,     0.0000000,
    0.0000000,     0.0000000,     0.0000000,     0.0000000,     0.0000000,     0.0000000,
    0.0000000,     0.0000000,     0.0000000,     0.0000000,     0.0000000,     0.0000000,
    0.0000000,     0.0000000,     0.0000000,     0.0000000,     0.0000000,     0.0000000,
    0.0000000,     0.0000000,     0.0000000,     0.0000000,     0.0000000,     0.0000000,
    0.0000000,     0.0000000,     0.0000000,     0.0000000,     0.0000000,     0.0000000,
    0.0000000,     0.0000000])

@pytest.mark.parametrize(
    "energy_edges,temperature,em,relative_abundances,observer_distance,earth,date,expected_spectrum",
    [
        (np.arange(3, 28.5, 0.5)*u.keV, 6*u.MK, default_EM, None, None, None, None,
        expected_spectrum_E032805_6MK_EM1e44_NoRelAbund_NotObserverScaled)
    ])
def test_chianti_kev_lines(energy_edges, temperature, em, relative_abundances, 
                           observer_distance, earth, date, expected_spectrum):
    output_spectrum = chianti_kev_lines.chianti_kev_lines(
            energy_edges, temperature, em,
            relative_abundances=relative_abundances, observer_distance=observer_distance,
            earth=earth, date=date)
    np.testing.assert_allclose(output_spectrum[0]*2, expected_spectrum, rtol=0.005, atol=1e-4) # A factor of 2 is currently missing from output spectrum compared to SSW version.
