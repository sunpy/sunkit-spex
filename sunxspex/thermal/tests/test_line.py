
import astropy.units as u
import numpy as np
import pytest

from sunxspex.thermal.line import line_emission

energy_edges = np.arange(3, 28.5, 0.5) * u.keV
default_T = 6 * u.MK
default_EM = 1e44 / u.cm**3

dist_scaled_spectrum_unit = u.ph / u.cm**2 / u.s / u.keV
dist_unscaled_spectrum_unit = u.ph / u.s / u.keV / u.sr
# Output spectrum from SSW given energy edges from 3 - 28.5 keV in 0.5keV bins with
# no relative abundances and not scaled to the observer distance.

expected_spectrum_E032805_6MK_EM1e44_RelAbunFe2_NotObserverScaled = np.array([
    1.6399155e+25,   3.1898577e+24,   9.9494481e+23,   1.5177821e+23,   5.2238873e+21,   1.1521569e+17,
    1.5242971e+22,   9.5837332e+21,   2.2554210e+20,   3.7478687e+15,   1.2419604e+15,   6.9386484e+13,
    4.3706957e+09,   2973542.5,       234939.45,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000]) * dist_unscaled_spectrum_unit

expected_spectrum_E032805_6MK_EM1e44_NoRelAbun_earth_20190522 = np.array([
    0.073248975,     0.014247916,     0.0044440511,    0.00067793718,   2.3333176e-05,   2.5751346e-10,
    3.4042361e-05,   2.1403499e-05,   5.0370664e-07,   8.3715751e-12,   2.7737142e-12,   1.5496721e-13,
    1.9522280e-17,   1.3281716e-20,   1.0493879e-21,   0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000]) * dist_scaled_spectrum_unit

def test_line_NoRelAbun():
   output = line_emission(energy_edges, default_T, default_EM)
   expected = expected_spectrum_E032805_6MK_EM1e44_NoRelAbun_earth_20190522.to_value(output.unit)
   np.testing.assert_allclose(output.value,
                              expected,
                              rtol=0.03,
                              atol=1e-30)
