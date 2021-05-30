import astropy.units as u
import numpy as np

from sunxspex.thermal import continuum_emission, line_emission, thermal_emission

energy_edges = np.arange(3, 28.5, 0.5) * u.keV
default_T = 6 * u.MK
default_EM = 1e44 / u.cm**3

# Define expected output from f_vth.pro, i.e. expected combined line and continuum emission.
expected_fvth_032805_6MK_1e44_1AU = np.array([
   0.49978617,    0.15907305,    0.052894916,   0.017032871,   0.0056818775,   0.0019678525,
   0.00071765773, 0.00026049651, 8.5432410e-05, 3.0082287e-05, 1.0713027e-05, 3.8201970e-06,
   1.3680836e-06, 4.9220034e-07, 1.7705435e-07, 6.4002016e-08, 2.3077330e-08, 8.3918881e-09,
   3.0453193e-09, 1.1047097e-09, 4.0377532e-10, 1.4734168e-10, 5.3671578e-11, 1.9628120e-11,
   7.2107064e-12, 2.6457057e-12, 9.6945607e-13, 3.5472713e-13, 1.3051763e-13, 4.8216642e-14,
   1.7797136e-14, 6.5629896e-15, 2.4178513e-15, 8.8982728e-16, 3.2711010e-16, 1.2110889e-16,
   4.4997199e-17, 1.6709174e-17, 6.2011332e-18, 2.2999536e-18, 8.5248536e-19, 3.1576185e-19,
   1.1687443e-19, 4.3226546e-20, 1.5974677e-20, 5.9133793e-21, 2.2103457e-21, 8.2594126e-22,
   3.0853276e-22, 1.1521560e-22]) * u.ph / u.cm**2 / u.s / u.keV

# Define expected continuum spectrum taken from chianti_kev_cont IDL routine.
expected_continuum_032805keV_6MK_1e44_1AU = np.array([
    0.435291,    0.144041,    0.0484909,   0.0164952,   0.00567440,  0.00196711,  0.000685918,
    0.000240666, 8.48974e-05, 3.00666e-05, 1.07070e-05, 3.81792e-06, 1.36722e-06, 4.91871e-07,
    1.76929e-07, 6.39545e-08, 2.30593e-08, 8.38502e-09, 3.04271e-09, 1.10372e-09, 4.03399e-10,
    1.47199e-10, 5.36175e-11, 1.96076e-11, 7.20292e-12, 2.64275e-12, 9.68337e-13, 3.54305e-13,
    1.30357e-13, 4.81556e-14, 1.77739e-14, 6.55418e-15, 2.41451e-15, 8.88565e-16, 3.26635e-16,
    1.20928e-16, 4.49286e-17, 1.66830e-17, 6.19118e-18, 2.29618e-18, 8.51055e-19, 3.15220e-19,
    1.16669e-19, 4.31491e-20, 1.59455e-20, 5.90236e-21, 2.20614e-21, 8.24339e-22, 3.07924e-22,
    1.14983e-22]) * u.ph / u.cm**2 / u.s / u.keV

dist_scaled_spectrum_unit = u.ph / u.cm**2 / u.s / u.keV
dist_unscaled_spectrum_unit = u.ph / u.s / u.keV / u.sr
# Output spectrum from SSW given energy edges from 3 - 28 keV in 0.5keV bins with
# no relative abundances and not scaled to the observer distance.
expected_line_spectrum_E032805_6MK_EM1e44_RelAbunFe2_NotObserverScaled = np.array([
    1.6399155e+25,   3.1898577e+24,   9.9494481e+23,   1.5177821e+23,   5.2238873e+21,   1.1521569e+17,
    1.5242971e+22,   9.5837332e+21,   2.2554210e+20,   3.7478687e+15,   1.2419604e+15,   6.9386484e+13,
    4.3706957e+09,   2973542.5,       234939.45,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000]) * dist_unscaled_spectrum_unit

expected_line_spectrum_E032805_6MK_EM1e44_NoRelAbun_earth_20190522 = np.array([
    0.073248975,     0.014247916,     0.0044440511,    0.00067793718,   2.3333176e-05,   2.5751346e-10,
    3.4042361e-05,   2.1403499e-05,   5.0370664e-07,   8.3715751e-12,   2.7737142e-12,   1.5496721e-13,
    1.9522280e-17,   1.3281716e-20,   1.0493879e-21,   0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,       0.0000000,
    0.0000000,       0.0000000]) * dist_scaled_spectrum_unit


def test_continuum_emission():
    output = continuum_emission(energy_edges, [6, 6]*u.MK, default_EM, abundance_type="sun_coronal",
                                observer_distance=(1 * u.AU).to(u.cm))
    np.testing.assert_allclose(output[0], expected_continuum_032805keV_6MK_1e44_1AU, rtol=0.03)


def test_line_emission():
   output = line_emission(energy_edges, default_T, default_EM, abundance_type="sun_coronal",
                          observer_distance=(1 * u.AU).to(u.cm))
   expected = expected_line_spectrum_E032805_6MK_EM1e44_NoRelAbun_earth_20190522.to_value(output.unit)
   np.testing.assert_allclose(output.value,
                              expected,
                              rtol=0.03,
                              atol=1e-30)

def test_thermal_emission():
    output = thermal_emission(energy_edges, default_T, default_EM, abundance_type="sun_coronal",
                          observer_distance=(1 * u.AU).to(u.cm))
    expected = expected_fvth_032805_6MK_1e44_1AU.to_value(output.unit)
    np.testing.assert_allclose(output.value,
                              expected,
                              rtol=0.03,
                              atol=1e-30)
