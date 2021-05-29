import astropy.units as u
import numpy as np

from sunxspex.thermal import continuum

energy_edges = np.arange(3, 28.5, 0.5) * u.keV

# Define expected continuum spectrum taken from chianti_kev_cont IDL routine.
expected_032805keV_6MK_1e44_1AU = np.array([
    0.435291,    0.144041,    0.0484909,   0.0164952,   0.00567440,  0.00196711,  0.000685918,
    0.000240666, 8.48974e-05, 3.00666e-05, 1.07070e-05, 3.81792e-06, 1.36722e-06, 4.91871e-07,
    1.76929e-07, 6.39545e-08, 2.30593e-08, 8.38502e-09, 3.04271e-09, 1.10372e-09, 4.03399e-10,
    1.47199e-10, 5.36175e-11, 1.96076e-11, 7.20292e-12, 2.64275e-12, 9.68337e-13, 3.54305e-13,
    1.30357e-13, 4.81556e-14, 1.77739e-14, 6.55418e-15, 2.41451e-15, 8.88565e-16, 3.26635e-16,
    1.20928e-16, 4.49286e-17, 1.66830e-17, 6.19118e-18, 2.29618e-18, 8.51055e-19, 3.15220e-19,
    1.16669e-19, 4.31491e-20, 1.59455e-20, 5.90236e-21, 2.20614e-21, 8.24339e-22, 3.07924e-22,
    1.14983e-22]) * u.ph / u.cm**2 / u.s / u.keV


def test_continuum_emission_suncoronal_NoRelAbund():
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    output = continuum.continuum_emission(
        energy_edges, 6 * u.MK, 1e44/u.cm**3, abundance_type="sun_coronal",
        observer_distance=(1 * u.AU).to(u.cm))
    np.testing.assert_allclose(output, expected_032805keV_6MK_1e44_1AU, rtol=0.03)
