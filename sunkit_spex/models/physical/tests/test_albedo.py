import numpy as np
import pytest

import astropy.units as u
from astropy.modeling.powerlaws import PowerLaw1D
from astropy.units import UnitsError

from sunkit_spex.models.physical.albedo import Albedo, get_albedo_matrix


def test_get_albedo_matrix():
    e = np.linspace(4, 600, 597) * u.keV
    theta = 0 * u.deg
    albedo_matrix = get_albedo_matrix(e, theta=theta)
    assert albedo_matrix[0, 0] == 0.006154127884656191
    assert albedo_matrix[298, 298] == 5.956079300274577e-24
    assert albedo_matrix[-1, -1] == 2.3302891436400413e-27


def test_get_albedo_matrix_bad_energy():
    e = [1, 4, 10, 20] * u.keV
    with pytest.raises(ValueError, match=r"Supported energy range 3.*"):
        get_albedo_matrix(e, theta=0 * u.deg)


def test_get_albedo_matrix_bad_angle():
    e = [4, 10, 20] * u.keV
    with pytest.raises(UnitsError, match=r".*Argument 'theta' to function 'get_albedo_matrix'.*'deg'."):
        get_albedo_matrix(e, theta=91 * u.m)
    with pytest.raises(ValueError, match=r"Theta must be between -90 and 90 degrees.*"):
        get_albedo_matrix(e, theta=-91 * u.deg)


def test_albedo_model():
    e_edges = np.linspace(10, 300, 10) * u.keV
    e_centers = e_edges[0:-1] + (0.5 * np.diff(e_edges))
    source = PowerLaw1D(amplitude=100 * u.ph, x_0=10 * u.keV, alpha=4)
    observed = source | Albedo(energy_edges=e_edges)
    observed(e_centers)
