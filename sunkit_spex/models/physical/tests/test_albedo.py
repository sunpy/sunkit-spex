import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.modeling.powerlaws import PowerLaw1D
from astropy.units import UnitsError

from sunkit_spex.models.physical.albedo import Albedo, get_albedo_matrix


def test_get_albedo_matrix():
    e = (np.arange(597) + 4) * u.keV
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


def test_albedo_idl():
    """
    IDL Code to generate values used below

    e_ph = dindgen(11)*2 + 10
    ph_edges = get_edges(e_ph)
    eye = identity(10)
    albedo = drm_correct_albedo(theta=45.0d, ani=1.0d, drm=eye, ph=ph_edges.EDGES_2)
    spec_in = ph_edges.mean^(-2)
    spec_out = albedo # spec_in
    """
    idl_spec_in = [
        0.0082644628099173556,
        0.0059171597633136093,
        0.0044444444444444444,
        0.0034602076124567475,
        0.0027700831024930748,
        0.0022675736961451248,
        0.0018903591682419660,
        0.0016000000000000001,
        0.0013717421124828531,
        0.0011890606420927466,
    ]
    idl_spec_out = [
        0.0095705470260438689,
        0.0076066757846289723,
        0.0063536931853246902,
        0.0055135516253148236,
        0.0045460615861343291,
        0.0037628884235374896,
        0.0031366857209791871,
        0.0025179715230395604,
        0.0019460260612382950,
        0.0013136881009379996,
    ]

    e_ph = np.arange(11) * 2 + 10
    albedo = Albedo(energy_edges=e_ph * u.keV, theta=45 * u.deg)
    e_c = e_ph[:-1] + 0.5 * np.diff(e_ph)
    spec_in = e_c**-2
    spec_out = albedo(spec_in[:])
    assert_allclose(idl_spec_in, spec_in)
    assert_allclose(idl_spec_out, spec_out)
