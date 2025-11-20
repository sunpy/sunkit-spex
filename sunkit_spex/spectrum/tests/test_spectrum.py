from operator import add, mul, sub, truediv

import numpy as np
import pytest
from ndcube import NDCube
from ndcube.extra_coords import QuantityTableCoordinate, TimeTableCoordinate
from ndcube.wcs.wrappers import CompoundLowLevelWCS
from numpy.testing import assert_array_equal

import astropy.units as u
from astropy.time import Time
from astropy.wcs import WCS

from sunkit_spex.spectrum.spectrum import SpectralAxis, Spectrum


def test_spectrum_quantity_bin_edges():
    spec = Spectrum(np.arange(1, 11) * u.watt, spectral_axis=np.arange(1, 12) * u.keV)
    assert_array_equal(spec._spectral_axis, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5] * u.keV)


def test_spectrum_quantity_bin_centers():
    spec = Spectrum(np.arange(1, 11) * u.watt, spectral_axis=(np.arange(1, 11) + 0.5) * u.keV)
    assert_array_equal(spec._spectral_axis, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5] * u.keV)


def test_spectrum_spectral_axis_bin_edges():
    spec_axis = SpectralAxis(np.arange(1, 12) * u.keV, bin_specification="edges")
    spec = Spectrum(np.arange(1, 11) * u.watt, spectral_axis=spec_axis)
    assert_array_equal(spec._spectral_axis, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5] * u.keV)


def test_spectrum_spectral_axis_bin_centers():
    spec_axis = SpectralAxis((np.arange(1, 11) + 0.5) * u.keV, bin_specification="centers")
    spec = Spectrum(np.arange(1, 11) * u.watt, spectral_axis=spec_axis)
    assert_array_equal(spec._spectral_axis, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5] * u.keV)


def test_spectrum_from_ndcube_wcs():
    header = {
        "CTYPE1": "TIME    ",
        "CUNIT1": "s",
        "CDELT1": 10,
        "CRPIX1": 0,
        "CRVAL1": 0,
        "CTYPE2": "ENER    ",
        "CUNIT2": "keV",
        "CDELT2": 1,
        "CRPIX2": 0,
        "CRVAL2": 1,
        "DATEREF": "2020-01-01T00:00:00",
    }
    wcs = WCS(header=header)
    shape = (10, 5)
    wcs.array_shape = shape
    data = np.arange(np.prod(shape), dtype=int).reshape(shape)
    cube = NDCube(data.T, wcs=wcs)
    with pytest.raises(ValueError, match="Input NDCube missing unit.*"):
        Spectrum(cube)

    cube = NDCube(data, wcs=wcs, unit=u.ph)
    spec = Spectrum(cube)
    assert isinstance(spec, Spectrum)


def test_spectrum_from_cube_wcs_tab():
    energy = np.arange(1, 11) * u.keV
    energy_coord = QuantityTableCoordinate(energy, names="energy", physical_types="em.energy")
    data = np.random.rand(len(energy))
    cube = NDCube(data=data, wcs=energy_coord.wcs, unit=u.ph)
    spec = Spectrum(cube)
    assert False


def test_spectrum_spectra_axis_detection():
    energy = np.arange(1, 11) * u.keV
    energy_coord = QuantityTableCoordinate(energy, names="energy", physical_types="em.energy")
    times = Time("2020-01-01") + 5 * np.arange(0, 11) * u.s
    time_coord = TimeTableCoordinate(times, names="time", physical_types="time")
    time_energy_wcs = (time_coord & energy_coord).wcs
    data = np.arange(5 * 10).reshape(5, 10)
    spec1 = Spectrum(data * u.ph, wcs=time_energy_wcs)

    energy_energy_wcs = (energy_coord & time_coord).wcs
    data = np.arange(5 * 10).reshape(10, 5)
    spec2 = Spectrum(data * u.ph, wcs=energy_energy_wcs)

    assert False


def test_spectrum_from_cubs_wcs_norm_tab():
    header = {
        "CTYPE1": "TIME    ",
        "CUNIT1": "s",
        "CDELT1": 10,
        "CRPIX1": 0,
        "CRVAL1": 0,
        "DATEREF": "2020-01-01T00:00:00",
    }
    time_wcs = WCS(header=header)
    energy = np.arange(10) * u.keV
    energy_coord = QuantityTableCoordinate(energy, names="energy", physical_types="em.energy")
    comp_wcs = CompoundLowLevelWCS(time_wcs, energy_coord.wcs)
    cube = NDCube(np.arange(10 * 5).reshape(10, 5), unit=u.ph, wcs=comp_wcs)
    spec = Spectrum(cube)
    assert False


def test_slice():
    spec = Spectrum(np.arange(1, 11) * u.watt, spectral_axis=(np.arange(1, 11) + 0.5) * u.keV)
    sliced_spec = spec[5:]
    assert sliced_spec.shape == (5,)
    assert sliced_spec.spectral_axis.shape == (5,)


@pytest.mark.parametrize(
    "op, value, res",
    [
        (add, 2 * u.W, np.arange(1.0, 11) + 2),
        (sub, 2 * u.W, np.arange(1.0, 11) - 2),
        (mul, 2, np.arange(1.0, 11) * 2),
        (truediv, 2 * u.W, np.arange(1.0, 11) / 2),
    ],
)
def test_arithmetic_operators(op, value, res):
    spec = Spectrum(np.arange(1, 11) * u.watt, spectral_axis=(np.arange(1, 11) + 0.5) * u.keV)
    res_spec = op(spec, value)
    assert_array_equal(res_spec.data, res)
