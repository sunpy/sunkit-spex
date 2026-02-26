from operator import add, mul, sub, truediv

import numpy as np
import pytest
from gwcs import coordinate_frames as cf
from ndcube import NDCube
from ndcube.extra_coords import QuantityTableCoordinate, TimeTableCoordinate
from ndcube.wcs.wrappers import CompoundLowLevelWCS
from numpy.testing import assert_array_equal

import astropy.units as u
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from astropy.wcs import WCS

from sunkit_spex.spectrum.spectrum import SpectralAxis, SpectralGWCS, Spectrum, gwcs_from_array

rng = np.random.default_rng()


def test_spectral_gwcs_init_and_copy():
    # Setup dummy transform and frames
    trans = models.Identity(1)
    # Create distinct frames with unique names
    # Usually 'detector' or 'pixel' for input, and 'world' or 'spectral' for output
    input_frame = cf.CoordinateFrame(naxes=1, axes_type=["SPECTRAL"], axes_order=(0,), name="pixel_frame")

    output_frame = cf.CoordinateFrame(naxes=1, axes_type=["SPECTRAL"], axes_order=(0,), name="world_frame")
    sgwcs = SpectralGWCS(
        forward_transform=trans, input_frame=input_frame, output_frame=output_frame, original_unit="Angstrom"
    )

    assert sgwcs.original_unit == "Angstrom"

    # Test shallow copy
    sgwcs_copy = sgwcs.copy()
    assert sgwcs_copy.original_unit == sgwcs.original_unit
    assert sgwcs_copy is not sgwcs

    # Test deep copy
    sgwcs_deepcopy = sgwcs.deepcopy()
    assert sgwcs_deepcopy.original_unit == sgwcs.original_unit
    assert sgwcs_deepcopy is not sgwcs


def test_gwcs_from_array_1d_wavelength():
    wavelengths = np.linspace(4000, 7000, 100) * u.AA
    flux_shape = (100,)

    wcs = gwcs_from_array(wavelengths, flux_shape)

    assert isinstance(wcs, SpectralGWCS)
    assert wcs.output_frame.unit[0] == u.AA
    assert wcs.output_frame.axes_names[0] == "wavelength"

    # Test forward transform (pixel to world)
    assert np.allclose(wcs(0), 4000 << u.AA)
    assert np.allclose(wcs(99), 7000 << u.AA)

    # Test inverse transform (world to pixel)
    assert np.allclose(wcs.invert(4000).value, 0)


def test_gwcs_from_array_3d_cube():
    # 3D cube: (Spatial, Spatial, Spectral) -> (y, x, lambda)
    # In numpy: shape is (ny, nx, nlambda)
    # We want spectral axis to be index 2
    n_lambda = 50
    flux_shape = (10, 20, n_lambda)
    freqs = np.linspace(100, 200, n_lambda) * u.keV

    # Note: spectral_axis_index is relative to numpy shape
    wcs = gwcs_from_array(freqs, flux_shape, spectral_axis_index=2)

    assert wcs.output_frame.naxes == 3
    assert wcs.forward_transform.n_inputs == 3

    # Test mapping: (x, y, lambda_pix) -> (spatial, spatial, freq)
    # GWCS/WCS usually expects (x, y, z) input order
    world = wcs.pixel_to_world(0, 0, 0)  # pixels for x, y, lambda
    assert world[0] == 100 * u.keV
    assert wcs.output_frame.frames[1].axes_names[0] == "energy"


def test_gwcs_from_array_invalid_units():
    data = np.arange(10) * u.Jy  # Flux units are not valid for spectral axis
    with pytest.raises(ValueError, match="Spectral axis units must be one of"):
        gwcs_from_array(data, (10,))


def test_gwcs_from_array_missing_index():
    data = np.linspace(1, 10, 10) * u.m
    # 2D flux but no index provided
    with pytest.raises(ValueError, match="spectral_axis_index must be set"):
        gwcs_from_array(data, (10, 10))


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


def test_spectrum_from_spectrum():
    spec_orig = Spectrum(np.arange(1, 11) * u.watt, spectral_axis=np.arange(1, 12) * u.keV)
    spec_new = Spectrum(spec_orig)
    spec_orig == spec_new


def test_spectrum_unknow_keywords():
    with pytest.raises(ValueError, match=r"Initializer contains unknown arguments."):
        Spectrum(np.arange(1, 11) * u.watt, spectral_axis=np.arange(1, 12) * u.keV, mykeyword="myvalue")


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
        "CRPIX2": 0.5,
        "CRVAL2": 0.0,
        "DATEREF": "2020-01-01T00:00:00",
    }
    wcs = WCS(header=header)
    shape = (10, 5)
    wcs.array_shape = shape

    spec_axis_edges = np.arange(11) * u.keV
    spec_axis_centers = spec_axis_edges[:-1] + np.diff(spec_axis_edges) * 0.5

    data = np.arange(np.prod(shape), dtype=int).reshape(shape)
    cube = NDCube(data.T, wcs=wcs)
    with pytest.raises(ValueError, match=r"Input NDCube missing unit.*"):
        Spectrum(cube)

    cube = NDCube(data, wcs=wcs, unit=u.ph)

    with pytest.raises(ValueError, match="Spectral axis must be specified"):
        Spectrum(cube)

    with pytest.raises(ValueError, match=r"Spectral axis"):
        Spectrum(cube, spectral_axis=spec_axis_edges[1:-1])

    spec = Spectrum(cube, spectral_axis=spec_axis_edges)
    assert isinstance(spec, Spectrum)
    assert spec.spectral_axis_index == 0
    assert spec.shape == (10, 5)
    assert_quantity_allclose(spec_axis_centers, spec.wcs.pixel_to_world(0, np.arange(10))[1].to("keV"))
    assert_quantity_allclose(spec_axis_edges, spec.spectral_axis.bin_edges)


def test_spectrum_from_cube_wcs_tab():
    spec_axis_edges = np.arange(11) * u.keV
    spec_axis_centers = spec_axis_edges[:-1] + np.diff(spec_axis_edges) * 0.5
    energy_coord = QuantityTableCoordinate(spec_axis_centers, names="energy", physical_types="em.energy")
    data = rng.random(len(spec_axis_centers))
    cube = NDCube(data=data, wcs=energy_coord.wcs, unit=u.ph)

    spec = Spectrum(cube, spectral_axis=spec_axis_edges)
    assert isinstance(spec, Spectrum)

    assert spec.spectral_axis_index == 0
    assert spec.shape == (10,)
    assert_quantity_allclose(spec_axis_centers, spec.wcs.pixel_to_world(np.arange(10)).to("keV"))


def test_spectrum_spectra_axis_detection():
    energy = (np.arange(0, 10) + 0.5) * u.keV
    energy_coord = QuantityTableCoordinate(energy, names="energy", physical_types="em.energy")
    times = Time("2020-01-01") + 5 * np.arange(0, 11) * u.s
    time_coord = TimeTableCoordinate(times, names="time", physical_types="time")
    time_energy_wcs = (time_coord & energy_coord).wcs
    data = np.arange(5 * 10).reshape(10, 5)
    spec1 = Spectrum(data * u.ph, wcs=time_energy_wcs, spectral_axis=np.arange(11) * u.keV)
    assert spec1.spectral_axis_index == 0

    energy_energy_wcs = (energy_coord & time_coord).wcs
    data = np.arange(10 * 5).reshape(5, 10)
    spec2 = Spectrum(data * u.ph, wcs=energy_energy_wcs, spectral_axis=np.arange(11) * u.keV)
    assert spec2.spectral_axis_index == 1


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
    energy = (0.5 + np.arange(10)) * u.keV
    energy_coord = QuantityTableCoordinate(energy, names="energy", physical_types="em.energy")
    comp_wcs = CompoundLowLevelWCS(time_wcs, energy_coord.wcs)
    cube = NDCube(np.arange(10 * 5).reshape(10, 5), unit=u.ph, wcs=comp_wcs)
    spec = Spectrum(cube, spectral_axis=np.arange(11) * u.keV)
    assert spec.shape == (10, 5)
    assert spec.spectral_axis_index == 0
    assert_quantity_allclose(energy, spec.spectral_axis)


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


def test_spectral_axis_bin_edges_from_centers():
    """Test that bin_edges are correctly calculated when SpectralAxis is created with centers."""
    spec_axis = SpectralAxis(np.array([1.5, 2.5, 3.5, 4.5]) * u.keV, bin_specification="centers")
    edges = spec_axis.bin_edges
    assert edges is None


def test_spectral_axis_bin_edges_preserved():
    """Test that bin_edges are preserved when SpectralAxis is created with edges."""
    input_edges = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * u.keV
    spec_axis = SpectralAxis(input_edges, bin_specification="edges")
    assert_quantity_allclose(spec_axis.bin_edges, input_edges)


def test_spectral_axis_centers_from_edges():
    """Test that centers are correctly calculated from edges."""
    input_edges = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * u.keV
    spec_axis = SpectralAxis(input_edges, bin_specification="edges")
    assert_quantity_allclose(spec_axis, [1.5, 2.5, 3.5, 4.5] * u.keV)


def test_spectral_axis_single_center():
    """Test SpectralAxis handles single-element arrays."""
    spec_axis = SpectralAxis(np.array([5.0]) * u.keV, bin_specification="centers")
    edges = spec_axis.bin_edges
    assert edges is None


def test_spectral_axis_single_bin():
    """Test SpectralAxis handles single bins"""
    with pytest.raises(ValueError, match="If bin_specification"):
        SpectralAxis(np.array([5.0]) * u.keV, bin_specification="edges")

    spec_axis = SpectralAxis(np.array([5.0, 6.0]) * u.keV, bin_specification="edges")
    edges = spec_axis.bin_edges
    assert edges is not None
    assert len(edges) == 2
    assert spec_axis == 5.5 * u.keV


def test_spectral_axis_empty_array():
    """Test SpectralAxis handles empty arrays."""
    edges = SpectralAxis(np.array([]), u.keV)
    assert len(edges) == 0


def test_spectral_axis_pixel_ascending():
    """Test that pixel spectral axes must be ascending."""
    with pytest.raises(ValueError, match=r"u\.pix spectral axes should always be ascending"):
        SpectralAxis(np.array([5, 4, 3, 2, 1]) * u.pix)


def test_spectral_axis_pixel_ascending_valid():
    """Test that ascending pixel spectral axes are accepted."""
    spec_axis = SpectralAxis(np.array([1, 2, 3, 4, 5]) * u.pix)
    assert len(spec_axis) == 5


def test_spectrum_from_spectrum_inherits_attributes():
    """Test that Spectrum created from another Spectrum inherits spectral_axis and spectral_axis_index."""
    spec_orig = Spectrum(np.arange(1, 11) * u.watt, spectral_axis=np.arange(1, 12) * u.keV)
    spec_new = Spectrum(spec_orig)

    # Verify spectral_axis_index is inherited (Bug #1 fix)
    assert spec_new.spectral_axis_index == spec_orig.spectral_axis_index
    assert spec_new.spectral_axis_index == 0

    # Verify spectral_axis is inherited
    assert_quantity_allclose(spec_new.spectral_axis, spec_orig.spectral_axis)


def test_spectrum_from_spectrum_preserves_data():
    """Test that Spectrum created from another Spectrum preserves data."""
    data = np.arange(1, 11) * u.watt
    spec_orig = Spectrum(data, spectral_axis=np.arange(1, 12) * u.keV)
    spec_new = Spectrum(spec_orig)

    assert_array_equal(spec_new.data, spec_orig.data)
    assert spec_new.unit == spec_orig.unit


def test_spectrum_strictly_increasing_spectral_axis():
    """Test that strictly increasing spectral axis is accepted."""
    spec = Spectrum(np.arange(1, 6) * u.watt, spectral_axis=np.array([1, 2, 3, 4, 5]) * u.keV)
    assert_quantity_allclose(spec.spectral_axis, [1, 2, 3, 4, 5] * u.keV)


def test_spectrum_non_monotonic_spectral_axis_raises():
    """Test that non-monotonic spectral axis raises ValueError."""
    with pytest.raises(ValueError, match="strictly increasing"):
        Spectrum(np.arange(1, 6) * u.watt, spectral_axis=np.array([1, 3, 2, 4, 5]) * u.keV)


def test_spectrum_duplicate_values_in_spectral_axis_raises():
    """Test that duplicate values in spectral axis raises ValueError."""
    with pytest.raises(ValueError, match="strictly increasing"):
        Spectrum(np.arange(1, 5) * u.watt, spectral_axis=np.array([1, 2, 2, 3]) * u.keV)


def test_spectrum_single_element_spectral_axis():
    """Test that single-element spectral axis is accepted."""
    spec = Spectrum(np.array([5]) * u.watt, spectral_axis=np.array([10]) * u.keV)
    assert spec.shape == (1,)
    assert_quantity_allclose(spec.spectral_axis, [10] * u.keV)


def test_spectrum_spectral_axis_length_mismatch():
    """Test that mismatched spectral axis length raises ValueError."""
    with pytest.raises(ValueError, match="Spectral axis length"):
        Spectrum(np.arange(1, 11) * u.watt, spectral_axis=np.arange(1, 5) * u.keV)


def test_spectrum_uncertainty_shape_mismatch():
    """Test that mismatched uncertainty shape raises ValueError."""
    data = np.arange(1, 11) * u.watt
    uncertainty = StdDevUncertainty(np.arange(1, 6))  # Wrong shape
    with pytest.raises(ValueError, match=r"Data axis .* and uncertainty .* shapes must be the same"):
        Spectrum(data, spectral_axis=np.arange(1, 12) * u.keV, uncertainty=uncertainty)


def test_spectrum_with_valid_uncertainty():
    """Test Spectrum with correctly shaped uncertainty."""
    data = np.arange(1, 11) * u.watt
    uncertainty = StdDevUncertainty(np.ones(10) * 0.1)
    spec = Spectrum(data, spectral_axis=np.arange(1, 12) * u.keV, uncertainty=uncertainty)
    assert spec.uncertainty is not None
    assert spec.uncertainty.array.shape == data.shape


def test_slice_preserves_spectral_axis_index():
    """Test that slicing preserves spectral_axis_index."""
    spec = Spectrum(np.arange(1, 11) * u.watt, spectral_axis=(np.arange(1, 11) + 0.5) * u.keV)
    sliced = spec[2:7]
    assert sliced.spectral_axis_index == spec.spectral_axis_index


def test_slice_updates_spectral_axis():
    """Test that slicing correctly slices spectral_axis."""
    spec = Spectrum(np.arange(1, 11) * u.watt, spectral_axis=(np.arange(1, 11) + 0.5) * u.keV)
    sliced = spec[2:5]
    assert_quantity_allclose(sliced.spectral_axis, [3.5, 4.5, 5.5] * u.keV)


def test_slice_single_element():
    """Test slicing to a single element."""
    spec = Spectrum(np.arange(1, 11) * u.watt, spectral_axis=(np.arange(1, 11) + 0.5) * u.keV)
    sliced = spec[5:6]
    assert sliced.shape == (1,)
    assert_quantity_allclose(sliced.spectral_axis, [6.5] * u.keV)


def test_arithmetic_preserves_spectral_axis():
    """Test that arithmetic operations preserve spectral_axis."""
    spec = Spectrum(np.arange(1, 11) * u.watt, spectral_axis=(np.arange(1, 11) + 0.5) * u.keV)
    result = spec + 1 * u.watt
    assert_quantity_allclose(result.spectral_axis, spec.spectral_axis)


def test_arithmetic_preserves_spectral_axis_index():
    """Test that arithmetic operations preserve spectral_axis_index."""
    spec = Spectrum(np.arange(1, 11) * u.watt, spectral_axis=(np.arange(1, 11) + 0.5) * u.keV)
    result = spec * 2
    assert result.spectral_axis_index == spec.spectral_axis_index
