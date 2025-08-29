import warnings

import numpy as np
import pytest

import astropy.units as u

from sunkit_spex.legacy import thermal

# Manually load file that was used to compile expected flux values.
thermal.setup_continuum_parameters(
    "https://soho.nascom.nasa.gov/solarsoft/packages/xray/dbase/chianti/chianti_cont_1_250_v71.sav"
)
thermal.setup_line_parameters(
    "https://soho.nascom.nasa.gov/solarsoft/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav"
)
SSW_INTENSITY_UNIT = u.ph / u.cm**2 / u.s / u.keV
DEFAULT_ABUNDANCE_TYPE = "sun_coronal_ext"


def fvth_simple():
    """Define expected thermal spectrum as returned by SSW routine f_vth.

    f_vth is the standard routine used to calculate a thermal spectrum in SSW.
    The output defined here uses a energy bins from 3-28 keV with a bin width of 0.5 keV,
    a temperature of 6 MK, a emission measure of 1e44 cm^-3, an observer distance of 1AU,
    default ('sun_coronal_ext') abundances, and default relative abundances.
    The SSW output can be reproduced in SSW/IDL with the following code:

    Returns
    -------
    inputs: `tuple`
        The Python inputs required to produce the spectrum

    ssw_output: `astropy.units.Quantity`
        The spectrum output by the SSW routine, f_vth, given the above inputs.

    Notes
    -----
    The spectrum output by this function can be reproduced in IDL with the following code.

    energy_in = fltarr(2, 50)
    energy_in[0, *] = findgen(50) * 0.5 + 3
    energy_in[1, *] = energy_in[0, *] + 0.5
    temp = 6.
    flux = f_vth(temp, energy_in, /kev, /earth)
    """
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    temperature = 6 * u.MK
    emission_measure = 1e44 / u.cm**3
    abundance_type = DEFAULT_ABUNDANCE_TYPE
    relative_abundances = None
    observer_distance = (1 * u.AU).to(u.cm)
    inputs = (energy_edges, temperature, emission_measure, abundance_type, relative_abundances, observer_distance)
    ssw_output = [
        0.49978617,
        0.15907305,
        0.052894916,
        0.017032871,
        0.0056818775,
        0.0019678525,
        0.00071765773,
        0.00026049651,
        8.5432410e-05,
        3.0082287e-05,
        1.0713027e-05,
        3.8201970e-06,
        1.3680836e-06,
        4.9220034e-07,
        1.7705435e-07,
        6.4002016e-08,
        2.3077330e-08,
        8.3918881e-09,
        3.0453193e-09,
        1.1047097e-09,
        4.0377532e-10,
        1.4734168e-10,
        5.3671578e-11,
        1.9628120e-11,
        7.2107064e-12,
        2.6457057e-12,
        9.6945607e-13,
        3.5472713e-13,
        1.3051763e-13,
        4.8216642e-14,
        1.7797136e-14,
        6.5629896e-15,
        2.4178513e-15,
        8.8982728e-16,
        3.2711010e-16,
        1.2110889e-16,
        4.4997199e-17,
        1.6709174e-17,
        6.2011332e-18,
        2.2999536e-18,
        8.5248536e-19,
        3.1576185e-19,
        1.1687443e-19,
        4.3226546e-20,
        1.5974677e-20,
        5.9133793e-21,
        2.2103457e-21,
        8.2594126e-22,
        3.0853276e-22,
        1.1521560e-22,
    ] * SSW_INTENSITY_UNIT
    return inputs, ssw_output


def chianti_kev_cont_simple():
    """Define expected thermal continuum spectrum as returned by SSW routine chianti_kev_cont.

    chianti_cont_kev is the standard routine used to calculate a continuum spectrum in SSW
    that includes free-free, free-bound and two-photon components.  It is used by f_vth.pro.
    The output defined here uses a energy bins from 3-28 keV with a bin width of 0.5 keV,
    a temperature of 6 MK, a emission measure of 1e44 cm^-3, an observer distance of 1AU,
    default ('sun_coronal_ext') abundances, and default relative abundances.
    The SSW output can be reproduced in SSW/IDL with the following code:

    Returns
    -------
    inputs: `tuple`
        The Python inputs required to produce the spectrum

    ssw_output: `astropy.units.Quantity`
        The spectrum output by the SSW routine, chianti_kev_cont, given the above inputs.

    Notes
    -----
    The spectrum output by this function can be reproduced in IDL with the following code.

    energy_in = fltarr(2, 50)
    energy_in[0, *] = findgen(50) * 0.5 + 3
    energy_in[1, *] = energy_in[0, *] + 0.5
    temp = 6.
    flux = chianti_kev_cont(temp, energy_in, /kev, /earth)
    """
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    temperature = 6 * u.MK
    emission_measure = 1e44 / u.cm**3
    abundance_type = DEFAULT_ABUNDANCE_TYPE
    relative_abundances = None
    observer_distance = (1 * u.AU).to(u.cm)
    inputs = (energy_edges, temperature, emission_measure, abundance_type, relative_abundances, observer_distance)
    ssw_output = [
        0.435291,
        0.144041,
        0.0484909,
        0.0164952,
        0.00567440,
        0.00196711,
        0.000685918,
        0.000240666,
        8.48974e-05,
        3.00666e-05,
        1.07070e-05,
        3.81792e-06,
        1.36722e-06,
        4.91871e-07,
        1.76929e-07,
        6.39545e-08,
        2.30593e-08,
        8.38502e-09,
        3.04271e-09,
        1.10372e-09,
        4.03399e-10,
        1.47199e-10,
        5.36175e-11,
        1.96076e-11,
        7.20292e-12,
        2.64275e-12,
        9.68337e-13,
        3.54305e-13,
        1.30357e-13,
        4.81556e-14,
        1.77739e-14,
        6.55418e-15,
        2.41451e-15,
        8.88565e-16,
        3.26635e-16,
        1.20928e-16,
        4.49286e-17,
        1.66830e-17,
        6.19118e-18,
        2.29618e-18,
        8.51055e-19,
        3.15220e-19,
        1.16669e-19,
        4.31491e-20,
        1.59455e-20,
        5.90236e-21,
        2.20614e-21,
        8.24339e-22,
        3.07924e-22,
        1.14983e-22,
    ] * SSW_INTENSITY_UNIT
    return inputs, ssw_output


def chianti_kev_lines_simple():
    """Define expected thermal line spectrum as returned by SSW routine chianti_kev_lines.

    chianti_cont_kev is the standard routine used to calculate a solar X-ray line spectrum
    in SSW.  It is used by f_vth.pro.
    The output defined here uses a energy bins from 3-28 keV with a bin width of 0.5 keV,
    a temperature of 6 MK, a emission measure of 1e44 cm^-3, an observer distance of 1AU,
    default ('sun_coronal_ext') abundances, and default relative abundances.
    The SSW output can be reproduced in SSW/IDL with the following code:

    Returns
    -------
    inputs: `tuple`
        The Python inputs required to produce the spectrum

    ssw_output: `astropy.units.Quantity`
        The spectrum output by the SSW routine, chianti_kev_cont, given the above inputs.

    Notes
    -----
    The spectrum output by this function can be reproduced in IDL with the following code.

    energy_in = fltarr(2, 50)
    energy_in[0, *] = findgen(50) * 0.5 + 3
    energy_in[1, *] = energy_in[0, *] + 0.5
    temp = 6.
    flux = chianti_kev_lines(temp, energy_in, /kev, /earth)
    """
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    temperature = 6 * u.MK
    emission_measure = 1e44 / u.cm**3
    abundance_type = "sun_coronal"
    relative_abundances = None
    observer_distance = (1 * u.AU).to(u.cm)
    inputs = (energy_edges, temperature, emission_measure, abundance_type, relative_abundances, observer_distance)
    ssw_output = [
        0.073248975,
        0.014247916,
        0.0044440511,
        0.00067793718,
        2.3333176e-05,
        2.5751346e-10,
        3.4042361e-05,
        2.1403499e-05,
        5.0370664e-07,
        8.3715751e-12,
        2.7737142e-12,
        1.5496721e-13,
        1.9522280e-17,
        1.3281716e-20,
        1.0493879e-21,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
        0.0000000,
    ] * SSW_INTENSITY_UNIT
    return inputs, ssw_output


def fvth_Fe2():
    """Define expected thermal continuum spectrum as returned by SSW routine chianti_kev_cont.

    chianti_cont_kev is the standard routine used to calculate a solar X-ray line spectrum
    in SSW.  It is used by f_vth.pro.
    The output defined here uses a energy bins from 3-28 keV with a bin width of 0.5 keV,
    a temperature of 6 MK, a emission measure of 1e44 cm^-3, an observer distance of 1AU,
    default ('sun_coronal_ext') abundances, and default relative abundances.
    The SSW output can be reproduced in SSW/IDL with the following code:

    Returns
    -------
    inputs: `tuple`
        The Python inputs required to produce the spectrum

    ssw_output: `astropy.units.Quantity`
        The spectrum output by the SSW routine, chianti_kev_lines, given the above inputs.

    Notes
    -----
    The spectrum output by this function can be reproduced in IDL with the following code.

    energy_in = fltarr(2, 50)
    energy_in[0, *] = findgen(50) * 0.5 + 3
    energy_in[1, *] = energy_in[0, *] + 0.5
    temp = 6.
    rel_abun = [[26, 2]]
    flux = chianti_kev_cont(temp, energy_in, rel_abun=rel_abun, /kev, /earth)

    Ensure you are using the same .sav file as used here.
    """
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    temperature = 6 * u.MK
    emission_measure = 1e44 / u.cm**3
    abundance_type = DEFAULT_ABUNDANCE_TYPE
    relative_abundances = ((26, 2),)
    observer_distance = (1 * u.AU).to(u.cm)
    inputs = (energy_edges, temperature, emission_measure, abundance_type, relative_abundances, observer_distance)
    ssw_output = [
        4.6152353e-01,
        1.5266217e-01,
        5.1370505e-02,
        1.7469261e-02,
        6.0074395e-03,
        2.0820354e-03,
        7.2583189e-04,
        2.5462240e-04,
        8.9805966e-05,
        3.1800522e-05,
        1.1323175e-05,
        4.0371842e-06,
        1.4456124e-06,
        5.2003952e-07,
        1.8704908e-07,
        6.7609605e-08,
        2.4375957e-08,
        8.8636174e-09,
        3.2163083e-09,
        1.1666705e-09,
        4.2640558e-10,
        1.5559361e-10,
        5.6675255e-11,
        2.0726003e-11,
        7.6138887e-12,
        2.7935852e-12,
        1.0236266e-12,
        3.7454293e-13,
        1.3780716e-13,
        5.0909286e-14,
        1.8790933e-14,
        6.9294558e-15,
        2.5528610e-15,
        9.3951687e-16,
        3.4537985e-16,
        1.2787355e-16,
        4.7510902e-17,
        1.7642638e-17,
        6.5476106e-18,
        2.4284929e-18,
        9.0014011e-19,
        3.3341828e-19,
        1.2341190e-19,
        4.5645426e-20,
        1.6869074e-20,
        6.2446062e-21,
        2.3341643e-21,
        8.7221399e-22,
        3.2582179e-22,
        1.2167254e-22,
    ] * SSW_INTENSITY_UNIT
    return inputs, ssw_output


def chianti_kev_cont_Fe2():
    """Define expected thermal continuum spectrum as returned by SSW routine chianti_kev_cont.

    chianti_cont_kev is the standard routine used to calculate a solar X-ray line spectrum
    in SSW.  It is used by f_vth.pro.
    The output defined here uses a energy bins from 3-28 keV with a bin width of 0.5 keV,
    a temperature of 6 MK, a emission measure of 1e44 cm^-3, an observer distance of 1AU,
    default ('sun_coronal_ext') abundances, and default relative abundances.
    The SSW output can be reproduced in SSW/IDL with the following code:

    Returns
    -------
    inputs: `tuple`
        The Python inputs required to produce the spectrum

    ssw_output: `astropy.units.Quantity`
        The spectrum output by the SSW routine, chianti_kev_lines, given the above inputs.

    Notes
    -----
    The spectrum output by this function can be reproduced in IDL with the following code.

    energy_in = fltarr(2, 50)
    energy_in[0, *] = findgen(50) * 0.5 + 3
    energy_in[1, *] = energy_in[0, *] + 0.5
    temp = 6.
    rel_abun = [[26, 2]]
    flux = chianti_kev_cont(temp, energy_in, rel_abun=rel_abun, /kev, /earth)

    Ensure you are using the same .sav file as used here.
    """
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    temperature = 6 * u.MK
    emission_measure = 1e44 / u.cm**3
    abundance_type = DEFAULT_ABUNDANCE_TYPE
    relative_abundances = ((26, 2),)
    observer_distance = (1 * u.AU).to(u.cm)
    inputs = (energy_edges, temperature, emission_measure, abundance_type, relative_abundances, observer_distance)
    ssw_output = [
        4.6152353e-01,
        1.5266217e-01,
        5.1370505e-02,
        1.7469261e-02,
        6.0074395e-03,
        2.0820354e-03,
        7.2583189e-04,
        2.5462240e-04,
        8.9805966e-05,
        3.1800522e-05,
        1.1323175e-05,
        4.0371842e-06,
        1.4456124e-06,
        5.2003952e-07,
        1.8704908e-07,
        6.7609605e-08,
        2.4375957e-08,
        8.8636174e-09,
        3.2163083e-09,
        1.1666705e-09,
        4.2640558e-10,
        1.5559361e-10,
        5.6675255e-11,
        2.0726003e-11,
        7.6138887e-12,
        2.7935852e-12,
        1.0236266e-12,
        3.7454293e-13,
        1.3780716e-13,
        5.0909286e-14,
        1.8790933e-14,
        6.9294558e-15,
        2.5528610e-15,
        9.3951687e-16,
        3.4537985e-16,
        1.2787355e-16,
        4.7510902e-17,
        1.7642638e-17,
        6.5476106e-18,
        2.4284929e-18,
        9.0014011e-19,
        3.3341828e-19,
        1.2341190e-19,
        4.5645426e-20,
        1.6869074e-20,
        6.2446062e-21,
        2.3341643e-21,
        8.7221399e-22,
        3.2582179e-22,
        1.2167254e-22,
    ] * SSW_INTENSITY_UNIT
    return inputs, ssw_output


def chianti_kev_lines_Fe2():
    """Define expected thermal line spectrum as returned by SSW routine chianti_kev_lines.

    chianti_lines_kev is the standard routine used to calculate a solar X-ray line spectrum
    in SSW.  It is used by f_vth.pro.
    The output defined here uses a energy bins from 3-28 keV with a bin width of 0.5 keV,
    a temperature of 6 MK, a emission measure of 1e44 cm^-3, an observer distance of 1AU,
    default ('sun_coronal_ext') abundances, and default relative abundances.
    The SSW output can be reproduced in SSW/IDL with the following code:

    Returns
    -------
    inputs: `tuple`
        The Python inputs required to produce the spectrum

    ssw_output: `astropy.units.Quantity`
        The spectrum output by the SSW routine, chianti_kev_lines, given the above inputs.

    Notes
    -----
    The spectrum output by this function can be reproduced in IDL with the following code.

    energy_in = fltarr(2, 50)
    energy_in[0, *] = findgen(50) * 0.5 + 3
    energy_in[1, *] = energy_in[0, *] + 0.5
    temp = 6.
    rel_abun = [[26, 2]]
    flux = chianti_kev_lines(temp, energy_in, rel_abun=rel_abun, /kev, /earth)

    Ensure you are using the same .sav file as used here.
    """
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    temperature = 6 * u.MK
    emission_measure = 1e44 / u.cm**3
    abundance_type = "sun_coronal"
    relative_abundances = ((26, 2),)
    observer_distance = (1 * u.AU).to(u.cm)
    inputs = (energy_edges, temperature, emission_measure, abundance_type, relative_abundances, observer_distance)
    ssw_output = [
        0.073248975,
        0.014247916,
        0.0044440511,
        0.00067793718,
        2.3333176e-05,
        5.1462595e-10,
        6.8084722e-05,
        4.2806998e-05,
        1.0074133e-06,
        1.6740345e-11,
        5.5473790e-12,
        3.0992380e-13,
        1.9522283e-17,
        1.3281716e-20,
        1.0493879e-21,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
        0.0000000e00,
    ] * SSW_INTENSITY_UNIT
    return inputs, ssw_output


@pytest.mark.parametrize("ssw", [fvth_simple])
def test_thermal_emission_against_ssw(ssw):
    input_args, expected = ssw()
    output = thermal.thermal_emission(*input_args)
    expected_value = expected.to_value(output.unit)
    np.testing.assert_allclose(output.value, expected_value, rtol=0.03)


@pytest.mark.parametrize("ssw", [chianti_kev_cont_simple, chianti_kev_cont_Fe2])
def test_continuum_emission_against_ssw(ssw):
    input_args, expected = ssw()
    output = thermal.continuum_emission(*input_args)
    expected_value = expected.to_value(output.unit)
    np.testing.assert_allclose(output.value, expected_value, rtol=0.03)


@pytest.mark.parametrize("ssw", [chianti_kev_lines_simple, chianti_kev_lines_Fe2])
def test_line_emission_against_ssw(ssw):
    input_args, expected = ssw()
    output = thermal.line_emission(*input_args)
    expected_value = expected.to_value(output.unit)
    np.testing.assert_allclose(output.value, expected_value, rtol=0.03, atol=1e-30)


def test_scalar_energy_input():
    with pytest.raises(ValueError):
        thermal.thermal_emission(10 * u.keV, 6 * u.MK, 1e44 / u.cm**3)


def test_len1_energy_input():
    with pytest.raises(ValueError):
        thermal.thermal_emission([10] * u.keV, 6 * u.MK, 1e44 / u.cm**3)


def test_energy_out_of_range_error():
    with pytest.raises(ValueError):
        thermal.thermal_emission([0.01, 10] * u.keV, 6 * u.MK, 1e44 / u.cm**3)


def test_temperature_out_of_range_error():
    with pytest.raises(ValueError):
        thermal.thermal_emission([5, 10] * u.keV, 0.1 * u.MK, 1e44 / u.cm**3)


def test_relative_abundance_negative_input():
    with pytest.raises(ValueError):
        thermal.thermal_emission([5, 10] * u.keV, 10 * u.MK, 1e44 / u.cm**3, relative_abundances=((26, -1)))


def test_relative_abundance_invalid_atomic_number_input():
    with pytest.raises(ValueError):
        thermal.thermal_emission([5, 10] * u.keV, 10 * u.MK, 1e44 / u.cm**3, relative_abundances=((100, 1)))


def test_energy_out_of_range_warning():
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        output = thermal.line_emission(np.arange(3, 28, 0.5) * u.keV, 6 * u.MK, 1e44 / u.cm**3)  # noqa
        assert issubclass(w[0].category, UserWarning)


def test_continuum_energy_out_of_range():
    with pytest.raises(ValueError):
        # Use an energy range that goes out of bounds
        # on the lower end--should error
        _ = thermal.continuum_emission(np.arange(0.1, 28, 0.5) * u.keV, 6 * u.MK, 1e44 / u.cm**3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # The continuum emission should only warn if we go out of
        # bounds on the upper end.
        _ = thermal.continuum_emission(np.arange(10, 1000, 0.5) * u.keV, 6 * u.MK, 1e44 / u.cm**3)
        assert issubclass(w[0].category, UserWarning)


def test_empty_flux_out_of_range():
    """The CHIANTI grid covers ~1 to 300 keV, but the values greater than
    of the grid max energy should all be zeros."""
    energy_edges = np.geomspace(10, 800, num=1000) << u.keV
    midpoints = energy_edges[:-1] + np.diff(energy_edges) / 2

    temperature = 20 << u.MK
    em = 1e49 << u.cm**-3

    flux = thermal.thermal_emission(energy_edges, temperature, em)
    # the continuum is the one we need to check
    max_e = thermal.CONTINUUM_GRID["energy range keV"][1] << u.keV
    should_be_zeros = midpoints >= max_e

    true_zero = 0 * (hopefully_zero := flux[should_be_zeros])
    np.testing.assert_allclose(true_zero, hopefully_zero)


def test_abundances_should_not_change():
    """
    Addressing the issue in PR #231, mainly that the DEFAULT_ABUNDANCES
    table is modified at the module level inadvertently if non-default
    abundances values are used in certain situations.

    The fix is that anywhere we have that table appear during a WRITE operation,
    we need to use a copy of it rather than modifying it in-place.

    This is just a smoke test to verify that the abundance array does not
    change between function calls with different abundances.
    """
    # Save the original abundance values for later comparison
    orig = thermal.DEFAULT_ABUNDANCES[thermal.DEFAULT_ABUNDANCE_TYPE].data.copy()

    rng = np.random.default_rng()
    edges = np.geomspace(3, 30, 100)
    # Repeat this a few times for good measure
    for _ in range(10):
        abundances = (
            (20, rng.uniform()),  # Ca
            (13, rng.uniform()),  # Al
            (18, rng.uniform()),  # Ar
        )
        # Apply the model several times;
        # if the DEFAULT_ABUNDANCES get modified, it will be multiplicative
        for _ in range(10):
            __ = thermal.thermal_emission(
                energy_edges=edges << u.keV,
                temperature=20 << u.MK,
                emission_measure=(1 << (1e49 * u.cm**-3)),
                relative_abundances=abundances,
            )

        after_models = thermal.DEFAULT_ABUNDANCES[thermal.DEFAULT_ABUNDANCE_TYPE].data
        assert np.allclose(after_models.data, orig.data)
