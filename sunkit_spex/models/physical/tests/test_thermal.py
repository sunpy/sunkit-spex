import warnings

import numpy as np
import pytest

import astropy.units as u

from sunkit_spex.models.physical import thermal

# Manually load file that was used to compile expected flux values.
thermal.setup_continuum_parameters(
    "https://soho.nascom.nasa.gov/solarsoft/packages/xray/dbase/chianti/chianti_cont_1_250_v71.sav"
)
thermal.setup_line_parameters(
    "https://soho.nascom.nasa.gov/solarsoft/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav"
)
SSW_INTENSITY_UNIT = u.ph / u.cm**2 / u.s / u.keV
DEFAULT_ABUNDANCE_TYPE = thermal.DEFAULT_ABUNDANCE_TYPE


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

    chianti_kev_common_load, contfile = 'chianti_cont_1_250_v71.sav', linefile = 'chianti_lines_1_10_v71.sav', /reload

    energy_in = fltarr(2, 50)
    energy_in[0, *] = findgen(50) * 0.5 + 3
    energy_in[1, *] = energy_in[0, *] + 0.5
    temp = 6.
    em = 1e-5
    flux = f_vth(energy_in, [em,temp], rel_abun=rel_abun, /kev, /earth)
    """
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    temperature = 6 * u.MK
    emission_measure = 1e-5 / u.cm**3
    abundance_type = DEFAULT_ABUNDANCE_TYPE
    observer_distance = (1 * u.AU).to(u.cm)
    # fmt: off
    mg, al, si, s, ar, ca, fe = 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1
    inputs = (
        energy_edges, temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type
    )
    inputs_class = (
        temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type
    )
    ssw_output = (
        [
            0.508525, 0.158277, 0.0529235, 0.0171709, 0.00569773, 0.00196599, 0.000719867,
            0.000262074, 8.53937e-05, 3.00764e-05, 1.06980e-05, 3.81875e-06, 1.36751e-06,
            4.91036e-07, 1.76798e-07, 6.38010e-08, 2.30734e-08, 8.36119e-09, 3.03534e-09,
            1.10357e-09, 4.01951e-10, 1.46575e-10, 5.35405e-11, 1.95792e-11, 7.16793e-12,
            2.62703e-12, 9.63826e-13, 3.53981e-13, 1.30125e-13, 4.78720e-14, 1.76232e-14,
            6.49645e-15, 2.39605e-15, 8.84008e-16, 3.26543e-16, 1.20678e-16, 4.46146e-17,
            1.65130e-17, 6.11063e-18, 2.26457e-18, 8.38897e-19, 3.11188e-19, 1.15416e-19,
            4.28376e-20, 1.59088e-20, 5.90513e-21, 2.19619e-21, 8.16363e-22, 3.03490e-22,
            1.12987e-22
        ]
        * SSW_INTENSITY_UNIT * (4 * np.pi * observer_distance**2)
    )
    # fmt: on
    return inputs, inputs_class, energy_edges, ssw_output


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

    chianti_kev_common_load, contfile = 'chianti_cont_1_250_v71.sav', linefile = 'chianti_lines_1_10_v71.sav', /reload

    energy_in = fltarr(2, 50)
    energy_in[0, *] = findgen(50) * 0.5 + 3
    energy_in[1, *] = energy_in[0, *] + 0.5
    temp = 6.
    flux = chianti_kev_cont(temp, energy_in, /kev, /earth)
    """
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    temperature = 6 * u.MK
    emission_measure = 1e-5 / u.cm**3
    abundance_type = DEFAULT_ABUNDANCE_TYPE
    observer_distance = (1 * u.AU).to(u.cm)
    # fmt: off
    mg, al, si, s, ar, ca, fe = 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1
    inputs = (
        energy_edges, temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type
    )
    inputs_class = (
        temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type
    )
    ssw_output = (
        [
            0.435182, 0.143992, 0.0484651, 0.0164877, 0.00567250, 0.00196526, 0.000685526,
            0.000240551, 8.48489e-05, 3.00608e-05, 1.06920e-05, 3.81647e-06, 1.36665e-06,
            4.90708e-07, 1.76672e-07, 6.37536e-08, 2.30554e-08, 8.35434e-09, 3.03274e-09,
            1.10259e-09, 4.01577e-10, 1.46433e-10, 5.34866e-11, 1.95588e-11, 7.16019e-12,
            2.62410e-12, 9.62714e-13, 3.53559e-13, 1.29965e-13, 4.78114e-14, 1.76002e-14,
            6.48773e-15, 2.39274e-15, 8.82754e-16, 3.26069e-16, 1.20498e-16, 4.45465e-17,
            1.64872e-17, 6.10082e-18, 2.26085e-18, 8.37490e-19, 3.10654e-19, 1.15214e-19,
            4.27608e-20, 1.58797e-20, 5.89412e-21, 2.19201e-21, 8.14780e-22, 3.02891e-22,
            1.12759e-22
        ]
        * SSW_INTENSITY_UNIT * (4 * np.pi * observer_distance**2)
    )
    # fmt: on
    return inputs, inputs_class, energy_edges, ssw_output


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

    chianti_kev_common_load, contfile = 'chianti_cont_1_250_v71.sav', linefile = 'chianti_lines_1_10_v71.sav', /reload

    energy_in = fltarr(2, 50)
    energy_in[0, *] = findgen(50) * 0.5 + 3
    energy_in[1, *] = energy_in[0, *] + 0.5
    temp = 6.
    flux = chianti_kev_lines(temp, energy_in, /kev, /earth)
    """
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    temperature = 6 * u.MK
    emission_measure = 1e-5 / u.cm**3
    abundance_type = DEFAULT_ABUNDANCE_TYPE
    observer_distance = (1 * u.AU).to(u.cm)
    # fmt: off
    mg, al, si, s, ar, ca, fe = 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1
    inputs = (
        energy_edges, temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type
    )
    inputs_class = (
        temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type
    )
    ssw_output = (
        [
            0.0732490, 0.0142479, 0.00444405, 0.000677937, 2.33332e-05, 2.57513e-10, 3.40424e-05,
            2.14035e-05, 5.03707e-07, 8.37158e-12, 2.77371e-12, 1.54967e-13, 1.95223e-17, 1.32817e-20,
            1.04939e-21, 5.60979545e-32, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
            0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
            0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
            0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000
        ]
        * SSW_INTENSITY_UNIT * (4 * np.pi * observer_distance**2)
    )
    # fmt: on
    return inputs, inputs_class, energy_edges, ssw_output


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

    chianti_kev_common_load, contfile = 'chianti_cont_1_250_v71.sav', linefile = 'chianti_lines_1_10_v71.sav', /reload

    energy_in = fltarr(2, 50)
    energy_in[0, *] = findgen(50) * 0.5 + 3
    energy_in[1, *] = energy_in[0, *] + 0.5
    temp = 0.5170399957287106
    rel_abun = [[26, 2]]
    em = 1e-5
    flux = f_vth(energy_in, [em,temp], rel_abun=rel_abun, /kev, /earth)

    Ensure you are using the same .sav file as used here.
    """
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    temperature = 6 * u.MK
    emission_measure = 1e-5 / u.cm**3
    abundance_type = DEFAULT_ABUNDANCE_TYPE
    observer_distance = (1 * u.AU).to(u.cm)
    # fmt: off
    mg, al, si, s, ar, ca, fe = 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.4010299956639812
    inputs = (
        energy_edges, temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type
    )
    inputs_class = (
        temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type
    )
    ssw_output = (
        [
            0.534751, 0.166895, 0.0558015, 0.0181445, 0.00603063, 0.00208080, 0.000793823, 0.000297441,
            9.08041e-05, 3.18103e-05, 1.13134e-05, 4.03799e-06, 1.44590e-06, 5.19144e-07, 1.86907e-07,
            6.74457e-08, 2.43905e-08, 8.83818e-09, 3.20843e-09, 1.16649e-09, 4.24865e-10, 1.54930e-10,
            5.65926e-11, 2.06955e-11, 7.57671e-12, 2.77691e-12, 1.01884e-12, 3.74194e-13, 1.37559e-13,
            5.06084e-14, 1.86312e-14, 6.86828e-15, 2.53330e-15, 9.34684e-16, 3.45278e-16, 1.27608e-16,
            4.71785e-17, 1.74628e-17, 6.46241e-18, 2.39505e-18, 8.87281e-19, 3.29154e-19, 1.22086e-19,
            4.53157e-20, 1.68300e-20, 6.24746e-21, 2.32363e-21, 8.63785e-22, 3.21138e-22, 1.19564e-22
        ]
        * SSW_INTENSITY_UNIT
        * (4 * np.pi * observer_distance**2)
    )
    # fmt: on
    return inputs, inputs_class, energy_edges, ssw_output


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

    chianti_kev_common_load, contfile = 'chianti_cont_1_250_v71.sav', linefile = 'chianti_lines_1_10_v71.sav', /reload

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
    emission_measure = 1e-5 / u.cm**3
    abundance_type = DEFAULT_ABUNDANCE_TYPE
    observer_distance = (1 * u.AU).to(u.cm)
    # fmt: off
    mg, al, si, s, ar, ca, fe = 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.4010299956639812
    inputs = (
        energy_edges, temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type
    )
    inputs_class = (
        temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type
    )
    ssw_output = (
        [
            0.461402, 0.152608, 0.0513423, 0.0174610, 0.00600528, 0.00208003, 0.000725404, 0.000254496,
            8.97528e-05, 3.17937e-05, 1.13071e-05, 4.03558e-06, 1.44498e-06, 5.18797e-07, 1.86775e-07,
            6.73955e-08, 2.43714e-08, 8.83094e-09, 3.20568e-09, 1.16545e-09, 4.24469e-10, 1.54780e-10,
            5.65355e-11, 2.06739e-11, 7.56853e-12, 2.77380e-12, 1.01766e-12, 3.73748e-13, 1.37390e-13,
            5.05443e-14, 1.86069e-14, 6.85906e-15, 2.52980e-15, 9.33357e-16, 3.44776e-16, 1.27417e-16,
            4.71065e-17, 1.74354e-17, 6.45203e-18, 2.39112e-18, 8.85792e-19, 3.28589e-19, 1.21872e-19,
            4.52345e-20, 1.67993e-20, 6.23582e-21, 2.31921e-21, 8.62109e-22, 3.20504e-22, 1.19323e-22
        ]
        * SSW_INTENSITY_UNIT
        * (4 * np.pi * observer_distance**2)
    )
    # fmt: on
    return inputs, inputs_class, energy_edges, ssw_output


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

    chianti_kev_common_load, contfile = 'chianti_cont_1_250_v71.sav', linefile = 'chianti_lines_1_10_v71.sav', /reload

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
    emission_measure = 1e-5 / u.cm**3
    abundance_type = DEFAULT_ABUNDANCE_TYPE
    observer_distance = (1 * u.AU).to(u.cm)
    # fmt: off
    mg, al, si, s, ar, ca, fe = 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.4010299956639812
    inputs = (
        energy_edges, temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type
    )
    inputs_class = (
        temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type
    )
    ssw_output = (
        [
            0.0732490, 0.0142479, 0.00444405, 0.000677937, 2.33332e-05, 5.14626e-10, 6.80847e-05,
            4.28070e-05, 1.00741e-06, 1.67403e-11, 5.54738e-12, 3.09924e-13, 1.95223e-17, 1.32817e-20,
            1.04939e-21, 5.60979545e-32, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
            0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
            0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
            0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000
        ]
        * SSW_INTENSITY_UNIT * (4 * np.pi * observer_distance**2)
    )
    # fmt: on
    return inputs, inputs_class, energy_edges, ssw_output


@pytest.mark.parametrize("ssw", [fvth_simple, fvth_Fe2])
def test_thermal_emission_against_ssw(ssw):
    input_args, input_args_class, energy_edges, expected = ssw()
    model_class = thermal.ThermalEmission(*input_args_class)
    output_class = model_class(energy_edges)
    expected_value = expected.to_value(output_class.unit)
    np.testing.assert_allclose(output_class.value, expected_value, rtol=0.03)


@pytest.mark.parametrize("ssw", [chianti_kev_cont_simple, chianti_kev_cont_Fe2])
def test_continuum_emission_against_ssw(ssw):
    input_args, input_args_class, energy_edges, expected = ssw()
    model_class = thermal.ContinuumEmission(*input_args_class)
    output_class = model_class(energy_edges)
    expected_value = expected.to_value(output_class.unit)
    np.testing.assert_allclose(output_class.value, expected_value, rtol=0.03)


@pytest.mark.parametrize("ssw", [chianti_kev_lines_simple, chianti_kev_lines_Fe2])
def test_line_emission_against_ssw(ssw):
    input_args, input_args_class, energy_edges, expected = ssw()
    model_class = thermal.LineEmission(*input_args_class)
    output_class = model_class(energy_edges)
    expected_value = expected.to_value(output_class.unit)
    np.testing.assert_allclose(output_class.value, expected_value, rtol=0.05, atol=1e-30)


def test_scalar_energy_input():
    with pytest.raises(ValueError, match="energy_edges must be a 1-D astropy Quantity with length greater than 1"):
        thermal.ThermalEmission(6 * u.MK, 1e-5 / u.cm**3)(10 * u.keV)


def test_len1_energy_input():
    with pytest.raises(ValueError, match="energy_edges must be a 1-D astropy Quantity with length greater than 1"):
        thermal.ThermalEmission(6 * u.MK, 1e-5 / u.cm**3)([10] * u.keV)


def test_energy_out_of_range_error():
    with pytest.raises(
        ValueError,
        match="Lower bound of the input energy must be within the range 1.0002920302956426--10.34753795157738 keV. ",
    ):
        thermal.ThermalEmission(6 * u.MK, 1e-5 / u.cm**3)([0.01, 10] * u.keV)


def test_temperature_out_of_range_error():
    with pytest.raises(ValueError, match="All input temperature values must be within the range"):
        thermal.ThermalEmission(0.1 * u.MK, 1e-5 / u.cm**3)([5, 10] * u.keV)


def test_line_energy_out_of_range_warning():
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        _ = thermal.LineEmission(6 * u.MK, 1e-5 / u.cm**3)(np.arange(3, 1000, 0.5) * u.keV)
        assert issubclass(w[0].category, (UserWarning, ResourceWarning))


def test_continuum_energy_out_of_range():
    with pytest.raises(
        ValueError,
        match="Lower bound of the input energy must be within the range 1.0009873438468269--200.15819869050395 keV.",
    ):
        # Use an energy range that goes out of bounds
        # on the lower end--should error
        _ = thermal.ContinuumEmission(6 * u.MK, 1e-5 / u.cm**3)(np.arange(0.1, 28, 0.5) * u.keV)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # The continuum emission should only warn if we go out of
        # bounds on the upper end.
        _ = thermal.ContinuumEmission(6 * u.MK, 1e-5 / u.cm**3)(np.arange(10, 1000, 0.5) * u.keV)
        assert issubclass(w[0].category, (UserWarning, ResourceWarning))


def test_empty_flux_out_of_range():
    """The CHIANTI grid covers ~1 to 300 keV, but the values greater than
    of the grid max energy should all be zeros."""
    energy_edges = np.geomspace(10, 800, num=1000) << u.keV
    midpoints = energy_edges[:-1] + np.diff(energy_edges) / 2

    temperature = 20 << u.MK
    em = 1e-5 << u.cm**-3

    flux = thermal.ThermalEmission(temperature, em)(energy_edges)
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
        model = thermal.ThermalEmission(
            temperature=20 << u.MK,
            emission_measure=(1 << (1e49 * u.cm**-3)),
            mg=thermal.ThermalEmission.mg.default - rng.uniform(),
            si=thermal.ThermalEmission.si.default + rng.uniform(),
            fe=thermal.ThermalEmission.fe.default + rng.uniform(),
        )

        # Apply the model several times;
        # if the DEFAULT_ABUNDANCES get modified, it will be multiplicative
        for _ in range(10):
            model(edges)

        after_models = thermal.DEFAULT_ABUNDANCES[thermal.DEFAULT_ABUNDANCE_TYPE].data
        assert np.allclose(after_models.data, orig.data)
