import warnings

import numpy as np
import pytest

import astropy.units as u
import matplotlib.pyplot as plt

from sunkit_spex.models.physical import thermal

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
    observer_distance = (1 * u.AU).to(u.cm)
    mg, al, si, s, ar, ca, fe = 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1
    inputs = (energy_edges, temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type)
    inputs_class = (temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type)

    ssw_output = (
        # [
        #     0.49978617,
        #     0.15907305,
        #     0.052894916,
        #     0.017032871,
        #     0.0056818775,
        #     0.0019678525,
        #     0.00071765773,
        #     0.00026049651,
        #     8.5432410e-05,
        #     3.0082287e-05,
        #     1.0713027e-05,
        #     3.8201970e-06,
        #     1.3680836e-06,
        #     4.9220034e-07,
        #     1.7705435e-07,
        #     6.4002016e-08,
        #     2.3077330e-08,
        #     8.3918881e-09,
        #     3.0453193e-09,
        #     1.1047097e-09,
        #     4.0377532e-10,
        #     1.4734168e-10,
        #     5.3671578e-11,
        #     1.9628120e-11,
        #     7.2107064e-12,
        #     2.6457057e-12,
        #     9.6945607e-13,
        #     3.5472713e-13,
        #     1.3051763e-13,
        #     4.8216642e-14,
        #     1.7797136e-14,
        #     6.5629896e-15,
        #     2.4178513e-15,
        #     8.8982728e-16,
        #     3.2711010e-16,
        #     1.2110889e-16,
        #     4.4997199e-17,
        #     1.6709174e-17,
        #     6.2011332e-18,
        #     2.2999536e-18,
        #     8.5248536e-19,
        #     3.1576185e-19,
        #     1.1687443e-19,
        #     4.3226546e-20,
        #     1.5974677e-20,
        #     5.9133793e-21,
        #     2.2103457e-21,
        #     8.2594126e-22,
        #     3.0853276e-22,
        #     1.1521560e-22,
        # ]
        [0.508525,     0.158277,    0.0529235,    0.0171709,   0.00569773,   0.00196599,  0.000719867,  0.000262074,  8.53937e-05,  3.00764e-05,  1.06980e-05,  3.81875e-06,  1.36751e-06,  4.91036e-07,  1.76798e-07,  6.38010e-08,
  2.30734e-08,  8.36119e-09,  3.03534e-09,  1.10357e-09 , 4.01951e-10,  1.46575e-10,  5.35405e-11,  1.95792e-11,  7.16793e-12,  2.62703e-12,  9.63826e-13,  3.53981e-13,  1.30125e-13,  4.78720e-14,  1.76232e-14,  6.49645e-15,
  2.39605e-15,  8.84008e-16,  3.26543e-16,  1.20678e-16,  4.46146e-17,  1.65130e-17,  6.11063e-18,  2.26457e-18,  8.38897e-19,  3.11188e-19,  1.15416e-19,  4.28376e-20,  1.59088e-20,  5.90513e-21 , 2.19619e-21,  8.16363e-22,
  3.03490e-22,  1.12987e-22]
        * SSW_INTENSITY_UNIT
        * (4 * np.pi * observer_distance**2)
    )
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
    observer_distance = (1 * u.AU).to(u.cm)
    mg, al, si, s, ar, ca, fe = 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1
    inputs = (energy_edges, temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type)
    inputs_class = (temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type)

    ssw_output = (
        [
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
        ]
        * SSW_INTENSITY_UNIT
        * (4 * np.pi * observer_distance**2)
    )
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
    observer_distance = (1 * u.AU).to(u.cm)
    mg, al, si, s, ar, ca, fe = 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1
    inputs = (energy_edges, temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type)
    inputs_class = (temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type)

    ssw_output = (
        [
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
            7.34942e-23,
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
        ]
        # [    0.0637739,
        #      0.0145299,
        #     0.00436042,
        #     0.000531023,
        #     5.58606e-06,
        #     9.94054e-10,
        #     3.14439e-05,
        #     1.97131e-05,
        #     4.93847e-07,
        #     2.66947e-11,
        #     7.63644e-12,
        #     1.18164e-14,
        #     1.12699e-16,
        #     1.72688e-17,
        #     7.27603e-21,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000,
        #     0.00000]
        * SSW_INTENSITY_UNIT
        * (4 * np.pi * observer_distance**2)
    )
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
    # relative_abundances = ((26, 2),)
    observer_distance = (1 * u.AU).to(u.cm)
    mg, al, si, s, ar, ca, fe = 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.4010299956639812
    inputs = (energy_edges, temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type)
    inputs_class = (temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type)
    ssw_output = (
        # [
        #     4.6152353e-01,
        #     1.5266217e-01,
        #     5.1370505e-02,
        #     1.7469261e-02,
        #     6.0074395e-03,
        #     2.0820354e-03,
        #     7.2583189e-04,
        #     2.5462240e-04,
        #     8.9805966e-05,
        #     3.1800522e-05,
        #     1.1323175e-05,
        #     4.0371842e-06,
        #     1.4456124e-06,
        #     5.2003952e-07,
        #     1.8704908e-07,
        #     6.7609605e-08,
        #     2.4375957e-08,
        #     8.8636174e-09,
        #     3.2163083e-09,
        #     1.1666705e-09,
        #     4.2640558e-10,
        #     1.5559361e-10,
        #     5.6675255e-11,
        #     2.0726003e-11,
        #     7.6138887e-12,
        #     2.7935852e-12,
        #     1.0236266e-12,
        #     3.7454293e-13,
        #     1.3780716e-13,
        #     5.0909286e-14,
        #     1.8790933e-14,
        #     6.9294558e-15,
        #     2.5528610e-15,
        #     9.3951687e-16,
        #     3.4537985e-16,
        #     1.2787355e-16,
        #     4.7510902e-17,
        #     1.7642638e-17,
        #     6.5476106e-18,
        #     2.4284929e-18,
        #     9.0014011e-19,
        #     3.3341828e-19,
        #     1.2341190e-19,
        #     4.5645426e-20,
        #     1.6869074e-20,
        #     6.2446062e-21,
        #     2.3341643e-21,
        #     8.7221399e-22,
        #     3.2582179e-22,
        #     1.2167254e-22,
        # ]
#         [          0.526024,     0.167697 ,   0.0557754 ,   0.0180073 ,  0.00601504 ,  0.00208282,  0.000789048,  0.000294182,  9.08375e-05,  3.18172e-05,  1.13295e-05,  4.03960e-06 , 1.44653e-06 , 5.20388e-07,  1.87182e-07,  6.76599e-08 ,2.43951e-08 , 8.87089e-09 , 3.21907e-09,  1.16771e-09,  4.26803e-10,  1.55745e-10,
#   5.67324e-11 , 2.07477e-11 , 7.62212e-12 , 2.79671e-12,  1.02481e-12 , 3.74989e-13,  1.37977e-13 , 5.09739e-14 , 1.88155e-14 , 6.93877e-15,  2.55639e-15,  9.40852e-16 , 3.45882e-16,  1.28065e-16,  4.75835e-17,  1.76703e-17,  6.55814e-18 , 2.43248e-18,  9.01653e-19 , 3.33991e-19 , 1.23629e-19 , 4.57274e-20,
#   1.68999e-20 , 6.25627e-21 , 2.33861e-21 , 8.73910e-22,  3.26466e-22, 1.21918e-22]
  [0.534751 ,    0.166895 ,   0.0558015  ,  0.0181445 ,  0.00603063 ,  0.00208080 , 0.000793823,  0.000297441,  9.08041e-05 , 3.18103e-05 , 1.13134e-05 , 4.03799e-06 , 1.44590e-06,  5.19144e-07,  1.86907e-07, 6.74457e-08 , 2.43905e-08,  8.83818e-09 , 3.20843e-09 , 1.16649e-09,  4.24865e-10,  1.54930e-10,
  5.65926e-11,  2.06955e-11,  7.57671e-12,  2.77691e-12,  1.01884e-12,  3.74194e-13,  1.37559e-13,  5.06084e-14,  1.86312e-14 , 6.86828e-15,  2.53330e-15,  9.34684e-16 , 3.45278e-16 , 1.27608e-16,  4.71785e-17,  1.74628e-17,  6.46241e-18,  2.39505e-18,  8.87281e-19,  3.29154e-19,  1.22086e-19 , 4.53157e-20,
  1.68300e-20,  6.24746e-21,  2.32363e-21,  8.63785e-22 , 3.21138e-22,  1.19564e-22]
        * SSW_INTENSITY_UNIT
        * (4 * np.pi * observer_distance**2)
    )
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
    # relative_abundances = ((26, 2),)
    observer_distance = (1 * u.AU).to(u.cm)
    mg, al, si, s, ar, ca, fe = 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.4010299956639812
    inputs = (energy_edges, temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type)
    inputs_class = (temperature, emission_measure, mg, al, si, s, ar, ca, fe, abundance_type)
    ssw_output = (
        [
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
        ]
        * SSW_INTENSITY_UNIT
        * (4 * np.pi * observer_distance**2)
    )
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


@pytest.mark.parametrize("ssw", [fvth_simple,fvth_Fe2])
def test_thermal_emission_against_ssw(ssw):
    input_args, input_args_class, energy_edges, expected = ssw()
    output = thermal.thermal_emission(*input_args)
    model_class = thermal.ThermalEmission(*input_args_class)
    output_class = model_class(energy_edges)
    # observer_distance = (1 * u.AU).to(u.cm)
    # print(output_class/(4 * np.pi * observer_distance**2))
    expected_value = expected.to_value(output.unit)
    # plt.figure()
    # plt.plot(output_class.value/expected_value)
    # plt.xscale('log')
    # plt.show()
    np.testing.assert_allclose(output.value, expected_value, rtol=0.03)
    np.testing.assert_allclose(output_class.value, expected_value, rtol=0.03)


@pytest.mark.parametrize("ssw", [chianti_kev_cont_simple, chianti_kev_cont_Fe2])
# @pytest.mark.parametrize("ssw", [chianti_kev_cont_simple, fvth_Fe2])
def test_continuum_emission_against_ssw(ssw):
    input_args, input_args_class, energy_edges, expected = ssw()
    output = thermal.continuum_emission(*input_args)
    model_class = thermal.ContinuumEmission(*input_args_class)
    output_class = model_class(energy_edges)
    expected_value = expected.to_value(output.unit)
    np.testing.assert_allclose(output.value, expected_value, rtol=0.03)
    np.testing.assert_allclose(output_class.value, expected_value, rtol=0.03)


# @pytest.mark.parametrize("ssw", [chianti_kev_lines_simple, chianti_kev_lines_Fe2])
# @pytest.mark.parametrize("ssw", [chianti_kev_lines_simple])
# def test_line_emission_against_ssw(ssw):
#     input_args, input_args_class, energy_edges, expected = ssw()
#     output = thermal.line_emission(*input_args)
#     model_class = thermal.LineEmission(*input_args_class)
#     output_class = model_class(energy_edges)
#     expected_value = expected.to_value(output.unit)
#     np.testing.assert_allclose(output.value, expected_value, rtol=0.05, atol=1e-30)
#     np.testing.assert_allclose(output_class.value, expected_value, rtol=0.05, atol=1e-30)


def test_scalar_energy_input():
    with pytest.raises(ValueError, match="energy_edges must be a 1-D astropy Quantity with length greater than 1"):
        thermal.thermal_emission(10 * u.keV, 6 * u.MK, 1e44 / u.cm**3, 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1)
        thermal.ThermalEmission(6 * u.MK, 1e44 / u.cm**3, 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1)(10 * u.keV)


def test_len1_energy_input():
    with pytest.raises(ValueError, match="energy_edges must be a 1-D astropy Quantity with length greater than 1"):
        thermal.thermal_emission([10] * u.keV, 6 * u.MK, 1e44 / u.cm**3, 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1)
        thermal.ThermalEmission(6 * u.MK, 1e44 / u.cm**3, 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1)([10] * u.keV)


def test_energy_out_of_range_error():
    with pytest.raises(
        ValueError,
        match="Lower bound of the input energy must be within the range 1.0002920302956426--200.15819869050395 keV.",
    ):
        thermal.thermal_emission([0.001, 10] * u.keV, 6 * u.MK, 1e44 / u.cm**3, 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1)
        thermal.ThermalEmission(6 * u.MK, 1e44 / u.cm**3, 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1)([0.01, 10] * u.keV)


def test_temperature_out_of_range_error():
    with pytest.raises(ValueError, match="All input temperature values must be within the range"):
        thermal.thermal_emission([5, 10] * u.keV, 0.1 * u.MK, 1e44 / u.cm**3, 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1)
        thermal.ThermalEmission(6 * u.MK, 1e44 / u.cm**3, 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1)([5, 10] * u.keV)


# def test_relative_abundance_negative_input():
#     with pytest.raises(ValueError, match="Relative abundances cannot be negative."):
#         thermal.thermal_emission([5, 10] * u.keV, 10 * u.MK, 1e44 / u.cm**3, 1 * u.AU, relative_abundances=((26, -1)))


# def test_relative_abundance_invalid_atomic_number_input():
#     with pytest.raises(
#         ValueError, match="Relative abundances can only be set for elements with atomic numbers in range"
#     ):
#         thermal.thermal_emission([5, 10] * u.keV, 10 * u.MK, 1e44 / u.cm**3, 1 * u.AU, relative_abundances=((100, 1)))


def test_energy_out_of_range_warning():
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        _ = thermal.line_emission(
            np.arange(3, 1000, 0.5) * u.keV, 6 * u.MK, 1e44 / u.cm**3, 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1
        )
        assert issubclass(w[0].category, UserWarning)


def test_continuum_energy_out_of_range():
    with pytest.raises(ValueError):
        # Use an energy range that goes out of bounds
        # on the lower end--should error
        _ = thermal.continuum_emission(
            np.arange(0.1, 28, 0.5) * u.keV, 6 * u.MK, 1e44 / u.cm**3, 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1
        )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # The continuum emission should only warn if we go out of
        # bounds on the upper end.
        _ = thermal.continuum_emission(
            np.arange(10, 1000, 0.5) * u.keV, 6 * u.MK, 1e44 / u.cm**3, 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1
        )
        assert issubclass(w[0].category, UserWarning)


def test_empty_flux_out_of_range():
    """The CHIANTI grid covers ~1 to 300 keV, but the values greater than
    of the grid max energy should all be zeros."""
    energy_edges = np.geomspace(10, 800, num=1000) << u.keV
    midpoints = energy_edges[:-1] + np.diff(energy_edges) / 2

    temperature = 20 << u.MK
    em = 1e49 << u.cm**-3

    flux = thermal.thermal_emission(energy_edges, temperature, em, 8.15, 7.04, 8.1, 7.27, 6.58, 6.93, 8.1)
    # the continuum is the one we need to check
    max_e = thermal.CONTINUUM_GRID["energy range keV"][1] << u.keV
    should_be_zeros = midpoints >= max_e

    true_zero = 0 * (hopefully_zero := flux[should_be_zeros])
    np.testing.assert_allclose(true_zero, hopefully_zero)
