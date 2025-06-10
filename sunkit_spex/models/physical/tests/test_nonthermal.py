import numpy as np
import pytest

import astropy.units as u

from sunkit_spex.models.physical import nonthermal

SSW_INTENSITY_UNIT = u.ph / u.cm**2 / u.s / u.keV


def thick_target():
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
    energy_edges = np.arange(25, 100.5, 0.5) * u.keV
    observer_distance = (1 * u.AU).to(u.cm)
    # fmt: off
    ssw_output = (
[
    7.0783946, 6.7634241, 6.4662366, 6.1855749, 5.9202898, 5.6693387, 5.4317374, 5.2066098, 4.9931420, 4.7905835,
    4.5982426, 4.4154809, 4.2417082, 4.0763780, 3.9189845, 3.7690581, 3.6261176, 3.4898520, 3.3598352, 3.2357162,
    3.1171676, 3.0038839, 2.8955799, 2.7919889, 2.6928615, 2.5979638, 2.5070771, 2.4191691, 2.3357588, 2.2557761,
    2.1790511, 2.1060428, 2.0353202, 1.9674067, 1.9021683, 1.8394788, 1.7792194, 1.7212775, 1.6655469, 1.6119271,
    1.5603233, 1.5106453, 1.4628081, 1.4167309, 1.3723373, 1.3295548, 1.2883144, 1.2485508, 1.2102052, 1.1732118,
    1.1375180, 1.1030704, 1.0698183, 1.0377134, 1.0067096, 0.97676294, 0.94784785, 0.91989206, 0.89287376, 0.86675665,
    0.84149056, 0.81707363, 0.79345854, 0.77061509, 0.74851438, 0.72712866, 0.70643177, 0.68639828, 0.66700399, 0.64822571,
    0.63004122, 0.61242926, 0.59536941, 0.57884214, 0.56282867, 0.54731103, 0.53227198, 0.51769487, 0.50356491, 0.48986456,
    0.47658033, 0.46369811, 0.45120434, 0.43908599, 0.42733053, 0.41592589, 0.40486049, 0.39412317, 0.38370317, 0.37359016,
    0.36377418, 0.35424549, 0.34499513, 0.33601463, 0.32729421, 0.31882624, 0.31060273, 0.30261598, 0.29485858, 0.28732337,
    0.28000343, 0.27289209, 0.26598319, 0.25926996, 0.25274667, 0.24640755, 0.24024951, 0.23426183, 0.22844209, 0.22278521,
    0.21728632, 0.21194068, 0.20674357, 0.20169089, 0.19677818, 0.19200134, 0.18735636, 0.18283934, 0.17844655, 0.17417436,
    0.17001924, 0.16597776, 0.16204668, 0.15822280, 0.15450299, 0.15088427, 0.14736371, 0.14393852, 0.14060578, 0.14187327,
    0.13420791, 0.13113760, 0.12814974, 0.12524205, 0.12241223, 0.11965801, 0.11697736, 0.11436812, 0.11182821, 0.10935569,
    0.10694881, 0.10460548, 0.10232414, 0.10010273, 0.097939692, 0.095833647, 0.093782356, 0.091785121, 0.089839397, 0.087944192
]
        * SSW_INTENSITY_UNIT * (4 * np.pi * observer_distance**2)
    )
    # fmt: on
    return energy_edges, ssw_output


def thin_target():
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
    energy_edges = np.arange(25, 100.5, 0.5) * u.keV
    observer_distance = (1 * u.AU).to(u.cm)
    # fmt: off
    ssw_output = (
[
    1.8709122, 1.7576846, 1.6530940, 1.5563399, 1.4667092, 1.3835649, 1.3063368, 1.2345137, 1.1676363, 1.1052912,
    1.0471056, 0.99274268, 0.94189773, 0.89429463, 0.84968272, 0.80783411, 0.76854137, 0.73161536, 0.69688338, 0.66418760,
    0.63338351, 0.60433867, 0.57693155, 0.55105048, 0.52659278, 0.50346385, 0.48157652, 0.46084637, 0.44120727, 0.42258623,
    0.40491964, 0.38814855, 0.37221830, 0.35707815, 0.34268095, 0.32898288, 0.31594313, 0.30352370, 0.29168919, 0.28040653,
    0.26964488, 0.25937542, 0.24957119, 0.24020697, 0.23125915, 0.22270560, 0.21452557, 0.20669961, 0.19920943, 0.19203787,
    0.18516879, 0.17858701, 0.17227824, 0.16622903, 0.16042670, 0.15485930, 0.14951556, 0.14438485, 0.13945710, 0.13472284,
    0.13017354, 0.12579979, 0.12159402, 0.11754866, 0.11365650, 0.10991071, 0.10630484, 0.10283275, 0.099488633, 0.096266956,
    0.093162480, 0.090170223, 0.087285451, 0.084503664, 0.081820581, 0.079232128, 0.076734428, 0.074323789, 0.071996694, 0.069749792,
    0.067579888, 0.065483936, 0.063459031, 0.061502398, 0.059611391, 0.057783482, 0.056016257, 0.054307410, 0.052654735, 0.051056124,
    0.049509562, 0.048013120, 0.046564952, 0.045163292, 0.043806448, 0.042492800, 0.041220796, 0.039988949, 0.038795834, 0.037640082,
    0.036520385, 0.035435483, 0.034384171, 0.033365291, 0.032377731, 0.031420423, 0.030492343, 0.029592505, 0.028719964, 0.027873809,
    0.027053167, 0.026257198, 0.025485092, 0.024736072, 0.024009390, 0.023304327, 0.022620189, 0.021956311, 0.021312049, 0.020686673,
    0.020079822, 0.019490800, 0.018919055, 0.018364133, 0.017825357, 0.017302319, 0.016794542, 0.016301568, 0.015822953, 0.015358272,
    0.014907111, 0.014469076, 0.014043785, 0.013630868, 0.013229970, 0.012840749, 0.012462876, 0.012096034, 0.011739917, 0.011394231,
    0.011058695, 0.010733037, 0.010416997, 0.010110328, 0.0098127935, 0.0095241690, 0.0092442439, 0.0089728231, 0.0087097301, 0.0084548167
]
        * SSW_INTENSITY_UNIT * (4 * np.pi * observer_distance**2)
    )
    # fmt: on
    return energy_edges, ssw_output


@pytest.mark.parametrize("ssw", [thick_target])
def test_thick_target_against_ssw(ssw):
    energy_edges, expected = ssw()
    model = nonthermal.ThickTarget()
    output = model(energy_edges)
    expected_value = expected.to_value(output.unit)
    np.testing.assert_allclose(output.value, expected_value, rtol=0.035)


@pytest.mark.parametrize("ssw", [thin_target])
def test_thin_target_against_ssw(ssw):
    energy_edges, expected = ssw()
    model = nonthermal.ThinTarget()
    output = model(energy_edges)
    expected_value = expected.to_value(output.unit)
    np.testing.assert_allclose(output.value, expected_value, rtol=0.035)
