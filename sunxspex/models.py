# -*- coding: utf-8 -*-
"""
A module to hold models useful for fitting solar X-ray spectra.
"""

import astropy.units as u
from astropy.modeling import models

"""
class SolarBremsstrahlungThermalContinuum(models.Fittable1DModel):
    '''
    Model of thermal X-ray bremsstrahlung emission.

    If energy array starts at >8 keV, then noline is set to calculate pure
    free-free continuum from either chianti_kev or mewe_kev.
    If edges weren't passed as 2xn array, use Brem_49 function.

    Parameters
    ----------
    temperature: `astropy.units.Quantity`
        Temperature or temperatures of the emitting plasma.
        If multiple temperatures desired, length can be >1.

    em: `astropy.units.Quantity`
        The emission measure of the emitting plasma at each temperature
        supplied in the temperature input.

    '''
    return (1.e8/9.26) * float(acgaunt(12.3985/E, KT0/.08617)) *exp(-(E/KT0 < 50)) /E / KT0^.5
"""
