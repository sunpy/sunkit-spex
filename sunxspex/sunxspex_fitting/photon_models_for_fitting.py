"""
The following code contains the default functions commonly used for solar X-ray spectral fitting in the for f(*args,energies=None).

Used to import `*` into fitter.py but changed as per the Zen of Python, the models here are now imported explicitly into fitter.py
so **remember** to add any new models there too as well as into `defined_photon_models` and `__all__`.
"""

import numpy as np
from astropy import units as u

from ..thermal import thermal_emission
from ..emission import bremsstrahlung_thick_target #bremsstrahlung_thin_target

__all__ = ["defined_photon_models", "f_vth", "thick_fn", "thick_warm"]

### Issue when using np.float64 numbers for the parameters as it ends up returning all nans and infs but rounding to 15 decimal places fixes this??????

# The defined models shouldn't have duplicate parameter input names
defined_photon_models = {"f_vth":["T", "EM"],
                         "thick_fn":["total_eflux", "index", "e_c"],
                         "thick_warm":["tot_eflux", "indx", "ec", "plasma_d", "loop_temp", "length"]}

def f_vth(temperature, emission_measure46, energies=None):
    """ Calculates optically thin thermal bremsstrahlung radiation as seen from Earth.

    [1] https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/f_vth.pro

    Parameters
    ----------
    energies : 2d array
            Array of energy bins for the model to be calculated over.
            E.g., [[1,1.5],[1.5,2],[2,2.5],...].

    temperature : int or float
            Plasma temperature in megakelvin.

    emission_measure46 : int or float
            Emission measure in units of 1e46 cm^-3.

    Returns
    -------
    A 1d array of optically thin thermal bremsstrahlung radiation in units
    of ph s^-1 cm^-2 keV^-1.
    """

    energies = np.unique(np.array(energies).flatten()) << u.keV # turn [[1,2],[2,3],[3,4]] into [1,2,3,4]
    temperature = temperature*1e6 << u.K
    emission_measure = emission_measure46*1e46 << u.cm**(-3)
    return thermal_emission(energies,temperature,emission_measure).value

def thick_fn(total_eflux, index, e_c, energies=None):
    """ Calculates the thick-target bremsstrahlung radiation of a single power-law electron distribution.

    [1] Brown, Solar Physics 18, 489 (1971) (https://link.springer.com/article/10.1007/BF00149070)
    [2] https://hesperia.gsfc.nasa.gov/ssw/packages/xray/doc/brm_thick_doc.pdf
    [3] https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_thicktarget.pro

    Parameters
    ----------
    energies : 2d array
            Array of energy bins for the model to be calculated over.
            E.g., [[1,1.5],[1.5,2],[2,2.5],...].

    total_eflux : int or float
            Total integrated electron flux, in units of 10^35 e^- s^-1.

    index : int or float
            Power-law index of the electron distribution.

    e_c : int or float
            Low-energy cut-off of the electron distribution in units of keV.

    Returns
    -------
    A 1d array of thick-target bremsstrahlung radiation in units
    of ph s^-1 cm^-2 keV^-1.
    """

    hack = np.round([total_eflux, index, e_c], 15)
    total_eflux, index, e_c = hack[0],  hack[1],  hack[2]

    energies = np.mean(energies, axis=1) # since energy bins are given, use midpoints though

    # total_eflux in units of 1e35 e/s
    # single power law so set eebrk==eehigh at a high value, high q also
    output = bremsstrahlung_thick_target(photon_energies=energies,
                                         p=index,
                                         eebrk=150,
                                         q=20,
                                         eelow=e_c,
                                         eehigh=150)*total_eflux*1e35

    output[np.isnan(output)] = 0
    output[~np.isfinite(output)] = 0
    return output


def thick_warm(total_eflux, index, e_c, plasma_d, loop_temp, length, energies=None):
    """ Calculates the warm thick-target bremsstrahlung radiation as seen from Earth.

    [1] Kontar et al, ApJ 2015 (http://adsabs.harvard.edu/abs/2015arXiv150503733K)
    [2] https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/f_thick_warm.pro

    Parameters
    ----------
    energies : 2d array
            Array of energy bins for the model to be calculated over.
            E.g., [[1,1.5],[1.5,2],[2,2.5],...].

    total_eflux : int or float
            Total integrated electron flux in units of 10^35 e^- s^-1.

    index : int or float
            Power-law index of the electron distribution.

    e_c : int or float
            Low-energy cut-off of the electron distribution in units of keV.

    plasma_d : int or float
            Plasma number density in units of 1e10 cm^-3.

    loop_temp : int or float
            Plasma temperature in megakelvin.

    length : int or float
            Plasma column length (usually half the full loop length?) in Mm.

    Returns
    -------
    A 1d array of the warm thick-target bremsstrahlung radiation in units
    of ph s^-1 cm^-2 keV^-1.
    """

    ME_KEV = 511 #[keV]
    CC = 2.99e10  # speed of light [cm/s]
    KK = 2.6e-18
    n_p = plasma_d*1e10 # was in 1e10 cm^-3, now in cm^-3
    tloop = loop_temp*8.6173e-2 # was in MK, now in keV
    l = length*1e8 # was in Mm, now in cm

    ll = tloop**2/(2*KK*n_p) # collisional stopping distance for electrons of Tloop energy

    emin = tloop*3*(5*ll/l)**4

    if emin>0.1:
        print(f"The loop_temp ({loop_temp}), plasma density ({plasma_d}), and loop length ({length}) make emin ({emin}) >0.1. Fixing emin to 0.1.")
        emin = 0.1

    lmin = e_c**2 / (2*KK*n_p) / 3
    if lmin>l:
        print("Minimum length>length")

    em_add = 3*np.pi/2/KK/CC*np.sqrt(ME_KEV/8.)*tloop**2/np.sqrt(emin)*total_eflux*1e35

    em46 = em_add*1e-46 # get EM in units of 10^46 cm^(-3)

    return thick_fn(total_eflux, index, e_c, energies=energies) + f_vth(loop_temp, em46, energies=energies)
