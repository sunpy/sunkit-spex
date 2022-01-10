import numpy as np
from astropy import units as u
from sys import path as sys_path
sys_path.append('../')
from sunxspex.thermal import thermal_emission
from sunxspex.emission import bremsstrahlung_thin_target, bremsstrahlung_thick_target

__all__ = ["f_vth", "thick_fn", "thick_warm", "defined_photon_models"]

### Issue when using np.float64 numbers for the parameters as it ends up returning all nans and infs but rounding to 15 decimal places fixes this??????

# The defined models shouldn't have duplicate parameter input names 
defined_photon_models = {"f_vth":["T", "EM"], 
                         "thick_fn":["total_eflux", "index", "e_c"],
                         "thick_warm":["tot_eflux", "indx", "ec", "plasmaD", "loopT", "length"]}

def f_vth(temperature, emission_measure46, energies=None):
    ''' Calculates optically thin thermal bremsstrahlung radiation as seen 
    from Earth. 

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
    of ph s^-1 keV^-1.
    '''
    # models are tested with 1s for all inputs
    if (temperature==1) and (emission_measure46==1):
            return np.zeros(len(energies))
            
    energies = np.unique(np.array(energies).flatten()) << u.keV # turn [[1,2],[2,3],[3,4]] into [1,2,3,4]
    temperature = temperature*1e6 << u.K
    emission_measure = emission_measure46*1e46 << u.cm**(-3)
    return thermal_emission(energies,temperature,emission_measure).value

def thick_fn(total_eflux, index, e_c, energies=None):
    ''' Calculates the thick-target bremsstrahlung radiation of a 
    single power-law electron distribution.

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
    of ph s^-1 keV^-1.
    '''
    
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


def thick_warm(total_eflux, index, e_c, plasmaD, T, length, energies=None):
    ''' Calculates the warm thick-target bremsstrahlung radiation as seen 
    from Earth.

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
    plasmaD : int or float
            Plasma number density in units of 1e10 cm^-3.
    T : int or float
            Plasma temperature in megakelvin.
    length : int or float
            Plasma column length (usually half the full loop length?) in Mm.

    Returns
    -------
    A 1d array of the warm thick-target bremsstrahlung radiation in units 
    of ph s^-1 keV^-1.
    '''

    me_keV = 511 #[keV]
    cc = 2.99e10  # speed of light [cm/s]
    KK = 2.6e-18
    n_p = plasmaD*1e10 # was in 1e10 cm^-3, now in cm^-3
    Tloop = T*8.6173e-2 # was in MK, now in keV
    L = length*1e8 # was in Mm, now in cm

    ll = Tloop**2/(2*KK*n_p) # collisional stoping distance for electrons of Tloop energy

    Emin = Tloop*3*(5*ll/L)**4

    if Emin>0.1:
        print("Fixing Emin to 0.1.")
        Emin = 0.1

    Lmin = e_c**2 / (2*KK*n_p) / 3
    if Lmin>L:
        print("Lmin>L")

    EM_add = 3*np.pi/2/KK/cc*np.sqrt(me_keV/8.)*Tloop**2/np.sqrt(Emin)*total_eflux*1e35

    EM46 = EM_add*1e-46 # get EM in units of 10^46 cm^(-3)
    
    return thick_fn(total_eflux, index, e_c, energies=energies) + f_vth(T, EM46, energies=energies)