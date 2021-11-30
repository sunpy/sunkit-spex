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

def f_vth(energies, temperature, emission_measure46):
    ''' Calculates...

    Parameters
    ----------
    energies : 2d array
            Array of energy bins for the model to be calculated over. E.g., [[1,1.5],[1.5,2], [2,2.5],...].

    temperature : int or float
            ...

    emission_measure46 : int or float
            ...

    Returns
    -------
    ...
    '''
    energies = np.unique(np.array(energies).flatten()) << u.keV # turn [[1,2],[2,3],[3,4]] into [1,2,3,4]
    temperature = temperature*1e6 << u.K
    emission_measure = emission_measure46*1e46 << u.cm**(-3)
    return thermal_emission(energies,temperature,emission_measure).value

def thick_fn(energies, total_eflux, index, e_c):
    ''' Calculates...

    Parameters
    ----------
    energies : 2d array
            Array of energy bins for the model to be calculated over. E.g., [[1,1.5],[1.5,2], [2,2.5],...].

    total_eflux : int or float
            ...

    index : int or float
            ...

    e_c : int or float
            ...

    Returns
    -------
    ...
    '''
    
    hack = np.round([total_eflux, index, e_c], 15)
    total_eflux, index, e_c = hack[0],  hack[1],  hack[2]
    
    energies = np.mean(energies, axis=1) # since energy bins are given, use midpoints though
    
    # total_eflux in units of 1e35 e/s
    output = bremsstrahlung_thick_target(photon_energies=energies, 
                                         p=index, 
                                         eebrk=150, 
                                         q=20, 
                                         eelow=e_c, 
                                         eehigh=150)*total_eflux*1e35
    output[np.isnan(output)] = 0
    output[~np.isfinite(output)] = 0
    return output


def thick_warm(energies, total_eflux, index, e_c, plasmaD, T, length):
    ''' Calculates...

    Parameters
    ----------
    energies : 2d array
            Array of energy bins for the model to be calculated over. E.g., [[1,1.5],[1.5,2], [2,2.5],...].

    total_eflux : int or float
            ...

    index : int or float
            ...

    e_c : int or float
            ...

    plasmaD : int or float
            ...

    T : int or float
            ...

    length : int or float
            ...

    Returns
    -------
    ...
    '''
    # effectively taken from https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/f_thick_warm.pro

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

    EM_add=3*np.pi/2/KK/cc*np.sqrt(me_keV/8.)*Tloop**2/np.sqrt(Emin)*total_eflux*1e35

    EM46 = EM_add*1e-46 # EM in units of 10^46 cm^(-3)

    return thick_fn(energies, total_eflux, index, e_c) + f_vth(energies, T, EM46)