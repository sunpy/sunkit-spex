"""
Solar X-ray fit functions for use with XSPEC (see https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/extended.html#local-models-in-python)

Nothing in this module requires any existing XSPEC or pyxspec installation. For Jupyter-friendly display and plot options for use with pyxspec, see https://raw.githubusercontent.com/elastufka/solar_all_purpose/main/xspec_utils.py

These models can be added via pyxspec:

thick=sun_xspec.ThickTargetModel()
xspec.AllModels.addPyMod(thick.model, thick.ParInfo, 'add')

or XSPEC: (tbd but read the documentation)
"""
import numpy as np
import pandas as pd
#from scipy.special import lpmv
from sunxspex.emission import split_and_integrate, split_and_integrate0 #, BrokenPowerLawElectronDistribution, bremsstrahlung_cross_section, collisional_loss
from sunxspex import thermal
from sunxspex import constants
from astropy import units as u
#import astropy.constants as c
#from quadpy.c1 import gauss_legendre
from datetime import datetime as dt
import logging

logging.basicConfig(filename='xspec.log', level=logging.DEBUG) #for now this helps catch Python errors that are not printed to the terminal by XSPEC

# Central constant management
global const #whyyyyyy there has got to be a better way to do this
const = constants.Constants()

class XspecModel:
    def __init__(self):
        self.ParInfo=''
        self.model=None
        self.description=None
        #should I deal with constants here?

    def __repr__(self):
        return self.description

    def print_ParInfo(self):
        '''print parameter info in a readable format with headers '''
        headers="Parameter,Unit,Default,Hard Min,Soft Min,Soft Max,Hard Max,Delta".split(',')
        partable=np.array([[p.split('  ') for p in self.ParInfo]]).reshape((len(self.ParInfo),8)).T
        df=pd.DataFrame({h:r for h,r in zip(headers,partable)}, index=range(1,len(self.ParInfo)+1))
        return df #prints nicely like this...

    def other_method(self):
        '''such as: easily set parameter defaults from a tuple or dictionary, descriptions of parameters, etc'''
        raise NotImplementedError


class ThickTargetModel0(XspecModel): #original integration for comparison
    def __init__(self):
        self.ParInfo=(
        "a0  1e-35  100.0  1.0  10.0  1e6  1e7  1.0",
        "p  \"\"  4.0  1.1  1.5  15.0  20.0  0.1",
           "eebrk  keV  150.0  1.0  5.0  100.  1e5  0.5" ,
         "q  \"\"  6.0  0.0  1.5  15.0  20.0  0.1",
         "eelow  keV  20.0  0.0  1.0  100.  1e3  1.0" ,
         "eehigh  keV  3200.0  1.0  10.0  1e6  1e7  1.0"
        ) #default parameters from OSPEX
        self.model=self.thick2_original
        self.description=f"Thick-target bremsstrahlung '{self.model.__name__}'"

    @staticmethod
    def thick2_original(photon_energies, params, flux):
        """
        tuples or lists containing photon energies, params
              flux is empty list of length nE-1

              The input array of energy bins gives the boundaries of the energy bins
              and hence has one more entry than the output flux arrays.

              The output flux array for an additive model should be in terms of photons/cm$^2$/s
              (not photons/cm$^2$/s/keV) i.e. it is the model spectrum integrated over the energy bin.
        """

        photon_energies=np.array(photon_energies)
        internal_flux=np.zeros(photon_energies.shape)

        # Constants
        #const=constants.Constants()
        mc2 = const.get_constant('mc2')
        clight = const.get_constant('clight')
        au = const.get_constant('au')
        r0 = const.get_constant('r0')

        # Max number of points
        maxfcn = 2048

        # Average atomic number
        z = 1.2

        # Relative error
        rerr = 1e-4

        # Numerical coefficient for photo flux
        fcoeff = ((clight ** 2 / mc2 ** 4) / (4 * np.pi * au ** 2))
        decoeff = 4.0 * np.pi * (r0 ** 2) * clight

        # Create arrays for the photon flux and error flags.
        #flux = np.zeros_like(photon_energies, dtype=np.float64)
        iergq = np.zeros_like(photon_energies, dtype=np.float64)
        try:
            a0,p, eebrk, q, eelow, eehigh,_=params #why is this not consistent within xspec?
        except ValueError:
            a0,p, eebrk, q, eelow, eehigh=params

        if eelow >= eehigh:
            return list(photon_energies)[1:]

        i, = np.where(np.logical_and(photon_energies < eehigh, photon_energies > 0)) #somewhat faster than &

        if i.size > 0:
            #logging.info(f"len(i)={i.size}, i[-1]={i[-1]}")
            try:
                internal_flux[i], iergq[i] = split_and_integrate0(model='thick-target', photon_energies=photon_energies[i],maxfcn=maxfcn, rerr=rerr, eelow=eelow,eebrk=eebrk, eehigh=eehigh, p=p, q=q, z=z,efd=False)
            except ValueError:
                flux[:]=[f*1e-10 for f in flux] #sometimes making it smaller might be better... but returning the original flux can also be misleading...have to figure out what to do about this
                return flux

            internal_flux = (fcoeff / decoeff) * internal_flux
            #logging.info(f"{dt.now()} PARAMS {params}")
            internal_flux[i]=internal_flux[i]*photon_energies[i]*a0*1e35
            #have to modify inplace, not return another pointer
            flux[:]=[internal_flux[j] if j in i and j!=i[-1] else prev for j,prev in enumerate(flux)]
            #logging.info(f"{flux[:20]}")

class ThickTargetModel(XspecModel):
    def __init__(self):
        self.ParInfo=(
        "a0  1e-35  100.0  1.0  10.0  1e6  1e7  1.0",
        "p  \"\"  4.0  1.1  1.5  15.0  20.0  0.1",
           "eebrk  keV  150.0  1.0  5.0  100.  1e5  0.5" ,
         "q  \"\"  6.0  0.0  1.5  15.0  20.0  0.1",
         "eelow  keV  20.0  0.0  1.0  100.  1e3  1.0" ,
         "eehigh  keV  3200.0  1.0  10.0  1e6  1e7  1.0"
        ) #default parameters from OSPEX
        self.model=self.thick2
        self.description=f"Thick-target bremsstrahlung '{self.model.__name__}'"

    @staticmethod
    def thick2(photon_energies, params, flux):
        """
        tuples or lists containing photon energies, params
              flux is empty list of length nE-1

              The input array of energy bins gives the boundaries of the energy bins
              and hence has one more entry than the output flux arrays.

              The output flux array for an additive model should be in terms of photons/cm$^2$/s
              (not photons/cm$^2$/s/keV) i.e. it is the model spectrum integrated over the energy bin.
        """

        photon_energies=np.array(photon_energies)
        internal_flux=np.zeros(photon_energies.shape)

        # Constants
        #const=constants.Constants()
        mc2 = const.get_constant('mc2')
        clight = const.get_constant('clight')
        au = const.get_constant('au')
        r0 = const.get_constant('r0')

        # Max number of points
        maxfcn = 2048

        # Average atomic number
        z = 1.2

        # Relative error
        rerr = 1e-4

        # Numerical coefficient for photo flux
        fcoeff = ((clight ** 2 / mc2 ** 4) / (4 * np.pi * au ** 2))
        decoeff = 4.0 * np.pi * (r0 ** 2) * clight

        # Create arrays for the photon flux and error flags.
        #flux = np.zeros_like(photon_energies, dtype=np.float64)
        iergq = np.zeros_like(photon_energies, dtype=np.float64)
        try:
            a0,p, eebrk, q, eelow, eehigh,_=params #why is this not consistent within xspec?
        except ValueError:
            a0,p, eebrk, q, eelow, eehigh=params

        #if eelow >= eehigh:
        #    return list(photon_energies)[1:]

        i, = np.where(np.logical_and(photon_energies < eehigh, photon_energies > 0)) #somewhat faster than &

        if i.size > 0:
            #logging.info(f"len(i)={i.size}, i[-1]={i[-1]}")
            try:
                internal_flux[i], iergq[i] = split_and_integrate(model='thick-target', photon_energies=photon_energies[i],maxfcn=maxfcn, rerr=rerr, eelow=eelow,eebrk=eebrk, eehigh=eehigh, p=p, q=q, z=z,efd=False)
            except ValueError:
                return flux #same as previous... might not always want this however

            internal_flux = (fcoeff / decoeff) * internal_flux

            #logging.info(f"PARAMS {params}")
            logging.info(f"{dt.now()} PARAMS {params}")

            internal_flux[i]=internal_flux[i]*photon_energies[i]*a0*1e35
            #have to modify inplace, not return another pointer
            flux[:]=[internal_flux[j] if j in i and j!=i[-1] else prev for j,prev in enumerate(flux)]
            #logging.info(f"{flux[:20]}")


class ThinTargetModel(XspecModel):
    def __init__(self):
        self.ParInfo=(
        "a0  1e-35  100.0  1.0  10.0  1e6  1e7  1.0",
        "p  \"\"  4.0  1.1  1.5  15.0  20.0  0.1",
           "eebrk  keV  150.0  1.0  5.0  100.  1e5  0.5" ,
         "q  \"\"  6.0  0.0  1.5  15.0  20.0  0.1",
         "eelow  keV  20.0  0.0  1.0  100.  1e3  1.0" ,
         "eehigh  keV  3200.0  1.0  10.0  1e6  1e7  1.0"
        ) #default parameters from OSPEX
        self.model=self.bremsstrahlung_thin_target
        self.description=f"Thin-target bremsstrahlung '{self.model.__name__}'"

    @staticmethod
    def bremsstrahlung_thin_target(photon_energies, params, flux):
        """
        Computes the thin-target bremsstrahlung x-ray/gamma-ray spectrum from an isotropic electron
        distribution function provided in `broken_powerlaw`. The units of the computed flux is photons
        per second per keV per square centimeter.

        The electron flux distribution function is a double power law in electron energy with a
        low-energy cutoff and a high-energy cutoff.

        Parameters
        ----------
        photon_energies : `numpy.array`
            Array of photon energies to evaluate flux at
        p : `float`
            Slope below the break energy
        eebrk : `float`
            Break energy
        q : `float`
            Slope above the break energy
        eelow : `float`
            Low energy electron cut off
        eehigh : `float`
            High energy electron cut off
        efd : `bool`
            True (default) - input electron distribution is electron flux density distribution
            (unit electrons cm^-2 s^-1 keV^-1),
            False - input electron distribution is electron density distribution.
            (unit electrons cm^-3 keV^-1),
            This input is not used in the main routine, but is passed to brm2_dmlin and Brm2_Fthin

        Returns
        -------
        flux: `numpy.array`
            Multiplying the output of Brm2_ThinTarget by a0 gives an array of
            photon fluxes in photons s^-1 keV^-1 cm^-2, corresponding to the photon energies in the
            input array eph. The detector is assumed to be 1 AU rom the source. The coefficient a0 is
            calculated as a0 = nth * V * nnth, where nth: plasma density; cm^-3) V:
            volume of source; cm^3) nnth: Integrated nonthermal electron flux density (cm^-2 s^-1), if
            efd = True, or Integrated electron number density (cm^-3), if efd = False

        Notes
        -----
        If you want to plot the derivative of the flux, or the spectral index of the photon spectrum as
        a function of photon energy, you should set RERR to 1.d-6, because it is more sensitive to RERR
        than the flux.

        Adapted from SSW `Brm2_ThinTarget
        <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_thintarget.pro>`_
        """
        mc2 = const.get_constant('mc2')
        clight = const.get_constant('clight')
        au = const.get_constant('au')

        # Max number of points
        maxfcn = 2048

        # Average atomic number
        z = 1.2

        # Relative error
        rerr = 1e-4

        # Numerical coefficient for photo flux
        fcoeff = (clight / (4 * np.pi * au ** 2)) / mc2 ** 2.

        # Create arrays for the photon flux and error flags.
        flux = np.zeros_like(photon_energies, dtype=np.float64)
        iergq = np.zeros_like(photon_energies, dtype=np.float64)

        if eelow >= eehigh:
            raise ValueError('eehigh must be larger than eelow!')

        l, = np.where((photon_energies < eehigh) & (photon_energies > 0))
        if l.size > 0:
            flux[l], iergq[l] = split_and_integrate(model='thin-target',
                                                    photon_energies=photon_energies[l], maxfcn=maxfcn,
                                                    rerr=rerr, eelow=eelow, eebrk=eebrk, eehigh=eehigh,
                                                    p=p, q=q, z=z, efd=efd)

            flux *= fcoeff

            return flux
        else:
            raise Warning('The photon energies are higher than the highest electron energy or not '
                          'greater than zero')


class ThermalModel(XspecModel):
    def __init__(self):
        self.ParInfo=(
        "EM  1e49  1.0  1e-20  1e-19  1e19  1e20  1.0",
        "kT  keV  2.0  1.  1.5  7.0  8.0  0.1",
        "abund  \"\"  1.0  0.01  0.1  9.0  10.0  0.1") #default parameters from OSPEX
        self.model=self.vth
        self.description=f"Thermal bremsstrahlung model '{self.model.__name__}'"
        global CONTINUUM_GRID
        CONTINUUM_GRID = thermal.setup_continuum_parameters()
        global LINE_GRID
        LINE_GRID= thermal.setup_line_parameters()
        #self.observer_distance=(1*u.AU).to(u.cm).value

    @staticmethod
    def vth(energy_edges,params,flux):
        # Convert inputs to known units and confirm they are within range.
        emission_measure,temperature,abund,_=params
        emission_measure*=1e49
        observer_distance=(1*u.AU).to(u.cm).value

        energy_edges_keV, temperature_K = np.array(energy_edges), np.array([((temperature*u.keV).to(u.J)/(c.k_B)).value])

        energy_range = (min(CONTINUUM_GRID["energy range keV"][0], LINE_GRID["energy range keV"][0]),
                        max(CONTINUUM_GRID["energy range keV"][1], LINE_GRID["energy range keV"][1]))
        #_error_if_input_outside_valid_range(energy_edges_keV, energy_range, "energy", "keV")
        temp_range = (min(CONTINUUM_GRID["temperature range K"][0], LINE_GRID["temperature range K"][0]),
                      max(CONTINUUM_GRID["temperature range K"][1], LINE_GRID["temperature range K"][1]))
        #_error_if_input_outside_valid_range(temperature_K, temp_range, "temperature", "K")
        # Calculate abundances
        abundances = thermal._calculate_abundances("sun_coronal_ext", None)
        # Calculate fluxes.
        continuum_flux = thermal._continuum_emission(energy_edges_keV, temperature_K, abundances) #get rid of units
        line_flux = thermal._line_emission(energy_edges_keV, temperature_K, abundances)
        internal_flux = ((continuum_flux +line_flux) * emission_measure /
                    (4 * np.pi * observer_distance**2)).flatten()

        logging.info(f"PARAMS: {params}")
        flux[:]=internal_flux.value
