"""
Solar X-ray fit functions for use with XSPEC (see https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/extended.html#local-models-in-python)

Nothing in this module requires any existing XSPEC or pyxspec installation. For Jupyter-friendly display and plot options for use with pyxspec, see https://raw.githubusercontent.com/elastufka/solar_all_purpose/main/xspec_utils.py

These models can be added via pyxspec:

thick=sun_xspec.ThickTargetModel()
xspec.AllModels.addPyMod(thick.model, thick.ParInfo, 'add')

or XSPEC: (tbd but read the documentation)

Fixes #57
"""
import numpy as np
import pandas as pd
from sunxspex.emission import _split_and_integrate as split_and_integrate
from sunxspex import thermal
from sunxspex import constants
from astropy import units as u

import logging
# For now this helps catch Python errors that are not printed to the terminal by XSPEC
logging.basicConfig(filename='xspec.log', level=logging.DEBUG)

# Central constant management
CONST = constants.Constants()


class XspecModel:
    '''Base class for Xspec models. Must include the model function and initial parameters

    These models can be added via pyxspec:
    import xspec
    thick=ThickTargetModel()
    xspec.AllModels.addPyMod(thick.model, thick.ParInfo, 'add')'''

    def __init__(self):
        self.ParInfo = ''
        self.model = None
        self.description = None

    def __repr__(self):
        return self.description

    def print_ParInfo(self):
        '''print parameter info in a readable format with headers '''
        headers = "Parameter,Unit,Default,Hard Min,Soft Min,Soft Max,Hard Max,Delta".split(',')
        partable = np.array([[p.split('  ') for p in self.ParInfo]]
                            ).reshape((len(self.ParInfo), 8)).T
        df = pd.DataFrame({h: r for h, r in zip(headers, partable)},
                          index=range(1, len(self.ParInfo) + 1))
        return df  # Table should be small enough to print well in both terminal and Jupyter

    def other_method(self):
        '''such as: easily set parameter defaults from a tuple or dictionary, descriptions of parameters, etc'''
        raise NotImplementedError


class ThickTargetModel(XspecModel):
    '''Thick-target bremsstrahlung model for use in Xspec. Default parameters are taken from OSPEX [f_thick2_defaults.pro](https://hesperia.gsfc.nasa.gov/ssw/packages/spex/idl/object_spex/f_thick2_defaults.pro)'''

    def __init__(self):
        self.ParInfo = (
            #"a0  1e-35  100.0  1.0  10.0  1e6  1e7  1.0",
            "p  \"\"  4.0  1.1  1.5  15.0  20.0  0.1",
            "eebrk  keV  150.0  1.0  5.0  100.  1e5  0.5",
            "q  \"\"  6.0  0.0  1.5  15.0  20.0  0.1",
            "eelow  keV  20.0  0.0  1.0  100.  1e3  1.0",
            "eehigh  keV  3200.0  1.0  10.0  1e6  1e7  1.0"
        )
        self.model = self.bremsstrahlung_thick_target
        self.description = f"Thick-target bremsstrahlung '{self.model.__name__}'"

    @staticmethod
    def bremsstrahlung_thick_target(photon_energies, params, flux):
        """
        tuples or lists containing photon energies, params
              flux is empty list of length nE-1

              The input array of energy bins gives the boundaries of the energy bins
              and hence has one more entry than the output flux arrays.

              The output flux array for an additive model should be in terms of photons/cm$^2$/s
              (not photons/cm$^2$/s/keV) i.e. it is the model spectrum integrated over the energy bin.
        """
        photon_energies = np.array(photon_energies)
        photon_energy_bins = photon_energies[1:] - photon_energies[:-1]
        photon_energy_bin_centers = (
            photon_energies[1:] - photon_energies[:-1]) / 2 + photon_energies[:-1]  # Assumes linear spacing
        internal_flux = np.zeros(photon_energy_bin_centers.shape)

        # Constants
        mc2 = CONST.get_constant('mc2')
        clight = CONST.get_constant('clight')
        au = CONST.get_constant('au')
        r0 = CONST.get_constant('r0')

        # Max number of points
        maxfcn = 2048

        # Average atomic number
        z = 1.2

        # Relative error
        rerr = 1e-4

        # Numerical coefficient for photon flux
        fcoeff = ((clight ** 2 / mc2 ** 4) / (4 * np.pi * au ** 2))
        decoeff = 4.0 * np.pi * (r0 ** 2) * clight

        try:
            # Sometimes model function is called without norm factor, sometimes with
            p, eebrk, q, eelow, eehigh, _ = params
        except ValueError:
            p, eebrk, q, eelow, eehigh = params

        if eelow < eehigh:  # If eelow >=eehigh, flux remains unchanged
            i, = np.where(np.logical_and(photon_energy_bin_centers <
                          eehigh, photon_energy_bin_centers > 0))

            if i.size > 0:
                try:
                    internal_flux[i], _ = split_and_integrate(model='thick-target', photon_energies=photon_energy_bin_centers[i],
                                                              maxfcn=maxfcn, rerr=rerr, eelow=eelow, eebrk=eebrk, eehigh=eehigh, p=p, q=q, z=z, efd=False)
                except ValueError:
                    return flux  # Same as previous iteration result

                # Parameter norm will have units 1e35. Equivalent to a0, total electron density
                internal_flux = (fcoeff / decoeff) * internal_flux * 1e35

                # Modify inplace, not return another pointer
                flux[:] = internal_flux * photon_energy_bins


class ThinTargetModel(XspecModel):
    '''Thin-target bremsstrahlung model for use in Xspec. Default parameters are taken from OSPEX [f_thin2_defaults.pro](https://hesperia.gsfc.nasa.gov/ssw/packages/spex/idl/object_spex/f_thin2_defaults.pro)'''

    def __init__(self):
        self.ParInfo = (
            #"a0  1e-35  100.0  1e-10  1e-9  1e14  1e15  1.0",
            "p  \"\"  2.0  0.1  1.0  15.0  20.0  0.1",
            "eebrk  keV  100.0  10.0  20.0  100.  500  0.5",
            "q  \"\"  2.0  0.0  1.5  15.0  20.0  0.1",
            "eelow  keV  10.0  1.0  2.0  100.  1e3  1.0",
            "eehigh  keV  3200.0  100.0  500.0  1e6  1e7  1.0"
        )
        self.model = self.bremsstrahlung_thin_target
        self.description = f"Thin-target bremsstrahlung '{self.model.__name__}'"

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
        photon_energies = np.array(photon_energies)
        photon_energy_bins = photon_energies[1:] - photon_energies[:-1]
        photon_energy_bin_centers = (
            photon_energies[1:] - photon_energies[:-1]) / 2 + photon_energies[:-1]
        internal_flux = np.zeros(photon_energy_bin_centers.shape)

        mc2 = CONST.get_constant('mc2')
        clight = CONST.get_constant('clight')
        au = CONST.get_constant('au')

        # Max number of points
        maxfcn = 2048

        # Average atomic number
        z = 1.2

        # Relative error
        rerr = 1e-4

        # Numerical coefficient for photo flux
        fcoeff = (clight / (4 * np.pi * au ** 2)) / mc2 ** 2.

        try:
            p, eebrk, q, eelow, eehigh, _ = params
        except ValueError:
            p, eebrk, q, eelow, eehigh = params

        if eelow < eehigh:

            i, = np.where(np.logical_and(photon_energy_bin_centers <
                          eehigh, photon_energy_bin_centers > 0))

            if i.size > 0:
                try:
                    internal_flux[i], _ = split_and_integrate(model='thin-target', photon_energies=photon_energy_bin_centers[i],
                                                              maxfcn=maxfcn, rerr=rerr, eelow=eelow, eebrk=eebrk, eehigh=eehigh, p=p, q=q, z=z, efd=True)
                except ValueError:
                    return flux

                flux[:] = internal_flux * photon_energy_bins * 1e35


class ThermalModel(XspecModel):
    '''Thermal bremsstrahlung model for use in Xspec. Default parameters are taken from OSPEX [link](). XSPEC's _apec_ model is more commonly used for thermal bremsstrahlung.'''

    def __init__(self):
        self.ParInfo = (
            "EM  1e49  1.0  1e-20  1e-19  1e19  1e20  1.0",
            "kT  keV  2.0  1.  1.5  7.0  8.0  0.1",
            "abund  \"\"  1.0  0.01  0.1  9.0  10.0  0.1")
        self.model = self.vth
        self.description = f"Thermal bremsstrahlung model '{self.model.__name__}'"
        global CONTINUUM_GRID
        CONTINUUM_GRID = thermal.setup_continuum_parameters()  # I don't like this...
        global LINE_GRID
        LINE_GRID = thermal.setup_line_parameters()
        # self.observer_distance=(1*u.AU).to(u.cm).value

    @staticmethod
    def vth(energy_edges, params, flux):
        # Convert inputs to known units and confirm they are within range.
        emission_measure, temperature, abund, _ = params
        emission_measure *= 1e49
        observer_distance = (1 * u.AU).to(u.cm).value

        energy_edges_keV, temperature_K = np.array(energy_edges), np.array(
            [((temperature * u.keV).to(u.J) / (c.k_B)).value])

        energy_range = (min(CONTINUUM_GRID["energy range keV"][0], LINE_GRID["energy range keV"][0]),
                        max(CONTINUUM_GRID["energy range keV"][1], LINE_GRID["energy range keV"][1]))
        #_error_if_input_outside_valid_range(energy_edges_keV, energy_range, "energy", "keV")
        temp_range = (min(CONTINUUM_GRID["temperature range K"][0], LINE_GRID["temperature range K"][0]),
                      max(CONTINUUM_GRID["temperature range K"][1], LINE_GRID["temperature range K"][1]))
        #_error_if_input_outside_valid_range(temperature_K, temp_range, "temperature", "K")
        # Calculate abundances
        abundances = thermal._calculate_abundances("sun_coronal_ext", None)
        # Calculate fluxes.
        continuum_flux = thermal._continuum_emission(
            energy_edges_keV, temperature_K, abundances)  # get rid of units
        line_flux = thermal._line_emission(energy_edges_keV, temperature_K, abundances)
        internal_flux = ((continuum_flux + line_flux) * emission_measure /
                         (4 * np.pi * observer_distance**2)).flatten()

        #logging.info(f"PARAMS: {params}")
        flux[:] = internal_flux.value
