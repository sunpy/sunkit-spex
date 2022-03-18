"""
Solar X-ray fit functions for use with XSPEC (see https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/extended.html#local-models-in-python)

Nothing in this module requires any existing XSPEC or pyxspec installation

These models can be added via pyxspec:

thick=sun_xspec.ThickTargetModel()
xspec.AllModels.addPyMod(thick.model, thick.ParInfo, 'add')

or XSPEC: (tbd but read the documentation)
"""
import numpy as np
import pandas as pd
#from scipy.special import lpmv
from sunxspex.emission import split_and_integrate, BrokenPowerLawElectronDistribution, bremsstrahlung_cross_section, collisional_loss
from sunxspex import thermal
from sunxspex import constants
from astropy import units as u
import astropy.constants as c
from quadpy.c1 import gauss_legendre
import logging

#logging.basicConfig(filename='xspec.log', level=logging.DEBUG) #for now this helps catch Python errors that are not printed to the terminal by XSPEC

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
        tuples containing photon energies, params
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

        i, = np.where((photon_energies < eehigh) & (photon_energies > 0))

        if i.size > 0:
            try:
                internal_flux[i], iergq[i] = split_and_integrate(model='thick-target', photon_energies=photon_energies[i],maxfcn=maxfcn, rerr=rerr, eelow=eelow,eebrk=eebrk, eehigh=eehigh, p=p, q=q, z=z,efd=False)
            except ValueError:
                return flux #same as previous... might not always want this however

            internal_flux = (fcoeff / decoeff) * internal_flux
            #logging.info(f"PARAMS {params}")
            internal_flux[i]=internal_flux[i]*photon_energies[i]*a0*1e35
            
            #try usual list comprehension one more time... nope doesn't work
            #lf=len(flux)
            #flux=[f for j,f in enumerate(internal_flux) if j < lf]
#            for j in i: #have to modify inplace, not return another pointer
#                if j < len(flux):
#                    flux[j]=internal_flux[j]
            flux[:]=[internal_flux[j] for j in i if j!=i[-1]] #yay finally
                
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
        self.model=self.thin
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
            flux[l], iergq[l] = split_and_integrate2(model='thin-target',
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

##### re-write thick2... use IDL documentation for reference https://hesperia.gsfc.nasa.gov/ssw/packages/xray/doc/brm_thick_doc.pdf

def integrate_part2(*, model, photon_energies, maxfcn, rerr, eelow, eebrk, eehigh,
               p, q, z, a_lg, b_lg, ll, efd): #3x faster than original
    """
    Perform numerical Gaussian-Legendre Quadrature integration for thick- and thin-target models.

    This integration is intended to be performed over continuous portions of the electron
    distribution.

    Parameters
    ----------
    model : `str`
        Either `thick-target` or `thin-target`
    maxfcn : `int`
        Maximum number of points used in Gaussian quadrature integration
    rerr : `float`
        Desired relative error for integral evaluation. For example, rerr = 0.01 indicates that
        the estimate of the integral is to be correct to one digit, whereas rerr = 0.001
        alls for two digits to be correct.
    photon_energies : `numpp.array`
        Photon energies
    eelow : `float`
        Low energy electron cut off
    eebrk : `float`
        Break energy
    eehigh : `float`
        High energy cutoff
    p : `float`
        Slope below the break energy
    q : `float`
        Slope above the break energy
    z : `float`
        Mean atomic number of plasma
    a_lg : `numpy.array`
        Logarithm of lower integration limits
    b_lg : `numpy.array`
        Logarithm of upper integration limit
    ll : `numpy.array`
        Indices for which to carry out integration
    efd: `boolean`
         `True` (default) electron flux density distribution, `False` electron density distribution.
        This input is not used in the main routine, but is passed to thin_target_integrand

    Returns
    -------
    `tuple`
        Array of integrated photon fluxes evaluation and array of integration status (0 converged,
        1 not converged)

    References
    ----------
    See SSW `Brm2_DmlinO_int.pro
    <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_dmlino_int.pro>`_
    and
    `brm2_dmlin.pro <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_dmlin.pro>`_.
    """
    intsum = np.zeros_like(photon_energies, dtype=np.float64)
    ier = np.zeros_like(photon_energies)
    lastsum = np.array(intsum) #faster than copy
   
    nlim=12
    lims=np.stack([a_lg,b_lg])
   
    electron_dist = BrokenPowerLawElectronDistribution(p=p, q=q, eelow=eelow, eebrk=eebrk,eehigh=eehigh)
   
    photon_energy=photon_energies[ll]
    def model_func(y): #basically emission.get_integrand()
        electron_energy=10**y

        gamma = (electron_energy / mc2) + 1.0
        brem_cross = bremsstrahlung_cross_section(electron_energy, photon_energy, z)
        collision_loss = collisional_loss(electron_energy)
        pc = np.sqrt(electron_energy * (electron_energy + 2.0 * mc2))
        density=electron_dist.density(electron_energy)
        integrand=10**y*np.log(10)* density * brem_cross * pc / collision_loss / gamma
        return integrand
   
    for ires in range(2, nlim + 1):
        npoint = 2 ** ires
        if npoint > maxfcn:
            ier[ll] = 1
            break
        scheme=gauss_legendre(npoint)
        intsum[ll]=scheme.integrate(model_func, lims)

        err = np.abs(intsum - lastsum)
        if (err < rerr*np.abs(intsum)).all():
            break
        return intsum, ier
    
def split_and_integrate2(*, model, photon_energies, maxfcn, rerr, eelow, eebrk, eehigh, p, q, z,
                    efd): #10x faster than original
    """
    Split and integrate the continuous parts of the electron spectrum.

    This is used for thin-target calculation from a double power-law electron density distribution
    To integrate a function via the method of Gaussian quadrature. Repeatedly doubles the number of
    points evaluated until convergence, specified by the input rerr, is obtained, or the maximum
    number of points, specified by the input maxfcn, is reached. If integral convergence is not
    achieved, this function raises a ValueError when either the maximum number of function
    evaluations is performed or the number of Gaussian points to be evaluated exceeds maxfcn.
    Maxfcn should be less than or equal to 2^nlim, or 4096 with nlim = 12. This function splits the
    numerical integration into up to three parts and returns the sum of the parts. This avoids
    numerical problems with discontinuities in the electron distribution function at eelow and
    eebrk.

    Parameters
    ----------
    model : `str`
        Electron model either `thick-target` or `thin-target`
    photon_energies : `numpy.array`
        Array containing lower integration limits
    maxfcn : `int`
        Maximum number of points used in Gaussian quadrature integration
    rerr : `float`
        Desired relative error for integral evaluation
    eelow : `float`
        Low energy electron cutoff
    eebrk : `float`
        Break energy
    eehigh : `float`
        High energy electron cutoff
    p : `float`
        Slope below the break energy
    q : `float`
        Slope above the break energy
    z : `float`
        Mean atomic number of plasma
    efd : `bool`
        True - electron flux density distribution, False - electron density distribution. This
        input is not used in the main routine, but is passed to Brm_Fthin()

    Returns
    -------
    `tuple`
        (DmlinO, irer) Array of integral evaluation and array of error flags

    References
    ----------
    Initial version modified from SSW
    `Brm2_DmlinO <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_dmlino.pro>`_ and
    `Brm2_Dmlin <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_dmlin.pro>`_.

    """
    mc2 = const.get_constant('mc2')
    clight = const.get_constant('clight')

    if not eelow <= eebrk <= eehigh:
        raise ValueError(f'Condition eelow <= eebrek <= eehigh not satisfied '
                         f'({eelow}<={eebrk}<={eehigh}).')

    # Create arrays for integral sums and error flags.
    intsum = np.zeros_like(photon_energies, dtype=np.float64)
    ier = np.zeros_like(photon_energies, dtype=np.float64)
#    intsum2 = np.zeros_like(photon_energies, dtype=np.float64)
#    ier2 = np.zeros_like(photon_energies, dtype=np.float64)
#    intsum3 = np.zeros_like(photon_energies, dtype=np.float64)
#    ier3 = np.zeros_like(photon_energies, dtype=np.float64)

    P1 = np.where(photon_energies < eelow)[0] #do this in the loop....
    P2 = np.where(photon_energies < eebrk)[0]
    P3 = np.where(photon_energies <= eehigh)[0]

    total_integral=0
    # Part 1, below en_val[0] (usually eelow)
    if model == 'thick-target':
        for n,(part,ulim) in enumerate(zip([P1,P2,P3],[eelow,eebrk,eehigh])):
            if part.size > 0:
        
                a_lg = np.log10(photon_energies[part])
                b_lg = np.log10(np.full_like(a_lg, ulim))
                i = np.array(part) #faster than copy
                intsum, ier = integrate_part2(model=model, maxfcn=maxfcn, rerr=rerr,
                                           photon_energies=photon_energies,
                                           eelow=eelow, eebrk=eebrk, eehigh=eehigh, p=p, q=q, z=z,
                                           a_lg=a_lg, b_lg=b_lg, ll=i, efd=efd)
                total_integral=np.add(total_integral,intsum)
                # ier = 1 indicates no convergence.
                if sum(ier):
                    raise ValueError(f'Part {n} integral did not converge for some photon energies.')

    # Part 2, between enval[0] and en_val[1](usually eelow and eebrk)

#    aa = np.copy(photon_energies)
###### probaly need to keep these checks!!! ##########
#    if (P2.size > 0) and (eebrk > eelow):
#        # TODO check if necessary as integration should only be carried out over point P2 which
#        # by definition are not in P1
#        if P1.size > 0:
#            aa[P1] = eelow
#
#        #print('Part2')
#        a_lg = np.log10(aa[P2])
#        b_lg = np.log10(np.full_like(a_lg, eebrk))
#        i = np.copy(P2)
#        intsum2, ier2 = integrate_part(model=model, maxfcn=maxfcn, rerr=rerr,
#                                       photon_energies=photon_energies,
#                                       eelow=eelow, eebrk=eebrk, eehigh=eehigh, p=p, q=q, z=z,
#                                       a_lg=a_lg, b_lg=b_lg, ll=i, efd=efd)
#
#        if sum(ier2) > 0:
#            raise ValueError('Part 2 integral did not converge for some photon energies.')
#
#    # Part 3: between eebrk and eehigh(usually eebrk and eehigh)
#    aa = np.copy(photon_energies)
#    if (P3.sum() > 0) and (eehigh > eebrk):
#        if P2.size > 0:
#            aa[P2] = eebrk
#
#        #print('Part3')
#        a_lg = np.log10(aa[P3])
#        b_lg = np.log10(np.full_like(a_lg, eehigh))
#        i = np.copy(P3)
#        intsum3, ier3 = integrate_part(model=model, maxfcn=maxfcn, rerr=rerr,
#                                       photon_energies=photon_energies,
#                                       eelow=eelow, eebrk=eebrk, eehigh=eehigh, p=p, q=q, z=z,
#                                       a_lg=a_lg, b_lg=b_lg, ll=i, efd=efd)
#        if sum(ier3) > 0:
#            raise ValueError('Part 3 integral did not converge for some photon energies.')

    # TODO check units here
    # Combine 3 parts and convert units and return
    if model == 'thick-target':
        DmlinO = total_integral * (mc2 / clight)
        #ier = ier1 + ier2 + ier3
        return DmlinO, ier
    elif model == 'thin-target':
        Dmlin = (intsum2 + intsum3)
        ier = ier2 + ier3
        return Dmlin, ier
