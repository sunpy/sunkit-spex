"""
Functions for computing the photon flux due to bremsstrahlung radiation from energetic electrons
impacting a dense plasma. See [1]_ and [2]_.


References
----------

.. [1] Thick-Target: https://hesperia.gsfc.nasa.gov/hessi/flarecode/bremthickdoc.pdf
.. [2] Thin-Target: https://hesperia.gsfc.nasa.gov/hessi/flarecode/bremdoc.pdf

"""
import logging

import numpy as np

from sunkit_spex.legacy import constants as const
from sunkit_spex.legacy.integrate import gauss_legendre

const = const.Constants()

logging = logging.getLogger(__name__)

__all__ = ['BrokenPowerLawElectronDistribution', 'collisional_loss', '_get_integrand',
           '_integrate_part', '_split_and_integrate', 'bremsstrahlung_thin_target',
           'bremsstrahlung_thick_target']


class BrokenPowerLawElectronDistribution:
    """
    A broken or double power law electron flux distribution and integral.

    This class is intended to be use with `sunpy.emission.bremsstrahlung_thin_target` and
    `bremsstrahlung_thick_target`.

    Parameters
    ----------
    p : `float`
        Power law index below the break energy `ebrk`
    q : `float`
        Power law index below the break energy `ebrk`
    eelow : `float`
        Low energy cutoff
    eebrk : `float`
        Break energy
    eehigh : `float`
        High energy cutoff
    norm : `bool` (optional)
        True (default) distribution function is normalized so that the integral from `eelow` to
        `eehigh` is 1.

    References
    ----------
    See SSW IDl functions
    `brm2_distrn <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_distrn.pro>`_ and
    `brm2_f_distrn <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_f_distrn.pro>`_.

    Examples
    --------

        >>> import numpy as np
        >>> from sunkit_spex.legacy.emission import BrokenPowerLawElectronDistribution
        >>> electron_dist = BrokenPowerLawElectronDistribution(p=5,q=7, eelow=10, eebrk=150,
        ...                                                    eehigh=500)
        >>> electron_dist
        BrokenPowerLawElectronDistribution(p=5, q=7, eelow=10, eebrk=150, eehigh=500, norm=True)
        >>> electron_dist.flux(np.array([15.0, 50.0, 100.0, 200.0, 500.0, 1000.0]))
        array([5.26752445e-02, 1.28000844e-04, 4.00002638e-06, 7.03129636e-08,
               1.15200760e-10, 0.00000000e+00])
        >>> electron_dist.density(np.array([15.0, 50.0, 100.0, 200.0, 500.0, 1000.0]))
        array([1.97525573e-01, 1.59341654e-03, 9.34066538e-05, 2.33416539e-06,
               2.68419888e-22, 0.00000000e+00])

    """

    def __init__(self, *, p, q, eelow, eebrk, eehigh, norm=True):
        """

        """
        self.p = p
        self.q = q
        self.eelow = eelow
        self.eebrk = eebrk
        self.eehigh = eehigh
        self.norm = norm
        if self.norm:
            n0 = (q - 1.0) / (p - 1.0) * eebrk ** (p - 1) * eelow ** (1 - p)
            n1 = n0 - (q - 1.0) / (p - 1.0)
            n2 = (1.0 - eebrk ** (q - 1) * eehigh ** (1 - q))
            self._norm_factor = 1.0 / (n1 + n2)
            self._n0 = n0
            self._n2 = n2
        else:
            self._norm_factor = 1.0
            self._n0 = 1.0
            self._n2 = 1.0

    def __eq__(self, other):
        return all([getattr(self, name) == getattr(other, name)
                    for name in ['p', 'q', 'eelow', 'eebrk', 'eehigh']]) and isinstance(other,
                                                                                        self.__class__)

    def flux(self, electron_energy):
        """
        Calculate the electron spectrum at the given energies.

        Parameters
        ----------
        electron_energy : `numpy.array`
            Electron energies

        Returns
        -------
        `numpy.array`
            The electron spectrum as a function of electron energy
        """
        res = np.zeros_like(electron_energy)

        index = np.where(electron_energy < self.eelow)
        if index[0].size > 0:
            res[index] = 0.

        index = np.where((electron_energy < self.eebrk) & (electron_energy >= self.eelow))
        if index[0].size > 0:
            res[index] = (self._norm_factor * self._n0 * (self.p - 1.)
                          * electron_energy[index] ** (-self.p) * self.eelow ** (self.p - 1.))

        index = np.where((electron_energy <= self.eehigh) & (electron_energy >= self.eebrk))
        if index[0].size > 0:
            res[index] = (self._norm_factor * (self.q - 1.)
                          * electron_energy[index] ** (-self.q) * self.eebrk ** (self.q - 1.))

        return res

    def density(self, electron_energy):
        """
        Return the electron flux at the given electron energies.

        Parameters
        ----------
        electron_energy : `numpy.array`
            Electron energies

        Returns
        -------
        `numpy.array`
            The electron flux as a function of electron energy
        """
        res = np.zeros_like(electron_energy)

        index = np.where(electron_energy < self.eelow)
        if index[0].size > 0:
            res[index] = 1.0

        index = np.where((electron_energy < self.eebrk) & (electron_energy >= self.eelow))
        if index[0].size > 0:
            res[index] = self._norm_factor * (self._n0 * self.eelow ** (self.p - 1) *
                                              electron_energy[index] ** (1.0 - self.p) -
                                              (self.q - 1.0) / (self.p - 1.0) + self._n2)

        index = np.where((electron_energy <= self.eehigh) & (electron_energy >= self.eebrk))
        if index[0].size > 0:
            res[index] = self._norm_factor * (self.eebrk ** (self.q - 1)
                                              * electron_energy[index] ** (1.0 - self.q)
                                              - (1.0 - self._n2))
        return res

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p}, q={self.q}, eelow={self.eelow}, ' \
            f'eebrk={self.eebrk}, eehigh={self.eehigh}, norm={self.norm})'


def collisional_loss(electron_energy):
    """
    Compute the energy dependant terms of the collisional energy loss rate for energetic electrons.

    Parameters
    ----------
    electron_energy : `numpy.array`
        Array of electron energies at which to evaluate loss

    Returns
    -------
    `numpy.array`
        Energy loss rate

    Notes
    -----
    Initial version modified from SSW
    `Brm_ELoss <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm/brm_eloss.pro>`_
    """
    electron_rest_mass = const.get_constant('mc2')  # * u.keV #c.m_e * c.c**2

    gamma = (electron_energy / electron_rest_mass) + 1.0

    beta = np.sqrt(1.0 - (1.0 / gamma ** 2))

    # TODO figure out what number is?
    energy_loss_rate = np.log(6.9447e+9 * electron_energy) / beta

    return energy_loss_rate


def bremsstrahlung_cross_section(electron_energy, photon_energy, z=1.2):
    """
    Compute the relativistic electron-ion bremsstrahlung cross section
    differential in energy (cm^2/mc^2 or 511 keV).

    Parameters
    ----------
    electron_energy : `numpy.array`
        Electron energies
    photon_energy : `numpy.array`
        Photon energies corresponding to electron_energy
    z : `float`
        Mean atomic number of target plasma

    Returns
    -------
    `np.array`
        The bremsstrahlung cross sections as a function of energy.

    Notes
    -----
    The cross section is from Equation (4) of [Haug]_. This closely follows Formula 3BN of [Koch]_,
    but requires fewer computational steps. The multiplicative factor introduced by [Elwert]_ is
    included.

    The initial version was heavily based of on [Brm_BremCross]_ from SSW IDL

    References
    ----------
    .. [Brm_BremCross] https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm/brm_bremcross.pro
    .. [Haug] Haug, E., 1997, Astronomy and Astrophysics, 326, 417,
       `ADS <https://ui.adsabs.harvard.edu/abs/1997A%26A...326..417H/abstract>`__
    .. [Koch] Koch, H. W., & Motz, J. W., 1959, Reviews of Modern Physics, 31, 920,
       `ADS <https://ui.adsabs.harvard.edu/abs/1959RvMP...31..920K/abstract>`__
    .. [Elwert] Elwert, G. 1939, Annalen der Physik, 426, 178,
       `ADS <https://ui.adsabs.harvard.edu/abs/1939AnP...426..178E/abstract>`__
    """

    mc2 = const.get_constant('mc2')
    alpha = const.get_constant('alpha')
    twoar02 = const.get_constant('twoar02')

    # Numerical coefficients
    c11 = 4.0 / 3.0
    c12 = 7.0 / 15.0
    c13 = 11.0 / 70.0
    c21 = 7.0 / 20.0
    c22 = 9.0 / 28.0
    c23 = 263.0 / 210.0

    # Calculate normalised photon and total electron energies.
    if electron_energy.ndim == 2:
        k = np.expand_dims(photon_energy / mc2, axis=1)
    else:
        k = photon_energy / mc2
    e1 = (electron_energy / mc2) + 1.0

    # Calculate energies of scattered electrons and normalized momenta.
    e2 = e1 - k
    p1 = np.sqrt(e1 ** 2 - 1.0)
    p2 = np.sqrt(e2 ** 2 - 1.0)

    # Define frequently used quantities.
    e1e2 = e1 * e2
    p1p2 = p1 * p2
    p2sum = p1 ** 2 + p2 ** 2
    k2 = k ** 2
    e1e23 = e1e2 ** 3
    pe = p2sum / e1e23

    # Define terms in cross section.
    ch1 = (c11 * e1e2 + k2) - (c12 * k2 / e1e2) - (c13 * k2 * pe / e1e2)
    ch2 = 1.0 + (1.0 / e1e2) + (c21 * pe) + (c22 * k2 + c23 * p1p2 ** 2) / e1e23

    # Collect terms.
    crtmp = ch1 * (2.0 * np.log((e1e2 + p1p2 - 1.0) / k) - (p1p2 / e1e2) * ch2)
    crtmp = z ** 2 * crtmp / (k * p1 ** 2)

    # Compute the Elwert factor.
    a1 = alpha * z * e1 / p1
    a2 = alpha * z * e2 / p2

    fe = (a2 / a1) * (1.0 - np.exp(-2.0 * np.pi * a1)) / (1.0 - np.exp(-2.0 * np.pi * a2))

    # Compute the differential cross section (units cm^2).
    cross_section = twoar02 * fe * crtmp

    return cross_section


def _get_integrand(x_log, *, model, electron_dist, photon_energy, z, efd=True):
    """
    Return the value of the integrand for the thick- or thin-target bremsstrahlung models.

    Parameters
    ----------
    x_log : `numpy.array`
        Log of the electron energies
    model : `str`
        Either `thick-target` or `thin-target`
    electron_dist : `BrokenPowerLawElectronDistribution`
        Electron distribution as function of energy
    photon_energy : `numpy.array`
        Photon energies
    z : `float`
        Mean atomic number of plasma
    efd: `bool` (optional)
        True (default) the electron flux distribution (electrons cm^-2 s^-1 keV^-1) is calculated
        with `~sunkit_spex.emission.BrokenPowerLawElectronDistribution.flux`. False, the electron
        density distribution (electrons cm^-3 keV^-1) is calculated with
        `~sunkit_spex.emission.BrokenPowerLawElectronDistribution.density`.

    Returns
    -------
    `numpy.array`
        The values of the integrand at the given electron_energies

    References
    ----------
    See SSW
    `brm2_fthin.pro <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_fthin.pro>`_ and
    `brm2_fouter.pro <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_fouter.pro>`_.

    """
    mc2 = const.get_constant('mc2')
    clight = const.get_constant('clight')

    # L=log10 (E), E=l0L and dE=10L ln(10) dL hence the electron_energy * np.log(10) below
    electron_energy = 10**x_log
    brem_cross = bremsstrahlung_cross_section(electron_energy, photon_energy, z)
    collision_loss = collisional_loss(electron_energy)
    pc = np.sqrt(electron_energy * (electron_energy + 2.0 * mc2))

    density = electron_dist.density(electron_energy)
    if model == 'thick-target':
        return (electron_energy * np.log(10) * density * brem_cross * pc
                / collision_loss / ((electron_energy / mc2) + 1.0))
    elif model == 'thin-target':
        if efd:
            return (electron_energy * np.log(10) * electron_dist.flux(electron_energy)
                    * brem_cross*(mc2/clight))
        else:
            return (electron_energy * np.log(10) * electron_dist.flux(electron_energy)
                    * brem_cross * pc/((electron_energy / mc2) + 1.0))


def _integrate_part(*, model, photon_energies, maxfcn, rerr, eelow, eebrk, eehigh,
                    p, q, z, a_lg, b_lg, ll, efd, integrator=None):
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
    nlim = 12

    intsum = np.zeros_like(photon_energies, dtype=np.float64)
    ier = np.zeros_like(photon_energies)

    if integrator is None:
        integrator = gauss_legendre
    elif not callable(integrator):
        raise TypeError('integrator must be a callable')

    # Copy indices over which to carry out the integration
    i = ll[:]

    electron_dist = BrokenPowerLawElectronDistribution(p=p, q=q, eelow=eelow, eebrk=eebrk,
                                                       eehigh=eehigh)

    for ires in range(2, nlim + 1):
        npoint = 2 ** ires
        if npoint > maxfcn:
            ier[i] = 1
            return intsum, ier

        lastsum = np.copy(intsum)

        intsum[i] = integrator(_get_integrand, a_lg[i], b_lg[i], n=npoint, func_kwargs={
            'model': model, 'electron_dist': electron_dist, 'photon_energy': photon_energies[i],
            'z': z, 'efd': efd})

        # Convergence criterion
        l1 = np.abs(intsum - lastsum)
        l2 = rerr * np.abs(intsum)
        i = np.where(l1 > l2)[0]

        # If all point have reached criterion return value and flags
        if i.size == 0:
            return intsum, ier


def _split_and_integrate(*, model, photon_energies, maxfcn, rerr, eelow, eebrk, eehigh, p, q, z,
                         efd, integrator=None):
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
    intsum1 = np.zeros_like(photon_energies, dtype=np.float64)
    ier1 = np.zeros_like(photon_energies, dtype=np.float64)
    intsum2 = np.zeros_like(photon_energies, dtype=np.float64)
    ier2 = np.zeros_like(photon_energies, dtype=np.float64)
    intsum3 = np.zeros_like(photon_energies, dtype=np.float64)
    ier3 = np.zeros_like(photon_energies, dtype=np.float64)

    P1 = np.where(photon_energies < eelow)[0]
    P2 = np.where(photon_energies < eebrk)[0]
    P3 = np.where(photon_energies <= eehigh)[0]

    # Part 1, below en_val[0] (usually eelow)
    if model == 'thick-target':
        if P1.size > 0:
            logging.debug('Part1')
            a_lg = np.log10(photon_energies[P1])
            b_lg = np.log10(np.full_like(a_lg, eelow))
            i = np.copy(P1)
            intsum1, ier1 = _integrate_part(model=model, maxfcn=maxfcn, rerr=rerr,
                                            photon_energies=photon_energies,
                                            eelow=eelow, eebrk=eebrk, eehigh=eehigh, p=p, q=q, z=z,
                                            a_lg=a_lg, b_lg=b_lg, ll=i, efd=efd, integrator=integrator)

            # ier = 1 indicates no convergence.
            if sum(ier1):
                raise ValueError('Part 1 integral did not converge for some photon energies.')

    # Part 2, between enval[0] and en_val[1](usually eelow and eebrk)

    aa = np.copy(photon_energies)
    if (P2.size > 0) and (eebrk > eelow):
        # TODO check if necessary as integration should only be carried out over point P2 which
        # by definition are not in P1
        if P1.size > 0:
            aa[P1] = eelow

        logging.debug('Part2')
        a_lg = np.log10(aa[P2])
        b_lg = np.log10(np.full_like(a_lg, eebrk))
        i = np.copy(P2)
        intsum2, ier2 = _integrate_part(model=model, maxfcn=maxfcn, rerr=rerr,
                                        photon_energies=photon_energies,
                                        eelow=eelow, eebrk=eebrk, eehigh=eehigh, p=p, q=q, z=z,
                                        a_lg=a_lg, b_lg=b_lg, ll=i, efd=efd, integrator=integrator)

        if sum(ier2) > 0:
            raise ValueError('Part 2 integral did not converge for some photon energies.')

    # Part 3: between eebrk and eehigh(usually eebrk and eehigh)
    aa = np.copy(photon_energies)
    if (P3.sum() > 0) and (eehigh > eebrk):
        if P2.size > 0:
            aa[P2] = eebrk

        logging.debug('Part3')
        a_lg = np.log10(aa[P3])
        b_lg = np.log10(np.full_like(a_lg, eehigh))
        i = np.copy(P3)
        intsum3, ier3 = _integrate_part(model=model, maxfcn=maxfcn, rerr=rerr,
                                        photon_energies=photon_energies,
                                        eelow=eelow, eebrk=eebrk, eehigh=eehigh, p=p, q=q, z=z,
                                        a_lg=a_lg, b_lg=b_lg, ll=i, efd=efd, integrator=integrator)
        if sum(ier3) > 0:
            raise ValueError('Part 3 integral did not converge for some photon energies.')

    # TODO check units here
    # Combine 3 parts and convert units and return
    if model == 'thick-target':
        DmlinO = (intsum1 + intsum2 + intsum3) * (mc2 / clight)
        ier = ier1 + ier2 + ier3
        return DmlinO, ier
    elif model == 'thin-target':
        Dmlin = (intsum2 + intsum3)
        ier = ier2 + ier3
        return Dmlin, ier


def bremsstrahlung_thin_target(photon_energies, p, eebrk, q, eelow, eehigh, efd=True, integrator=None):
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
    integrator : callable
        A Python function or method to integrate must support vector limits and match signture
        `fun(x, a, b, n, *args, **kwargs)`

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
        flux[l], iergq[l] = _split_and_integrate(model='thin-target',
                                                 photon_energies=photon_energies[l], maxfcn=maxfcn,
                                                 rerr=rerr, eelow=eelow, eebrk=eebrk, eehigh=eehigh,
                                                 p=p, q=q, z=z, efd=efd, integrator=integrator)

        flux *= fcoeff

        return flux
    else:
        raise Warning('The photon energies are higher than the highest electron energy or not '
                      'greater than zero')


def bremsstrahlung_thick_target(photon_energies, p, eebrk, q, eelow, eehigh, integrator=None):
    """
    Computes the thick-target bremsstrahlung x-ray/gamma-ray spectrum from an isotropic electron
    distribution function provided in `broken_powerlaw_f`. The units of the computed flux is photons
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
    integrator : callable
        A Python function or method to integrate must support vector limits and match signture
        `fun(x, a, b, n, *args, **kwargs)`

    Returns
    -------
    `numpy.array`
        flux The computed bremsstrahlung photon flux at the given photon energies.
        Array of photon fluxes (in photons s^-1 keV^-1 cm^-2), when multiplied by a0 * 1.0d+35,
        corresponding to the photon energies in the input array eph.
        The detector is assumed to be 1 AU from the source.
        a0 is the total integrated electron flux, in units of 10^35 electrons s^-1.

    Notes
    -----
    If you want to plot the derivative of the flux, or the spectral index of the photon spectrum as
    a function of photon energy, you should set RERR to 1.d-6, because it is more sensitive to RERR
    than the flux.

    Adapted from SSW `Brm2_ThickTarget
    <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_thicktarget.pro>`_
    """
    # Constants
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
    flux = np.zeros_like(photon_energies, dtype=np.float64)
    iergq = np.zeros_like(photon_energies, dtype=np.float64)

    if eelow >= eehigh:
        return flux

    i, = np.where((photon_energies < eehigh) & (photon_energies > 0))

    if i.size > 0:
        flux[i], iergq[i] = _split_and_integrate(model='thick-target',
                                                 photon_energies=photon_energies[i],
                                                 maxfcn=maxfcn, rerr=rerr, eelow=eelow,
                                                 eebrk=eebrk, eehigh=eehigh, p=p, q=q, z=z,
                                                 efd=False, integrator=integrator)

        flux = (fcoeff / decoeff) * flux

        return flux
    else:
        raise Warning('The photon energies are higher than the highest electron energy or not '
                      'greater than zero')
