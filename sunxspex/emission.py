"""
Functions for computing the photon flux due to bremsstrahlung radiation from energetic electrons
impacting a dense plasma.

References
----------

.. [1] Thick-Target: https://hesperia.gsfc.nasa.gov/hessi/flarecode/bremthickdoc.pdf
.. [2] Thin-Target: https://hesperia.gsfc.nasa.gov/hessi/flarecode/bremdoc.pdf

"""
import numpy as np
from scipy.special import lpmv

from sunxspex.constants import Constants

# Central constant management
const = Constants()

np.seterr(all='raise')

def broken_powerlaw(x, p, q, eelow, eebrk, eehigh):
    """
    This routine calculates the value of the electron distribution function at electron energy x.
    The distribution function is normalized so that the integral from eelow to eehigh is 1.

    Parameters
    ----------
    x : np.array
        electron energy in keV
    p : float
        Slope below the break (x < eebrk)
    q : float
        Slope above the break (x > eebrk)
    eelow : float
        Low cutoff
    eebrk : float
        Break
    eehigh : float
        High cutoff
    Returns
    -------
    np.array
        Power law of x

    Notes
    -----
    A normalized, double power law electron distribution function is computed.
    p is the lower power-law index, between eelow and eebrk, and q is the
    upper power-law index, between eebrk and eehigh.

    Adapted from SSW Brm2_Distrn.pro:
        https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_distrn.pro

    """
    # Obtain normalization coefficient, norm.
    n0 = (q - 1.0) / (p - 1.0) * eebrk ** (p - 1) * eelow ** (1 - p)
    n1 = n0 - (q - 1.0) / (p - 1.0)
    n2 = (1.0 - eebrk ** (q - 1) * eehigh ** (1 - q))

    norm = 1.0 / (n1 + n2)

    res = np.zeros_like(x)

    ind0 = np.where(x < eelow)
    if ind0[0].size > 0:
        res[ind0] = 0.

    ind1 = np.where((x < eebrk) & (x >= eelow))
    if ind1[0].size > 0:
        res[ind1] = norm * n0 * (p - 1.) * x[ind1] ** (-p) * eelow ** (p - 1.)

    ind2 = np.where((x <= eehigh) & (x >= eebrk))
    if ind2[0].size > 0:
        res[ind2] = norm * (q - 1.) * x[ind2] ** (-q) * eebrk ** (q - 1.)

    return res

def broken_powerlaw_f(x, p, q, eelow, eebrk, eehigh):
    """
    Return power law of x with a break and low and high cutoffs

    Parameters
    ----------
    x : np.array

    p : float
        Slope below the break (x < eebrk)
    q : float
        Slope above the break (x > eebrk)
    eelow : float
        Low cutoff
    eebrk : float
        Break
    eehigh : float
        High cutoff
    Returns
    -------
    np.array
        Power law of x

    Notes
    -----
    The *integral* of a normalized, double power law electron distribution function is computed.
    p is the lower power-law index, between eelow and eebrk, and q is the
    upper power-law index, between eebrk and eehigh.

    Adapted from SSW Brm2_F_Distrn.pro:
        https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_f_distrn.pro
    """
    # Obtain normalization coefficient, norm.
    n0 = (q - 1.0) / (p - 1.0) * eebrk ** (p - 1) * eelow ** (1 - p)
    n1 = n0 - (q - 1.0) / (p - 1.0)
    n2 = (1.0 - eebrk ** (q - 1) * eehigh ** (1 - q))

    norm = 1.0 / (n1 + n2)

    res = np.zeros_like(x)

    index = np.where(x < eelow)
    if index[0].size > 0:
        res[index] = 1.0

    index = np.where((x < eebrk) & (x >= eelow))
    if index[0].size > 0:
        res[index] = norm * (n0 * eelow ** (p - 1) * x[index] ** (1.0 - p) - (q - 1.0) / (p - 1.0) + n2)

    index = np.where((x <= eehigh) & (x >= eebrk))
    if index[0].size > 0:
        res[index] = norm * (eebrk ** (q - 1) * x[index] ** (1.0 - q) - (1.0 - n2))

    return res

def powerlaw(x, low_energy_cutoff=10, high_energy_cutoff=100, index=3):
    """

    Parameters
    ----------
    x
    low_energy_cutoff
    high_energy_cutoff
    index

    Returns
    -------

    """
    # normalisation = (high_energy_cutoff**(index-1)/(index-1) +
    #     (low_energy_cutoff**(index-1)/(index-1)))

    normalisation = 1 / (((high_energy_cutoff ** (2 - 2 * index)) / ((1 - index) ** 2)) - (
            (low_energy_cutoff ** (2 - 2 * index)) / ((1 - index) ** 2)))

    return normalisation * x ** (1 - index)


def collisional_loss(electron_energy):
    """
    Compute the energy dependant terms of the collisional energy loss rate for energetic electrons

    Parameters
    ----------
    electron_energy : np.array
        Array of electron energies at which to evaluate loss

    Returns
    -------
    np.array
        Energy loss rate

    Notes
    -----
    Initial version modified from SSW Brm_ELoss.pro:
        https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm/brm_eloss.pro
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
    differential in energy (unit: cm^2 per mc2 or 511 keV).

    Parameters
    ----------
    electron_energy : np.array
        Electron energies
    photon_energy : np.arry
        Photon energies corresponding to electron_energy
    z : float
        Mean atomic number of target plasma

    Returns
    -------
    np.array
        Bremsstrahlung cross sections.

    Notes
    -----
    The cross section is from Equation (4) of E. Haug (Astron. Astrophys. 326,
    417, 1997).  This closely follows Formula 3BN of H. W. Koch & J. W. Motz
    (Rev. Mod. Phys. 31, 920, 1959), but requires fewer computational steps.
    The multiplicative factor introduced by G. Elwert (Ann. Physik 34, 178,
    1939) is included.

    Initial version modified from SSW Brm_BremCross.pro
        https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm/brm_bremcross.pro
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

    # Calculate normalized photon and total electron energies.
    if electron_energy.ndim == 2:
        k = np.expand_dims(photon_energy / mc2, axis=1)
    else:
        k = photon_energy / mc2
    e1 = (electron_energy / mc2) + 1.0

    # Calculate energies of scatter electrons and normalized momenta.
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


def brm2_fthin(electron_energy, photon_energy, eelow, eebrk, eehigh, p, q, z=1.2, efd=True):
    """
    Returns the integrand for the thin-target bremsstrahlung integration.

    Parameters
    ----------
    electron_energy : np.array
        Electron energies in keV. Two-dimensional array of abscissas calculated in brm_gauss_legendre()
    photon_energy : np.array
        Photon energies in keV
    eelow : float
        Low energy electron cut off
    eebrk : float
        Break energy
    eehigh :
        High energy cutoff
    p : float
        Slope below the break energy
    q : float
        Slope above the break energy
    z : float
        Mean atomic number of plasma
    efd: bool
        If set to True (default), the electron flux distribution (electrons cm^-2 s^-1 keV^-1)
            is calculated from broken_powerlaw().
        If set to False, the electron density distribution (electrons cm^-3 keV^-1)
            is calculated from broken_powerlaw().

    Returns
    -------
    np.array
        Bremsstrahlung photon flux at given photon energies

    Notes
    -----
    Adapted from SSW Brm2_FThin.pro:
        https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_fthin.pro
    """
    mc2 = const.get_constant('mc2')
    clight = const.get_constant('clight')
    gamma = (electron_energy / mc2) + 1.0
    pc = np.sqrt(electron_energy * (electron_energy + 2.0 * mc2))
    #pdb.set_trace()
    brem_cross = bremsstrahlung_cross_section(electron_energy, photon_energy, z)
    electron_dist = broken_powerlaw(electron_energy, p, q, eelow, eebrk, eehigh)
    if efd:
        # if electron flux distribution is assumed (default)
        photon_flux = electron_dist * brem_cross * (mc2 / clight)
    else:
        # if electron density distribution is assumed
        photon_flux = electron_dist * brem_cross * pc / gamma  # that is n_e * sigma * mc2 * (v / c)

    return photon_flux


def brem_outer(electron_energy, photon_energy, eelow, eebrk, eehigh, p, q, z=1.2):
    """
    Calculate the outer integration over electron energies.

    Parameters
    ----------
    electron_energy : np.array
        Electron energies
    photon_energy : np.array
        Photon energies
    eelow : float
        Low energy electron cut off
    eebrk : float
        Break energy
    eehigh :
        High energy cutoff
    p : float
        Slope below the break energy
    q : flaot
        Slope above the break energy
    z : float
        Mean atomic number of plasma

    Returns
    -------
    np.array
        Bremsstrahlung photon flux at given photon energies

    Notes
    -----
    Initial version modified from SSW Brm2_Fouter.pro
        https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_fouter.pro
    """
    mc2 = const.get_constant('mc2')

    gamma = (electron_energy / mc2) + 1.0

    brem_cross = bremsstrahlung_cross_section(electron_energy, photon_energy, z)

    collision_loss = collisional_loss(electron_energy)

    pc = np.sqrt(electron_energy * (electron_energy + 2.0 * mc2))

    electron_flux = broken_powerlaw_f(electron_energy, p, q, eelow, eebrk, eehigh)

    photon_flux = electron_flux * brem_cross * pc / collision_loss / gamma

    return photon_flux


def brm_gauss_legendre(x1, x2, npoints):
    """
    Calculate the positions and weights for a Gauss-Legendre integration scheme.

    Parameters
    ----------
    x1 : np.array

    x2 : np.array

    npoints : int
        Degree or number of point evalute fuction at
    Returns
    -------
    tuple : (x, w)
        The positions and weights for the integration.

    Notes
    -----

    History
    ------
    Adapted from SSW Brm_GauLeg54.pro:
            https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm/brm_gauleg54.pro
    """
    eps = 3e-14
    m = (npoints + 1) // 2

    x = np.zeros((x1.shape[0], npoints))
    w = np.zeros((x1.shape[0], npoints))

    # Normalise from -1 to +1 as Legendre polynomial only valid in this range
    xm = 0.5 * (x2 + x1)
    xl = 0.5 * (x2 - x1)

    for i in range(1, m + 1):

        z = np.cos(np.pi * (i - 0.25) / (npoints + 0.5))
        # Init to np.inf so loop runs at least once
        z1 = np.inf

        # Some kind of integration/update loop
        # TODO put back as while condition loop
        while True:
            # Evaluate Legendre polynomial of degree npoints at z points P_m^l(z) m=0, l=npoints
            p1 = lpmv(0, npoints, z)
            p2 = lpmv(0, npoints - 1, z)

            pp = npoints * (z * p1 - p2) / (z ** 2 - 1.0)

            z1 = np.copy(z)
            z = z1 - p1 / pp
            if np.abs(z - z1) <= eps:
                break

        # Update ith components
        x[:, i - 1] = xm - xl * z
        x[:, npoints - i] = xm + xl * z
        w[:, i - 1] = 2.0 * xl / ((1.0 - z ** 2) * pp ** 2)
        w[:, npoints - i] = w[:, i - 1]

    return x, w


def brm2_dmlin(a, b, maxfcn, rerr, eph, eelow, eebrk, eehigh, p, q, z, efd=True, verbose=False):
    """
    This is used for thin-target calculation from a double power-law electron density distribution
    To integrate a function via the method of Gaussian quadrature.  Repeatedly
    doubles the number of points evaluated until convergence, specified by the
    input rerr, is obtained, or the maximum number of points, specified by the
    input maxfcn, is reached.  If integral convergence is not achieved, this
    function returns the error code ier = 1 when either the maximum number of
    function evaluations is performed or the number of Gaussian points to be
    evaluated exceeds maxfcn.  Maxfcn should be less than or equal to 2^nlim,
    or 4096 with nlim = 12.

    This function splits the numerical integration into up to three parts and returns the sum of
    the parts. This avoids numerical problems with discontinuities in the electron distribution
    function at eelow and eebrk.

    Parameters
    ----------
    a : np.array
        Array containing lower integration limits
    b : np.array
        Array containing upper integration limits
    maxfcn : int
        Maximum number of points used in Gaussian quadrature integration
    rerr : float
        Desired relative error for integral evaluation. For example, rerr = 0.01 indicates that
        the estimate of the integral is to be correct to one digit, whereas rerr = 0.001
        alls for two digits to be correct.
    eph : np.array
        Array of photon energies at which to calculate fluxes
    eelow : float
        Low energy cut off of electron energy distribution, see broken_powerlaw()
    eebrk : float
        Break energy of electron energy distribution, see broken_powerlaw()
    eehigh : float
        High energy cut off of electron energy distribution, see broken_powerlaw()
    p : float
        Slope below the break energy
    q : flaot
        Slope above the break energy
    z : float
        Mean atomic number of plasma
    efd: True (default) - electron flux density distribution, False - electron density distribution.
         This input is not used in the main routine, but is passed to Brm_Fthin()
    verbose: bool. Default False
         If True, Print extra information in command line

    Returns
    -------
    tuple : (Dmlin, irer)

        Array of integral evaluation (Dmlin) and array of error flags
        (ier, 0 for converged and 1 for nonconverge)

    Notes
    -------
    Adapted from SSW Brm2_Dmlin.pro:
        https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_dmlin.pro

    """
    mc2 = const.get_constant('mc2')
    clight = const.get_constant('clight')

    if eebrk < eelow:
        eebrk = eelow

    if eebrk > eehigh:
        eebrk = eehigh

    en_vals = [eelow, eebrk, eehigh]
    en_vals = sorted(en_vals)

    # Part 1, below en_val[0] (usually eelow)
    # Create arrays for integral sum and error flags.
    intsum1 = np.zeros_like(a, dtype=np.float64)
    ier1 = np.zeros_like(a, dtype=np.float64)

    aa = np.copy(a)
    P1 = np.where(a < en_vals[0])[0]  # photon energy lower than low energy cutoff
    P2 = np.where(a < en_vals[1])[0]  # photon energy lower than break energy

    if P2.size > 0 and (en_vals[1] > en_vals[0]):
        if verbose:
            print('======Integrating Part1 between eelow and eebrk======')
        if P1.size > 0:
            aa[P1] = eelow

        a_lg = np.log10(aa[P2])
        b_lg = np.log10(np.full_like(a_lg, en_vals[1]))

        l = np.copy(P2)
        nlim = 12  # Maximum possible order of the Gaussian quadrature integration is 2^nlim

        for ires in range(2, nlim+1):
            npoint = 2 ** ires
            if (npoint > maxfcn):
                ier1[l] = 1
            eph1 = eph[l]

            # generate positions and weights
            xi, wi, = brm_gauss_legendre(a_lg[l], b_lg[l], npoint)

            lastsum1 = np.copy(intsum1)

            # Perform integration sum w_i * f(x_i)  i=1 to npoints
            intsum1[l] = np.sum((10.0 ** xi * np.log(10.0) * wi
                                 * brm2_fthin(10.0 ** xi, eph1, eelow, eebrk, eehigh, p, q, z, efd)), axis=1)

            # Convergence criterion
            l1 = np.abs(intsum1 - lastsum1)
            l2 = rerr * np.abs(intsum1)
            l = np.where(l1 > l2)[0]

        # ier = 1 indicates no convergence.
        if sum(ier1) > 0:
            raise ValueError('Part 1 integral did not converge for some photon energies.')

    # Part 2, between enval[1] and en_val[2](usually eebrk and eehigh)
    intsum2 = np.zeros_like(a, dtype=np.float64)
    ier2 = np.zeros_like(a, dtype=np.float64)
    aa = np.copy(a)
    P3 = np.where(a <= en_vals[2])[0]

    if (P3.size > 0) and (en_vals[2] > en_vals[1]):
        if P2.size > 0:
            aa[P2] = en_vals[1]
        if verbose:
            print('======Integrating Part2 between eebrk and eehigh======')

        a_lg = np.log10(aa[P3])
        b_lg = np.log10(np.full_like(a_lg, en_vals[2]))

        l = np.copy(P3)
        nlim = 12  # Maximum possible order of the Gaussian quadrature integration is 2^nlim

        for ires in range(2, nlim+1):
            npoint = 2 ** ires
            if (npoint > maxfcn):
                ier2[l] = 1
            eph1 = eph[l]

            # generate positions and weights
            xi, wi, = brm_gauss_legendre(a_lg[l], b_lg[l], npoint)

            lastsum2 = np.copy(intsum2)

            # Perform integration sum w_i * f(x_i)  i=1 to npoints
            intsum2[l] = np.sum((10.0 ** xi * np.log(10.0) * wi
                                 * brm2_fthin(10.0 ** xi, eph1, eelow, eebrk, eehigh, p, q, z, efd)), axis=1)

            # Convergence criterion
            l1 = np.abs(intsum2 - lastsum2)
            l2 = rerr * np.abs(intsum2)
            l = np.where(l1 > l2)[0]

        # ier = 1 indicates no convergence.
        if sum(ier2) > 0:
            raise ValueError('Part 2 integral did not converge for some photon energies.')

    # Combine 2 parts and return
    Dmlin = (intsum1 + intsum2)
    ier = ier1 + ier2

    return Dmlin, ier


def brm2_dmlino_int(maxfcn, rerr, eph, eelow, eebrk, eehigh, p, q, z, a_lg, b_lg, ll):
    """
    Perform numerical Gaussian-Legendre Quadrature integration for thick target model.

    Double the number of points until convergence criterion is reached then return

    Parameters
    ----------
    maxfcn : int
        Max number of point for Gaussian-Legendre Quadrature integration
    rerr : float
        Desired relative error level (0.01 correct to one digit, 0.001 correct to two digits)
    eph :
        Array of photon energies for which the flux is caculated
    eelow : float
        Low energy electron cut off
    eebrk : float
        Break energy
    eehigh :
    p : float
        Slope below the break energy
    q : flaot
        Slope above the break energy
    z : float
        Mean atomic number of plasma
    a_lg : np.array:
        Array of logarithm of lower integration limit
    b_lg :
        Array of logarithm of upper integration limit
    ll : np.array
        Indices for which to carry out integration

    Returns
    -------
    tuple : (intsum, ier)
        Array of results and array of error flags

    Notes
    -----
    Initial version modified from SSW Brm2_DmlinO_int.pro
        https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_dmlino_int.pro
    """
    nlim = 12  # 4096 points

    # Output arrays
    intsum = np.zeros_like(eph, dtype=np.float64)

    ier = np.zeros_like(eph)

    # TODO Probably not need test and remove
    l = ll[:]

    for ires in range(2, nlim + 1):
        npoint = 2 ** ires
        if npoint > maxfcn:
            ier[l] = 1
            return intsum, ier

        eph1 = eph[l]

        # generate positions and weights
        xi, wi, = brm_gauss_legendre(a_lg[l], b_lg[l], npoint)
        lastsum = np.copy(intsum)

        # Perform integration sum w_i * f(x_i)  i=1 to npoints
        intsum[l] = np.sum((10.0 ** xi * np.log(10.0) * wi
                            * brem_outer(10.0 ** xi, eph1, eelow, eebrk, eehigh, p, q, z)), axis=1)
        # Convergence criterion
        l1 = np.abs(intsum - lastsum)
        l2 = rerr * np.abs(intsum)
        l = np.where(l1 > l2)[0]

        # If all point have reached criterion return value and flags
        if l.size == 0:
            return intsum, ier


def brm2_dmlino(a, b, maxfcn, rerr, eph, eelow, eebrk, eehigh, p, q, z):
    """
    Integrates and broken power law electron distribution ot obtain photon fluxes at the given
    photon energies and electron limits.

    This function splits the numerical integration into up to three parts and returns the sum of
    the parts. This avoids numerical problems with discontinuities in the electron distribution
    function at eelow and eebrk.

    Parameters
    ----------
    a : np.array
        Array containing lower integration limits
    b : np.array
        Array containing upper integration limits
    maxfcn : int
        Maximum number of points used in Guassian quadrature integration
    rerr : float
        Desired relative error for integral evaluation
    eph : np.array
        Array of photon energies at which to calculate fluxes
    eelow : float
        Low energy electron cut off
    eebrk : float
        Break energy
    eehigh :
    p : float
        Slope below the break energy
    q : flaot
        Slope above the break energy
    z : float
        Mean atomic number of plasma

    Returns
    -------
    tuple : (DmlinO, irer)

        Array of integral evaluation and array of error flags
    Notes
    -----
    Initial version modified from SSW Brm2_DmlinO.pro:
        https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_dmlino.pro

    """
    mc2 = const.get_constant('mc2')
    clight = const.get_constant('clight')

    # TODO raise or also return input values
    if eebrk < eelow:
        eebrk = eelow

    if eebrk > eehigh:
        eebrk = eehigh

    en_vals = [eelow, eebrk, eehigh]
    en_vals = sorted(en_vals)

    # Part 1, below en_val[0] (usually eelow)
    # Create arrays for integral sum and error flags.
    intsum1 = np.zeros_like(a, dtype=np.float64)
    ier1 = np.zeros_like(a, dtype=np.float64)

    # TODO put repeated code in private _function

    P1 = np.where(a < en_vals[0])[0]

    if P1.size > 0:
        print('Part1')
        a_lg = np.log10(a[P1])
        b_lg = np.log10(np.full_like(a_lg, en_vals[0]))

        l = np.copy(P1)

        intsum1, ier1 = brm2_dmlino_int(maxfcn, rerr, eph, eelow, eebrk, eehigh,
                                        p, q, z, a_lg, b_lg, l)

    # ier = 1 indicates no convergence.
    if sum(ier1) > 0:
        raise ValueError('Part 1 integral did not converge for some photon energies.')

    # Part 2, between enval[0 and en_val[1](usually eelow and eebrk)
    intsum2 = np.zeros_like(a, dtype=np.float64)
    ier2 = np.zeros_like(a, dtype=np.float64)
    aa = np.copy(a)
    P2 = np.where(a < en_vals[1])[0]

    if (P2.size > 0) and (en_vals[1] > en_vals[0]):
        if P1.size > 0:
            aa[P1] = en_vals[0]

        print('Part2')

        a_lg = np.log10(aa[P2])
        b_lg = np.log10(np.full_like(a_lg, en_vals[1]))

        l = np.copy(P2)

        intsum2, ier2 = brm2_dmlino_int(maxfcn, rerr, eph, eelow, eebrk, eehigh,
                                        p, q, z, a_lg, b_lg, l)

        if sum(ier2) > 0:
            raise ValueError('Part 2 integral did not converge for some photon energies.')

    # Part 3: between en_vals[1] and en_vals[2](usually eebrk and eehigh)
    intsum3 = np.zeros_like(a, dtype=np.float64)
    ier3 = np.zeros_like(a, dtype=np.float64)

    aa = np.copy(a)
    P3 = np.where(a <= en_vals[2])[0]

    if (P3.sum() > 0) and (en_vals[2] > en_vals[1]):
        if P2.size > 0:
            aa[P2] = en_vals[1]

        print('Part3')

        a_lg = np.log10(aa[P3])
        b_lg = np.log10(np.full_like(a_lg, en_vals[2]))

        l = np.copy(P3)

        intsum3, ier3 = brm2_dmlino_int(maxfcn, rerr, eph, eelow, eebrk, eehigh,
                                        p, q, z, a_lg, b_lg, l)

        if sum(ier3) > 0:
            raise ValueError('Part 3 integral did not converge for some photon energies.')

    # TODO check units here
    # Combine 3 parts and convert units and return
    DmlinO = (intsum1 + intsum2 + intsum3) * (mc2 / clight)
    ier = ier1 + ier2 + ier3

    return DmlinO, ier


def bremsstrahlung_thin_target(eph, p, eebrk, q, eelow, eehigh, efd=True):
    """
    Computes the thin-target bremsstrahlung x-ray/gamma-ray spectrum from an isotropic electron
    distribution function provided in `broken_powerlaw`. The units of the computed flux is photons
    per second per keV per square centimeter.

    The electron flux distribution function is a double power law in electron energy with a
    low-energy cutoff and a high-energy cutoff.

    Parameters
    ----------
    eph : np.array
        Array of photon energies to evaluate flux at
    p   : float
        Slope below the break energy
    eebrk : float
        Break energy
    q   : float
        Slope above the break energy
    eelow : float
        Low energy electron cut off
    eehigh : float
        High energy electron cut off
    efd: bool.
        True (default) - input electron distribution is electron flux density distribution
            (unit electrons cm^-2 s^-1 keV^-1),
        False - input electron distribution is electron density distribution.
            (unit electrons cm^-3 keV^-1),
         This input is not used in the main routine, but is passed to brm2_dmlin() (and Brm2_Fthin())

    Returns
    -------
    flux: np.array
        Multiplying the output of Brm2_ThinTarget by a0 gives an array of
        photon fluxes in photons s^-1 keV^-1 cm^-2, corresponding to the photon energies in the input array eph.
        The detector is assumed to be 1 AU rom the source.
        The coefficient a0 is calculated as a0 = nth * V * nnth, where
            nth: plasma density; cm^-3)
            V: volume of source; cm^3)
            nnth: Integrated nonthermal electron flux density (cm^-2 s^-1), if efd = True, or
                  Integrated electron number density (cm^-3), if efd = False

    Notes
    -----
    If you want to plot the derivative of the flux, or the spectral index of the photon spectrum as
    a function of photon energy, you should set RERR to 1.d-6, because it is more sensitive to RERR
    than the flux.

    Adapted from SSW Brm2_ThinTarget.pro:
        https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_thintarget.pro
    """
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
    fcoeff = (clight / (4 * np.pi * au ** 2)) / mc2 ** 2.

    # Create arrays for the photon flux and error flags.
    flux = np.zeros_like(eph, dtype=np.float64)
    iergq = np.zeros_like(eph, dtype=np.float64)

    if eelow >= eehigh:
        raise ValueError('eehigh must be larger than eelow!')
        return flux

    l, = np.where((eph < eehigh) & (eph > 0))
    if l.size > 0:
        #pdb.set_trace()
        flux[l], iergq[l] = brm2_dmlin(eph[l], np.full_like(l, eehigh), maxfcn, rerr,
                                       eph[l], eelow,
                                       eebrk, eehigh, p, q, z, efd)

        flux *= fcoeff

        return flux
    else:
        raise Warning('The photon energies are higher than the highest electron energy or not greater than zero')

def bremsstrahlung_thick_target(eph, p, eebrk, q, eelow, eehigh):
    """
    Computes the thick-target bremsstrahlung x-ray/gamma-ray spectrum from an isotropic electron
    distribution function provided in `broken_powerlaw_f`. The units of the computed flux is photons
    per second per keV per square centimeter.

    The electron flux distribution function is a double power law in electron energy with a
    low-energy cutoff and a high-energy cutoff.

    Parameters
    ----------
    eph : np.array
        Array of photon energies to evaluate flux at
    p   : float
        Slope below the break energy
    eebrk : float
        Break energy
    q   : float
        Slope above the break energy
    eelow : float
        Low energy electron cut off
    eehigh : float
        High energy electron cut off

    Returns
    -------
    np.array : flux
        The computed bremsstrahlung photon flux at the given photon energies.
        Array of photon fluxes (in photons s^-1 keV^-1 cm^-2), when multiplied by a0 * 1.0d+35,
        corresponding to the photon energies in the input array eph.
        The detector is assumed to be 1 AU from the source.
        a0 is the total integrated electron flux, in units of 10^35 electrons s^-1.

    Notes
    -----
    If you want to plot the derivative of the flux, or the spectral index of the photon spectrum as
    a function of photon energy, you should set RERR to 1.d-6, because it is more sensitive to RERR
    than the flux.

    Adapted from SSW Brm2_ThickTarget.pro:
        https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_thicktarget.pro
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
    flux = np.zeros_like(eph, dtype=np.float64)
    iergq = np.zeros_like(eph, dtype=np.float64)

    if eelow >= eehigh:
        return flux

    l = np.where((eph < eehigh) & (eph > 0))

    if l[0].size > 0:
        flux[l[0]], iergq[l[0]] = brm2_dmlino(eph[l[0]], np.full_like(l[0], eehigh), maxfcn, rerr,
                                              eph[l[0]], eelow,
                                              eebrk, eehigh, p, q, z)

        flux = (fcoeff / decoeff) * flux

        return flux
    else:
        raise Warning('The photon energies are higher than the highest electron energy or not greater than zero')
