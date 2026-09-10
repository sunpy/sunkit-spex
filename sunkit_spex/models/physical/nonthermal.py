"""
Functions for computing the photon flux due to bremsstrahlung radiation from energetic electrons
impacting a dense plasma. See [1]_ and [2]_.


References
----------

.. [1] Thick-Target: https://hesperia.gsfc.nasa.gov/hessi/flarecode/bremthickdoc.pdf
.. [2] Thin-Target: https://hesperia.gsfc.nasa.gov/hessi/flarecode/bremdoc.pdf

"""

import logging
from dataclasses import field, dataclass

import numpy as np

import astropy.units as u
from astropy.modeling import FittableModel, Parameter

from sunkit_spex.legacy import constants
from sunkit_spex.models.physical.integrate import fixed_quad_batch

const = constants.Constants()

# Cached once here rather than calling const.get_constant(...) repeatedly
MC2 = const.get_constant("mc2")  # electron rest mass energy, keV
CLIGHT = const.get_constant("clight")  # speed of light, cm/s
ALPHA = const.get_constant("alpha")  # fine structure constant
TWOAR02 = const.get_constant("twoar02")  # 2 * alpha * classical electron radius^2, cm^2
R0 = const.get_constant("r0")  # classical electron radius, cm

# ln(10), used in `_get_integrand` to convert a log10-space integral back to electron-energy
# space (dE = E * ln(10) * dx_log)
LN_10 = np.log(10)

logger = logging.getLogger(__name__)

__all__ = ["ThickTarget", "ThinTarget"]

# Mean atomic number of the target plasma, from SSW `Brm2_BremCross <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_bremcross.pro>`_.
MEAN_ATOMIC_NUMBER = 1.2

# Maximum number of Gauss-Legendre quadrature points tried before giving up on convergence.
MAX_QUADRATURE_POINTS = 2048

# log2(MAX_QUADRATURE_POINTS); `_integrate_part` doubles the point count from 2**2 up to 2**NLIM.
NLIM = 12

# Desired relative error for the Gauss-Legendre integral evaluation in `_integrate_part`.
RELATIVE_ERROR = 1e-4

# `total_eflux` for `ThickTarget` is expressed in units of this many electrons/s.
THICK_TARGET_EFLUX_SCALE = 1e35  # e/s

# `total_eflux` for `ThinTarget` is expressed in units of this many cm^-2 s^-1.
THIN_TARGET_EFLUX_SCALE = 1e55  #  cm^-2 s^-1.

# Coefficient inside the argument of the Coulomb logarithm ln(Lambda) in `collisional_loss`, in
# units of keV^-1 (so ``COULOMB_LOGARITHM_COEFFICIENT * electron_energy[keV]`` is dimensionless).
# This gives ln(Lambda) ~ 25-27 for electron energies of 10-100 keV, consistent with the
# "typically ln(Lambda) ~ 20" quoted by Leach & Petrosian (1981, ApJ 251, 781) for fast electron,
# Can't see the exact closed-form derivation of this specific coefficient
# against those formulas. Carried over unchanged from SSW `Brm_ELoss
# <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm/brm_eloss.pro>`_.
COULOMB_LOGARITHM_COEFFICIENT = 6.9447e9  # keV^-1


class ThickTarget(FittableModel):
    r"""Calculates the thick-target bremsstrahlung radiation of a dual power-law electron distribution.

    [1] Brown, Solar Physics 18, 489 (1971) (https://link.springer.com/article/10.1007/BF00149070)
    [2] https://hesperia.gsfc.nasa.gov/ssw/packages/xray/doc/brm_thick_doc.pdf
    [3] https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_thicktarget.pro

    Parameters
    ----------
    energy_edges : 1d array
        Edges of energy bins in units of keV.
    total_eflux : int or float
        Total integrated electron flux, in units of 10^35 e^- s^-1.
        Need to take care here as the model returns units of cm-2 sec-1 as the scaling factor of 1e35 is hidden.
        So actual units are 1.0d35 e^- s^-1.
    p : int or float
        Power-law index of the electron distribution below the break.
    break_energy : int or float
        Break energy of power law.
    q : int or float
        Power-law index of the electron distribution above the break.
    low_e_cutoff : int or float
        Low-energy cut-off of the electron distribution in units of keV.
    high_e_cutoff : int or float
        High-energy cut-off of the electron distribution in units of keV.

    Returns
    -------
    A 1d array of thick-target bremsstrahlung radiation in units
    of ph s^-1 keV^-1.
    """

    n_inputs = 1
    n_outputs = 1

    p = Parameter(name="p", default=2, description="Slope below break", fixed=False)

    break_energy = Parameter(name="break_energy", default=100, unit=u.keV, description="Break Energy", fixed=False)

    q = Parameter(name="q", default=5, min=0.01, description="Slope above break", fixed=True)

    low_e_cutoff = Parameter(
        name="low_e_cutoff", default=7, unit=u.keV, description="Low energy electron cut off", fixed=False
    )

    high_e_cutoff = Parameter(
        name="high_e_cutoff", default=1500, unit=u.keV, description="High energy electron cut off", fixed=True
    )

    total_eflux = Parameter(
        name="total_eflux", default=1.5, unit=u.electron * u.s**-1, description="Total electron flux", fixed=True
    )

    _input_units_allow_dimensionless = True

    def __init__(
        self,
        p=p.default,
        break_energy=u.Quantity(break_energy.default, break_energy.unit),
        q=q.default,
        low_e_cutoff=u.Quantity(low_e_cutoff.default, low_e_cutoff.unit),
        high_e_cutoff=u.Quantity(high_e_cutoff.default, high_e_cutoff.unit),
        total_eflux=u.Quantity(total_eflux.default, total_eflux.unit),
        integrator=None,
        **kwargs,
    ):
        self.integrator = integrator

        super().__init__(
            p=p,
            break_energy=break_energy,
            q=q,
            low_e_cutoff=low_e_cutoff,
            high_e_cutoff=high_e_cutoff,
            total_eflux=total_eflux,
            **kwargs,
        )

    def evaluate(self, energy_edges, p, break_energy, q, low_e_cutoff, high_e_cutoff, total_eflux):
        energy_centers = energy_edges[:-1] + 0.5 * np.diff(energy_edges)

        # `<<` attaches a unit to a bare value, or converts an existing (compatible) one so bremsstrahlung_thick_target
        # always receives plain floats/arrays
        output = bremsstrahlung_thick_target(
            (energy_centers << u.keV).value,
            p,
            (break_energy << u.keV).value,
            q,
            (low_e_cutoff << u.keV).value,
            (high_e_cutoff << u.keV).value,
            self.integrator,
        )
        output[~np.isfinite(output)] = 0

        return output * (total_eflux << (u.electron / u.s)).value * THICK_TARGET_EFLUX_SCALE

    @property
    def input_units(self):
        # The units for the 'energy_edges' variable should be an energy (default keV)
        return {self.inputs[0]: u.keV}

    @property
    def return_units(self):
        return {self.outputs[0]: u.ph * u.keV**-1 * u.s**-1}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {
            "break_energy": u.keV,
            "low_e_cutoff": u.keV,
            "high_e_cutoff": u.keV,
            "total_eflux": u.electron * u.s**-1,
        }


class ThinTarget(FittableModel):
    r"""Calculates the thin-target bremsstrahlung radiation of a dual power-law electron distribution.

    [1] Brown, Solar Physics 18, 489 (1971) (https://link.springer.com/article/10.1007/BF00149070)
    [2] https://hesperia.gsfc.nasa.gov/ssw/packages/xray/doc/brm_thin_doc.pdf
    [3] https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_thintarget.pro

    Parameters
    ----------
    energy_edges : 1d array
        Edges of energy bins in units of keV.
    total_eflux : int or float
        normalization factor in units of 1.0d55 cm-2 sec-1,
        i.e. plasma density * volume of source * integrated nonthermal electron flux density
        Need to take care here as the model returns units of cm-2 sec-1 as the scaling factor of 1e55 is hidden.
        So actual units are 1.0d55 cm-2 sec-1.
    p : int or float
        Power-law index of the electron distribution below the break.
    break_energy : int or float
        Break energy of power law.
    q : int or float
        Power-law index of the electron distribution above the break.
    low_e_cutoff : int or float
        Low-energy cut-off of the electron distribution in units of keV.
    high_e_cutoff : int or float
        High-energy cut-off of the electron distribution in units of keV.

    Returns
    -------
    A 1d array of thin-target bremsstrahlung radiation in units
    of ph s^-1 keV^-1.
    """

    n_inputs = 1
    n_outputs = 1

    p = Parameter(
        name="p",
        default=2,
        description="Slope below break",
        fixed=False,
    )

    break_energy = Parameter(name="break_energy", default=100, unit=u.keV, description="Break Energy", fixed=False)

    q = Parameter(name="q", default=5, min=0.01, description="Slope above break", fixed=True)

    low_e_cutoff = Parameter(
        name="low_e_cutoff", default=7, unit=u.keV, description="Low energy electron cut off", fixed=False
    )

    high_e_cutoff = Parameter(
        name="high_e_cutoff", default=1500, unit=u.keV, description="High energy electron cut off", fixed=True
    )

    total_eflux = Parameter(
        name="total_eflux", default=1.5, unit=u.s**-1 * u.cm**-2, description="Total electron flux", fixed=True
    )

    _input_units_allow_dimensionless = True

    def __init__(
        self,
        p=p.default,
        break_energy=u.Quantity(break_energy.default, break_energy.unit),
        q=q.default,
        low_e_cutoff=u.Quantity(low_e_cutoff.default, low_e_cutoff.unit),
        high_e_cutoff=u.Quantity(high_e_cutoff.default, high_e_cutoff.unit),
        total_eflux=u.Quantity(total_eflux.default, total_eflux.unit),
        integrator=None,
        **kwargs,
    ):
        self.integrator = integrator

        super().__init__(
            p=p,
            break_energy=break_energy,
            q=q,
            low_e_cutoff=low_e_cutoff,
            high_e_cutoff=high_e_cutoff,
            total_eflux=total_eflux,
            **kwargs,
        )

    def evaluate(self, energy_edges, p, break_energy, q, low_e_cutoff, high_e_cutoff, total_eflux):
        energy_centers = energy_edges[:-1] + 0.5 * np.diff(energy_edges)

        # `<<` attaches a unit to a bare value, or converts an existing (compatible) one so bremsstrahlung_thin_target
        # always receives plain floats/arrays
        output = bremsstrahlung_thin_target(
            (energy_centers << u.keV).value,
            p,
            (break_energy << u.keV).value,
            q,
            (low_e_cutoff << u.keV).value,
            (high_e_cutoff << u.keV).value,
            efd=True,
            integrator=self.integrator,
        )
        output[~np.isfinite(output)] = 0

        return output * (total_eflux << (u.s**-1 * u.cm**-2)).value * THIN_TARGET_EFLUX_SCALE

    @property
    def input_units(self):
        # The units for the 'energy_edges' variable should be an energy (default keV)
        return {self.inputs[0]: u.keV}

    @property
    def return_units(self):
        return {self.outputs[0]: u.ph * u.keV**-1 * u.s**-1}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {
            "break_energy": u.keV,
            "low_e_cutoff": u.keV,
            "high_e_cutoff": u.keV,
            "total_eflux": u.s**-1 * u.cm**-2,
        }


@dataclass(kw_only=True)
class BrokenPowerLawElectronDistribution:
    """
    A broken or double power law electron flux distribution and integral.

    This class is intended to be used with `~sunkit_spex.models.physical.nonthermal.bremsstrahlung_thin_target` and
    `~sunkit_spex.models.physical.nonthermal.bremsstrahlung_thick_target`.

    Parameters
    ----------
    p : `float`
        Power law index below the break energy `ebrk`
    q : `float`
        Power law index below the break energy `ebrk`
    low_e_cutoff : `float`
        Low energy cutoff
    break_energy : `float`
        Break energy
    high_e_cutoff : `float`
        High energy cutoff
    norm : `bool` (optional)
        True (default) distribution function is normalized so that the integral from `low_e_cutoff` to
        `high_e_cutoff` is 1. Not included in equality comparisons, to match the pre-existing behavior
        of this class.

    References
    ----------
    See SSW IDl functions
    `brm2_distrn <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_distrn.pro>`_ and
    `brm2_f_distrn <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_f_distrn.pro>`_.
    """

    p: float
    q: float
    low_e_cutoff: float
    break_energy: float
    high_e_cutoff: float
    norm: bool = field(default=True, compare=False)

    def __post_init__(self):
        if self.norm:
            n0 = (self.q - 1.0) / (self.p - 1.0) * self.break_energy ** (self.p - 1) * self.low_e_cutoff ** (1 - self.p)
            n1 = n0 - (self.q - 1.0) / (self.p - 1.0)
            n2 = 1.0 - self.break_energy ** (self.q - 1) * self.high_e_cutoff ** (1 - self.q)
            self._norm_factor = 1.0 / (n1 + n2)
            self._n0 = n0
            self._n2 = n2
        else:
            self._norm_factor = 1.0
            self._n0 = 1.0
            self._n2 = 1.0

    def flux(self, electron_energy):
        """
        The differential electron flux density at the given energies.

        This is the normalized double power-law distribution itself (units of keV^-1, since it
        integrates to 1 over [low_e_cutoff, high_e_cutoff]). Used for the thin-target model, where
        only the instantaneous flux at each energy matters (a single, optically-thin interaction).

        Parameters
        ----------
        electron_energy : `numpy.array`
            Electron energies

        Returns
        -------
        `numpy.array`
            The electron flux density (keV^-1) as a function of electron energy
        """
        conditions = [
            (electron_energy >= self.low_e_cutoff) & (electron_energy < self.break_energy),
            (electron_energy >= self.break_energy) & (electron_energy <= self.high_e_cutoff),
        ]
        functions = [
            lambda e: (
                self._norm_factor * self._n0 * (self.p - 1.0) * e ** (-self.p) * self.low_e_cutoff ** (self.p - 1.0)
            ),
            lambda e: self._norm_factor * (self.q - 1.0) * e ** (-self.q) * self.break_energy ** (self.q - 1.0),
        ]
        return np.piecewise(electron_energy, conditions, functions)

    def density(self, electron_energy):
        """
        The cumulative electron flux from `electron_energy` up to `high_e_cutoff`.

        Despite the name (inherited from SSW `Brm2_F_Distrn
        <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_f_distrn.pro>`_), this is
        *not* a density: it is the integral of `flux` from `electron_energy` to `high_e_cutoff`,
        i.e. the dimensionless fraction of the total (normalized) electron flux carried by
        electrons at or above `electron_energy`. Used for the thick-target model, where electrons
        continuously lose energy through collisions as they penetrate the target (Brown 1971), so
        the relevant quantity at a given energy is how much flux is left above it, not the
        instantaneous flux at that energy alone.

        Parameters
        ----------
        electron_energy : `numpy.array`
            Electron energies

        Returns
        -------
        `numpy.array`
            The dimensionless cumulative electron flux fraction as a function of electron energy
        """
        conditions = [
            electron_energy < self.low_e_cutoff,
            (electron_energy >= self.low_e_cutoff) & (electron_energy < self.break_energy),
            (electron_energy >= self.break_energy) & (electron_energy <= self.high_e_cutoff),
        ]
        functions = [
            1.0,
            lambda e: (
                self._norm_factor
                * (
                    self._n0 * self.low_e_cutoff ** (self.p - 1) * e ** (1.0 - self.p)
                    - (self.q - 1.0) / (self.p - 1.0)
                    + self._n2
                )
            ),
            lambda e: self._norm_factor * (self.break_energy ** (self.q - 1) * e ** (1.0 - self.q) - (1.0 - self._n2)),
        ]
        return np.piecewise(electron_energy, conditions, functions)


def collisional_loss(electron_energy):
    """
    Compute the energy dependent terms of the collisional energy loss rate for energetic electrons.

    Parameters
    ----------
    electron_energy : `numpy.array`
        Array of electron kinetic energies at which to evaluate loss, in keV

    Returns
    -------
    `numpy.array`
        Energy loss rate (dimensionless; see Notes)

    Notes
    -----
    Initial version modified from SSW
    `Brm_ELoss <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm/brm_eloss.pro>`_
    """
    electron_rest_mass = MC2  # m_e c^2, keV

    # Lorentz factor (dimensionless): gamma = (kinetic energy / rest-mass energy) + 1
    gamma = (electron_energy / electron_rest_mass) + 1.0

    # Speed as a fraction of the speed of light (dimensionless): beta = v / c
    beta = np.sqrt(1.0 - (1.0 / gamma**2))

    return np.log(COULOMB_LOGARITHM_COEFFICIENT * electron_energy) / beta


def bremsstrahlung_cross_section(electron_energy, photon_energy, z=MEAN_ATOMIC_NUMBER):
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

    mc2 = MC2
    alpha = ALPHA
    twoar02 = TWOAR02

    # Numerical coefficients
    c11 = 4.0 / 3.0
    c12 = 7.0 / 15.0
    c13 = 11.0 / 70.0
    c21 = 7.0 / 20.0
    c22 = 9.0 / 28.0
    c23 = 263.0 / 210.0

    # Calculate normalised photon and total electron energies. photon_energy may already be
    # broadcastable against electron_energy; only add trailing axes if it genuinely has fewer dimensions.
    k = photon_energy / mc2
    if k.ndim < electron_energy.ndim:
        k = k.reshape(k.shape + (1,) * (electron_energy.ndim - k.ndim))
    e1 = (electron_energy / mc2) + 1.0

    # Calculate energies of scattered electrons and normalized momenta.
    e2 = e1 - k
    p1 = np.sqrt(e1**2 - 1.0)
    p2 = np.sqrt(e2**2 - 1.0)

    # Define reused quantities.
    e1e2 = e1 * e2
    p1p2 = p1 * p2
    p1_sq = p1 * p1
    p2sum = p1_sq + p2**2
    k2 = k**2
    # e1e2 * e1e2 * e1e2 rather than e1e2**3: numpy's ** does not fast-path integer exponents
    # above 2 the way it does for **2, making e1e2**3 measurably slower here
    e1e23 = e1e2 * e1e2 * e1e2
    pe = p2sum / e1e23

    # Define terms in cross section.
    ch1 = (c11 * e1e2 + k2) - (c12 * k2 / e1e2) - (c13 * k2 * pe / e1e2)
    ch2 = 1.0 + (1.0 / e1e2) + (c21 * pe) + (c22 * k2 + c23 * p1p2**2) / e1e23

    # Collect terms.
    crtmp = ch1 * (2.0 * np.log((e1e2 + p1p2 - 1.0) / k) - (p1p2 / e1e2) * ch2)
    crtmp = z**2 * crtmp / (k * p1_sq)

    # Compute the Elwert factor.
    a1 = alpha * z * e1 / p1
    a2 = alpha * z * e2 / p2

    fe = (a2 / a1) * (1.0 - np.exp(-2.0 * np.pi * a1)) / (1.0 - np.exp(-2.0 * np.pi * a2))

    # Compute the differential cross section (units cm^2).
    return twoar02 * fe * crtmp


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
        with `~sunkit_spex.models.physical.nonthermal.BrokenPowerLawElectronDistribution.flux`. False, the electron
        density distribution (electrons cm^-3 keV^-1) is calculated with
        `~sunkit_spex.models.physical.nonthermal.BrokenPowerLawElectronDistribution.density`.

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
    mc2 = MC2
    clight = CLIGHT

    # L=log10 (E), E=l0L and dE=10L ln(10) dL hence the electron_energy * LN_10 below.
    # exp(x_log * LN_10) rather than 10**x_log: numpy's ** has no fast path for a scalar base
    # raised to an array exponent, making it ~2.5x slower than the equivalent exp/log form here
    electron_energy = np.exp(x_log * LN_10)
    brem_cross = bremsstrahlung_cross_section(electron_energy, photon_energy, z)
    collision_loss = collisional_loss(electron_energy)
    pc = np.sqrt(electron_energy * (electron_energy + 2.0 * mc2))

    if model == "thick-target":
        density = electron_dist.density(electron_energy)
        return electron_energy * LN_10 * density * brem_cross * pc / collision_loss / ((electron_energy / mc2) + 1.0)
    if model == "thin-target":
        if efd:
            return electron_energy * LN_10 * electron_dist.flux(electron_energy) * brem_cross * (mc2 / clight)
        return (
            electron_energy
            * LN_10
            * electron_dist.flux(electron_energy)
            * brem_cross
            * pc
            / ((electron_energy / mc2) + 1.0)
        )
    return None


def _integrate_part(
    *,
    model,
    photon_energies,
    maxfcn,
    rerr,
    low_e_cutoff,
    break_energy,
    high_e_cutoff,
    p,
    q,
    z,
    a_lg,
    b_lg,
    indices,
    efd,
    integrator=None,
):
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
        calls for two digits to be correct.
    photon_energies : `numpp.array`
        Photon energies
    low_e_cutoff : `float`
        Low energy electron cut off
    break_energy : `float`
        Break energy
    high_e_cutoff : `float`
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
    indices : `numpy.array`
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
    nlim = NLIM

    intsum = np.zeros_like(photon_energies, dtype=np.float64)
    ier = np.zeros_like(photon_energies)

    if integrator is None:
        integrator = fixed_quad_batch
    elif not callable(integrator):
        raise TypeError("integrator must be a callable")

    electron_dist = BrokenPowerLawElectronDistribution(
        p=p, q=q, low_e_cutoff=low_e_cutoff, break_energy=break_energy, high_e_cutoff=high_e_cutoff
    )

    for ires in range(2, nlim + 1):
        npoint = 2**ires
        if npoint > maxfcn:
            ier[indices] = 1
            return intsum, ier

        lastsum = np.copy(intsum)

        intsum[indices] = integrator(
            _get_integrand,
            a_lg[indices],
            b_lg[indices],
            n=npoint,
            func_kwargs={
                "model": model,
                "electron_dist": electron_dist,
                "photon_energy": photon_energies[indices],
                "z": z,
                "efd": efd,
            },
        )

        # Convergence criterion: narrow `indices` down to the points that haven't converged yet.
        l1 = np.abs(intsum - lastsum)
        l2 = rerr * np.abs(intsum)
        indices = np.where(l1 > l2)[0]

        # If all point have reached criterion return value and flags
        if indices.size == 0:
            return intsum, ier
    return None


def _split_and_integrate(
    *, model, photon_energies, maxfcn, rerr, low_e_cutoff, break_energy, high_e_cutoff, p, q, z, efd, integrator=None
):
    """
    Split and integrate the continuous parts of the electron spectrum.

    Used for both thin- and thick-target calculations from a double power-law electron
    distribution, to integrate a function via the method of Gaussian quadrature. Repeatedly doubles the number of
    points evaluated until convergence, specified by the input rerr, is obtained, or the maximum
    number of points, specified by the input maxfcn, is reached. If integral convergence is not
    achieved, this function raises a ValueError when either the maximum number of function
    evaluations is performed or the number of Gaussian points to be evaluated exceeds maxfcn.
    Maxfcn should be less than or equal to 2^nlim, or 4096 with nlim = 12. This function splits the
    numerical integration into up to three parts and returns the sum of the parts. This avoids
    numerical problems with discontinuities in the electron distribution function at low_e_cutoff and
    break_energy.

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
    low_e_cutoff : `float`
        Low energy electron cutoff
    break_energy : `float`
        Break energy
    high_e_cutoff : `float`
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
    mc2 = MC2
    clight = CLIGHT

    if not low_e_cutoff <= break_energy <= high_e_cutoff:
        raise ValueError(
            f"Condition low_e_cutoff <= eebrek <= high_e_cutoff not satisfied ({low_e_cutoff}<={break_energy}<={high_e_cutoff})."
        )

    # The electron distribution has kinks at low_e_cutoff and break_energy, so for each photon
    # energy the integral up to high_e_cutoff is split into up to three continuous segments at
    # those two points, each integrated separately, then summed. Segment 1 (below low_e_cutoff)
    # only applies to the thick-target model, where electrons below low_e_cutoff still contribute
    # via prior collisional energy loss; the thin-target model only sees electrons above
    # low_e_cutoff directly. Assumes photon energies montonicaly ascending
    below_low_e_cutoff = np.where(photon_energies < low_e_cutoff)[0]
    below_break_energy = np.where(photon_energies < break_energy)[0]
    below_high_e_cutoff = np.where(photon_energies <= high_e_cutoff)[0]

    segments = []
    if model == "thick-target" and below_low_e_cutoff.size > 0:
        segments.append(("Part 1", below_low_e_cutoff, photon_energies, low_e_cutoff))
    if below_break_energy.size > 0 and break_energy > low_e_cutoff:
        # Below low_e_cutoff, this segment's lower bound is clamped to low_e_cutoff: that part of
        # the range is already covered by segment 1 (thick-target) or doesn't apply (thin-target).
        lower_bound = np.copy(photon_energies)
        lower_bound[below_low_e_cutoff] = low_e_cutoff
        segments.append(("Part 2", below_break_energy, lower_bound, break_energy))
    if below_high_e_cutoff.size > 0 and high_e_cutoff > break_energy:
        # Likewise clamped to break_energy below it, which segment 2 already covers.
        lower_bound = np.copy(photon_energies)
        lower_bound[below_break_energy] = break_energy
        segments.append(("Part 3", below_high_e_cutoff, lower_bound, high_e_cutoff))

    total = np.zeros_like(photon_energies, dtype=np.float64)
    ier_total = np.zeros_like(photon_energies, dtype=np.float64)
    for name, indices, lower_bound, upper_bound in segments:
        logger.debug(name)
        a_lg = np.log10(lower_bound)
        b_lg = np.log10(np.full_like(lower_bound, upper_bound))
        segment_sum, segment_ier = _integrate_part(
            model=model,
            maxfcn=maxfcn,
            rerr=rerr,
            photon_energies=photon_energies,
            low_e_cutoff=low_e_cutoff,
            break_energy=break_energy,
            high_e_cutoff=high_e_cutoff,
            p=p,
            q=q,
            z=z,
            a_lg=a_lg,
            b_lg=b_lg,
            indices=indices,
            efd=efd,
            integrator=integrator,
        )

        # ier = 1 indicates no convergence.
        if segment_ier.sum():
            raise ValueError(f"{name} integral did not converge for some photon energies.")

        total += segment_sum
        ier_total += segment_ier

    # Segment 1 (and its contribution to `total`) is always zero for the thin-target model, so no
    # separate combination step is needed there.
    if model == "thick-target":
        return total * (mc2 / clight), ier_total
    if model == "thin-target":
        return total, ier_total
    return None


def bremsstrahlung_thin_target(
    photon_energies, p, break_energy, q, low_e_cutoff, high_e_cutoff, efd=True, integrator=None
):
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
    break_energy : `float`
        Break energy
    q : `float`
        Slope above the break energy
    low_e_cutoff : `float`
        Low energy electron cut off
    high_e_cutoff : `float`
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
    mc2 = MC2
    clight = CLIGHT

    maxfcn = MAX_QUADRATURE_POINTS
    z = MEAN_ATOMIC_NUMBER
    rerr = RELATIVE_ERROR

    # Numerical coefficient for photon flux. SSW's Brm2_ThinTarget includes an extra
    # 1/(4*pi*au**2) factor here to normalise to a detector at 1 AU from the source; this Python
    # port deliberately omits it, so the returned flux is NOT distance-normalised (see
    # test_nonthermal.py, which compensates by scaling the SSW reference values by 4*pi*(1 AU)**2
    # before comparing).
    fcoeff = clight / mc2**2.0

    # Create array for the photon flux. _split_and_integrate also returns a per-point convergence
    # flag array, but it's redundant here: _split_and_integrate already raises ValueError itself
    # if any point fails to converge, so there's nothing left to do with the flags afterward.
    flux = np.zeros_like(photon_energies, dtype=np.float64)

    if low_e_cutoff >= high_e_cutoff:
        raise ValueError("high_e_cutoff must be larger than low_e_cutoff!")

    valid = (photon_energies < high_e_cutoff) & (photon_energies > 0)
    if valid.any():
        flux[valid], _ = _split_and_integrate(
            model="thin-target",
            photon_energies=photon_energies[valid],
            maxfcn=maxfcn,
            rerr=rerr,
            low_e_cutoff=low_e_cutoff,
            break_energy=break_energy,
            high_e_cutoff=high_e_cutoff,
            p=p,
            q=q,
            z=z,
            efd=efd,
            integrator=integrator,
        )

        flux *= fcoeff

        return flux
    raise ValueError("The photon energies are higher than the highest electron energy or not greater than zero")


def bremsstrahlung_thick_target(photon_energies, p, break_energy, q, low_e_cutoff, high_e_cutoff, integrator=None):
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
    break_energy : `float`
        Break energy
    q : `float`
        Slope above the break energy
    low_e_cutoff : `float`
        Low energy electron cut off
    high_e_cutoff : `float`
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
    mc2 = MC2
    clight = CLIGHT
    r0 = R0

    maxfcn = MAX_QUADRATURE_POINTS
    z = MEAN_ATOMIC_NUMBER
    rerr = RELATIVE_ERROR

    # Numerical coefficient for photon flux. SSW's Brm2_ThickTarget includes an extra
    # 1/(4*pi*au**2) factor here to normalize to a detector at 1 AU from the source; this Python
    # port deliberately omits it, so the returned flux is NOT distance-normalized (see
    # test_nonthermal.py, which compensates by scaling the SSW reference values by 4*pi*(1 AU)**2
    # before comparing).
    fcoeff = clight**2 / mc2**4

    decoeff = 4.0 * np.pi * (r0**2) * clight

    # Create array for the photon flux. _split_and_integrate also returns a per-point convergence
    # flag array, but it's redundant here: _split_and_integrate already raises ValueError itself
    # if any point fails to converge, so there's nothing left to do with the flags afterward.
    flux = np.zeros_like(photon_energies, dtype=np.float64)

    if low_e_cutoff >= high_e_cutoff:
        return flux

    valid = (photon_energies < high_e_cutoff) & (photon_energies > 0)

    if valid.any():
        flux[valid], _ = _split_and_integrate(
            model="thick-target",
            photon_energies=photon_energies[valid],
            maxfcn=maxfcn,
            rerr=rerr,
            low_e_cutoff=low_e_cutoff,
            break_energy=break_energy,
            high_e_cutoff=high_e_cutoff,
            p=p,
            q=q,
            z=z,
            efd=False,
            integrator=integrator,
        )

        return (fcoeff / decoeff) * flux

    raise ValueError("The photon energies are higher than the highest electron energy or not greater than zero")
