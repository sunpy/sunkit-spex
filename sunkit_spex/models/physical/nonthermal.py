import logging

import numpy as np

import astropy.units as u
from astropy.modeling import FittableModel, Parameter

from sunkit_spex.legacy import constants as const
from sunkit_spex.legacy.integrate import gauss_legendre

const = const.Constants()


logging = logging.getLogger(__name__)

"""
Functions for computing the photon flux due to bremsstrahlung radiation from energetic electrons
impacting a dense plasma. See [1]_ and [2]_.


References
----------

.. [1] Thick-Target: https://hesperia.gsfc.nasa.gov/hessi/flarecode/bremthickdoc.pdf
.. [2] Thin-Target: https://hesperia.gsfc.nasa.gov/hessi/flarecode/bremdoc.pdf

"""


__all__ = ["ThickTarget", "ThinTarget"]


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

        if (
            hasattr(break_energy, "unit")
            or hasattr(energy_centers, "unit")
            or hasattr(low_e_cutoff, "unit")
            or hasattr(high_e_cutoff, "unit")
            or hasattr(total_eflux, "unit")
        ):
            flux = thick_fn(
                energy_centers.value,
                p,
                break_energy.value,
                q,
                low_e_cutoff.value,
                high_e_cutoff.value,
                total_eflux.value,
                self.integrator,
            )
        else:
            flux = thick_fn(
                energy_centers, p, break_energy, q, low_e_cutoff, high_e_cutoff, total_eflux, self.integrator
            )

        return flux

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
    [2] https://hesperia.gsfc.nasa.gov/ssw/packages/xray/doc/brm_thick_doc.pdf
    [3] https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_thicktarget.pro

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

        if (
            hasattr(break_energy, "unit")
            or hasattr(energy_centers, "unit")
            or hasattr(low_e_cutoff, "unit")
            or hasattr(high_e_cutoff, "unit")
            or hasattr(total_eflux, "unit")
        ):
            flux = thin_fn(
                energy_centers.value,
                p,
                break_energy.value,
                q,
                low_e_cutoff.value,
                high_e_cutoff.value,
                total_eflux.value,
                self.integrator,
            )
        else:
            flux = thin_fn(
                energy_centers, p, break_energy, q, low_e_cutoff, high_e_cutoff, total_eflux, self.integrator
            )

        return flux

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
            # "total_eflux": u.electron * u.s**-1,
        }


def thick_fn(energy_centers, p, break_energy, q, low_e_cutoff, high_e_cutoff, total_eflux, integrator):
    """Calculates the thick-target bremsstrahlung radiation of a dual power-law electron distribution.

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

    # hack = np.round([p, break_energy, q, low_e_cutoff, high_e_cutoff, total_eflux], 15)
    # p, break_energy, q, low_e_cutoff, high_e_cutoff, total_eflux = hack[0], hack[1], hack[2], hack[3], hack[4], hack[5]

    # energies = np.mean(energies, axis=1)  # since energy bins are given, use midpoints though

    # we want a single power law electron distribution,
    # so set break_energy == high_e_cutoff at a high value.
    # we don't care about q at E > break_energy.
    # high_break = energies.max() * 10

    output = bremsstrahlung_thick_target(energy_centers, p, break_energy, q, low_e_cutoff, high_e_cutoff, integrator)

    output[np.isnan(output)] = 0
    output[~np.isfinite(output)] = 0

    # convert to 1e35 e-/s
    return output * total_eflux * 1e35


# def thin_fn(total_eflux, index, e_c, energies=None):
def thin_fn(energy_centers, p, break_energy, q, low_e_cutoff, high_e_cutoff, total_eflux, integrator):
    """Calculates the thin-target bremsstrahlung radiation of a dual power-law electron distribution.

    [1] Brown, Solar Physics 18, 489 (1971) (https://link.springer.com/article/10.1007/BF00149070)
    [2] https://hesperia.gsfc.nasa.gov/ssw/packages/xray/doc/brm_thick_doc.pdf
    [3] https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_thicktarget.pro

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

    # hack = np.round([total_eflux, index, e_c], 15)
    # total_eflux, index, e_c = hack[0], hack[1], hack[2]

    # energies = np.mean(energies, axis=1)  # since energy bins are given, use midpoints though
    # energies = energy_centers
    # we want a single power law electron distribution,
    # so set break_energy == high_e_cutoff at a high value.
    # we don't care about q at E > break_energy.
    # high_break = energies.max() * 10
    output = bremsstrahlung_thin_target(
        energy_centers, p, break_energy, q, low_e_cutoff, high_e_cutoff, total_eflux, integrator
    )

    output[np.isnan(output)] = 0
    output[~np.isfinite(output)] = 0

    # convert to 1e35 e-/s
    return output * total_eflux * 1e55


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
    low_e_cutoff : `float`
        Low energy cutoff
    break_energy : `float`
        Break energy
    high_e_cutoff : `float`
        High energy cutoff
    norm : `bool` (optional)
        True (default) distribution function is normalized so that the integral from `low_e_cutoff` to
        `high_e_cutoff` is 1.

    References
    ----------
    See SSW IDl functions
    `brm2_distrn <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_distrn.pro>`_ and
    `brm2_f_distrn <https://hesperia.gsfc.nasa.gov/ssw/packages/xray/idl/brm2/brm2_f_distrn.pro>`_.
    """

    def __init__(self, *, p, q, low_e_cutoff, break_energy, high_e_cutoff, norm=True):
        """ """
        self.p = p
        self.q = q
        self.low_e_cutoff = low_e_cutoff
        self.break_energy = break_energy
        self.high_e_cutoff = high_e_cutoff
        self.norm = norm
        if self.norm:
            n0 = (q - 1.0) / (p - 1.0) * break_energy ** (p - 1) * low_e_cutoff ** (1 - p)
            n1 = n0 - (q - 1.0) / (p - 1.0)
            n2 = 1.0 - break_energy ** (q - 1) * high_e_cutoff ** (1 - q)
            self._norm_factor = 1.0 / (n1 + n2)
            self._n0 = n0
            self._n2 = n2
        else:
            self._norm_factor = 1.0
            self._n0 = 1.0
            self._n2 = 1.0

    def __eq__(self, other):
        return all(
            getattr(self, name) == getattr(other, name)
            for name in ["p", "q", "low_e_cutoff", "break_energy", "high_e_cutoff"]
        ) and isinstance(other, self.__class__)

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

        index = np.where(electron_energy < self.low_e_cutoff)
        if index[0].size > 0:
            res[index] = 0.0

        index = np.where((electron_energy < self.break_energy) & (electron_energy >= self.low_e_cutoff))
        if index[0].size > 0:
            res[index] = (
                self._norm_factor
                * self._n0
                * (self.p - 1.0)
                * electron_energy[index] ** (-self.p)
                * self.low_e_cutoff ** (self.p - 1.0)
            )

        index = np.where((electron_energy <= self.high_e_cutoff) & (electron_energy >= self.break_energy))
        if index[0].size > 0:
            res[index] = (
                self._norm_factor
                * (self.q - 1.0)
                * electron_energy[index] ** (-self.q)
                * self.break_energy ** (self.q - 1.0)
            )

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

        index = np.where(electron_energy < self.low_e_cutoff)
        if index[0].size > 0:
            res[index] = 1.0

        index = np.where((electron_energy < self.break_energy) & (electron_energy >= self.low_e_cutoff))
        if index[0].size > 0:
            res[index] = self._norm_factor * (
                self._n0 * self.low_e_cutoff ** (self.p - 1) * electron_energy[index] ** (1.0 - self.p)
                - (self.q - 1.0) / (self.p - 1.0)
                + self._n2
            )

        index = np.where((electron_energy <= self.high_e_cutoff) & (electron_energy >= self.break_energy))
        if index[0].size > 0:
            res[index] = self._norm_factor * (
                self.break_energy ** (self.q - 1) * electron_energy[index] ** (1.0 - self.q) - (1.0 - self._n2)
            )
        return res

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, q={self.q}, low_e_cutoff={self.low_e_cutoff}, "
            f"break_energy={self.break_energy}, high_e_cutoff={self.high_e_cutoff}, norm={self.norm})"
        )


def collisional_loss(electron_energy):
    """
    Compute the energy dependent terms of the collisional energy loss rate for energetic electrons.

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
    electron_rest_mass = const.get_constant("mc2")  # * u.keV #c.m_e * c.c**2

    gamma = (electron_energy / electron_rest_mass) + 1.0

    beta = np.sqrt(1.0 - (1.0 / gamma**2))

    # TODO figure out what number is?
    return np.log(6.9447e9 * electron_energy) / beta


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

    mc2 = const.get_constant("mc2")
    alpha = const.get_constant("alpha")
    twoar02 = const.get_constant("twoar02")

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
    p1 = np.sqrt(e1**2 - 1.0)
    p2 = np.sqrt(e2**2 - 1.0)

    # Define frequently used quantities.
    e1e2 = e1 * e2
    p1p2 = p1 * p2
    p2sum = p1**2 + p2**2
    k2 = k**2
    e1e23 = e1e2**3
    pe = p2sum / e1e23

    # Define terms in cross section.
    ch1 = (c11 * e1e2 + k2) - (c12 * k2 / e1e2) - (c13 * k2 * pe / e1e2)
    ch2 = 1.0 + (1.0 / e1e2) + (c21 * pe) + (c22 * k2 + c23 * p1p2**2) / e1e23

    # Collect terms.
    crtmp = ch1 * (2.0 * np.log((e1e2 + p1p2 - 1.0) / k) - (p1p2 / e1e2) * ch2)
    crtmp = z**2 * crtmp / (k * p1**2)

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
    mc2 = const.get_constant("mc2")
    clight = const.get_constant("clight")

    # L=log10 (E), E=l0L and dE=10L ln(10) dL hence the electron_energy * np.log(10) below
    electron_energy = 10**x_log
    brem_cross = bremsstrahlung_cross_section(electron_energy, photon_energy, z)
    collision_loss = collisional_loss(electron_energy)
    pc = np.sqrt(electron_energy * (electron_energy + 2.0 * mc2))

    density = electron_dist.density(electron_energy)
    if model == "thick-target":
        return (
            electron_energy * np.log(10) * density * brem_cross * pc / collision_loss / ((electron_energy / mc2) + 1.0)
        )
    if model == "thin-target":
        if efd:
            return electron_energy * np.log(10) * electron_dist.flux(electron_energy) * brem_cross * (mc2 / clight)
        return (
            electron_energy
            * np.log(10)
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
    ll,
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
        raise TypeError("integrator must be a callable")

    # Copy indices over which to carry out the integration
    i = ll[:]

    electron_dist = BrokenPowerLawElectronDistribution(
        p=p, q=q, low_e_cutoff=low_e_cutoff, break_energy=break_energy, high_e_cutoff=high_e_cutoff
    )

    for ires in range(2, nlim + 1):
        npoint = 2**ires
        if npoint > maxfcn:
            ier[i] = 1
            return intsum, ier

        lastsum = np.copy(intsum)

        intsum[i] = integrator(
            _get_integrand,
            a_lg[i],
            b_lg[i],
            n=npoint,
            func_kwargs={
                "model": model,
                "electron_dist": electron_dist,
                "photon_energy": photon_energies[i],
                "z": z,
                "efd": efd,
            },
        )

        # Convergence criterion
        l1 = np.abs(intsum - lastsum)
        l2 = rerr * np.abs(intsum)
        i = np.where(l1 > l2)[0]

        # If all point have reached criterion return value and flags
        if i.size == 0:
            return intsum, ier
    return None


def _split_and_integrate(
    *, model, photon_energies, maxfcn, rerr, low_e_cutoff, break_energy, high_e_cutoff, p, q, z, efd, integrator=None
):
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
    mc2 = const.get_constant("mc2")
    clight = const.get_constant("clight")

    if not low_e_cutoff <= break_energy <= high_e_cutoff:
        raise ValueError(
            f"Condition low_e_cutoff <= eebrek <= high_e_cutoff not satisfied ({low_e_cutoff}<={break_energy}<={high_e_cutoff})."
        )

    # Create arrays for integral sums and error flags.
    intsum1 = np.zeros_like(photon_energies, dtype=np.float64)
    ier1 = np.zeros_like(photon_energies, dtype=np.float64)
    intsum2 = np.zeros_like(photon_energies, dtype=np.float64)
    ier2 = np.zeros_like(photon_energies, dtype=np.float64)
    intsum3 = np.zeros_like(photon_energies, dtype=np.float64)
    ier3 = np.zeros_like(photon_energies, dtype=np.float64)

    P1 = np.where(photon_energies < low_e_cutoff)[0]
    P2 = np.where(photon_energies < break_energy)[0]
    P3 = np.where(photon_energies <= high_e_cutoff)[0]

    # Part 1, below en_val[0] (usually low_e_cutoff)
    if model == "thick-target":
        if P1.size > 0:
            logging.debug("Part1")
            a_lg = np.log10(photon_energies[P1])
            b_lg = np.log10(np.full_like(a_lg, low_e_cutoff))
            i = np.copy(P1)
            intsum1, ier1 = _integrate_part(
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
                ll=i,
                efd=efd,
                integrator=integrator,
            )

            # ier = 1 indicates no convergence.
            if sum(ier1):
                raise ValueError("Part 1 integral did not converge for some photon energies.")

    # Part 2, between enval[0] and en_val[1](usually low_e_cutoff and break_energy)

    aa = np.copy(photon_energies)
    if (P2.size > 0) and (break_energy > low_e_cutoff):
        # TODO check if necessary as integration should only be carried out over point P2 which
        # by definition are not in P1
        if P1.size > 0:
            aa[P1] = low_e_cutoff

        logging.debug("Part2")
        a_lg = np.log10(aa[P2])
        b_lg = np.log10(np.full_like(a_lg, break_energy))
        i = np.copy(P2)
        intsum2, ier2 = _integrate_part(
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
            ll=i,
            efd=efd,
            integrator=integrator,
        )

        if sum(ier2) > 0:
            raise ValueError("Part 2 integral did not converge for some photon energies.")

    # Part 3: between break_energy and high_e_cutoff(usually break_energy and high_e_cutoff)
    aa = np.copy(photon_energies)
    if (P3.sum() > 0) and (high_e_cutoff > break_energy):
        if P2.size > 0:
            aa[P2] = break_energy

        logging.debug("Part3")
        a_lg = np.log10(aa[P3])
        b_lg = np.log10(np.full_like(a_lg, high_e_cutoff))
        i = np.copy(P3)
        intsum3, ier3 = _integrate_part(
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
            ll=i,
            efd=efd,
            integrator=integrator,
        )
        if sum(ier3) > 0:
            raise ValueError("Part 3 integral did not converge for some photon energies.")

    # TODO check units here
    # Combine 3 parts and convert units and return
    if model == "thick-target":
        DmlinO = (intsum1 + intsum2 + intsum3) * (mc2 / clight)
        ier = ier1 + ier2 + ier3
        return DmlinO, ier
    if model == "thin-target":
        Dmlin = intsum2 + intsum3
        ier = ier2 + ier3
        return Dmlin, ier
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
    mc2 = const.get_constant("mc2")
    clight = const.get_constant("clight")
    # au = const.get_constant("au")

    # Max number of points
    maxfcn = 2048

    # Average atomic number
    z = 1.2

    # Relative error
    rerr = 1e-4

    # Numerical coefficient for photo flux
    # fcoeff = (clight / (4 * np.pi * au**2)) / mc2**2.0
    fcoeff = clight / mc2**2.0

    # Create arrays for the photon flux and error flags.
    flux = np.zeros_like(photon_energies, dtype=np.float64)
    iergq = np.zeros_like(photon_energies, dtype=np.float64)

    if low_e_cutoff >= high_e_cutoff:
        raise ValueError("high_e_cutoff must be larger than low_e_cutoff!")

    (l,) = np.where((photon_energies < high_e_cutoff) & (photon_energies > 0))  # noqa: E741
    if l.size > 0:
        flux[l], iergq[l] = _split_and_integrate(
            model="thin-target",
            photon_energies=photon_energies[l],
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
    raise Warning("The photon energies are higher than the highest electron energy or not greater than zero")


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
    # Constants
    mc2 = const.get_constant("mc2")
    clight = const.get_constant("clight")
    # au = const.get_constant("au")
    r0 = const.get_constant("r0")

    # Max number of points
    maxfcn = 2048

    # Average atomic number
    z = 1.2

    # Relative error
    rerr = 1e-4

    # Numerical coefficient for photo flux
    # fcoeff = (clight**2 / mc2**4) / (4 * np.pi * au**2)

    # decoeff = 4.0 * np.pi * (r0**2) * clight

    fcoeff = clight**2 / mc2**4

    decoeff = 4.0 * np.pi * (r0**2) * clight

    # Create arrays for the photon flux and error flags.
    flux = np.zeros_like(photon_energies, dtype=np.float64)
    iergq = np.zeros_like(photon_energies, dtype=np.float64)

    if low_e_cutoff >= high_e_cutoff:
        return flux

    (i,) = np.where((photon_energies < high_e_cutoff) & (photon_energies > 0))

    if i.size > 0:
        flux[i], iergq[i] = _split_and_integrate(
            model="thick-target",
            photon_energies=photon_energies[i],
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

    raise Warning("The photon energies are higher than the highest electron energy or not greater than zero")
