import functools

import numpy as np

import astropy.units as u

PHOTON_RATE_UNIT = u.ph / u.keV / u.cm**2 / u.s


@u.quantity_input
def power_law_integral(
    energy: u.keV,
    norm_energy: u.keV,
    norm: PHOTON_RATE_UNIT,
    index: u.one
) -> (PHOTON_RATE_UNIT * u.keV):
    """Evaluate the antiderivative of a power law at a given energy.

    :param energy: Array of energies to evaluate.
    :type energy: u.keV
    :param norm_energy: Normalization energy.
    :param norm: Function normalization.
    :param index: Power law index.
    :return: Antiderivative of power law evaluated at given energies.
    :rtype: PHOTON_RATE_UNIT
    """
    prefactor = norm * norm_energy
    arg = energy / norm_energy
    if index == 1:
        return prefactor * np.log(arg)
    return prefactor * arg**(1 - index) / (1 - index)


@u.quantity_input
def broken_power_law_binned_flux(
    energy_edges: u.keV,
    reference_energy: u.keV,
    reference_flux: PHOTON_RATE_UNIT,
    break_energy: u.keV,
    lower_index: u.one,
    upper_index: u.one
) -> PHOTON_RATE_UNIT:
    """Analytically evaluate a photon-space broken power law.

    :param energy_edges: 1D array of energy edges.
    :type energy_edges: u.keV
    :param reference_energy: Normalization energy.
    :type reference_energy: u.keV
    :param reference_flux: Normalization flux.
    :type reference_flux: PHOTON_RATE_UNIT
    :param break_energy: Power law break energy.
    :type break_energy: u.keV
    :param lower_index: Power law index below the break.
    :type lower_index: u.one
    :param upper_index: Power law index above the break.
    :type upper_index: u.one
    :return: Photon flux of the broken power law.
    :rtype: PHOTON_RATE_UNIT
    """
    norm_idx = lower_index if reference_energy <= break_energy else upper_index
    # norm is from continuity at break energy and solving.
    norm = reference_flux * (reference_energy / break_energy)**(norm_idx)

    cnd = energy_edges <= break_energy
    lower = energy_edges[cnd]
    upper = energy_edges[~cnd]

    up_integ = functools.partial(
        power_law_integral,
        norm_energy=break_energy,
        norm=norm,
        index=upper_index
    )
    low_integ = functools.partial(
        power_law_integral,
        norm_energy=break_energy,
        norm=norm,
        index=lower_index
    )

    lower_portion = low_integ(energy=lower[1:])
    lower_portion -= low_integ(energy=lower[:-1])

    upper_portion = up_integ(energy=upper[1:])
    upper_portion -= up_integ(energy=upper[:-1])

    twixt_portion = []
    # bin between the portions is comprised of both power laws
    if lower.size > 0 and upper.size > 0:
        twixt_portion = np.diff(
            low_integ(energy=np.array([break_energy.value, upper[0].value]) << u.keV)
        )
        twixt_portion += np.diff(
            up_integ(energy=np.array([lower[-1].value, break_energy.value]) << u.keV)
        )

    ret = np.concatenate((lower_portion, twixt_portion, upper_portion))
    assert ret.size == (energy_edges.size - 1)
    # go back to units of cm2/sec/keV
    return ret / np.diff(energy_edges)
