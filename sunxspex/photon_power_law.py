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
    """Evaluate the antiderivative of a power law at a given energy,
       or vector of energies.
    """
    prefactor = norm * norm_energy
    arg = (energy / norm_energy).to(u.one)
    if index == 1:
        return prefactor * np.log(arg)
    return (prefactor / (1 - index)) * arg**(1 - index)


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
    """

    eng_arg = (break_energy / reference_energy).to(u.one)
    if reference_energy < break_energy:
        low_norm = reference_flux
        up_norm = reference_flux * eng_arg**(upper_index - lower_index)
    else:
        up_norm = reference_flux
        low_norm = reference_flux * eng_arg**(lower_index - upper_index)

    cnd = energy_edges < break_energy
    lower = energy_edges[cnd]
    upper = energy_edges[~cnd]

    up_integ = functools.partial(
        power_law_integral,
        norm_energy=reference_energy,
        norm=up_norm,
        index=upper_index
    )
    low_integ = functools.partial(
        power_law_integral,
        norm_energy=reference_energy,
        norm=low_norm,
        index=lower_index
    )

    lower_portion = low_integ(energy=lower[1:])
    lower_portion -= low_integ(energy=lower[:-1])

    upper_portion = up_integ(energy=upper[1:])
    upper_portion -= up_integ(energy=upper[:-1])

    twixt_portion = []
    # bin between the portions is comprised of both power laws
    # import pdb; pdb.set_trace()
    if lower.size > 0 and upper.size > 0:
        twixt_portion = np.diff(
            low_integ(energy=u.Quantity([lower[-1], break_energy]))
        )
        twixt_portion += np.diff(
            up_integ(energy=u.Quantity([break_energy, upper[0]]))
        )

    ret = np.concatenate((lower_portion, twixt_portion, upper_portion))
    if ret.size != (energy_edges.size - 1):
        raise ValueError('Bin or edge size mismatch. Bug?')

    # go back to units of cm2/sec/keV
    return ret / np.diff(energy_edges)
