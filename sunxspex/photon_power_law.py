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
    r"""Evaluate the antiderivative of a power law at a given energy or vector of energies.

    The power law antiderivative evaluated by this function is assumed to take the following form:
    .. math:: f(E) = N \left( \frac{E}{E_0} \right)^{- \gamma},

    where $E$ is the energy, $N$ is the normalization, $E_0$ is the normalization energy,
    and $\gamma$ is the power law index.

    The value of $\gamma$ is assumed to be positive, but the functional form
    includes a negative sign.

    The special case of $\gamma = 1$ is handled.

    Parameters
    ----------
    energy : `astropy.units.Quantity`
        Energy (or vector of energies) at which to evaluate the power law antiderivative, i.e. $E$.
    norm_energy : `astropy.units.Quantity`
        Energy used for the normalization of the power law argument, i.e. $E_0$.
    norm : `astropy.units.Quantity`
        Normalization of the power law integral, i.e. $N$, in units convertible to ph / (cm2 . keV . s).
    index : `astropy.units.Quantity`
        The power law index, i.e. $\gamma$.

    Returns
    -------
    `astropy.units.Quantity`
        Analytical antiderivative of a power law evaluated at the given energies.
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
    """Analytically evaluate a photon-space broken power law and bin the flux.

    The broken power law is assumed to take the following form:
    .. math:: f(E \le E_b) = N_1 \left( \frac{E}{E_0} \right)^{-\gamma_1},
    .. math:: f(E > E_b) = N_2 \left( \frac{E}{E_0} \right)^{-\gamma_2}

    where $E$ is the energy, $N_1$ and $N_2$ are the normalization below and above the break,
    $E_0$ is the normalization energy, $E_b$ is the break energy, and $\gamma_1$ and $\gamma_2$ are the upper and lower
    power law indices.

    Only one normalization flux and energy are given. Continuity is enforced at the break energy
    so that the normalization is correct at the chosen energy.

    The values of $\gamma_1$ and $\gamma_2$ are assumed to be positive, but the functional form
    includes negative signs.

    Parameters
    ----------
    energy_edges : `astropy.units.Quantity`
        1D array of energy edges defining the energy bins.
    reference_energy : `astropy.units.Quantity`
        Energy at which the normalization is applied, i.e. $E_0$.
    reference_flux : `astropy.units.Quantity`
        Normalization flux for the photon power law.
    break_energy : `astropy.units.Quantity`
        Break energy of the broken power law. The energy bin containing the break energy
        will be a combination of the lower and upper power laws.
    lower_index : `astropy.units.Quantity`
        Lower power law index.
    upper_index : `astropy.units.Quantity`
        Upper power law index.

    Returns
    -------
    `astropy.units.Quantity`
        Photon broken power law, where the flux in each energy bin is equal
        to the broken power law analytically averaged over each bin.
    """

    if energy_edges.size <= 1:
        raise ValueError('Need at least two energy edges.')
    if reference_flux == 0:
        return np.zeros(energy_edges.size - 1) << PHOTON_RATE_UNIT

    up_norm, low_norm = _broken_power_law_normalizations(
        reference_flux, reference_energy, break_energy, lower_index, upper_index
    )

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

    lower_portion = low_integ(energy=lower[1:]) - low_integ(energy=lower[:-1])
    upper_portion = up_integ(energy=upper[1:]) - up_integ(energy=upper[:-1])

    twixt_portion = []
    # bin between the portions is comprised of both power laws
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


@u.quantity_input
def power_law_binned_flux(
    energy_edges: u.keV,
    reference_energy: u.keV,
    reference_flux: PHOTON_RATE_UNIT,
    index: u.one
) -> PHOTON_RATE_UNIT:
    '''Single power law, defined by setting the break energy to -inf and the lower index to nan.

    See docstring of `broken_power_law_binned_flux` for more details.

    Parameters
    ----------
    energy_edges : `astropy.units.Quantity`
        1D array of energy edges defining the energy bins.
    reference_energy : `astropy.units.Quantity`
        Energy at which the normalization is applied, i.e. $E_0$.
    reference_flux : `astropy.units.Quantity`
        Normalization flux for the photon power law.
    index : `astropy.units.Quantity`
        Power law index.

    Returns
    -------
    `astropy.units.Quantity`
        Photon power law, where the flux in each energy bin is equal
        to the power law analytically averaged over each bin.

    '''
    return broken_power_law_binned_flux(
        energy_edges=energy_edges,
        reference_energy=reference_energy,
        reference_flux=reference_flux,
        break_energy=-np.inf << u.keV,
        lower_index=np.nan,
        upper_index=index
    )


@u.quantity_input
def _broken_power_law_normalizations(
    reference_flux: PHOTON_RATE_UNIT,
    reference_energy: u.keV,
    break_energy: u.keV,
    lower_index: u.one,
    upper_index: u.one
) -> PHOTON_RATE_UNIT:
    '''(internal)
    give the correct upper and lower power law normalizations given the
    desired flux and locations of break & norm energies
    '''
    eng_arg = (break_energy / reference_energy).to(u.one)
    if reference_energy < break_energy:
        low_norm = reference_flux
        up_norm = reference_flux * eng_arg**(upper_index - lower_index)
    else:
        up_norm = reference_flux
        low_norm = reference_flux * eng_arg**(lower_index - upper_index)
    return u.Quantity((up_norm, low_norm))
