import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import readsav

import astropy.units as u
from astropy.units import Quantity

from sunpy.data import cache


@u.quantity_input
def albedo(spec, energy: Quantity[u.keV], theta: Quantity[u.deg, 0, 90], anisotropy=1):
    r"""
    Add albedo correction to input spectrum

    Correct input model spectrum for the component reflected by the solar atmosphere following [Kontar20006]_ using
    precomputed green matrices from SSW.

    .. [Kontar20006] https://doi.org/10.1051/0004-6361:20053672

    Parameters
    ----------
    spec :
        Input spectrum
    energy :
        Energy edges associated with the spectrum
    theta :
        Angle between Sun-observer line and X-ray source
    anisotropy :
        Ratio of the flux in observer direction to the flux downwards, anisotropy=1 (default) the source is isotropic

    Example
    -------
    >>> import astropy.units as u
    >>> import numpy as np
    >>> from sunkit_spex.models.physical import albedo
    >>> e = np.linspace(5,  500, 1000)
    >>> e_c = e[:-1] + np.diff(e)
    >>> s = 125*e_c**-3
    >>> corrected = albedo(s, e, theta=45*u.deg)
    """
    base_url = "https://soho.nascom.nasa.gov/solarsoft/packages/xray/dbase/albedo/"
    mu = np.cos(theta)

    # what about 0 and 1 assume so close to 05 and 95 that it doesn't matter
    # load precomputed green matrices
    if 0.5 <= mu <= 0.95:
        low = 5 * np.floor(mu * 20)
        high = 5 * np.ceil(mu * 20)
        low_name = f"green_compton_mu{low:03.0f}.dat"
        high_name = f"green_compton_mu{high:03.0f}.dat"
        low_file = cache.download(base_url + low_name)
        high_file = cache.download(base_url + high_name)
        green = readsav(low_file)
        albedo_low = green["p"].albedo[0]
        green_high = readsav(high_file)
        albedo_high = green_high["p"].albedo[0]
        # why 20?
        albedo = albedo_low + (albedo_high - albedo_low) * (mu - (np.floor(mu * 20)) / 20)

    elif mu < 0.5:
        file = "green_compton_mu005.dat"
        file = cache.download(base_url + file)
        green = readsav(file)
        albedo = green["p"].albedo[0]
    elif mu > 0.95:
        file = "green_compton_mu095.dat"
        file = cache.download(base_url + file)
        green = readsav(file)
        albedo = green["p"].albedo[0]

    albedo = albedo.T
    energy_grid_edges = green["p"].edges[0] * u.keV
    energy_grid_centers = energy_grid_edges[:, 0] + (np.diff(energy_grid_edges, axis=1) / 2).reshape(-1)

    interp = RegularGridInterpolator((energy_grid_centers.to_value(u.keV), energy_grid_centers.to_value(u.keV)), albedo)

    de = np.diff(energy)
    energy_centers = energy[:-1] + de / 2

    X, Y = np.meshgrid(energy_centers.to_value(u.keV), energy_centers.to_value(u.keV))
    albedo_interp = interp((X, Y))

    albedo_interp = albedo_interp * de.value / anisotropy

    return spec + spec @ albedo_interp.T
