"""
The following code is used to generate matrices for albedo correction
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import readsav

from sunpy.data import cache
from astropy.units import Quantity
from functools import lru_cache

from numpy.typing import NDArray
import astropy.units as u

@lru_cache
def _get_green_matrix(theta: float) -> RegularGridInterpolator:
    r"""
    Get greens matrix for given angle.

    Interpolates pre-computed green matrices for fixed angles. The resulting greens matrix is then loaded into an
    interpolator for later energy interpolation.

    Parameters
    ==========
    theta : float
        Angle between the observer and the source

    Returns
    =======
        Greens matrix interpolator
    """
    mu = np.cos(theta)

    base_url = "https://soho.nascom.nasa.gov/solarsoft/packages/xray/dbase/albedo/"
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

    # By construction in keV
    energy_grid_edges = green["p"].edges[0]
    energy_grid_centers = energy_grid_edges[:, 0] + (np.diff(energy_grid_edges, axis=1) / 2).reshape(-1)

    return RegularGridInterpolator((energy_grid_centers, energy_grid_centers), albedo)


@lru_cache
def _calculate_albedo_matrix(energy_edges: tuple[float], theta: float, anisotropy: float) -> NDArray:
    r"""
    Calculate green matrix for given energies and angle.

    Interpolates precomputed green matrices for given energies and angle.

    Parameters
    ==========
    energy_edges :
        Energy edges associated with the spectrum
    theta :
        Angle between the observer and the source
    anisotropy :
        Ratio of the flux in observer direction to the flux downwards, 1 for an isotropic source
    """
    albedo_interpolator = _get_green_matrix(theta)
    de = np.diff(energy_edges)
    energy_centers = np.mean(energy_edges, axis=1)

    X, Y = np.meshgrid(energy_centers, energy_centers)

    albedo_interp = albedo_interpolator((X, Y))

    # Scale by anisotropy
    albedo_interp = (albedo_interp * de) / anisotropy

    # Take a transpose
    return albedo_interp.T


@u.quantity_input
def get_albedo_matrix(energy_edges: Quantity[u.keV], theta: Quantity[u.deg], anisotropy=1):
    r"""
    Get albedo correction matrix.

    Matrix used to correct a photon spectrum for the component reflected by the solar atmosphere following interpolated
    to given angle and energy indices.

    Parameters
    ----------
    energy_edges :
        Energy edges associated with the spectrum
    theta :
        Angle between Sun-observer line and X-ray source
    anisotropy :
        Ratio of the flux in observer direction to the flux downwards, 1 for an isotropic source

    Example
    -------
    >>> import astropy.units as u
    >>> import numpy as np
    >>> from sunkit_spex.models.physical.albedo import get_albedo_matrix
    >>> e = np.linspace(5,  500, 5)*u.keV
    >>> albedo_matrix = get_albedo_matrix(e,theta=45*u.deg)
    >>> albedo_matrix
    array([[3.80274484e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [5.10487362e-01, 8.35309813e-07, 0.00000000e+00, 0.00000000e+00],
    [3.61059918e-01, 2.48711099e-01, 2.50744411e-09, 0.00000000e+00],
    [3.09323903e-01, 2.66485260e-01, 1.23563372e-01, 1.81846722e-10]])
    """
    # if energy_edges[0].to_value(u.keV) < 3 or energy_edges[-1].to_value(u.keV) > 600:
    #     raise ValueError("Supported energy range 3 <= E <= 600 keV")
    # theta = np.array(theta).squeeze() << theta.unit
    # if np.abs(theta) > 90 * u.deg:
    #     raise ValueError(f"Theta must be between -90 and 90 degrees: {theta}.")
    anisotropy = np.array(anisotropy).squeeze()

    energy_edges = energy_edges.to_value(u.keV)
    energy_edges = [tuple(l) for l in energy_edges]

    return _calculate_albedo_matrix(tuple(energy_edges), theta.to_value(u.deg), anisotropy.item())