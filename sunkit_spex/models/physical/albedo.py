import warnings
import functools

from functools import lru_cache

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import readsav

# from astropy.units import Quantity
from astropy.modeling import Fittable1DModel, Parameter

from sunpy.data import cache

__all__ = ["AlbedoModel"]


class AlbedoModel(Fittable1DModel):
    def __init__(self, energy, anisotropy, theta):
        self.energy = energy
        self.anisotropy = Parameter(
            default=anisotropy, description="The anisotropy used for albedo correction", fixed=True
        )
        self.theta = Parameter(
            default=theta, description="Angle between the flare and the telescope in radians", fixed=True
        )
        super().__init__()

    def evaluate(self, count_model):
        """Corrects the composite count model for albedo. To be used as: ct_model = (ph_model|srm)|albedo"""
        albedo_matrix_T = albedo(self.energy, self.theta.value, self.anisotropy.value)

        return count_model + count_model @ albedo_matrix_T


# Wrapper for chaching function with numpy array args
def np_cache(function):
    @functools.cache
    def cached_wrapper(*args, **kwargs):
        args = [np.array(a) if isinstance(a, tuple) else a for a in args]
        kwargs = {k: np.array(v) if isinstance(v, tuple) else v for k, v in kwargs.items()}

        return function(*args, **kwargs)

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        args = [tuple(a) if isinstance(a, np.ndarray) else a for a in args]
        kwargs = {k: tuple(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
        return cached_wrapper(*args, **kwargs)

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


@functools.cache
def load_green_matrices(theta):
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

    # energy_grid_edges = green["p"].edges[0] * u.keV
    energy_grid_edges = green["p"].edges[0]

    energy_grid_centers = energy_grid_edges[:, 0] + (np.diff(energy_grid_edges, axis=1) / 2).reshape(-1)

    # interp_green_matrix = RegularGridInterpolator((energy_grid_centers.to_value(u.keV), energy_grid_centers.to_value(u.keV)), albedo)
    interp_green_matrix = RegularGridInterpolator((energy_grid_centers, energy_grid_centers), albedo)

    return interp_green_matrix


@np_cache
def calc_albedo_matrix(energy, interp_green_matrix, anisotropy):
    # Re-introduce units
    # energy = energy * u.keV
    de = np.diff(energy)
    energy_centers = energy[:-1] + de / 2

    # X, Y = np.meshgrid(energy_centers.to_value(u.keV), energy_centers.to_value(u.keV))
    X, Y = np.meshgrid(energy_centers, energy_centers)

    albedo_interp = interp_green_matrix((X, Y))

    # Scale by anisotropy
    # albedo_interp = albedo_interp * de.value / anisotropy
    albedo_interp = albedo_interp * de / anisotropy

    # Take a transpose
    albedo_interp_T = albedo_interp.T

    return albedo_interp_T


# @u.quantity_input
# def albedo(energy: Quantity[u.keV], theta: Quantity[u.deg, 0, 90], anisotropy=1):
def albedo(energy, theta, anisotropy=1):
    r"""
    Add albedo correction to input spectrum

    Correct input model spectrum for the component reflected by the solar atmosphere following [Kontar20006]_ using
    precomputed green matrices from SSW.

    .. [Kontar2006] https://doi.org/10.1051/0004-6361:20053672

    Parameters
    ----------
    spec :
        Input count spectrum
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
    >>> from sunkit_spex.models.physical.albedo import albedo
    >>> e = np.linspace(5,  500, 1000)
    >>> e_c = e[:-1] + np.diff(e)
    >>> s = 125*e_c**-3
    >>> albedo_matrix = albedo(e, theta=0.2)
    >>> corrected = s + s@albedo_matrix
    """

    # Add check for energy range restricted by the Green matrices
    if energy[0] < 3 or energy[-1] > 600:
        warnings.warn("\nCount energies are required to be >= 3keV and <=600 keV ")

    # Green matrix for a given theta
    interp_green = load_green_matrices(theta)

    # Transpose of a matrix used for albedo correction, need to strip energy of units otherwise problem with caching
    # albedo_matrix_T = calc_albedo_matrix(energy.value, interp_green, anisotropy)
    albedo_matrix_T = calc_albedo_matrix(energy, interp_green, anisotropy)

    return albedo_matrix_T
