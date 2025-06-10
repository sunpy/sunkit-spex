from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.io import readsav

import astropy.units as u
from astropy.modeling import FittableModel, Parameter
from astropy.units import Quantity

from sunpy.data import cache

__all__ = ["Albedo", "get_albedo_matrix"]


class Albedo(FittableModel):
    r"""
    Aldedo model which adds albdeo correction to input spectrum.

    Following [Kontar2006]_ using precomputed green matrices distributed as part of [SSW]_.

    .. [Kontar2006] https://doi.org/10.1051/0004-6361:20053672
    .. [SSW] https://www.lmsal.com/solarsoft/

    Parameters
    ==========
    energy_edges :
        Energy edges associated with input spectrum
    theta :
        Angle between Sun-observer line and X-ray source
    anisotropy :
        Ratio of the flux in observer direction to the flux downwards, 1 for an isotropic source

    Examples
    ========
    .. plot::
        :include-source:

        import astropy.units as u
        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.powerlaws import PowerLaw1D
        from astropy.visualization import quantity_support

        from sunkit_spex.models.physical.albedo import Albedo

        e_edges = np.linspace(5, 550, 600) * u.keV
        e_centers = e_edges[0:-1] + (0.5 * np.diff(e_edges))
        source = PowerLaw1D(amplitude=1*u.ph/(u.cm*u.s), x_0=5*u.keV, alpha=3)
        albedo = Albedo(energy_edges=e_edges)
        observed = source | albedo

        with quantity_support():
            plt.figure()
            plt.plot(e_centers,  source(e_centers), 'k', label='Source')
            for i, t in enumerate([0, 45, 90]*u.deg):
                albedo.theta = t
                plt.plot(e_centers,  observed(e_centers), '--', label=f'Observed, theta={t}', color=f'C{i+1}')
                plt.plot(e_centers,  observed(e_centers) - source(e_centers), ':',
                         label=f'Reflected, theta={t}', color=f'C{i+1}')

            plt.ylim(1e-6,  1)
            plt.xlim(5, 550)
            plt.loglog()
            plt.legend()
            plt.show()

    """

    n_inputs = 1
    n_outputs = 1
    theta = Parameter(
        name="theta",
        default=0,
        unit=u.deg,
        min=-90,
        max=90,
        description="Angle between the observer and the source",
        fixed=False,
    )
    anisotropy = Parameter(
        name="anisotropy", default=1, description="The anisotropy used for albedo correction", fixed=True
    )

    name = "Albedo"

    _input_units_allow_dimensionless = True

    def __init__(self, *args, **kwargs):
        self.energy_edges = kwargs.pop("energy_edges")

        super().__init__(*args, **kwargs)

    def evaluate(self, spectrum, theta, anisotropy):
        if not isinstance(theta, Quantity):
            theta = theta * u.deg

        albedo_matrix = get_albedo_matrix(self.energy_edges, theta, anisotropy)

        return spectrum + spectrum @ albedo_matrix

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {"theta": u.deg}


@lru_cache
def _get_green_matrix(theta: float) -> RegularGridInterpolator:
    r"""
    Get greens matrix for given angle.

    Interpolates pre-computed green matrices for fixed angles. The resulting greens matrix is then loaded into an
    interpolator for later energy interpolation.

    Parameters
    ==========
    theta : float
        Angle in degrees between the observer and the source

    Returns
    =======
        Greens matrix interpolator
    """
    mu = np.cos(np.deg2rad(theta))

    base_url = "https://soho.nascom.nasa.gov/solarsoft/packages/xray/dbase/albedo/"
    # what about 0 and 1 assume so close to 05 and 95 that it doesn't matter
    # load precomputed green matrices
    if 0.05 <= mu <= 0.95:
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
        # There are 20 files from 005 to 095 in steps of 005
        albedo = albedo_low + (albedo_high - albedo_low) * (mu - (np.floor(mu * 20)) / 20)

    elif mu < 0.05:
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
    energy_centers = energy_edges[:-1] + de / 2

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
    array([[7.64944936e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [7.17787454e-01, 1.54795970e-10, 0.00000000e+00, 0.00000000e+00],
           [5.22059171e-01, 3.02951100e-01, 1.46291699e-13, 0.00000000e+00],
           [4.52582540e-01, 3.69821128e-01, 1.13435321e-01, 5.95953019e-15]])
    """
    if energy_edges[0].to_value(u.keV) < 3 or energy_edges[-1].to_value(u.keV) > 600:
        raise ValueError("Supported energy range 3 <= E <= 600 keV")
    theta = np.array(theta).squeeze() << theta.unit
    if np.abs(theta) > 90 * u.deg:
        raise ValueError(f"Theta must be between -90 and 90 degrees: {theta}.")
    anisotropy = np.array(anisotropy).squeeze()

    return _calculate_albedo_matrix(tuple(energy_edges.to_value(u.keV)), theta.to_value(u.deg), anisotropy.item())
