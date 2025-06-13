"""
The following code is used to generate matrices for albedo correction
in the legacy portion of the package.
"""

import numpy as np

import astropy.units as u

from sunkit_spex.models.physical.albedo import get_albedo_matrix as _get_albedo_matrix


@u.quantity_input
def get_albedo_matrix(energy_edges: u.keV, theta: u.deg, anisotropy=1):
    r"""
    Get an albedo correction matrix.

    Wraps `~sunkit_spex.models.physical.albedo.get_albedo_matrix`,
    adding support for 2D energy edges.

    Parameters
    ----------
    energy_edges :
        Energy edges associated with the spectrum (2D array)
    theta :
        Angle between Sun-observer line and X-ray source
    anisotropy :
        Ratio of the flux in observer direction to the flux downwards, 1 for an isotropic source
    """
    # The physical model Albedo matrix function expects 1D
    # energy edges, so flatten the 2D edges from the legacy side,
    # and just call the other function.
    energy_edges = energy_edges.to_value(u.keV)
    flat_edges = np.concatenate((energy_edges[:, 0], [energy_edges[-1, -1]])) << u.keV
    return _get_albedo_matrix(flat_edges, theta, anisotropy)
