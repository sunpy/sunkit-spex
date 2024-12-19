import numpy as np

from sunkit_spex.models.physical.albedo import albedo


def test_albedo():
    e = np.linspace(4, 600, 597)
    e_c = e[:-1] + np.diff(e)
    s = 125 * e_c**-3
    # e = e *u.keV
    theta = 0.2  # u.rad
    albedo_matrix = albedo(e, theta=theta)
    out = s + s @ albedo_matrix
    print(out)
