import numpy as np

import astropy.units as u

from sunkit_spex.models.physical.albedo import albedo


def test_albedo():
    e = np.linspace(4, 600, 597)
    e_c = e[:-1] + np.diff(e)
    s = 125 * e_c**-3
    out = albedo(s, e * u.keV, 45 * u.deg)
    print(out)
