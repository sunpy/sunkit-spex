import numpy as np

import astropy.units as u
from astropy.wcs import WCS

from sunkit_spex.spectrum import Spectrum


@pytest.fixture
def wcs_et():
    header = {
        "CTYPE1": "TIME    ",  # data type
        "CUNIT1": "min",  # data unit
        "CDELT1": 0.4,  # interval
        "CRPIX1": 0,  # home pixel (units = pixels)
        "CRVAL1": 0,  # home coordinate (units = data unit) eg here the home coordinate is time = 0 min
        "CTYPE2": "ENERGY   ",
        "CUNIT2": "keV",
        "CDELT2": 0.2,
        "CRPIX2": 0,
        "CRVAL2": 0,
        "DATEREF": "2020-01-01T00:00:00",
    }
    return WCS(header=header)


timeenergy = wcs_et()
data = np.arange(1, 11) * u.watt
spec = Spectrum(data, wcs=timeenergy)
