# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure.


# Uncomment the following line to treat all DeprecationWarnings as
# exceptions. For Astropy v2.0 or later, there are 2 additional keywords,
# as follow (although default should work for most cases).
# To ignore some packages that produce deprecation warnings on import
# (in addition to 'compiler', 'scipy', 'pygments', 'ipykernel', and
# 'setuptools'), add:
# modules_to_ignore_on_import=['module_1', 'module_2']
# To ignore some specific deprecation warning messages for Python version
# MAJOR.MINOR or later, add:
# warnings_to_ignore_by_pyver={(MAJOR, MINOR): ['Message to ignore']}
# enable_deprecations_as_exceptions()

# Uncomment and customize the following lines to add/remove entries from
# the list of packages for which version numbers are displayed when running
# the tests. Making it pass for KeyError is essential in some cases when
# the package uses other astropy affiliated packages.
# try:
#     PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
#     PYTEST_HEADER_MODULES['scikit-image'] = 'skimage'
#     del PYTEST_HEADER_MODULES['h5py']
# except (NameError, KeyError):  # NameError is needed to support Astropy < 1.0
#     pass

# Uncomment the following lines to display the version number of the
# package rather than the version number of Astropy in the top line when
# running the tests.
# import os
#
# This is to figure out the package version, rather than
# using Astropy's
# try:
#     from .version import version
# except ImportError:
#     version = 'dev'
#
# try:
#     packagename = os.path.basename(os.path.dirname(__file__))
#     TESTED_VERSIONS[packagename] = version
# except NameError:   # Needed to support Astropy <= 1.0.0
#     pass

import numpy as np
import pytest

import astropy.units as u
from astropy.wcs import WCS

from sunkit_spex.spectrum import Spectrum


@pytest.fixture
def wcs_et():
    header = {
        "CTYPE1": "TIME    ",
        "CUNIT1": "min",
        "CDELT1": 0.4,
        "CRPIX1": 0,
        "CRVAL1": 0,
        "CTYPE2": "ENERGY   ",
        "CUNIT2": "keV",
        "CDELT2": 0.2,
        "CRPIX2": 0,
        "CRVAL2": 0,
        "DATEREF": "2020-01-01T00:00:00",
    }
    return WCS(header=header)


@pytest.fixture
def spec(wcs_et):
    timeenergy = wcs_et
    data = np.arange(1, 11) * u.watt
    return Spectrum(data, wcs=timeenergy)


@pytest.fixture
def wcs_dt():
    header = {
        "CTYPE1": "DETECTOR",
        "CUNIT1": "",
        "CDELT1": 0,
        "CRPIX1": 0,
        "CRVAL1": 0,
        "CTYPE2": "ENERGY   ",
        "CUNIT2": "keV",
        "CDELT2": 0.2,
        "CRPIX2": 0,
        "CRVAL2": 0,
        "DATEREF": "2020-01-01T00:00:00",
    }
    return WCS(header=header)


@pytest.fixture
def dve():
    detectorenergy = wcs_dt()
    data = np.arange(1, 11) * u.watt
    return Spectrum(data, wcs=detectorenergy)


@pytest.fixture
def wcs_ess():
    header = {
        "CTYPE1": "ENERGY    ",
        "CUNIT1": "keV",
        "CDELT1": 0.2,
        "CRPIX1": 0,
        "CRVAL1": 0,
        "CTYPE2": "HPLT-TAN",
        "CUNIT2": "deg",
        "CDELT2": 0.5,
        "CRPIX2": 2,
        "CRVAL2": 0.5,
        "CTYPE3": "HPLN-TAN",
        "CUNIT3": "deg",
        "CDELT3": 0.4,
        "CRPIX3": 2,
        "CRVAL3": 1,
        "DATEREF": "2020-01-01T00:00:00",
    }
    return WCS(header=header)


@pytest.fixture
def evsvs():
    energyspacespace = wcs_ess()
    data = np.arange(1, 11) * u.watt
    return Spectrum(data, wcs=energyspacespace)


@pytest.fixture
def wcs_etd():
    header = {
        "CTYPE1": "ENERGY    ",
        "CUNIT1": "keV",
        "CDELT1": 0.2,
        "CRPIX1": 0,
        "CRVAL1": 0,
        "CTYPE2": "TIME",
        "CUNIT2": "min",
        "CDELT2": 0.4,
        "CRPIX2": 0,
        "CRVAL2": 0,
        "CTYPE3": "DETECTOR",
        "CUNIT3": "",
        "CDELT3": 0,
        "CRPIX3": 0,
        "CRVAL3": 0,
        "DATEREF": "2020-01-01T00:00:00",
    }
    return WCS(header=header)


@pytest.fixture
def evtvd():
    energytimedetector = wcs_etd()
    data = np.arange(1, 11) * u.watt
    return Spectrum(data, wcs=energytimedetector)


@pytest.fixture
def wcs_esstp():
    header = {
        "CTYPE1": "ENERGY    ",
        "CUNIT1": "keV",
        "CDELT1": 0.2,
        "CRPIX1": 0,
        "CRVAL1": 0,
        "CTYPE2": "HPLT-TAN",
        "CUNIT2": "deg",
        "CDELT2": 0.5,
        "CRPIX2": 2,
        "CRVAL2": 0.5,
        "CTYPE3": "HPLN-TAN",
        "CUNIT3": "deg",
        "CDELT3": 0.4,
        "CRPIX3": 2,
        "CRVAL3": 1,
        "CTYPE4": "TIME",
        "CUNIT4": "min",
        "CDELT4": 0.4,
        "CRPIX4": 0,
        "CRVAL4": 0,
        "CTYPE5": "POLARISATION",
        "CUNIT5": "C/m2",
        "CDELT5": 0.4,
        "CRPIX5": 0,
        "CRVAL5": 0,
        "DATEREF": "2020-01-01T00:00:00",
    }
    return WCS(header=header)


@pytest.fixture
def evsvsvtvp():
    esstp = wcs_esstp()
    data = np.arange(1, 11) * u.watt
    return Spectrum(data, wcs=esstp)
