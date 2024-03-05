import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import astropy.units as u

from sunkit_spex.spectrum import CountSpectrum, Fitter, SpectrometerResponseMatrix, Spectrum


@pytest.fixture
def spectrum():
    spec = Spectrum(np.arange(1, 11)*u.watt, spectral_axis=np.arange(1, 12)*u.keV)
    return spec


@pytest.fixture
def countspectrum():
    count_spec = CountSpectrum(np.arange(1, 11)*u.ct, spectral_axis=np.arange(1, 12)*u.keV, norm=np.ones(10)*u.s)
    return count_spec


@pytest.fixture
def perfect_srm():
    srm = SpectrometerResponseMatrix(np.eye(10, 10)*u.count/u.photon)
    return srm


@pytest.fixture()
def diag_edep_srm():
    srm = SpectrometerResponseMatrix(np.diag(1/np.arange(1, 11))*u.count/u.photon)
    return srm


def test_srm(perfect_srm):
    data_in = np.arange(1, 11) * u.photon
    data_out = perfect_srm.forward(data_in)
    assert_array_equal(data_out, np.arange(1, 11)*u.count)


def test_fitter(spectrum):
    def model(x, slope: u.watt/u.keV, intercept: u.watt):
        return x*slope+intercept

    fitter = Fitter(spectrum, model)
    result = fitter.fit()

    assert_allclose(1, result.slope)
    assert_allclose(-0.5, result.intercept)


def test_fitter_response_perfect(countspectrum, perfect_srm):
    def model(x, slope: u.ph/u.keV, intercept: u.ph):
        return x*slope+intercept

    fitter = Fitter(countspectrum, model, perfect_srm)
    result = fitter.fit()

    assert_allclose(1, result.slope)
    assert_allclose(-0.5, result.intercept)


def test_fitter_response_edep(countspectrum, diag_edep_srm):
    countspectrum.data[:] = countspectrum.data @ diag_edep_srm.matrix.value * u.ct

    def model(x, slope: u.ph/u.keV, intercept: u.ph):
        return x*slope+intercept

    fitter = Fitter(countspectrum, model, diag_edep_srm)
    result = fitter.fit()

    assert_allclose(1, result.slope)
    assert_allclose(-0.5, result.intercept)
