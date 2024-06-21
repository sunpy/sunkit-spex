import numpy as np
from numpy.testing import assert_array_equal

from astropy.nddata import NDDataRef

from sunkit_spex.spectrum.uncertainty import PoissonUncertainty


def test_add():
    data = np.array([0, 1, 2])
    uncert = np.sqrt(data)
    a = NDDataRef(data, uncertainty=PoissonUncertainty(uncert))
    b = NDDataRef(data.copy(), uncertainty=PoissonUncertainty(uncert.copy()))
    aplusb = a.add(b)
    assert_array_equal(aplusb.data, 2 * data)
    assert_array_equal(aplusb.uncertainty.array, np.sqrt(2 * uncert**2))


def test_subtract():
    data = np.array([0, 1, 2])
    uncert = np.sqrt(data)
    a = NDDataRef(data, uncertainty=PoissonUncertainty(uncert))
    b = NDDataRef(data.copy(), uncertainty=PoissonUncertainty(uncert.copy()))
    aminusb = a.subtract(b)
    assert_array_equal(aminusb.data, data - data)
    assert_array_equal(aminusb.uncertainty.array, np.sqrt(2 * uncert**2))


def test_multiply():
    data = np.array([0, 1, 2])
    uncert = np.sqrt(data)
    a = NDDataRef(data, uncertainty=PoissonUncertainty(uncert))
    b = NDDataRef(data.copy(), uncertainty=PoissonUncertainty(uncert.copy()))
    atimesb = a.multiply(b)
    assert_array_equal(atimesb.data, data**2)
    assert_array_equal(atimesb.uncertainty.array, np.sqrt(2 * data**2 * uncert**2))  # (b**2*da**2 + a**2db**2)**0.5


def test_divide():
    data = np.array([0, 1, 2])
    uncert = np.sqrt(data)
    a = NDDataRef(data, uncertainty=PoissonUncertainty(uncert))
    b = NDDataRef(data.copy(), uncertainty=PoissonUncertainty(uncert.copy()))
    adivb = a.divide(b)
    assert_array_equal(adivb.data, data / data)
    assert_array_equal(
        adivb.uncertainty.array, np.sqrt(((1 / data) ** 2 * uncert**2) * 2)
    )  # ((1/b)**2*da**2 + (a/b**2)**2db**2)**0.5
