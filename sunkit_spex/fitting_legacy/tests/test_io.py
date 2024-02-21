from unittest.mock import MagicMock, patch

import numpy as np

from sunkit_spex.fitting_legacy.io import (
    _read_arf,
    _read_pha,
    _read_rhessi_spec_file,
    _read_rhessi_srm_file,
    _read_rmf,
    _read_stix_spec_file,
    _read_stix_srm_file,
)


@patch('astropy.io.fits.open')
def test_read_pha(mock_open):
    channel = np.array([0, 1, 2])
    counts = np.array([10, 20, 30])
    livetime = 1.0
    hdul = MagicMock()
    hdul[1].data = {'channel': np.array([0, 1, 2]), 'counts': np.array([10, 20, 30])}
    hdul[0].header = {'LIVETIME': 1.0}
    mock_open.return_value.__enter__.return_value = hdul
    res = _read_pha('test.pha')
    for t, r in zip((channel, counts, livetime), res):
        assert np.all(t == r)


@patch('astropy.io.fits.open')
def test_read_arf(mock_open):
    energ_lo = 1
    energ_hi = 10
    specresp = np.eye(2)
    hdul = MagicMock()
    hdul[1].data = {'energ_lo': energ_lo, 'energ_hi': energ_hi, 'specresp': specresp}
    mock_open.return_value.__enter__.return_value = hdul
    res = _read_arf('test.pha')
    assert res == (energ_lo, energ_hi, specresp)


@patch('astropy.io.fits.open')
def test_read_rmf(mock_open):
    energ_lo = 1
    energ_hi = 10
    n_grp = 1
    f_chan = 1
    n_chan = 1
    matrix = np.eye(2)
    hdul = MagicMock()
    hdul[1].data = {'energ_lo': energ_lo, 'energ_hi': energ_hi, 'n_grp': n_grp, 'f_chan': f_chan,
                    'n_chan': n_chan, 'matrix': matrix}
    hdul[0].header = {'LIVETIME': 1.0}
    mock_open.return_value.__enter__.return_value = hdul
    res = _read_rmf('test.pha')
    assert res == (energ_lo, energ_hi, n_grp, f_chan, n_chan, matrix)


@patch('astropy.io.fits.open')
def test_read_rhessi_spec_file(mock_open):
    hdul = []
    for i in range(4):
        m = MagicMock()
        m.data = {i: np.arange(i)}
        m.header = {'ext': i}
        hdul.append(m)
    mock_open.return_value.__enter__.return_value = hdul
    res = _read_rhessi_spec_file('test.fits')
    for k, v in res.items():
        k = int(k)
        assert v[0]['ext'] == k
        assert np.array_equal(v[1][k], np.arange(k))


@patch('astropy.io.fits.open')
def test_read_rhessi_srm_file(mock_open):
    hdul = []
    for i in range(4):
        m = MagicMock()
        m.data = {i: np.arange(i)}
        m.header = {'ext': i}
        hdul.append(m)
    mock_open.return_value.__enter__.return_value = hdul
    res = _read_rhessi_srm_file('test.fits')
    for k, v in res.items():
        k = int(k)
        assert v[0]['ext'] == k
        assert np.array_equal(v[1][k], np.arange(k))


@patch('astropy.io.fits.open')
def test_read_stix_spec_file(mock_open):
    hdul = []
    for i in range(5):
        m = MagicMock()
        m.data = {i: np.arange(i)}
        m.header = {'ext': i}
        hdul.append(m)
    mock_open.return_value.__enter__.return_value = hdul
    res = _read_stix_spec_file('test.fits')
    for k, v in res.items():
        k = int(k)
        assert v[0]['ext'] == k
        assert np.array_equal(v[1][k], np.arange(k))


@patch('astropy.io.fits.open')
def test_read_stix_srm_file(mock_open):

    photon_bins = np.array([[1, 2], [2, 3]])
    count_bins = np.array([[2, 3], [3, 4]])
    drm = 2*np.eye(2)

    hdul = []
    headers = [{}, {'GEOAREA': 2}, {}]
    data = [{}, {'MATRIX': np.eye(2), 'ENERG_LO': np.array([1, 2]), 'ENERG_HI': np.array([2, 3])}, {'E_MIN': np.array([2, 3]), 'E_MAX': np.array([3, 4])}]

    for i in range(len(headers)):
        m = MagicMock()
        m.header = headers[i]
        m.data = data[i]
        hdul.append(m)

    mock_open.return_value.__enter__.return_value = hdul
    res = _read_stix_srm_file('test.fits')

    assert np.array_equal(res['photon_energy_bin_edges'], photon_bins)
    assert np.array_equal(res['count_energy_bin_edges'], count_bins)
    assert np.array_equal(res['drm'], drm)
