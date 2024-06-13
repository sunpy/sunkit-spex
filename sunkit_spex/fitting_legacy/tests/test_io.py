from unittest.mock import MagicMock, patch

import numpy as np

from sunkit_spex.fitting_legacy.io import (
    _read_arf,
    _read_pha,
    _read_rmf,
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
