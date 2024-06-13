"""
The ``io`` module contains code to read instrument specific spectral data.
"""

from astropy.io import fits

__all__ = ["_read_pha", "_read_arf", "_read_rmf"]


def _read_pha(file):
    """
    Read a .pha file and extract useful information from it.

    Parameters
    ----------
    file : `str`, `file-like` or `pathlib.Path`
        A .pha file (see `~astropy.fits.io.open` for details).

    Returns
    -------
    `tuple`
        The channel numbers, counts, and the livetime for the observation.
    """
    with fits.open(file) as hdul:
        data = hdul[1].data
        header_for_livetime = hdul[0].header

    return data['channel'], data['counts'], header_for_livetime['LIVETIME']


def _read_arf(file):
    """
    Read a .arf file and extract useful information from it.

    Parameters
    ----------
    file :  `str`, `file-like` or `pathlib.Path`
        A .arf file (see `~astropy.fits.io.open` for details ).

    Returns
    -------
    `tuple`
        The low and high boundary of energy bins, and the ancillary response [cm^2] (data['specresp']).
    """
    with fits.open(file) as hdul:
        data = hdul[1].data

    return data['energ_lo'], data['energ_hi'], data['specresp']


def _read_rmf(file):
    """
    Read a .rmf file and extract useful information from it.

    Parameters
    ----------
    file :  `str`, `file-like` or `pathlib.Path`
        A .rmf file (see `~astropy.fits.io.open` for details).

    Returns
    -------
    `tuple`
        The low and high boundary of energy bins (data['energ_lo'], data['energ_hi']), number of sub-set channels in the energy
        bin (data['n_grp']), starting index of each sub-set of channels (data['f_chan']),
        number of channels in each sub-set (data['n_chan']), redistribution matrix [counts per photon] (data['matrix']).
    """

    with fits.open(file) as hdul:
        data = hdul[2].data

    return data['energ_lo'], data['energ_hi'], data['n_grp'], data['f_chan'], data['n_chan'], data['matrix']
