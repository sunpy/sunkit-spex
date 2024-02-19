"""
The ``io`` module contains code to read instrument specific spectral data.
"""
import numpy as np

from astropy.io import fits

__all__ = ["_read_pha", "_read_arf", "_read_rmf", "_read_rhessi_spec_file", "_read_rhessi_srm_file",
           "_read_stix_spec_file", "_read_stix_srm_file"]


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


def _read_rhessi_spec_file(spec_file):
    """
    Read RHESSI spectral fits file and extract useful information from it.

    Parameters
    ----------
    spec_file :  `str`, `file-like` or `pathlib.Path`
        A RHESSI spectral fits file (see `~astropy.fits.io.open` for details)

    Returns
    -------
    `dict`
        RHESSI spectal data
    """
    rdict = {}
    with fits.open(spec_file) as hdul:
        for i in range(4):
            rdict[str(i)] = [hdul[i].header, hdul[i].data]
    return rdict


def _read_rhessi_srm_file(srm_file):
    """
    Read RHESSI SRM fits file and extract useful information from it.

    Parameters
    ----------
    srm_file : `str`, `file-like` or `pathlib.Path`
        A RHESSI SRM fits file (see `~astropy.fits.io.open` for details)

    Returns
    -------
    `dict`
        RHESSI SRM data
    """
    srmrdict = {}
    with fits.open(srm_file) as hdul:
        for i in range(4):
            srmrdict[str(i)] = [hdul[i].header, hdul[i].data]
    return srmrdict


def _read_stix_spec_file(spec_file):
    """
    Read STIX spectral fits file and extracts useful information from it.

    Parameters
    ----------
    spec_file : `str`, `file-like` or `pathlib.Path`
            STIX spectral fits file (see `~astropy.fits.io.open` for details)

    Returns
    -------
    `dict`
        STIX spectral data.
    """
    sdict = {}
    with fits.open(spec_file) as hdul:
        for i in range(5):
            sdict[str(i)] = [hdul[i].header, hdul[i].data]
    return sdict


def _read_stix_srm_file(srm_file):
    """
    Read a STIX SRM spectral fits file and extract useful information from it.

    Parameters
    ----------
    srm_file : `str` or `pathlib.Path`
        STIX SRM fits file

    Returns
    -------
    `dict`
        STIX SRM data (photon bins, count bins, and SRM in units of [counts/keV/photons]).
    """
    with fits.open(srm_file) as hdul:
        d0 = hdul[1].header
        d1 = hdul[1].data
        d3 = hdul[2].data

    pcb = np.concatenate((d1['ENERG_LO'][:, None], d1['ENERG_HI'][:, None]), axis=1)

    return {"photon_energy_bin_edges": pcb,
            "count_energy_bin_edges": np.concatenate((d3['E_MIN'][:, None], d3['E_MAX'][:, None]), axis=1),
            "drm": d1['MATRIX']*d0["GEOAREA"]}
