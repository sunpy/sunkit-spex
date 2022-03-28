"""
The following code is used to read in instrument spectral data.
"""

from astropy.io import fits
from sunpy.io.special.genx import read_genx

__all__ = ["_read_pha", "_read_arf", "_read_rmf", "_read_rspec_file", "_read_rsrm_file", "_read_sspec_file", "_read_ssrm_file"]


def _read_pha(file):
    """ Takes a .pha file and extracts useful information from it.

    Parameters
    ----------
    file : Str
            String for the .pha file of the spectrum under investigation.

    Returns
    -------
    The channel numbers, counts, and the livetime for the observation.
    """

    with fits.open(file) as hdul:
        data = hdul[1].data
        header_for_livetime = hdul[0].header

    return data['channel'], data['counts'], header_for_livetime['LIVETIME']


def _read_arf(file):
    """ Takes a .arf file and extracts useful information from it.

    Parameters
    ----------
    file : Str
            String for the .arf file of the spectrum under investigation.

    Returns
    -------
    The low and high boundary of energy bins, and the ancillary response [cm^2] (data['specresp']).
    """
    with fits.open(file) as hdul:
        data = hdul[1].data

    return data['energ_lo'], data['energ_hi'], data['specresp']


def _read_rmf(file):
    """ Takes a .rmf file and extracts useful information from it.

    Parameters
    ----------
    file : Str
            String for the .rmf file of the spectrum under investigation.

    Returns
    -------
    The low and high boundary of energy bins (data['energ_lo'], data['energ_hi']), number of sub-set channels in the energy
    bin (data['n_grp']), starting index of each sub-set of channels (data['f_chan']),
    number of channels in each sub-set (data['n_chan']), redistribution matrix [counts per photon] (data['matrix']).
    """

    with fits.open(file) as hdul:
        data = hdul[2].data

    return data['energ_lo'], data['energ_hi'], data['n_grp'], data['f_chan'], data['n_chan'], data['matrix']


def _read_rspec_file(spec_file):
    """ Takes the RHESSI spectral file and extracts useful information from it.

    Parameters
    ----------
    spec_file : str
            String for the RHESSI spectral file under investigation.

    Returns
    -------
    Dictionary of RHESSI information.
    """
    rdict = {}
    with fits.open(spec_file) as hdul:
        for i in range(4):
            rdict[str(i)] = [hdul[i].header, hdul[i].data]
    return rdict

def _read_rsrm_file(srm_file):
    """ Takes the RHESSI SRM spectral file and extracts useful information from it.

    Parameters
    ----------
    srm_file : str
            String for the RHESSI SRM spectral file under investigation.

    Returns
    -------
    Dictionary of RHESSI SRM information.
    """
    srmrdict = {}
    with fits.open(srm_file) as hdul:
        for i in range(4):
            srmrdict[str(i)] = [hdul[i].header, hdul[i].data]
    return srmrdict

def _read_sspec_file(spec_file):
    """ Takes the STIX spectral file and extracts useful information from it.

    Parameters
    ----------
    spec_file : str
            String for the STIX spectral file under investigation.

    Returns
    -------
    Dictionary of STIX information.
    """
    sdict = {}
    with fits.open(spec_file) as hdul:
        for i in range(5):
            sdict[str(i)] = [hdul[i].header, hdul[i].data]
    return sdict

def _read_ssrm_file(srm_file):
    """ Takes the STIX SRM spectral file and extracts useful information from it.

    Parameters
    ----------
    srm_file : str
            String for the STIX SRM spectral file under investigation.

    Returns
    -------
    Dictionary of STIX SRM information (photon bins, count bins, and SRM in units of [counts/keV/photons]).
    """
    contents = read_genx(srm_file)
    return {"photon_energy_bin_edges":contents["DRM"]['E_2D'],"count_energy_bin_edges":contents["DRM"]['EDGES_OUT'],"drm":contents['DRM']['SMATRIX']}
