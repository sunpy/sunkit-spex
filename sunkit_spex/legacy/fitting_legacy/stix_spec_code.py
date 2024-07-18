"""
The following code is used to make SRM/counts data in consistent units from STIX spectral data.
"""

import numpy as np

from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.io import fits


__all__ = ["_get_spec_file_info", "_spec_file_units_check", "_get_srm_file_info"]


def _get_spec_file_info(spec_file):
    """
    Read STIX spectrogram fits file and extracts useful information from it.

    IMPORTANT NOTE: This reader assumes the file was generated from stx_convert_spectrogram IDL routine. This shouldn't be used with science data x-ray compaction level 4 (spectrogram) files.
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
        for i in range(len(hdul)):
                sdict[str(i)] = [hdul[i].header, hdul[i].data]

    #All useful spectra info is in index "2", so get the energy bins
    channel_bins = np.concatenate((sdict["2"][1]['E_MIN'][:, None], sdict["2"][1]['E_MAX'][:, None]), axis=1)

    #Create channel bin arrays
    channel_bins_inds=np.empty(len(channel_bins))
    channel_bins_inds.fill(True)

    #Get TIME_DEL as we only get counts/s but also want values of counts
    times_mids = sdict["1"][1]["TIME"]
    time_del = sdict["1"][1]["TIMEDEL"]

    time_diff_so2e = sdict["1"][0]["TIME_SHI"]  # time difference between Sun2Earth and Sun2SO

    # if odd `t` in seconds then edges are [midt-floor(bin_width/2), midt+ceil(bin_width/2)]
    # e.g., mid_t=19, del_t=13 then edges would be [19-6, 19+7]=[13,26]
    _minus_half_bin_width = (time_del)/2
    t_lo = times_mids - _minus_half_bin_width

    _plus_half_bin_width = time_del/2
    t_hi = times_mids + _plus_half_bin_width

    #Calculating spectral times absTime[i] = mjd2any(MJDREF + TIMEZERO) + TIME[i]
    # So the time of first observation:
    timemjd = sdict["1"][0]["MJDREF"] + sdict["1"][0]["TIMEZERO"]
    date_beg = Time(timemjd, format='mjd', scale='utc')
    date_beg.format = 'isot'


    spec_stimes = [date_beg +TimeDelta(dt * u.s) for dt in t_lo]

    spec_etimes = [date_beg+TimeDelta(dt * u.s) for dt in t_hi]

    time_bins = np.concatenate((np.array(spec_stimes)[:, None], np.array(spec_etimes)[:, None]), axis=1)

    #Getting livetime
    lvt = sdict["1"][1]["LIVETIME"]

    #Getting count rates and errors
    cts_rates, cts_rate_err = _spec_file_units_check(sdict)
    cts_rates[np.nonzero(cts_rates < 0)] = 0

    #Calculating counts
    counts_err = cts_rate_err * lvt[:, None] * time_del[:, None]
    counts = cts_rates * lvt[:, None] * time_del[:, None]

    # Adding the 3% above 10 keV, 5% bellow 10keV  or 7% systematic errors bellow 7keV - matching OSPEX
    #Getting the array with percentages coresponding to each energy bin
    energy_bin_low = channel_bins[:,0]
    energy_conditions = [energy_bin_low < 7, (energy_bin_low < 10) & (energy_bin_low >= 7), energy_bin_low >= 10]
    percentage = [0.07, 0.05, 0.03]

    systematic_err_percentage = np.select(energy_conditions, percentage)

    #Calculating systematic error
    systematic_err = (systematic_err_percentage * counts)

    #Adding the two errors in quadrature
    counts_err = np.sqrt(counts_err**2 + systematic_err**2)

    #Count rate error
    cts_rate_err = counts_err / lvt[:, None] / time_del[:, None]

    return channel_bins, channel_bins_inds, time_bins, lvt, counts, counts_err, cts_rates, cts_rate_err


def _spec_file_units_check(stix_dict):
    """ Make sure STIX count data is in the correct units.

    This file is regularly saved out using different units.

    Parameters
    ----------
    stix_dict : dict
            Dictionary containing all STIX spectral file information.

    Returns
    -------
    A 2d array of the counts [counts], count rates [counts/s], and the count and count
    rate errors (counts, counts_err, cts_rates, cts_rate_err).
    """
    # The processed spectrogram files only come in counts/sec so need
    if stix_dict["1"][0]["TTYPE1"] == "RATE":
        cts_rates = stix_dict["1"][1]["RATE"]
        cts_rate_err = stix_dict["1"][1]["STAT_ERR"]
    else:
        print("I don\'t know what units STIX has.")

    return cts_rates, cts_rate_err


def _get_srm_file_info(srm_file):
    """ Return all STIX SRM data needed for fitting.

    SRM units returned as counts ph^(-1) cm^(2).

    Parameters
    ----------
    srm_file : str
            String for the STIX SRM spectral file under investigation.

    Returns
    -------
    A 2d array of the photon and channel bin edges (photon_bins, channel_bins), number of sub-set channels
    in the energy bin (ngrp), starting index of each sub-set of channels (fchan), number of channels in each
    sub-set (nchan), 2d array that is the spectral response (srm).
    """
    with fits.open(srm_file) as hdul:
        d0 = hdul[1].header
        d1 = hdul[1].data
        d3 = hdul[2].data

    pcb = np.concatenate((d1['ENERG_LO'][:, None], d1['ENERG_HI'][:, None]), axis=1)

    srmfsdict = {"photon_energy_bin_edges": pcb,
                 "count_energy_bin_edges": np.concatenate((d3['E_MIN'][:, None], d3['E_MAX'][:, None]), axis=1),
                 "drm": d1['MATRIX']*d0["GEOAREA"]}

    photon_bins = srmfsdict["photon_energy_bin_edges"]

    srm = srmfsdict["drm"]  # counts ph^-1 keV^-1

    channel_bins = srmfsdict["count_energy_bin_edges"]

    # srm units counts ph^(-1) kev^(-1); i.e., photons cm^(-2) go in and counts cm^(-2) kev^(-1) comes out # https://hesperia.gsfc.nasa.gov/ssw/hessi/doc/params/hsi_params_srm.htm#***
    # need srm units are counts ph^(-1) cm^(2)
    srm = srm * np.diff(channel_bins, axis=1).flatten()

    return photon_bins, channel_bins, srm