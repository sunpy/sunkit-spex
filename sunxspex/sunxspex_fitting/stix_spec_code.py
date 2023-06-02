"""
The following code is used to make SRM/counts data in consistent units from STIX spectral data.
"""

import numpy as np

from astropy import units as u
from astropy.time import Time, TimeDelta

from . import io

__all__ = ["_get_spec_file_info", "_spec_file_units_check", "_get_srm_file_info"]


def _get_spec_file_info(spec_file):
    """ Return all STIX data needed for fitting.

    Parameters
    ----------
    spec_file : str
            String for the STIX spectral file under investigation.

    Returns
    -------
    A 2d array of the channel bin edges (channel_bins), 2d array of the channel bins (channel_bins_inds),
    2d array of the time bins for each spectrum (time_bins), 2d array of livetimes/counts/count rates/count
    rate errors per channel bin and spectrum (lvt/counts/cts_rates/cts_rate_err, respectively).
    """
    sdict = io._read_stix_spec_file(spec_file)

    times_mids = sdict["2"][1]["time"]  # mid-times of spectra, entries -> times. Mid-times from start of observation
    time_deltas = sdict["2"][1]["timedel"]  # times deltas of spectra, entries -> times
    time_diff_so2e = sdict["0"][0]["EAR_TDEL"]  # time difference between Sun2Earth and Sun2SO, so time at earth for measurement is this difference added on to the actual detection time
    # spectrum number in the file, entries -> times # spec_num = sdict["1"][1]["SPEC_NUM"]

    # if odd `t` in seconds then edges are [midt-floor(bin_width/2), midt+ceil(bin_width/2)]
    # e.g., mid_t=19, del_t=13 then edges would be [19-6, 19+7]=[13,26]
    _minus_half_bin_width = np.floor(time_deltas/2)
    t_lo = times_mids - _minus_half_bin_width
    _plus_half_bin_width = np.ceil(time_deltas/2)
    t_hi = times_mids + _plus_half_bin_width

    spec_stimes = [Time(sdict["0"][0]["DATE-BEG"], format='isot', scale='utc')+TimeDelta(time_diff_so2e * u.s)+TimeDelta(dt * u.cs) for dt in t_lo]
    spec_etimes = [Time(sdict["0"][0]["DATE-BEG"], format='isot', scale='utc')+TimeDelta(time_diff_so2e * u.s)+TimeDelta(dt * u.cs) for dt in t_hi]
    time_bins = np.concatenate((np.array(spec_stimes)[:, None], np.array(spec_etimes)[:, None]), axis=1)

    channel_bins_inds, channel_bins = _return_masked_bins(sdict)

    lvt = np.ones((len(t_lo), len(channel_bins)))  # livetimes, (rows,columns) -> (times, channels), for STIX it is 1 at the minute?

    # get counts [counts], count rate [counts/s], and error on count and count rate
    counts, counts_err, cts_rates, cts_rate_err = _spec_file_units_check(stix_dict=sdict, time_dels=time_deltas)

    return channel_bins, channel_bins_inds, time_bins, lvt, counts, counts_err, cts_rates, cts_rate_err


def _return_masked_bins(sdict):
    """ Return the energy bins where there is data.

    Parameters
    ----------
    sdict : dict
            Dictionary containing all STIX spectral file information.

    Returns
    -------
    A 2d array of the energy bin edges.
    """
    # get all energy bin edges (0--inf keV)
    e_bins = np.concatenate((sdict["4"][1]['e_low'][:, None], sdict["4"][1]['e_high'][:, None]), axis=1)

    # get all indices of the energy bins needed
    mask_inds = sdict["1"][1]['energy_bin_edge_mask'][0].astype(bool)

    return mask_inds, e_bins


def _spec_file_units_check(stix_dict, time_dels):
    """ Make sure STIX count data is in the correct units.

    This file is regularly saved out using different units.

    Parameters
    ----------
    stix_dict : dict
            Dictionary containing all STIX spectral file information.

    time_dels : 1d array
            The time duration of each recorded spectrum.

    Returns
    -------
    A 2d array of the counts [counts], count rates [counts/s], and the count and count
    rate errors (counts, counts_err, cts_rates, cts_rate_err).
    """
    # stix can be saved out with counts, counts/sec, or counts/sec/cm^2/keV using counts, rate, or flux, respectively
    if stix_dict["0"][0]["BUNIT"] == "counts":
        counts = stix_dict["2"][1]["counts"][:, :]
        counts_err = np.sqrt(counts)  # should this be added in quadrature to the estimated count compression error
        cts_rates = counts / time_dels[:, None]
        cts_rate_err = counts_err / time_dels[:, None]
    else:
        print("I don\'t know what units STIX has.")

    return counts, counts_err, cts_rates, cts_rate_err


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
    srmfsdict = io._read_stix_srm_file(srm_file)

    photon_bins = srmfsdict["photon_energy_bin_edges"]

    srm = srmfsdict["drm"]  # counts ph^-1 keV^-1

    channel_bins = srmfsdict["count_energy_bin_edges"]

    # srm units counts ph^(-1) kev^(-1); i.e., photons cm^(-2) go in and counts cm^(-2) kev^(-1) comes out # https://hesperia.gsfc.nasa.gov/ssw/hessi/doc/params/hsi_params_srm.htm#***
    # need srm units are counts ph^(-1) cm^(2)
    srm = srm * np.diff(channel_bins, axis=1).flatten()

    return photon_bins, channel_bins, srm
