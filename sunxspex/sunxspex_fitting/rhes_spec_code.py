"""
The following code is used to make SRM/counts data in consistent units from RHESSI spectral data.
"""

import numpy as np

from astropy import units as u
from astropy.time import Time, TimeDelta

from . import io

__all__ = ["_get_spec_file_info", "_spec_file_units_check", "_get_srm_file_info"]


def _get_spec_file_info(spec_file):
    """ Return all RHESSI data needed for fitting.

    Parameters
    ----------
    spec_file : str
            String for the RHESSI spectral file under investigation.

    Returns
    -------
    A 2d array of the channel bin edges (channel_bins), 2d array of the channel bins (channel_bins_inds),
    2d array of the time bins for each spectrum (time_bins), 2d array of livetimes/counts/count rates/count
    rate errors per channel bin and spectrum (lvt/counts/cts_rates/cts_rate_err, respectively).
    """
    rdict = io._read_rhessi_spec_file(spec_file)

    if rdict["1"][0]["SUMFLAG"] != 1:
        print("Apparently spectrum file\'s `SUMFLAG` should be one and I don\'t know what to do otherwise at the moment.")
        return

    # Note that for rate, the units are per detector, i.e. counts sec-1 detector-1. https://hesperia.gsfc.nasa.gov/rhessi3/software/spectroscopy/spectrum-software/index.html
    # -> I tnhink this is the default but false for the first simple case I tried. I think sum_flag=1 sums the detectors up for spectra and srm

    channel_bins_inds = rdict["1"][1]["CHANNEL"]  # channel numbers, (rows,columns) -> (times, channels)
    lvt = rdict["1"][1]["LIVETIME"]  # livetimes, (rows,columns) -> (times, channels)
    times_s = rdict["1"][1]["TIME"]  # times of spectra, entries -> times. Times from start of the day of "DATE_OBS"; e.g.,"DATE_OBS"='2002-10-05T10:38:00.000' then times measured from '2002-10-05T00:00:00.000'
    time_deltas = rdict["1"][1]["TIMEDEL"]  # times deltas of spectra, entries -> times
    # spectrum number in the file, entries -> times # spec_num = rdict["1"][1]["SPEC_NUM"]

    td = times_s-times_s[0]
    spec_stimes = [Time(rdict["0"][0]["DATE_OBS"], format='isot', scale='utc')+TimeDelta(dt * u.s) for dt in td]
    spec_etimes = [st+TimeDelta(dt * u.s) for st, dt in zip(spec_stimes, time_deltas)]
    time_bins = np.concatenate((np.array(spec_stimes)[:, None], np.array(spec_etimes)[:, None]), axis=1)

    channels = rdict["2"][1]  # [(chan, lowE, hiE), ...], rdict["2"][0] has units etc.
    channel_bins = np.concatenate((np.array(channels['E_MIN'])[:, None], np.array(channels['E_MAX'])[:, None]), axis=1)

    # get counts [counts], count rate [counts/s], and error on count rate
    counts, counts_err, cts_rates, cts_rate_err = _spec_file_units_check(rhessi_dict=rdict, livetimes=lvt, time_dels=time_deltas, kev_binning=channel_bins)

    return channel_bins, channel_bins_inds, time_bins, lvt, counts, counts_err, cts_rates, cts_rate_err


def _spec_file_units_check(rhessi_dict, livetimes, time_dels, kev_binning):
    """ Make sure RHESSI count data is in the correct units.

    This file is regularly saved out using different units.

    Parameters
    ----------
    rhessi_dict : dict
            Dictionary containing all RHESSI spectral file information.

    livetimes : 2d array
            Fraction livetimes for each energy bin and each recorded spectrum.

    time_dels : 1d array
            The time duration of each recorded spectrum.

    kev_binning : 2d array
            Array of the energy bin edges; e.g., [[1,1.5],[1.5,2],...].

    Returns
    -------
    A 2d array of the counts [counts], count rates [counts/s], and the count and count
    rate errors (counts, counts_err, cts_rates, cts_rate_err).
    """
    # rhessi can be saved out with counts, counts/sec, or counts/sec/cm^2/keV using counts, rate, or flux, respectively
    if rhessi_dict["1"][0]["TTYPE1"] == "RATE":
        cts_rates = rhessi_dict["1"][1]["RATE"]  # count rate for every time, (rows,columns) -> (times, channels) [counts/sec]
        cts_rate_err = rhessi_dict["1"][1]["STAT_ERR"]  # errors, (rows,columns) -> (times, channels)
        counts = cts_rates * livetimes * time_dels[:, None]
        counts_err = cts_rate_err * livetimes * time_dels[:, None]
    elif rhessi_dict["1"][0]["TTYPE1"] == "COUNTS":
        # **** Not Tested ****
        counts = rhessi_dict["1"][1]["COUNTS"]  # counts for every time, (rows,columns) -> (times, channels) [counts]
        counts_err = rhessi_dict["1"][1]["STAT_ERR"]
        cts_rates = counts / livetimes / time_dels[:, None]
        cts_rate_err = np.sqrt(counts) / livetimes / time_dels[:, None]
    elif rhessi_dict["1"][0]["TTYPE1"] == "FLUX":
        # **** Not Tested ****
        _flux = rhessi_dict["1"][1]["FLUX"]  # flux for every time, (rows,columns) -> (times, channels) [counts/sec/cm^2/keV]
        cts_rates = _flux * rhessi_dict["1"][0]["GEOAREA"] * np.diff(kev_binning, axis=1).flatten()
        cts_rate_err = rhessi_dict["1"][1]["STAT_ERR"] * rhessi_dict["1"][0]["GEOAREA"] * np.diff(kev_binning, axis=1).flatten()
        counts = cts_rates * livetimes * time_dels[:, None]
        counts_err = cts_rate_err * livetimes * time_dels[:, None]
    else:
        print("I don\'t know what units RHESSI has.")

    return counts, counts_err, cts_rates, cts_rate_err


def _get_srm_file_info(srm_file):
    """ Return all RHESSI SRM data needed for fitting.

    SRM units returned as counts ph^(-1) cm^(2).

    Parameters
    ----------
    srm_file : str
            String for the RHESSI SRM spectral file under investigation.

    Returns
    -------
    A 2d array of the photon and channel bin edges (photon_bins, channel_bins), number of sub-set channels
    in the energy bin (ngrp), starting index of each sub-set of channels (fchan), number of channels in each
    sub-set (nchan), 2d array that is the spectral response (srm).
    """
    srmfrdict = io._read_rhessi_srm_file(srm_file)

    if srmfrdict["1"][0]["SUMFLAG"] != 1:
        print("Apparently srm file\'s `SUMFLAG` should be one and I don\'t know what to do otherwise at the moment.")
        return

    photon_channels_elo = srmfrdict["1"][1]['ENERG_LO']  # photon channel edges, different to count channels
    photon_channels_ehi = srmfrdict["1"][1]['ENERG_HI']
    photon_bins = np.concatenate((np.array(photon_channels_elo)[:, None], np.array(photon_channels_ehi)[:, None]), axis=1)

    # other info but left out
    # ngrp = srmfrdict["1"][1]['N_GRP'] # number of count groups along a photon bin to construct the SRM
    # fchan = srmfrdict["1"][1]['F_CHAN'] # starting index for each count group along a photon channel
    # nchan = srmfrdict["1"][1]['N_CHAN'] # number of matrix entries each count group along a photon channel
    srm = srmfrdict["1"][1]['MATRIX']  # counts ph^-1 keV^-1
    geo_area = srmfrdict["3"][0]['GEOAREA']

    channels = srmfrdict["2"][1]  # [(chan, lowE, hiE), ...], srmfrdict["2"][0] has units etc. count channels for SRM
    channel_bins = np.concatenate((np.array(channels['E_MIN'])[:, None], np.array(channels['E_MAX'])[:, None]), axis=1)

    # srm units counts ph^(-1) kev^(-1); i.e., photons cm^(-2) go in and counts cm^(-2) kev^(-1) comes out # https://hesperia.gsfc.nasa.gov/ssw/hessi/doc/params/hsi_params_srm.htm#***
    # need srm units are counts ph^(-1) cm^(2)
    srm = srm * np.diff(channel_bins, axis=1).flatten() * geo_area

    return photon_bins, channel_bins, srm
