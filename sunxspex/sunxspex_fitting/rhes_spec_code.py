"""
The following code is used to make SRM/counts data in consistent units from RHESSI spectral data.
"""

import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.time import Time, TimeDelta
import astropy.table as atab

from . import io

__all__ = ["_get_spec_file_info", "_spec_file_units_check", "_get_srm_file_info"]


def load_spectrum(spec_fn: str):
    """ Return all RHESSI data needed for fitting.

    Parameters
    ----------
    spec_fn : str
            String for the RHESSI spectral file under investigation.

    Returns
    -------
    `dict` of:
        - A 2d array of the channel bin edges (channel_bins),
        - 2d array of the time bins for each spectrum (time_bins),
        - 2d array of livetimes/counts/count rates/count rate errors per
          channel bin and spectrum (lvt/counts/cts_rates/cts_rate_err, respectively).
    """
    with fits.open(spec_fn) as spec:
        rate_dat = spec['RATE']
        if rate_dat.header["SUMFLAG"] != 1:
            raise ValueError("Cannot perform spectroscopy on un-summed RHESSI data.")

        # Note that for rate, the units are per detector, i.e. counts sec-1 detector-1.
        # https://hesperia.gsfc.nasa.gov/rhessi3/software/spectroscopy/spectrum-software/index.html

        # livetimes, (rows,columns) -> (times, channels)
        lvt = rate_dat.data["LIVETIME"]
        # times of spectra, entries -> times. Times from start of the day of "DATE_OBS";
        # e.g.,"DATE_OBS"='2002-10-05T10:38:00.000' then times measured from '2002-10-05T00:00:00.000'
        start_time = Time(rate_dat.header['DATE_OBS'], format='isot', scale='utc')
        bin_starts = TimeDelta(rate_dat.data["TIME"], format='sec')
        bin_starts -= bin_starts[0]
        time_deltas = TimeDelta(rate_dat.data["TIMEDEL"], format='sec')

        spec_stimes = start_time + bin_starts
        spec_etimes = spec_stimes + time_deltas
        time_bins = np.column_stack((spec_stimes, spec_etimes))

        channels = spec['ENEBAND'].data
        channel_bins = np.column_stack((channels['E_MIN'], channels['E_MAX']))

        # get counts [counts], count rate [counts/s], and error on count rate
        counts, counts_err, cts_rates, cts_rate_err = _spec_file_units_check(
            hdu=spec['RATE'],
            livetimes=lvt,
            time_dels=time_deltas.to_value(u.s),
            kev_binning=channel_bins
        )

        attenuator_state_info = _extract_attenunator_info(
            spec['HESSI Spectral Object Parameters']
        )

    return dict(
        channel_bins=channel_bins,
        time_bins=time_bins,
        livetime=lvt,
        counts=counts,
        counts_err=counts_err,
        count_rate=cts_rates,
        count_rate_error=cts_rate_err,
        attenuator_state_info=attenuator_state_info
    )


def _extract_attenunator_info(att_dat) -> dict[str, list]:
    '''Pull out attenuator states and times'''
    # hack for now: swap the UNIX year with the date_obs year
    # they don't match (???)
    # waiting on reply from Kim Tolbert
    obs_year = Time(att_dat.header['DATE_OBS']).datetime.year
    dtimes = Time(att_dat.data['SP_ATTEN_STATE$$TIME'], format='unix').datetime[0]
    dtimes = [dt.replace(year=obs_year) for dt in dtimes]
    return {
        'change_times': Time(dtimes, format='datetime'),
        'states': list(*att_dat.data['SP_ATTEN_STATE$$STATE'])
    }


def _spec_file_units_check(hdu, livetimes, time_dels, kev_binning):
    """ Make sure RHESSI count data is in the correct units.

    This file is regularly saved out using different units.

    Parameters
    ----------
    hdu: astropy HDU
            rhessi 'RATE' data

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
    if hdu.header["TTYPE1"] != "RATE":
        raise ValueError("Only tested with units of 'RATE'."
                         "Go back to OSPEX and save counts data as 'RATE'")

    # count rate for every time, (rows,columns) -> (times, channels) [counts/sec]
    cts_rates = hdu.data["RATE"]
    # errors, (rows,columns) -> (times, channels)
    cts_rate_err = hdu.data["STAT_ERR"]
    counts = cts_rates * livetimes * time_dels[:, None]
    counts_err = cts_rate_err * livetimes * time_dels[:, None]
    return counts, counts_err, cts_rates, cts_rate_err


def srm_options_by_attenuator_state(hdu_list: list[dict[str, atab.QTable]]) -> dict[int, np.ndarray]:
    ''' Enumerate all possible SRMs for RHESSI based on attenuator state'''
    ret = dict()
    for hdu in hdu_list:
        if hdu['data'] is None:
            continue
        if 'MATRIX' not in hdu['data'].columns:
            continue
        state = hdu['header']['filter']
        ret[state] = hdu

    return ret


def load_srm(srm_file: str):
    """ Return all RHESSI SRM data needed for fitting.

    SRM units returned as counts ph^(-1) cm^(2).

    Parameters
    ----------
    srm_file : str
            String for the RHESSI SRM spectral file under investigation.

    Returns
    -------
    Dict of relevant SRM data.
    Notably returns all available SRM states from the given SRM .fits file.
    """
    srm_file_dat = io._read_rhessi_srm_file(srm_file)

    # handle attenuated responses and `None` simultaneously
    all_srms = srm_options_by_attenuator_state(srm_file_dat)

    if not all(h['header']['SUMFLAG'] for h in all_srms.values()):
        raise ValueError('SRM SUMFLAG\'s must be 1 for RHESSI spectroscopy')

    sample_key = list(all_srms.keys())[0]
    sample_srm = all_srms[sample_key]['data']
    low_photon_bins = sample_srm['ENERG_LO']
    high_photon_bins = sample_srm['ENERG_HI']
    photon_bins = np.column_stack((low_photon_bins, high_photon_bins)).to_value(u.keV)

    geo_area = srm_file_dat[3]['data']['GEOM_AREA'].astype(float).sum()

    channels = srm_file_dat[2]['data']
    channel_bins = np.column_stack((channels['E_MIN'], channels['E_MAX'])).to_value(u.keV)

    # need srm units in counts ph^(-1) cm^(2)
    ret_srms = {
        state: srm['data']['MATRIX'].data * np.diff(channel_bins, axis=1).flatten() * geo_area
        for (state, srm) in all_srms.items()
    }

    return dict(
        channel_bins=channel_bins,
        photon_bins=photon_bins,
        srm_options=ret_srms
    )
