import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from astropy.time import Time, TimeDelta
from astropy import units as u
import astropy.table as atab
import astropy.time as atime
from astropy.io import fits

from sunkit_spex.legacy.fitting import instruments


class STIXLoader(instruments.InstrumentBlueprint):
    # Formats for string times passed to different methods
    TIME_FMT = "isot"
    TIME_SCALE = "utc"

    """
    Loader specifically for processed STIX spectrogram data exported from the IDL software.

    This loader has the same format as the STIX/RHESSI loader from the older version of sunkit-spex (sunxspex, in leggacy fitting now).
    Once the format of the instrument loaders is decided, the loader should be moved to sunkit_spex.extern

    StixLoader Specifics
    ----------------------
    Has methods to plot time series and perform time selection on the data. A background time can be added or removed and can fit
    the event data with the model+background (recommended) or fit a model to data-background using the `data2data_minus_background`
    setter with False or True, respectively.

    We assume that the background (if calculated) and the event emission is calculated from the same sized area. If this is not
    true then the background effective exposure time should be multiplied by the ratio of the background area and the event area.
    I.e., background_effective_exposure * (background_area / event_area) as described in [1]. This may be automated (16/03/2022).

    [1] https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html

    Properties
    ----------
    data2data_minus_background : bool
            Returns True if the data to be fitted has been converted to event time minus background.

    end_background_time : `astropy.Time`
            Returns the set background end time.

    end_event_time : `astropy.Time`
            Returns the set event end time.

    start_background_time : `astropy.Time`
            Returns the set background start time.

    start_event_time : `astropy.Time`
            Returns the set event start time.

    Setters
    -------
    data2data_minus_background : bool
            Change the way the data is fitted; either event time minus background is fitted with the model (True) or
            event time is fitted with background+model (False, recommended and default behaviour).

    end_background_time : str, `astropy.Time`, None
            Sets the end time for the background time spectrum. None (default behaviour) sets this to None and removes or
            doesn't add a background component. The `start_background_time` setter can be assigned separately but must also
            be set to produce a background time.
            See `select_time` method to set both end and/or start time(s) in just one line.

    end_event_time : str, `astropy.Time`, None
            Sets the end time for the event time spectrum. None (default behaviour) sets this to the last time the loaded
            spectrum has data for.
            See `select_time` method to set both end and/or start time(s) in just one line.

    start_background_time : str, `astropy.Time`, None
            Sets the start time for the background time spectrum. None (default behaviour) sets this to None and removes or
            doesn't add a background component. The `end_background_time` setter can be assigned separately but must also
            be set to produce a background time.
            See `select_time` method to set both end and/or start time(s) in just one line.

    start_event_time : str, `astropy.Time`, None
            Sets the start time for the event time spectrum. None (default behaviour) sets this to the first time the loaded
            spectrum has data for.
            See `select_time` method to set both end and/or start time(s) in just one line.

    Methods
    -------
    lightcurve : energy_ranges (list of length=2, lists of length=2 lists, or None), axes (axes object or None)
            Plots the STIX time profile in the energy range given and on the axes provided. Default behaviour (energy_ranges=None)
            is to include all energies.

    update_event_time : start (str, `astropy.Time`, None), end (str, `astropy.Time`, None)
            Set the start and end times for the event data. The event time is assumed to commence/finish at the
            first/final data time if the start/end time is not given.

    update_background_time : start (str, `astropy.Time`, None), end (str, `astropy.Time`, None)
            Set the start and end times for the background data. Both start and end time need to be defined.

    spectrogram : axes (axes object or None) and any kwargs are passed to matplotlib.pyplot.imshow
            Plots the STIX spectrogram of all the data.

    Attributes
    ----------
    _channel_bins_inds_perspec : 2d array
            Array of channel bins (columns) per spectrum (rows).

    _construction_string : string
            String to show how class was constructed.

    _count_rate_perspec : 2d array
            Array of count rates per channel bin (columns) and spectrum (rows).

    _count_rate_error_perspec 2d array
            Array of count rate errors per channel bin (columns) and spectrum (rows).

    _counts_perspec : 2d array
            Array of counts per channel bin (columns) and spectrum (rows).

    _counts_err_perspec : 2d array
            Array of count error per channel bin (columns) and spectrum (rows).

    _end_background_time : `astropy.Time`
            End time for the defined background.
            Default: None

    _end_event_time : `astropy.Time`
            End time for the defined event.
            Default: Last time in loaded data.

    _full_obs_time : [`astropy.Time`, `astropy.Time`]
            Start and end time of the data loaded in.

    _lightcurve_data : {"mdtimes":_ts, "lightcurves":_lcs, "lightcurve_error":_lcs_err, "energy_ranges":energy_ranges}
            Arrays used to plots the lightcurves if the lightcurve method has been run.

    _loaded_spec_data : dict
            Instrument loaded spectral data.

    _lvt_perspec : 2d array
            Array of livetimes per channel bin (columns) and spectrum (rows).

    _spectrogram_data : {"spectrogram":_spect, "extent":_ext}
            Arrays used to plots the spectrogram if the spectrogram method has been run.

    _start_background_time
            Starting time for the defined background.
            Default: None

    _start_event_time
            Starting time for the defined event.
            Default: First time in loaded data.

    _time_bins_perspec : 2d array
            Array of time bins per spectrum.
    """

    def __init__(self, spectrum_file, srm_file, custom_channel_bins=None, custom_photon_bins=None, **kwargs):
        """Construct a string to show how the class was constructed (`_construction_string`) and set the `_loaded_spec_data` dictionary attribute."""

        self._construction_string = f"StixLoader(spectrum_file={spectrum_file},srm_file={srm_file},custom_channel_bins={custom_channel_bins},custom_photon_bins={custom_photon_bins},**{kwargs})"
        self._loaded_spec_data = self._load1spec(
            spectrum_file, srm_file, channel_bins=custom_channel_bins, photon_bins=custom_photon_bins
        )
        self._start_background_time, self._end_background_time = None, None
        self._start_event_time, self._end_event_time = self._full_obs_time[0], self._full_obs_time[1]

    def _getspec(self, spectrum_fn):
        """Load in STIX spectral data.

        Parameters
        ----------
        spectrum_fn : str
                String for the STIX spectral file under investigation.

        Returns
        -------
        A 2d array of the channel bin edges (channel_bins), 2d array of the channel bins (channel_bins_inds),
        2d array of the time bins for each spectrum (time_bins), 2d array of livetimes/counts/count rates/count
        rate errors per channel bin and spectrum (lvt/counts/cts_rates/cts_rate_err, respectively).
        """
        return self._get_spec_file_info(spectrum_fn)

    def _get_spec_file_info(self, spec_file):
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

        # All useful spectra info is in index "2", so get the energy bins
        channel_bins = np.concatenate((sdict["2"][1]["E_MIN"][:, None], sdict["2"][1]["E_MAX"][:, None]), axis=1)

        # Create channel bin arrays
        channel_bins_inds = np.empty(len(channel_bins))
        channel_bins_inds.fill(True)

        # Get TIME_DEL as we only get counts/s but also want values of counts
        times_mids = sdict["1"][1]["TIME"]
        time_del = sdict["1"][1]["TIMEDEL"]

        # if odd `t` in seconds then edges are [midt-floor(bin_width/2), midt+ceil(bin_width/2)]
        # e.g., mid_t=19, del_t=13 then edges would be [19-6, 19+7]=[13,26]
        _minus_half_bin_width = time_del / 2
        t_lo = times_mids - _minus_half_bin_width

        _plus_half_bin_width = time_del / 2
        t_hi = times_mids + _plus_half_bin_width

        # Calculating spectral times absTime[i] = mjd2any(MJDREF + TIMEZERO) + TIME[i]
        # So the time of first observation:
        timemjd = sdict["1"][0]["MJDREF"] + sdict["1"][0]["TIMEZERO"]
        date_beg = Time(timemjd, format="mjd", scale="utc")
        date_beg.format = "isot"

        spec_stimes = [date_beg + TimeDelta(dt * u.s) for dt in t_lo]
        spec_etimes = [date_beg + TimeDelta(dt * u.s) for dt in t_hi]
        time_bins = np.concatenate((np.array(spec_stimes)[:, None], np.array(spec_etimes)[:, None]), axis=1)

        # Getting livetime
        lvt = sdict["1"][1]["LIVETIME"]

        # Getting count rates and errors
        cts_rates, cts_rate_err = self._spec_file_units_check(sdict)
        cts_rates[np.nonzero(cts_rates < 0)] = 0

        # Calculating counts
        counts_err = cts_rate_err * lvt[:, None] * time_del[:, None]
        counts = cts_rates * lvt[:, None] * time_del[:, None]

        # Adding the 3% above 10 keV, 5% below 10keV  or 7% systematic errors below 7keV - matching OSPEX
        # Getting the array with percentages corresponding to each energy bin
        energy_bin_low = channel_bins[:, 0]
        energy_conditions = [energy_bin_low < 7, (energy_bin_low < 10) & (energy_bin_low >= 7), energy_bin_low >= 10]
        percentage = [0.07, 0.05, 0.03]

        systematic_err_percentage = np.select(energy_conditions, percentage)

        # Calculating systematic error
        systematic_err = systematic_err_percentage * counts

        # Adding the two errors in quadrature
        counts_err = np.sqrt(counts_err**2 + systematic_err**2)

        # Count rate error
        cts_rate_err = counts_err / lvt[:, None] / time_del[:, None]

        return channel_bins, channel_bins_inds, time_bins, lvt, counts, counts_err, cts_rates, cts_rate_err

    def _spec_file_units_check(self, stix_dict):
        """Make sure STIX count data is in the correct units.

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
            raise ValueError("I don't know what units STIX has.")

        return cts_rates, cts_rate_err

    def _getsrm(self, srm_file: str):
        """Return all STIX SRM data needed for fitting.

        SRM units returned as counts ph^(-1) cm^(2).

        Parameters
        ----------
        f_srm : str
                String for the STIX SRM spectral file under investigation.

        Returns
        -------
        A 2d array of the photon and channel bin edges (photon_bins, channel_bins), number of sub-set channels
        in the energy bin (ngrp), starting index of each sub-set of channels (fchan), number of channels in each
        sub-set (nchan), 2d array that is the spectral response (srm).
        """
        srm_file_dat = list()
        with fits.open(srm_file) as hdul:
            for hdu_idx in range(len(hdul)):
                try:
                    cur_hdu = atab.QTable.read(hdul, format="fits", hdu=hdu_idx)
                except ValueError:
                    cur_hdu = None
                srm_file_dat.append({"header": hdul[hdu_idx].header, "data": cur_hdu})

        # handle attenuated responses and `None` simultaneously
        all_srms = srm_options_by_attenuator_state(srm_file_dat)

        sample_key = list(all_srms.keys())[0]
        sample_srm = all_srms[sample_key]["data"]
        low_photon_bins = sample_srm["ENERG_LO"]
        high_photon_bins = sample_srm["ENERG_HI"]
        photon_bins = np.column_stack((low_photon_bins, high_photon_bins)).to_value(u.keV)

        geo_area = srm_file_dat[1]["header"]["GEOAREA"]

        channels = srm_file_dat[2]["data"]
        channel_bins = np.column_stack((channels["E_MIN"], channels["E_MAX"])).to_value(u.keV)

        # need srm units in counts ph^(-1) cm^(2)
        ret_srms = {
            state: srm["data"]["MATRIX"].data * np.diff(channel_bins, axis=1).flatten() * geo_area
            for (state, srm) in all_srms.items()
        }

        self._attenuator_state_info = _extract_attenunator_info(srm_file_dat[3], self._full_obs_time[1])

        return dict(channel_bins=channel_bins, photon_bins=photon_bins, srm_options=ret_srms)

    def _load1spec(self, spectrum_fn, srm_fn, channel_bins=None, photon_bins=None):
        """Loads all the information in for a given spectrum.

        Parameters
        ----------
        spectrum_fn, srm_fn : string
                Filenames for the relevant spectral files.

        photon_bins, channel_bins: 2d array
                User defined channel bins for the rows and columns of the SRM matrix.
                E.g., custom_channel_bins=[[1,1.5],[1.5,2],...]
                Default: None

        Returns
        -------
        Dictionary of the loaded in spectral information in the form {"photon_channel_bins":channel_bins,
                                                                      "photon_channel_mids":np.mean(channel_bins, axis=1),
                                                                      "photon_channel_binning":channel_binning,
                                                                      "count_channel_bins":channel_bins,
                                                                      "count_channel_mids":np.mean(channel_bins, axis=1),
                                                                      "count_channel_binning":channel_binning,
                                                                      "counts":counts,
                                                                      "count_error":count_error,
                                                                      "count_rate":count_rate,
                                                                      "count_rate_error":count_rate_error,
                                                                      "effective_exposure":eff_exp,
                                                                      "srm":srm,
                                                                      "extras":{"pha.file":f_pha,
                                                                                "srm.file":f_srm,
                                                                                "srm.ngrp":ngrp,
                                                                                "srm.fchan":fchan,
                                                                                "srm.nchan":nchan,
                                                                                "counts=data-bg":False}
                                                                     }.
        """
        # need effective exposure and energy binning since likelihood works on counts, not count rates etc.
        (
            obs_channel_bins,
            self._channel_bins_inds_perspec,
            self._time_bins_perspec,
            self._lvt_perspec,
            self._counts_perspec,
            self._counts_err_perspec,
            self._count_rate_perspec,
            self._count_rate_error_perspec,
        ) = self._getspec(spectrum_fn)

        # Full observation time range
        self._full_obs_time = [self._time_bins_perspec[0, 0], self._time_bins_perspec[-1, -1]]

        # Loading in the SRM
        self._srm = self._getsrm(srm_fn)
        srm_photon_bins = self._srm["photon_bins"]
        srm_channel_bins = self._srm["channel_bins"]
        # needs an srm file load it in
        srm = self._srm["srm_options"][0]

        # make sure the SRM will only produce counts to match the data
        data_inds2match = np.where(
            (obs_channel_bins[0, 0] <= srm_channel_bins[:, 0]) & (srm_channel_bins[:, 1] <= obs_channel_bins[-1, -1])
        )
        srm = srm[:, data_inds2match[0]]

        photon_bins = srm_photon_bins if type(photon_bins) is type(None) else photon_bins
        photon_binning = np.diff(photon_bins).flatten()

        # from the srm file #channel_binning = np.diff(channel_bins).flatten()
        channel_bins = obs_channel_bins if type(channel_bins) is type(None) else channel_bins

        # default is no background and all data is the spectrum to be fitted
        counts = np.sum(
            self._data_time_select(
                stime=self._full_obs_time[0], full_data=self._counts_perspec, etime=self._full_obs_time[1]
            ),
            axis=0,
        )

        count_rate = np.sum(
            self._data_time_select(
                stime=self._full_obs_time[0], full_data=self._count_rate_perspec, etime=self._full_obs_time[1]
            ),
            axis=0,
        )

        counts_err = np.sqrt(
            np.sum(
                self._data_time_select(
                    stime=self._full_obs_time[0], full_data=self._counts_err_perspec, etime=self._full_obs_time[1]
                )
                ** 2,
                axis=0,
            )
        )

        count_rate_error = np.sqrt(
            np.sum(
                self._data_time_select(
                    stime=self._full_obs_time[0], full_data=self._count_rate_error_perspec, etime=self._full_obs_time[1]
                )
                ** 2,
                axis=0,
            )
        )

        _livetimes = np.mean(
            self._data_time_select(
                stime=self._full_obs_time[0], full_data=self._lvt_perspec, etime=self._full_obs_time[1]
            ),
            axis=0,
        )  # to convert a model count rate to counts, so need mean
        eff_exp = np.diff(self._full_obs_time)[0].to_value("s") * _livetimes

        channel_binning = np.diff(obs_channel_bins, axis=1).flatten()
        count_rate = count_rate / channel_binning
        count_rate_error = count_rate_error / channel_binning

        # what spectral info you want to know from this observation
        return {
            "photon_channel_bins": photon_bins,
            "photon_channel_mids": np.mean(photon_bins, axis=1),
            "photon_channel_binning": photon_binning,
            "count_channel_bins": channel_bins,
            "count_channel_mids": np.mean(channel_bins, axis=1),
            "count_channel_binning": channel_binning,
            "counts": counts,
            "count_error": counts_err,
            "count_rate": count_rate,
            "count_rate_error": count_rate_error,
            "effective_exposure": eff_exp,
            "srm": srm,
            "extras": {"pha.file": spectrum_fn, "srm.file": srm_fn, "counts=data-bg": False},
        }  # this might make it easier to add different observations together

    def _update_srm_state(self):
        """
        Updates SRM state (attenuator state) given the event times.
        If the times span attenuator states, throws an error.
        """

        start_time, end_time = self._start_event_time, self._end_event_time
        change_times = self._attenuator_state_info["change_times"]
        if len(change_times) > 1:
            for t in change_times:
                if start_time <= t <= end_time:
                    warnings.warn(
                        f"\ndo not update event times to ({start_time}, {end_time}): "
                        "covers attenuator state change. Don't trust this fit!"
                    )
        n_states = len(self._attenuator_state_info["states"])
        new_att_state = self._attenuator_state_info["states"][0]  # default to first
        if n_states > 1:
            for i in range(n_states):
                state = self._attenuator_state_info["states"][i]
                if change_times[i] < start_time and end_time < change_times[i + 1]:
                    new_att_state = state
                    break
        self._loaded_spec_data["srm"] = self._srm["srm_options"][new_att_state].astype(float)

    @property
    def data2data_minus_background(self):
        """***Property*** States whether the the data is event-background or not.

        Returns
        -------
        Bool.
        """
        return self._loaded_spec_data["extras"]["counts=data-bg"]

    @data2data_minus_background.setter
    def data2data_minus_background(self, boolean):
        """***Property Setter*** Allows the data to be changed to be event-background.

        Original data is stored in `_full_data` attribute. Default fitting will essentially have this as False and fit the event
        time data with model+background (recommended). If this is set to True then the data to be fitting will be converted to
        the event time data minus the background data and fitting with just model; this is the way STIX/OSPEX analysis has been
        done in the past but is not strictly correct.

        To convert back to fitting the event time data with model+background then this setter need only be set to False again.

        Every time a background is set this is set to False.

        Parameters
        ----------
        boolean : bool
                If True then the event data is changed to event-background and the _loaded_spec_data["extras"]["counts=data-bg"]
                entry is changed to True. If False (recommended) then the event data to be fitted will remain as is or will be
                converted back with the _loaded_spec_data["extras"]["counts=data-bg"] entry changed to False.

        Returns
        -------
        None.
        """
        # make sure a background is set
        if "background_rate" not in self._loaded_spec_data["extras"]:
            return

        # check you want to make data data-bg and that the data isn't already data-bg
        if boolean:
            # make sure to save the full data first (without background subtraction) if not already done
            if not hasattr(self, "_full_data"):
                self._full_data = {
                    "counts": self._loaded_spec_data["counts"],
                    "count_error": self._loaded_spec_data["count_error"],
                    "count_rate": self._loaded_spec_data["count_rate"],
                    "count_rate_error": self._loaded_spec_data["count_rate_error"],
                }

            new_cts = self._full_data["counts"] - (
                self._loaded_spec_data["extras"]["background_rate"]
                * self._loaded_spec_data["effective_exposure"]
                * self._loaded_spec_data["count_channel_binning"]
            )
            new_cts_err = np.sqrt(
                self._full_data["count_error"] ** 2
                + (
                    self._loaded_spec_data["extras"]["background_count_error"]
                    * (
                        self._loaded_spec_data["effective_exposure"]
                        / self._loaded_spec_data["extras"]["background_effective_exposure"]
                    )
                )
                ** 2
            )
            new_cts_rates = self._full_data["count_rate"] - self._loaded_spec_data["extras"]["background_rate"]
            new_cts_rates_err = np.sqrt(
                self._full_data["count_rate_error"] ** 2
                + self._loaded_spec_data["extras"]["background_rate_error"] ** 2
            )

            self._loaded_spec_data.update(
                {
                    "counts": new_cts,
                    "count_error": new_cts_err,
                    "count_rate": new_cts_rates,
                    "count_rate_error": new_cts_rates_err,
                }
            )
            self._loaded_spec_data["extras"]["counts=data-bg"] = True
        elif not boolean:
            # reset
            if hasattr(self, "_full_data"):
                self._loaded_spec_data.update(self._full_data)
                del self._full_data
            self._loaded_spec_data["extras"]["counts=data-bg"] = False

    def _data_time_select(self, stime, full_data, etime):
        """Index and return data in time range stime<=data<=etime.

        If stime/etime is None then they are assumed to be the first/last data time.

        Parameters
        ----------
        stime, etime : `astropy.Time`
                The start and end range (inclusive) of the data required.

        full_data : array
                Array to be indexed along axis 0 such that the data returned in within the time range.

        Returns
        -------
        Indexed array.
        """
        if (type(stime) is type(None)) and (type(etime) is type(None)):
            return full_data
        elif type(stime) is type(None):
            return full_data[np.where(self._time_bins_perspec[:, 1] <= etime)]
        elif type(etime) is type(None):
            return full_data[np.where(stime <= self._time_bins_perspec[:, 0])]
        else:
            return full_data[
                np.where((stime <= self._time_bins_perspec[:, 0]) & (self._time_bins_perspec[:, 1] <= etime))
            ]

    def _update_event_data_with_times(self):
        """Changes the data in `_loaded_spec_data` to the data in the defined event time range.

        Default time is the whole file time range.

        Returns
        -------
        None.
        """

        self._update_srm_state()

        # sum counts over time range
        self._loaded_spec_data["counts"] = np.sum(
            self._data_time_select(
                stime=self._start_event_time, full_data=self._counts_perspec, etime=self._end_event_time
            ),
            axis=0,
        )

        counts_err = np.sqrt(
            np.sum(
                self._data_time_select(
                    stime=self._start_event_time, full_data=self._counts_err_perspec, etime=self._end_event_time
                )
                ** 2,
                axis=0,
            )
        )

        self._loaded_spec_data["count_error"] = counts_err

        # isolate livetimes and time binning
        _livetimes = np.mean(
            self._data_time_select(
                stime=self._start_event_time, full_data=self._lvt_perspec, etime=self._end_event_time
            ),
            axis=0,
        )  # to convert a model count rate to counts, so need mean
        _actual_first_bin = self._data_time_select(
            stime=self._start_event_time, full_data=self._time_bins_perspec[:, 0], etime=self._end_event_time
        )[0]
        _actual_last_bin = self._data_time_select(
            stime=self._start_event_time, full_data=self._time_bins_perspec[:, 1], etime=self._end_event_time
        )[-1]
        self._loaded_spec_data["effective_exposure"] = (
            np.diff([_actual_first_bin, _actual_last_bin])[0].to_value("s") * _livetimes
        )

        # calculate new count rates and errors assuming we need livetime correction???
        # self._loaded_spec_data["count_rate"] = self._loaded_spec_data["counts"]/self._loaded_spec_data["effective_exposure"]/self._loaded_spec_data["count_channel_binning"]
        # self._loaded_spec_data["count_rate_error"] = counts_err/self._loaded_spec_data["effective_exposure"]/self._loaded_spec_data["count_channel_binning"]

        self._loaded_spec_data["count_rate"] = (
            np.mean(
                self._data_time_select(
                    stime=self._start_event_time, full_data=self._count_rate_perspec, etime=self._end_event_time
                ),
                axis=0,
            )
            / self._loaded_spec_data["count_channel_binning"]
        )
        self._loaded_spec_data["count_rate_error"] = (
            np.sqrt(
                np.sum(
                    self._data_time_select(
                        stime=self._start_event_time,
                        full_data=self._count_rate_error_perspec,
                        etime=self._end_event_time,
                    )
                    ** 2,
                    axis=0,
                )
            ) / len(self._data_time_select(
                        stime=self._start_event_time,
                        full_data=self._count_rate_error_perspec,
                        etime=self._end_event_time,
                    ))
            / self._loaded_spec_data["count_channel_binning"]
        )

    @property
    def start_event_time(self):
        """***Property*** States the set event starting time.

        Returns
        -------
        Astropy.Time of the set event starting time.
        """
        return self._start_event_time

    @start_event_time.setter
    def start_event_time(self, evt_stime):
        """***Property Setter*** Sets the event start time.

        Parameters
        ----------
        evt_stime : str, `astropy.Time`, None
                String to be given to astropy's Time, `astropy.Time` is used directly, None sets the
                start event time to be the first time of the data.

        Returns
        -------
        None.
        """
        # self._update_event_data_with_times()
        self.update_event_times(evt_stime, self._end_event_time)

    @property
    def end_event_time(self):
        """***Property*** States the set event end time.

        Returns
        -------
        Astropy.Time of the set event end time.
        """
        return self._end_event_time

    @end_event_time.setter
    def end_event_time(self, evt_etime):
        """***Property Setter*** Sets the event end time.

        Parameters
        ----------
        evt_stime : str, `astropy.Time`, None
                String to be given to astropy's Time, `astropy.Time` is used directly, None sets the
                start event time to be the last time of the data.

        Returns
        -------
        None.
        """

        self.update_event_times(self._start_event_time, evt_etime)

    def _update_bg_data_with_times(self):
        """Changes/adds the background data in `_loaded_spec_data["extras"]` to the data in the defined background time range.

        Background data is removed from `_loaded_spec_data["extras"]` is either the start or end time is set to None.

        Default is that there is no background.

        Returns
        -------
        None.
        """
        if (type(self._start_background_time) is not type(None)) and (
            type(self._end_background_time) is not type(None)
        ):
            # get background data, woo!
            # sum counts over time range
            self._loaded_spec_data["extras"]["background_counts"] = np.sum(
                self._data_time_select(
                    stime=self._start_background_time, full_data=self._counts_perspec, etime=self._end_background_time
                ),
                axis=0,
            )
            self._loaded_spec_data["extras"]["background_count_error"] = np.sqrt(
                self._loaded_spec_data["extras"]["background_counts"]
            )

            # isolate livetimes and time binning
            _livetimes = np.mean(
                self._data_time_select(
                    stime=self._start_background_time, full_data=self._lvt_perspec, etime=self._end_background_time
                ),
                axis=0,
            )  # to convert a model count rate to counts, so need mean
            _actual_first_bin = self._data_time_select(
                stime=self._start_background_time,
                full_data=self._time_bins_perspec[:, 0],
                etime=self._end_background_time,
            )[0]
            _actual_last_bin = self._data_time_select(
                stime=self._start_background_time,
                full_data=self._time_bins_perspec[:, 1],
                etime=self._end_background_time,
            )[-1]
            self._loaded_spec_data["extras"]["background_effective_exposure"] = (
                np.diff([_actual_first_bin, _actual_last_bin])[0].to_value("s") * _livetimes
            )

            # calculate new count rates and errors
            self._loaded_spec_data["extras"]["background_rate"] = (
                self._loaded_spec_data["extras"]["background_counts"]
                / self._loaded_spec_data["extras"]["background_effective_exposure"]
                / self._loaded_spec_data["count_channel_binning"]
            )
            self._loaded_spec_data["extras"]["background_rate_error"] = (
                np.sqrt(self._loaded_spec_data["extras"]["background_counts"])
                / self._loaded_spec_data["extras"]["background_effective_exposure"]
                / self._loaded_spec_data["count_channel_binning"]
            )

        else:
            # if either the start or end background time is None, or set to None, then makes sure the background data is removed if it is there
            for key in self._loaded_spec_data["extras"].copy().keys():
                if key.startswith("background"):
                    del self._loaded_spec_data["extras"][key]

    def _atimes2mdates(self, astrotimes):
        """Convert a list of `astropy.Time`s to matplotlib dates for plotting.

        Parameters
        ----------
        astrotimes : list of `astropy.Time`
                List of `astropy.Time`s to convert to list of matplotlib dates.

        Returns
        -------
        List of matplotlib dates.
        """
        # convert astro time to datetime then use list comprehension to convert to matplotlib dates
        md = []

        for dt in astrotimes:
            if dt is None:
                md.append(dt)
            else:
                md.append(mdates.date2num(dt.utc.datetime))

        return md

    def _mdates_minute_locator(self, _obs_dt=None):
        """Try to determine a nice tick separation for time axis on the lightcurve.

        Parameters
        ----------
        _obs_dt : float
                Number of seconds the axis spans.

        Returns
        -------
        `mdates.MinuteLocator`.
        """
        obs_dt = np.diff(self._full_obs_time)[0].to_value("s") if type(_obs_dt) is type(None) else _obs_dt
        if obs_dt > 3600 * 12:
            return mdates.MinuteLocator(byminute=[0], interval=1)
        elif 3600 * 3 < obs_dt <= 3600 * 12:
            return mdates.MinuteLocator(byminute=[0, 30], interval=1)
        elif 3600 < obs_dt <= 3600 * 3:
            return mdates.MinuteLocator(byminute=[0, 20, 40], interval=1)
        elif 3600 * 0.5 < obs_dt <= 3600:
            return mdates.MinuteLocator(byminute=[0, 10, 20, 30, 40, 50], interval=1)
        elif 600 < obs_dt <= 3600 * 0.5:
            return mdates.MinuteLocator(byminute=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], interval=1)
        elif 240 < obs_dt <= 600:
            return mdates.MinuteLocator(interval=2)
        else:
            return mdates.SecondLocator(bysecond=[0, 20, 40], interval=1)

    def _rebin_lc(self, arr, clump_bins):
        """Combines array elements in groups of `clump_bins`.

        Parameters
        ----------
        arr : numpy array
                Number of contiguous bins to combine.

        clump_bins : int>1
                Number of contiguous bins to combine.

        Returns
        -------
        Rebinned array.
        """
        return np.add.reduceat(arr, np.arange(0, len(arr), clump_bins))

    def _rebin_ts(self, times, clump_bins):
        """Combines bin array elements in groups of `clump_bins`.

        Mainly used to rebin time bins for lightcurve and spectrogram methods.

        Parameters
        ----------
        times : numpy array (2D)
                Number of contiguous bins to combine.

        clump_bins : int>1
                Number of contiguous bins to combine.

        Returns
        -------
        Rebinned 2D array.
        """
        _t_to_clump, _endt_to_clump = times[0::clump_bins], times[clump_bins - 1 :: clump_bins]
        _clumped_start_ts = _t_to_clump[:, 0]
        _clumped_end_ts = np.concatenate((_t_to_clump[1:, 0], [_endt_to_clump[-1, -1]]))
        return np.concatenate((_clumped_start_ts[:, None], _clumped_end_ts[:, None]), axis=1)

    def lightcurve(self, energy_ranges=None, axes=None, rebin_time=1):
        """Creates a STIX lightcurve.

        Helps the user see the STIX time profile. The defined event time (defined either through `start_event_time`,
        `end_event_time` setters, or `select_time(...)` method) is shown with a purple shaded region and if a background
        time (defined either through `start_background_time`, `end_background_time` setters, or
        `select_time(...,background=True)` method) is defined then it is shown with an orange shaded region.

        Parameters
        ----------
        energy_ranges : list of length=2 or lists of length=2 lists
                Energy ranges to plot. Default behaviour is full energy range.
                Default: None

        axes : axes object
                Axes object to plot on. Default gets changed to matplotlib.pyplot.
                Default: None

        rebin_time : int>1
                Number of contiguous time bins to combine.
                Default: 1

        Returns
        -------
        The axes object.

        Examples
        --------
        # use the class to load in data
        ar = StixLoader(pha_file=spec_file, srm_file=srm_file)

        # define a background range if we like; equivalent to ar.select_time(start="2002-10-05T10:38:32", end="2002-10-05T10:40:32", background=True)
        ar.start_background_time = "2002-10-05T10:38:32"
        ar.end_background_time = "2002-10-05T10:40:32"

        # change the event time range to something other than the full time range; equivalent to ar.select_time(start="2002-10-05T10:41:20", end="2002-10-05T10:42:24")
        ar.start_event_time = "2002-10-05T10:41:20"
        ar.end_event_time = "2002-10-05T10:42:24"

        # see the lightcurves for 5--10 keV, 10--30 keV, and 25--50 keV
        plt.figure(figsize=(9,6))
        ar.lightcurve(energy_ranges=[[5,10], [10,30], [25,50]])
        plt.show()

        """
        # just make sure we have a list of lists for the energy ranges
        if type(energy_ranges) is type(None):
            energy_ranges = [
                [
                    self._loaded_spec_data["count_channel_bins"][0, 0],
                    self._loaded_spec_data["count_channel_bins"][-1, -1],
                ]
            ]
        elif len(np.shape(energy_ranges)) == 1:
            energy_ranges = [energy_ranges]
        elif len(np.shape(energy_ranges)) == 0:
            print(
                "The `energy_ranges` input should be a range of two energy values (e.g., [4,8] meaning 4-8 keV inclusive) or a list of these ranges."
            )
            return

        ax = axes if type(axes) is not type(None) else plt.gca()
        _def_fs = plt.rcParams["font.size"]

        _lcs, _lcs_err = [], []
        _times = self._time_bins_perspec
        _times = self._rebin_ts(_times, rebin_time) if isinstance(rebin_time, int) and rebin_time > 1 else _times
        _ts = self._atimes2mdates(_times.flatten())

        # plot each energy range
        for er in energy_ranges:
            i = np.where(
                (self._loaded_spec_data["count_channel_bins"][:, 0] >= er[0])
                & (self._loaded_spec_data["count_channel_bins"][:, -1] <= er[-1])
            )
            time_binning = np.array([dt.to_value("s") for dt in np.diff(self._time_bins_perspec).flatten()])
            e_range_cts = np.sum(self._counts_perspec[:, i].reshape((len(time_binning), -1)), axis=1)

            if isinstance(rebin_time, int) and rebin_time > 1:
                e_range_cts = self._rebin_lc(e_range_cts, rebin_time)
                time_binning = self._rebin_lc(time_binning, rebin_time)

            e_range_ctr, e_range_ctr_err = e_range_cts / time_binning, np.sqrt(e_range_cts) / time_binning
            lc = np.concatenate((e_range_ctr[:, None], e_range_ctr[:, None]), axis=1).flatten()
            _lcs.append(lc)
            _lcs_err.append(e_range_ctr_err)
            _p = ax.plot(
                _ts, lc, label=f"{er[0]}$-${er[-1]} keV"
            )  # in case of uncommon time binning just plot cts/s per bin edge
            ax.errorbar(
                np.mean(np.reshape(_ts, (int(len(_ts) / 2), -1)), axis=1),
                e_range_ctr,
                yerr=e_range_ctr_err,
                c=_p[0].get_color(),
                ls="",
            )  # error bar in middle of the bin

        fmt = mdates.DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(self._mdates_minute_locator())

        ax.set_yscale("log")
        ax.set_xlabel(f"Time (Start Time: {self._full_obs_time[0]})")
        ax.set_ylabel("Counts s$^{-1}$")

        ax.set_title("STIX Lightcurve")
        plt.legend(fontsize=_def_fs - 5)

        # plot background time range if there is one
        _y_pos = (
            ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95
        )  # stop region label overlapping axis spine
        if (
            hasattr(self, "_start_background_time")
            and (type(self._start_background_time) is not type(None))
            and hasattr(self, "_end_background_time")
            and (type(self._end_background_time) is not type(None))
        ):
            ax.axvspan(
                *self._atimes2mdates([self._start_background_time, self._end_background_time]),
                alpha=0.1,
                color="orange",
            )
            ax.annotate(
                "BG",
                (self._atimes2mdates([self._start_background_time])[0], _y_pos),
                color="orange",
                va="top",
                size=_def_fs - 2,
            )

        # plot event time range
        if hasattr(self, "_start_event_time") and hasattr(self, "_end_event_time"):
            ax.axvspan(*self._atimes2mdates([self._start_event_time, self._end_event_time]), alpha=0.1, color="purple")
            ax.annotate(
                "Evt",
                (self._atimes2mdates([self._start_event_time])[0], _y_pos),
                color="purple",
                va="top",
                size=_def_fs - 2,
            )

        self._lightcurve_data = {
            "mdtimes": _ts,
            "lightcurves": _lcs,
            "lightcurve_error": _lcs_err,
            "energy_ranges": energy_ranges,
        }

        return ax

    def spectrogram(self, axes=None, rebin_time=1, rebin_energy=1, **kwargs):
        """Creates a STIX spectrogram.

        Helps the user see the STIX time and energy evolution. The defined event time (defined either through
        `start_event_time`, `end_event_time` setters, or `select_time(...)` method) is shown with a violet line
        and if a background time (defined either through `start_background_time`, `end_background_time` setters,
        or `select_time(...,background=True)` method) is defined then it is shown with an orange line.

        Parameters
        ----------
        axes : axes object
                Axes object to plot on. Default gets changed to matplotlib.pyplot.
                Default: None

        rebin_time : int>1
                Number of contiguous time bins to combine.
                Default: 1

        rebin_rate : int>1
                Number of contiguous count rate bins to combine.
                Default: 1

        kwargs :  passed to matplotlib.pyplot.imshow()

        Returns
        -------
        The axes object.

        Examples
        --------
        # use the class to load in data
        ar = StixLoader(pha_file=spec_file, srm_file=srm_file)

        # define a background range if we like; equivalent to ar.select_time(start="2002-10-05T10:38:32", end="2002-10-05T10:40:32", background=True)
        ar.start_background_time = "2002-10-05T10:38:32"
        ar.end_background_time = "2002-10-05T10:40:32"

        # change the event time range to something other than the full time range; equivalent to ar.select_time(start="2002-10-05T10:41:20", end="2002-10-05T10:42:24")
        ar.start_event_time = "2002-10-05T10:41:20"
        ar.end_event_time = "2002-10-05T10:42:24"

        # see the spectrogram
        plt.figure(figsize=(9,6))
        ar.spectrogram()
        plt.show()

        """

        ax = axes if type(axes) is not type(None) else plt.gca()

        _cmap = "plasma" if "cmap" not in kwargs else kwargs["cmap"]
        kwargs.pop("cmap", None)
        _aspect = "auto" if "aspect" not in kwargs else kwargs["aspect"]
        kwargs.pop("aspect", None)

        _def_fs = plt.rcParams["font.size"]

        # get cts/s, and times and energy bin ranges
        time_binning = np.array([dt.to_value("s") for dt in np.diff(self._time_bins_perspec).flatten()])
        e_range_cts = self._counts_perspec
        _times = self._time_bins_perspec

        # check if the times are being rebinned
        if isinstance(rebin_time, int) and rebin_time > 1:
            e_range_cts = self._rebin_lc(e_range_cts, rebin_time)
            time_binning = self._rebin_lc(time_binning, rebin_time)
            _times = self._rebin_ts(_times, rebin_time)

        _cts_rate = e_range_cts / time_binning[:, None]
        _e_bins = self._loaded_spec_data["count_channel_bins"]

        # rebin the energies if needed
        if isinstance(rebin_energy, int) and rebin_energy > 1:
            _cts_rate = self._rebin_lc(_cts_rate.T, rebin_energy).T
            _e_bins = self._rebin_ts(self._loaded_spec_data["count_channel_bins"], rebin_energy)

        # check if the time bins need combined
        _t = self._atimes2mdates(_times.flatten())

        # plot spectrogram
        etop = _e_bins[-1][-1]
        _ext = [_t[0], _t[-1], _e_bins[0][0], etop]
        _spect = _cts_rate.T
        ax.imshow(_spect, origin="lower", extent=_ext, aspect=_aspect, cmap=_cmap, **kwargs)

        fmt = mdates.DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(self._mdates_minute_locator())

        ax.set_xlabel(f"Time (Start Time: {self._full_obs_time[0]})")
        ax.set_ylabel("Energy [keV]")

        ax.set_title("STIX Spectrogram [Counts s$^{-1}$]")

        # change event and background start and end times from astropy dates to matplotlib dates
        start_evt_time, end_evt_time, start_bg_time, end_bg_time = self._atimes2mdates(
            [self._start_event_time, self._end_event_time, self._start_background_time, self._end_background_time]
        )

        # plot background time range if there is one
        _y_pos = (
            ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95
        )  # stop region label overlapping axis spine
        if (
            hasattr(self, "_start_background_time")
            and (type(self._start_background_time) is not type(None))
            and hasattr(self, "_end_background_time")
            and (type(self._end_background_time) is not type(None))
        ):
            ax.hlines(y=etop, xmin=start_bg_time, xmax=end_bg_time, alpha=0.9, color="orange", capstyle="butt", lw=10)
            ax.annotate("BG", (start_bg_time, _y_pos), color="orange", va="top", size=_def_fs - 2)

        # plot event time range
        if hasattr(self, "_start_event_time") and hasattr(self, "_end_event_time"):
            ax.hlines(
                y=etop, xmin=start_evt_time, xmax=end_evt_time, alpha=0.9, color="#F37AFF", capstyle="butt", lw=10
            )
            ax.annotate("Evt", (start_evt_time, _y_pos), color="#F37AFF", va="top", size=_def_fs - 2)

        self._spectrogram_data = {"spectrogram": _spect, "extent": _ext}

        return ax

    @property
    def start_background_time(self):
        """
        Returns
        -------
        Time of the set background starting time.
        """
        return self._start_background_time

    @start_background_time.setter
    def start_background_time(self, bg_stime):
        """
        Parameters
        ----------
        bg_stime : str | `astropy.Time` to update to update
        """
        self.update_background_times(start=bg_stime, end=self._end_background_time)

    @property
    def end_background_time(self):
        """
        Returns
        -------
        Time of the set background end time.
        """
        return self._end_background_time

    @end_background_time.setter
    def end_background_time(self, bg_etime):
        """
        Parameters
        ----------
        bg_etime : str | `astropy.Time` to update
        """
        self.update_background_times(start=self._start_background_time, end=bg_etime)

    def update_event_times(self, start: str | Time, end: str | Time):
        """Provides method to set start and end time of the event.

        Parameters
        ----------
        start, end : str, `astropy.Time`
                String to be given to astropy's Time, `astropy.Time` is used directly
        Returns
        -------
        None.

        Examples
        --------
        # use the class to load in data
        ar = StixLoader(pha_file=spec_file, srm_file=srm_file)

        # change the event time range to something other than the full time range; equivalent to doing ar.start_event_time = "2022-10-05T10:41:20" and ar.end_event_time = "2022-10-05T10:42:24"
        ar.update_event_times(start="2022-10-05T10:41:20", end="2022-10-05T10:42:24")
        """
        start, end = self._time_verification(start, end)
        self._start_event_time = start
        self._end_event_time = end
        self._update_event_data_with_times()

    def update_background_times(self, start: str | Time, end: str | Time) -> None:
        """Provides method to set start and end time of the background
        Parameters
        ----------
        start, end : str, `astropy.Time`
                String to be given to astropy's Time, `astropy.Time` is used directly


        The `data2data_minus_background` setter is reset to False; if the `data2data_minus_background` setter
        was used by the user and they still want to fit the event time data minus the background then this
        must be set to True again. If you don't know what the `data2data_minus_background` setter is then
        don't worry about it.

        Returns
        -------
        None.

        Examples
        --------
        # use the class to load in data
        ar = StixLoader(pha_file=spec_file, srm_file=srm_file)

        # define a background range if we like; equivalent to doing ar.start_background_time = "2002-10-05T10:38:32" and ar.end_background_time = "2002-10-05T10:40:32"
        ar.update_background_times(start="2002-10-05T10:38:32", end="2002-10-05T10:40:32")

        """
        start, end = self._time_verification(start, end)
        self._start_background_time = start
        self._end_background_time = end
        self._update_bg_data_with_times()
        # change back to separate event time and background data
        self.data2data_minus_background = False

    def _check_relative_times(self, new: str | Time, other: str | Time, compare_less: bool) -> Time:
        """
        Ensures that the `new` time doesn't go past the `other`.
        If `compare_less` is True, then the comparison is `new < other`,
        else it is `other < new`.
        On success, returns the time.
        It will coerce a `str` argument into an `astropy.time.Time`."""
        try:
            t = Time(new, format=STIXLoader.TIME_FMT, scale=STIXLoader.TIME_SCALE)
        except ValueError as e:
            raise ValueError(
                f"Time '{new}' is not convertable to {STIXLoader.TIME_FMT} format at {STIXLoader.TIME_SCALE} scale."
            ) from e

        passed_check = (t < other) if compare_less else (t > other)
        if passed_check:
            return new

        # Otherwise ...
        self._time_error()

    def _time_verification(self, start: str | Time, end: str | Time) -> Time:
        """Ensure that a time range is consistent with the observing time,
           and coerce strings into astropy.time.Time objects."""
        data_start, data_end = self._full_obs_time
        start = self._check_relative_times(start, data_end, compare_less=True)
        end = self._check_relative_times(end, data_start, compare_less=False)

        if start > end:
            self._time_error()

        return Time((start, end))

    def _time_error(self):
        raise ValueError(
            "The start and/or end time being set is not appropriate. "
            "The data will not be changed. Please set start < end."
        )

    def select_time(self, *_):
        raise DeprecationWarning(
            "Selecting background and event time with this method is deprecated. "
            "In the future, please use `STIXLoader.update_event_times` to update the event time "
            "or `STIXLoader.update_background_times` to update the background time."
        )


def _extract_attenunator_info(att_dat, spectrum_end_time) -> dict[str, list]:
    """Pull out attenuator states and times"""
    n_attenuator_changes = att_dat["data"]["SP_ATTEN_STATE$$TIME"].size
    atten_change_times = atime.Time(att_dat["data"]["SP_ATTEN_STATE$$TIME"], format="utime").utc
    atten_change_times = atten_change_times.reshape(n_attenuator_changes)  # reshape so always 1d array

    # Use lists to fix deprecated numpy API breaking change in astropy.times
    atten_change_times = atime.Time(list(atten_change_times) + [spectrum_end_time])

    return {
        "change_times": atten_change_times,
        "states": att_dat["data"]["SP_ATTEN_STATE$$STATE"].reshape(n_attenuator_changes).tolist(),
    }


def srm_options_by_attenuator_state(hdu_list: list[dict[str, atab.QTable]]) -> dict[int, np.ndarray]:
    """Enumerate all possible SRMs for STIX based on attenuator state"""
    ret = dict()
    for hdu in hdu_list:
        if hdu["data"] is None:
            continue
        if "MATRIX" not in hdu["data"].columns:
            continue
        state = hdu["header"]["filter"]
        ret[state] = hdu

    return ret
