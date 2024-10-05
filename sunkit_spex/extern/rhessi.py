import copy
import warnings

from astropy.io import fits
import astropy.table as atab
import astropy.units as u
import astropy.time as atime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from sunkit_spex.legacy.fitting import instruments
from sunpy.time import time  # noqa: F401


class RhessiLoader(instruments.InstrumentBlueprint):
    """
    RhessiLoader Specifics
    ----------------------
    Has methods to plot time series and perform time selection on the data. A background time can be added or removed and can fit
    the event data with the model+background (recommended) or fit a model to data-background using the `data2data_minus_background`
    setter with False or True, respectively.

    We assume that the background (if calculated) and the event emission is calculated from the same sized area. If this is not
    true then the background effective exposure time should be multiplied by the ratio of the background area and the event area.
    I.e., background_effective_exposure * (background_area / event_area) as described in [1]. This may be automated (16/03/2022).

    [1] https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html
    """

    def __init__(self, spectrum_fn, srm_fn, **kwargs):
        """
        Spectrum and SRM files are both required: attenuator state change times
            are in the spectrum file,
            and the state determines which SRM will be used.
        """
        self._construction_string = f"RhessiLoader(spectrum_fn={spectrum_fn}, " f"srm_fn={srm_fn}," f"**{kwargs})"
        self._systematic_error = 0
        self.load_prepare_spectrum_srm(spectrum_fn, srm_fn)
        self._start_background_time, self._end_background_time = None, None

    @property
    def systematic_error(self):
        """Allow a systemtaic error to be added in quadrature with the counts data."""
        return self._systematic_error

    @systematic_error.setter
    def systematic_error(self, err):
        self._systematic_error = err
        self._update_event_data_with_times()

    def load_prepare_spectrum_srm(self, spectrum_fn, srm_fn):
        """Loads all the information in for a given spectrum.

        Parameters
        ----------
        spectrum_fn, srm_fn : string
            Filenames for the spectrum and SRM .fits files from OSPEX.

        Returns
        -------
        Standard sunkit_spex dictionary (see docs)
        """
        # Load spectrum & SRM
        self._spectrum = (spec := load_spectrum(spectrum_fn))
        self._attenuator_state_info = spec.pop("attenuator_state_info")
        self._srm = (srm := load_srm(srm_fn))

        # make sure the SRM will only produce counts to match the data
        counts_indices_which_match = np.where(
            (spec["channel_bins"][0, 0] <= srm["channel_bins"][:, 0])
            & (srm["channel_bins"][:, 1] <= spec["channel_bins"][-1, -1])
        )[0]
        srm["srm_options"] = {
            state: matrix[:, counts_indices_which_match] for (state, matrix) in srm["srm_options"].items()
        }

        photon_bins = srm["photon_bins"]
        photon_binning = np.diff(photon_bins).flatten()
        channel_bins = spec["channel_bins"]

        # default is no background and all data is the spectrum to be fitted
        self.start_data_time, self.end_data_time = (spec["time_bins"][0, 0], spec["time_bins"][-1, -1])
        # avoid SRM setting issues
        self._start_event_time, self._end_event_time = self.start_data_time, self.end_data_time

        self._loaded_spec_data = {
            "photon_channel_bins": photon_bins,
            "photon_channel_binning": photon_binning,
            "photon_channel_mids": photon_bins[:, 0] + photon_binning / 2,
            "count_channel_bins": channel_bins,
            "count_channel_binning": (ch_de := np.diff(channel_bins, axis=1).flatten()),
            "count_channel_mids": channel_bins[:, 0] + ch_de / 2,
            "srm": list(srm["srm_options"].values())[0],  # default to first state
            "extras": dict(),
        }
        self._update_event_data_with_times()
        self._original_data = copy.deepcopy(self._loaded_spec_data)

    @property
    def subtract_background(self):
        """
        States whether the the data is event minus background or not.
        """
        return self._loaded_spec_data["extras"]["counts=data-bg"]

    @subtract_background.setter
    def subtract_background(self, do_subtract: bool):
        """
        Allows the data to be changed to be event minus background.

        Original data is stored in `_original_data` attribute. Default fitting will essentially have this as False and fit the event
        time data with model+background (recommended). If this is set to True then the data to be fitting will be converted to
        the event time data minus the background data and fitting with just model; this is the way RHESSI/OSPEX analysis has been
        done in the past but is not strictly correct.

        To convert back to fitting the event time data with model+background then this setter need only be set to False again.

        Everytime a background is set this is set to False.
        """
        if "background_rate" not in self._loaded_spec_data["extras"]:
            raise ValueError("No background set. Cannot update subtraction state.")
        if not do_subtract:
            self._loaded_spec_data["extras"]["counts=data-bg"] = False
            # remove SRM from the original data so we dont accidentally modify it
            # and background: not set initially
            no_srm = copy.deepcopy(self._original_data)
            _ = no_srm.pop("srm", None)
            _ = no_srm.pop("extras", None)
            self._loaded_spec_data.update(no_srm)
            return

        scaled_bg = (
            self._loaded_spec_data["extras"]["background_rate"]
            * self._loaded_spec_data["effective_exposure"]
            * self._loaded_spec_data["count_channel_binning"]
        )
        new_cts = self._original_data["counts"] - scaled_bg

        scaled_bg_err = (
            self._loaded_spec_data["extras"]["background_count_error"]
            * self._loaded_spec_data["effective_exposure"]
            / self._loaded_spec_data["extras"]["background_effective_exposure"]
        )
        new_cts_err = np.sqrt(self._original_data["count_error"] ** 2 + scaled_bg_err**2)

        new_rate = self._original_data["count_rate"] - self._loaded_spec_data["extras"]["background_rate"]
        new_rate_err = np.sqrt(
            self._original_data["count_rate_error"] ** 2
            + self._loaded_spec_data["extras"]["background_rate_error"] ** 2
        )

        self._loaded_spec_data.update(
            {"counts": new_cts, "count_error": new_cts_err, "count_rate": new_rate, "count_rate_error": new_rate_err}
        )
        self._loaded_spec_data["extras"]["counts=data-bg"] = True

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
        if stime is None and etime is None:
            return full_data
        elif stime is None:
            return full_data[np.where(self._spectrum["time_bins"][:, 1] <= etime)]
        elif etime is None:
            return full_data[np.where(stime <= self._spectrum["time_bins"][:, 0])]
        return full_data[
            np.where((stime <= self._spectrum["time_bins"][:, 0]) & (self._spectrum["time_bins"][:, 1] <= etime))
        ]

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
            for i in range(n_states - 1):
                state = self._attenuator_state_info["states"][i]
                if change_times[i] < start_time and end_time < change_times[i + 1]:
                    new_att_state = state
                    break
        self._loaded_spec_data["srm"] = self._srm["srm_options"][new_att_state].astype(float)

    def _update_event_data_with_times(self):
        """
        Changes the data in `_loaded_spec_data` to the data in the defined event time range.
        Returns
        -------
        None.
        """
        self._update_srm_state()

        # sum counts over time range
        self._loaded_spec_data["counts"] = (
            cts := np.sum(
                self._data_time_select(
                    stime=self._start_event_time, full_data=self._spectrum["counts"], etime=self._end_event_time
                ),
                axis=0,
            )
        )
        # do not assume Poisson error
        self._loaded_spec_data["count_error"] = (
            err := np.sqrt(
                np.sum(
                    self._data_time_select(
                        stime=self._start_event_time, full_data=self._spectrum["counts_err"], etime=self._end_event_time
                    )
                    ** 2,
                    axis=0,
                )
            )
        )

        if np.any(np.atleast_1d(self.systematic_error) > 0):
            self._loaded_spec_data["count_error"] = (err := np.sqrt(err**2 + (self._systematic_error * cts) ** 2))

        livetimes = np.mean(
            self._data_time_select(
                stime=self._start_event_time, full_data=self._spectrum["livetime"], etime=self._end_event_time
            ),
            axis=0,
        )
        actual_first_bin = self._data_time_select(
            stime=self._start_event_time, full_data=self._spectrum["time_bins"][:, 0], etime=self._end_event_time
        )[0]
        actual_last_bin = self._data_time_select(
            stime=self._start_event_time, full_data=self._spectrum["time_bins"][:, 1], etime=self._end_event_time
        )[-1]
        self._loaded_spec_data["effective_exposure"] = (
            np.diff([actual_first_bin, actual_last_bin])[0].to_value("s") * livetimes
        )

        # calculate new count rates and errors
        effe = self._loaded_spec_data["effective_exposure"]
        de = self._loaded_spec_data["count_channel_binning"]
        self._loaded_spec_data["count_rate"] = cts / effe / de
        self._loaded_spec_data["count_rate_error"] = err / effe / de

    def update_event_times(self, ta, tb):
        """Update start, end event times"""
        if not isinstance(ta, atime.Time) or not isinstance(tb, atime.Time):
            raise ValueError("Event times must be astropy.time.Time")
        if ta > tb:
            self._time_error()
        self._start_event_time = ta
        self._end_event_time = tb
        self._update_event_data_with_times()

    def _update_bg_data_with_times(self):
        """
        Changes/adds the background data in `_loaded_spec_data["extras"]` to the data in the defined background time range.
        Background data is removed from `_loaded_spec_data["extras"]` is either the start or end time is set to None.
        Default is that there is no background.
        """
        clear_all_bg_data = (self._start_background_time is None) or (self._end_background_time is None)
        if clear_all_bg_data:
            keyz = list(self._loaded_spec_data["extras"].keys())
            for key in keyz:
                if "background" in key:
                    del self._loaded_spec_data["extras"][key]
            return

        # sum counts over time range
        self._loaded_spec_data["extras"]["background_counts"] = (
            cts := np.sum(
                self._data_time_select(
                    stime=self._start_background_time,
                    full_data=self._spectrum["counts"],
                    etime=self._end_background_time,
                ),
                axis=0,
            )
        )

        # TODO: this is an assumption of Poisson statistics (not necessarily correct)
        self._loaded_spec_data["extras"]["background_count_error"] = (
            cts_err := np.sqrt(self._loaded_spec_data["extras"]["background_counts"])
        )

        # isolate livetimes and time binning
        livetimes = np.mean(
            self._data_time_select(
                stime=self._start_background_time, full_data=self._spectrum["livetime"], etime=self._end_background_time
            ),
            axis=0,
        )
        actual_first_bin = self._data_time_select(
            stime=self._start_background_time,
            full_data=self._spectrum["time_bins"][:, 0],
            etime=self._end_background_time,
        )[0]
        actual_last_bin = self._data_time_select(
            stime=self._start_background_time,
            full_data=self._spectrum["time_bins"][:, 1],
            etime=self._end_background_time,
        )[-1]
        self._loaded_spec_data["extras"]["background_effective_exposure"] = (
            effe := ((actual_last_bin - actual_first_bin).to_value("s") * livetimes)
        )

        # calculate new count rates and errors
        de = self._loaded_spec_data["count_channel_binning"]
        self._loaded_spec_data["extras"]["background_rate"] = cts / effe / de
        self._loaded_spec_data["extras"]["background_rate_error"] = cts_err / effe / de

    def update_background_times(self, ta: atime.Time, tb: atime.Time):
        """Update start and end background times."""
        if (ta is not None) and (tb is not None) and (ta > tb):
            self._time_error()

        self._start_background_time = ta
        self._end_background_time = tb
        self._update_bg_data_with_times()
        self.subtract_background = False

    def _time_error(self):
        raise ValueError(
            "The start and/or end time being set is not appropriate."
            "The data will not be changed. Please set start < end."
        )

    def _mdates_minute_locator(self):
        """Try to determine a nice tick separation for time axis on the lightcurve.

        Returns
        -------
        `mdates.MinuteLocator`.
        """
        obs_dt = (self.end_data_time - self.start_data_time).to_value("s")
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
        return np.column_stack((_clumped_start_ts, _clumped_end_ts))

    def lightcurve(self, energy_ranges=None, axes=None, rebin_time=1):
        """Creates a RHESSI lightcurve.

        Helps the user see the RHESSI time profile. The defined event time (defined either through `start_event_time`,
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
        """
        # just make sure we have a list of lists for the energy ranges
        if energy_ranges is None:
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

        ax = axes or plt.gca()
        font_size = plt.rcParams["font.size"]

        orig_bins = self._spectrum["time_bins"]
        flat_times = atime.Time(np.concatenate((orig_bins[:, 0], [orig_bins[-1, -1]])))

        if rebin_time > 1:
            flat_times = self._rebin_ts(flat_times, rebin_time)
        time_mids = flat_times[:-1] + (flat_times[1:] - flat_times[:-1]) / 2

        # plot each energy range
        time_binning = (flat_times[1:] - flat_times[:-1]).to_value("s")
        for er in energy_ranges:
            i = np.where(
                (self._loaded_spec_data["count_channel_bins"][:, 0] >= er[0])
                & (self._loaded_spec_data["count_channel_bins"][:, -1] <= er[-1])
            )
            e_range_cts = np.sum(self._spectrum["counts"][:, i].reshape((len(time_binning), -1)), axis=1)

            if rebin_time > 1:
                e_range_cts = self._rebin_lc(e_range_cts, rebin_time)
                time_binning = self._rebin_lc(time_binning, rebin_time)

            e_range_ctr, e_range_ctr_err = e_range_cts / time_binning, np.sqrt(e_range_cts) / time_binning
            lc = e_range_ctr
            p = ax.stairs(lc, flat_times.datetime, label=f"{er[0]}$-${er[-1]} keV")
            ax.errorbar(
                x=time_mids.datetime, y=e_range_ctr, yerr=e_range_ctr_err, c=p.get_edgecolor(), ls=""
            )  # error bar in middle of the bin

        fmt = mdates.DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(self._mdates_minute_locator())

        ax.set_yscale("log")
        ax.set_xlabel(f"Time (Start Time: {self.start_data_time})")
        ax.set_ylabel("Counts s$^{-1}$")

        ax.set_title("RHESSI Lightcurve")
        plt.legend(fontsize=font_size - 5)

        # plot background time range if there is one
        _y_pos = (
            ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95
        )  # stop region label overlapping axis spine
        if (self._start_background_time is not None) and (self._end_background_time is not None):
            ax.axvspan(
                *atime.Time([self._start_background_time, self._end_background_time]).datetime,
                alpha=0.1,
                color="orange",
            )
            ax.annotate(
                "BG", (self._start_background_time.datetime, _y_pos), color="orange", va="top", size=font_size - 2
            )

        # plot event time range
        if hasattr(self, "_start_event_time") and hasattr(self, "_end_event_time"):
            ax.axvspan(self._start_event_time.datetime, self._end_event_time.datetime, alpha=0.1, color="purple")
            ax.annotate("Evt", (self._start_event_time.datetime, _y_pos), color="purple", va="top", size=font_size - 2)

        return ax

    def spectrogram(self, axes=None, rebin_time=1, rebin_energy=1, **kwargs):
        """Creates a RHESSI spectrogram.

        Helps the user see the RHESSI time and energy evolution. The defined event time (defined either through
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

        """
        ax = axes or plt.gca()

        _def_fs = plt.rcParams["font.size"]

        # get cts/s, and times and energy bin ranges
        time_binning = np.array([dt.to_value("s") for dt in np.diff(self._spectrum["time_bins"]).flatten()])
        e_range_cts = self._spectrum["counts"]
        t = atime.Time(np.concatenate((self._spectrum["time_bins"][:, 0], [self._spectrum["time_bins"][-1, -1]])))

        # check if the times are being rebinned
        if rebin_time > 1:
            e_range_cts = self._rebin_lc(e_range_cts, rebin_time)
            time_binning = self._rebin_lc(time_binning, rebin_time)
            t = self._rebin_ts(t, rebin_time)

        cts_rate = e_range_cts / time_binning[:, None]
        ebins = self._loaded_spec_data["count_channel_bins"]

        # rebin the energies if needed
        if rebin_energy > 1:
            cts_rate = self._rebin_lc(cts_rate.T, rebin_energy).T
            ebins = self._rebin_ts(ebins, rebin_energy)

        ebins = np.unique(ebins.flatten())

        # check if the time bins need combined
        t = t.datetime

        # plot spectrogram
        spect = cts_rate.T
        opts = {"cmap": "plasma"}
        opts.update(kwargs)
        ax.pcolormesh(t, ebins, spect, **opts)

        fmt = mdates.DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(self._mdates_minute_locator())

        ax.set_xlabel(f"Time (Start Time: {self._start_event_time})")
        ax.set_ylabel("Energy [keV]")

        ax.set_title("RHESSI Spectrogram [Counts s$^{-1}$]")

        # change event and background start and end times from astropy dates to matplotlib dates
        start_evt_time, end_evt_time, start_bg_time, end_bg_time = atime.Time(
            [self._start_event_time, self._end_event_time, self._start_background_time, self._end_background_time]
        ).datetime

        # plot background time range if there is one
        _y_pos = (
            ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95
        )  # stop region label overlapping axis spine
        etop = ebins.max()
        if (self._start_background_time is not None) and (self._end_background_time is not None):
            ax.hlines(y=etop, xmin=start_bg_time, xmax=end_bg_time, alpha=0.9, color="orange", capstyle="butt", lw=10)
            ax.annotate("BG", (start_bg_time, _y_pos), color="orange", va="top", size=_def_fs - 2)

        # plot event time range
        ax.hlines(y=etop, xmin=start_evt_time, xmax=end_evt_time, alpha=0.9, color="#F37AFF", capstyle="butt", lw=10)
        ax.annotate("Evt", (start_evt_time, _y_pos), color="#F37AFF", va="top", size=_def_fs - 2)

        return ax

    def select_time(self, start=None, end=None, background=False):
        """Provides method to set start and end time of the event or background in one line.

        Parameters
        ----------
        start, end : str, `astropy.Time`, None
                String to be given to astropy's Time, `astropy.Time` is used directly, None sets the
                start/end event time to be the first time of the data. None doesn't add, or will remove,
                any background data in `_loaded_spec_data["extras"]` if background is set to True.
                Default: None

        background : bool
                Determines whether the start and end times are for the event (False) or background
                time (True).
                Default: False

        Returns
        -------
        None.
        """
        self.__warn = False if (start is not None) and (end is not None) else True

        if background:
            self.start_background_time, self.end_background_time = start, end
        else:
            self.start_event_time, self.end_event_time = start, end


def load_spectrum(spec_fn: str):
    """Return all RHESSI data needed for fitting.

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
        rate_dat = spec["RATE"]
        if rate_dat.header["SUMFLAG"] != 1:
            raise ValueError("Cannot perform spectroscopy on un-summed RHESSI data.")

        # Note that for rate, the units are per detector, i.e. counts sec-1 detector-1.
        # https://hesperia.gsfc.nasa.gov/rhessi3/software/spectroscopy/spectrum-software/index.html

        # livetimes, (rows,columns) -> (times, channels)
        lvt = rate_dat.data["LIVETIME"]
        # times of spectra, entries -> times. Times from start of the day of "DATE_OBS";
        # e.g.,"DATE_OBS"='2002-10-05T10:38:00.000' then times measured from '2002-10-05T00:00:00.000'
        start_time = atime.Time(rate_dat.header["DATE_OBS"], format="isot", scale="utc")
        bin_starts = atime.TimeDelta(rate_dat.data["TIME"], format="sec")
        bin_starts -= bin_starts[0]
        time_deltas = atime.TimeDelta(rate_dat.data["TIMEDEL"], format="sec")

        spec_stimes = start_time + bin_starts
        spec_etimes = spec_stimes + time_deltas
        time_bins = np.column_stack((spec_stimes, spec_etimes))

        channels = spec["ENEBAND"].data
        channel_bins = np.column_stack((channels["E_MIN"], channels["E_MAX"]))

        # get counts [counts], count rate [counts/s], and error on count rate
        counts, counts_err, cts_rates, cts_rate_err = _spec_file_units_check(
            hdu=spec["RATE"], livetimes=lvt, time_dels=time_deltas.to_value(u.s), kev_binning=channel_bins
        )

        attenuator_state_info = _extract_attenunator_info(spec["HESSI Spectral Object Parameters"])

    return dict(
        channel_bins=channel_bins,
        time_bins=time_bins,
        livetime=lvt,
        counts=counts,
        counts_err=counts_err,
        count_rate=cts_rates,
        count_rate_error=cts_rate_err,
        attenuator_state_info=attenuator_state_info,
    )


def _extract_attenunator_info(att_dat) -> dict[str, list]:
    """Pull out attenuator states and times"""
    n_attenuator_changes = att_dat.data["SP_ATTEN_STATE$$TIME"].size
    atten_change_times = atime.Time(att_dat.data["SP_ATTEN_STATE$$TIME"], format="utime").utc
    atten_change_times = atten_change_times.reshape(n_attenuator_changes)  # reshape so always 1d array
    return {
        "change_times": atten_change_times,
        "states": att_dat.data["SP_ATTEN_STATE$$STATE"].reshape(n_attenuator_changes).tolist(),
    }


def _spec_file_units_check(hdu, livetimes, time_dels, kev_binning):
    """Make sure RHESSI count data is in the correct units.

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
        raise ValueError("Only tested with units of 'RATE'." "Go back to OSPEX and save counts data as 'RATE'")

    # count rate for every time, (rows,columns) -> (times, channels) [counts/sec]
    cts_rates = hdu.data["RATE"]
    # errors, (rows,columns) -> (times, channels)
    cts_rate_err = hdu.data["STAT_ERR"]
    counts = cts_rates * livetimes * time_dels[:, None]
    counts_err = cts_rate_err * livetimes * time_dels[:, None]
    return counts, counts_err, cts_rates, cts_rate_err


def srm_options_by_attenuator_state(hdu_list: list[dict[str, atab.QTable]]) -> dict[int, np.ndarray]:
    """Enumerate all possible SRMs for RHESSI based on attenuator state"""
    ret = dict()
    for hdu in hdu_list:
        if hdu["data"] is None:
            continue
        if "MATRIX" not in hdu["data"].columns:
            continue
        state = hdu["header"]["filter"]
        ret[state] = hdu

    return ret


def load_srm(srm_file: str):
    """Return all RHESSI SRM data needed for fitting.

    SRM units returned as counts ph^(-1) cm^(2).

    Parameters
    ----------
    srm_file : str
            String for the RHESSI SRM spectral file under investigation.

    Returns
    -------
    Dictionary of relevant SRM data.
    Notably returns all available SRM states from the given SRM .fits file.
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

    if not all(h["header"]["SUMFLAG"] for h in all_srms.values()):
        raise ValueError("SRM SUMFLAG's must be 1 for RHESSI spectroscopy")

    sample_key = list(all_srms.keys())[0]
    sample_srm = all_srms[sample_key]["data"]
    low_photon_bins = sample_srm["ENERG_LO"]
    high_photon_bins = sample_srm["ENERG_HI"]
    photon_bins = np.column_stack((low_photon_bins, high_photon_bins)).to_value(u.keV)

    geo_area = srm_file_dat[3]["data"]["GEOM_AREA"].astype(float).sum()

    channels = srm_file_dat[2]["data"]
    channel_bins = np.column_stack((channels["E_MIN"], channels["E_MAX"])).to_value(u.keV)

    # need srm units in counts ph^(-1) cm^(2)
    ret_srms = {
        state: srm["data"]["MATRIX"].data * np.diff(channel_bins, axis=1).flatten() * geo_area
        for (state, srm) in all_srms.items()
    }

    return dict(channel_bins=channel_bins, photon_bins=photon_bins, srm_options=ret_srms)
