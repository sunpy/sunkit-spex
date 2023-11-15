"""
The following code is for instrument specific classes each using their own methods to create and edit their `_loaded_spec_data` attrbutes.

Tips that I have been following:
    * None of the instrument loaders should have public attributes; ie., all attributes should be preceded with `_`
    * Only obvious and useful methods and setters should be public, all else preceded with `_`
"""

import copy
from os import path as os_path
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from astropy.time import Time

from . import io
from . import nu_spec_code as nu_spec
from . import rhes_spec_code as rhes_spec

__all__ = ["NustarLoader", "RhessiLoader", "CustomLoader", "rebin_any_array"]

# Get a default class for the instrument specfic loaders
# Once the instrument specific loaders inherit from this then all they really have to do is get the spectral
#    data they want to fit in the correct dictionary form and assigned to `self._loaded_spec_data`.


class InstrumentBlueprint:
    """ The blueprint class for an instruemnt to be given to the `DataLoader` class in data_loader.py.

    The main aim of these classes is to:
            (1) produce a `_loaded_spec_data` attribute with the instrument spectral data in the
                form {"photon_channel_bins":Photon Space Bins (e.g., [keV,keV],[keV,keV],...]),
                      "photon_channel_mids":Photon Space Bin Mid-points (e.g., [keV,...]),
                      "photon_channel_binning":Photon Space Binwidths (e.g., [keV,...]),
                      "count_channel_bins":Count Space Bins (e.g., [keV,keV],[keV,keV],...]),
                      "count_channel_mids":Count Space Bin Mid-points (e.g., [keV,...]),
                      "count_channel_binning":Count Space Binwidths (e.g., [keV,...]),
                      "counts":counts (e.g., cts),
                      "count_error":Count Error for `counts`,
                      "count_rate":Count Rate (e.g., cts/keV/s),
                      "count_rate_error":Count Rate Error for `count_rate`,
                      "effective_exposure":Effective Exposure (e.g., s),
                      "srm":Spectral Response Matrix (e.g., cts/ph * cm^2),
                      "extras":{"any_extra_info":or_empty_dict}
                     };
            (2) provide instrument specific methods such as time/spatial/spectral range selectors
                and SRM rebinning methods that then update the `_loaded_spec_data` attrbute
                appropriately.

    Instrument loader classes are expected to receive the PHA spctral file (`pha_file`) as the first
    argument then other spectral information (`arf_file`, `rmf_file`, `srm_custom`,
    `custom_channel_bins`). Obviously not all of these files need be used and so just pass then through
    as **kwargs.

    The `DataLoader` class in data_loader.py then creates a dictionary attrbute called `loaded_spec_data`
    (note no underscore) that is then getable by the user when spectral fitting with the `SunXspex` class
    in fitter.py where the keys are each spectum's ID (e.g, spectrum1, spectrum2, etc.).

    This means that, while fitting STIX data with spectrum ID "spectrum1" for example, if the user wants
    to change the time interval for the spectrum (e.g., with an intrument specific method time_range)
    they can do this by `loaded_spec_data["spectrum1"].time_range(new_time_range)` which will update the
    StixLoader `_loaded_spec_data` attribute located in loaded_spec_data["spectrum1"].
    """

    _UNIVERSAL_DOC_ = """Parameters
                         ----------
                         pha_file : string
                                 The PHA file for the spectrum to be loaded.

                         arf_file, rmf_file : string
                                 The ARF and RMF files associated with the PHA file(s). If none are given (e.g, with
                                 NuSTAR data) it is assumed that these are in the same directory with same filename
                                 as the PHA file(s) but with extensions '.arf' and '.rmf', respectively.

                         srm_file : string
                                 The file that contains the spectral response matrix for the given spectrum.

                         srm_custom : 2d array
                                 User defined spectral response matrix. This is accepted over the SRM created from any
                                 ARF and RMF files given.

                         custom_channel_bins, custom_photon_bins : 2d array
                                 User defined channel bins for the columns and rows of the SRM matrix.
                                 E.g., custom_channel_bins=[[1,1.5],[1.5,2],...]

                         Attributes
                         ----------
                         _construction_string : string
                                 String to show how class was constructed.

                         _loaded_spec_data : dict
                                 Loaded spectral data.
                         """

    def _rebin_rmf(self, matrix, old_count_bins=None, new_count_bins=None, old_photon_bins=None, new_photon_bins=None, axis="count"):
        """ Rebins the photon and/or count channels of the redistribution matrix if needed.

        This will rebin any 2d array by taking the mean across photon space (rows) and summing
        across count space (columns).

        If no effective area information from the instrument then this is passed straight
        to `_rebin_srm`, if there is then the `_rebin_srm` should be overwritten.

        Parameters
        ----------
        matrix : 2d array
                Redistribution matrix.

        old_count_bins, new_count_bins : 1d arrays
                The old count channel binning and the new binning to be for the redistribution matrix columns (sum columns).

        old_photon_bins, new_photon_bins : 1d arrays
                The old photon channel binning and the new binning to be for the redistribution matrix columns (average rows).

        axis : string
                Define what \'axis\' the binning should be applied to. E.g., \'photon\', \'count\', or \'photon_and_count\'.

        Returns
        -------
        The rebinned 2d redistribution matrix.
        """
        # across channel bins, we sum. across energy bins, we average
        # appears to be >2x faster to average first then sum if needing to do both
        if (axis == "photon") or (axis == "photon_and_count"):
            # very slight difference to rbnrmf when binning across photon axis, <2% of entries have a ratio (my way/rbnrmf) >1 (up to 11)
            # all come from where the original rmf has zeros originally so might be down to precision being worked in, can't expect the exact same numbers essentially
            matrix = rebin_any_array(data=matrix, old_bins=old_photon_bins, new_bins=new_photon_bins, combine_by="mean")
        if (axis == "count") or (axis == "photon_and_count"):
            matrix = rebin_any_array(data=matrix.T, old_bins=old_count_bins, new_bins=new_count_bins, combine_by="sum").T  # need to go along columns so .T then .T back

        return matrix

    def _channel_bin_info(self, axis):
        """ Returns the old and new channel bins for the indicated axis (count axis, photon axis or both).

        Parameters
        ----------
        axis : string
                Set to "count", "photon", or "photon_and_count" to return the old and new count
                channel bins, photon channel bins, or both.

        Returns
        -------
        Arrays of old_count_bins, new_count_bins, old_photon_bins, new_photon_bins or Nones.
        """
        old_count_bins, new_count_bins, old_photon_bins, new_photon_bins = None, None, None, None
        if (axis == "count") or (axis == "photon_and_count"):
            old_count_bins = self._loaded_spec_data["extras"]["original_count_channel_bins"]
            new_count_bins = self._loaded_spec_data["count_channel_bins"]
        if (axis == "photon") or (axis == "photon_and_count"):
            old_photon_bins = self._loaded_spec_data["extras"]["orignal_photon_channel_bins"]
            new_photon_bins = self._loaded_spec_data["photon_channel_bins"]
        return old_count_bins, new_count_bins, old_photon_bins, new_photon_bins

    def _rebin_srm(self, axis="count"):
        """ Rebins the photon and/or count channels of the spectral response matrix (SRM) if needed.

        Note: If the instrument has a spatial aspect and effective information is present (e.g.,
        NuSTAR from its ARF file) then this method should be overwritten in the instrument
        specific loader in order to rebin the redistribution matrix and effective area separately
        before re-construction the new SRM.

        Parameters
        ----------
        matrix : 2d array
                Spectral response matrix.

        old_count_bins, new_count_bins : 1d arrays
                The old count channel binning and the new binning to be for the spectral response matrix columns (sum columns).

        old_photon_bins, new_photon_bins : 1d arrays
                The old photon channel binning and the new binning to be for the spectral response matrix columns (average rows).

        axis : string
                Define what \'axis\' the binning should be applied to. E.g., \'photon\', \'count\', or \'photon_and_count\'.

        Returns
        -------
        The rebinned 2d spectral response matrix.
        """
        old_count_bins, new_count_bins, old_photon_bins, new_photon_bins = self._channel_bin_info(axis)
        matrix = self._loaded_spec_data["srm"]
        return self._rebin_rmf(matrix, old_count_bins=old_count_bins, new_count_bins=new_count_bins, old_photon_bins=old_photon_bins, new_photon_bins=new_photon_bins, axis="count")

    def __getitem__(self, item):
        """Index the entries in `_loaded_spec_data`"""
        return self._loaded_spec_data[item]

    def __setitem__(self, item, new_value):
        """Allows entries in `_loaded_spec_data` to be changed."""
        self._loaded_spec_data[item] = new_value

    def __call__(self):
        """When the class is called (n=NustarLoader()->n()) then `_loaded_spec_data` is returned."""
        return self._loaded_spec_data

    def __repr__(self):
        """String representation of `_loaded_spec_data`."""
        return str(self._loaded_spec_data)


# Instrument specific data loaders
#    As long as these loaders get the spectral data to fit into the correct dictionary form and assigned to self._loaded_spec_data then
#    they should work but they can also overwrite the _rebin_srm(self, axis="count") method if the SRM rebinning is instrument specific.
#       The benefit here is that the class can have other methods/properties/setters (like time selection for STIX/RHESSI;e.g.,
#    .select_time(new_time)?) which can be accessed at the user level easily when fitting through the loaded_spec_data attribute
#    (e.g., .loaded_spec_data["spectrum1"].select_time(new_time)).


class NustarLoader(InstrumentBlueprint):
    """
    Loader specifically for NuSTAR spectral data.

    NustarLoader Specifics
    ----------------------
    Changes how the spectral response matrix (SRM) is rebinned. The NuSTAR SRM is constructed from
    the effective areas (EFs) and redistribution matrix (RM) and so the EFs and RM are rebinned
    separately then used to construct the rebinned SRM.

    Superclass Override: _rebin_srm()

    Attributes
    ----------
    _construction_string : string
            String to show how class was constructed.

    _loaded_spec_data : dict
            Instrument loaded spectral data.
    """
    __doc__ += InstrumentBlueprint._UNIVERSAL_DOC_

    def __init__(self, pha_file, arf_file=None, rmf_file=None, srm_custom=None, custom_channel_bins=None, custom_photon_bins=None, **kwargs):
        """Construct a string to show how the class was constructed (`_construction_string`) and set the `_loaded_spec_data` dictionary attribute."""

        self._construction_string = f"NustarLoader(pha_file={pha_file},arf_file={arf_file},rmf_file={rmf_file},srm_custom={srm_custom},custom_channel_bins={custom_channel_bins},custom_photon_bins={custom_photon_bins},**{kwargs})"
        self._loaded_spec_data = self._load1spec(pha_file, f_arf=arf_file, f_rmf=rmf_file, srm=srm_custom, channel_bins=custom_channel_bins, photon_bins=custom_photon_bins)

    def _load1spec(self, f_pha, f_arf=None, f_rmf=None, srm=None, channel_bins=None, photon_bins=None):
        """ Loads all the information in for a given spectrum.

        Parameters
        ----------
        f_pha, f_arf, f_rmf : string
                Filenames for the relevant spectral files. If f_arf, f_rmf are None it is assumed
                that these are in the same directory with same filename as the PHA file but with
                extensions '.arf' and '.rmf', respectively.
                Default of f_arf, f_rmf: None

        srm : 2d array
                User defined spectral response matrix. This is accepted over the SRM created from any
                ARF and RMF files given.
                Default: None

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
                                                                                "arf.file":f_arf,
                                                                                "arf.e_lo":e_lo_arf,
                                                                                "arf.e_hi":e_hi_arf,
                                                                                "arf.effective_area":eff_area,
                                                                                "rmf.file":f_rmf,
                                                                                "rmf.e_lo":e_lo_rmf,
                                                                                "rmf.e_hi":e_hi_rmf,
                                                                                "rmf.ngrp":ngrp,
                                                                                "rmf.fchan":fchan,
                                                                                "rmf.nchan":nchan,
                                                                                "rmf.matrix":matrix,
                                                                                "rmf.redistribution_matrix":redist_m}
                                                                     }.
        """

        # what files might be needed (for NuSTAR)
        f_arf = f_pha[:-3]+"arf" if type(f_arf) == type(None) else f_arf
        f_rmf = f_pha[:-3]+"rmf" if type(f_rmf) == type(None) else f_rmf

        # need effective exposure and energy binning since likelihood works on counts, not count rates etc.
        _, counts, eff_exp = io._read_pha(f_pha)

        # now calculate the SRM or use a custom one if given
        if type(srm) == type(None):

            # if there is an ARF file load it in
            if os_path.isfile(f_arf):
                e_lo_arf, e_hi_arf, eff_area = io._read_arf(f_arf)

            # if there is an RMF file load it in and convert to a redistribution matrix
            if os_path.isfile(f_rmf):
                e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m = self._load_rmf(f_rmf)

            srm = nu_spec.make_srm(rmf_matrix=redist_m, arf_array=eff_area)
        else:
            e_lo_arf, e_hi_arf, eff_area = None, None, None
            e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m = None, None, None, None, None, None, None

        channel_bins = self._calc_channel_bins(e_lo_rmf, e_hi_rmf) if type(channel_bins) == type(None) else channel_bins
        channel_binning = np.diff(channel_bins).flatten()

        phot_channels = channel_bins if type(photon_bins) == type(None) else photon_bins
        phot_binning = np.diff(phot_channels).flatten()

        # get the count rate information
        count_rate, count_rate_error = nu_spec.flux_cts_spec(f_pha, bin_size=channel_binning)

        # what spectral info you want to know from this observation
        return {"photon_channel_bins": phot_channels,
                "photon_channel_mids": np.mean(phot_channels, axis=1),
                "photon_channel_binning": phot_binning,
                "count_channel_bins": channel_bins,
                "count_channel_mids": np.mean(channel_bins, axis=1),
                "count_channel_binning": channel_binning,
                "counts": counts,
                "count_error": np.sqrt(counts),
                "count_rate": count_rate,
                "count_rate_error": count_rate_error,
                "effective_exposure": eff_exp,
                "srm": srm,
                "extras": {"pha.file": f_pha,
                           "arf.file": f_arf,
                           "arf.e_lo": e_lo_arf,
                           "arf.e_hi": e_hi_arf,
                           "arf.effective_area": eff_area,
                           "rmf.file": f_rmf,
                           "rmf.e_lo": e_lo_rmf,
                           "rmf.e_hi": e_hi_rmf,
                           "rmf.ngrp": ngrp,
                           "rmf.fchan": fchan,
                           "rmf.nchan": nchan,
                           "rmf.matrix": matrix,
                           "rmf.redistribution_matrix": redist_m}
                }  # this might make it easier to add different observations together

    def _load_rmf(self, rmf_file):
        """ Extracts all information, mainly the redistribution matrix ([counts/photon]) from a given RMF file.

        Parameters
        ----------
        rmf_file : string
                The file path and name of the RMF file.

        Returns
        -------
        The lower/higher photon bin edges (e_lo_rmf, e_hi_rmf), the number of counts channels activated by each photon channel (ngrp),
        starting indices of the count channel groups (fchan), number counts channels from each starting index (nchan), the coresponding
        counts/photon value for each count and photon entry (matrix), and the redistribution matrix (redist_m: with rows of photon channels,
        columns of counts channels, and in the units of counts/photon).
        """

        e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix = io._read_rmf(rmf_file)
        fchan_array = nu_spec.col2arr_py(fchan)
        nchan_array = nu_spec.col2arr_py(nchan)
        redist_m = nu_spec.vrmf2arr_py(data=matrix,
                                       n_grp_list=ngrp,
                                       f_chan_array=fchan_array,
                                       n_chan_array=nchan_array)  # 1.5 s of the total 2.4 s (1spec) is spent here

        return e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m

    def _calc_channel_bins(self, e_low, e_hi):
        """ Calculates the count channel bins from the given rmf files. Assumes that the photon and count channel bins are the same.

        Parameters
        ----------
        e_low : 1d array
                Array of the lower bounds of all the channel bins.

        e_hi : 1d array
                Array of the higher bounds of all the channel bins.

        Returns
        -------
        None if no e_low or e_hi is given or 2d array where each row is the lower and higher bound of that bin.
        """
        if (e_low is None) or (e_hi is None):
            print("If no rmf/arf files are given and a custom srm is provided, please provide the custom_channel_bins.\nE.g., custom_channel_bins=[[1,2],[2,3],...]")
            return None
        else:
            return np.stack((e_low, e_hi), axis=-1)

    def _rebin_srm(self, axis="count"):
        """ Rebins the photon and/or count channels of the spectral response matrix by rebinning the redistribution matrix and the effective area array.

        Parameters
        ----------
        axis : string
                Define what \'axis\' the binning should be applied to. E.g., \'photon\', \'count\', or \'photon_and_count\'.

        Returns
        -------
        The rebinned 2d spectral response matrix.
        """
        old_count_bins, new_count_bins, old_photon_bins, new_photon_bins = self._channel_bin_info(axis)

        old_rmf = self._loaded_spec_data["extras"]["rmf.redistribution_matrix"]
        old_eff_area = self._loaded_spec_data["extras"]["arf.effective_area"]

        # checked with ftrbnrmf
        new_rmf = self._rebin_rmf(matrix=old_rmf,
                                  old_count_bins=old_count_bins,
                                  new_count_bins=new_count_bins,
                                  old_photon_bins=old_photon_bins,
                                  new_photon_bins=new_photon_bins,
                                  axis=axis)

        # average eff_area, checked with ftrbnarf
        new_eff_area = rebin_any_array(data=old_eff_area, old_bins=old_photon_bins, new_bins=new_photon_bins, combine_by="mean") if (axis != "count") else old_eff_area
        return nu_spec.make_srm(rmf_matrix=new_rmf, arf_array=new_eff_area)


class RhessiLoader(InstrumentBlueprint):
    """
    Loader specifically for RHESSI spectral data.

    RhessiLoader Specifics
    ----------------------
    Has methods to plot time series and perform time selection on the data. A background time can be added or removed and can fit
    the event data with the model+background (recommended) or fit a model to data-background using the `data2data_minus_background`
    setter with False or True, respectively.

    We assume that the background (if calculated) and the event emission is calculated from the same sized area. If this is not
    true then the background effective exposure time should be multiplied by the ratio of the background area and the event area.
    I.e., background_effective_exposure * (background_area / event_area) as described in [1]. This may be automated (16/03/2022).

    [1] https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html
    ----------
    """

    def __init__(self, spectrum_fn, srm_fn, **kwargs):
        '''
        Spectrum and SRM files are both required: attenuator state change times
            are in the spectrum file,
            and the state determines which SRM will be used.
        '''
        self._construction_string = (
            f"RhessiLoader(spectrum_fn={spectrum_fn}, "
            f"srm_fn={srm_fn},"
            f"**{kwargs})"
        )
        self._systematic_error = 0
        self.load_prepare_spectrum_srm(spectrum_fn, srm_fn)
        self._start_background_time, self._end_background_time = None, None

    @property
    def systematic_error(self):
        return self._systematic_error

    @systematic_error.setter
    def systematic_error(self, err):
        self._systematic_error = err
        self._update_event_data_with_times()

    def load_prepare_spectrum_srm(self, spectrum_fn, srm_fn):
        """ Loads all the information in for a given spectrum.

        Parameters
        ----------
        spectrum_fn, srm_fn : string
            Filenames for the spectrum and SRM .fits files from OSPEX.

        Returns
        -------
        Standard sunxspex dictionary (see docs)
        """
        # Load spectrum & SRM
        self._spectrum = (spec := rhes_spec.load_spectrum(spectrum_fn))
        self._attenuator_state_info = spec.pop('attenuator_state_info')
        self._srm = (srm := rhes_spec.load_srm(srm_fn))

        # make sure the SRM will only produce counts to match the data
        counts_indices_which_match = np.where(
            (spec['channel_bins'][0, 0] <=  srm['channel_bins'][ :,  0]) &
            ( srm['channel_bins'][:, 1] <= spec['channel_bins'][-1, -1])
        )[0]
        srm['srm_options'] = {
            state: matrix[:, counts_indices_which_match]
            for (state, matrix) in srm['srm_options'].items()
        }

        photon_bins = srm['photon_bins']
        photon_binning = np.diff(photon_bins).flatten()
        channel_bins = spec['channel_bins']

        # default is no background and all data is the spectrum to be fitted
        self.start_data_time, self.end_data_time = (
            spec['time_bins'][0, 0],
            spec['time_bins'][-1, -1]
        )
        # avoid SRM setting issues
        self._start_event_time, self._end_event_time =\
            self.start_data_time, self.end_data_time

        # Assume unattenuated SRM until spectroscopy interval is specified
        UNATTENUATED_STATE = 0
        self._loaded_spec_data = {
            'photon_channel_bins': photon_bins,
            'photon_channel_binning': photon_binning,
            'photon_channel_mids': photon_bins[:, 0] + photon_binning/2,
            'count_channel_bins': channel_bins,
            'count_channel_binning': (ch_de := np.diff(channel_bins, axis=1).flatten()),
            'count_channel_mids': channel_bins[:, 0] + ch_de/2,
            'srm': srm['srm_options'][UNATTENUATED_STATE],
            'extras': dict()
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
            _ = no_srm.pop('srm', None)
            _ = no_srm.pop('extras', None)
            self._loaded_spec_data.update(no_srm)
            return

        scaled_bg = (
            self._loaded_spec_data["extras"]["background_rate"] *
            self._loaded_spec_data["effective_exposure"] *
            self._loaded_spec_data["count_channel_binning"]
        )
        new_cts = self._original_data["counts"] - scaled_bg

        scaled_bg_err = (
            self._loaded_spec_data["extras"]["background_count_error"] *
            self._loaded_spec_data["effective_exposure"] /
            self._loaded_spec_data["extras"]["background_effective_exposure"]
        )
        new_cts_err = np.sqrt(self._original_data["count_error"]**2 + scaled_bg_err**2)

        new_rate = self._original_data["count_rate"] - self._loaded_spec_data["extras"]["background_rate"]
        new_rate_err = np.sqrt(self._original_data["count_rate_error"]**2 + self._loaded_spec_data["extras"]["background_rate_error"]**2)

        self._loaded_spec_data.update({"counts": new_cts,
                                       "count_error": new_cts_err,
                                       "count_rate": new_rate,
                                       "count_rate_error": new_rate_err})
        self._loaded_spec_data["extras"]["counts=data-bg"] = True

    def _data_time_select(self, stime, full_data, etime):
        """ Index and return data in time range stime<=data<=etime.

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
            return full_data[np.where(self._spectrum['time_bins'][:, 1] <= etime)]
        elif etime is None:
            return full_data[np.where(stime <= self._spectrum['time_bins'][:, 0])]
        return full_data[np.where((stime <= self._spectrum['time_bins'][:, 0]) & (self._spectrum['time_bins'][:, 1] <= etime))]

    def _update_srm_state(self):
        '''
        Updates SRM state (attenuator state) given the event times.
        If the times span attenuator states, throws an error.
        '''
        start_time, end_time = self._start_event_time, self._end_event_time
        change_times = self._attenuator_state_info['change_times']
        for t in change_times:
            if start_time <= t <= end_time:
                warnings.warn(
                    f"\ndo not update event times to ({start_time}, {end_time}): "
                    "covers attenuator state change. Don't trust this fit!"
                )

        new_att_state = 0
        for i in range(len(self._attenuator_state_info['states']) - 1):
            state = self._attenuator_state_info['states'][i]
            if change_times[i] < start_time and end_time < change_times[i+1]:
                new_att_state = state
                break

        self._loaded_spec_data['srm'] = self._srm['srm_options'][new_att_state].astype(float)

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
                    stime=self._start_event_time,
                    full_data=self._spectrum['counts'],
                    etime=self._end_event_time
                ),
                axis=0
            )
        )
        # do not assume Poisson error
        self._loaded_spec_data["count_error"] = (
            err := np.sqrt(np.sum(
                self._data_time_select(
                    stime=self._start_event_time,
                    full_data=self._spectrum['counts_err'],
                    etime=self._end_event_time
                )**2)
            )
        )

        if np.any(np.atleast_1d(self.systematic_error) > 0):
            self._loaded_spec_data['count_error'] = (err := np.sqrt(
                err**2 + (self._systematic_error * cts)**2
            ))

        livetimes = np.mean(
            self._data_time_select(
                stime=self._start_event_time,
                full_data=self._spectrum['livetime'],
                etime=self._end_event_time
            ),
            axis=0
        )
        actual_first_bin = self._data_time_select(
            stime=self._start_event_time,
            full_data=self._spectrum['time_bins'][:, 0],
            etime=self._end_event_time
        )[0]
        actual_last_bin = self._data_time_select(
            stime=self._start_event_time,
            full_data=self._spectrum['time_bins'][:, 1],
            etime=self._end_event_time
        )[-1]
        self._loaded_spec_data["effective_exposure"] = np.diff([actual_first_bin, actual_last_bin])[0].to_value("s") * livetimes

        # calculate new count rates and errors
        effe = self._loaded_spec_data['effective_exposure']
        de = self._loaded_spec_data["count_channel_binning"]
        self._loaded_spec_data["count_rate"] = cts / effe / de
        self._loaded_spec_data["count_rate_error"] = err / effe / de

    def update_event_times(self, ta, tb):
        ''' Update start, end event times '''
        if not isinstance(ta, Time) or not isinstance(tb, Time):
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
            keyz = list(self._loaded_spec_data['extras'].keys())
            for key in keyz:
                if 'background' in key:
                    del self._loaded_spec_data["extras"][key]
            return

        # sum counts over time range
        self._loaded_spec_data["extras"]["background_counts"] = (cts := np.sum(
            self._data_time_select(
                stime=self._start_background_time,
                full_data=self._spectrum['counts'],
                etime=self._end_background_time
            ),
            axis=0
        ))

        # TODO: this is an assumption of Poisson statistics (not necessarily correct)
        self._loaded_spec_data["extras"]["background_count_error"] = (cts_err := np.sqrt(self._loaded_spec_data["extras"]["background_counts"]))

        # isolate livetimes and time binning
        livetimes = np.mean(
            self._data_time_select(
                stime=self._start_background_time,
                full_data=self._spectrum['livetime'],
                etime=self._end_background_time
            ),
            axis=0
        )
        actual_first_bin = self._data_time_select(
            stime=self._start_background_time,
            full_data=self._spectrum['time_bins'][:, 0],
            etime=self._end_background_time
        )[0]
        actual_last_bin = self._data_time_select(
            stime=self._start_background_time,
            full_data=self._spectrum['time_bins'][:, 1],
            etime=self._end_background_time
        )[-1]
        self._loaded_spec_data["extras"]["background_effective_exposure"] = (
            effe := ((actual_last_bin - actual_first_bin).to_value("s") * livetimes)
        )

        # calculate new count rates and errors
        de = self._loaded_spec_data['count_channel_binning']
        self._loaded_spec_data["extras"]["background_rate"] = cts / effe / de
        self._loaded_spec_data["extras"]["background_rate_error"] = cts_err / effe / de

    def update_background_times(self, ta: Time, tb: Time):
        ''' Update start and end background times.'''
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
        """ Try to determine a nice tick separation for time axis on the lightcurve.

        Returns
        -------
        `mdates.MinuteLocator`.
        """
        obs_dt = (self.end_data_time  - self.start_data_time).to_value("s")
        if obs_dt > 3600*12:
            return mdates.MinuteLocator(byminute=[0], interval=1)
        elif 3600*3 < obs_dt <= 3600*12:
            return mdates.MinuteLocator(byminute=[0, 30], interval=1)
        elif 3600 < obs_dt <= 3600*3:
            return mdates.MinuteLocator(byminute=[0, 20, 40], interval=1)
        elif 3600*0.5 < obs_dt <= 3600:
            return mdates.MinuteLocator(byminute=[0, 10, 20, 30, 40, 50], interval=1)
        elif 600 < obs_dt <= 3600*0.5:
            return mdates.MinuteLocator(byminute=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], interval=1)
        elif 240 < obs_dt <= 600:
            return mdates.MinuteLocator(interval=2)
        else:
            return mdates.SecondLocator(bysecond=[0, 20, 40], interval=1)

    def _rebin_lc(self, arr, clump_bins):
        """ Combines array elements in groups of `clump_bins`.

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
        """ Combines bin array elements in groups of `clump_bins`.

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
        _t_to_clump, _endt_to_clump = times[0::clump_bins], times[clump_bins-1::clump_bins]
        _clumped_start_ts = _t_to_clump[:, 0]
        _clumped_end_ts = np.concatenate((_t_to_clump[1:, 0], [_endt_to_clump[-1, -1]]))
        return np.column_stack((_clumped_start_ts, _clumped_end_ts))

    def lightcurve(self, energy_ranges=None, axes=None, rebin_time=1):
        """ Creates a RHESSI lightcurve.

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

        Examples
        --------
        # use the class to load in data
        ar = RhessiLoader(pha_file=spec_file, srm_file=srm_file)

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
        if energy_ranges is None:
            energy_ranges = [[self._loaded_spec_data["count_channel_bins"][0, 0], self._loaded_spec_data["count_channel_bins"][-1, -1]]]
        elif len(np.shape(energy_ranges)) == 1:
            energy_ranges = [energy_ranges]
        elif len(np.shape(energy_ranges)) == 0:
            print("The `energy_ranges` input should be a range of two energy values (e.g., [4,8] meaning 4-8 keV inclusive) or a list of these ranges.")
            return

        ax = axes or plt.gca()
        font_size = plt.rcParams['font.size']

        orig_bins = self._spectrum['time_bins']
        flat_times = Time(np.concatenate((orig_bins[:, 0], [orig_bins[-1, -1]])))

        if rebin_time > 1:
            flat_times = self._rebin_ts(flat_times, rebin_time)
        time_mids = flat_times[:-1] + (flat_times[1:] - flat_times[:-1])/2

        # plot each energy range
        time_binning = (flat_times[1:] - flat_times[:-1]).to_value('s')
        for er in energy_ranges:
            i = np.where((self._loaded_spec_data["count_channel_bins"][:, 0] >= er[0]) & (self._loaded_spec_data["count_channel_bins"][:, -1] <= er[-1]))
            e_range_cts = np.sum(self._spectrum['counts'][:, i].reshape((len(time_binning), -1)), axis=1)

            if rebin_time > 1:
                e_range_cts = self._rebin_lc(e_range_cts, rebin_time)
                time_binning = self._rebin_lc(time_binning, rebin_time)

            e_range_ctr, e_range_ctr_err = e_range_cts/time_binning, np.sqrt(e_range_cts)/time_binning
            lc = e_range_ctr
            p = ax.stairs(lc, flat_times.datetime, label=f"{er[0]}$-${er[-1]} keV")
            ax.errorbar(
                x=time_mids.datetime,
                y=e_range_ctr,
                yerr=e_range_ctr_err,
                c=p.get_edgecolor(),
                ls=""
            )  # error bar in middle of the bin

        fmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(self._mdates_minute_locator())

        ax.set_yscale("log")
        ax.set_xlabel(f"Time (Start Time: {self.start_data_time})")
        ax.set_ylabel("Counts s$^{-1}$")

        ax.set_title("RHESSI Lightcurve")
        plt.legend(fontsize=font_size-5)

        # plot background time range if there is one
        _y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.95  # stop region label overlapping axis spine
        if (self._start_background_time is not None) and (self._end_background_time is not None):
            ax.axvspan(*Time([self._start_background_time, self._end_background_time]).datetime, alpha=0.1, color='orange')
            ax.annotate("BG", (self._start_background_time.datetime, _y_pos), color='orange', va="top", size=font_size-2)

        # plot event time range
        if hasattr(self, "_start_event_time") and hasattr(self, "_end_event_time"):
            ax.axvspan(self._start_event_time.datetime, self._end_event_time.datetime, alpha=0.1, color='purple')
            ax.annotate("Evt", (self._start_event_time.datetime, _y_pos), color='purple', va="top", size=font_size-2)

        return ax

    def spectrogram(self, axes=None, rebin_time=1, rebin_energy=1, **kwargs):
        """ Creates a RHESSI spectrogram.

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

        Examples
        --------
        # use the class to load in data
        ar = RhessiLoader(pha_file=spec_file, srm_file=srm_file)

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
        ax = axes or plt.gca()

        _def_fs = plt.rcParams['font.size']

        # get cts/s, and times and energy bin ranges
        time_binning = np.array([dt.to_value("s") for dt in np.diff(self._spectrum['time_bins']).flatten()])
        e_range_cts = self._spectrum['counts']
        t = Time(np.concatenate((self._spectrum['time_bins'][:, 0], [self._spectrum['time_bins'][-1, -1]])))

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
        opts = {'cmap': 'plasma'}
        opts.update(kwargs)
        ax.pcolormesh(t, ebins, spect, **opts)

        fmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(self._mdates_minute_locator())

        ax.set_xlabel(f"Time (Start Time: {self._start_event_time})")
        ax.set_ylabel("Energy [keV]")

        ax.set_title("RHESSI Spectrogram [Counts s$^{-1}$]")

        # change event and background start and end times from astropy dates to matplotlib dates
        start_evt_time, end_evt_time, start_bg_time, end_bg_time = Time(
            [self._start_event_time, self._end_event_time, self._start_background_time, self._end_background_time]
        ).datetime

        # plot background time range if there is one
        _y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.95  # stop region label overlapping axis spine
        etop = ebins.max()
        if (self._start_background_time is not None) and (self._end_background_time is not None):
            ax.hlines(y=etop, xmin=start_bg_time, xmax=end_bg_time, alpha=0.9, color='orange', capstyle='butt', lw=10)
            ax.annotate("BG", (start_bg_time, _y_pos), color='orange', va="top", size=_def_fs-2)

        # plot event time range
        ax.hlines(y=etop, xmin=start_evt_time, xmax=end_evt_time, alpha=0.9, color='#F37AFF', capstyle='butt', lw=10)
        ax.annotate("Evt", (start_evt_time, _y_pos), color='#F37AFF', va="top", size=_def_fs-2)

        return ax

    def select_time(self, start=None, end=None, background=False):
        """ Provides method to set start and end time of the event or background in one line.

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

        Examples
        --------
        # use the class to load in data
        ar = RhessiLoader(pha_file=spec_file, srm_file=srm_file)

        # define a background range if we like; equivalent to doing ar.start_background_time = "2002-10-05T10:38:32" and ar.end_background_time = "2002-10-05T10:40:32"
        ar.select_time(start="2002-10-05T10:38:32", end="2002-10-05T10:40:32", background=True)

        # change the event time range to something other than the full time range; equivalent to doing ar.start_event_time = "2002-10-05T10:41:20" and ar.end_event_time = "2002-10-05T10:42:24"
        ar.select_time(start="2002-10-05T10:41:20", end="2002-10-05T10:42:24")

        """
        self.__warn = False if (type(start) != type(None)) and (type(end) != type(None)) else True

        if background:
            self.start_background_time, self.end_background_time = start, end
        else:
            self.start_event_time, self.end_event_time = start, end


class CustomLoader(InstrumentBlueprint):
    """
    Loader specifically for custom spectral data.

    CustomLoader Specifics
    ----------------------
    Offers a user a way to provide custom spectral data as long as they can organise it in the accepted dictionary
    form. Accepted dictionary format: {"photon_channel_bins":Photon-space Bins (e.g., [keV,keV],[keV,keV],...]),
                                       "photon_channel_mids":Photon-space Bin Mid-points (e.g., [keV,...]),
                                       "photon_channel_binning":Photon-space Binwidths (e.g., [keV,...]),
                                       "count_channel_bins":Count-space Bins (e.g., [keV,keV],[keV,keV],...]),
                                       "count_channel_mids":Count-space Bin Mid-points (e.g., [keV,...]),
                                       "count_channel_binning":Count-space Binwidths (e.g., [keV,...]),
                                       "counts":counts (e.g., cts),
                                       "count_error":count error (e.g., sqrt(cts)),
                                       "count_rate":Count Rate (e.g., cts/keV/s),
                                       "count_rate_error":Count Rate Error for `count_rate`,
                                       "effective_exposure":Effective Exposure (e.g., s),
                                       "srm":Spectral Response Matrix (e.g., cts/ph * cm^2),
                                       "extras":{"any_extra_info":or_empty_dict}
                                       },
    where the "count_channel_bins" and "counts" are essential entries with the rest being assigned a default if
    they are not given.

    Parameters
    ----------
    spec_data_dict : dict
            Dictionary for custom spectral data. Essential entries are 'count_channel_bins'
            and 'counts'.

    Attributes
    ----------
    _construction_string : string
            String to show how class was constructed.
    _loaded_spec_data : dict
            Custom loaded spectral data.
    """

    def __init__(self, spec_data_dict, **kwargs):
        """Construct a string to show how the class was constructed (`_construction_string`) and set the `_loaded_spec_data` dictionary attribute."""
        self._construction_string = f"CustomLoader({spec_data_dict},**{kwargs})"

        # needed keys
        ess_keys = ["count_channel_bins",
                    "counts"]
        non_ess_keys = ["photon_channel_bins",
                        "photon_channel_mids",
                        "photon_channel_binning",
                        "count_channel_mids",
                        "count_channel_binning",
                        "count_error",
                        "count_rate",
                        "count_rate_error",
                        "effective_exposure",
                        "srm",
                        "extras"]

        # check essential keys are given
        essentials_not_present = set(ess_keys)-set(list(spec_data_dict.keys()))
        assert len(essentials_not_present) == 0, f"Essential dict. entries are not present: {essentials_not_present}"

        # check non-essential keys are given, if not then defaults are 1s
        nonessentials_not_present = set(non_ess_keys)-set(list(spec_data_dict.keys()))
        _def = self._nonessential_defaults(nonessentials_not_present, spec_data_dict["count_channel_bins"], spec_data_dict["counts"])
        _def.update(spec_data_dict)

        self._loaded_spec_data = _def

    def _nonessential_defaults(self, nonessential_list, count_channels, counts):
        """ To return a dictionary of all "non-essential" `_loaded_spec_data` values.

        These then get overwritten with the user provided dictionary in __init__.

        Parameters
        ----------
        nonessential_list : list of strings
                List of the "non-essential" entries in the custom `_loaded_spec_data`
                dict that need defaults.
        count_channels : 2d np.array, shape (N,2)
                Array of the count channel bin edges.
        counts : np.array, length N
                Array of counts data.

        Returns
        -------
        Defaults of all "non-essential" `_loaded_spec_data` values as a dictionary if
        needed, else an empty dictionary.
        """
        if len(nonessential_list) > 0:
            _count_length_default = np.ones(len(count_channels))
            _chan_mids_default = np.mean(count_channels, axis=1)
            defaults = {"photon_channel_bins": count_channels,
                        "photon_channel_mids": _chan_mids_default,
                        "photon_channel_binning": _count_length_default,
                        "count_channel_mids": _chan_mids_default,
                        "count_channel_binning": _count_length_default,
                        "count_error": _count_length_default,
                        "count_rate": counts,
                        "count_rate_error": _count_length_default,
                        "effective_exposure": 1,
                        "srm": np.identity(len(counts)),
                        "extras": {}}
            return defaults
        else:
            return {}


def rebin_any_array(data, old_bins, new_bins, combine_by="sum"):
    """ Takes any array of data in old_bins space and rebins along data array axis==0 to have new_bins.

    Can specify how the bins are combined.

    Parameters
    ----------
    data, old_bins, new_bins : np.array
            Array of the data, current bins (for data axis==0), and new bins (for data axis==0).
            Need len(data)==len(old_bins).

    combine_by : string
            Defines how to combine multiple bins along axis 0. E.g., "sum" adds the data, "mean" averages
            the data, and "quadrature" sums the data in quadrature.
            Default: "sum"

    Returns
    -------
    The new bins and the corresponding grouped counts.
    """
    new_binned_data = []
    for nb in new_bins:
        # just loop through new bins and bin data from between new_bin_lower<=old_bin_lowers and old_bin_highers<new_bin_higher
        if combine_by == "sum":
            new_binned_data.append(np.sum(data[np.where((nb[0] <= old_bins[:, 0]) & (nb[-1] >= old_bins[:, -1]))], axis=0))
        elif combine_by == "mean":
            new_binned_data.append(np.mean(data[np.where((nb[0] <= old_bins[:, 0]) & (nb[-1] >= old_bins[:, -1]))], axis=0))
        elif combine_by == "quadrature":
            new_binned_data.append(np.sqrt(np.sum(data[np.where((nb[0] <= old_bins[:, 0]) & (nb[-1] >= old_bins[:, -1]))]**2, axis=0)))
    return np.array(new_binned_data)
