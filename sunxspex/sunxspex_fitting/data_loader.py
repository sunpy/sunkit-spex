"""
The following code Handles how all data is loaded in from the individual instrument loaders and how the data is used (e.g., rebinning, etc.).
"""

import numpy as np
from copy import deepcopy
from astropy.io import fits

from .parameter_handler import _make_into_list, isnumber # sunxspex.sunxspex_fitting.parameter_handler
from . import instruments as inst # sunxspex.sunxspex_fitting.instruments

__all__ = ["LoadSpec"]


class LoadSpec:
    """
    This class's job is to load in spectral file(s), obtain count spectra, and calculate/store the info for fitting.

    This class holds all data required to perform spectral fitting as well as methods used for appropriately handling
    or changing the loaded data.

    Parameters
    ----------
    *args : dict
            Dictionaries for custom data to be passed to `sunxspex.sunxspex_fitting.instruments.CustomLoader`.
            These will be added before any instrument file entries from `pha_file`.

    pha_file : string or list of strings
            The PHA file or list of PHA files for the spectrum to be loaded.

    arf_file, rmf_file : string or list of strings
            The ARF and RMF files associated with the PHA file(s). If none are given (e.g, with
            NuSTAR data) it is assumed that these are in the same directory with same filename
            as the PHA file(s) but with extensions '.arf' and '.rmf', respectively.

    srm_file : string
            The file that contains the spectral response matrix for the given spectrum.

    srm_custom : 2d array
            User defined spectral response matrix. This is accepted over the SRM created from any
            ARF and RMF files given.

    custom_channel_bins : 2d array
            User defined channel bins for the columns of the SRM matrix.
            E.g., custom_channel_bins=[[1,1.5],[1.5,2],...]

    **kwargs :
            Any kwargs will get passed to all instrument loaders.

    Properties
    ----------
    rebin : list/array
            Returns the new energy bins of the data (self._rebinned_edges), None if the has not been rebinned (has setter).

    undo_rebin : None
            Has the code that uses self._undo_rebin to undo the rebinning for spectra (has setter).

    Setters
    -------
    rebin : int, {specID:int}, {"all":int}
            Minimum number of counts in each bin. Changes data but saves original in "extras" key
            in loaded_spec_data attribute.

    undo_rebin : int, str of specID, "all"
            Undo the rebinning. Move the original data from "extras" in loaded_spec_data attribute
            back to main part of the dict and set self._undo_rebin.

    Methods
    -------
    group : channel_bins (array (n,2)), counts (array (n)), group_min (int)
            Groups bins so they have at least a `group_min` number of counts.

    group_pha_finder : channels (array (n,2)), counts (array (n)), group_min (int), print_tries (bool)
            Check, incrementally from a minimum number, what group minimum is needed to leave no counts unbinned
            after rebinning. Returns binned channels and the minimum group number if one exists.

    group_spec : spectrum (str), group_min (int), _orig_in_extras (bool)
            Returns new bins and new binned counts, count errors, and effective exposures for a given spectrum and
            minimun bin gorup number.

    Attributes
    ----------
    instruments : dict
            Spectrum identifiers as keys with the spectrum's instrument as a string for values.

    intrument_loaders : dict
            Dictionary with keys of the supported instruments and values of their repsective loaders.

    loaded_spec_data : dict
            All loaded spectral data.


    _construction_string : string
            String to be returned from __repr__() dunder method.

    _rebinned_edges : dict
            Dictionary of energy bins if they have been rebinned for each loaded spectrum. Set in rebin().

    _rebin_setting : dict
            Dictionary of rebin setting for each loaded spectrum. Set in rebin().

    _undo_rebin : string
            Indicates the spectral rebinning to be undone. E.g., 'all', 'spectrumN', or None. Set in undo_rebin().

    Examples
    --------
    # load in 2 spectra, rebin the count channels to have a minimum of 10 counts then undo that rebinning
    s = LoadSpec(pha_file=['filename1.pha', 'filename2.pha'],
                    arf_file=['filename1.arf', 'filename2.arf'],
                    rmf_file=['filename1.rmf', 'filename2.rmf'])
    s.rebin = 10
    s.undo_rebin
    """
    def __init__(self, *args, pha_file=None, arf_file=None, rmf_file=None, srm_file=None, srm_custom=None, custom_channel_bins=None, **kwargs):
        """Construct a string to show how the class was constructed (`_construction_string`) and set the `loaded_spec_data` dictionary attribute."""

        self._construction_string = f"LoadSpec(*{args},pha_file={pha_file},arf_file={arf_file},rmf_file={rmf_file},srm_file={srm_file},srm_custom={srm_custom},custom_channel_bins={custom_channel_bins}, **{kwargs})"

        # from sunxspex.sunxspex_fitting.instruments import * gives us the instrument specific loaders, keys should match up to the "TELESCOP" header entry in spec file
        self.instrument_loaders = {"NuSTAR":inst.NustarLoader, "SOLO/STIX":inst.StixLoader, "RHESSI":inst.RhessiLoader}

        pha_file, arf_file, rmf_file, srm_file, srm_custom, custom_channel_bins, instruments = self._sort_files(pha_file=pha_file,
                                                                                                                arf_file=arf_file,
                                                                                                                rmf_file=rmf_file,
                                                                                                                srm_file=srm_file,
                                                                                                                srm_custom=srm_custom,
                                                                                                                custom_channel_bins=custom_channel_bins)
        # get ready to load multiple spectra if needed
        num_of_files, num_of_custom = len(pha_file), len(args)
        self.loaded_spec_data, self.instruments = {}, {}
        for s in range(num_of_files+num_of_custom):
            if s<num_of_custom:
                # if a custom dict is given or if the user has set up the instrument loader class themselves and just wants to pass it straight in
                if type(args[s])==dict:
                    self.loaded_spec_data["spectrum"+str(s+1)] = inst.CustomLoader(args[s], **kwargs)
                    self.instruments["spectrum"+str(s+1)] = "CustomLoader"
                elif issubclass(args[s], inst.InstrumentBlueprint) or issubclass(args[s].__class__, inst.InstrumentBlueprint):
                    self.loaded_spec_data["spectrum"+str(s+1)] = args[s]
                    _usr_loader_name = args[s].__class__.__name__ if issubclass(args[s].__class__, inst.InstrumentBlueprint) else args[s].__name__
                    self.instruments["spectrum"+str(s+1)] = "UserLoader:"+_usr_loader_name
            else:
                file_indx = s-num_of_custom
                self.loaded_spec_data["spectrum"+str(s+1)] = self.instrument_loaders[instruments[s]](pha_file[file_indx],
                                                                                                    arf_file=arf_file[file_indx],
                                                                                                    rmf_file=rmf_file[file_indx],
                                                                                                    srm_file=srm_file[file_indx],
                                                                                                    srm_custom=srm_custom[file_indx],
                                                                                                    custom_channel_bins=custom_channel_bins[file_indx],
                                                                                                    **kwargs)
                self.instruments["spectrum"+str(s+1)] = instruments[s]

        # Adding these classes should also yield {"spectrum1":..., "spectrum2":..., etc.}

    def _sort_files(self, pha_file=None, arf_file=None, rmf_file=None, srm_file=None, srm_custom=None, custom_channel_bins=None):
        """ Takes in spectral data files and turns then all into list.

        Parameters
        ----------
        pha_file : string or list of strings
                The PHA file or list of PHA files for the spectrum to be loaded.

        arf_file, rmf_file : string or list of strings
                The ARF and RMF files associated with the PHA file(s). If none are given (e.g, with
                NuSTAR data) it is assumed that these are in the same directory with same filename
                as the PHA file(s) but with extensions '.arf' and '.rmf', respectively.

        srm_file : string
                The file that contains the spectral response matrix for the given spectrum.

        srm_custom : 2d array
                User defined spectral response matrix. This is accepted over the SRM created from any
                ARF and RMF files given.

        custom_channel_bins : 2d array
                User defined channel bins for the columns of the SRM matrix.
                E.g., custom_channel_bins=[[1,1.5],[1.5,2],...]

        Returns
        -------
        Equal length lists such that each spectral data set has an entry for all inputs (pha_file,
        arf_file, rmf_file, srm_custom, custom_channel_bins) along with a list of corresponding
        instrument names.
        """
        if type(pha_file)==type(None):
            return [], [], [], [], [], [], []

        # if only one observation is given then it won't be a list so make it one
        file_pha = _make_into_list(pha_file)
        file_arf = _make_into_list(arf_file)
        file_rmf = _make_into_list(rmf_file)
        file_srm = _make_into_list(srm_file)

        # the following should be numpy arrays so _make_into_list would turn the array to a list, not put it into a list
        custom_srm = srm_custom if type(srm_custom)==list else [srm_custom]
        custom_channel_bins = custom_channel_bins if type(custom_channel_bins)==list else [custom_channel_bins]

        # check if arf and rmf and custom srm is either None in which case everything is found via the pha
        #  file naming (a list of None the same length as the pha input will also acheive this) or if there
        #  is a corresponding arf, rmf, and srm for every pha file
        assert ((type(arf_file)==type(None)) and (type(arf_file)==type(None)) and (len(file_pha)>=1)) \
                or \
                ((len(file_arf)==len(file_pha)) and (len(file_rmf)==len(file_pha))), \
                """Names can be taken from the \"pha_file\" input if your \"arf_file\", \"rmf_file\", and \"srm_file\" are not
                supplied. This means that if your \"arf_file\", \"rmf_file\", and \"srm_file\" are supplied then they can
                either be of list length==1 or the same number of entries as your \"pha_file\" input."""

        assert (type(file_srm)==type(None)) or (len(file_srm)==1) or (len(file_srm)==len(file_pha)), \
                """The \"file_srm\" should either be None, list length 1, or the same length as the \"pha_file\" input."""

        assert (type(srm_custom)==type(None)) or (len(custom_srm)==1) or (len(custom_srm)==len(file_pha)), \
                """The \"srm_custom\" should either be None, list length 1, or the same length as the \"pha_file\" input."""

        assert (type(custom_channel_bins)==type(None)) or (len(custom_channel_bins)==1) or (len(custom_channel_bins)==len(file_pha)), \
                """The \"custom_channel_bins\" should either be None, list length 1, or the same length as the \"pha_file\" input."""

        # make sure lists of None are same length for inputs to self.load1spec()
        file_pha, file_arf, file_rmf, file_srm, custom_srm, custom_channel_bins = self._lists_same_length(file_pha, file_arf, file_rmf, file_srm, custom_srm, custom_channel_bins)

        instruments = self._files2instruments(file_pha)

        return file_pha, file_arf, file_rmf, file_srm, custom_srm, custom_channel_bins, instruments

    def _lists_same_length(self, file_pha, file_arf, file_rmf, file_srm, custom_srm, custom_channel_bins):
        """ Ensure all file_pha entries have corresponding entries in all other lists.

        Parameters
        ----------
        blah : blah
                .

        Returns
        -------
        .
        """
        # make sure lists of None are same length for inputs to self.load1spec()
        if (len(file_arf)==1) and (len(file_rmf)==1) and (len(file_pha)>1):
            file_arf *= len(file_pha)
            file_rmf *= len(file_pha)
        if (len(file_srm)==1):
            file_srm *= len(file_pha)
        if (len(custom_srm)==1) and (len(file_pha)>1):
            custom_srm *= len(file_pha)
        if (len(custom_channel_bins)==1) and (len(file_pha)>1):
            custom_channel_bins *= len(file_pha)

        return file_pha, file_arf, file_rmf, file_srm, custom_srm, custom_channel_bins

    def _files2instruments(self, pha_files):
        """ Finds the instruments that correspond to the list of input `pha_files` (list of fits files).

        Parameters
        ----------
        pha_files : list of strings
                List of PHA (fits) files.

        Returns
        -------
        List of corresponding instrument names (strings) to the input fits files.
        """
        _instruments_names = []
        for pf in pha_files:
            with fits.open(pf) as hdul:
                if ("TELESCOP" in hdul[0].header):
                    # works for 'NuSTAR' and 'RHESSI'
                    _instruments_names.append(hdul[0].header["TELESCOP"])
                else:
                    print("How do I know the instument?")
        return _instruments_names

    @property
    def rebin(self):
        """ ***Property*** Allows energy channels to be rebinned.

        Returns
        -------
        None if the data has not been rebinned, the new energy bins for the rebinned data.
        """
        if not hasattr(self, "_rebinned_edges"):
            return None
        return self._rebinned_edges

    @rebin.setter
    def rebin(self, group_mins):
        """ ***Property Setter*** Allows energy channels to be rebinned.

        Parameters
        ----------
        group_mins : int, list, or dict
                The minimum number of counts in a bin.

        Returns
        -------
        None.

        Example
        -------
        # load in 1 spectra, rebin the count channels to have a minimum of 10 counts
        s = LoadSpec(pha_file='filename1.pha',arf_file='filename1.arf',rmf_file='filename1.rmf')
        s.rebin = 10
        s.rebin = {"spectrum1":10}
        """
        # check what the setter has been given
        # if dict, can group all loaded spectral data differently, "all" key takes priority and is applied to all spectra
        if type(group_mins)==dict:
            if "all" in group_mins:
                group_mins = group_mins["all"]
            else:
                spec_list = list(self.loaded_spec_data.keys())
                gms = [None]*len(spec_list)
                for k in list(group_mins):
                    if k in spec_list:
                        gms[spec_list.index(k)] = group_mins[k]
                group_mins = gms
        # if None: do nothing, if int:apply to all,
        if type(group_mins)==type(None):
            return None
        elif (type(group_mins)==int):
            group_mins = [group_mins]*len(list(self.loaded_spec_data.keys()))
        elif not self._rebin_list_and_one2one(group_mins):
            return None

        # now rebin the data
        bin_edges = []
        for s, c in zip(list(self.loaded_spec_data.keys()), group_mins):
            # should be able to rebin across photon and both axes too but not sure how user would set those yet
            bin_edges.append(self._rebin_loaded_spec(spectrum=s, group_min=c, axis="count"))

        # need to group response file stuff https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node29.html
        self._rebinned_edges = dict(zip(self.loaded_spec_data.keys(), bin_edges))
        # remember how it was rebinned
        self._rebin_setting = dict(zip(self.loaded_spec_data.keys(), group_mins))

    def _rebin_effective_exposures(self, old_bins, new_bins, old_counts, new_counts, old_effective_exposures):
        """ Rebin arrays of effective exposures.

        If (like RHESSI) we have a different effective exposure per channel bin then these need to be rebinned too.
        This is done by rebinning the counts array and rebinning a calculated count rate array (counts/s that was in
        counts/s/keV), both being rebinned by summing across combined bins. Then to find the new "rebinned effective
        exposures we will just divide the counts array by the rebinned counts/s array. I.e.,

        .. math::
         new_rebinned_effective_exposures = rebinned_counts / rebinned_counts_per_second

        Parameters
        ----------
        old_bins, new_bins, old_counts, new_counts, old_effective_exposures : np.array
                Array of the original counts/bins, and effective exposures. New bins are needed and shuold already
                have new counts.

        Returns
        -------
        Array of rebinned "effective exposures" or just the effective exposure if it was a single number.
        """
        # only rebin if different "exposures" per energy channel
        if not isnumber(old_effective_exposures):
            rebinned_counts_per_second = inst.rebin_any_array(data=old_counts/old_effective_exposures,
                                                            old_bins=old_bins,
                                                            new_bins=new_bins,
                                                            combine_by="sum")
            return new_counts / rebinned_counts_per_second
        else:
            return old_effective_exposures

    def _rebin_list_and_one2one(self, group_mins):
        """ Check if the group minimum (minima) given is in a list form and with a one-to-one entry to the loaded stectra.

        Parameters
        ----------
        group_mins : string
                Spectrum to be checked. E.g., \'spectrum1\'.

        Returns
        -------
        Boolean.
        """
        if (type(group_mins) in (list, np.ndarray)) and len(group_mins)==len(list(self.loaded_spec_data.keys())):
            return True
        else:
            print("rebin must be int or list of int with one-to-one match to the loaded spectra or dict with keys as the spectrum identifiers or with key \"all\".")
            return False

    def _rebin_check(self, spectrum):
        """ Check if the spectrum given has been rebinned.

        Parameters
        ----------
        spectrum : string
                Spectrum to be checked. E.g., \'spectrum1\'

        Returns
        -------
        Boolean.
        """
        _orig_in_extras = False if ("original_srm" not in self.loaded_spec_data[spectrum]["extras"]) else True
        return _orig_in_extras

    @property
    def undo_rebin(self):
        """ ***Property*** Allows the energy channel's rebinning to be undone.

        Returns
        -------
        None.
        """
        if not hasattr(self, "_undo_rebin"):
            self._undo_rebin = "all"

        if type(self._undo_rebin)==type(None):
            return

        spec_list = list(self.loaded_spec_data.keys()) if self._undo_rebin=="all" else self._undo_rebin

        for spec in spec_list:
            _orig_in_extras = self._rebin_check(spectrum=spec)

            if _orig_in_extras:
                del self._rebin_setting[spec], self._rebinned_edges[spec]
                # move original binning/counts/etc. into extras entry
                for s_att in self.loaded_spec_data[spec]().keys():
                    if s_att!="extras":
                        self.loaded_spec_data[spec][s_att] = self.loaded_spec_data[spec]["extras"]["original_"+s_att]
                        del self.loaded_spec_data[spec]["extras"]["original_"+s_att]
                self._check_if_known_background_and_rebin(spectrum=spec, undo=True)
            else:
                print(f"Nothing to undo in {spec} as data has not been rebinned.")

    @undo_rebin.setter
    def undo_rebin(self, spectrum):
        """ ***Property Setter*** Allows the energy channel's rebinning to be undone.

        Parameters
        ----------
        spectrum : int, string
                Number of spectrum to be undone. E.g., 1 or \'spectrum1\'. If None then nothing is undone,
                if \'all\' then all spectral rebinning will be undone.

        Returns
        -------
        None.

        Example
        -------
        # load in 1 spectra, rebin the count channels to have a minimum of 10 counts
        s = LoadSpec(pha_file='filename1.pha',arf_file='filename1.arf',rmf_file='filename1.rmf')
        s.rebin = 10
        s.undo_rebin = \'all\' <equivalent to> s.undo_rebin
        """
        if type(spectrum)==type(None):
            self._undo_rebin = None
        elif type(spectrum)==list:
            specs_no = ["spectrum"+s for s in spectrum if isnumber(s)]
            specs_id = [s.lower() for s in spectrum if (s.lower().startswith("spectrum"))]
            self._undo_rebin = specs_no + specs_id
        elif isnumber(spectrum):
            self._undo_rebin = ["spectrum"+str(spectrum)]
        elif spectrum.lower().startswith("spectrum"):
            self._undo_rebin = [spectrum.lower()]
        elif (spectrum.lower()=="all"):
            self._undo_rebin = spectrum.lower()
        else:
            self._undo_rebin = None

        if (type(self._undo_rebin)==type(None)) or ((len(self._undo_rebin)==0) and (self._undo_rebin!="all")):
            print("Please provide the spectrum number (N or \"N\") indicated by spectrumN in loaded_spec_data attribute, the full spectrum identifier (\"spectrumN\"), or set to \"all\".")
            print("Setting the undo_rebin property to nothing will undo all spectral rebinnings but it will be set to None and nothing will be undone here.")

        self.undo_rebin # now that _undo_rebin list is set, actually undo the rebinning

    def _rebin_data(self, spectrum, group_min):
        """ Rebins the data and channels to return them.

        Parameters
        ----------
        spectrum : string
                Number of spectrum to be undone. E.g., \'spectrum1\'.

        group_min : int
                Minimum number of counts in a bin.

        Returns
        -------
        The new bins (new_bins), counts (new_counts), binning widths (new_binning), bin centres (bin_mids), count
        rates (ctr), count rate errors (ctr_err), and whether the spectrum was already binned (_orig_in_extras).
        """
        # check if data has been rebinned already
        _orig_in_extras = self._rebin_check(spectrum=spectrum)

        # get new bins and binned counts
        new_bins, new_counts, counts_error, new_effective_exposure = self.group_spec(spectrum=spectrum, group_min=group_min, _orig_in_extras=_orig_in_extras)

        # calculate the new widths, centres, count errors, count rates and count rate errors
        new_binning = np.diff(new_bins).flatten()
        bin_mids = np.mean(new_bins, axis=1)
        ctr = (new_counts / new_binning) / new_effective_exposure
        #old way the now ctr_err = (np.sqrt(new_counts) / new_binning) / new_effective_exposure
        ctr_err = counts_error / new_binning / new_effective_exposure # no guarentee always will have poisson errors

        return new_bins, new_counts, counts_error, new_binning, bin_mids, ctr, ctr_err, new_effective_exposure, _orig_in_extras

    def _rebin_loaded_spec(self, spectrum, group_min, axis="count"):
        """ Rebins all the relevant data for a spectrum and moves original information into the \'extras\' key in the loaded_spec_data attribute.

        Parameters
        ----------
        spectrum : string
                SRM's spectrum identifier to be rebinned. E.g., \'spectrum1\'.

        group_min : int
                Minimum number of counts in a bin.

        axis : string
                Define what \'axis\' the binning should be applied to. E.g., \'photon\', \'count\', or \'photon_and_count\'.

        Returns
        -------
        New bin edges from the rebinning process.
        """

        if (axis=="count") or (axis=="photon_and_count"):
            new_bins, new_counts, count_error, new_binning, bin_mids, ctr, ctr_err, new_effective_exposure, _orig_in_extras = self._rebin_data(spectrum, group_min)

        if not _orig_in_extras:
            # move original binning/counts/etc. into extras entry
            for s_att in self.loaded_spec_data[spectrum]().keys():
                if s_att!="extras":
                    # print("putting in extras", s_att)
                    self.loaded_spec_data[spectrum]["extras"]["original_"+s_att] = self.loaded_spec_data[spectrum][s_att]

        # put new rebinned data into the loaded_spec_data dictionary
        if (axis=="count") or (axis=="photon_and_count"):
            self.loaded_spec_data[spectrum]["count_channel_bins"] = new_bins
            self.loaded_spec_data[spectrum]["count_channel_mids"] = bin_mids
            self.loaded_spec_data[spectrum]["count_channel_binning"] = new_binning
        if (axis=="photon") or (axis=="photon_and_count"):
            self.loaded_spec_data[spectrum]["photon_channel_bins"] = new_bins
            self.loaded_spec_data[spectrum]["photon_channel_mids"] = bin_mids
            self.loaded_spec_data[spectrum]["photon_channel_binning"] = new_binning
        self.loaded_spec_data[spectrum]["counts"] = new_counts
        self.loaded_spec_data[spectrum]["count_rate"] = ctr
        self.loaded_spec_data[spectrum]["count_rate_error"] = ctr_err
        self.loaded_spec_data[spectrum]["effective_exposure"] = new_effective_exposure

        # update the count errors
        self.loaded_spec_data[spectrum]["count_error"] = count_error
        self.loaded_spec_data[spectrum]["count_rate_error"] = self.loaded_spec_data[spectrum]["count_error"] / self.loaded_spec_data[spectrum]["effective_exposure"] / self.loaded_spec_data[spectrum]["count_channel_binning"]

        # if spec has a known background, (e.g.,RHESSI) then have it here
        self._check_if_known_background_and_rebin(spectrum, new_bins)

        # https://heasarc.gsfc.nasa.gov/docs/rosat/ros_xselect_guide_v1.1/node7.html#SECTION00712000000000000000
        # https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node94.html
        # https://heasarc.gsfc.nasa.gov/docs/asca/abc/node9.html#SECTION00940000000000000000
        # effectively need to replicate https://heasarc.gsfc.nasa.gov/docs/software/ftools/caldb/rbnrmf.html for rmf (then get srm)
        # the rebinning I think XSPEC uses internally https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/ftrbnrmf.html
        # good website for XSPEC commands https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/heasptools.html
        # self.loaded_spec_data[spectrum]["srm"] = self._rebin_srm(spectrum=spectrum, axis="count")
        self.loaded_spec_data[spectrum]["srm"] = self.loaded_spec_data[spectrum]._rebin_srm(axis="count")
        return new_bins

    def _check_if_known_background_and_rebin(self, spectrum, new_bins=None, undo=False):
        """ Checks if the spectrum has a known background component and rebins it. Looks for 'original_background' in
        the 'extras' key (units counts) and calculates the 'background_rate' (units counts/s).

        Mainly for RHESSI stuff at the minute.

        Parameters
        ----------
        spectrum : string
                Spectrum identifier to be rebinned. E.g., \'spectrum1\'.

        new_bins: 2d array
                New bins for the background to be rebinned into.
                Default: None

        undo : bool
                True to revert any rebinning, False to do the rebinning.

        Returns
        -------
        None.
        """
        # so we can loop
        _bg_entries = ["background_counts", "background_count_error", "background_rate", "background_rate_error", "background_effective_exposure"]

        if undo and ("original_background_counts" in self.loaded_spec_data[spectrum]["extras"]):
            # undo the binning
            for e in _bg_entries:
                self.loaded_spec_data[spectrum]["extras"][e] = self.loaded_spec_data[spectrum]["extras"]["original_"+e]
                del self.loaded_spec_data[spectrum]["extras"]["original_"+e]

        elif "background_counts" in self.loaded_spec_data[spectrum]["extras"]:
            # do the binning
            if "original_background_counts" not in self.loaded_spec_data[spectrum]["extras"]:
                for e in _bg_entries:
                    self.loaded_spec_data[spectrum]["extras"]["original_"+e] = self.loaded_spec_data[spectrum]["extras"][e]

            self.loaded_spec_data[spectrum]["extras"]["background_counts"] = inst.rebin_any_array(data=self.loaded_spec_data[spectrum]["extras"]["original_background_counts"],
                                                                                                  old_bins=self.loaded_spec_data[spectrum]["extras"]["original_count_channel_bins"],
                                                                                                  new_bins=new_bins,
                                                                                                  combine_by="sum")

            self.loaded_spec_data[spectrum]["extras"]["background_effective_exposure"] = self._rebin_effective_exposures(old_bins=self.loaded_spec_data[spectrum]["extras"]["original_count_channel_bins"],
                                                                                                                         new_bins=new_bins,
                                                                                                                         old_counts=self.loaded_spec_data[spectrum]["extras"]["original_background_counts"],
                                                                                                                         new_counts=self.loaded_spec_data[spectrum]["extras"]["background_counts"],
                                                                                                                         old_effective_exposures=self.loaded_spec_data[spectrum]["extras"]["original_background_effective_exposure"])
            self.loaded_spec_data[spectrum]["extras"]["background_count_error"] = np.sqrt(self.loaded_spec_data[spectrum]["extras"]["background_counts"])
            self.loaded_spec_data[spectrum]["extras"]["background_rate"] = self.loaded_spec_data[spectrum]["extras"]["background_counts"] / self.loaded_spec_data[spectrum]["extras"]["background_effective_exposure"] / self.loaded_spec_data[spectrum]["count_channel_binning"]
            self.loaded_spec_data[spectrum]["extras"]["background_rate_error"] = np.sqrt(self.loaded_spec_data[spectrum]["extras"]["background_counts"]) / self.loaded_spec_data[spectrum]["extras"]["background_effective_exposure"] / self.loaded_spec_data[spectrum]["count_channel_binning"]

    def group(self, channel_bins, counts, group_min):
        """ Groups bins so they have at least a `group_min` number of counts.

        Parameters
        ----------
        channel_bins, counts : np.array
                Array of the channel bins and counts.

        group_min : Int
                The minimum number of counts allowed in a bin. This input is a starting number and then is checked
                incrementally.
                Default: None

        Returns
        -------
        The number of counts left over at the end that could not be grouped into the minimum number and new bins/counts arrays.
        """
        binned_channel = []
        binned_counts = []
        combin = 0
        reset_bin_counter = True
        for c, count in enumerate(counts):
            if count>=group_min and combin==0 and reset_bin_counter:
                binned_channel.append(channel_bins[c])
                binned_counts.append(count)
            else:
                if reset_bin_counter:
                    start_e_bin = channel_bins[c][0]
                    reset_bin_counter = False
                combin += count
                if combin >= group_min:
                    binned_channel.append([start_e_bin, channel_bins[c][1]]) # starting at the last bin edge and the last edge of the bin we're on
                    binned_counts.append(combin)
                    combin = 0
                    reset_bin_counter = True

        return combin, binned_channel, binned_counts

    def group_pha_finder(self, channels, counts, group_min=None, print_tries=False):
        """ Takes the counts, and checks the bins left over from grouping the bins with a minimum value.

        Parameters
        ----------
        channel, counts : np.array
                Array of the channel bins and counts.

        group_min : Int
                The minimum number of counts allowed in a bin. This input is a starting number and then is checked
                incrementally.
                Default: None

        print_tries : Bool
                States whether the result of every try of 'group_min' should be displayed (True) or only the final
                result (False, default).
                Default: False

        Returns
        -------
        The new bins and the minimum bin number that gives zero counts left over at the end, if they exist, else None.
        """

        if not self._valid_group_min_entry(group_min):
            return

        # grppha groups in counts, not counts s^-1 or anything
        total_counts = np.sum(counts)

        while True:
            # group the counts
            combin, binned_channel, _ = self.group(channels, counts, group_min)

            if print_tries:
                print('Group min, ', group_min, ', has counts left over: ', combin)

            # check if all counts are binned, if not increment group_min and start again
            if combin != 0:
                group_min += 1
            elif group_min == total_counts:
                print('The minimum group number being tried is the same as the total number of counts.')
                return
            else:
                print('Group minimum that works is: ', group_min)
                return np.array(binned_channel), group_min

    def group_spec(self, spectrum, group_min=None, _orig_in_extras=False):
        """ Takes the counts, and checks the bins left over from grouping the bins with a minimum value.

        Parameters
        ----------
        spectrum : string
                SRM's spectrum identifier to be rebinned. E.g., \'spectrum1\'.

        group_min : Int
                The minimum number of counts allowed in a bin. This input is a starting number and the is checked
                incrementally.
                Default: None

        print_remainders : Bool
                States whether the result's remainder should be printed.
                Default: False

        _orig_in_extras : Bool
                Check if \'original_srm\' is in self.loaded_spec_data[spectrum]["extras"]. If it is then the data
                has been rebinned (True), if not then the data has not been rebinned (False).
                Default: False

        Returns
        -------
        The bin edges and new counts for you minimum group number. Also the count errors and effective exposures too.
        Any bins left over are now included with zero counts.
        """

        counts = self.loaded_spec_data[spectrum]["counts"] if not _orig_in_extras else self.loaded_spec_data[spectrum]["extras"]["original_counts"]
        count_error = self.loaded_spec_data[spectrum]["count_error"] if not _orig_in_extras else self.loaded_spec_data[spectrum]["extras"]["original_count_error"]
        channel_bins = self.loaded_spec_data[spectrum]["count_channel_bins"] if not _orig_in_extras else self.loaded_spec_data[spectrum]["extras"]["original_count_channel_bins"]
        old_effective_exposures = self.loaded_spec_data[spectrum]["effective_exposure"] if not _orig_in_extras else self.loaded_spec_data[spectrum]["extras"]["original_effective_exposure"]

        new_bins, new_counts = self._group_cts(channel_bins, counts, group_min=group_min, spectrum=spectrum)
        new_effective_exposure = self._rebin_effective_exposures(old_bins=channel_bins, new_bins=new_bins, old_counts=counts, new_counts=new_counts, old_effective_exposures=old_effective_exposures)

        counts_error = inst.rebin_any_array(data=count_error, old_bins=channel_bins, new_bins=new_bins, combine_by="quadrature")
        counts_error[new_counts==0] = 0 # avoid plotting anything for bins that could not be combined

        return new_bins, new_counts, counts_error, new_effective_exposure

    def _group_cts(self, channel_bins, counts, group_min=None, spectrum=None, verbose=True):
        """ Takes the counts and bins and groups the counts to have a minimum number of group_min.

        Parameters
        ----------
        channel_bins, counts : np.array
                Array of the channel bins and counts.

        group_min : Int
                The minimum number of counts allowed in a bin.
                Default: None

        spectrum : string
                SRM's spectrum identifier to be rebinned. E.g., \'spectrum1\'.

        verbose : Bool
                If True, will print out the number counts not able to be binned.
                Default: True

        Returns
        -------
        The new bins and the corresponding grouped counts.
        """

        if not self._valid_group_min_entry(group_min):
            return channel_bins, counts

        # grppha groups in counts, not counts s^-1 or anything
        combin, binned_channel, binned_counts = self.group(channel_bins, counts, group_min)

        # since SRM is going to be binned, add the rest of the photon bins on at the end with native binning
        # any counts in these bins will be ignored, only 0s taken into consideration in photon space for these
        remainder_bins = channel_bins[np.where(channel_bins[:,1] > np.array(binned_channel)[-1,-1])]

        binned_counts += [0]*len(remainder_bins)
        new_bins = np.concatenate((np.array(binned_channel), remainder_bins))
        new_counts = np.array(binned_counts)

        self._verbose_tries(spectrum, group_min, combin, verbose)

        return new_bins, new_counts# np.unique(np.array(binned_channel).flatten())

    def _valid_group_min_entry(self, group_min):
        """ Checks if a valid `group_min` entry has been given. An entry of None is valid but still returns False.

        Parameters
        ----------
        group_min : Int > 0
                The minimum number of counts allowed in a bin.

        Returns
        -------
        Boolean.
        """
        if type(group_min)!=int or group_min<=0:
            if type(group_min)==type(None):
                return False
            print('The \'group_min\' parameter must be an integer > 0.')
            return False
        return True

    def _verbose_tries(self, spectrum, group_min, combin, verbose):
        """ Executes print statements on the result of count rebinning.

        Any counts that were left over and could not form a bin are indicated in the print statements.

        Parameters
        ----------
        spectrum : string
                SRM's spectrum identifier to be rebinned. E.g., \'spectrum1\'.

        group_min : Int > 0
                The minimum number of counts allowed in a bin. This input is a starting number and the is checked
                incrementally.

        combin : Int > 0
                The number of bins left over that were not able to be binned.

        verbose : Bool
                If True, will print out the number counts not able to be binned.

        Returns
        -------
        None.
        """
        if combin>0 and verbose:
            if type(spectrum)!=type(None):
                print("In "+spectrum+", ", end="")
            print(combin,f' counts are left over from binning (bin min. {group_min}) and will not be included when fitting or shown when plotted.')

    def __add__(self, other):
        """ Define what adding means to the function.

        Just combine other's loaded_spec_data with self's while changing other's spectrum numbers. E.g.,
        self.loaded_spec_data={"spectrum1":...}, other.loaded_spec_data={"spectrum1":...}, then
        (self+other).loaded_spec_data={"spectrum1":...,"spectrum2":...}"spectrum2" was other's "spectrum1"
        """

        # combine loaded_spec_data attribute between classes
        # can't just do a = {**b, **c} since keys need to change in the second dict
        _other_spec_ = list(other.loaded_spec_data.keys())
        _last_in_self_ = int(list(self.loaded_spec_data.keys())[-1].split("spectrum")[-1])

        # combine the loaded_spec_data dicts
        new_self = deepcopy(self)
        for c, key_other in enumerate(_other_spec_):
            new_self.loaded_spec_data["spectrum"+str(_last_in_self_+1+c)] = other.loaded_spec_data[key_other]

        return new_self

    def __repr__(self):
        """Provide a representation to construct the class from scratch."""
        return self._construction_string

    def __str__(self):
        """Provide a printable, user friendly representation of what the class contains."""
        _loadedspec = ""
        plural = ["Spectrum", "is"] if len(self.loaded_spec_data.keys())==1 else ["Spectra", "are"]
        tag = f"{plural[0]} Loaded {plural[1]}: "
        for s in self.loaded_spec_data.keys():
            if "pha.file" in self.loaded_spec_data[s]["extras"]:
                _loadedspec += str(self.loaded_spec_data[s]["extras"]["pha.file"])+"\n"+" "*len(tag)
            else:
                _loadedspec += str(None)+"\n"+" "*len(tag)
        return f"No. of Spectra Loaded: {len(self.loaded_spec_data.keys())} \n{tag}{_loadedspec}"
