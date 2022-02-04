"""
The following code is for instrument specific classes each using their own methods to create and edit their `_loaded_spec_data` attrbutes.
"""

import numpy as np
from os import path as os_path

from . import nu_spec_code as nu_spec # sunxspex.sunxspex_fitting

__all__ = ["NustarLoader", "StixLoader", "RhessiLoader", "CustomLoader", "rebin_any_array"]

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
                                 The ARF and RMF files associated with the PHA file(s). If none are given it is assumed 
                                 that these are in the same directory with same filename as the PHA file(s) but with 
                                 extensions '.arf' and '.rmf', respectively.

                         srm_custom : 2d array
                                 User defined spectral response matrix. This is accepted over the SRM created from any 
                                 ARF and RMF files given.

                         custom_channel_bins : 2d array
                                 User defined channel bins for the columns of the SRM matrix. 
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
        if (axis=="photon") or (axis=="photon_and_count"):
            # very slight difference to rbnrmf when binning across photon axis, <2% of entries have a ratio (my way/rbnrmf) >1 (up to 11)
            # all come from where the original rmf has zeros originally so might be down to precision being worked in, can't expect the exact same numbers essentially
            matrix = rebin_any_array(data=matrix, old_bins=old_photon_bins, new_bins=new_photon_bins, combine_by="mean")
        if (axis=="count") or (axis=="photon_and_count"):
            matrix = rebin_any_array(data=matrix.T, old_bins=old_count_bins, new_bins=new_count_bins, combine_by="sum").T # need to go along columns so .T then .T back
        
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
        if (axis=="count") or (axis=="photon_and_count"):
            old_count_bins = self._loaded_spec_data["extras"]["original_count_channel_bins"]
            new_count_bins = self._loaded_spec_data["count_channel_bins"] 
        if (axis=="photon") or (axis=="photon_and_count"):
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
    """
    __doc__ += InstrumentBlueprint._UNIVERSAL_DOC_
    
    def __init__(self, pha_file, arf_file=None, rmf_file=None, srm_custom=None, custom_channel_bins=None):
        """Construct a string to show how the class was constructed (`_construction_string`) and set the `_loaded_spec_data` dictionary attribute."""

        self._construction_string = f"NustarLoader(pha_file={pha_file},arf_file={arf_file},rmf_file={rmf_file},srm_custom={srm_custom},custom_channel_bins={custom_channel_bins})"
        self._loaded_spec_data = self._load1spec(pha_file, f_arf=arf_file, f_rmf=rmf_file, srm=srm_custom, channel_bins=custom_channel_bins)

    def _load1spec(self, f_pha, f_arf=None, f_rmf=None, srm=None, channel_bins=None):
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

        channel_bins: 2d array
                User defined channel bins for the columns of the SRM matrix. 
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
        f_arf = f_pha[:-3]+"arf" if type(f_arf)==type(None) else f_arf
        f_rmf = f_pha[:-3]+"rmf" if type(f_rmf)==type(None) else f_rmf
        
        # need effective exposure and energy binning since likelihood works on counts, not count rates etc.
        _, counts, eff_exp = nu_spec.read_pha(f_pha)
        
        # now calculate the SRM or use a custom one if given
        if type(srm)==type(None):
        
            # if there is an ARF file load it in
            if os_path.isfile(f_arf):
                e_lo_arf, e_hi_arf, eff_area = nu_spec.read_arf(f_arf)

            # if there is an RMF file load it in and convert to a redistribution matrix
            if os_path.isfile(f_rmf):
                e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m = self._load_rmf(f_rmf)
        
            srm = nu_spec.make_srm(rmf_matrix=redist_m, arf_array=eff_area)
        else:
            e_lo_arf, e_hi_arf, eff_area = None, None, None
            e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m = None, None, None, None, None, None, None
        
        
        channel_bins = self._calc_channel_bins(e_lo_rmf, e_hi_rmf) if type(channel_bins)==type(None) else channel_bins
        channel_binning = np.diff(channel_bins).flatten()  
        
        # get the count rate information
        count_rate, count_rate_error = nu_spec.flux_cts_spec(f_pha, bin_size=channel_binning)
            
        # what spectral info you want to know from this observation
        return {"photon_channel_bins":channel_bins, 
                "photon_channel_mids":np.mean(channel_bins, axis=1), 
                "photon_channel_binning":channel_binning, 
                "count_channel_bins":channel_bins, 
                "count_channel_mids":np.mean(channel_bins, axis=1), 
                "count_channel_binning":channel_binning, 
                "counts":counts, 
                "count_error":np.sqrt(counts),
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
                } # this might make it easier to add different observations together

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
        
        e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix = nu_spec.read_rmf(rmf_file)
        fchan_array = nu_spec.col2arr_py(fchan)
        nchan_array = nu_spec.col2arr_py(nchan)
        redist_m = nu_spec.vrmf2arr_py(data=matrix,  
                                       n_grp_list=ngrp,
                                       f_chan_array=fchan_array, 
                                       n_chan_array=nchan_array) # 1.5 s of the total 2.4 s (1spec) is spent here
        
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
        new_eff_area = rebin_any_array(data=old_eff_area, old_bins=old_photon_bins, new_bins=new_photon_bins, combine_by="mean") if (axis!="count") else old_eff_area
        return nu_spec.make_srm(rmf_matrix=new_rmf, arf_array=new_eff_area)
        

class StixLoader(InstrumentBlueprint):
    """
    Loader specifically for STIX spectral data.

    StixLoader Specifics
    --------------------
    Short description of specifics.

    Superclass Override: 

    Properties
    ----------

    Setters
    -------

    Methods
    -------

    Attributes
    ----------
    """
    __doc__ += InstrumentBlueprint._UNIVERSAL_DOC_

    def __init__(self, pha_file, arf_file=None, rmf_file=None, srm_custom=None, custom_channel_bins=None):
        """Construct a string to show how the class was constructed (`_construction_string`) and set the `_loaded_spec_data` dictionary attribute."""

        self._construction_string = f"StixLoader(pha_file={pha_file},arf_file={arf_file},rmf_file={rmf_file},srm_custom={srm_custom},custom_channel_bins={custom_channel_bins})"
        self._loaded_spec_data = {}

class RhessiLoader(InstrumentBlueprint):
    """
    Loader specifically for RHESSI spectral data.

    RhessiLoader Specifics
    ----------------------
    Short description of specifics.

    Superclass Override:

    Properties
    ----------

    Setters
    -------

    Methods
    -------

    Attributes
    ----------
    """
    __doc__ += InstrumentBlueprint._UNIVERSAL_DOC_

    def __init__(self, pha_file, arf_file=None, rmf_file=None, srm_custom=None, custom_channel_bins=None):
        """Construct a string to show how the class was constructed (`_construction_string`) and set the `_loaded_spec_data` dictionary attribute."""

        self._construction_string = f"RhessiLoader(pha_file={pha_file},arf_file={arf_file},rmf_file={rmf_file},srm_custom={srm_custom},custom_channel_bins={custom_channel_bins})"
        self._loaded_spec_data = {}

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

    def __init__(self, spec_data_dict):
        """Construct a string to show how the class was constructed (`_construction_string`) and set the `_loaded_spec_data` dictionary attribute."""
        self._construction_string = f"CustomLoader({spec_data_dict})"

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
        assert len(essentials_not_present)==0, f"Essential dict. entries are not present: {essentials_not_present}"
        
        # check non-essential keys are given, if not then defaults are 1s
        nonessentials_not_present = set(non_ess_keys)-set(list(spec_data_dict.keys()))
        _def = self._nonessential_defaults(nonessentials_not_present,spec_data_dict["count_channel_bins"],spec_data_dict["counts"])
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
        if len(nonessential_list)>0:
            _count_length_default = np.ones(len(count_channels))
            _chan_mids_default = np.mean(count_channels, axis=1)
            defaults = {"photon_channel_bins":count_channels,
                        "photon_channel_mids":_chan_mids_default,
                        "photon_channel_binning":_count_length_default,
                        "count_channel_mids":_chan_mids_default,
                        "count_channel_binning":_count_length_default,
                        "count_error":_count_length_default,
                        "count_rate":counts,
                        "count_rate_error":_count_length_default,
                        "effective_exposure":1,
                        "srm":np.identity(len(counts)),
                        "extras":{}}
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
        if combine_by=="sum":
            new_binned_data.append(np.sum(data[np.where((nb[0]<=old_bins[:,0]) & (nb[-1]>=old_bins[:,-1]))], axis=0))
        elif combine_by=="mean":
            new_binned_data.append(np.mean(data[np.where((nb[0]<=old_bins[:,0]) & (nb[-1]>=old_bins[:,-1]))], axis=0))
        elif combine_by=="quadrature":
            new_binned_data.append(np.sqrt(np.sum(data[np.where((nb[0]<=old_bins[:,0]) & (nb[-1]>=old_bins[:,-1]))]**2, axis=0)))
    return np.array(new_binned_data)