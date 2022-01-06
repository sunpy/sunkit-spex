import numpy as np
from os import path as os_path

from sunxspex.sunxspex_fitting import nu_spec_code as nu_spec

__all__ = ["NustarLoader", "StixLoader", "RhessiLoader", "CustomLoader", "rebin_any_array"]

# Get a default class for the instrument specfic loaders
# Once the instrument specific loaders inherit from this then all they really have to do is get the spectral
#    data they want to fit in the correct dictionary form and assigned to `self._loaded_spec_data`.
class InstrumentBlueprint:

    def _rebin_rmf(self, matrix, old_count_bins=None, new_count_bins=None, old_photon_bins=None, new_photon_bins=None, axis="count"):
        ''' Rebins the photon and/or count channels of the redistribution matrix if needed.
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
        '''
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
        old_count_bins, new_count_bins, old_photon_bins, new_photon_bins = None, None, None, None
        if (axis=="count") or (axis=="photon_and_count"):
            old_count_bins = self._loaded_spec_data["extras"]["original_count_channel_bins"]
            new_count_bins = self._loaded_spec_data["count_channel_bins"] 
        if (axis=="photon") or (axis=="photon_and_count"):
            old_photon_bins = self._loaded_spec_data["extras"]["orignal_photon_channel_bins"]
            new_photon_bins = self._loaded_spec_data["photon_channel_bins"]
        return old_count_bins, new_count_bins, old_photon_bins, new_photon_bins

    def _rebin_srm(self, axis="count"):
        ''' Rebins the photon and/or count channels of the spectral response matrix (SRM) if needed. 
        
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
        '''
        old_count_bins, new_count_bins, old_photon_bins, new_photon_bins = self._channel_bin_info(axis)
        matrix = self._loaded_spec_data["srm"]
        return self._rebin_rmf(matrix, old_count_bins=old_count_bins, new_count_bins=new_count_bins, old_photon_bins=old_photon_bins, new_photon_bins=new_photon_bins, axis="count")

    def __getitem__(self, item):
        '''Index the entries in `_loaded_spec_data`'''
        return self._loaded_spec_data[item]

    def __setitem__(self, item, new_value):
        '''Allows entries in `_loaded_spec_data` to be changed.'''
        self._loaded_spec_data[item] = new_value

    def __call__(self):
        '''When the class is called (n=NustarLoader()->n()) then `_loaded_spec_data` 
        is returned.'''
        return self._loaded_spec_data

    def __repr__(self):
        '''String representation of `_loaded_spec_data`.'''
        return str(self._loaded_spec_data)


# Instrument specific data loaders
#    As long as these loaders get the spectral data to fit into the correct dictionary form and assigned to self._loaded_spec_data then
#    they should work but they can also overwrite the _rebin_srm(self, axis="count") method if the SRM rebinning is instrument specific.
#       The benefit here is that the class can have other methods/properties/setters (like time selection for STIX/RHESSI;e.g., 
#    .select_time(new_time)?) which can be accessed at the user level easily when fitting through the loaded_spec_data attribute 
#    (e.g., .loaded_spec_data["spectrum1"].select_time(new_time)).


class NustarLoader(InstrumentBlueprint):
    def __init__(self, pha_file, arf_file=None, rmf_file=None, srm_custom=None, custom_channel_bins=None):
        self._construction_string = f"NustarLoader(pha_file={pha_file},arf_file={arf_file},rmf_file={rmf_file},srm_custom={srm_custom},custom_channel_bins={custom_channel_bins})"
        self._loaded_spec_data = self.load1spec(pha_file, f_arf=arf_file, f_rmf=rmf_file, srm=srm_custom, channel_bins=custom_channel_bins)

    def load1spec(self, f_pha, f_arf=None, f_rmf=None, srm=None, channel_bins=None):
        ''' Loads all the information in for a given spectrum.

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
        '''
        
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
                e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m = self.load_rmf(f_rmf)
        
            srm = nu_spec.make_srm(rmf_matrix=redist_m, arf_array=eff_area)
        else:
            e_lo_arf, e_hi_arf, eff_area = None, None, None
            e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m = None, None, None, None, None, None, None
        
        
        channel_bins = self.calc_channel_bins(e_lo_rmf, e_hi_rmf) if type(channel_bins)==type(None) else channel_bins
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

    def load_rmf(self, rmf_file):
        ''' Extracts all information, mainly the redistribution matrix ([counts/photon]) from a given RMF file.

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
        '''
        
        e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix = nu_spec.read_rmf(rmf_file)
        fchan_array = nu_spec.col2arr_py(fchan)
        nchan_array = nu_spec.col2arr_py(nchan)
        redist_m = nu_spec.vrmf2arr_py(data=matrix,  
                                       n_grp_list=ngrp,
                                       f_chan_array=fchan_array, 
                                       n_chan_array=nchan_array) # 1.5 s of the total 2.4 s (1spec) is spent here
        
        return e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m

    def calc_channel_bins(self, e_low, e_hi):
        ''' Calculates the count channel bins from the given rmf files. Assumes that the photon and count 
        channel bins are the same.

        Parameters
        ----------
        e_low : 1d array
                Array of the lower bounds of all the channel bins.

        e_hi : 1d array
                Array of the higher bounds of all the channel bins.

        Returns
        -------
        None if no e_low or e_hi is given or 2d array where each row is the lower and higher bound of that bin.
        '''
        if (e_low is None) or (e_hi is None):
            print("If no rmf/arf files are given and a custom srm is provided, please provide the custom_channel_bins.\nE.g., custom_channel_bins=[[1,2],[2,3],...]")
            return None
        else:
            return np.stack((e_low, e_hi), axis=-1)

    def _rebin_srm(self, axis="count"):
        ''' Rebins the photon and/or count channels of the spectral response matrix by rebinning the redistribution 
        matrix and the effective area array.

        Parameters
        ----------
        axis : string
                Define what \'axis\' the binning should be applied to. E.g., \'photon\', \'count\', or \'photon_and_count\'.

        Returns
        -------
        The rebinned 2d spectral response matrix.
        '''
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
    def __init__(self):
        self._loaded_spec_data = {}

class RhessiLoader(InstrumentBlueprint):
    def __init__(self):
        self._loaded_spec_data = {}

class CustomLoader(InstrumentBlueprint):
    def __init__(self, spec_data_dict):

        # needed keys
        ess_keys = ["count_channel_bins",
                    "counts"]
        non_ess_keys = ["photon_channel_bins",
                        "photon_channel_mids",
                        "photon_channel_binning",
                        "count_channel_mids",
                        "count_channel_binning",
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

    def _nonessential_defaults(self,nonessential_list,count_channels,counts):
        if len(nonessential_list)>0:
            _count_length_default = np.ones(len(count_channels))
            defaults = {"photon_channel_bins":count_channels,
                        "photon_channel_mids":_count_length_default,
                        "photon_channel_binning":_count_length_default,
                        "count_channel_mids":_count_length_default,
                        "count_channel_binning":_count_length_default,
                        "count_rate":counts,
                        "count_rate_error":_count_length_default,
                        "effective_exposure":1,
                        "srm":np.identity(len(counts)),
                        "extras":{}}
            return defaults
        else:
            return {}


def rebin_any_array(data, old_bins, new_bins, combine_by="sum"):
    ''' Takes any array of data in old_bins space and rebins along data array axis==0 to have new_bins.
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
    '''
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