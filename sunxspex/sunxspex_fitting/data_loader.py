import numpy as np
#from os import path as os_path
from copy import deepcopy
from astropy.io import fits

#from sunxspex.sunxspex_fitting import nu_spec_code as nu_spec
from sunxspex.sunxspex_fitting.parameter_handler import _make_into_list
from sunxspex.sunxspex_fitting.instruments import *

__all__ = ["LoadSpec", "isnumber"]

# class LoadSpec:
#     def __init__(self, pha_file=None, arf_file=None, rmf_file=None, srm_custom=None, custom_channel_bins=None):
#         """
#         This class's job is to load in spectral file(s). I.e., load in the count spectrum and then 
#         calculate/store the SRM for fitting.

#         Parameters
#         ----------
#         pha_file : string or list of strings
#                 The PHA file or list of PHA files for the spectrum to be loaded.

#         arf_file, rmf_file : string or list of strings
#                 The ARF and RMF files associated with the PHA file(s). If none are given it is assumed 
#                 that these are in the same directory with same filename as the PHA file(s) but with 
#                 extensions '.arf' and '.rmf', respectively.

#         srm_custom : 2d array
#                 User defined spectral response matrix. This is accepted over the SRM created from any 
#                 ARF and RMF files given.

#         custom_channel_bins : 2d array
#                 User defined channel bins for the columns of the SRM matrix. 
#                 E.g., custom_channel_bins=[[1,1.5],[1.5,2],...]

#         Properties
#         ----------
#         rebin : list/array
#                 Returns the new energy bins of the data (self._rebinned_edges), None if the has not been rebinned (has setter). 
#         undo_rebin : None
#                 Has the code that uses self._undo_rebin to undo the rebinning for spectra (has setter).

#         Setters
#         -------
#         rebin : int, {specID:int}, {"all":int}
#                 Minimum number of counts in each bin. Changes data but saves original in "extras" key 
#                 in loaded_spec_data attribute. 
#         undo_rebin : int, specID, "all"
#                 Undo the rebinning. Move the original data from "extras" in loaded_spec_data attribute
#                 back to main part of the dict and set self._undo_rebin.

#         Attributes
#         ----------
#         loaded_spec_data : dict
#                 All loaded spectral data.
#         _construction_string : string
#                 String to be returned from __repr__() dunder method.
#         _rebinned_edges : dict
#                 Dictionary of energy bins if they have been rebinned for each loaded spectrum. Set in rebin().
#         _rebin_setting : dict
#                 Dictionary of rebin setting for each loaded spectrum. Set in rebin().
#         _undo_rebin : string
#                 Indicates the spectral rebinning to be undone. E.g., 'all', 'spectrumN', or None. Set in undo_rebin().

#         Examples
#         --------
#         # load in 2 spectra, rebin the count channels to have a minimum of 10 counts then undo that rebinning
#         s = LoadSpec(pha_file=['filename1.pha', 'filename2.pha'],
#                      arf_file=['filename1.arf', 'filename2.arf'],
#                      rmf_file=['filename1.rmf', 'filename2.rmf'])
#         s.rebin = 10
#         s.undo_rebin = 10
#         """

#         self._construction_string = f"LoadSpec(pha_file={pha_file},arf_file={arf_file},rmf_file={rmf_file},srm_custom={srm_custom},custom_channel_bins={custom_channel_bins})"
        
#         # if only one observation is given then it won't be a list so make it one
#         file_pha = pha_file if type(pha_file)==list else [pha_file]
#         file_arf = arf_file if type(arf_file)==list else [arf_file]
#         file_rmf = rmf_file if type(rmf_file)==list else [rmf_file]
#         custom_srm = srm_custom if type(srm_custom)==list else [srm_custom]
#         custom_channel_bins = custom_channel_bins if type(custom_channel_bins)==list else [custom_channel_bins]
        
#         # check if are and rmf and custom srm is either None in which case everything is found via the pha 
#         #  file naming (a list of None the same length as the pha input will also acheive this) or if there 
#         #  is a corresponding arf, rmf, and srm for every pha file
#         assert ((type(arf_file)==type(None)) and (type(arf_file)==type(None)) and (len(file_pha)>=1)) \
#                 or \
#                 ((len(file_arf)==len(file_pha)) and (len(file_rmf)==len(file_pha))), \
#                 """Names are taken from the \"pha_file\" input if your \"arf_file\" and \"rmf_file\" are not 
#                 supplied. This means that if your \"arf_file\" and \"rmf_file\" are supplied then they can 
#                 either be of list length==1 or the same number of entries as your \"pha_file\" input."""
        
#         assert (type(srm_custom)==type(None)) or (len(custom_srm)==1) or (len(custom_srm)==len(file_pha)), \
#                 """The \"srm_custom\" should either be None, 1, or the same length as the \"pha_file\" input."""
        
#         assert (type(custom_channel_bins)==type(None)) or (len(custom_channel_bins)==1) or (len(custom_channel_bins)==len(file_pha)), \
#                 """The \"custom_channel_bins\" should either be None, 1, or the same length as the \"pha_file\" input."""
        
#         # make sure lists of None are same length for inputs to self.load1spec()
#         if (len(file_arf)==1) and (len(file_rmf)==1) and (len(file_pha)>1):
#             file_arf *= len(file_pha)
#             file_rmf *= len(file_pha)
#         if (len(custom_srm)==1) and (len(file_pha)>1):
#             custom_srm *= len(file_pha)
#         if (len(custom_channel_bins)==1) and (len(file_pha)>1):
#             custom_channel_bins *= len(file_pha)
        
#         # get ready to load multiple spectra if needed
#         self.loaded_spec_data = {}
#         for s in range(len(file_pha)):
#             self.loaded_spec_data["spectrum"+str(s+1)] = self.load1spec(file_pha[s], 
#                                                                         f_arf=file_arf[s], 
#                                                                         f_rmf=file_rmf[s], 
#                                                                         srm=custom_srm[s],
#                                                                         channel_bins=custom_channel_bins[s])
        
#         # Adding these classes should also yield {"spectrum1":..., "spectrum2":..., etc.}
        
        
#     def load1spec(self, f_pha, f_arf=None, f_rmf=None, srm=None, channel_bins=None):
#         ''' Loads all the information in for a given spectrum.

#         Parameters
#         ----------
#         f_pha, f_arf, f_rmf : string
#                 Filenames for the relevant spectral files. If f_arf, f_rmf are None it is assumed 
#                 that these are in the same directory with same filename as the PHA file but with 
#                 extensions '.arf' and '.rmf', respectively.
#                 Default of f_arf, f_rmf: None

#         srm : 2d array
#                 User defined spectral response matrix. This is accepted over the SRM created from any 
#                 ARF and RMF files given.
#                 Default: None

#         channel_bins: 2d array
#                 User defined channel bins for the columns of the SRM matrix. 
#                 E.g., custom_channel_bins=[[1,1.5],[1.5,2],...]
#                 Default: None

#         Returns
#         -------
#         Dictionary of the loaded in spectral information in the form {"photon_channel_bins":channel_bins, 
#                                                                       "photon_channel_mids":np.mean(channel_bins, axis=1), 
#                                                                       "photon_channel_binning":channel_binning, 
#                                                                       "count_channel_bins":channel_bins, 
#                                                                       "count_channel_mids":np.mean(channel_bins, axis=1), 
#                                                                       "count_channel_binning":channel_binning, 
#                                                                       "counts":counts, 
#                                                                       "count_rate":count_rate, 
#                                                                       "count_rate_error":count_rate_error, 
#                                                                       "effective_exposure":eff_exp,
#                                                                       "srm":srm,
#                                                                       "extras":{"pha.file":f_pha,
#                                                                                 "arf.file":f_arf,
#                                                                                 "arf.e_lo":e_lo_arf,
#                                                                                 "arf.e_hi":e_hi_arf,
#                                                                                 "arf.effective_area":eff_area,
#                                                                                 "rmf.file":f_rmf,
#                                                                                 "rmf.e_lo":e_lo_rmf,
#                                                                                 "rmf.e_hi":e_hi_rmf,
#                                                                                 "rmf.ngrp":ngrp,
#                                                                                 "rmf.fchan":fchan,
#                                                                                 "rmf.nchan":nchan,
#                                                                                 "rmf.matrix":matrix,
#                                                                                 "rmf.redistribution_matrix":redist_m} 
#                                                                      }.
#         '''
        
#         # what files might be needed (for NuSTAR)
#         f_arf = f_pha[:-3]+"arf" if type(f_arf)==type(None) else f_arf
#         f_rmf = f_pha[:-3]+"rmf" if type(f_rmf)==type(None) else f_rmf
        
#         # need effective exposure and energy binning since likelihood works on counts, not count rates etc.
#         _, counts, eff_exp = nu_spec.read_pha(f_pha)
        
#         # now calculate the SRM or use a custom one if given
#         if type(srm)==type(None):
        
#             # if there is an ARF file load it in
#             if os_path.isfile(f_arf):
#                 e_lo_arf, e_hi_arf, eff_area = nu_spec.read_arf(f_arf)

#             # if there is an RMF file load it in and convert to a redistribution matrix
#             if os_path.isfile(f_rmf):
#                 e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m = self.load_rmf(f_rmf)
        
#             srm = nu_spec.make_srm(rmf_matrix=redist_m, arf_array=eff_area)
#         else:
#             e_lo_arf, e_hi_arf, eff_area = None, None, None
#             e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m = None, None, None, None, None, None, None
        
        
#         channel_bins = self.calc_channel_bins(e_lo_rmf, e_hi_rmf) if type(channel_bins)==type(None) else channel_bins
#         channel_binning = np.diff(channel_bins).flatten()  
        
#         # get the count rate information
#         count_rate, count_rate_error = nu_spec.flux_cts_spec(f_pha, bin_size=channel_binning)
            
#         # what spectral info you want to know from this observation
#         return {"photon_channel_bins":channel_bins, 
#                 "photon_channel_mids":np.mean(channel_bins, axis=1), 
#                 "photon_channel_binning":channel_binning, 
#                 "count_channel_bins":channel_bins, 
#                 "count_channel_mids":np.mean(channel_bins, axis=1), 
#                 "count_channel_binning":channel_binning, 
#                 "counts":counts, 
#                 "count_rate":count_rate, 
#                 "count_rate_error":count_rate_error, 
#                 "effective_exposure":eff_exp,
#                 "srm":srm,
#                 "extras":{"pha.file":f_pha,
#                           "arf.file":f_arf,
#                           "arf.e_lo":e_lo_arf,
#                           "arf.e_hi":e_hi_arf,
#                           "arf.effective_area":eff_area,
#                           "rmf.file":f_rmf,
#                           "rmf.e_lo":e_lo_rmf,
#                           "rmf.e_hi":e_hi_rmf,
#                           "rmf.ngrp":ngrp,
#                           "rmf.fchan":fchan,
#                           "rmf.nchan":nchan,
#                           "rmf.matrix":matrix,
#                           "rmf.redistribution_matrix":redist_m} 
#                 } # this might make it easier to add different observations together
        
#     def calc_channel_bins(self, e_low, e_hi):
#         ''' Calculates the count channel bins from the given rmf files. Assumes that the photon and count 
#         channel bins are the same.

#         Parameters
#         ----------
#         e_low : 1d array
#                 Array of the lower bounds of all the channel bins.

#         e_hi : 1d array
#                 Array of the higher bounds of all the channel bins.

#         Returns
#         -------
#         None if no e_low or e_hi is given or 2d array where each row is the lower and higher bound of that bin.
#         '''
#         if (e_low is None) or (e_hi is None):
#             print("If no rmf/arf files are given and a custom srm is provided, please provide the custom_channel_bins.\nE.g., custom_channel_bins=[[1,2],[2,3],...]")
#             return None
#         else:
#             return np.stack((e_low, e_hi), axis=-1)
    
#     def load_rmf(self, rmf_file):
#         ''' Extracts all information, mainly the redistribution matrix ([counts/photon]) from a given RMF file.

#         Parameters
#         ----------
#         rmf_file : string
#                 The file path and name of the RMF file.

#         Returns
#         -------
#         The lower/higher photon bin edges (e_lo_rmf, e_hi_rmf), the number of counts channels activated by each photon channel (ngrp), 
#         starting indices of the count channel groups (fchan), number counts channels from each starting index (nchan), the coresponding 
#         counts/photon value for each count and photon entry (matrix), and the redistribution matrix (redist_m: with rows of photon channels, 
#         columns of counts channels, and in the units of counts/photon).
#         '''
        
#         e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix = nu_spec.read_rmf(rmf_file)
#         fchan_array = nu_spec.col2arr_py(fchan)
#         nchan_array = nu_spec.col2arr_py(nchan)
#         redist_m = nu_spec.vrmf2arr_py(data=matrix,  
#                                        n_grp_list=ngrp,
#                                        f_chan_array=fchan_array, 
#                                        n_chan_array=nchan_array) # 1.5 s of the total 2.4 s (1spec) is spent here
        
#         return e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m
    
#     @property
#     def rebin(self):
#         ''' ***Property*** 
#         Allows energy channels to be rebinned.

#         Parameters
#         ----------

#         Returns
#         -------
#         None if the data has not been rebinned, the new energy bins for the rebinned data.
#         '''
#         if not hasattr(self, "_rebinned_edges"):
#             return None
#         return self._rebinned_edges

#     @rebin.setter
#     def rebin(self, group_mins):
#         ''' ***Property Setter*** 
#         Allows energy channels to be rebinned.

#         Parameters
#         ----------
#         group_mins : int
#                 The minimum number of counts in a bin.

#         Returns
#         -------
#         None.

#         Example
#         -------
#         # load in 1 spectra, rebin the count channels to have a minimum of 10 counts
#         s = LoadSpec(pha_file='filename1.pha',arf_file='filename1.arf',rmf_file='filename1.rmf')
#         s.rebin = 10
#         '''
#         # check what the setter has been given
#         # if dict, can group all loaded spectral data differently, "all" key takes priority and is applied to all spectra
#         if type(group_mins)==dict:
#             if "all" in group_mins:
#                 group_mins = group_mins["all"]
#             else:
#                 spec_list = list(self.loaded_spec_data.keys())
#                 gms = [None]*len(spec_list)
#                 for k in list(group_mins):
#                     if k in spec_list:
#                         gms[spec_list.index(k)] = group_mins[k]
#                 group_mins = gms
#         # if None: do nothing, if int:apply to all, 
#         if type(group_mins)==type(None):
#             return None
#         elif (type(group_mins)==int):
#             group_mins = [group_mins]*len(list(self.loaded_spec_data.keys()))
#         elif (type(group_mins) in (list, np.ndarray)):
#             if len(group_mins)!=len(list(self.loaded_spec_data.keys())):
#                 print("rebin must be int or list of int with one-to-one match to the loaded spectra or dict with keys as the spectrum identifiers or with key \"all\".")
#                 return None
        
#         # now rebin the data
#         bin_edges = []           
#         for s, c in zip(list(self.loaded_spec_data.keys()), group_mins):
#             # should be able to rebin across photon and both axes too but not sure how user would set those yet
#             bin_edges.append(self._rebin_loaded_spec(spectrum=s, group_min=c, axis="count")) 
        
#         # need to group response file stuff https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node29.html
#         self._rebinned_edges = dict(zip(self.loaded_spec_data.keys(), bin_edges))
#         # remember how it was rebinned
#         self._rebin_setting = dict(zip(self.loaded_spec_data.keys(), group_mins))
        
#     def _rebin_check(self, spectrum):
#         ''' Check if the spectrum given has been rebinned.

#         Parameters
#         ----------
#         spectrum : string
#                 Spectrum to be checked. E.g., \'spectrum1\'

#         Returns
#         -------
#         Boolean.
#         '''
#         _orig_in_extras = False if ("original_srm" not in self.loaded_spec_data[spectrum]["extras"]) else True
#         return _orig_in_extras
        
#     @property
#     def undo_rebin(self):
#         ''' ***Property*** 
#         Allows the energy channel's rebinning to be undone.

#         Parameters
#         ----------

#         Returns
#         -------
#         None.
#         '''
#         if not hasattr(self, "_undo_rebin"):
#             self._undo_rebin = "all"
            
#         if type(self._undo_rebin)==type(None):
#             return
        
#         spec_list = list(self.loaded_spec_data.keys()) if self._undo_rebin=="all" else self._undo_rebin
        
#         for spec in spec_list:
#             _orig_in_extras = self._rebin_check(spectrum=spec)
        
#             if _orig_in_extras:
#                 del self._rebin_setting[spec], self._rebinned_edges[spec]
#                 # move original binning/counts/etc. into extras entry
#                 for s_att in self.loaded_spec_data[spec].keys():
#                     if s_att not in ("effective_exposure", "extras"):
#                         self.loaded_spec_data[spec][s_att] = self.loaded_spec_data[spec]["extras"]["original_"+s_att]
#                         del self.loaded_spec_data[spec]["extras"]["original_"+s_att]
#             else:
#                 print(f"Nothing to undo in {spec} as data has not been rebinned.")
    
#     @undo_rebin.setter
#     def undo_rebin(self, spectrum):
#         ''' ***Property Setter*** 
#         Allows the energy channel's rebinning to be undone.

#         Parameters
#         ----------
#         spectrum : int, string
#                 Number of spectrum to be undone. E.g., 1 or \'spectrum1\'. If None then nothing is undone, 
#                 if \'all\' then all spectral rebinning will be undone.

#         Returns
#         -------
#         None.

#         Example
#         -------
#         # load in 1 spectra, rebin the count channels to have a minimum of 10 counts
#         s = LoadSpec(pha_file='filename1.pha',arf_file='filename1.arf',rmf_file='filename1.rmf')
#         s.rebin = 10
#         s.undo_rebin = \'all\' <equivalent to> s.undo_rebin
#         '''
#         if type(spectrum)==type(None):
#             self._undo_rebin = None
#         elif type(spectrum)==list:
#             specs_no = ["spectrum"+s for s in spectrum if isnumber(s)]
#             specs_id = [s.lower() for s in spectrum if (s.lower().startswith("spectrum"))]
#             self._undo_rebin = specs_no + specs_id
#         elif isnumber(spectrum):
#             self._undo_rebin = ["spectrum"+str(spectrum)]
#         elif spectrum.lower().startswith("spectrum"):
#             self._undo_rebin = [spectrum.lower()]
#         elif (spectrum.lower()=="all"):
#             self._undo_rebin = spectrum.lower()
#         else:
#             self._undo_rebin = None
            
#         if (type(self._undo_rebin)==type(None)) or ((len(self._undo_rebin)==0) and (self._undo_rebin!="all")):
#             print("Please provide the spectrum number (N or \"N\") indicated by spectrumN in loaded_spec_data attribute, the full spectrum identifier (\"spectrumN\"), or set to \"all\".")
#             print("Setting the undo_rebin property to nothing will undo all spectral rebinnings but it will be set to None and nothing will be undone here.")
            
#         self.undo_rebin # now that _undo_rebin list is set, actually undo the rebinning
            
        
#     def _rebin_data(self, spectrum, group_min): 
#         ''' Rebins the data and channels to return them.

#         Parameters
#         ----------
#         spectrum : string
#                 Number of spectrum to be undone. E.g., \'spectrum1\'. 
        
#         group_min : int
#                 Minimum number of counts in a bin.

#         Returns
#         -------
#         The new bins (new_bins), counts (new_counts), binning widths (new_binning), bin centres (bin_mids), count 
#         rates (ctr), count rate errors (ctr_err), and whether the spectrum was already binned (_orig_in_extras).
#         '''
#         # check if data has been rebinned already
#         _orig_in_extras = self._rebin_check(spectrum=spectrum)

#         # get new bins and binned counts
#         new_bins, new_counts = self.group_spec(spectrum=spectrum, group_min=group_min, _orig_in_extras=_orig_in_extras)

#         # calculate the new widths, centres, count rates and count rate errors
#         new_binning = np.diff(new_bins).flatten()
#         bin_mids = np.mean(new_bins, axis=1)
#         ctr = (new_counts / new_binning) / self.loaded_spec_data[spectrum]["effective_exposure"]
#         ctr_err = (np.sqrt(new_counts) / new_binning) / self.loaded_spec_data[spectrum]["effective_exposure"]
        
#         return new_bins, new_counts, new_binning, bin_mids, ctr, ctr_err, _orig_in_extras
    
#     def _rebin_rmf(self, matrix, old_count_bins=None, new_count_bins=None, old_photon_bins=None, new_photon_bins=None, axis="count"):
#         ''' Rebins the photon and/or count channels of the redistribution matrix if needed.

#         Parameters
#         ----------
#         matrix : 2d array
#                 Redistribution matrix. 
        
#         old_count_bins, new_count_bins : 1d arrays
#                 The old count channel binning and the new binning to be for the redistribution matrix columns (sum columns).

#         old_photon_bins, new_photon_bins : 1d arrays
#                 The old photon channel binning and the new binning to be for the redistribution matrix columns (average rows).

#         axis : string
#                 Define what \'axis\' the binning should be applied to. E.g., \'photon\', \'count\', or \'photon_and_count\'.

#         Returns
#         -------
#         The rebinned 2d redistribution matrix.
#         '''
#         # across channel bins, we sum. across energy bins, we average
#         # appears to be >2x faster to average first then sum if needing to do both
#         if (axis=="photon") or (axis=="photon_and_count"):
#             # very slight difference to rbnrmf when binning across photon axis, <2% of entries have a ratio (my way/rbnrmf) >1 (up to 11)
#             # all come from where the original rmf has zeros originally so might be down to precision being worked in, can't expect the exact same numbers essentially
#             matrix = self._rebin_any_array(data=matrix, old_bins=old_photon_bins, new_bins=new_photon_bins, combine_by="mean")
#         if (axis=="count") or (axis=="photon_and_count"):
#             matrix = self._rebin_any_array(data=matrix.T, old_bins=old_count_bins, new_bins=new_count_bins, combine_by="sum").T # need to go along columns so .T then .T back
        
#         return matrix
    
#     def _rebin_srm(self, spectrum, axis="count"):
#         ''' Rebins the photon and/or count channels of the spectral response matrix by rebinning the redistribution 
#         matrix and the effective area array.

#         Parameters
#         ----------
#         spectrum : string
#                 SRM's spectrum identifier to be rebinned. E.g., \'spectrum1\'

#         axis : string
#                 Define what \'axis\' the binning should be applied to. E.g., \'photon\', \'count\', or \'photon_and_count\'.

#         Returns
#         -------
#         The rebinned 2d spectral response matrix.
#         '''
#         old_count_bins, new_count_bins, old_photon_bins, new_photon_bins = None, None, None, None
#         if (axis=="count") or (axis=="photon_and_count"):
#             old_count_bins = self.loaded_spec_data[spectrum]["extras"]["original_count_channel_bins"]
#             new_count_bins = self.loaded_spec_data[spectrum]["count_channel_bins"] 
#         if (axis=="photon") or (axis=="photon_and_count"):
#             old_photon_bins = self.loaded_spec_data[spectrum]["extras"]["orignal_photon_channel_bins"]
#             new_photon_bins = self.loaded_spec_data[spectrum]["photon_channel_bins"]
        
#         old_rmf = self.loaded_spec_data[spectrum]["extras"]["rmf.redistribution_matrix"]
#         old_eff_area = self.loaded_spec_data[spectrum]["extras"]["arf.effective_area"]
        
#         # check if effective areas are given, if not then just rebin the SRM as ist is
#         if type(old_eff_area) is not type(None):
#             # checked with ftrbnrmf
#             new_rmf = self._rebin_rmf(matrix=old_rmf, 
#                                     old_count_bins=old_count_bins, 
#                                     new_count_bins=new_count_bins, 
#                                     old_photon_bins=old_photon_bins, 
#                                     new_photon_bins=new_photon_bins, 
#                                     axis=axis)
            
#             # average eff_area, checked with ftrbnarf
#             new_eff_area = self._rebin_any_array(data=old_eff_area, old_bins=old_photon_bins, new_bins=new_photon_bins, combine_by="mean") if (axis!="count") else old_eff_area
#             return nu_spec.make_srm(rmf_matrix=new_rmf, arf_array=new_eff_area)
#         else:
#             return self._rebin_rmf(matrix=self.loaded_spec_data[spectrum]["srm"], old_count_bins=old_count_bins, new_count_bins=new_count_bins, old_photon_bins=old_photon_bins, new_photon_bins=new_photon_bins, axis=axis)
        
#     def _rebin_loaded_spec(self, spectrum, group_min, axis="count"):
#         ''' Rebins all the relevant data for a spectrum and moves original information into the \'extras\' key in the loaded_spec_data attribute.

#         Parameters
#         ----------
#         spectrum : string
#                 SRM's spectrum identifier to be rebinned. E.g., \'spectrum1\'.

#         group_min : int
#                 Minimum number of counts in a bin.

#         axis : string
#                 Define what \'axis\' the binning should be applied to. E.g., \'photon\', \'count\', or \'photon_and_count\'.

#         Returns
#         -------
#         New bin edges from teh rebinning process.
#         '''
        
#         if (axis=="count") or (axis=="photon_and_count"):
#             new_bins, new_counts, new_binning, bin_mids, ctr, ctr_err, _orig_in_extras = self._rebin_data(spectrum, group_min)
        
#         if not _orig_in_extras:
#             # move original binning/counts/etc. into extras entry
#             for s_att in self.loaded_spec_data[spectrum].keys():
#                 if s_att not in ("effective_exposure", "extras"):
#                     self.loaded_spec_data[spectrum]["extras"]["original_"+s_att] = self.loaded_spec_data[spectrum][s_att]
        
#         # put new rebinned data into the loaded_spec_data dictionary
#         if (axis=="count") or (axis=="photon_and_count"):
#             self.loaded_spec_data[spectrum]["count_channel_bins"] = new_bins
#             self.loaded_spec_data[spectrum]["count_channel_mids"] = bin_mids
#             self.loaded_spec_data[spectrum]["count_channel_binning"] = new_binning
#         if (axis=="photon") or (axis=="photon_and_count"):
#             self.loaded_spec_data[spectrum]["photon_channel_bins"] = new_bins
#             self.loaded_spec_data[spectrum]["photon_channel_mids"] = bin_mids
#             self.loaded_spec_data[spectrum]["photon_channel_binning"] = new_binning
#         self.loaded_spec_data[spectrum]["counts"] = new_counts
#         self.loaded_spec_data[spectrum]["count_rate"] = ctr
#         self.loaded_spec_data[spectrum]["count_rate_error"] = ctr_err
         
#         # https://heasarc.gsfc.nasa.gov/docs/rosat/ros_xselect_guide_v1.1/node7.html#SECTION00712000000000000000
#         # https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node94.html
#         # https://heasarc.gsfc.nasa.gov/docs/asca/abc/node9.html#SECTION00940000000000000000
#         # effectively need to replicate https://heasarc.gsfc.nasa.gov/docs/software/ftools/caldb/rbnrmf.html for rmf (then get srm)
#         # the rebinning I think XSPEC uses internally https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/ftrbnrmf.html
#         # good website for XSPEC commands https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/heasptools.html
#         self.loaded_spec_data[spectrum]["srm"] = self._rebin_srm(spectrum=spectrum, axis="count")
#         return new_bins
        
        
#     def group_pha_finder(self, channels, counts, group_min=None, print_tries=False):
#         ''' Takes the counts, and checks the bins left over from grouping the bins with a minimum 
#         value.

#         Parameters
#         ----------
#         channel, counts : np.array
#                 Array of the channel bins and counts.

#         group_min : Int
#                 The minimum number of counts allowed in a bin. This input is a starting number and then is checked 
#                 incrementally.
#                 Default: None

#         print_tries : Bool
#                 States whether the result of every try of 'group_min' should be displayed (True) or only the final 
#                 result (False, default).
#                 Default: False

#         Returns
#         -------
#         The new bins and the minimum bin number that gives zero counts left over at the end, if they exist, else None.
#         '''

#         if type(group_min)!=int or group_min<=0: 
#             print('The \'group_min\' parameter must be an integer > 0.')
#             return

#         # grppha groups in counts, not counts s^-1 or anything
#         total_counts = np.sum(counts)

#         while True:
#             binned_channel = []
#             combin = 0
#             # check if we can make it through the whole count list with 0 left over when grouping with group_min
#             for c, count in enumerate(counts):
#                 if count>=group_min and combin==0:
#                     binned_channel.append(channels[c])
#                 else:
#                     if combin==0:
#                         start_e_bin = channels[c][0] 
#                     combin += count
#                     if combin >= group_min:
#                         binned_channel.append([start_e_bin, channels[c][1]]) # starting at the last bin edge and the last edge of the bin we're on
#                         combin = 0

#             if print_tries == True:
#                 print('Group min, ', group_min, ', has counts left over: ', combin)

#             if combin != 0:
#                 group_min += 1
#             elif group_min == total_counts:
#                 print('The minimum group number being tried is the same as the total number of counts.')
#                 return
#             else:
#                 print('Group minimum that works is: ', group_min)
#                 return np.array(binned_channel), group_min
            
    
#     def group_spec(self, spectrum, group_min=None, _orig_in_extras=False):
#         ''' Takes the counts, and checks the bins left over from grouping the bins with a minimum 
#         value.

#         Parameters
#         ----------
#         spectrum : string
#                 SRM's spectrum identifier to be rebinned. E.g., \'spectrum1\'.

#         group_min : Int
#                 The minimum number of counts allowed in a bin. This input is a starting number and the is checked 
#                 incrementally.
#                 Default: None

#         print_remainders : Bool
#                 States whether the result's remainder should be printed.
#                 Default: False

#         _orig_in_extras : Bool
#                 Check if \'original_srm\' is in self.loaded_spec_data[spectrum]["extras"]. If it is then the data 
#                 has been rebinned (True), if not then the data has not been rebinned (False).
#                 Default: False

#         Returns
#         -------
#         The bin edges for you minimum group number. Any bins left over are now included.
#         '''

#         counts = self.loaded_spec_data[spectrum]["counts"] if not _orig_in_extras else self.loaded_spec_data[spectrum]["extras"]["original_counts"]
#         channel_bins = self.loaded_spec_data[spectrum]["count_channel_bins"] if not _orig_in_extras else self.loaded_spec_data[spectrum]["extras"]["original_count_channel_bins"]
        
        
#         return self.group_cts(channel_bins, counts, group_min=group_min, spectrum=spectrum)
    
#     def group_cts(self, channel_bins, counts, group_min=None, spectrum=None, verbose=True):
#         ''' Takes the counts and bins and groups the counts to have a minimum number of group_min.

#         Parameters
#         ----------
#         channel_bins, counts : np.array
#                 Array of the channel bins and counts.

#         group_min : Int
#                 The minimum number of counts allowed in a bin. This input is a starting number and the is checked 
#                 incrementally.
#                 Default: None

#         spectrum : string
#                 SRM's spectrum identifier to be rebinned. E.g., \'spectrum1\'.

#         verbose : Bool
#                 If True, will print out the number counts not able to be binned.
#                 Default: True

#         Returns
#         -------
#         The new bins and the corresponding grouped counts.
#         '''

#         if type(group_min)!=int or group_min<=0: 
#             if type(group_min)==type(None):
#                 return channel_bins, counts
#             print('The \'group_min\' parameter must be an integer > 0.')
#             return channel_bins, counts
        
#         # grppha groups in counts, not counts s^-1 or anything

#         binned_channel = []
#         binned_counts = []
#         combin = 0
#         reset_bin_counter = True
#         for c, count in enumerate(counts):
#             if count>=group_min and combin==0 and reset_bin_counter:
#                 binned_channel.append(channel_bins[c])
#                 binned_counts.append(count)
#             else:
#                 if reset_bin_counter:
#                     start_e_bin = channel_bins[c][0] 
#                     reset_bin_counter = False
#                 combin += count
#                 if combin >= group_min:
#                     binned_channel.append([start_e_bin, channel_bins[c][1]]) # starting at the last bin edge and the last edge of the bin we're on
#                     binned_counts.append(combin)
#                     combin = 0
#                     reset_bin_counter = True
                
#         # since SRM is going to be binned, add the rest of the photon bins on at the end with native binning
#         # any counts in these bins will be ignored, only 0s taken into consideration in photon space for these
#         remainder_bins = channel_bins[np.where(channel_bins[:,1] > np.array(binned_channel)[-1,-1])]
#         binned_counts += [0]*len(remainder_bins)
                    
#         if combin>0 and verbose:
#             if type(spectrum)!=type(None):
#                 print("In "+spectrum+": ", end="")
#             print(combin,f' counts are left over from binning (bin min. {group_min}) and will not be included when fitting or shown when plotted.')
        
#         return np.concatenate((np.array(binned_channel), remainder_bins)), np.array(binned_counts)# np.unique(np.array(binned_channel).flatten())

#     def _rebin_any_array(self, data, old_bins, new_bins, combine_by="sum"):
#         ''' Takes any array of data in old_bins space and rebins along data array axis==0 to have new_bins.
#         Can specify how the bins are combined.

#         Parameters
#         ----------
#         data, old_bins, new_bins : np.array
#                 Array of the data, current bins (for data axis==0), and new bins (for data axis==0). 
#                 Need len(data)==len(old_bins).

#         combine_by : string
#                 Defines how to combine multiple bins along axis 0. E.g., "sum" adds the data, "mean" averages 
#                 the data, and "quadrature" sums the data in quadrature.
#                 Default: "sum"

#         Returns
#         -------
#         The new bins and the corresponding grouped counts.
#         '''
#         new_binned_data = []
#         for nb in new_bins:
#             # just loop through new bins and bin data from between new_bin_lower<=old_bin_lowers and old_bin_highers<new_bin_higher
#             if combine_by=="sum":
#                 new_binned_data.append(np.sum(data[np.where((nb[0]<=old_bins[:,0]) & (nb[-1]>=old_bins[:,-1]))], axis=0))
#             elif combine_by=="mean":
#                 new_binned_data.append(np.mean(data[np.where((nb[0]<=old_bins[:,0]) & (nb[-1]>=old_bins[:,-1]))], axis=0))
#             elif combine_by=="quadrature":
#                 new_binned_data.append(np.sqrt(np.sum(data[np.where((nb[0]<=old_bins[:,0]) & (nb[-1]>=old_bins[:,-1]))]**2, axis=0)))
#         return np.array(new_binned_data)
    
#     def __add__(self, other):
#         ''' Define what adding means to the function, just combine other's loaded_spec_data with self's while changing other's 
#         spectrum numbers. E.g., self.loaded_spec_data={"spectrum1":...}, other.loaded_spec_data={"spectrum1":...}, 
#         then (self+other).loaded_spec_data={"spectrum1":...,"spectrum2":...}"spectrum2" was other's "spectrum1"'''

#         # combine loaded_spec_data attribute between classes
#         # can't just do a = {**b, **c} since keys need to change in the second dict
#         _other_spec_ = list(other.loaded_spec_data.keys())
#         _last_in_self_ = int(list(self.loaded_spec_data.keys())[-1].split("spectrum")[-1])
        
#         # combine the loaded_spec_data dicts 
#         new_self = deepcopy(self)
#         for c, key_other in enumerate(_other_spec_):
#             new_self.loaded_spec_data["spectrum"+str(_last_in_self_+1+c)] = other.loaded_spec_data[key_other]
            
#         return new_self
    
#     def __repr__(self):
#         '''Provide a representation to construct the class from scratch.'''
#         return self._construction_string
    
#     def __str__(self):
#         '''Provide a printable, user friendly representation of what the class contains.'''
#         _loadedspec = ""
#         plural = ["Spectrum", "is"] if len(self.loaded_spec_data.keys())==1 else ["Spectra", "are"]
#         tag = f"{plural[0]} Loaded {plural[1]}: "
#         for s in self.loaded_spec_data.keys():
#             _loadedspec += str(self.loaded_spec_data[s]["extras"]["pha.file"])+"\n"+" "*len(tag)
#         return f"No. of Spectra Loaded: {len(self.loaded_spec_data.keys())} \n{tag}{_loadedspec}"



class LoadSpec:
    def __init__(self, *args, pha_file=None, arf_file=None, rmf_file=None, srm_custom=None, custom_channel_bins=None):
        """
        This class's job is to load in spectral file(s). I.e., load in the count spectrum and then 
        calculate/store the SRM for fitting.

        Parameters
        ----------
        *args : dict
                Dictionaries for custom data to be passed to `CustomLoader`. These will be added before 
                any instrument file entries from `pha_file`.

        pha_file : string or list of strings
                The PHA file or list of PHA files for the spectrum to be loaded.

        arf_file, rmf_file : string or list of strings
                The ARF and RMF files associated with the PHA file(s). If none are given it is assumed 
                that these are in the same directory with same filename as the PHA file(s) but with 
                extensions '.arf' and '.rmf', respectively.

        srm_custom : 2d array
                User defined spectral response matrix. This is accepted over the SRM created from any 
                ARF and RMF files given.

        custom_channel_bins : 2d array
                User defined channel bins for the columns of the SRM matrix. 
                E.g., custom_channel_bins=[[1,1.5],[1.5,2],...]

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
        undo_rebin : int, specID, "all"
                Undo the rebinning. Move the original data from "extras" in loaded_spec_data attribute
                back to main part of the dict and set self._undo_rebin.

        Attributes
        ----------
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
        s.undo_rebin = 10
        """

        self._construction_string = f"LoadSpec(pha_file={pha_file},arf_file={arf_file},rmf_file={rmf_file},srm_custom={srm_custom},custom_channel_bins={custom_channel_bins})"
        
        # from sunxspex.sunxspex_fitting.instruments import * gives us the instrument specific loaders
        self.intrument_loaders = {"NuSTAR":NustarLoader, "STIX":StixLoader, "RHESSI":RhessiLoader}

        pha_file, arf_file, rmf_file, srm_custom, custom_channel_bins, instruments = self.sort_files(pha_file=pha_file, 
                                                                                                     arf_file=arf_file, 
                                                                                                     rmf_file=rmf_file, 
                                                                                                     srm_custom=srm_custom, 
                                                                                                     custom_channel_bins=custom_channel_bins)
        # get ready to load multiple spectra if needed
        num_of_files, num_of_custom = len(pha_file), len(args)
        self.loaded_spec_data = {}
        for s in range(num_of_files+num_of_custom):
            if s<num_of_custom:
                self.loaded_spec_data["spectrum"+str(s+1)] = CustomLoader(args[s])
            else:
                file_indx = s-num_of_custom
                self.loaded_spec_data["spectrum"+str(s+1)] = self.intrument_loaders[instruments[s]](pha_file[file_indx], 
                                                                                                    arf_file=arf_file[file_indx], 
                                                                                                    rmf_file=rmf_file[file_indx], 
                                                                                                    srm_custom=srm_custom[file_indx],
                                                                                                    custom_channel_bins=custom_channel_bins[file_indx])

        # Adding these classes should also yield {"spectrum1":..., "spectrum2":..., etc.}

    def sort_files(self, pha_file=None, arf_file=None, rmf_file=None, srm_custom=None, custom_channel_bins=None):
        # if only one observation is given then it won't be a list so make it one
        file_pha = _make_into_list(pha_file)
        file_arf = _make_into_list(arf_file)
        file_rmf = _make_into_list(rmf_file) 
        # the following should be numpy arrays so _make_into_list would turn the array to a list, not put it into a list
        custom_srm = srm_custom if type(srm_custom)==list else [srm_custom]
        custom_channel_bins = custom_channel_bins if type(custom_channel_bins)==list else [custom_channel_bins]
        
        # check if are and rmf and custom srm is either None in which case everything is found via the pha 
        #  file naming (a list of None the same length as the pha input will also acheive this) or if there 
        #  is a corresponding arf, rmf, and srm for every pha file
        assert ((type(arf_file)==type(None)) and (type(arf_file)==type(None)) and (len(file_pha)>=1)) \
                or \
                ((len(file_arf)==len(file_pha)) and (len(file_rmf)==len(file_pha))), \
                """Names are taken from the \"pha_file\" input if your \"arf_file\" and \"rmf_file\" are not 
                supplied. This means that if your \"arf_file\" and \"rmf_file\" are supplied then they can 
                either be of list length==1 or the same number of entries as your \"pha_file\" input."""
        
        assert (type(srm_custom)==type(None)) or (len(custom_srm)==1) or (len(custom_srm)==len(file_pha)), \
                """The \"srm_custom\" should either be None, 1, or the same length as the \"pha_file\" input."""
        
        assert (type(custom_channel_bins)==type(None)) or (len(custom_channel_bins)==1) or (len(custom_channel_bins)==len(file_pha)), \
                """The \"custom_channel_bins\" should either be None, 1, or the same length as the \"pha_file\" input."""
        
        # make sure lists of None are same length for inputs to self.load1spec()
        if (len(file_arf)==1) and (len(file_rmf)==1) and (len(file_pha)>1):
            file_arf *= len(file_pha)
            file_rmf *= len(file_pha)
        if (len(custom_srm)==1) and (len(file_pha)>1):
            custom_srm *= len(file_pha)
        if (len(custom_channel_bins)==1) and (len(file_pha)>1):
            custom_channel_bins *= len(file_pha)

        instruments = self.files2instruments(file_pha)

        return file_pha, file_arf, file_rmf, custom_srm, custom_channel_bins, instruments

    def files2instruments(self, pha_files):
        _instruments_names = []
        for pf in pha_files:
            with fits.open(pf) as hdul:
                if "TELESCOP" in hdul[0].header:
                    _instruments_names.append(hdul[0].header["TELESCOP"])
                else:
                    print("How do I know the instument?")
        return _instruments_names
        
    
    @property
    def rebin(self):
        ''' ***Property*** 
        Allows energy channels to be rebinned.

        Parameters
        ----------

        Returns
        -------
        None if the data has not been rebinned, the new energy bins for the rebinned data.
        '''
        if not hasattr(self, "_rebinned_edges"):
            return None
        return self._rebinned_edges

    @rebin.setter
    def rebin(self, group_mins):
        ''' ***Property Setter*** 
        Allows energy channels to be rebinned.

        Parameters
        ----------
        group_mins : int
                The minimum number of counts in a bin.

        Returns
        -------
        None.

        Example
        -------
        # load in 1 spectra, rebin the count channels to have a minimum of 10 counts
        s = LoadSpec(pha_file='filename1.pha',arf_file='filename1.arf',rmf_file='filename1.rmf')
        s.rebin = 10
        '''
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
        elif (type(group_mins) in (list, np.ndarray)):
            if len(group_mins)!=len(list(self.loaded_spec_data.keys())):
                print("rebin must be int or list of int with one-to-one match to the loaded spectra or dict with keys as the spectrum identifiers or with key \"all\".")
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
        
    def _rebin_check(self, spectrum):
        ''' Check if the spectrum given has been rebinned.

        Parameters
        ----------
        spectrum : string
                Spectrum to be checked. E.g., \'spectrum1\'

        Returns
        -------
        Boolean.
        '''
        _orig_in_extras = False if ("original_srm" not in self.loaded_spec_data[spectrum]["extras"]) else True
        return _orig_in_extras
        
    @property
    def undo_rebin(self):
        ''' ***Property*** 
        Allows the energy channel's rebinning to be undone.

        Parameters
        ----------

        Returns
        -------
        None.
        '''
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
                    if s_att not in ("effective_exposure", "extras"):
                        self.loaded_spec_data[spec][s_att] = self.loaded_spec_data[spec]["extras"]["original_"+s_att]
                        del self.loaded_spec_data[spec]["extras"]["original_"+s_att]
            else:
                print(f"Nothing to undo in {spec} as data has not been rebinned.")
    
    @undo_rebin.setter
    def undo_rebin(self, spectrum):
        ''' ***Property Setter*** 
        Allows the energy channel's rebinning to be undone.

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
        '''
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
        ''' Rebins the data and channels to return them.

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
        '''
        # check if data has been rebinned already
        _orig_in_extras = self._rebin_check(spectrum=spectrum)

        # get new bins and binned counts
        new_bins, new_counts = self.group_spec(spectrum=spectrum, group_min=group_min, _orig_in_extras=_orig_in_extras)

        # calculate the new widths, centres, count rates and count rate errors
        new_binning = np.diff(new_bins).flatten()
        bin_mids = np.mean(new_bins, axis=1)
        ctr = (new_counts / new_binning) / self.loaded_spec_data[spectrum]["effective_exposure"]
        ctr_err = (np.sqrt(new_counts) / new_binning) / self.loaded_spec_data[spectrum]["effective_exposure"]
        
        return new_bins, new_counts, new_binning, bin_mids, ctr, ctr_err, _orig_in_extras
    
    def _rebin_loaded_spec(self, spectrum, group_min, axis="count"):
        ''' Rebins all the relevant data for a spectrum and moves original information into the \'extras\' key in the loaded_spec_data attribute.

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
        New bin edges from teh rebinning process.
        '''
        
        if (axis=="count") or (axis=="photon_and_count"):
            new_bins, new_counts, new_binning, bin_mids, ctr, ctr_err, _orig_in_extras = self._rebin_data(spectrum, group_min)
        
        if not _orig_in_extras:
            # move original binning/counts/etc. into extras entry
            for s_att in self.loaded_spec_data[spectrum]().keys():
                if s_att not in ("effective_exposure", "extras"):
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
         
        # https://heasarc.gsfc.nasa.gov/docs/rosat/ros_xselect_guide_v1.1/node7.html#SECTION00712000000000000000
        # https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node94.html
        # https://heasarc.gsfc.nasa.gov/docs/asca/abc/node9.html#SECTION00940000000000000000
        # effectively need to replicate https://heasarc.gsfc.nasa.gov/docs/software/ftools/caldb/rbnrmf.html for rmf (then get srm)
        # the rebinning I think XSPEC uses internally https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/ftrbnrmf.html
        # good website for XSPEC commands https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/heasptools.html
        # self.loaded_spec_data[spectrum]["srm"] = self._rebin_srm(spectrum=spectrum, axis="count")
        self.loaded_spec_data[spectrum]["srm"] = self.loaded_spec_data[spectrum]._rebin_srm(axis="count")
        return new_bins
        
        
    def group_pha_finder(self, channels, counts, group_min=None, print_tries=False):
        ''' Takes the counts, and checks the bins left over from grouping the bins with a minimum 
        value.

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
        '''

        if type(group_min)!=int or group_min<=0: 
            print('The \'group_min\' parameter must be an integer > 0.')
            return

        # grppha groups in counts, not counts s^-1 or anything
        total_counts = np.sum(counts)

        while True:
            binned_channel = []
            combin = 0
            # check if we can make it through the whole count list with 0 left over when grouping with group_min
            for c, count in enumerate(counts):
                if count>=group_min and combin==0:
                    binned_channel.append(channels[c])
                else:
                    if combin==0:
                        start_e_bin = channels[c][0] 
                    combin += count
                    if combin >= group_min:
                        binned_channel.append([start_e_bin, channels[c][1]]) # starting at the last bin edge and the last edge of the bin we're on
                        combin = 0

            if print_tries == True:
                print('Group min, ', group_min, ', has counts left over: ', combin)

            if combin != 0:
                group_min += 1
            elif group_min == total_counts:
                print('The minimum group number being tried is the same as the total number of counts.')
                return
            else:
                print('Group minimum that works is: ', group_min)
                return np.array(binned_channel), group_min
            
    
    def group_spec(self, spectrum, group_min=None, _orig_in_extras=False):
        ''' Takes the counts, and checks the bins left over from grouping the bins with a minimum 
        value.

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
        The bin edges for you minimum group number. Any bins left over are now included.
        '''

        counts = self.loaded_spec_data[spectrum]["counts"] if not _orig_in_extras else self.loaded_spec_data[spectrum]["extras"]["original_counts"]
        channel_bins = self.loaded_spec_data[spectrum]["count_channel_bins"] if not _orig_in_extras else self.loaded_spec_data[spectrum]["extras"]["original_count_channel_bins"]
        
        
        return self.group_cts(channel_bins, counts, group_min=group_min, spectrum=spectrum)
    
    def group_cts(self, channel_bins, counts, group_min=None, spectrum=None, verbose=True):
        ''' Takes the counts and bins and groups the counts to have a minimum number of group_min.

        Parameters
        ----------
        channel_bins, counts : np.array
                Array of the channel bins and counts.

        group_min : Int
                The minimum number of counts allowed in a bin. This input is a starting number and the is checked 
                incrementally.
                Default: None

        spectrum : string
                SRM's spectrum identifier to be rebinned. E.g., \'spectrum1\'.

        verbose : Bool
                If True, will print out the number counts not able to be binned.
                Default: True

        Returns
        -------
        The new bins and the corresponding grouped counts.
        '''

        if type(group_min)!=int or group_min<=0: 
            if type(group_min)==type(None):
                return channel_bins, counts
            print('The \'group_min\' parameter must be an integer > 0.')
            return channel_bins, counts
        
        # grppha groups in counts, not counts s^-1 or anything

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
                
        # since SRM is going to be binned, add the rest of the photon bins on at the end with native binning
        # any counts in these bins will be ignored, only 0s taken into consideration in photon space for these
        remainder_bins = channel_bins[np.where(channel_bins[:,1] > np.array(binned_channel)[-1,-1])]
        binned_counts += [0]*len(remainder_bins)
                    
        if combin>0 and verbose:
            if type(spectrum)!=type(None):
                print("In "+spectrum+": ", end="")
            print(combin,f' counts are left over from binning (bin min. {group_min}) and will not be included when fitting or shown when plotted.')
        
        return np.concatenate((np.array(binned_channel), remainder_bins)), np.array(binned_counts)# np.unique(np.array(binned_channel).flatten())
    
    def __add__(self, other):
        ''' Define what adding means to the function, just combine other's loaded_spec_data with self's while changing other's 
        spectrum numbers. E.g., self.loaded_spec_data={"spectrum1":...}, other.loaded_spec_data={"spectrum1":...}, 
        then (self+other).loaded_spec_data={"spectrum1":...,"spectrum2":...}"spectrum2" was other's "spectrum1"'''

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
        '''Provide a representation to construct the class from scratch.'''
        return self._construction_string
    
    def __str__(self):
        '''Provide a printable, user friendly representation of what the class contains.'''
        _loadedspec = ""
        plural = ["Spectrum", "is"] if len(self.loaded_spec_data.keys())==1 else ["Spectra", "are"]
        tag = f"{plural[0]} Loaded {plural[1]}: "
        for s in self.loaded_spec_data.keys():
            if "pha.file" in self.loaded_spec_data[s]["extras"]:
                _loadedspec += str(self.loaded_spec_data[s]["extras"]["pha.file"])+"\n"+" "*len(tag)
            else:
                _loadedspec += str(None)+"\n"+" "*len(tag)
        return f"No. of Spectra Loaded: {len(self.loaded_spec_data.keys())} \n{tag}{_loadedspec}"


# # Get a default class for the instrument specfic loaders
# # Once the instrument specific loaders inherit from this then all they really have to do is get the spectral
# #    data they want to fit in the correct dictionary form and assigned to `self._loaded_spec_data`.
# class LoaderBlueprint:

#     def _rebin_rmf(self, matrix, old_count_bins=None, new_count_bins=None, old_photon_bins=None, new_photon_bins=None, axis="count"):
#         ''' Rebins the photon and/or count channels of the redistribution matrix if needed.
#         This will rebin any 2d array by taking the mean across photon space (rows) and summing
#         across count space (columns).

#         If no effective area information from the instrument then this is passed straight 
#         to `_rebin_srm`, if there is then the `_rebin_srm` should be overwritten.

#         Parameters
#         ----------
#         matrix : 2d array
#                 Redistribution matrix. 
        
#         old_count_bins, new_count_bins : 1d arrays
#                 The old count channel binning and the new binning to be for the redistribution matrix columns (sum columns).

#         old_photon_bins, new_photon_bins : 1d arrays
#                 The old photon channel binning and the new binning to be for the redistribution matrix columns (average rows).

#         axis : string
#                 Define what \'axis\' the binning should be applied to. E.g., \'photon\', \'count\', or \'photon_and_count\'.

#         Returns
#         -------
#         The rebinned 2d redistribution matrix.
#         '''
#         # across channel bins, we sum. across energy bins, we average
#         # appears to be >2x faster to average first then sum if needing to do both
#         if (axis=="photon") or (axis=="photon_and_count"):
#             # very slight difference to rbnrmf when binning across photon axis, <2% of entries have a ratio (my way/rbnrmf) >1 (up to 11)
#             # all come from where the original rmf has zeros originally so might be down to precision being worked in, can't expect the exact same numbers essentially
#             matrix = rebin_any_array(data=matrix, old_bins=old_photon_bins, new_bins=new_photon_bins, combine_by="mean")
#         if (axis=="count") or (axis=="photon_and_count"):
#             matrix = rebin_any_array(data=matrix.T, old_bins=old_count_bins, new_bins=new_count_bins, combine_by="sum").T # need to go along columns so .T then .T back
        
#         return matrix

#     def _channel_bin_info(self, axis):
#         old_count_bins, new_count_bins, old_photon_bins, new_photon_bins = None, None, None, None
#         if (axis=="count") or (axis=="photon_and_count"):
#             old_count_bins = self._loaded_spec_data["extras"]["original_count_channel_bins"]
#             new_count_bins = self._loaded_spec_data["count_channel_bins"] 
#         if (axis=="photon") or (axis=="photon_and_count"):
#             old_photon_bins = self._loaded_spec_data["extras"]["orignal_photon_channel_bins"]
#             new_photon_bins = self._loaded_spec_data["photon_channel_bins"]
#         return old_count_bins, new_count_bins, old_photon_bins, new_photon_bins

#     def _rebin_srm(self, axis="count"):
#         ''' Rebins the photon and/or count channels of the spectral response matrix (SRM) if needed. 
        
#         Note: If the instrument has a spatial aspect and effective information is present (e.g., 
#         NuSTAR from its ARF file) then this method should be overwritten in the instrument 
#         specific loader in order to rebin the redistribution matrix and effective area separately
#         before re-construction the new SRM.

#         Parameters
#         ----------
#         matrix : 2d array
#                 Spectral response matrix. 
        
#         old_count_bins, new_count_bins : 1d arrays
#                 The old count channel binning and the new binning to be for the spectral response matrix columns (sum columns).

#         old_photon_bins, new_photon_bins : 1d arrays
#                 The old photon channel binning and the new binning to be for the spectral response matrix columns (average rows).

#         axis : string
#                 Define what \'axis\' the binning should be applied to. E.g., \'photon\', \'count\', or \'photon_and_count\'.

#         Returns
#         -------
#         The rebinned 2d spectral response matrix.
#         '''
#         old_count_bins, new_count_bins, old_photon_bins, new_photon_bins = self._channel_bin_info(axis)
#         matrix = self._loaded_spec_data["srm"]
#         return self._rebin_rmf(matrix, old_count_bins=old_count_bins, new_count_bins=new_count_bins, old_photon_bins=old_photon_bins, new_photon_bins=new_photon_bins, axis="count")

#     def __getitem__(self, item):
#         '''Index the entries in `_loaded_spec_data`'''
#         return self._loaded_spec_data[item]

#     def __setitem__(self, item, new_value):
#         '''Allows entries in `_loaded_spec_data` to be changed.'''
#         self._loaded_spec_data[item] = new_value

#     def __call__(self):
#         '''When the class is called (n=NustarLoader()->n()) then `_loaded_spec_data` 
#         is returned.'''
#         return self._loaded_spec_data

#     def __repr__(self):
#         '''String representation of `_loaded_spec_data`.'''
#         return str(self._loaded_spec_data)


# # Instrument specific data loaders
# #    As long as these loaders get the spectral data to fit into the correct dictionary form and assigned to self._loaded_spec_data then
# #    they should work but they can also overwrite the _rebin_srm(self, axis="count") method if the SRM rebinning is instrument specific.
# #       The benefit here is that the class can have other methods/properties/setters (like time selection for STIX/RHESSI;e.g., 
# #    .select_time(new_time)?) which can be accessed at the user level easily when fitting through the loaded_spec_data attribute 
# #    (e.g., .loaded_spec_data["spectrum1"].select_time(new_time)).


# class NustarLoader(LoaderBlueprint):
#     def __init__(self, pha_file, arf_file=None, rmf_file=None, srm_custom=None, custom_channel_bins=None):
#         self._construction_string = f"NustarLoader(pha_file={pha_file},arf_file={arf_file},rmf_file={rmf_file},srm_custom={srm_custom},custom_channel_bins={custom_channel_bins})"
#         self._loaded_spec_data = self.load1spec(pha_file, f_arf=arf_file, f_rmf=rmf_file, srm=srm_custom, channel_bins=custom_channel_bins)

#     def load1spec(self, f_pha, f_arf=None, f_rmf=None, srm=None, channel_bins=None):
#         ''' Loads all the information in for a given spectrum.

#         Parameters
#         ----------
#         f_pha, f_arf, f_rmf : string
#                 Filenames for the relevant spectral files. If f_arf, f_rmf are None it is assumed 
#                 that these are in the same directory with same filename as the PHA file but with 
#                 extensions '.arf' and '.rmf', respectively.
#                 Default of f_arf, f_rmf: None

#         srm : 2d array
#                 User defined spectral response matrix. This is accepted over the SRM created from any 
#                 ARF and RMF files given.
#                 Default: None

#         channel_bins: 2d array
#                 User defined channel bins for the columns of the SRM matrix. 
#                 E.g., custom_channel_bins=[[1,1.5],[1.5,2],...]
#                 Default: None

#         Returns
#         -------
#         Dictionary of the loaded in spectral information in the form {"photon_channel_bins":channel_bins, 
#                                                                       "photon_channel_mids":np.mean(channel_bins, axis=1), 
#                                                                       "photon_channel_binning":channel_binning, 
#                                                                       "count_channel_bins":channel_bins, 
#                                                                       "count_channel_mids":np.mean(channel_bins, axis=1), 
#                                                                       "count_channel_binning":channel_binning, 
#                                                                       "counts":counts, 
#                                                                       "count_rate":count_rate, 
#                                                                       "count_rate_error":count_rate_error, 
#                                                                       "effective_exposure":eff_exp,
#                                                                       "srm":srm,
#                                                                       "extras":{"pha.file":f_pha,
#                                                                                 "arf.file":f_arf,
#                                                                                 "arf.e_lo":e_lo_arf,
#                                                                                 "arf.e_hi":e_hi_arf,
#                                                                                 "arf.effective_area":eff_area,
#                                                                                 "rmf.file":f_rmf,
#                                                                                 "rmf.e_lo":e_lo_rmf,
#                                                                                 "rmf.e_hi":e_hi_rmf,
#                                                                                 "rmf.ngrp":ngrp,
#                                                                                 "rmf.fchan":fchan,
#                                                                                 "rmf.nchan":nchan,
#                                                                                 "rmf.matrix":matrix,
#                                                                                 "rmf.redistribution_matrix":redist_m} 
#                                                                      }.
#         '''
        
#         # what files might be needed (for NuSTAR)
#         f_arf = f_pha[:-3]+"arf" if type(f_arf)==type(None) else f_arf
#         f_rmf = f_pha[:-3]+"rmf" if type(f_rmf)==type(None) else f_rmf
        
#         # need effective exposure and energy binning since likelihood works on counts, not count rates etc.
#         _, counts, eff_exp = nu_spec.read_pha(f_pha)
        
#         # now calculate the SRM or use a custom one if given
#         if type(srm)==type(None):
        
#             # if there is an ARF file load it in
#             if os_path.isfile(f_arf):
#                 e_lo_arf, e_hi_arf, eff_area = nu_spec.read_arf(f_arf)

#             # if there is an RMF file load it in and convert to a redistribution matrix
#             if os_path.isfile(f_rmf):
#                 e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m = self.load_rmf(f_rmf)
        
#             srm = nu_spec.make_srm(rmf_matrix=redist_m, arf_array=eff_area)
#         else:
#             e_lo_arf, e_hi_arf, eff_area = None, None, None
#             e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m = None, None, None, None, None, None, None
        
        
#         channel_bins = self.calc_channel_bins(e_lo_rmf, e_hi_rmf) if type(channel_bins)==type(None) else channel_bins
#         channel_binning = np.diff(channel_bins).flatten()  
        
#         # get the count rate information
#         count_rate, count_rate_error = nu_spec.flux_cts_spec(f_pha, bin_size=channel_binning)
            
#         # what spectral info you want to know from this observation
#         return {"photon_channel_bins":channel_bins, 
#                 "photon_channel_mids":np.mean(channel_bins, axis=1), 
#                 "photon_channel_binning":channel_binning, 
#                 "count_channel_bins":channel_bins, 
#                 "count_channel_mids":np.mean(channel_bins, axis=1), 
#                 "count_channel_binning":channel_binning, 
#                 "counts":counts, 
#                 "count_rate":count_rate, 
#                 "count_rate_error":count_rate_error, 
#                 "effective_exposure":eff_exp,
#                 "srm":srm,
#                 "extras":{"pha.file":f_pha,
#                           "arf.file":f_arf,
#                           "arf.e_lo":e_lo_arf,
#                           "arf.e_hi":e_hi_arf,
#                           "arf.effective_area":eff_area,
#                           "rmf.file":f_rmf,
#                           "rmf.e_lo":e_lo_rmf,
#                           "rmf.e_hi":e_hi_rmf,
#                           "rmf.ngrp":ngrp,
#                           "rmf.fchan":fchan,
#                           "rmf.nchan":nchan,
#                           "rmf.matrix":matrix,
#                           "rmf.redistribution_matrix":redist_m} 
#                 } # this might make it easier to add different observations together

#     def load_rmf(self, rmf_file):
#         ''' Extracts all information, mainly the redistribution matrix ([counts/photon]) from a given RMF file.

#         Parameters
#         ----------
#         rmf_file : string
#                 The file path and name of the RMF file.

#         Returns
#         -------
#         The lower/higher photon bin edges (e_lo_rmf, e_hi_rmf), the number of counts channels activated by each photon channel (ngrp), 
#         starting indices of the count channel groups (fchan), number counts channels from each starting index (nchan), the coresponding 
#         counts/photon value for each count and photon entry (matrix), and the redistribution matrix (redist_m: with rows of photon channels, 
#         columns of counts channels, and in the units of counts/photon).
#         '''
        
#         e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix = nu_spec.read_rmf(rmf_file)
#         fchan_array = nu_spec.col2arr_py(fchan)
#         nchan_array = nu_spec.col2arr_py(nchan)
#         redist_m = nu_spec.vrmf2arr_py(data=matrix,  
#                                        n_grp_list=ngrp,
#                                        f_chan_array=fchan_array, 
#                                        n_chan_array=nchan_array) # 1.5 s of the total 2.4 s (1spec) is spent here
        
#         return e_lo_rmf, e_hi_rmf, ngrp, fchan, nchan, matrix, redist_m

#     def calc_channel_bins(self, e_low, e_hi):
#         ''' Calculates the count channel bins from the given rmf files. Assumes that the photon and count 
#         channel bins are the same.

#         Parameters
#         ----------
#         e_low : 1d array
#                 Array of the lower bounds of all the channel bins.

#         e_hi : 1d array
#                 Array of the higher bounds of all the channel bins.

#         Returns
#         -------
#         None if no e_low or e_hi is given or 2d array where each row is the lower and higher bound of that bin.
#         '''
#         if (e_low is None) or (e_hi is None):
#             print("If no rmf/arf files are given and a custom srm is provided, please provide the custom_channel_bins.\nE.g., custom_channel_bins=[[1,2],[2,3],...]")
#             return None
#         else:
#             return np.stack((e_low, e_hi), axis=-1)

#     def _rebin_srm(self, axis="count"):
#         ''' Rebins the photon and/or count channels of the spectral response matrix by rebinning the redistribution 
#         matrix and the effective area array.

#         Parameters
#         ----------
#         axis : string
#                 Define what \'axis\' the binning should be applied to. E.g., \'photon\', \'count\', or \'photon_and_count\'.

#         Returns
#         -------
#         The rebinned 2d spectral response matrix.
#         '''
#         old_count_bins, new_count_bins, old_photon_bins, new_photon_bins = self._channel_bin_info(axis)
        
#         old_rmf = self._loaded_spec_data["extras"]["rmf.redistribution_matrix"]
#         old_eff_area = self._loaded_spec_data["extras"]["arf.effective_area"]
        
#         # checked with ftrbnrmf
#         new_rmf = self._rebin_rmf(matrix=old_rmf, 
#                                     old_count_bins=old_count_bins, 
#                                     new_count_bins=new_count_bins, 
#                                     old_photon_bins=old_photon_bins, 
#                                     new_photon_bins=new_photon_bins, 
#                                     axis=axis)
        
#         # average eff_area, checked with ftrbnarf
#         new_eff_area = rebin_any_array(data=old_eff_area, old_bins=old_photon_bins, new_bins=new_photon_bins, combine_by="mean") if (axis!="count") else old_eff_area
#         return nu_spec.make_srm(rmf_matrix=new_rmf, arf_array=new_eff_area)
        

# class StixLoader(LoaderBlueprint):
#     def __init__(self):
#         self._loaded_spec_data = {}

# class RhessiLoader(LoaderBlueprint):
#     def __init__(self):
#         self._loaded_spec_data = {}

# class CustomLoader(LoaderBlueprint):
#     def __init__(self, spec_data_dict):

#         # needed keys
#         ess_keys = ["count_channel_bins",
#                     "counts"]
#         non_ess_keys = ["photon_channel_bins",
#                         "photon_channel_mids",
#                         "photon_channel_binning",
#                         "count_channel_mids",
#                         "count_channel_binning",
#                         "count_rate",
#                         "count_rate_error",
#                         "effective_exposure",
#                         "srm",
#                         "extras"]
        
#         # check essential keys are given
#         essentials_not_present = set(ess_keys)-set(list(spec_data_dict.keys()))
#         assert len(essentials_not_present)==0, f"Essential dict. entries are not present: {essentials_not_present}"
        
#         # check non-essential keys are given, if not then defaults are 1s
#         nonessentials_not_present = set(non_ess_keys)-set(list(spec_data_dict.keys()))
#         _def = self._nonessential_defaults(nonessentials_not_present,spec_data_dict["count_channel_bins"],spec_data_dict["counts"])
#         _def.update(spec_data_dict)

#         self._loaded_spec_data = _def

#     def _nonessential_defaults(self,nonessential_list,count_channels,counts):
#         if len(nonessential_list)>0:
#             _count_length_default = np.ones(len(count_channels))
#             defaults = {"photon_channel_bins":count_channels,
#                         "photon_channel_mids":_count_length_default,
#                         "photon_channel_binning":_count_length_default,
#                         "count_channel_mids":_count_length_default,
#                         "count_channel_binning":_count_length_default,
#                         "count_rate":counts,
#                         "count_rate_error":_count_length_default,
#                         "effective_exposure":1,
#                         "srm":np.identity(len(counts)),
#                         "extras":{}}
#             return defaults
#         else:
#             return {}


def isnumber(word):
    """ Checks if a string is a string of a number.

    Parameters
    ----------
    word : string
            String of the possible number.

    Returns
    -------
    Boolean.
    """
    try:
        float(word)
    except ValueError:
        return False
    return True

# def rebin_any_array(data, old_bins, new_bins, combine_by="sum"):
#     ''' Takes any array of data in old_bins space and rebins along data array axis==0 to have new_bins.
#     Can specify how the bins are combined.

#     Parameters
#     ----------
#     data, old_bins, new_bins : np.array
#             Array of the data, current bins (for data axis==0), and new bins (for data axis==0). 
#             Need len(data)==len(old_bins).

#     combine_by : string
#             Defines how to combine multiple bins along axis 0. E.g., "sum" adds the data, "mean" averages 
#             the data, and "quadrature" sums the data in quadrature.
#             Default: "sum"

#     Returns
#     -------
#     The new bins and the corresponding grouped counts.
#     '''
#     new_binned_data = []
#     for nb in new_bins:
#         # just loop through new bins and bin data from between new_bin_lower<=old_bin_lowers and old_bin_highers<new_bin_higher
#         if combine_by=="sum":
#             new_binned_data.append(np.sum(data[np.where((nb[0]<=old_bins[:,0]) & (nb[-1]>=old_bins[:,-1]))], axis=0))
#         elif combine_by=="mean":
#             new_binned_data.append(np.mean(data[np.where((nb[0]<=old_bins[:,0]) & (nb[-1]>=old_bins[:,-1]))], axis=0))
#         elif combine_by=="quadrature":
#             new_binned_data.append(np.sqrt(np.sum(data[np.where((nb[0]<=old_bins[:,0]) & (nb[-1]>=old_bins[:,-1]))]**2, axis=0)))
#     return np.array(new_binned_data)