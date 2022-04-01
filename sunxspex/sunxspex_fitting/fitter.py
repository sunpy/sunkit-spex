"""
The following code hosts the main class that handles all processes relevant to fitting.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from mpl_toolkits.axes_grid1 import make_axes_locatable

## to fit the model
from scipy.optimize import minimize

## for MCMC
import emcee
import corner

# for nested sampling
import nestle

# run mcmc in parallel
from multiprocessing import Pool
##import### dill

# function construction packages
import inspect
import re
from keyword import iskeyword
import itertools

from scipy.interpolate import interp1d

from copy import copy, deepcopy

import types
import math as maths

# for colour list cycling
from itertools import cycle

# for Hessian matrix calculation for the minimise errors
import numdifftools as nd
from scipy.linalg import LinAlgError
import pickle

# let's not have warnings for every /0 when we have 0 counts
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from sunxspex.sunxspex_fitting.rainbow_text import rainbow_text_lines
from sunxspex.sunxspex_fitting.photon_models_for_fitting import (defined_photon_models, f_vth, thick_fn, thick_warm)
from sunxspex.sunxspex_fitting.likelihoods import LogLikelihoods
from sunxspex.sunxspex_fitting.data_loader import LoadSpec
from sunxspex.sunxspex_fitting.instruments import rebin_any_array
from sunxspex.sunxspex_fitting.parameter_handler import Parameters, isnumber


__all__ = ["add_var", "del_var","add_photon_model", "del_photon_model", "SunXspex", "load"]

DYNAMIC_FUNCTION_SOURCE = {}

# models should return a photon spectrum with units photons s^-1 cm^-2 keV^-1
def add_photon_model(function, overwrite=False):
    """ Add user photon model to fitting namespace.

    Takes a user defined function intended to be used as a model or model component when giving a
    string to the SunXspex.model property. Puts defined_photon_models[function.__name__]=param_inputs
    in `defined_photon_models` for it to be known to the fititng code. The energies argument must be
    first and accept photon bins.

    The given function needs to have parameters as arguments then \'energies\' as a keyword argument
    where energies accepts the energy bin array. E.g.,

    .. math::
     gauss = = a$\cdot$e$^{-\frac{(energies - b)^{2}}{2 c^{2}}}$

    would be
     `gauss = lambda a, b, c, energies=None: a * np.exp(-((np.mean(energies, axis=1)-b)**2/(2*c**2)))`

    Parameters
    ----------
    function : function
            The function object.

    overwrite : bool
            Set True to overwrite a model that already exists. User needs to be explicit if they wish
            to overwrite their models.
            Default: False

    Returns
    -------
    None.

    Example
    -------
    from fitter import SunXspex, defined_photon_models, add_photon_model

    # Define gaussian model (doesn't have to be a lambda function)
    gauss = lambda a, b, c, energies=None: a * np.exp(-((np.mean(energies, axis=1)-b)**2/(2*c**2)))

    # Add the gaussian model to fitter.py namespace
    add_photon_model(gauss)

    # Now can use it in fitting with string defined model. Will be plotted separately to the total model
    Sx = SunXspex(pha_file=[...])
    Sx.model = "gauss+gauss"
    Sx.fit()
    Sx.plot()
    """
    # if user wants to define any component model to use in the fitting and reference as a string
    _glbls_cp, _dfs_cp = copy(globals()), copy(DYNAMIC_FUNCTION_SOURCE)
    # check if lambda function and return the user function
    # a 'self-contained' check will also take place and will fail if function is not of the form f(...,energies=None)
    usr_func = deconstruct_lambda(function, add_underscore=False)
    param_inputs, _ = get_func_inputs(function) # get the param inputs for the function
    # check if the function has already been added
    if (usr_func.__name__ in defined_photon_models.keys()) and not overwrite:
        print("Model: \'", usr_func.__name__,"\' already in \'defined_photon_models\'. Please set `overwrite=True`\nor use `del_photon_model()` to remove the existing model entirely.")
        # revert changes back, model needs to be initialised to know its name to hceck if it already existed but doing this overwrites it if it is there
        globals()[usr_func.__name__] = _glbls_cp[usr_func.__name__]
        DYNAMIC_FUNCTION_SOURCE[usr_func.__name__] = _dfs_cp[usr_func.__name__]
        return

    # if overwrite is True and it gets to this stage remove the function entry from defined_photon_models quietly, else this line does nothing
    defined_photon_models.pop(usr_func.__name__, None)

    # make list of all inputs for all already defined models
    def_pars = list(itertools.chain.from_iterable(defined_photon_models.values()))
    assert len(set(def_pars)-set(param_inputs))==len(def_pars), f"Please use different parameter names to the ones already defined: {def_pars}"

    # add user model to defined_photon_models from photon_models_for_fitting
    defined_photon_models[usr_func.__name__] = param_inputs
    print(f"Model {usr_func.__name__} added.")

def del_photon_model(function_name):
    """ Remove user defined sub-models that have been added via `add_photon_model`.

    Parameters
    ----------
    function_name : str
            Name of the function to be removed.

    Returns
    -------
    None.

    Example
    -------
    from fitter import add_photon_model, del_photon_model

    # Define gaussian model (doesn't have to be a lambda function)
    gauss = lambda a, b, c, energies=None: a * np.exp(-((np.mean(energies, axis=1)-b)**2/(2*c**2)))

    # Add the gaussian model to fitter.py namespace
    add_photon_model(gauss)

    # realise the model is wrong or just want it removed
    del_photon_model("gauss")
    """
    # quickly check if the function exists under function_name
    if function_name not in defined_photon_models:
        print(function_name, "is not in `defined_photon_models` to be removed.")
        return

    # can only remove if the user added the model, defined models from photon_models_for_fitting.py are protected
    if inspect.getmodule(globals()[function_name]).__name__ in (__name__):
        del defined_photon_models[function_name], globals()[function_name], DYNAMIC_FUNCTION_SOURCE[function_name]
        print(f"Model {function_name} removed.")
    else:
        print("Default models imported from sunxspex.sunxspex_fitting.photon_models_for_fitting are protected.")


DYNAMIC_VARS = {}
def add_var(overwrite=False, quiet=False, **user_kwarg):
    """ Add user variable to fitting namespace.

    Takes user defined variables and makes them available to the models being used within the
    fitting. E.g., the user could define a variable in their own namespace (obtained from a file?)
    and, instead of loading the file in with every function call, they can add the variable using
    this method.

    Parameters
    ----------
    overwrite : bool
            Set True to overwrite an argument that already exists. User needs to be explicit if they wish
            to overwrite their arguments.
            Default: False

    quiet : bool
            Suppress any print statement to announce the variable has been added or not added. To be used
            when loading session back in as if the variables were save then they were fine to add in the
            first place.
            Default: False

    **user_kwarg :
            User added variables. Arrays, lists, constants, etc., to be used in user defined models. This
            enables sessions with complex models (say that use a constants from a file) to still be save
            and work nroally when loaded back in.

    Returns
    -------
    None.

    Example
    -------
    from fitter import SunXspex, defined_photon_models, add_photon_model, add_var

    # the user variable that might be too costly to run every function call or too hard to hard code
    some_user_var = something_complicated

    # Define gaussian model (doesn't have to be a lambda function), but include some user variable outside the scope of the model
    gauss = lambda a, b, c, energies=None: a * np.exp(-((np.mean(energies, axis=1)-b)**2/(2*c**2))) * some_user_var

    # add user variable
    add_var(some_user_var=some_user_var)

    # Add the gaussian model to fitter.py namespace
    add_photon_model(gauss)

    # Now can use it in fitting with string defined model. Will be plotted separately to the total model
    Sx = SunXspex(pha_file=[...])
    Sx.model = "gauss+gauss"
    Sx.fit()
    Sx.plot()
    """
    for k,i in user_kwarg.items():
        if k in DYNAMIC_VARS and not overwrite:
            vb = f"Variable {k} already exists. Please set `overwrite=True`, delete this with `del_var({k})`,\nor use a different variable name."
        elif not k in DYNAMIC_VARS and k in globals():
            vb = f"Argument name {k} already exists **in globals** and is not a good idea to overwrite. Please use a different variable name."
        else:
            DYNAMIC_VARS.update({k:i})
            globals().update({k:i})
            vb = f"Variable {k} added."
        if not quiet:
            print(vb)

def del_var(*user_arg_name):
    """ Remove user defined variables that have been added via `add_var`.

    Parameters
    ----------
    *user_arg_name : str
            Name(s) of the variable(s) to be removed.

    Returns
    -------
    None.

    Example
    -------
    from fitter import add_var, del_var

    # the user variable that might be too costly to run every function call or too hard to hard code
    some_user_var = something_complicated

    # add user variable
    add_var(some_user_var=some_user_var)

    # realise the variable should be there or want it gome to update it
    del_var("some_user_var")
    """
    _removed, _not_removed = [], []
    for uan in user_arg_name:
        if uan in DYNAMIC_VARS:
            del globals()[uan], DYNAMIC_VARS[uan]
            _removed.append(uan)
        else:
            _not_removed.append(uan)
    _rmstr, spc = (f"Variables {_removed} were removed.", "\n") if len(_removed)>0 else ("", "")
    _nrmstr, spc  = (f"Variables {_not_removed} are not ones added by user and so were not removed.", spc) if len(_not_removed)>0 else ("", "")
    print(_rmstr, _nrmstr, sep=spc)

# Easily access log-likelihood/fit-stat methods from the one place, if SunXpsex class inherits this then data is duplicated
LL_CLASS = LogLikelihoods()

class SunXspex(LoadSpec):
    """
    Load's in spectral file(s) and then provide a framework for fitting models to the spectral data.

    Parameters
    ----------
    *args : dict
            Dictionaries for custom data to be passed to `sunxspex.sunxspex_fitting.instruments.CustomLoader`.
            These will be added before any instrument file entries from `pha_file`.

    pha_file : string or list of strings
            The PHA file or list of PHA files for the spectrum to be loaded.
            See LoadSpec class.

    arf_file, rmf_file : string or list of strings
            The ARF and RMF files associated with the PHA file(s). If none are given it is assumed
            that these are in the same directory with same filename as the PHA file(s) but with
            extensions '.arf' and '.rmf', respectively.
            See LoadSpec class.

    srm_file : string
            The file that contains the spectral response matrix for the given spectrum.
            See LoadSpec class.

    srm_custom : 2d array
            User defined spectral response matrix. This is accepted over the SRM created from any
            ARF and RMF files given.
            See LoadSpec class.

    custom_channel_bins, custom_photon_bins : 2d array
            User defined channel bins for the columns and rows of the SRM matrix.
            E.g., custom_channel_bins=[[1,1.5],[1.5,2],...]
            See LoadSpec class.

    **kwargs : Passed to LoadSpec.

    Properties
    ----------
    burn_mcmc : Int
            Returns the burn-in number used with the MCMC samples (has setter).

    confidence_range : 0<float<=1
            Returns the confidence range used in the MCMC analysis; default 0.6827 (has setter).

    energy_fitting_range : dict
            Returns the defined fitting ranges for each spectrum. Default [0,np.inf] for all (has setter).

    model : function object
            Returns self._model which is the functional model used for fitting (has setter).

    rebin : list/array
            Returns the new energy bins of the data (self._rebinned_edges), None if the has not been rebinned (has setter).
            See LoadSpec class.

    renew_model : None
            Only exists to allow the setter to be defined (has setter).

    show_params : astropy table
            Returns an table showing all model spectral parameter information.

    show_rParams : astropy table
            Returns an table showing all response parameter information.

    undo_burn_mcmc : None
            Undoes any burn-in applied to the calculated MCMC samples.

    undo_rebin : None
            Has the code that uses self._undo_rebin to undo the rebinning for spectra (has setter).
            See LoadSpec class.

    update_model : None
            Only exists to allow the setter to be defined (has setter).

    Setters
    -------
    burn_mcmc : Int>0
            Applies a burn-in to the calculated MCMC sampels.

    confidence_range : 0<float<=1
            Set to the confidence range used in the MCMC analysis (default 0.6827). Setting this after the MCMC has run
            will still update the parameter/mcmc tables.

    energy_fitting_range : dict or list
            A dictionary of spectrum identifier and fitting range or list of range for all spectra (default [0,np.inf]).

    model : function object or str
            Model function or mathematical string using the names in defined_photon_models.

    rebin : int, {specID:int}, {"all":int}
            Minimum number of counts in each bin. Changes data but saves original in "extras" key
            in loaded_spec_data attribute.
            See LoadSpec class.

    renew_model : function object or str
            Same input as model setter. Defines the model function again without changing the parameter tables, etc. To be
            used if model function couldn't be save to then be loaded back in by the user. At the start of the next session,
            once the class has been loaded back in, then this setter can be used to renew the complicated user defined model.

    undo_rebin : int, specID, "all"
            Undo the rebinning. Move the original data from "extras" in loaded_spec_data attribute
            back to main part of the dict and set self._undo_rebin.
            See LoadSpec class.

    update_model : function object or str
            Same input as model setter. Saves previous model info and resets model setter.

    Methods
    -------
    corner_mcmc : _fix_titles (bool), **kwargs
            Produces a corner plot of the MCMC run. The kwargs are passed to `corner.corner`.

    fit : **kwargs
            Used to initiate the spectral fitting. The kwargs are passed to minimiser, `bounds` kwarg is overwritten,
            `_hess_step` can be passed to hessian calculation for errors.

    group : channel_bins (array (n,2)), counts (array (n)), group_min (int)
            Groups bins so they have at least a `group_min` number of counts.
            See LoadSpec class.

    group_pha_finder : channels (array (n,2)), counts (array (n)), group_min (int), print_tries (bool)
            Check, incrementally from a minimum number, what group minimum is needed to leave no counts unbinned
            after rebinning. Returns binned channels and the minimum group number if one exists.
            See LoadSpec class.

    group_spec : spectrum (str), group_min (int), _orig_in_extras (bool)
            Returns new bins and new binned counts, count errors, and effective exposures for a given spectrum and
            minimun bin gorup number.
            See LoadSpec class.

    plot : subplot_axes_grid (list of axes), rebin (int,list,dict,None), num_of_samples (int), hex_grid (bool), plot_final_result (bool)
            Used to produce a plot. No inputs are required but they offer customisation for the default plot. Creates the
            `plotting_info` attribute that contains may of the arrays (channel bins, counts, etc.) that produced the plot.

    plot_log_prob_chain :
            Produces a plot of the log-probability chain from all MCMC samples.

    run_mcmc : code (str), number_of_walkers (int), walker_spread (str), steps_per_walker (int), mp_workers (int,None), append_runs (bool), **kwargs
            Used to initiate posterior probability distribution sampling via MCMC. The kwargs are passed to the MCMC sampler,
            `backend` takes priority over `append` keyword, `pool` is overwritten if `mp_workers` is provided.

    run_nested : code (str), nlive (int), method (str), tol (float), **kwargs
            Used to initiate posterior probability distribution sampling to perform nested sampling. Can be used for model
            comparison. The kwargs are passed to the nested sampling method.

    save : filename (str)
            Used to save out session. Saves out the __dict__ of the class as well as the source code for any created models
            via `DYNAMIC_FUNCTION_SOURCE` and any user added variables via `DYNAMIC_VARS`. Use `load` method to load back in.

    Attributes
    ----------
    all_mcmc_samples : array
            Array of all MCMC used samples if run.

    all_models : dict
            Information of previous models if they exist.

    correlation_matrix : 2d square array
            Array holding correlation info between the free, fitted parameters.

    error_confidence_range : float
            Fraction of the error confidence. I.e., 0.6827 is 1-sigma or 68.27%.
            Default = 0.6827

    instruments : dict
            Spectrum identifiers as keys with the spectrum's instrument as a string for values.
            See LoadSpec class.

    intrument_loaders : dict
            Dictionary with keys of the supported instruments and values of their repsective loaders.
            See LoadSpec class.

    loaded_spec_data : dict
            All loaded spectral data. See LoadSpec class.

    loglikelihood : str
            Identifier in LogLikelihoods().log_likelihoods dict for the log-likelihood/fit
            statistic to be used. E.g., (gaussian, chi2, poisson, cash, cstat).
            Default = "cstat"

    mcmc_sampler : sampler object
            The MCMC sampler object if the MCMC has been run.

    mcmc_table : astropy table
            Table of [lower_conf_range, MAP, higher_conf_range, max_log_prob] for each
            free parameter the MCMC was run for.

    nestle : sampler object
            The nested sampling sampler object.

    nwalkers : int
            The number of walkers to set for the MCMC run. Set with `number_of_walkers`
            arg to `run_mcmc()` (must be >=2*_ndim).
            Default: 2*_ndim

    params : Parameters object
            Parameter table for all model parameters. See Parameters class.

    plotting_info : dict
            Contains most arrays used for plotting.

    rParams : Parameters object
            Parameter table for all response parameters. See Parameters class.

    sigmas : 1d array
            Array of the standard error on the free parameters.


    _colour_list : list of colours
            Colour cycle to be used when plotting submodels.
            Default = plt.rcParams['axes.prop_cycle'].by_key()['color']

    _construction_string_sunxspex : str
            String to be returned from __repr__() dunder method.

    _corresponding_submod_inputs : list of strings
            Parameter names for each sub-model.

    _covariance_matrix : 2d square array
            Array with covariance info for each free, fitted parameter.

    _discard_sample_number : Int
            The burn-in used for the MCMC samples.

    _energy_fitting_range : dict
            Defined fitting ranges for each spectrum (default {"spectrum1":[0,np.inf],...}).

    _energy_fitting_indices : list of arrays
            List of indices describing values in the count bins to be used in fitting.

    _fpl : Int
            The number of free model parameters.

    _free_model_param_bounds : list of tuples
            List of bounds for each free parameter for parameter space exploration.

    _free_model_param_names : list of strings
            List of the free model parameter names.

    _free_rparam_names : list of strings
            List of the free response parameter names.

    _latest_fit_run : str
            Stores the last method used to fill the param table, either scipy or emcee.

    _lpc : 1d array
            Orignal list of all log-probabilities.

    _max_prob : float
            Maximum probability/log-likelihood/fot statistic found during the MCMC.

    _minimize_solution : OptimizeResult
            Output from the Scipy minimise fitting funciton.

    _model : function object
            Named function of the model used to fit the data.

    _model_param_names : list
            All model parameter names in the one list.
    _ndim : int
            Number of dimensions the MCMC is sampling over.

    _orig_params : list
            List of model parameters.

    _param_groups : list of lists
            Lists of the model parameters for each spectrum.

    _pickle_reason : str
            Determines how the class is pickled, used for parallelisation.
            Default = "normal"

    _plr : bool
            Set through plot() method's `plot_final_result` entry. True will plot the final model fit
            result, False will leave it out.
            Default: True

    _rebin_setting : See LoadSpec class.

    _rebinned_edges : See LoadSpec class.

    _response_param_names : list
            All response parameter names in the one list.

    _scaled_backgrounds : dict
            Holds each spectrum's scaled background counts/s, if it has any.

    _scaled_background_rates_cut : dict
            Holds each spectrum's scaled background counts/s cut to the fitting range, if it has any.

    _scaled_background_rates_full : dict
            Holds each spectrum's full list scaled background counts/s, if it has any.

    _separate_models : list
            List of separated component models if models is given a string with >1 defined model.

    _submod_functions : list of functions
            List of component functions.

    _submod_value_inputs : lists of floats
            Lists of the sub-model value inputs.

    _other_model_inputs : dict
            Model inputs that are not model parameters, i.e., energies=None.

    _undo_rebin : See LoadSpec class.

    Examples
    --------
    # load in 2 spectra, rebin the count channels to have a minimum of 10 counts then undo that rebinning, then fit the data
    s = SunXspex(pha_file=['filename1.pha', 'filename2.pha'],
                    arf_file=['filename1.arf', 'filename2.arf'],
                    rmf_file=['filename1.rmf', 'filename2.rmf'])
    s.rebin = 10
    s.undo_rebin = 10

    # two thermal models multiplied by a constant
    s.model = "C*f_vth"
    # define a fitting range in keV
    energy_fitting_range = [2.5,8.1]

    # thermal
    s.params["T1_spectrum1"] = {"Value":3.05, "Bounds":(2.5, 6)}
    s.params["T1_spectrum2"] = s.params["T1_spectrum1"] # tie T1_spectrum2's value to T1_spectrum1's
    s.params["EM1_spectrum1"] = {"Value":1.7, "Bounds":(0.5, 3.5)}
    s.params["EM1_spectrum2"] = s.params["EM1_spectrum1"]

    # constant multiplier, vary for spectral data from filename2.pha
    s.params["C_spectrum1"] = {"Status":"freeze"}
    s.params["C_spectrum2"] = {"Bounds":(0.5, 2)}

    s_minimised_params = s.fit()
    """

    def __init__(self, *args, pha_file=None, arf_file=None, rmf_file=None, srm_file=None, srm_custom=None, custom_channel_bins=None, custom_photon_bins=None, **kwargs):
        """Construct the class and set up some defaults."""

        LoadSpec.__init__(self, *args, pha_file=pha_file, arf_file=arf_file, rmf_file=rmf_file, srm_file=srm_file, srm_custom=srm_custom, custom_channel_bins=custom_channel_bins, custom_photon_bins=custom_photon_bins, **kwargs)

        self._construction_string_sunxspex = f"SunXspex({args},pha_file={pha_file},arf_file={arf_file},rmf_file={rmf_file},srm_file={srm_file},srm_custom={srm_custom},custom_channel_bins={custom_channel_bins},custom_photon_bins={custom_photon_bins},**{kwargs})"

        self.loglikelihood = "cstat"

        # set self.error_confidence_range att and check 0<cr<=1
        self.confidence_range = 0.6827

        # define the whole energy range by default
        self.energy_fitting_range = [0, np.inf]

        # for when plotting submodels
        self._colour_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # attribute to determine how the class is pickled
        self._pickle_reason = "normal"

    @property
    def model(self):
        """ ***Property*** Allows a model to be set and a parameter table to be generated straight away.

        Sets the _model attribute.

        Returns
        -------
        The named model function used for fitting the data.
        """
        # to get get param names
        return self._model

    @model.setter
    def model(self, model_function):
        """ ***Property Setter*** Allows a model to be set and a parameter table to be generated straight away.

        Sets the _model attribute.

        Parameters
        ----------
        model_function : function object or str
                The model to be used to fit the spectral data.

        Returns
        -------
        None.

        Example
        -------
        # fit the spectral data with two thermal models and a constant
        model_2therm = lambda T1, EM1, T2, EM2, C, energies=None: C*(f_vth(T1, EM1, energies=energies) + f_vth(T2, EM2, energies=energies))

        def model_2therm(T1, EM1, T2, EM2, C, energies=None):
            return C*(f_vth(T1, EM1, energies=energies) + f_vth(T2, EM2, energies=energies))

        model_2therm = "C*(f_vth + f_vth)"

        # all above model_2therm are equivalent
        spec.model = model_2therm
        """
        if hasattr(self, '_model'):
            # if we already have a model defined then don't want to do all this again
            print("Model already assigned. If you want to change model please set property \"update_model\".")
        elif self._set_model_attr(model_function):

            # get parameter names from the function
            # need the same parameter for every spectrum
            self._orig_params = []
            self._param_groups = []
            self._model_param_names = []
            self._response_param_names = []
            self._other_model_inputs = {}
            for s in range(len(self.loaded_spec_data)):
                self._response_param_names.append("gain_slope_spectrum"+str(s+1))
                self._response_param_names.append("gain_offset_spectrum"+str(s+1))
                pg = []
                param_inputs, other_inputs = get_func_inputs(self._model)
                for p in param_inputs:
                    self._model_param_names.append(p+"_spectrum"+str(s+1))
                    pg.append(p+"_spectrum"+str(s+1))
                    if s==0:
                        self._orig_params.append(p)
                for oi in other_inputs.keys():
                    self._other_model_inputs[oi+"_spectrum"+str(s+1)] = other_inputs[oi]
                self._param_groups.append(pg)
            # now set basic param defaults
            self.params = Parameters(self._model_param_names)
            self.rParams = Parameters(self._response_param_names, rparams=True)

    def _set_model_attr(self, model_function):
        """ Sets the `_model` attribute if it can.

        True means the `_model` attribute was set and things like parameter table construction can happen.
        False means _model` attribute was not set.

        Parameters
        ----------
        model_function : function object or str
                The model to be used to fit the spectral data.

        Returns
        -------
        Bool.
        """
        if isinstance(model_function, (types.FunctionType, types.BuiltinFunctionType)):
            self._model = deconstruct_lambda(model_function)#model_function
        elif type(model_function) is str:
            self._model = self._mod_from_str(model_string=model_function, _create_separate_models_for_one=True)
        else:
            print("Model not set.")
            return False
        return True

    @property
    def update_model(self):
        """ ***Property*** Allows the existing model to be replaced and a new parameter table to be created.

        Returns
        -------
        None.
        """
        # only here to allow the setter
        pass

    @update_model.setter
    def update_model(self, new_model_function):
        """ ***Property Setter*** Allows the existing model to be replaced and a new parameter table to be created.

        Previous model, params, rParams, and mcmc_samples are save to all_models.

        Parameters
        ----------
        new_model_function : function object or str
                The model to be used to fit the spectral data. Same as input to
                model property.

        Returns
        -------
        None.
        """
        # make sure all_models is there to save previous models
        if not hasattr(self, "all_models"):
            self.all_models = {}

        # save MCMC result if it was run
        if hasattr(self, "all_mcmc_samples"):
            _mcmc_sampler = {"mcmc_sampler":self.mcmc_sampler}
            del self.mcmc_sampler
        else:
            _mcmc_sampler = {}

        # save nested sampling if it was run
        if hasattr(self, "nestle"):
            _nested_run = {"nested_run":self.nestle}
            del self.nestle
        else:
            _nested_run = {}

        self.all_models["model"+str(len(self.all_models)+1)] = {"function":self.model,"params":self.params,"rParams":self.rParams,**_mcmc_sampler,**_nested_run} # to be used for model comparison?
        del self._model
        if hasattr(self, "_latest_fit_run"):
            del self._latest_fit_run
        self.model = new_model_function

    @property
    def renew_model(self):
        """ ***Property*** Allows the model to be replaced while keeping the same parameter table.

        Can be used if the user provided a complicated model function that was not self-contained,
        hence could not be pickled and loaded back in, where this setter is used to insert the user
        model back into the class.

        The renewing model must at least have the same model parameters as the previous model.

        Returns
        -------
        None.
        """
        # only here to allow the setter
        pass

    @update_model.setter
    def renew_model(self, renewed_model_function):
        """ ***Property Setter*** Allows the model to be replaced while keeping the same parameter table.

        Can be used if the user provided a complicated model function that was not self-contained, hence
        could not be pickled and loaded back in, where this setter is used to insert the user model back
        into the class.

        The renewing model must at least have the same model parameters as the previous model.

        Parameters
        ----------
        renewed_model_function : function object or str
                The model to be used to fit the spectral data. Same as input to
                model property.

        Returns
        -------
        None.

        Example
        -------
        # fit the spectral data with one thermal model and a constant and something_complicated or a function dependent on something_complicated
        def complicated_model(T1, EM1, C, something_complicated, energies=None):
            return C*(f_vth(T1, EM1, energies=energies)) * f(something_complicated)

        # set the model
        spec.model = complicated_model

        # this may produce warnings if the user has not been able to add their own functions and variables using `add_photon_model()` and `add_var()`, respectively
        # parallelisation will not work if this is the case since variables/functions will not be getable when freshly loaded in for each thread

        # do stuff
        spec.fit() # etc.

        # save and load back in
        spec.save("./test.pickle")
        new_spec = load("./test.pickle")
        # will get the same

        new_spec.renew_model = complicated_model
        """
        # if the new function fails because of typo/bug then this method becomes unusable since _model is deleted early on but is needed at the start, just make sure its here
        self._model = self._model if hasattr(self, '_model') else None

        # save a copy of the parameter tables since the model parameters must be the same as model it's renewing
        if hasattr(self, 'params'):
            _model, _ps, _rps = deepcopy(self._model), deepcopy(self.params), deepcopy(self.rParams)
            _periph = deepcopy((self._orig_params, self._param_groups, self._model_param_names, self._response_param_names, self._other_model_inputs))

        # remove _model attr to attempt to add renewed model
        del self._model

        self.model = renewed_model_function

        # parameters that the original model (being renewed) has that the new function doesnt have
        _orig_params_mismatch = set(_ps.param_name)-set(self.params.param_name)
        # parameters that the new model (renewing) has that the original function doesnt have
        _new_params_mismatch = set(self.params.param_name)-set(_ps.param_name)
        if not (_orig_params_mismatch==set()) or not (_new_params_mismatch==set()):
            del self.params, self.rParams
            self._orig_params, self._param_groups, self._model_param_names, self._response_param_names, self._other_model_inputs = _periph
            self.params, self.rParams = Parameters(_ps.param_name), Parameters(_rps.param_name, rparams=True)
            self._model = _model
            print("Model cannot be renewed as the number of parameters are different.")
            print(f"The following parameters are missing: {_orig_params_mismatch}, and the following are new: {_new_params_mismatch}")

        self.params["Status"], self.params["Value"], self.params["Bounds"], self.params["Error"] = list(_ps.param_status), list(_ps.param_value), list(_ps.param_bounds), list(_ps.param_error)
        self.rParams["Status"], self.rParams["Value"], self.rParams["Bounds"], self.rParams["Error"] = list(_rps.param_status), list(_rps.param_value), list(_rps.param_bounds), list(_rps.param_error)

    @property
    def energy_fitting_range(self):
        """ ***Property*** Allows a fitting range to be defined.

        Returns
        -------
        Dictionary of the fitting ranges defined for each loaded spectrum.
        """
        if not hasattr(self, "_energy_fitting_range"):
            self.energy_fitting_range = [0, np.inf]
        return self._energy_fitting_range

    @energy_fitting_range.setter
    def energy_fitting_range(self, fitting_ranges):
        """ ***Property Setter*** Allows a fitting range to be defined.

        Parameters
        ----------
        fitting_ranges : dict or list
                Dict where the keys are the spectra identifiers (e.g., "spectrum1")
                and values of valid energy ranges, or a list of energy ranges to
                be applied to all spectra.

        Returns
        -------
        None.

        Example
        -------
        spec.energy_fitting_range = [2.5,8.1]
        # <equivalent> spec.energy_fitting_range = [[2.5,8.1]]

        # To fit the energy range while missing bins:
        spec.energy_fitting_range = [[2.5,4], [4.5,8.1]]
        # This only will fit the counts from 2.5--4 keV and 4.5--8.1 keV and is applied to all spectra loaded

        # To vary the fitting range per spectrum, say if we have two spectra loaded:
        spec.energy_fitting_range = {"spectrum1":[[2.5,4], [4.5,8.1]], "spectrum2":[[2.5,8.1]]}
        """
        _default_fitting_range = [0, np.inf]

        # if a dict with keys of the spectra identifiers and energy ranges
        if (type(fitting_ranges)==dict):
            default = dict(zip(self.loaded_spec_data.keys(), np.tile(_default_fitting_range, (len(self.loaded_spec_data.keys()), 1))))
            default_updated = {**default, **fitting_ranges} # incase a dict is given only updating some spectra
            self._energy_fitting_range = {k: default_updated[k] for k in list(self.loaded_spec_data.keys())}
            return

        # if not type list or array then set to default
        if (type(fitting_ranges) not in (list, np.ndarray)):
            warnings.warn(self._energy_fitting_range_instructions())
            fitting_ranges = _default_fitting_range

        # if a list is given then it is the fitting range for all spectra loaded
        if np.size(fitting_ranges)==2:
            # if, e.g., [2,3] or [[2,3]] then fitting range is 2--3 keV for all spectra
            frs = np.tile(fitting_ranges, (len(self.loaded_spec_data.keys()), 1))
        elif len(np.shape(fitting_ranges))==2:
            # if, e.g., [[2,3], [4,8]] then fitting range is 2--3 and 4--8 keV for all spectra
            frs = [fitting_ranges]*len(self.loaded_spec_data.keys())
        else:
            # if (somehow) none of the above then default it
            frs = np.tile(_default_fitting_range, (len(self.loaded_spec_data.keys()), 1))
            warnings.warn(self._energy_fitting_range_instructions())


        self._energy_fitting_range = dict(zip(self.loaded_spec_data.keys(), frs))

    def _energy_fitting_range_instructions(self):
        """ Function to store string needed for multiple points in the energy_fitting_range setter.

        Returns
        -------
        String.
        """
        return "\nNeed one fitting_energy_range entry (e.g., [2,6.5]) for a fitting range for each loaded spectrum. Back to default [0,inf].\nExamples: energy_fitting_range=[1,2] or energy_fitting_range=[[1,2], [5,6]] for fitting 1--2 keV or 1--2 and 5--6 keV, respectively, for all loaded spectra.\nenergy_fitting_range={\"Spectrum1\":[1,2], \"Spectrum2\":[[1,2], [5,6]]} for fitting spectrum1 over 1--2 keV and spectrum2 over 1--2 and 5--6 keV, respectively."


    @property
    def confidence_range(self):
        """ ***Property*** Allows the confidence region for mcmc errors and plots to be set.

        Checks 0<confidence region<=1.

        Returns
        -------
        Float.
        """
        if not hasattr(self, "error_confidence_range"):
            self.error_confidence_range = 0.6827
        return self.error_confidence_range

    @confidence_range.setter
    def confidence_range(self, conf_range):
        """ ***Property Setter*** Allows the confidence region for mcmc errors and plots to be set.

        Checks 0<confidence region<=1.

        Parameters
        ----------
        conf_range : 0<float<=1
                Fractional confidence range used for MCMC stuff.

        Returns
        -------
        None.
        """
        if 0<conf_range<=1:
            self.error_confidence_range = conf_range
            if hasattr(self, "all_mcmc_samples") and hasattr(self, "_latest_fit_run") and (self._latest_fit_run=="mcmc"):
                self._update_tables_mcmc(orig_free_param_len=self._fpl)
        else:
            warnings.warn("Need 0<confidence_range<=1. Setting back to default: 0.6827")
            self.error_confidence_range = 0.6827

    def _component_mods_from_str(self, model_string, _create_separate_models_for_one=False):
        """ Deconstructs a given model string into its component models.

        Returns a list of lists with each model isolated and its counter.

        E.g., C*(f_vth+f_vth) would return [["C*(f_vth+0)", 1], ["C*(0+f_vth)", 2]].

        Parameters
        ----------
        model_string : str
                String of the model.

        _create_separate_models_for_one : bool
                If True then even if only one sub-model in total model it will be "separated"
                out anyway and the `_separate_models` attribute will be created. If False then
                the `_separate_models` attribute will only be created if there are multiple
                sub-models in the model.
                Default: False

        Returns
        -------
        List.
        """

        model_names = list(defined_photon_models.keys())
        mods_removed = model_string.split(model_names[0]) # split the first one to start
        inds = np.array(list(map(len, model_string.split(model_names[0])))) # get the indices of the gaps in the string (now broken down into a list)
        ind_and_mod = [(i,model_names[0]) for i in np.cumsum(inds+len(model_names[0])*np.arange(len(inds)))[:-1]] # starting index(/indices) of the first model in defined_photon_models in your custom string
        for mn in model_names[1:]:
            mods_removed = sum([s.split(mn) for s in mods_removed],[])# each gap is where one of the defined models should be
            inds = np.array(list(map(len, model_string.split(mn)))) # get the indices of the gaps in the string (now brocken down into a list)
            ind_and_mod += [(i,mn) for i in np.cumsum(inds+len(mn)*np.arange(len(inds)))[:-1]]
        model_order_in_model_string = [m[1] for m in sorted(ind_and_mod)] # use indices of where the models are in the original string to order the component models correctly

        # only create the separate model strings if we have (multiple sub-models) or (only have one sub-model but if from the original user input string and not a previously determined sub-model string)
        if len(model_order_in_model_string)>1 or (_create_separate_models_for_one and len(model_order_in_model_string)==1):
            diag = np.diag(model_order_in_model_string) # now each row is the replacement for mods_removed, meaning we can plot each model spearately
            diag[diag==""] = "0"

            put_mods_back = [None]*(len(model_order_in_model_string)+len(mods_removed))
            _isolated_model_strings = []
            __mod_counter = []
            for iso_mods, _model in zip(diag, model_order_in_model_string):
                put_mods_back[::2] = mods_removed
                put_mods_back[1::2] = iso_mods
                _mod_counter = str(__mod_counter.count(_model)+1)
                __mod_counter.append(_model)
                _isolated_model_strings.append(["".join(put_mods_back), _mod_counter])
                # model shorthand string for _mod_from_str and the number for the input parameters. E.g., 1st f_vth->f_vth(T1,EM1, ...), 2nd f_vth->f_vth(T2,EM2, ...)
            self._separate_models = _isolated_model_strings


    def _mod_from_str(self, model_string, custom_param_number=None, _create_separate_models_for_one=False):
        """ Construct a named function object from a given string.

        Function name is made up of the string and model inputs.

        Parameters
        ----------
        model_string : str
                String of the model, e.g., "C*(f_vth+f_vth)".

        custom_param_number : None or int
                When building up parameters each duplicate of the same
                model increments it's number by 1 (default). A custom
                number can be provide for all parameter for the string
                (helps when plotting component models).
                Default : None

        _create_separate_models_for_one : bool
                If True then even if only one sub-model in total model it will be "separated"
                out anyway and the `_separate_models` attribute will be created. If False then
                the `_separate_models` attribute will only be created if there are multiple
                sub-models in the model.
                Default: False

        Returns
        -------
        Function.
        """
        # take in a mathematical expression with functions and parameters and constants and return the named function of the expression
        if check_allowed_names(model_string=model_string):
            _mod = copy(model_string)
            _params = []
            self._component_mods_from_str(model_string=model_string, _create_separate_models_for_one=_create_separate_models_for_one) # try to break down into separate models and assign them to self._separate_models, need >1 sub-model
            for mn, mp in defined_photon_models.items():
                # how many of this model are in the string
                number_of_this_model = _mod.count(mn)
                # number the parameter accordingly
                mp_numbered = [[mod_par+str(i) for mod_par in mp] for i in range(1,number_of_this_model+1)] if type(custom_param_number)==type(None) else [[mod_par+str(custom_param_number) for mod_par in mp] for _ in range(1,number_of_this_model+1)]
                mods_removed = _mod.split(mn)
                _mods_with_params = []
                for numbered_params in mp_numbered:
                    # add in inputs to the function constructor string for each of this model
                    _params += numbered_params
                    # _mods_with_params.append(mn+"(energies,"+",".join(numbered_params)+")") # no need to have f(e,...) AND f(...,e=None), change all to latter
                    _mods_with_params.append(mn+"("+",".join(numbered_params)+", energies=energies)")
                put_mods_back = [None]*(len(mods_removed)+len(_mods_with_params))
                put_mods_back[::2] = mods_removed
                put_mods_back[1::2] = _mods_with_params
                _mod = "".join(put_mods_back) # replace the model string with with latest model component added
            # build the function string
            _params += get_nonsubmodel_params(model_string=model_string, _defined_photon_models=defined_photon_models)
            fun_name = re.sub(r'[^a-zA-Z0-9]+', '_', model_string).lstrip('0123456789')+"".join(_params) # replace non-word/non-numbers from string with "_", remove any starting numbers, join params to the end too->should be a legit and unique enough function name
            def_line = "def "+fun_name+"("+",".join(_params)+", energies=None):\n"
            return_line = "    return "+_mod+"\n"
            return function_creator(function_name=fun_name, function_text="".join([def_line, return_line]))
        else:
            print("The above are not valid identifiers (or are keywords) in Python. Please change this in your model string.")

    def _free_and_other(self):
        """ Find all inputs to the model being used.

        Return the starting values of the free parameters and the tied/frozen parameters.
        Also list the parameter names such that the free parameter names are all first.

        Returns
        -------
        The starting values for the free parameters (free_params_list), the values
        for the tied or frozen parameters (tied_or_frozen_params_list), other inputs
        to the model being used in a dictionary (other_inputs, just `energies` at the
        moment), a list of the new parameter order with the free ones first followed by
        the tied or frozen ones (param_name_list_order), a list of the free parameter
        bounds (free_bounds).
        """
        # to sort the free params to be first in your actual model
        # find the free param names and values and bounds, find the tied or frozen params
        free_params = {}
        free_bounds = []
        tied_or_frozen_params = {}
        for key in self.params.param_name:
            if self.params["Status", key].startswith("free"):
                free_params[key] = self.params["Value", key]
                free_bounds.append(self.params["Bounds", key])
            elif self.params["Status", key].startswith("frozen") or self.params["Status", key].startswith("tie"):
                tied_or_frozen_params[key] = self.params["Value", key]

        # does the function have any other inputs, might be needed for future
        other_inputs = {}
        for key in self._other_model_inputs.keys():
            other_inputs[key] = None

        free_params_list = list(free_params.values())
        tied_or_frozen_params_list = list(tied_or_frozen_params.values())
        param_name_list_order = list(free_params.keys())+list(tied_or_frozen_params.keys())

        return free_params_list, tied_or_frozen_params_list, other_inputs, param_name_list_order, free_bounds

    def _tie_params(self, param_name_dict_order):
        """ Change tied parameter values to the ones they are tied to.

        Before the _pseudo_model passes the reordered parameter info to the user model,
        change the tied parameter values to the values of the parameter they are tied to.

        E.g.,
        If self.params["Status", "param1"] = "tie_param2"
        Then self.params["Value", "param1"] = self.params["Value", "param2"]

        Parameters
        ----------
        param_name_dict_order : dict
                Dictionary with keys of the parameter names and values of the parameter values.

        Returns
        -------
        The new `free_tied_or_frozen_params_list` where the parameters that are tied have
        their value changed to the value of the parameter they are tied to.
        """

        for key in self.params.param_name:
            _status = self.params["Status", key]
            if _status.startswith("tie"):
                # tie one parameter to another. E.g., self.model_param_names["T1"]="tie_T2".
                # This would tie T1's value to T2's.
                try:
                    param_name_dict_order[key] = param_name_dict_order[_status[4:]]
                except ValueError:
                    warnings.warn(f"Either the {key} or {_status[4:]} parameter has not been passed to the _pseudo_model. Value for parameter {key} will act like it is frozen.")

        # now check the response parameters if we have any but same process
        for key in self.rParams.param_name:
            _rstatus = self.rParams["Status", key]
            if _rstatus.startswith("tie"):
                # tie one parameter to another. E.g., self.model_param_names["T1"]="tieT2".
                # This would tie T1's value to T2's.
                # if an rparam is tied to a frozen rparam then this is handled in sort_fixed_gain_method()
                try:
                    param_name_dict_order = self._update_rtable_or_kwargs(key, param_name_dict_order, _rstatus)
                except ValueError:
                    warnings.warn(f"Either the {key} or {_rstatus[4:]} response parameter has not been passed to the _pseudo_model. Value for parameter {key} will act like it is frozen.")

        return param_name_dict_order

    def _update_rtable_or_kwargs(self, key, param_name_dict_order, _rstatus):
        """ Either update the parameter list or the response parameter table.

        If a response parameter is tied to a frozen rparam then might as well just update the table and pull
        the values from there later. Otherwise, add the rparam to the input parameter list.

        Parameters
        ----------
        key : str
                The response parameter under question.

        param_name_dict_order : dict
                Dictionary with keys of the parameter names and values of the parameter values.

        _rstatus : str
                Status of the response parameter under question. If it starts with `tie` then can find the
                response parameter it is tied to.

        Returns
        -------
        The input parameter list either as it was (rParam tied to frozen rParam) or updated parameter list
        with the tied to rParam included.
        """
        if self.rParams[_rstatus[4:], "Status"]=="frozen":
            self.rParams[key, "Value"] = self.rParams[_rstatus[4:], "Value"]
        else:
            param_name_dict_order[key] = param_name_dict_order[_rstatus[4:]]
        return param_name_dict_order

    def _gain_energies(self, energies, array, gain_slope, gain_offset):
        """ Allow a gain shift to be applied.

        The output model spectra are redefined as they get new energies ($E_{new}$) the counts a
        re said to correspond to. We have

        .. math::
         E_{new} = E_{orig}/gain_slope - gain_offset ,

        where $E_{orig}$ are the orignal energies for the counts, and gain_slope/gain_offset
        are the gradient and offset for the conversion, respectively. This is the same
        representation as in the XSPEC fitting software [1].

        New count values are returned interpolated back onto the original energies.

        An equivalent method is to use
        .. math::
         E_{new} = (E_{orig} + gain_offset) \times gain_slope ,

        and interpolate the counts from these new energies to the orignal ones.

        [1] https://heasarc.gsfc.nasa.gov/xanadu/xspec/xspec11/manual/node34.html

        Parameters
        ----------
        energies : list or array
                Original energies corresponding to the array of count values (`array`).

        array : list or array
                Count data corresponding to the energies given.

        gain_slope, gain_offset : float or int
                The gradient and offset, respectively, of the linear transformation
                applied to the energies.

        Returns
        -------
        Array of the new count values to be used with the original energies given
        (not with the new energy values they were interpolated to).
        """
        # find the new energy values
        new_energies = np.array(energies)/gain_slope - gain_offset
        # interpolate the counts to them for them to be used with the original energies
        return interp1d(energies, array, bounds_error=False, fill_value="extrapolate")(new_energies)


    def _match_kwargs2orig_params(self, original_parameters, expected_kwargs, given_kwargs):
        """ Returns the coorect order of inputs from being arrange with the free ones first.

        Takes the original parameters for the model, the corresponding spectrum specific parameters,
        and all inputs to _counts_model and returns a list of the inputs to the user model in the
        correct order.

        E.g.,
        Takes model ["p1","p2"] from model(p1,p2), corresponding param table entries for each
        spectrum like ["p1_spectrum1","p2_spectrum1"], all model info for the count model creation
        and then returns [p1_spectrum1's value, p2_spectrum1's value] to be given to the model again
        in _counts_model like model(p1_spectrum1's value, p2_spectrum1's value).

        Parameters
        ----------
        original_parameters : list of str
                List of the original parameters without any spectra identifier. E.g., ["p1","p2"] not
                ["p1_spectrum1","p2_spectrum1","p1_spectrum2","p2_spectrum2"]

        expected_kwargs : None or int
                The expected parameter arguments for the model being used from each spectrum.
                E.g., like above, need ["p1_spectrum1","p2_spectrum1"] to match model's ["p1","p2"]

        given_kwargs : dict
                All inputs needed for the model to be calculated and converted to a counts model.

        Returns
        -------
        List of floats ordered to go into the user defined model.
        """

        # check all parameters are given
        ordered_kwarg_values = [0]*len(original_parameters)
        for ek in expected_kwargs:
            p_name = ek.split("_spectrum")[0] # find the model param this model-spectrum param corresponds to
            p_index = original_parameters.index(p_name) # find the position it is to be given to the user model
            ordered_kwarg_values[p_index] = given_kwargs[ek] # put the value of model-spectrum param in the position of model param

        return ordered_kwarg_values

    def _counts_model(self, **kwargs):
        """ Calculates the count rate models from user's photon flux model for fitting.

        Takes the user model, and all other relavent inputs, to calculate the user's photon spectrum
        model (_model) into a count rate spectrum (cts s^-1).

        Note: The energy binning (keV^-1) is removed when the photon spectrum is calculated. This makes
        log-likelihood and count space rebinning easier.

        Parameters
        ----------
        **kwargs used:
                all spectral model parameters

        **kwargs used & named:
                photon_channels
                photon_channel_widths
                total_response

        **kwargs needed for gain (optional):
                gain_slope_spectrumN
                gain_offset_spectrumN
                count_channel_mids

        Returns
        -------
        List of count rate models (counts s^-1) for all loaded spectra.
        """
        # return counts s^-1 (keV^-1 has been multiplied out) photon_channel_widths=ph_e_binning

        cts_models = []
        # loop through the parameter groups (params for spectrum1, then for spectrum2, etc)
        for s, pgs in enumerate(self._param_groups):
            # take the spectrum parameters (e.g., [p2_spectrum1,p1_spectrum1]) and order to be the same as the model parameters (e.g., [p1,p2])
            ordered_kwarg_values = self._match_kwargs2orig_params(original_parameters=self._orig_params,
                                                                 expected_kwargs=pgs,
                                                                 given_kwargs=kwargs)

            # assign the spectrum parameter values to their model param counterpart
            sep_params = dict(zip(self._orig_params, ordered_kwarg_values))

            # calculate the [photon s^-1 cm^-2]
            m = self._model(**sep_params, energies=kwargs["photon_channels"][s]) * kwargs["photon_channel_widths"][s]# np.diff(kwargs["photon_channels"][s]).flatten() # remove energy bin dependence

            # fold the photon model through the SRM to create the count rate model, [photon s^-1 cm^-2] * [count photon^-1 cm^2] = [count s^-1]
            cts_model = make_model(energies=kwargs["photon_channels"][s],
                                   photon_model=m,
                                   parameters=None,
                                   srm=kwargs["total_responses"][s])

            if "scaled_background_spectrum"+str(s+1) in self._scaled_backgrounds:
                cts_model += self._scaled_backgrounds["scaled_background_spectrum"+str(s+1)]

            # apply a response gain correction if need be
            if ("gain_slope_spectrum"+str(s+1) in kwargs) or ("gain_offset_spectrum"+str(s+1) in kwargs):
                cts_model = self._gain_energies(energies=kwargs["count_channel_mids"][s],
                                                array=cts_model,
                                                gain_slope=kwargs["gain_slope_spectrum"+str(s+1)],
                                                gain_offset=kwargs["gain_offset_spectrum"+str(s+1)])

            # if the model returns nans then it's a rubbish fit so change to zeros, just to be sure
            cts_model = cts_model if len(LL_CLASS.remove_non_numbers(cts_model[cts_model!=0]))!=0 else np.zeros(cts_model.shape)

            cts_models.append(cts_model[None,:]) # need [None, :] these lines get rid of a dimension meaning later concatenation fails

        return cts_models

    def _pseudo_model(self, free_params_list, tied_or_frozen_params_list, param_name_list_order, **other_inputs):
        """ Bridging method between the input args (free,other) and different ordered args for the model calculation.

        Takes model parameters with the free ones in one list and tied/frozen in another with the names of all in order.
        All parameter values are match with their names and tied parameter values are changed to be the same as the
        parameter they are tied to before passing to _counts_model.

        This is here because minimisers and samplers mainly just want a list or array to vary so need to take in a
        list/array (free_params_list) of the parameters to vary then construct order for the user model later. Also
        need to change tied parameter values to the ones they are tied to.

        Parameters
        ----------
        free_params_list : list of floats
                Values for the free parameters.

        tied_or_frozen_params_list : list of floats
                Values for the tied or frozen parameters (the tied parameter values
                get changed by _tie_params).

        param_name_list_order : list of str
                All parameter names in order of [*free_params_list, *tied_or_frozen_params_list]

        **other_inputs:
                photon_channels
                photon_channel_widths
                total_response
                count_channel_mids

        Returns
        -------
        Output from _counts_model.
        """
        # match all parameters up with their names
        dictionary = dict(zip(param_name_list_order, list(free_params_list)+list(tied_or_frozen_params_list)))

        # change the tied parameter's value to the value of the param it is tied to
        dictionary = self._tie_params(dictionary)

        return self._counts_model(**dictionary, **other_inputs)


    @property
    def show_params(self):
        """ ***Property*** Returns an astropy table of the model parameter info.

        Returns
        -------
        Astropy table.
        """
        t = self.params.to_astropy
        t.add_row(["Fit Stat.", self.loglikelihood+" ln(L)", self._get_max_fit_stat(), (None, None), (0.0, 0.0)]) # add fit stat info
        t = Table(t, masked=True, copy=False) # allow values to be masked
        dont_masked_params = [False]*len(self.params.param_name)
        t["Error"].mask = [*dont_masked_params, True] # mask these columns for the stat value
        t["Bounds"].mask = [*dont_masked_params, True]
        t["Value"].format, t["Error"].format = "%10.2e", "(%10.2e, %10.2e)"
        return t

    @property
    def show_rParams(self):
        """ ***Property*** Returns an astropy table of the response parameter info.

        Returns
        -------
        Astropy table.
        """
        t = self.rParams.to_astropy
        t["Value"].format, t["Error"].format = "%10.3f", "(%10.2f, %10.2f)"
        return t

    def _fit_range(self, channel, fitting_range):
        """ Get channel bin indices for fitting range.

        Takes the energy channels and a fitting range and returns the indices of
        channels and counts within that range. Works with bin mid-points and is
        non-inclusive. I.e., fitting_range_lower<channels<fitting_range_higher.

        Parameters
        ----------
        channel : list or array
                List or arrays of the energy channels for all loaded spectra.

        fitting_range : dict
                A dictionary of the fitting ranges for each spectrum, see
                energy_fitting_range and _energy_fitting_range_instructions
                methods for more detail.

        Returns
        -------
        Array or lists of indices of the channels/counts in the given
        fitting range.
        """
        useful_inds = []
        # if multiple spectra are loaded then have multiple channel lists
        if (len(np.shape(channel))==2) or (type(channel)==list):
            for c, f in zip(channel, fitting_range.values()):
                c = np.array(c)
                in_range = np.where((c>np.squeeze(f)[0])&(c<np.squeeze(f)[1])) if np.size(f)==2 else [(c>f_range[0])&(c<f_range[1]) for f_range in f]
                allowed = in_range if np.size(f)==2 else np.where([any(b) for b in zip(*in_range)])
                useful_inds.append(allowed)#np.where((c>f[0])&(c<f[1])))
            return useful_inds
        else:
            # if just one channel list is given
            channel = np.array(channel)
            in_range = np.where((channel>np.squeeze(fitting_range)[0])&(channel<np.squeeze(fitting_range)[1])) if np.size(fitting_range)==2 else [(channel>f_range[0])&(channel<f_range[1]) for f_range in fitting_range]
            allowed = in_range if np.size(fitting_range)==2 else np.where([any(b) for b in zip(*in_range)])
            return allowed

    def _count_rate2count(self, counts_model, livetime):
        """ Convert a count rate [counts s^-1] to just counts.

        Parameters
        ----------
        counts_model : list or array
                Count rate models [counts s^-1].

        fitting_range : dict
                The effective exposure of the observation.

        Returns
        -------
        Array or lists of count spectra.
        """
        return (counts_model * livetime).astype(int)

    def _choose_loglikelihood(self):
        """ Access the log_likelihoods attribute.

        This is located in the likelihoods module and return the log_likelihoods
        associtated with the loglikelihood attribute of this class.

        Parameters
        ----------

        Returns
        -------
        Function.
        """
        return LL_CLASS.log_likelihoods[self.loglikelihood.lower()]

    def _minus_2lnL(self):
        """ Return -2*log_likelihood for a minimiser.

        Returns
        -------
        Lambda function.
        """
        return lambda *args: -2*self._choose_loglikelihood()(*args)

    def _update_tied(self, table):
        """ Updates the tied parameter values in the given parameter table.

        Parameters
        ----------
        table : parameter_handler.Parameters
                The parameter table to update.

        Returns
        -------
        None.
        """
        # only let "free" params vary
        for key in table.param_name:

            if table["Status", key].startswith("tie"):
                # tie one parameter to another. E.g., self.model_param_names["T1"]="tie_T2".
                # This would tie T1's value to T2's.
                param_tied_2 = table["Status", key][4:]
                table["Value", key] = table["Value", param_tied_2]

    def _update_free(self, table, updated_free, errors):
        """ Updates the free parameter values in the given parameter table to the values found by the minimiser.

        Parameters
        ----------
        table : parameter_handler.Parameters
                The parameter table to update.

        updated_free : 1d array
                Array of the minimiser values found for the free parameters
                in order (top to bottom) of the parameter table entries.

        errors : 1d array
                Array of errors calculated from the Hessian that correspond
                to the updated_free array (see _calc_minimize_error method).

        Returns
        -------
        None.
        """
        # only update the free params that were varied
        c = 0
        for key in table.param_name:

            if table["Status", key].startswith("free"):
                # just for completeness
                table["Value", key] = updated_free[c]
                table["Error", key] = (errors[c],  errors[c])
                c += 1

    def _fit_stat(self,
                  free_params_list,
                  photon_channels,
                  count_channel_mids,
                  srm,
                  livetime,
                  ph_e_binning,
                  observed_counts,
                  observed_count_errors,
                  tied_or_frozen_params_list,
                  param_name_list_order,
                  maximize_or_minimize,
                  **kwargs):
        """ Calculate the fit statistic from given parameter values.

        Calculates the model count spectrum and calculates the fit statistic
        in relation to the data. The ln(L) (or -2ln(L)) is added for all spectra
        that are loaded, fitting all spectra simultaneously, and returned.

        Parameters
        ----------
        free_params_list : 1d array or list of 1d arrays
                The values for all free parameters.

        photon_channels : 2d array or list of 2d arrays
                The photon energy bins for each spectrum (2 entries per bin).

        count_channel_mids : 1d array or list of 1d arrays
                The mid-points of the count energy channels.

        srm : 2d array or list of 2d arrays
                The spectral response matrices for all loaded spectra.

        livetime : list of floats
                List of spectra effective exposures
                (s.t., count s^-1 * livetime = counts).

        ph_e_binning : 1d array or list of 1d arrays
                The photon energy binning widths for all loaded spectra.

        observed_counts : 1d array or list of 1d arrays
                The observed counts for all loaded spectra.

        observed_count_errors : 1d array or list of 1d arrays
                Errors for the observed counts for all loaded spectra.

        tied_or_frozen_params_list :
                Values for the parameters that are needed for the fitting but
                are tied or frozen.

        param_name_list_order : list of strings
                List of all parameter names to match the order of
                [*free_params_list,*tied_or_frozen_params_list].

        maximize_or_minimize : str
                Determines whether to calculate the log-likelihood for
                the mcmc ("maximize") or -2*log-likelihood for a
                minimiser ("minimize").

        **kwargs :
                Passed to _pseudo_model method.

        Returns
        -------
        Float, the combined ln(L) or -2ln(L) for all spectra loaded and model.
        """

        # make sure only the free parameters are getting varied so put them first
        mu = self._pseudo_model(free_params_list,
                                tied_or_frozen_params_list,
                                param_name_list_order,
                                photon_channels=photon_channels,
                                photon_channel_widths=ph_e_binning,
                                count_channel_mids=count_channel_mids,
                                total_responses=srm,
                                **kwargs)

        ll = 0
        for m, o, l, err in zip(mu, observed_counts, livetime, observed_count_errors):

            # calculate the count rate model for each spectrum
            model_cts = self._count_rate2count(m, l)

            # either calculate ln(L) or -2ln(L)
            if maximize_or_minimize == "maximize":
                ll += self._choose_loglikelihood()(model_cts, o, err)
            elif maximize_or_minimize == "minimize":
                ll += self._minus_2lnL()(model_cts, o, err)

        return ll

    def _cut_srm(self, srms, spectrum=None):
        """ Select the columns in the SRM (count space) that are appropriate for the defined fitting range.

        Parameters
        ----------
        srms : list of 2d arrays
                A list SRM arrays.

        spectrum : int or None
                The spectrum number ID corresponding to the SRM given.
                E.g., "spectrum1" would mean spectrum=1. If set to None
                then assume given SRMs for all loaded spectra.
                Default: None

        Returns
        -------
        List of 2d arrays.
        """
        # clip the counts bins in the srm (keeping all photon bins) to cut down on matrix multiplication
        # don't need all the count bins from the model for fitting
        if spectrum is None:
            for c, srm in enumerate(srms):
                srms[c] = srm[:, self._energy_fitting_indices[c][0]]
        else:
            srms = srms[:,self._energy_fitting_indices[int(spectrum)-1][0]]
        return srms

    def _cut_counts(self, counts, spectrum=None):
        """ Select the entries in counts information that correspond to the defined fitting range.

        Parameters
        ----------
        counts : list of 1d arrays
                A list counts data. This could be a list of channel-mids,
                counts, or bin widths.

        spectrum : int or None
                The spectrum number ID corresponding to the array given.
                E.g., "spectrum1" would mean spectrum=1. If set to None
                then assume given arrays for all loaded spectra.
                Default: None

        Returns
        -------
        List of 1d arrays.
        """
        # clip the counts bins in the count spectrum
        if spectrum is None:
            for c, count in enumerate(counts):
                counts[c] = count[self._energy_fitting_indices[c]]
        else:
            counts = np.array(counts)[self._energy_fitting_indices[int(spectrum)-1]]
        return counts

    def _loadSpec4fit(self):
        """ Loads all photon and count bin information.

        Includes SRMs, effective exposure, energy binning, data and data errors needed
        for fitting.

        Returns
        -------
        Photon channel bins (hoton_channel_bins), photon channel mid-points
        (photon_channel_mids), count channel mid-points (count_channel_mids),
        spectral response matrices (srm), effective exposures (livetime),
        count channel bin widths (e_binning), photon channel bin widths
        (ph_e_binning), observed count data (observed_counts), observed
        count data errors (observed_count_errors).
        """

        photon_channel_bins, photon_channel_mids, count_channel_mids, srm, livetime, e_binning, ph_e_binning, observed_counts, observed_count_errors = [], [], [], [], [], [], [], [], []
        for k in self.loaded_spec_data:
            photon_channel_bins.append(self.loaded_spec_data[k]['photon_channel_bins'])
            photon_channel_mids.append(self.loaded_spec_data[k]['photon_channel_mids'])
            count_channel_mids.append(self.loaded_spec_data[k]['count_channel_mids'])
            srm.append(self.loaded_spec_data[k]['srm'])
            e_binning.append(self.loaded_spec_data[k]['count_channel_binning'])
            ph_e_binning.append(self.loaded_spec_data[k]['photon_channel_binning'])
            observed_counts.append(self.loaded_spec_data[k]['counts'])
            observed_count_errors.append(self.loaded_spec_data[k]['count_error'])
            livetime.append(self.loaded_spec_data[k]['effective_exposure'])

        return photon_channel_bins, photon_channel_mids, count_channel_mids, srm, livetime, e_binning, ph_e_binning, observed_counts, observed_count_errors

    def _tied2frozen(self, spectrum_num):
        """ Checks if any gain parameters are tied to another frozen rparameter.

        Returns True is both slope and offset are tied to frozen parameters.

        Parameters
        ----------
        spectrum_num : int
                Spectrum ID number as an integer.

        Returns
        -------
        Bool.
        """
        # if either the slope OR the offset is tied to another response param that is frozen then this flags them to be
        #  checked if the frozen rparam has default 1 and 0 values
        tied2frozen = []
        for g in ["slope", "offset"]:
            gain_rparam = "gain_"+g+"_spectrum"+str(spectrum_num)
            if self.rParams["Status", gain_rparam].startswith("tie"):
                gain_tied2 = self.rParams["Status", gain_rparam][4:]
                # if these are tied to frozen parameters then change them to the right value
                if self.rParams["Status", gain_tied2]=="frozen":
                    tied2frozen.append(True)
                    self.rParams["Value", gain_rparam] = self.rParams["Value", gain_tied2]
                else:
                    tied2frozen.append(False)
        tied2frozen = False if len(tied2frozen)==0 else all(tied2frozen)
        return tied2frozen

    def _gain_froz_or_tied2froz_notdefault(self, update_fixed_params, s):
        """ Updates `update_fixed_params` for any gain params that are fixed and not default.

        Parameters
        ----------
        update_fixed_params : dict
                Dictionary of fixed parameter names and their values.

        s : int
                Spectrum ID index as an integer.

        Returns
        -------
        Updated fixed parameter dictionary.
        """
        # defaults should be offset=0 and slope=1
        if (self.rParams["Value", "gain_slope_spectrum"+str(s+1)]!=1) or (self.rParams["Value", "gain_offset_spectrum"+str(s+1)]!=0):
                update_fixed_params.update(**{"gain_slope_spectrum"+str(s+1):self.rParams["Value", "gain_slope_spectrum"+str(s+1)],
                                                "gain_offset_spectrum"+str(s+1):self.rParams["Value", "gain_offset_spectrum"+str(s+1)]})
        return update_fixed_params

    def _gain_free_or_tie(self, update_fixed_params, update_free_params, update_free_bounds, s):
        """ Updates `update_fixed_params`, `update_free_params`, and `update_free_bounds`.

        Any free parameters are added to `update_free_params`, and `update_free_bounds` else they are
        added to `update_fixed_params`.

        Parameters
        ----------
        update_fixed_params, update_free_params, update_free_bounds : dict, dict, dict
                Dictionary of fixed, free parameter and free bounds names and their values, respectively.

        s : int
                Spectrum ID index as an integer.

        Returns
        -------
        Updated fixed, free parameter dictionaries and free bounds list.
        """
        for g in ["slope", "offset"]:
            gain_rparam = "gain_"+g+"_spectrum"+str(s+1)
            # if free then update free params and bounds
            if (self.rParams["Status", gain_rparam]=="free"):
                update_free_params.update(**{gain_rparam:self.rParams["Value", gain_rparam]})
                update_free_bounds.append(self.rParams["Bounds", gain_rparam])
            else:
                # elif update the fixed param list (this is needed if, e.g., slope is free but offset is fixed since both are needed to gain shift)
                update_fixed_params.update(**{gain_rparam:self.rParams["Value", gain_rparam]})
        return update_fixed_params, update_free_params, update_free_bounds

    def _sort_gain(self):
        """ Inspect whether gain parameters need to be added/not added to fitting process.

        Assesses whether the response parameters for each spectrum (slope and offset)
        should be passed to the model fitting process. If slope=1 and offset=0 then nothing
        is passed; however, if one of them is different, allowed to vary, or tied to a
        response parameter that is different or allowed to vary then then both the slope and
        the offset need to be passed to the model calculation process.

        Returns
        -------
        Lists of the gain parameters if any need to be passed to the model calculation to
        update the total fixed, free, and bounds lists.
        """
        update_fixed_params = {}
        update_free_params = {}
        update_free_bounds = []
        # loop through both spectra to check the response parameters
        for s in range(len(self.loaded_spec_data)):
            # check if both gain rparams for a spec are tied but tied to frozen rparams
            tied2frozen = self._tied2frozen(spectrum_num=int(s+1))

            ## Now sort the r params into th efixed and free param lists
            # if an rparam is frozen, or both are tied to a frozen parameter, that is not the default then add them to the lists to send to the model calculation functions
            if ((self.rParams["Status", "gain_slope_spectrum"+str(s+1)]=="frozen") and (self.rParams["Status", "gain_offset_spectrum"+str(s+1)]=="frozen")) or tied2frozen:
                # don't waste time if they are the default values that don't change anything
                update_fixed_params = self._gain_froz_or_tied2froz_notdefault(update_fixed_params, s)

            elif (self.rParams["Status", "gain_slope_spectrum"+str(s+1)]=="free") or (self.rParams["Status", "gain_slope_spectrum"+str(s+1)].startswith("tie")) or (self.rParams["Status", "gain_offset_spectrum"+str(s+1)]=="free") or (self.rParams["Status", "gain_offset_spectrum"+str(s+1)].startswith("tie")):
                # elif either (or both) rparam is free or tied to a free rpraram then need to pass both to model calculation functions
                update_fixed_params, update_free_params, update_free_bounds = self._gain_free_or_tie(update_fixed_params, update_free_params, update_free_bounds, s)

        return update_fixed_params, update_free_params, update_free_bounds

    def _include_background(self, _for_plotting=False):
        """ Inspect whether a background is in the extras entry to be added to the model.

        Make the `_scaled_background_rates_cut` and `_scaled_background_rates_full` attributes which
        is a dictionary of all instrument backgrounds  cut to the fit range size (for fitting) and
        all of them (for plotting) respectively

        Parameters
        ----------
        _for_plotting : bool
                If plotting then we don't need to cut the values like when fitting.
                Default: False

        Returns
        -------
        None.
        """
        self._scaled_background_rates_cut, self._scaled_background_rates_full = {}, {}
        # loop through both spectra to check the response parameters
        for s in range(len(self.loaded_spec_data)):
            # do not want to include bg spectrum if the data is structured to be event-background
            if ("background_rate" in self.loaded_spec_data["spectrum"+str(s+1)]["extras"]) and (not self.loaded_spec_data["spectrum"+str(s+1)]["extras"]["counts=data-bg"]):
                # turn the background rate (cts/keV/s) into just cts/s scaled to the event time
                bg_cts = self.loaded_spec_data["spectrum"+str(s+1)]["extras"]["background_rate"]*self.loaded_spec_data["spectrum"+str(s+1)]["count_channel_binning"]
                self._scaled_background_rates_cut["scaled_background_spectrum"+str(s+1)] = self._cut_counts(bg_cts, spectrum=s+1) if not _for_plotting else bg_cts
                self._scaled_background_rates_full["scaled_background_spectrum"+str(s+1)] = bg_cts

    def _fit_stat_minimize(self, *args, **kwargs):
        """ Return the chosen fit statistic defined to minimise for the best fit.

        I.e., returns -2ln(L).

        Parameters
        ----------
        *args, **kwargs :
                All passed to the fit_stat method.

        Returns
        -------
        Float.
        """
        return self._fit_stat(*args, maximize_or_minimize="minimize", **kwargs)

    def _photon_space_reduce(self, ph_bins, ph_mids, ph_widths, srm):
        """ Cuts out photon bins that only include rows of 0s at the top and bottom in the SRM.

        Returns the new photon bins, mid-points, and SRMs.

        Parameters
        ----------
        ph_bins : list of 2d arrays
                A list of all the photon bins for all loaded spectra.

        ph_mids : list of 1d arrays
                A list of the photon energy mid-points for all loaded spectra.

        ph_widths : list of 1d arrays
                A list of the photon energy bin widths for all loaded spectra.

        srm : list of SRMs
                A list of the SRMS for all loaded spectra.

        Returns
        -------
        The 2d (nx2) arrays (photon bins), 1d (n) arrays (mid-points), and 2d (nxm) arrays (SRMs).
        """
        _ph_bins, _ph_mids, _ph_widths, _srm = [], [], [], []
        for i in range(len(srm)):
            non_zero_rows = np.where(np.sum(srm[i], axis=1)!=0)[0] # remove photon energy bins that only have zeros in the srm
            # can't have gaps in the srm rows so only remove from the first or last row in, any zero rows in the middle will stay
            non_zero_rows = np.arange(non_zero_rows[0],non_zero_rows[-1]+1)
            _ph_bins.append(ph_bins[i][non_zero_rows])
            _ph_mids.append(ph_mids[i][non_zero_rows])
            _ph_widths.append(ph_widths[i][non_zero_rows])
            _srm.append(srm[i][non_zero_rows])
        del ph_bins, ph_mids, ph_widths, srm
        return _ph_bins, _ph_mids, _ph_widths, _srm

    def _count_space_reduce(self, ct_binning, ct_mids, ct_obs, ct_err, srm):
        """ Cuts out count bins that only include columns of 0s from the left and right in the SRM.

        Returns the new count bins, mid-points, and SRMs.

        Parameters
        ----------
        ct_binning : list of 1d arrays
                A list of count bin widths for all loaded spectra.

        ct_mids : list of 1d arrays
                A list of the photon energy mid-points for all loaded spectra.

        ct_obs : list of 1d arrays
                A list of observed counts for all loaded spectra.

        ct_err : list of 1d arrays
                A list of the observed count errors for all loaded spectra.

        srm : list of SRMs
                A list of the SRMS for all loaded spectra.

        Returns
        -------
        The 1d (m) arrays (count binning), 1d (m) arrays (mid-points), 1d (m) arrays (observed counts),
        1d (m) arrays (observed count errors), and 2d (nxm) arrays (SRMs).
        """
        _ct_binning, _ct_mids, _ct_obs, _ct_err, _srm = [], [], [], [], []
        for i in range(len(srm)):
            non_zero_cols = np.where(np.sum(srm[i], axis=0)!=0)[0] # remove photon energy bins that only have zeros in the srm
            # can't have gaps in the srm rows so only remove from the first or last row in, any zero rows in the middle will stay
            non_zero_cols = np.arange(non_zero_cols[0],non_zero_cols[-1]+1)
            _ct_binning.append(ct_binning[i][non_zero_cols])
            _ct_mids.append(ct_mids[i][non_zero_cols])
            _ct_obs.append(ct_obs[i][non_zero_cols])
            _ct_err.append(ct_err[i][non_zero_cols])
            _srm.append(srm[i][:,non_zero_cols])
        del ct_binning, ct_mids, ct_obs, ct_err, srm
        return _ct_binning, _ct_mids, _ct_obs, _ct_err, _srm

    def _fit_setup(self):
        """ Returns all information ready to be given the fitting process.

        Returns
        -------
        List of the free parameter float values (free_params_list). All info need to produce model
        and fitting [i.e., photon_channel_bins (list of 2d array), count_channel_mids (list of
        1d arrays), srm (list of 2d arrays), livetime (list of floats), photon_channel_widths (list
        of 1d arrays), observed_counts (list of 1d arrays), observed_count_errors (list of 1d arrays),
        tied_or_frozen_params_list (list of floats), param_name_list_order (list of strings)].
        The correspsonding bounds for each parameter's parameter space (free_bounds, list of tuples),
        and finally the number of free parameters (excluding rParams, orig_free_param_len, int).
        """

        photon_channel_bins, photon_channel_mids, count_channel_mids, srm, livetime, _, photon_e_binning, observed_counts, observed_count_errors = self._loadSpec4fit()

        self._energy_fitting_indices = self._fit_range(count_channel_mids, self.energy_fitting_range) # get fitting indices from the bin midpoints

        # find the free, tie, and frozen params + other model inputs
        free_params_list, tied_or_frozen_params_list, _, param_name_list_order, free_bounds = self._free_and_other()
        self._free_model_param_names = param_name_list_order[:len(free_params_list)]
        self._free_model_param_bounds = free_bounds
        orig_free_param_len = len(free_params_list)

        # sort gain params
        update_fixed_params, update_free_params, update_free_bounds = self._sort_gain()
        self._free_rparam_names = list(update_free_params.keys())
        param_name_list_order[orig_free_param_len:orig_free_param_len] = self._free_rparam_names
        param_name_list_order.extend(list(update_fixed_params.keys()))
        free_params_list.extend(list(update_free_params.values()))
        tied_or_frozen_params_list.extend(list(update_fixed_params.values()))
        self._free_model_param_bounds.extend(update_free_bounds)

        # check if a background is to be included and make the self._scaled_backgrounds attr
        self._include_background()
        self._scaled_backgrounds = self._scaled_background_rates_cut

        # only want values in energy range specified
        srm = self._cut_srm(srm) # saves a couple of seconds
        count_channel_mids = self._cut_counts(count_channel_mids)
        observed_counts = self._cut_counts(observed_counts) # same with the observed counts
        observed_count_errors = self._cut_counts(observed_count_errors)

        # cut the livetimes too if each channel bin is livetime dependent like rhessi
        livetime = [self._cut_counts([lvt]) if not isnumber(lvt) else lvt for lvt in livetime]

        # don't waste time on full rows/columns of 0s in the srms
        photon_channel_bins, photon_channel_mids, photon_channel_widths, srm = self._photon_space_reduce(ph_bins=photon_channel_bins,
                                                                                                         ph_mids=photon_channel_mids,
                                                                                                         ph_widths=photon_e_binning,
                                                                                                         srm=srm) # arf (for NuSTAR at least) makes ~half of the rows all zeros (>80 keV), remove them and cut fitting time by a third
        #photon_e_binning

        # remove the count space reduce since this now needs to reduce the livetimes and baclgrounds if they are there
        # e_binning, count_channel_mids, observed_counts, observed_count_errors, srm = self._count_space_reduce(ct_binning=e_binning,
        #                                                                                                       ct_mids=count_channel_mids,
        #                                                                                                       ct_obs=observed_counts,
        #                                                                                                       ct_err=observed_count_errors,
        #                                                                                                       srm=srm) # this may not do anything if a fitting range has already cut away a lot across counts space
        # return free_params_list, (photon_channel_bins, count_channel_mids, srm, livetime, e_binning, observed_counts, observed_count_errors, tied_or_frozen_params_list, param_name_list_order), self._free_model_param_bounds, orig_free_param_len
        return free_params_list, (photon_channel_bins, count_channel_mids, srm, livetime, photon_channel_widths, observed_counts, observed_count_errors, tied_or_frozen_params_list, param_name_list_order), self._free_model_param_bounds, orig_free_param_len

    def _run_minimiser_core(self, minimise_func, free_parameter_list, statistic_args, free_param_bounds, **kwargs):
        """ Allows user (or us) to define their own (different) minimiser easily.

        This should return the same type of output as Scipy's minimize at the minute. This just passes
        the inputs straight to Scipy's minimiser.

        Parameters
        ----------
        minimise_func : fun, callable
                The fit statistic function that returns value to be minimised.

        free_parameter_list : list of floats
                List of the free parameter float values.

        statistic_args : list of arrays
                All info need to produce model and fitting [i.e., photon_channel_bins (list of 2d array),
                count_channel_mids (list of 1d arrays), srm (list of 2d arrays), livetime (list of floats),
                e_binning (list of 1d arrays), observed_counts (list of 1d arrays), observed_count_errors
                (list of 1d arrays), tied_or_frozen_params_list (list of floats), param_name_list_order
                (list of strings)].

        free_param_bounds : list of tuples
                The correspsonding bounds for each parameter's parameter space.

        **kwargs :
                Passed to Scipy's minimize funciton.
                The `bounds` entry should be handled in the parameter table (.params) and is not
                        passed to minimize.

        Returns
        -------
        Minimiser result as a Scipy OptimizeResult object. Effectively as long as the result.x returns
        the parameter results in the order of `free_parameter_list`.
        """
        return minimize(minimise_func, free_parameter_list, args=statistic_args, bounds=free_param_bounds, **kwargs)

    def fit(self, **kwargs):
        """ Runs the fitting process and returns all found parameter values.

        Parameters
        ----------
        **kwargs :
                Passed to Scipy's minimize funciton (default).
                The `bounds` entry should be handled in the parameter table (.params) and is not
                        passed to minimize.
                A `_hess_step` input can be provided to be passed to the _calc_minimize_error method.

        Returns
        -------
        List of all parameter values after the fitting has taken place.
        """

        free_params_list, stat_args, free_bounds, orig_free_param_len = self._fit_setup()

        kwargs["method"] = "Nelder-Mead" if "method" not in kwargs else kwargs["method"]
        kwargs.pop("bounds", None) # handle the bounds in the Bounds column of the params table attribute

        # step size for Hessian gradient calculation when calculating the covariance matrix
        # step=percentage of each best fit param. E.g., step=0.02 means that the step will be 2% for each param when differentiating
        _hess_step = 0.01 if "_hess_step" not in kwargs else kwargs["_hess_step"]
        kwargs.pop("_hess_step", None)

        # any issues with the step size being a fractional input (e.g., parameter=0) then provide a fix, this gives a direct input to nd.Hessian
        _abs_hess_step = None if "_abs_hess_step" not in kwargs else kwargs["_abs_hess_step"]
        kwargs.pop("_abs_hess_step", None)

        # this has been replaced by a `_run_minimiser_core` so that this can be swapped out easily down the line
        # soltn = minimize(self._fit_stat_minimize,
        #                  free_params_list,
        #                  args=stat_args,
        #                  bounds=free_bounds,
        #                  **kwargs)
        soltn = self._run_minimiser_core(minimise_func=self._fit_stat_minimize,
                                         free_parameter_list=free_params_list,
                                         statistic_args=stat_args,
                                         free_param_bounds=free_bounds,
                                         **kwargs)

        self._minimize_solution = soltn

        std_err = self._calc_minimize_error(stat_args, step=_hess_step, _abs_step=_abs_hess_step)

        # update the model parameters
        self._update_free(table=self.params, updated_free=soltn.x[:orig_free_param_len], errors=std_err[:orig_free_param_len])
        self._update_tied(self.params)

        self._update_free(table=self.rParams, updated_free=soltn.x[orig_free_param_len:], errors=std_err[orig_free_param_len:])
        self._update_tied(self.rParams)

        self._latest_fit_run = "minimiser"# "scipy"

        return list(self.params.param_value)#self.model_params

    def _calc_minimize_error(self, stat_args, step, _abs_step=None):
        """ Calculates errors on the minimize solutions best fit parameters.

        This assumes that free parameters have Gaussian posterior distributions and are
        independent. For a better handle on errors an mcmc run is recommended.

        Parameters
        ----------
        stat_args : list
                A list of all info needed to produce a fit to the data for a model.

        step : float
                A float fraction (>0) indicating the percentage of step for the Hessian matrix
                calculation to take for each parameter. I.e., if step=0.01 and
                free_params=[6,0.05] then Hessian steps would be [0.06,0.0005] (1%).

        _abs_step : float, array-like or StepGenerator object, optional
                Direct input to the `step` arg in nd.Hessian. Takes priority over `step`.

        Returns
        -------
        Array of errors for the free parameters.
        """

        self.sigmas = np.zeros(len(self._minimize_solution.x))
        self.correlation_matrix = np.identity(len(self._minimize_solution.x))
        try:
            self._covariance_matrix = self._calc_hessian(mod_without_x=(lambda free_args: self._fit_stat_minimize(free_args, *stat_args)), fparams=self._minimize_solution.x, step=step, _abs_step=_abs_step)

            for (i,j), cov in np.ndenumerate(self._covariance_matrix):
                # diagonals are the variances
                if i==j:
                    self.sigmas[i] = np.sqrt(cov)
                # now get the correlation between param i and j sqrt( cov(i, j) / (var(i)*var(j)) )
                else:
                    self.correlation_matrix[i,j] = cov / (np.sqrt(self._covariance_matrix[i,i]) * np.sqrt(self._covariance_matrix[j,j]))
        except LinAlgError:
            warnings.warn(f"LinAlgError when calculating the hessian. Errors may not be calculated.")
        return self.sigmas

    def _calc_hessian(self, mod_without_x, fparams, step=0.01, _abs_step=None):
        """ Calculates 2*inv(Hessian).

        Calculates and returns 2*inv(Hessian) which is equal to the covariance matrix
        from a Gaussian posterior -2ln(L) distribution.

        Parameters
        ----------
        mod_without_x : function
                Fit statistic calculation function that takes the free parameter values
                to be passed to nd.Hessian().

        fparams : list of floats
                Best fit values for the free parameters from th efitting process.

        step : float
                A float fraction (>0) indicating the percentage of step for the Hessian matrix
                calculation to take for each parameter. I.e., if step=0.01 and
                free_params=[6,0.05] then Hessian steps would be [0.06,0.0005] (1%). If the param
                is 0 then a best guess at the correct order of magnitude for the step is made.
                Default: 0.01 (or 1% steps)

        _abs_step : float, array-like or StepGenerator object, optional
                Direct input to the `step` arg in nd.Hessian. Takes priority over `step`.

        Returns
        -------
        The 2d square covariance matrix.
        """
        if type(_abs_step)==type(None):
            # get step sizes
            steps = abs(fparams)*step

            # step sized must be >0 but if using fractional step and parameter is 0 then try to get a good step size
            zero_steps = np.where(steps==0)
            try:
                # use the last few params the minimizer used to get an order of magnitude for the step to be
                replacement_steps = np.mean(self._minimize_solution.final_simplex[0][:,zero_steps], axis=0)*step
            except AttributeError:
                replacement_steps = 0
            steps[zero_steps] = replacement_steps if replacement_steps>0 else step
        else:
            steps = _abs_step

        hessian = nd.Hessian(mod_without_x, step=steps)(fparams)
        return 2 * np.linalg.inv(hessian) # 2 appears since our objective function is -2ln(L) and not -ln(L)

    def _calculate_model(self, **kwargs):
        """ Calculates the total count models (in ph. s^-1 keV^-1) for all loaded spectra.

        Parameters
        ----------
        **kwargs :
                Passed to `_pseudo_model`.

        Returns
        -------
        A list (or array) of the count models for each loaded spectrum.
        """

        # let's get all the info needed from LoadSpec for the fit if not provided
        photon_channel_bins, _, count_channel_mids, srm, _, e_binning, ph_e_binning, _, _ = self._loadSpec4fit()

        # want all energies plotted, not just ones in fitting range so change for now and change back later
        _energy_fitting_indices_orig = copy(self._energy_fitting_indices)
        self._energy_fitting_indices = self._fit_range(count_channel_mids, dict(zip(self.loaded_spec_data.keys(), np.tile([0, np.inf], (len(self.loaded_spec_data.keys()), 1)))))

        # make sure using full background counts, not cut version for fitting
        self._scaled_backgrounds = self._scaled_background_rates_full

        # not sure if there are multiple spectra for args
        # if just one spectrum then its just the appropriate entries in the dict from LoadSpec

        # find the free, tie, and frozen params + other model inputs
        free_params_list, tied_or_frozen_params_list, other_inputs, param_name_list_order, _ = self._free_and_other()
        orig_free_param_len = len(free_params_list)
        # clip the counts bins in the srm (keeping all photon bins) to cut down on matrix multiplication
        update_fixed_params, update_free_params, update_free_bounds = self._sort_gain()
        param_name_list_order[orig_free_param_len:orig_free_param_len] = list(update_free_params.keys())
        param_name_list_order.extend(list(update_fixed_params.keys()))
        free_params_list.extend(list(update_free_params.values()))
        tied_or_frozen_params_list.extend(list(update_fixed_params.values()))

        # make sure only the free parameters are getting varied
        mu = self._pseudo_model(free_params_list,
                                tied_or_frozen_params_list,
                                param_name_list_order,
                                photon_channels=photon_channel_bins,
                                photon_channel_widths=ph_e_binning,
                                count_channel_mids=count_channel_mids,
                                total_responses=srm,
                                **kwargs)
        # turn counts s^-1 into counts s^-1 keV^-1
        for m, e in enumerate(e_binning):
            mu[m][0] /= e

        self._energy_fitting_indices = _energy_fitting_indices_orig
        # numpy is better to store but all models need to be the same length, this might not be the case
        try:
            return np.concatenate(tuple(mu))
        except ValueError:
            return [mu[i][0] for i in range(len(mu))]

    def _calc_counts_model(self, photon_model, parameters=None, spectrum="spectrum1", include_bg=False, **kwargs):
        """ Easily calculate a spectrum's count model from the parameter table and user photon model.

        Given a model (a calculated model array or a photon model function) then calcutes the counts
        model for the given spectrum. If a functional input is provided for `photon_model` then
        `parameters` must also be set with a list or dictionary.

        Parameters
        ----------
        photon_model : array/list or function
                A photon model array or a function for a photon model.

        parameters : list or dict
                The corresponding model parameters for a funcitonal photon model.
                Ignored if array is given for `photon_model`.
                Default: None

        spectrum : str
                The spectrum identifier, e.g., "spectrum2".
                Default: "spectrum1"

        include_bg : bool
                Determines whether the background should be added to the model if it is present.
                Don't want it added to sub-models but will want it added to overall models.
                Default: False

        **kwargs :
                Used to set the response parameters differently than what was calculated,
                "gain_slope_spectrum1" and "gain_offset_spectrum1" must be given.

        Returns
        -------
        The count channel mid-points and the calculated count spectrum.
        """

        spec_no = int(spectrum.split("spectrum")[1])

        # don't waste time on full rows/columns of 0s in the srms
        photon_channel_bins, _, _, srm = self._photon_space_reduce(ph_bins=[self.loaded_spec_data[spectrum]['photon_channel_bins']],
                                                                   ph_mids=[self.loaded_spec_data[spectrum]['photon_channel_bins']],
                                                                   ph_widths=[self.loaded_spec_data[spectrum]['photon_channel_binning']],
                                                                   srm=[self.loaded_spec_data[spectrum]['srm']]) # arf (for NuSTAR at least) makes ~half of the rows all zeros (>80 keV), remove them and cut fitting time by a third
        photon_channel_bins, srm = photon_channel_bins[0], srm[0]

        if type(parameters)==type(None):
            m = photon_model
        elif type(parameters) is dict:
            m = photon_model(**parameters, energies=photon_channel_bins)
        elif type(parameters) in [list, type(np.array([]))]:
            m = photon_model(*parameters, energies=photon_channel_bins)
        else:
            print("parameters needs to be a dictionary or list (or np.array) of the photon_model inputs (excluding energies input) or None if photon_model is values and not a function.")
            return

        cts_model = make_model(energies=photon_channel_bins,
                               photon_model=m * np.diff(photon_channel_bins).flatten(),
                               parameters=None,
                               srm=srm)

        if include_bg and ("scaled_background_"+spectrum in self._scaled_backgrounds):
                cts_model += self._scaled_backgrounds["scaled_background_"+spectrum]

        cts_model /= np.diff(self.loaded_spec_data[spectrum]['count_channel_bins']).flatten()

        # if the spectrum has been gain shifted then this will be done but if user provides their own values they will take priority
        if ("gain_slope_spectrum"+str(spec_no) in kwargs) and ("gain_offset_spectrum"+str(spec_no) in kwargs):
            cts_model = self._gain_energies(energies=self.loaded_spec_data[spectrum]['count_channel_mids'],
                                           array=cts_model,
                                           gain_slope=kwargs["gain_slope_spectrum"+str(spec_no)],
                                           gain_offset=kwargs["gain_offset_spectrum"+str(spec_no)])
        elif (self.rParams["Value", "gain_slope_spectrum"+str(spec_no)]!=1) or (self.rParams["Value", "gain_offset_spectrum"+str(spec_no)]!=0):
            cts_model = self._gain_energies(energies=self.loaded_spec_data[spectrum]['count_channel_mids'],
                                           array=cts_model,
                                           gain_slope=self.rParams["Value", "gain_slope_spectrum"+str(spec_no)],
                                           gain_offset=self.rParams["Value", "gain_offset_spectrum"+str(spec_no)])


        return self.loaded_spec_data[spectrum]['count_channel_mids'], cts_model

    def _prepare_submodels(self):
        """ Prepare individual sub-models for use.

        If a string was given to create the overall model for fitting then the component
        models should have been separated into attribute `_separate_models`. If this is the
        case then create the functions for the components and which parameters in the
        parameter table belongs to them.

        Returns
        -------
        None.
        """
        if hasattr(self, '_separate_models'):
            # get [['C*(f_vth + 0)', '1'], ['C*(0 + f_vth)', '2']] from "C*(f_vth + f_vth)"
            # make functions of all the submods, the param_number should correspond with what is in the param table
            self._submod_functions = [self._mod_from_str(model_string=sm[0], custom_param_number=sm[1]) for sm in self._separate_models]
            # get the the params for each model, e.g., from above [['T1', 'EM1'], ['T2', 'EM2']] from 'C*(f_vth(T1,EM1,energies=None) + 0)' and 'C*(0 + f_vth(T2,EM2,energies=None))'
            self._corresponding_submod_inputs = [get_func_inputs(submod_fun)[0] for submod_fun in self._submod_functions]
            # get the values from the param table in the same structure as corresponding_submod_inputs for each loaded spectrum
            self._submod_value_inputs = [[[self.params["Value", _p+"_spectrum"+str(s+1)] for _p in p] for p in self._corresponding_submod_inputs] for s in range(len(self.loaded_spec_data))]

    def _spec_loop_range(self, spectrum):
        """ Finds the range limits to loop through loaded spectra of choice.

        Parameters
        ----------
        spectrum : None, str, int
                The spectrum number identifier, e.g., 2 or "2" for "spectrum2". If set to "combined"
                then the submodel for multiple spectra will be averaged, if set to "all" then
                they will not be averaged and all returned as a list.
                Default: None

        Returns
        -------
        The range as a tuple and a boolean as to whether the count models should be combined and
        averaged or not.
        """
        combine_submods = False
        if str(spectrum)=='combined':
            spec2pick = (1, len(self.loaded_spec_data)+1)
            combine_submods = True
        elif str(spectrum)=='all':
            spec2pick = (1, len(self.loaded_spec_data)+1)
        else:
            spec2pick = (int(spectrum), int(spectrum)+1)
        return spec2pick, combine_submods

    def _calculate_submodels(self, spectrum=None):
        """ After running `_prepare_submodels`, this then calculates the actual count spectrum array for each submodel.

        Parameters
        ----------
        spectrum : None, str, int
                The spectrum number identifier, e.g., 2 or "2" for "spectrum2". If set to "combined"
                then the submodel for multiple spectra will be averaged, if set to "all" then
                they will not be averaged and all returned as a list.
                Default: None

        Returns
        -------
        The count sub-model for the given spectrum (spectra) or None if there are no component models.
        """
        if hasattr(self, '_submod_value_inputs'):

            combine_submods = False
            if type(spectrum)==type(None):
                return None
            elif type(spectrum) in [str, int]:
                spec2pick, combine_submods = self._spec_loop_range(spectrum)
            else:
                print("Not valid spectrum input.")

            all_spec_submods = []
            for s in range(*spec2pick):
                spec_submods = []
                for p in range(len(self._submod_value_inputs[s-1])):
                    energies, cts_mod = self._calc_counts_model(photon_model=self._submod_functions[p], parameters=self._submod_value_inputs[s-1][p], spectrum="spectrum"+str(s))
                    spec_submods.append(cts_mod)
                all_spec_submods.append(spec_submods)

            if combine_submods:
                all_spec_submods = [np.mean(all_spec_submods, axis=0)]

            return all_spec_submods

        else:
            return None

    def _get_max_fit_stat(self):
        """ Return the maximum ln(L) from the minimiser or MCMC analysis result.

        Returns
        -------
        Float.
        """
        if hasattr(self, "_latest_fit_run"):
            if self._latest_fit_run=="minimiser":
                # minimize minimises -2ln(L), so -0.5* to get ln(L)
                return  -0.5*self._minimize_solution.fun
            elif self._latest_fit_run=="mcmc":
                return  self._max_prob
        return 0

    def _fit_stat_str(self):
        """ Creates string to indicate the fit statistic and method used.

        Produce a string that indicates the method used to produce the last
        ln(L) value with the ln(L) value. E.g., last run was Scipy's minimiser
        and it found a maximum likelihood of N then string would be
        "Scipy Max. Total ln(L): N".

        Returns
        -------
        String.
        """
        if hasattr(self, "_latest_fit_run"):
            string = self.loglikelihood.lower()+" Max. Total ln(L): "+str(round(self._get_max_fit_stat(), 1))
            if self._latest_fit_run=="minimiser":
                return  "Minimiser " + string
            elif self._latest_fit_run=="mcmc":
                return  "MCMC " + string
        return ""

    def _bin_data(self, rebin_and_spec):
        """ Bins the count data for a given spectrum.

        Parameters
        ----------
        rebin_and_spec : list
                List of the minimum counts in a group (int) and the spectrum identifier
                for the spectrum to be binned (str, e.g., "spectrum1").

        Returns
        -------
        Eight arrays: the new count bins that have the minimum number of counts in them
        (new_bins), the new bin widths (new_bin_width), new bin mid-points
        (energy_channels), corresponding count rates and count rate errors (count_rates,
        count_rate_errors), old bins and old bin widths for binning other array in the
        same way (old_bins, old_bin_width), and the new bin channel "error"
        (energy_channel_error).
        """
        print("Apply binning for plotting. ", end="")
        new_bins, _, _, new_bin_width, energy_channels, count_rates, count_rate_errors, _, _orig_in_extras = self._rebin_data(spectrum=rebin_and_spec[1], group_min=rebin_and_spec[0])
        old_bins = self.loaded_spec_data[rebin_and_spec[1]]["count_channel_bins"] if not _orig_in_extras else self.loaded_spec_data[rebin_and_spec[1]]["extras"]["original_count_channel_bins"]
        old_bin_width = self.loaded_spec_data[rebin_and_spec[1]]["count_channel_binning"] if not _orig_in_extras else self.loaded_spec_data[rebin_and_spec[1]]["extras"]["original_count_channel_binning"]
        energy_channel_error = new_bin_width/2
        return new_bins, new_bin_width, energy_channels, count_rates, count_rate_errors, old_bins, old_bin_width, energy_channel_error

    def _bin_model(self, count_rate_model, old_bin_width, old_bins, new_bins, new_bin_width):
        """ Rebins a given array of data in count bins energies to a new set of bins given.

        Parameters
        ----------
        count_rate_model, old_bin_width, old_bins, new_bins : array
                Array of the data (cts s^-1 keV^-1), current bins widths, current bins (for data axis==0), and new bins (for data axis==0).
                Need len(data)==len(old_bins).

        Returns
        -------
        Array.
        """
        return rebin_any_array(count_rate_model*old_bin_width, old_bins, new_bins, combine_by="sum") / new_bin_width

    def _bin_spec4plot(self, rebin_and_spec, count_rate_model, _return_cts_rate_mod=True):
        """ Bins the count model given based on the given spectrum's rebinning.

        Parameters
        ----------
        rebin_and_spec : list
                List of the minimum counts in a group (int) and the spectrum identifier
                for the spectrum to be binned (str, e.g., "spectrum1").

        count_rate_model : array
                The count model array.

        _return_cts_rate_mod : bool
                Defines whether the counts rate model or None should be returned. Only here so that the
                plot() method can be used before a model is even defined to allow the user to see the spectra.
                Default: True

        Returns
        -------
        Nine arrays: the new count bins that have the minimum number of counts in them
        (new_bins), the new bin widths (new_bin_width), old bins and old bin widths for
        binning other array in the same way (old_bins, old_bin_width), new bin mid-points
        (energy_channels), corresponding count rate errors and count rates (energy_channel_error,
        count_rates), the new bin channel "error" (energy_channel_error), and the binned count
        rate model.
        """
        new_bins, new_bin_width, energy_channels, count_rates, count_rate_errors, old_bins, old_bin_width, energy_channel_error = self._bin_data(rebin_and_spec)

        count_rate_model = self._bin_model(count_rate_model, old_bin_width, old_bins, new_bins, new_bin_width) if _return_cts_rate_mod else None

        return new_bins, new_bin_width, old_bins, old_bin_width, energy_channels, energy_channel_error, count_rates, count_rate_errors, count_rate_model

    def _get_mean_counts_across_specs(self):
        """ Calculates the mean counts for each count energy bin for all loaded spectra.

        Used when producing the combined spectral plots.

        Returns
        -------
        An array of the mean counts for each count energy bin for all loaded spectra.
        """
        _counts = []
        for s in range(len(self.loaded_spec_data)):
            _counts.append(self.loaded_spec_data['spectrum'+str(s+1)]['counts'])
        return np.mean(np.array(_counts), axis=0)

    def _bin_comb4plot(self, rebin_and_spec, count_rate_model, energy_channels, energy_channel_error, count_rates, count_rate_errors, _return_cts_rate_mod=True):
        """ Returns all information to rebin some counts data.

        Mainly rebins the data to plot for the combined plot.

        Parameters
        ----------
        rebin_and_spec : list
                List of the minimum counts in a group (int) and the spectrum identifier
                for the spectrum to be binned (str, e.g., "spectrum1").

        count_rate_model : 1d array
                The count model array.

        energy_channels, energy_channel_error, count_rates, count_rate_errors : 1d arrays
                The energy channel mid-points, energy channel half bin widths, count rates,
                and count rate errors, respectively.

        _return_cts_rate_mod : bool
                Defines whether the counts rate model or None should be returned. Only here so that the
                plot() method can be used before a model is even defined to allow the user to see the spectra.
                Default: True

        Returns
        -------
        A 2d array of new bins (new_bins), 1d array of bin widths (new_bin_width), 2d array of old bins (old_bins),
        1d array of old bin widths (old_bin_width), 1d array of new bin mid-points (energy_channels), 1d array of
        energy errors/half bin width (energy_channel_error), and three more 1d arrays (count_rates, count_rate_errors,
        count_rate_model).
        """
        old_bins = np.column_stack((energy_channels-energy_channel_error, energy_channels+energy_channel_error))
        old_bin_width = energy_channel_error*2
        new_bins, mask = self._group_cts(channel_bins=old_bins, counts=self._get_mean_counts_across_specs(), group_min=rebin_and_spec[0], spectrum=rebin_and_spec[1], verbose=True)

        mask[mask!=0] = 1
        # use group counts to make sure see where the grouping stops, any zero entries don't have grouped counts and so should be nans to avoid plotting
        mask[mask==0] = np.nan

        new_bin_width = np.diff(new_bins).flatten()
        count_rates = (rebin_any_array(data=count_rates*old_bin_width, old_bins=old_bins, new_bins=new_bins, combine_by="sum") / new_bin_width) * mask
        count_rate_errors = (rebin_any_array(data=count_rate_errors*old_bin_width, old_bins=old_bins, new_bins=new_bins, combine_by="quadrature") / new_bin_width) * mask
        energy_channels = np.mean(new_bins, axis=1)
        count_rate_model = rebin_any_array(count_rate_model*old_bin_width, old_bins, new_bins, combine_by="sum") / new_bin_width if _return_cts_rate_mod else None
        energy_channel_error = new_bin_width/2

        return new_bins, new_bin_width, old_bins, old_bin_width, energy_channels, energy_channel_error, count_rates, count_rate_errors, count_rate_model

    def _plot_mcmc_mods_combin(self, ax, res_ax, res_info, hex_grid=False, _rebin_info=None):
        """ Plots model runs for a combined data plot.

        Parameters
        ----------
        ax : axes object
                Axes for the data and model.

        res_ax : axes object
                Axes for the residuals.

        res_info : list of length 2
                First entry is the combined data and second entry is the correpsonding uncertainty list.

        hex_grid : bool
                Indicates whether separate model lines should be drawn for MCMC runs or produce hexagonal
                histogram grid for all runs.
                Default: False

        _rebin_info : list of lrngth 4
                Inputs for `_bin_model` method: [old_bin_width, old_bins, new_bins, new_bin_width].
                Default: None

        Returns
        -------
        None.
        """
        comb = np.mean(self._mcmc_mod_runs,axis=0)
        res_comb = []
        for comb_ctr in comb:
            e_mids = self._mcmc_mod_runs_emids
            if _rebin_info is not None:
                comb_ctr = self._bin_model(comb_ctr, *_rebin_info)
                e_mids = np.mean(_rebin_info[2], axis=1)

            residuals = [(res_info[0][i] - comb_ctr[i])/res_info[1][i] if res_info[1][i]>0 else 0 for i in range(len(res_info[1]))]
            res_comb.append(residuals)
            residuals = np.column_stack((residuals,residuals)).flatten() # non-uniform binning means we have to plot every channel edge instead of just using drawstyle='steps-mid'in .plot()
            if not hex_grid:
                ax.plot(e_mids, comb_ctr, color="orange", alpha=0.05, zorder=0)#, color="grey"
                res_ax.plot(res_info[2], residuals, color="orange", alpha=0.05, zorder=0)#, color="grey"
        if hex_grid:
            es, cts_list, res_list = np.array(list(e_mids)*len(comb_ctr)), np.array(comb_ctr).flatten(), np.array(res_comb).flatten()
            keep = np.where((self.plot_xlims[0]<=es) & (es<=self.plot_xlims[1]) & (cts_list>0)) #& (self.plot_ylims[0]<=cts_list) & (cts_list<=self.plot_ylims[1])
            ax.hexbin(es[keep], cts_list[keep], gridsize=100, cmap='Oranges', yscale='log', zorder=0, mincnt=1)#, alpha=0.8, bins='log'
            res_ax.hexbin(es[keep], res_list[keep], gridsize=(100,20), cmap='Oranges', zorder=0, mincnt=1)

    def _plot_mcmc_mods_sep_pars(self, spectrum_indx, mcmc_freepar_labels, hex_grid):
        """ Creates lists fo the parameters.

        All frozen (or params tied to frozen) are set to their fixed value. If the parameter is one that was
        sampled over (or tied to a param that was sampled over) then the entry is the name of the sampled
        parameter. When looping through the MCMC samples the named (string) entries are then replaced with
        their sample value.

        Parameters
        ----------
        spectrum_indx : int
                Index of spectrum in the `loaded_spec_data` attribute.

        mcmc_freepar_labels : list of strings
                List of parameter/rparameter names sampled over.

        hex_grid : bool
                Indicates whether separate model lines should be drawn for MCMC runs or produce hexagonal
                histogram grid for all runs.

        Returns
        -------
        Two lists (_spec_pars, _spec_rpars). Either fixed value entries in the list or names of the MCMC
        sample to go in that entry.
        """
        s = spectrum_indx

        # fixed values take their values, tied values entry displays the name they are attached to, free are set to "free"
        spec_pars = [self.params["Value",name] if (self.params["Status",name]=="frozen") else self.params["Status",name][4:] if (self.params["Status",name].startswith("tie")) else name for name in self._param_groups[s]]
        spec_rpars = {name:(self.rParams["Value",name] if (self.rParams["Status",name]=="frozen") else self.rParams["Status",name][4:] if (self.rParams["Status",name].startswith("tie")) else name) for name in self._response_param_names[s*2:2*s+2]}

        # if tied to something fixed, if entry for param is a string (free or tied), if entry is a param that is frozen then it isn't in mcmc_freepar_labels
        _spec_pars = [self.params["Value",name] if ((type(name) is str) and (name not in mcmc_freepar_labels)) else name for name in spec_pars]
        _spec_rpars = {name:(self.rParams["Value",val] if ((type(val) is str) and (val not in mcmc_freepar_labels)) else val) for name,val in spec_rpars.items()}

        # assign _samp_inds to random ones if we have them AND lines are to be plotted, else make it all samples
        self._samp_inds = self._samp_inds if hasattr(self, "_samp_inds") and (not hex_grid) else np.arange(len(self.all_mcmc_samples))

        return _spec_pars, _spec_rpars

    def _randsamples_or_all(self, hex_grid, num_of_samples):
        """ Sets `__mcmc_samples__` and maybe `_samp_inds`.

        If `hex_grid` is True then `__mcmc_samples__` becomes all samples from the MCMC. If `hex_grid` is
        False then `__mcmc_samples__` becomes a random number of samples (`num_of_samples`) from the MCMC
        run with `all_mcmc_samples` indices `_samp_inds`.

        Parameters
        ----------
        hex_grid : bool
                Indicates whether separate model lines should be drawn for MCMC runs or produce hexagonal
                histogram grid for all runs.

        num_of_samples : int
                Number of random sample entries to use for MCMC model run plotting.

        Returns
        -------
        None.
        """
        if not hex_grid:
            self._samp_inds = np.random.randint(len(self.all_mcmc_samples), size=num_of_samples)
            self.__mcmc_samples__ = self.all_mcmc_samples[self._samp_inds]
        elif hex_grid:
            self.__mcmc_samples__ = self.all_mcmc_samples

    def _make_or_add_mcmc_mod_runs(self, _randcts, e_mids):
        """ From individual spectra, create or add a sampled models list.

        Parameters
        ----------
        _randcts : int
                Lists of count rates produced from the MCMC samples.

        e_mids : list
                The energy mid-points list for the count rate models.

        Returns
        -------
        None.
        """
        # can only have a run for each loaded spectrum, if there is more then it must be from this being run multiple times
        if hasattr(self, "_mcmc_mod_runs") and len(self._mcmc_mod_runs)>=len(self.loaded_spec_data):
            del self._mcmc_mod_runs

        if hasattr(self, "_mcmc_mod_runs"):
            self._mcmc_mod_runs.append(_randcts)
        else:
            self._mcmc_mod_runs = [_randcts]
        self._mcmc_mod_runs_emids = e_mids

    def _no_mcmc_change(self):
        """ Checks if the MCMC samples being plotted have changed since the last
        time this was run. E.g., return False if more runs have been burned, etc.

        Returns
        -------
        True if there has not been a change to the MCMC samples to plot since this
        was last run, False if there has been a change.
        """
        try:
            return np.array_equal(self.__mcmc_samples__, self.all_mcmc_samples[self._samp_inds])
        except IndexError:
            return False

    def _recalc_plotting_mcmc_samples(self, hex_grid, num_of_samples):
        """ Decides whether to recalculate information to plot MCMC runs.

        This is based on whether the MCMC samples have changed since the last time they were plotted, if
        all samples are needed for a hexagonal grid, or if the number of samples to use has changed. This
        method helps to reproduce the same plots if the user doesn't change anything rather than just new
        samples being randomly chosen everytime a plot is created.

        Parameters
        ----------
        num_of_samples : int
                Number of random sample entries to use for MCMC model run plotting.
                Default: 100

        hex_grid : bool
                Indicates whether separate model lines should be drawn for MCMC runs or produce hexagonal
                histogram grid for all runs. If True `num_of_samples` is ignored.
                Default: False

        Returns
        -------
        None.
        """
        if not hasattr(self, "__mcmc_samples__") or not self._no_mcmc_change() or (hasattr(self, "_samp_inds") and len(self._samp_inds)!=num_of_samples):
            self._randsamples_or_all(hex_grid, num_of_samples)

    def _plot_mcmc_mods(self, ax, res_ax, res_info, spectrum="combined", num_of_samples=100, hex_grid=False, _rebin_info=None):
        """ Plots MCMC runs (and residuals) on the given axes.

        Parameters
        ----------
        ax : axes object
                Axes for the data and model.

        res_ax : axes object
                Axes for the residuals.

        res_info : list of length 2
                First entry is the combined data and second entry is the correpsonding uncertainty list.

        spectrum_indx : str
                Spectrum ID in the `loaded_spec_data` attribute. E.g., "spectrum1", etc.
                Default: "combined"

        num_of_samples : int
                Number of random sample entries to use for MCMC model run plotting.
                Default: 100

        hex_grid : bool
                Indicates whether separate model lines should be drawn for MCMC runs or produce hexagonal
                histogram grid for all runs. If True `num_of_samples` is ignored.
                Default: False

        _rebin_info : list of lrngth 4
                Inputs for `_bin_model` method: [old_bin_width, old_bins, new_bins, new_bin_width].
                Default: None

        Returns
        -------
        None.
        """

        if spectrum=="combined":
            self._plot_mcmc_mods_combin(ax, res_ax, res_info, hex_grid=hex_grid, _rebin_info=_rebin_info)
            return

        mcmc_freepar_labels = self._free_model_param_names+self._free_rparam_names
        s = int(spectrum.split("spectrum")[-1])-1

        _spec_pars, _spec_rpars = self._plot_mcmc_mods_sep_pars(s, mcmc_freepar_labels, hex_grid)

        # ensure same samples are used across all spectra (needs to be when combining spectra), only update if more runs have been added since __mcmc_samples__ was created
        # or if plotting lines then hexagons
        self._recalc_plotting_mcmc_samples(hex_grid, num_of_samples)

        _randcts = []
        _randctsres = []
        for _params in self.__mcmc_samples__:
            # take list of [1,0.1,3,par1,70] where par1 was sampled over and replace par1 a sample value
            _pars = [_params[mcmc_freepar_labels.index(p)] if type(p)==str else p for p in _spec_pars]
            _rpars = {name:(_params[mcmc_freepar_labels.index(val)] if type(val)==str else val) for name,val in _spec_rpars.items()}
            e_mids, ctr = self._calc_counts_model(photon_model=self._model, parameters=_pars, spectrum="spectrum"+str(s+1), include_bg=True, **_rpars)
            _randcts.append(ctr)
            if _rebin_info is not None:
                ctr = self._bin_model(ctr, *_rebin_info)
                e_mids = np.mean(_rebin_info[2], axis=1)

            residuals = [(res_info[0][i] - ctr[i])/res_info[1][i] if res_info[1][i]>0 else 0 for i in range(len(res_info[1]))]
            _randctsres.append(residuals)
            residuals = np.column_stack((residuals,residuals)).flatten() # non-uniform binning means we have to plot every channel edge instead of just using drawstyle='steps-mid'in .plot()
            if not hex_grid:
                ax.plot(e_mids, ctr, color="orange", alpha=0.05, zorder=0)#, color="grey"
                res_ax.plot(res_info[2], residuals, color="orange", alpha=0.05, zorder=0)#, color="orangegrey"

        if hex_grid:
            es, cts_list, res_list = np.array(list(e_mids)*len(_randcts)), np.array(_randcts).flatten(), np.array(_randctsres).flatten()
            keep = np.where((self.plot_xlims[0]<=es) & (es<=self.plot_xlims[1]) & (cts_list>0)) #& (self.plot_ylims[0]<=cts_list) & (cts_list<=self.plot_ylims[1])
            ax.hexbin(es[keep], cts_list[keep], gridsize=100, cmap='Oranges', yscale='log', zorder=0, mincnt=1)#, alpha=0.8, bins='log'
            res_ax.hexbin(es[keep], res_list[keep], gridsize=(100,20), cmap='Oranges', zorder=0, mincnt=1)

        self._make_or_add_mcmc_mod_runs(_randcts, e_mids)

    def _combin_fitting_range(self, submod_spec, fitting_range):
        """ Get the fitting range(s) for the combined plot.

        Parameters
        ----------
        submod_spec : str, int
                The spectrum number identifier, e.g., 2 or "2" for "spectrum2". Set to "combined"
                for combined plot.

        fitting_range : dict
                Fitting range corresponding to the spectrum being plotted.

        Returns
        -------
        The (potentially modified) fitting range list (fitting_range).
        """
        if submod_spec=="combined":
            frvs = np.array(list(fitting_range.values()))
            if (frvs == frvs[0]).all():
                fitting_range = [frvs[0]] if np.size(frvs[0])>2 else frvs[0]
            else:
                fitting_range = frvs
        return fitting_range

    def _plot_fitting_range(self, res, fitting_range, submod_spec):
        """ Handles the fitting range plotting for one loaded spectrum.

        Parameters
        ----------
        res : axes object
                Axes for the residual data.
                Default: None

        fitting_range : list, None
                Fitting range corresponding to the spectrum being plotted.
                Default: None

        submod_spec : str, int
                The spectrum number identifier, e.g., 2 or "2" for "spectrum2". Set to "combined"
                for combined plot.
                Default: None

        Returns
        -------
        None.
        """
        if type(fitting_range)!=type(None):
            # if combined then need all fitting ranges
            fitting_range = self._combin_fitting_range(submod_spec, fitting_range)

            if (np.size(fitting_range)==2) and (type(fitting_range[0])!=list):
                # simple fitting range; e.g., [2,4]
                res.plot([fitting_range[0], fitting_range[1]], [self.res_ylim[1],self.res_ylim[1]], linewidth=9, color="#17A589", solid_capstyle="butt")
            elif (np.size(fitting_range)>2) and (submod_spec!="combined"):
                # more complex fitting range; e.g., [[2,4], [5,9]]
                fitting_range = np.array(fitting_range)
                gaps_in_ranges = np.concatenate((fitting_range[:,-1][:-1][:,None], fitting_range[:,0][1:][:,None]), axis=1)
                xs = np.insert(fitting_range, np.arange(1, len(fitting_range)), gaps_in_ranges, axis=0).flatten()
                ys = ([self.res_ylim[1],self.res_ylim[1], np.nan, np.nan]*len(fitting_range))[:-2]
                res.plot(xs, ys, linewidth=9, color="#17A589", solid_capstyle="butt")
            else:
                # for more multiple fitting ranges for combined plots; e.g., [[2,4], [[2,4], [5,9]]]
                for suba, f in enumerate(fitting_range):
                    if np.size(f)>2:
                        f = np.array(f)
                        gaps_in_ranges = np.concatenate((f[:,-1][:-1][:,None], f[:,0][1:][:,None]), axis=1)
                        xs = np.insert(f, np.arange(1, len(f)), gaps_in_ranges, axis=0).flatten()
                        ys = ([self.res_ylim[1],self.res_ylim[1], np.nan, np.nan]*len(f))[:-2]
                    else:
                        f = np.squeeze(f) # since this can sometimes be a list of one list depending on how the fitting range is defined
                        xs, ys = [f[0], f[1]], [self.res_ylim[1],self.res_ylim[1]]
                    res.plot(xs, np.array(ys), linewidth=9*1.2**suba, color="#17A589", solid_capstyle="butt", alpha=0.7-0.6*(suba/len(fitting_range))) # span the aphas between 0.1 and 0.7

    def _plot_submodels(self, axs, submod_spec, rebin, bin_info):
        """ Plots sub/component models if available.

        Parameters
        ----------
        axs : axes object
                Axes for the sub-model.

        submod_spec : str
                The spectrum number identifier, e.g., 2 or "2" for "spectrum2". Set to "combined"
                for combined plot.

        rebin : list
                List of the minimum counts in a group (int) and the spectrum identifier for the spectrum to
                be binned (str, e.g., "spectrum1").

        bin_info : tuple of arrays
                The old energy bin widths (old_bin_width), old bins (old_bins), new bins and bin widths
                (new_bins, new_bin_width), and x-axis poitions for model points (energy_channels),
                respectively.

        Returns
        -------
        List of sub-model arrays (spec_submods) and a list of strings (hex colour codes) for each sub-model's
        parameters (submod_param_cols). Colour for every parameter, each group of parameters for a sub-model
        is has the same colour.
        """
        old_bin_width, old_bins, new_bins, new_bin_width, energy_channels = bin_info
        spec_submods, submod_param_cols = None, None

        if hasattr(self, '_corresponding_submod_inputs'):
            spec_submods = self._calculate_submodels(spectrum=submod_spec)
            colour_pool = cycle(self._colour_list)
            submod_cols = [next(colour_pool) for _ in range(len(self._corresponding_submod_inputs))]

            if submod_spec!="combined":
                common = set([common for pl in range(len(self._corresponding_submod_inputs)-1) for common in (set(self._corresponding_submod_inputs[pl]) & set(self._corresponding_submod_inputs[pl+1]))])
                self._params2get = [[unique+"_spectrum"+submod_spec for unique in pl if unique not in common] for pl in self._corresponding_submod_inputs]
                submod_param_cols = [[smcs]*len(params) for smcs, params in zip(submod_cols, self._params2get)] + [["blue"]*len(common)]
                submod_param_cols = [param for params in submod_param_cols for param in params]

            for sm, col in zip(spec_submods[0], submod_cols):
                if type(rebin)!=type(None):
                    sm = rebin_any_array(sm*old_bin_width, old_bins, new_bins, combine_by="sum") / new_bin_width
                axs.plot(energy_channels, sm, alpha=0.7, color=col)

        return spec_submods, submod_param_cols

    def _annotate_params(self, axs, param_str, submod_param_cols, _xycoords):
        """ Annotates the plot with parameter names and values.

        Parameters
        ----------
        axs : axes objects
                Axes for data.

        param_str : list of str
                List of strings with all formatted parameter names and values.

        submod_param_cols : list of strings (hex colour codes)
                Colours for each (sub-)model's parameters.

        _xycoords : str
                Tells matplotlib the coordinate system to use when placing the
                text (e.g., "axes fraction").

        Returns
        -------
        None.
        """
        if hasattr(self, '_params2get'):
            # if we know about sub-models
            rainbow_text_lines((0.95, 0.95), strings=param_str, colors=submod_param_cols, xycoords=_xycoords, verticalalignment="top", horizontalalignment="right", ax=axs)
        else:
            axs.annotate("".join(param_str), (0.95, 0.95), xycoords=_xycoords, verticalalignment="top", horizontalalignment="right", color="navy")

    def _plot_params(self, axs, res, plot_params, submod_param_cols):
        """ Annotates the plot with parameter/response parameter names and values.

        Parameters
        ----------
        axs, res : axes objects
                Axes for data and residuals, respectively.

        plot_params : list of strings
                List of parameter names from the parameter table corresponding to this spectrum.

        submod_param_cols : list of strings (hex colour codes)
                Colours for each (sub-)model's parameters.

        Returns
        -------
        None.
        """
        if type(plot_params)==list:
            param_str = []
            _xycoords = "axes fraction"
            for p in plot_params:
                par_spec = p.split("_spectrum")
                error = self.params["Error", p]
                if np.all(error)==np.all([0,0]):
                    param_str += [par_spec[0]+": {0:.2e}".format(self.params["Value", p])+"\n"]
                else:
                    param_str += [par_spec[0]+": {0:.2e}".format(self.params["Value", p])+"$^{{+{0:.2e}}}_{{-{1:.2e}}}$".format(error[1], error[0])+"\n"] # str(round(self.params["Value", p], 2))
            # join param strings correctly and annotate the axes depending on whether they should be coloured with sub-models if present
            self._annotate_params(axs, param_str, submod_param_cols, _xycoords)

            rparam_str = ""
            for rp, rname in zip(["gain_slope_spectrum"+par_spec[1], "gain_offset_spectrum"+par_spec[1]], ["slope", "offset"]):
                rerror = self.rParams["Error", rp]
                if np.all(rerror)==np.all([0,0]):
                    rparam_str += "\n"+rname+": "+str(round(self.rParams["Value", rp], 2))
                else:
                    rparam_str += "\n"+rname+": {0:.2f}".format(self.rParams["Value", rp])+"$^{{+{0:.2f}}}_{{-{1:.2f}}}$".format(rerror[1], rerror[0])
            axs.annotate(rparam_str, (0.01, 0.01), xycoords=_xycoords, verticalalignment="bottom", horizontalalignment="left", fontsize="small", color="red", alpha=0.5)

            res.annotate(self._fit_stat_str(), (0.99, 0.01), xycoords=_xycoords, verticalalignment="bottom", horizontalalignment="right", fontsize="small", color="navy", alpha=0.7)

    def _setup_rebin_plotting(self, rebin_and_spec, data_arrays, _return_cts_rate_mod=True):
        """ Checks if plot wants rebinned data/models and returns relevant information.

        Parameters
        ----------
        data_arrays : tuple of 1d arrays
                The energy channel mid-points (energy_channels), energy channel half bin widths
                (energy_channel_error), count rates (count_rates), count rate errors
                (count_rate_errors), and count rate model array (count_rate_model), respectively.

        rebin_and_spec : list
                List of the minimum counts in a group (int) and the spectrum identifier for the spectrum to
                be binned (str, e.g., "spectrum1").

        _return_cts_rate_mod : bool
                Defines whether the counts rate model or None should be returned. Only here so that the
                plot() method can be used before a model is even defined to allow the user to see the spectra.
                Default: True

        Returns
        -------
        A 2d array of new bins (new_bins), 1d array of bin widths (new_bin_width), 2d array of old bins (old_bins),
        1d array of old bin widths (old_bin_width), 1d array of new bin mid-points (energy_channels), 1d array of
        energy errors/half bin width (energy_channel_error), three more 1d arrays (count_rates, count_rate_errors,
        count_rate_model), and a 1d array of energies for residual plotting (energy_channels_res), respectively.
        """
        rebin_val, rebin_spec = rebin_and_spec[0], rebin_and_spec[1]
        energy_channels, energy_channel_error, count_rates, count_rate_errors, count_rate_model = data_arrays
        if type(rebin_val)!=type(None):
            if rebin_spec in list(self.loaded_spec_data.keys()):
                new_bins, new_bin_width, old_bins, old_bin_width, energy_channels, energy_channel_error, count_rates, count_rate_errors, count_rate_model = self._bin_spec4plot(rebin_and_spec,
                                                                                                                                                                                count_rate_model,
                                                                                                                                                                                _return_cts_rate_mod=_return_cts_rate_mod)
            elif rebin_spec=='combined':
                new_bins, new_bin_width, old_bins, old_bin_width, energy_channels, energy_channel_error, count_rates, count_rate_errors, count_rate_model = self._bin_comb4plot(rebin_and_spec,
                                                                                                                                                                                count_rate_model,
                                                                                                                                                                                energy_channels,
                                                                                                                                                                                energy_channel_error,
                                                                                                                                                                                count_rates,
                                                                                                                                                                                count_rate_errors,
                                                                                                                                                                                _return_cts_rate_mod=_return_cts_rate_mod)
            energy_channels_res = new_bins.flatten()
        else:
            old_bin_width, old_bins, new_bins, new_bin_width = None, None, None, None
            energy_channels_res = np.column_stack((energy_channels-energy_channel_error, energy_channels+energy_channel_error)).flatten()

        return new_bins, new_bin_width, old_bins, old_bin_width, energy_channels, energy_channel_error, count_rates, count_rate_errors, count_rate_model, energy_channels_res

    def _get_xlims(self, energy_channels, count_rates):
        """ Get sensible x-limits for plotting.

        Parameters
        ----------
        energy_channels, count_rates : 1d arrays
                The energy channel mid-points (energy_channels) and count rates
                (count_rates), respectively.

        Returns
        -------
        None.
        """
        if not hasattr(self, 'plot_xlims'):
            extrema_x = LL_CLASS.remove_non_numbers(np.where(count_rates!=0)[0])
            minx, maxx = energy_channels[extrema_x[0]], energy_channels[extrema_x[-1]]
            self.plot_xlims = [0.9*minx, 1.1*maxx]

    def _get_ylims(self, count_rates, count_rate_model):
        """ Get sensible y-limits for plotting.

        Parameters
        ----------
        count_rates, count_rate_model : 1d arrays
                Count rates (count_rates) and count rate model array (count_rate_model),
                respectively.

        Returns
        -------
        None.
        """
        if not hasattr(self, 'plot_ylims'):
            miny = np.min(LL_CLASS.remove_non_numbers(count_rates[count_rates!=0]))
            maxy = np.max([np.max(LL_CLASS.remove_non_numbers(count_rates[count_rates!=0])), np.max(LL_CLASS.remove_non_numbers(count_rate_model[count_rate_model!=0]))])
            self.plot_ylims = [0.9*miny, 1.1*maxy]

    def _plot_1spec(self,
                    data_arrays,
                    axes=None,
                    fitting_range=None,
                    plot_params=None,
                    log_plotting_info=None,
                    submod_spec=None,
                    rebin_and_spec=None,
                    num_of_samples=100,
                    hex_grid=False):
        """ Handles the plotting for one loaded spectrum.

        Parameters
        ----------
        data_arrays : tuple of 1d arrays
                The energy channel mid-points (energy_channels), energy channel half bin widths
                (energy_channel_error), count rates (count_rates), count rate errors
                (count_rate_errors), and count rate model array (count_rate_model), respectively.

        axes : axes object
                Axes for the data and model. If None use `plt.gca()`.
                Default: None

        fitting_range : list
                Fitting range corresponding to this spectrum.
                Default: None

        plot_params : list of strings
                List of parameter names from the parameter table corresponding to this spectrum.
                Default: None

        log_plotting_info : dict
                Dictionary to record the arrays used to plot the spectrum.
                Default: None

        submod_spec : None, str, int
                The spectrum number identifier, e.g., 2 or "2" for "spectrum2". Set to "combined"
                for combined plot.
                Default: None

        rebin_and_spec : list
                List of the minimum counts in a group (int) and the spectrum identifier for the spectrum to
                be binned (str, e.g., "spectrum1").
                Default: None

        num_of_samples : int
                Number of random sample entries to use for MCMC model run plotting.
                Default: 100

        hex_grid : bool
                Indicates whether separate model lines should be drawn for MCMC runs or produce hexagonal
                histogram grid for all runs. If True `num_of_samples` is ignored.
                Default: False

        Returns
        -------
        Spectrum axes and residuals axes.
        """
        energy_channels, energy_channel_error, count_rates, count_rate_errors, count_rate_model = data_arrays

        axs = axes if type(axes)!=type(None) else plt.gca()
        fitting_range = fitting_range if type(fitting_range)!=type(None) else self.energy_fitting_range
        self.res_ylim = self.res_ylim if hasattr(self, 'res_ylim') else [-7,7]

        axs.set_xlabel('Energy [keV]')
        axs.set_ylabel('Count Spectrum [cts s$^{-1}$ keV$^{-1}$]')

        # axes limits
        self._get_xlims(energy_channels, count_rates)
        axs.set_xlim(self.plot_xlims)

        self._get_ylims(count_rates, count_rate_model)
        axs.set_ylim(self.plot_ylims)

        axs.set_yscale('log')

        # check if the plot is to produce rebinned data and models
        _return_cts_rate_mod = True if hasattr(self, "_model") else False
        new_bins, new_bin_width, old_bins, old_bin_width, energy_channels, energy_channel_error, count_rates, count_rate_errors, count_rate_model, energy_channels_res = self._setup_rebin_plotting(rebin_and_spec, data_arrays, _return_cts_rate_mod=_return_cts_rate_mod)
        self._plot_rebin_info = [old_bin_width, old_bins, new_bins, new_bin_width] # for background is needed

        ## plot data
        axs.errorbar(energy_channels,
                     count_rates,
                     xerr=energy_channel_error,
                     yerr=count_rate_errors,
                     color='k',
                     fmt='.',
                     markersize=0.01,
                     label='Data',
                     alpha=0.8)

        if not hasattr(self, "_model"):
            return axs, None

        axs.xaxis.set_tick_params(labelbottom=False)
        axs.get_xaxis().set_visible(False)

        residuals = [(count_rates[i] - count_rate_model[i])/count_rate_errors[i] if count_rate_errors[i]>0 else 0 for i in range(len(count_rate_errors))]
        residuals = np.column_stack((residuals,residuals)).flatten() # non-uniform binning means we have to plot every channel edge instead of just using drawstyle='steps-mid'in .plot()

        # if we have submodels, plot them
        spec_submods, submod_param_cols = self._plot_submodels(axs, submod_spec, rebin_and_spec[0], (old_bin_width, old_bins, new_bins, new_bin_width, energy_channels))

        #residuals plotting
        divider = make_axes_locatable(axs)
        res = divider.append_axes('bottom', 1.2, pad=0.2, sharex=axs)
        res.axhline(0, linestyle=':', color='k')
        res.set_ylim(self.res_ylim)
        # res.set_ylabel('(y$_{Data}$ - y$_{Model}$)/$\sigma_{Error}$')
        res.set_ylabel('($D - M$)/$\sigma$')
        res.set_xlim(self.plot_xlims)
        res.set_xlabel("Energy [keV]")

        # plot final result, final model and resulting residuals
        if self._plr:
            axs.plot(energy_channels, count_rate_model, linewidth=2, color="k")
            res.plot(energy_channels_res, residuals, color='k', alpha=0.8)#, drawstyle='steps-mid'

        if self._latest_fit_run=="mcmc":
            _rebin_info = [old_bin_width, old_bins, new_bins, new_bin_width] if type(rebin_and_spec[0])!=type(None) else None
            self._plot_mcmc_mods(axs, res, [count_rates, count_rate_errors, energy_channels_res], spectrum=submod_spec, num_of_samples=num_of_samples, hex_grid=hex_grid, _rebin_info=_rebin_info)

        # plot fitting range
        self._plot_fitting_range(res, fitting_range, submod_spec)

        if type(log_plotting_info)==dict:
            # model
            log_plotting_info["count_channels"], log_plotting_info["count_rate_model"] = energy_channels, count_rate_model
            # data
            log_plotting_info["count_rates"], log_plotting_info["count_channel_error"], log_plotting_info["count_rate_errors"] = count_rates, energy_channel_error, count_rate_errors
            # other
            log_plotting_info["residuals"], log_plotting_info["fitting_range"] = residuals, fitting_range
            # do we have submodels
            if hasattr(self, '_corresponding_submod_inputs'):
                log_plotting_info["submodels"] = spec_submods[0]

            if hasattr(self, '_mcmc_mod_runs'):
                log_plotting_info["mcmc_model_runs"] = self._mcmc_mod_runs

        # annotate with parameter names and values
        self._plot_params(axs, res, plot_params, submod_param_cols)

        return axs, res

    def _rebin_input_handler(self, _rebin_input):
        """ Handles any acceptable rebin input.

        Parameters
        ----------
        _rebin_input : list, np.ndarray, dict, int, None
                The rebin input to be converted to the standard dictionary format.

        Returns
        -------
        Returns a rebinning dictionary with keys of spectra IDs and rebin number.
        """

        _default = dict(zip(self.loaded_spec_data.keys(), [None]*len(self.loaded_spec_data.keys())))
        if type(_rebin_input)==int:
            rebin_dict = dict(zip(self.loaded_spec_data.keys(), [_rebin_input]*len(self.loaded_spec_data.keys())))
        elif type(_rebin_input) in (list, np.ndarray):
            rebin_dict = self._if_rebin_input_list(_rebin_input, _default)
        elif type(_rebin_input) is dict:
            rebin_dict = self._if_rebin_input_dict(_rebin_input)
        else:
            if type(_rebin_input)!=type(None):
                print("Rebin input needs to be a single int (applied to all spectra), a list with a rebin entry for each spectrum, or a dict with the spec. identifier and rebin value for any of the loaded spectra.")
            rebin_dict = _default
        rebin_dict["combined"] = rebin_dict["spectrum1"] if "combined" not in rebin_dict else rebin_dict["combined"]

        return rebin_dict

    def _if_rebin_input_list(self, _rebin_input, _default):
        """ Handles rebin input as a list.

        Parameters
        ----------
        _rebin_input : list, np.ndarray
                The rebin input. Need to check if there is a one-to-one match between
                input entries and spectra loaded.

        _default : dict
                A default dictionary for rebinning.

        Returns
        -------
        Returns a rebinning dictionary with keys of spectra IDs and rebin number.
        """
        if len(self.loaded_spec_data.keys())==len(_rebin_input):
            rebin_dict = dict(zip(self.loaded_spec_data.keys(), _rebin_input))
        else:
            print("rebin input list must have an entry for each spectrum; e.g., 3 spectra could be [10,15,None].")
            rebin_dict = _default
        return rebin_dict

    def _if_rebin_input_dict(self, _rebin_input):
        """ Handles rebin input as a list.

        Parameters
        ----------
        _rebin_input : dict
                The rebin input. Keys of spectra IDs and rebin values. If `all` key
                exists then it takes priority over all other keys.

        Returns
        -------
        Returns a rebinning dictionary with keys of spectra IDs and rebin number.
        """
        if "all" in _rebin_input.keys():
            rebin_dict = dict(zip(self.loaded_spec_data.keys(), [_rebin_input["all"]]*len(self.loaded_spec_data.keys())))
        else:
            labels = list(self.loaded_spec_data.keys())+['combined']
            rebin_dict = dict(zip(labels, [None]*(len(self.loaded_spec_data.keys())+1)))
            for k in _rebin_input.keys():
                if k in labels:
                    rebin_dict[k] = _rebin_input[k]
            if rebin_dict['combined'] is None:
                rebin_dict['combined'] = rebin_dict['spectrum1']
        return rebin_dict

    def _plot_from_dict(self, subplot_axes_grid):
        """ Try to build plot from `plotting_info` dict.

        ** Under Construction **

        To be used if no access to a model function.

        Parameters
        ----------
        subplot_axes_grid : list, None
                List of axes objects for the spectra to be plotted on, if None then
                this list will be automatically created.

        Returns
        -------
        A list of axes objects for spectra and a list of axes objects for residuals.
        """
        number_of_plots = len(self.plotting_info)
        subplot_axes_grid = self._build_axes(subplot_axes_grid, number_of_plots)
        axes, res_axes = [], []
        for s, ax in zip(self.plotting_info.keys(), subplot_axes_grid):
            # need this at some point fitting_range = self.plotting_info[s]['fitting_range']
            self.res_ylim = self.res_ylim if hasattr(self, 'res_ylim') else [-7,7]

            ax.set_xlabel('Energy [keV]')
            ax.set_ylabel('Count Spectrum [cts s$^{-1}$ keV$^{-1}$]')

            # axes limits
            energy_channels = self.plotting_info[s]['count_channels']
            if not hasattr(self, 'plot_xlims'):
                extrema_x = LL_CLASS.remove_non_numbers(np.where(count_rates!=0)[0])
                minx, maxx = energy_channels[extrema_x[0]], energy_channels[extrema_x[-1]]
                self.plot_xlims = [0.9*minx, 1.1*maxx]
            ax.set_xlim(self.plot_xlims)

            count_rates = self.plotting_info[s]['count_rates']
            count_rate_model = self.plotting_info[s]['count_rate_model']
            if not hasattr(self, 'plot_ylims'):
                miny = np.min(LL_CLASS.remove_non_numbers(count_rates[count_rates!=0]))
                maxy = np.max([np.max(LL_CLASS.remove_non_numbers(count_rates[count_rates!=0])), np.max(LL_CLASS.remove_non_numbers(count_rate_model[count_rate_model!=0]))])
                self.plot_ylims = [0.9*miny, 1.1*maxy]
            ax.set_ylim(self.plot_ylims)

            ax.set_yscale('log')
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.get_xaxis().set_visible(False)

            energy_channel_error = self.plotting_info[s]['count_channel_error']
            count_rate_errors = self.plotting_info[s]['count_rate_errors']
            ax.errorbar(energy_channels,
                     count_rates,
                     xerr=energy_channel_error,
                     yerr=count_rate_errors,
                     color='k',
                     fmt='.',
                     markersize=0.01,
                     label='Data',
                     alpha=0.8)

            ax.plot(energy_channels, count_rate_model, linewidth=2, color="k")

            energy_channels_res = np.column_stack((energy_channels-energy_channel_error,energy_channels+energy_channel_error)).flatten()
            residuals = self.plotting_info[s]['residuals']
            #residuals plotting
            divider = make_axes_locatable(ax)
            res = divider.append_axes('bottom', 1.2, pad=0.2, sharex=ax)
            res.plot(energy_channels_res, residuals, color='k', alpha=0.8)#, drawstyle='steps-mid'
            res.axhline(0, linestyle=':', color='k')
            res.set_ylim(self.res_ylim)
            # res.set_ylabel('(y$_{Data}$ - y$_{Model}$)/$\sigma_{Error}$')
            res.set_ylabel('($D - M$)/$\sigma$')
            res.set_xlim(self.plot_xlims)
            res.set_xlabel("Energy [keV]")

            axes.append(ax)
            res_axes.append(res)
        return axes, res_axes

    def _build_axes(self, subplot_axes_grid, number_of_plots):
        """ Handles cutsom axes for plotting or creates the axes list.

        Parameters
        ----------
        subplot_axes_grid : list, None
                List of axes objects for the spectra to be plotted on, if None then
                this list will be automatically created.

        number_of_plots : int
                Number of plots needed; i.e., the number of loaded spectra (+1 if
                there is a combined spectra plot).

        Returns
        -------
        A list of axes objects for spectra information.
        """
        if type(subplot_axes_grid)!=type(None):
            assert len(subplot_axes_grid)>=number_of_plots, "Custom list of axes objects needs to be >= number of plots to be created. I.e., >="+str(number_of_plots)+"."
        else:
            # work out rows and columns
            if number_of_plots==0:
                print("No spectra to plot.")
                return
            elif 0<number_of_plots<=4:
                rows, cols = "1", str(number_of_plots)
            elif 4<number_of_plots<=16:
                pos_cols = np.array([3,4])
                mod = number_of_plots%pos_cols
                col_indx = np.where(mod==0)[0]
                if len(col_indx)==1:
                    cols = pos_cols[col_indx][0]
                elif (len(col_indx)==2) and (np.diff(mod)[0]==0):
                    cols = 4
                else:
                    # if both have a remainder then pick the largest one, get a fuller last row
                    cols = pos_cols[np.argmax(mod)]
                rows = maths.ceil(number_of_plots/cols)
            else:
                # 16<number_of_plots always use 4 rows and just build to the right
                rows, cols = 4, maths.ceil(number_of_plots/4)

            # make list of axes
            subplot_axes_grid = []
            for p in range(number_of_plots):
                plot_position = str(int(rows))+str(int(cols))+str(int(p+1))
                subplot_axes_grid.append(plt.subplot(int(plot_position)))
        return subplot_axes_grid

    def _unbin_combin_check_and_rebin_input_format(self, rebin):
        """ Get the correct dictionary rebin format.

        If the data has been rebinned previously then this is undone to check if multiple data can be
        combined. The biinuing to the data is then re-applied at the end of plotting.

        Parameters
        ----------
        rebin : int, list, or dict
                The minimum number of counts in a bin.

        Returns
        -------
        Boolean if the data needs rebinned after plot and the rebin dictionary to do this.
        """
        # rather than having caveats and diff calc for diff spectra, just unbin all to check if the spec can be combined
        _rebin_after_plot = False
        if hasattr(self, "_rebin_setting") and type(rebin)!=type(None):# and (len(self.loaded_spec_data)>1):
            # check if rebinning needs done to at least one spectrum
            # e.g., self._rebin_setting={'spectrum1': 8, 'spectrum2': None} -> yes, self._rebin_setting={'spectrum1': None, 'spectrum2': None} -> no
            _need_rebinning = [0 if type(s)==type(None) else s for s in self._rebin_setting.values()]
            # if len(self._rebin_setting)>0:
            if np.sum(_need_rebinning)>0:
                self._data_rebin_setting = copy(self._rebin_setting)
                self.undo_rebin
                _rebin_after_plot = True
        rebin = self._rebin_input_handler(_rebin_input=rebin) # get this as a dict, {"spectrum1":rebinValue1, "spectrum2":rebinValue2, ..., "combined":rebinValue}
        return _rebin_after_plot, rebin

    def _plot_background(self, axes, spectrum, rebin=None):
        """ Plot the background spectrum if there is one.

        If a background exists in the extras key in the `loaded_spec_data` entry for a given
        spectrum then plot it in grey and behind all other lines. Add some useful annotation to
        the plot.

        Parameters
        ----------
        axes : axes object
                Axes for the data and model.

        spectrum : str
                Spectrum identifier; e.g., "spectrum1".

        rebin : into or None
                If it is a number, rebin the background to the same energy bins as the event time.
                Default: None

        Returns
        -------
        None.
        """
        if 'background_rate' in self.loaded_spec_data[spectrum]['extras']:

            if isnumber(rebin):
                # self._plot_rebin_info defined as [old_bin_width, old_bins, new_bins, new_bin_width] purely for this method
                energies = self._plot_rebin_info[2].flatten()
                _bg_rate_binned = self._bin_model(self.loaded_spec_data[spectrum]['extras']['background_rate'], *self._plot_rebin_info)
                bg_rate = np.concatenate((_bg_rate_binned[:,None],_bg_rate_binned[:,None]),axis=1).flatten()
            else:
                energies = self.loaded_spec_data[spectrum]['count_channel_bins'].flatten()
                bg_rate = np.concatenate((self.loaded_spec_data[spectrum]['extras']['background_rate'][:,None],self.loaded_spec_data[spectrum]['extras']['background_rate'][:,None]),axis=1).flatten()

            axes.plot(energies, bg_rate, color="grey", zorder=0)
            self.plotting_info[spectrum]["background_rate"] = bg_rate

            str_list, c_list = ["BG"], ["grey"]
            # check if the data is actually the event-background
            if self.loaded_spec_data[spectrum]["extras"]["counts=data-bg"]:
                str_list.insert(0, "Counts=Evt-BG\n")
                c_list.insert(0, "k")
            rainbow_text_lines((0.01, 0.99), strings=str_list, colors=c_list, xycoords="axes fraction", verticalalignment="top", horizontalalignment="left", ax=axes, alpha=0.8, fontsize="small")

    def _same_instruments(self, can_combine):
        """ Only combine if all spectra are from the same instrument the now.

        Obviously if the previous checks have revealed they can then False should be return
        regardless of the instruments.

        Parameters
        ----------
        can_combine : bool
                Result from previous checks to see if several spectra can be combined.

        Returns
        -------
        Bool.
        """
        if (len(list(dict.fromkeys(list(self.instruments.values()))))==1) and (can_combine):
            return True
        else:
            return False

    def _get_models(self, number_of_models):
        """ Get the total and sub-models of the fit.

        If there is no attribute `_model`, or it is None, then return [0] as a model
        for all spectra.

        Parameters
        ----------
        number_of_models : int
                Number of models needed if no `_model` or it is None.

        Returns
        -------
        Models or fake models.
        """
        if hasattr(self, "_model") and not (self._model is None):
            self._prepare_submodels() # calculate all submodels if we have them
            return self._calculate_model()
        else:
            self._param_groups = [None]*int(number_of_models)
            return [np.array([1])]*int(number_of_models)

    def plot(self, subplot_axes_grid=None, rebin=None, num_of_samples=100, hex_grid=False, plot_final_result=True):
        """ Plots the latest fit or sampling result.

        Parameters
        ----------
        subplot_axes_grid : list, None
                List of axes objects for the spectra to be plotted on, if None then
                this list will be automatically created.
                Default: None

        rebin : list, np.ndarray, dict, int, None
                The rebin input to be converted to the standard dictionary format. If list, need one-to-one
                match with loaded spectra.
                Default: None

        num_of_samples : int
                Number of random sample entries to use for MCMC model run plotting.
                Default: 100

        hex_grid : bool
                Indicates whether separate model lines should be drawn for MCMC runs or produce hexagonal
                histogram grid for all runs. If True `num_of_samples` is ignored.
                Default: False

        plot_final_result : bool
                Indicates whether the final result should be plotted. Mainly for MCMC runs if only the runs
                are to be plotted without the MAP parameter value models.
                Default: True

        Returns
        -------
        A list of axes objects for spectra information.
        """
        # if the user doesn't want the final result to be shown, perhaps want the MCMC runs to be seen but not the MAP value on top
        self._plr = plot_final_result

        # rebin is ignored if the data has been rebinned
        _rebin_after_plot, rebin = self._unbin_combin_check_and_rebin_input_format(rebin)

        # check if the spectra combined plot can be made
        _channels, _channel_error = [], []
        for s in range(len(self.loaded_spec_data)):
            _channels.append(self.loaded_spec_data['spectrum'+str(s+1)]['count_channel_mids'])
            _channel_error.append(self.loaded_spec_data['spectrum'+str(s+1)]["count_channel_binning"]/2)
        _same_chans = all([np.array_equal(np.array(_channels[0]), np.array(c)) for c in _channels[1:]])
        _same_errs = all([np.array_equal(np.array(_channel_error[0]), np.array(c)) for c in _channel_error[1:]])
        if _same_chans and _same_errs:
            can_combine = True
        else:
            can_combine = False
            print("The energy channels and/or binning are different for at least one fitted spectrum. Not sure how to combine all spectra so won\'t show combined plot.")

        # check for same instruments, no point in combining counts from different instruments?
        can_combine = self._same_instruments(can_combine)

        if hasattr(self,"_model") and (self._model is None) and hasattr(self,"_plotting_info"):
            return self._plot_from_dict(subplot_axes_grid)

        number_of_plots = len(self.loaded_spec_data)+1 if (len(self.loaded_spec_data)>1) and (can_combine) else len(self.loaded_spec_data) # plus one for combined plot
        subplot_axes_grid = self._build_axes(subplot_axes_grid, number_of_plots)

        # reset backgrounds for plotting
        self._include_background(_for_plotting=True)
        self._scaled_backgrounds = self._scaled_background_rates_full

        # only need enough axes for the number of spectra to plot so doesn't matter if more axes are given
        models = self._get_models(number_of_models=number_of_plots)

        axes, res_axes = [], []
        _count_rates, _count_rate_errors = [], []
        self.plotting_info = {}
        for s, ax in zip(range(number_of_plots), subplot_axes_grid):
            if (s<number_of_plots-1) or ((s==number_of_plots-1) and not can_combine) or (number_of_plots==1):
                self.plotting_info['spectrum'+str(s+1)] = {}
                axs, res = self._plot_1spec((self.loaded_spec_data['spectrum'+str(s+1)]['count_channel_mids'],
                                             self.loaded_spec_data['spectrum'+str(s+1)]["count_channel_binning"]/2,
                                             self.loaded_spec_data['spectrum'+str(s+1)]["count_rate"],
                                             self.loaded_spec_data['spectrum'+str(s+1)]["count_rate_error"],
                                             models[s]),
                                            axes=ax,
                                            fitting_range=self.energy_fitting_range['spectrum'+str(s+1)],
                                            plot_params=self._param_groups[s],
                                            submod_spec=str(s+1),
                                            log_plotting_info=self.plotting_info['spectrum'+str(s+1)],
                                            rebin_and_spec=[rebin["spectrum"+str(s+1)], "spectrum"+str(s+1)],
                                            num_of_samples=num_of_samples,
                                            hex_grid=hex_grid)
                _count_rates.append(self.loaded_spec_data['spectrum'+str(s+1)]["count_rate"])
                _count_rate_errors.append(self.loaded_spec_data['spectrum'+str(s+1)]["count_rate_error"])
                axs.set_title('Spectrum '+str(s+1))

                # do we have a background for this spectrum?
                self._plot_background(axes=axs, spectrum='spectrum'+str(s+1), rebin=rebin["spectrum"+str(s+1)])

            else:
                self.plotting_info['combined'] = {}
                axs, res = self._plot_1spec((self.loaded_spec_data['spectrum'+str(s)]['count_channel_mids'],
                                             self.loaded_spec_data['spectrum'+str(s)]["count_channel_binning"]/2,
                                             np.mean(np.array(_count_rates), axis=0),
                                             np.sqrt(np.sum(np.array(_count_rate_errors)**2, axis=0))/len(_count_rate_errors),
                                             np.mean(models, axis=0)),
                                            axes=ax,
                                            fitting_range=self.energy_fitting_range,
                                            submod_spec='combined',
                                            log_plotting_info=self.plotting_info['combined'],
                                            rebin_and_spec=[rebin['combined'], 'combined'],
                                            num_of_samples=num_of_samples,
                                            hex_grid=hex_grid)
                axs.set_ylabel('Count Spectrum [cts s$^{-1}$ keV$^{-1}$ Det$^{-1}$]')
                axs.set_title("Combined Spectra")

            axes.append(axs)
            res_axes.append(res)

        # if the data was unbinned for plotting (to be binned in the plotting methods) then rebin the data here
        if _rebin_after_plot:
            print("Reapply original binning to data. ", end="")
            self.rebin = self._data_rebin_setting
            del self._data_rebin_setting

        return axes, res_axes


    def _prior(self, *args):
        """ Defines parameter priors.

        Takes the bounds from the parameter tables and defines a uniform prior for the
        model params between those bounds.

        Parameters
        ----------
        *args : List of the free parameter bounds.

        Returns
        -------
        Returns 0 if parameter is in the bounds or -np.inf if not.
        """
        for p, b in zip(args, self._free_model_param_bounds):
            lower = b[0] if type(b[0])!=type(None) else -np.inf
            upper = b[1] if type(b[1])!=type(None) else np.inf
            if lower <= p <= upper:
                continue
            else:
                return -np.inf

        return 0.0

    def _model_probability(self,
                           free_params_list,
                           photon_channels,
                           count_channel_mids,
                           srm,
                           livetime,
                           e_binning,
                           observed_counts,
                           observed_count_errors,
                           tied_or_frozen_params_list,
                           param_name_list_order):
        """ Calculates the non-normalised posterior probability for a given set of parameter values.

        I.e., this calculates p(D|M)p(M) from

        .. math::
         p(D|M) = \frac{p(D|M)p(M)}{p(D)}

        where D is the data, M is the model.

        Parameters
        ----------
        *Same list given to fit_stat method*

        free_params_list : 1d array or list of 1d arrays
                The values for all free parameters.

        photon_channels : 2d array or list of 2d arrays
                The photon energy bins for each spectrum (2 entries per bin).

        count_channel_mids : 1d array or list of 1d arrays
                The mid-points of the count energy channels.

        srm : 2d array or list of 2d arrays
                The spectral response matrices for all loaded spectra.

        livetime : list of floats
                List of spectra effective exposures (s.t., count s^-1 * livetime = counts).

        e_binning : 1d array or list of 1d arrays
                The energy binning widths for all loaded spectra.

        observed_counts : 1d array or list of 1d arrays
                The observed counts for all loaded spectra.

        observed_count_errors : 1d array or list of 1d arrays
                Errors for the observed counts for all loaded spectra.

        tied_or_frozen_params_list :
                Values for the parameters that are needed for the fitting but are tied or frozen.

        param_name_list_order : list of strings
                List of all parameter names to match the order of
                [*free_params_list,*tied_or_frozen_params_list].

        Returns
        -------
        Returns the non-normalised posterior probability value for a given set of
        parameter values.
        """
        # calculate prior ln(p(M))
        lp = self._prior(*free_params_list)

        if not np.isfinite(lp):
            return -np.inf

        # now ln(p(D|M)p(M))
        return lp + self._fit_stat_maximize(free_params_list,
                                            photon_channels,
                                            count_channel_mids,
                                            srm,
                                            livetime,
                                            e_binning,
                                            observed_counts,
                                            observed_count_errors,
                                            tied_or_frozen_params_list,
                                            param_name_list_order)

    def _mag_order_spread(self, value, number):
        """ Takes a `value` and calculates a `number` of random values within an order of magnitude.

        Parameters
        ----------
        value : float
                Input value/parameter value.

        number : int
                Number of random points to generate around the `value`.

        Returns
        -------
        Returns a `number` of random values within an order of magnitude of
        `value`.
        """
        # starting points
        # make sure random numbers across rows and columns
        return value + 10**(np.floor(np.log10(value))-1) * np.random.randn(number, len(value))

    def _boundary_spread(self, value_bounds, number):
        """ Takes parameter boundaries (`value_bounds`) and calculates a `number` of random values within that range.

        Parameters
        ----------
        value_bounds : tuple, length 2
                Input value/parameter bounds.

        number : int
                Number of random points to generate around the `value`.

        Returns
        -------
        Returns a `number` of random values within a given range.
        """
        # starting points
        # make sure random numbers across rows and columns

        # sort boundaries into numbers first if there are Nones
        value_bounds = np.array(value_bounds)
        lowers = value_bounds[:, 0][:,None] #= []
        lowers[lowers==None] = -2**32 #-np.inf # for the random int generatation, these have to be numbers. Make it largest 32-bit number
        uppers = value_bounds[:, 1][:,None] #= []
        uppers[uppers==None] = 2**32 #np.inf
        vbounds = np.concatenate((lowers,uppers), axis=1)*100

        return np.array([list(np.random.randint(*vb, number)/100) for vb in vbounds]).T

    def _walker_spread(self, value, value_bounds, number, spread_type="mixed"):
        """ Calculate a spread of walker starting positions.

        Takes parameter values and boundaries and calculates a `number`
        of random values about the value given (spread_type="over_bounds"),
        within the boundary range (spread_type="mag_order"), or half of the
        number about the value with the other half across the boundary
        (spread_type="mixed").

        Parameters
        ----------
        value : float
                Input value/parameter value.

        value_bounds : tuple, length 2
                Input value/parameter bounds.

        number : int
                Number of random points to generate around the `value`.

        spread_type : str
                Defines where the random values are located with respect
                to the given value or value bounds.
                Default: "mixed"

        Returns
        -------
        Returns values surrounding the given value and/or across a given boundary.
        """

        if spread_type=="mag_order":
            spread = self._mag_order_spread(value, number)
        elif spread_type=="over_bounds":
            spread = self._boundary_spread(value_bounds, number)
        elif spread_type=="mixed":
            first_half = int(number/2)
            spread1 = self._mag_order_spread(value, number)[:first_half,:] # half start around the starting value
            spread2 = self._boundary_spread(value_bounds, number)[first_half:,:] # half spread across the boundaries given
            spread = np.concatenate((spread1, spread2))
        else:
            print("spread_type needs to be mag_order, over_bounds, or mixed.")
            return
        return spread

    def _remove_non_number_logprob(self, trials, logprob):
        """ Removes any MCMC trials that produced a np.nan or infinite (will be -np.inf) log-probability.

        Parameters
        ----------
        trials : 2d array
                The trial parameters of an MCMC of shape nxm where n is the
                number of trials and m is the number of free parameters.

        logprob : 1d array
                List of the log-probabilities of all the trials of length n.

        Returns
        -------
        A 2d array of all the filtered trials and 1d array of filtered
        log-probabilities.
        """
        trials = trials[~np.isnan(logprob)]
        logprob = logprob[~np.isnan(logprob)]
        trials = trials[np.isfinite(logprob)]
        logprob = logprob[np.isfinite(logprob)]
        return trials, logprob

    def _combine_samples_and_logProb(self, discard_samples=0):
        """ Makes full sample run with log-probabilities.

        Extracts the parameter trial chains and the corresponding log-probability
        from the MCMC sampler and combines them column-wise.

        Parameters
        ----------
        discard_samples : int
                Number of MCMC samples from the original sample-set to
                discard/burn.
                Default: 0

        Returns
        -------
        A 2d array with columns of each free parameter sampling chain with the last
        column being the log-probability chain.
        """
        flat_samples = self.mcmc_sampler.get_chain(discard=discard_samples, flat=True)
        log_prob_samples = self.mcmc_sampler.get_log_prob(discard=discard_samples, flat=True)

        if not hasattr(self, "_lpc"):
            # keep orignal list of log-probs for plotting
            self._lpc = log_prob_samples

        flat_samples, log_prob_samples = self._remove_non_number_logprob(trials=flat_samples, logprob=log_prob_samples)

        return np.concatenate((flat_samples, log_prob_samples[:,None]), axis=1)

    def _produce_mcmc_table(self, names, relevant_values):
        """ Creates table with MCMC values.

        Produces an astropy table with the MAP MCMC, confidence range, and
        maximum log-probability values from the MCMC samples.

        Parameters
        ----------
        names : list of strings
                List of free parameter names.

        all_mcmc_samples : list of 1d array
                List of [lower_conf_range, MAP, higher_conf_range, max_log_prob]
                for each free parameter from the MCMC samples.

        Returns
        -------
        None.
        """
        a = np.array(relevant_values)
        b = np.array(names)

        # update if we can
        if len(a)!=0:
            # see if we need to make the table or just update the one that's there
            _mcmc_result = [[name, *vals] for name, vals in zip(b, a)] # can't do this with np since all values get changed to str
            if not hasattr(self, "mcmc_table"):
                _value_types = ["LowB", "Mid", "HighB", "MaxLog"]
                self.mcmc_table = Table(rows=_mcmc_result,
                                        names=["Param", *_value_types])#
                for p in _value_types:
                    self.mcmc_table[p].format = "%10.2f"
            else:
                for n, r in zip(names, _mcmc_result):
                    if n in list(self.mcmc_table["Param"]):
                        # update row
                        self.mcmc_table[list(self.mcmc_table["Param"]).index(n)] = r
                    else:
                        # add row
                        self.mcmc_table.add_row(r)

    def _get_mcmc_values(self, all_mcmc_samples, names):
        """ Find useful MCMC values.

        Given the MCMC samples, find the maximum a postiori (MAP value) of the
        sampled posterior distribution, the confidence range values (via
        `confidence_range` setter), and the maximum log-likelihood value found.

        Parameters
        ----------
        all_mcmc_samples : 2d array
                All MCMC samples for the free parameters with the last
                column being the log-probabilities.
        names : list of strings
                List of free parameter names to update.

        Returns
        -------
        List of [lower_conf_range, MAP, higher_conf_range, max_log_prob] for each
        free parameter from the MCMC samples.
        """
        l, m, h = (0.5 - self.error_confidence_range/2)*100, 0.5*100, (0.5 + self.error_confidence_range/2)*100
        quantiles_and_max_prob = []
        self._max_prob = np.max(all_mcmc_samples[:, -1])
        max_prob_index = np.argmax(all_mcmc_samples[:, -1])
        for p in range(len(all_mcmc_samples[0, :])-1):
            # cycle through number of parameters, last one should be the log probability
            qs = np.percentile(all_mcmc_samples[:, p], [l, m, h])
            mlp = all_mcmc_samples[max_prob_index, p]
            qs_mlp = np.append(qs,mlp)
            quantiles_and_max_prob.append(list(qs_mlp))

        self._produce_mcmc_table(names=names, relevant_values=quantiles_and_max_prob)

        return quantiles_and_max_prob

    def _update_free_mcmc(self, updated_free, names, table):
        """ Updates the free parameter values in the given parameter table to the values found by the MCMC run.

        Parameters
        ----------
        updated_free : 2d array
                All MCMC samples for the free parameters with the last
                column being the log-probabilities.

        names : list of strings
                List of free parameter names to update.

        table : parameter_handler.Parameters
                The parameter table to update.

        Returns
        -------
        None.
        """
        quantiles_and_max_prob = self._get_mcmc_values(updated_free, names)
        # only update the free params that were varied
        c = 0
        for key in table.param_name:

            if table["Status", key].startswith("free"):
                # update table
                table["Value", key] = quantiles_and_max_prob[c][1]
                table["Error", key] = (table["Value", key] - quantiles_and_max_prob[c][0],
                                       quantiles_and_max_prob[c][2] - table["Value", key])
                c += 1

    def _fit_stat_maximize(self, *args, **kwargs):
        """ Return the chosen fit statistic defined to maximise for the best fit.

        I.e., returns ln(L).

        Parameters
        ----------
        *args, **kwargs :
                All passed to the fit_stat method.

        Returns
        -------
        Float.
        """
        return self._fit_stat(*args, maximize_or_minimize="maximize", **kwargs)

    def _run_mcmc_setup(self, number_of_walkers=None, walker_spread="mixed"):
        """ Sets up the MCMC run specific variables.

        Parameters
        ----------
        number_of_walkers : int
                The number of walkers to set for the MCMC run.

        walker_spread : str
                Dictates how the walkers are spread out over the parameter
                space with respect to the starting values. If "mag_order"
                then the walkers are spread about an ordser of magnitude
                of the starting values, if "over_bounds" then the walkers
                will be spread randomly over the boundary range, and if
                "mixed" then half will be "mag_order" and half will be
                "over_bounds".
                Default: "mixed"

        Returns
        -------
        List of number of walkers (int), dimensions (int), model probability function (func).
        All info need to produce model and fitting [i.e., photon_channel_bins (list of 2d array),
        count_channel_mids (list of 1d arrays), srm (list of 2d arrays), livetime (list of floats),
        e_binning (list of 1d arrays), observed_counts (list of 1d arrays), observed_count_errors
        (list of 1d arrays), tied_or_frozen_params_list (list of floats), param_name_list_order
        (list of strings)].The starting position of all the walkers (list of floats), and finally
        the number of free parameters (excluding rParams, orig_free_param_len, int).
        """
        free_params_list, stat_args, free_bounds, orig_free_param_len = self._fit_setup()

        self._ndim = len(free_params_list)
        # sort out the number of walkers
        if type(number_of_walkers)==type(None):
            self.nwalkers = 2*self._ndim
        elif type(number_of_walkers)==int:
            if number_of_walkers>=2*self._ndim:
                self.nwalkers = number_of_walkers
            else:
                print("\'self.nwalkers=number_of_walkers\' must be >= 2*number of free parameters (\'self._ndim\').")
                print("Setting \'self.nwalkers\' to \'2*self._ndim\'.")
                self.nwalkers = 2*self._ndim
        else:
            print("\'number_of_walkers\' must be of type \'int\' and >= 2*number of free parameters (\'self._ndim\') or \'None\'.")


        # make sure the random number from normal distribution are of the same order as the earlier solution, or -1 orde, and get a good spread across the boundaries given
        walkers_start = self._walker_spread(free_params_list, free_bounds, self.nwalkers, spread_type=walker_spread)

        return [self.nwalkers, self._ndim, self._model_probability], stat_args, walkers_start, orig_free_param_len

    def _run_mcmc_core(self, mcmc_essentials, prob_args, walkers_start, steps_per_walker=1200, **kwargs):
        """ Passes the information of the MCMC set up to the MCMC sampler.

        Parameters
        ----------
        mcmc_essentials : list
                List of the number of walkers, number of dimensions, and the
                probability function being used.

        prob_args : list
                All the arguments for the `_model_probability()` method.

        walkers_start : 2d array
                Starting positions for the walkers.

        steps_per_walker : int
                The number of steps each walker will take to sample the
                parameter space.

        **kwargs :
                Passed to the MCMC sampler.

        Returns
        -------
        The MCMC sampler object.
        """
        # find the free, tie, and frozen params + other model inputs
        if "pool" in kwargs:
            # for parallelisation
            with kwargs["pool"] as pool:
                kwargs.pop("pool", None)
                mcmc_sampler = emcee.EnsembleSampler(*mcmc_essentials,
                                                     args=prob_args,
                                                     pool=pool,
                                                     **kwargs)
                mcmc_sampler.run_mcmc(walkers_start, steps_per_walker, progress=True);
        else:
            mcmc_sampler = emcee.EnsembleSampler(*mcmc_essentials,
                                                 args=prob_args,
                                                 **kwargs)
            mcmc_sampler.run_mcmc(walkers_start, steps_per_walker, progress=True);

        return mcmc_sampler

    def _update_tables_mcmc(self, orig_free_param_len):
        """ Updates the parameter table with MAP value and confidence range given.

        Parameters
        ----------
        orig_free_param_len : int
                Number of free parameters (excluding rParams).

        Returns
        -------
        None.
        """
        # [params, rParams, lopProb], here want [params, lopProb]
        self._update_free_mcmc(updated_free=np.concatenate((self.all_mcmc_samples[:,:orig_free_param_len], self.all_mcmc_samples[:,-1][:,None]), axis=1), names = self._free_model_param_names, table=self.params) # last one is the logProb
        self._update_tied(self.params)

        # to update the rParams want [rParams, lopProb]
        self._update_free_mcmc(updated_free=self.all_mcmc_samples[:,orig_free_param_len:], names = self._free_rparam_names, table=self.rParams)
        self._update_tied(self.rParams)

    def _run_mcmc_post(self, orig_free_param_len, discard_samples=0):
        """ Handles the results from the MCMC sampling.

        I.e., updates the parameter table and set relevant attributes.

        Parameters
        ----------
        orig_free_param_len : int
                Number of free parameters (excluding rParams).

        discard_samples : int
                Number of MCMC samples to be burned from original samples.
                Default: 0

        Returns
        -------
        A 2d array of the MCMC samples after burning has taken place.
        """
        self.all_mcmc_samples = self._combine_samples_and_logProb(discard_samples=discard_samples)

        # update the model parameters from the mcmc
        self._update_tables_mcmc(orig_free_param_len)

        self._latest_fit_run = "mcmc"#"emcee"

        return self.mcmc_sampler.chain.reshape((-1, self._ndim))

    def _multiprocessing_setup(self, workers):
        """ To return the pool of workers that the MCMC sampler uses to
        run in parallel.

        Parameters
        ----------
        workers : int
                The number of parallel workers that split up the walker's
                steps for the MCMC.

        Returns
        -------
        The pool of workers.
        """
        return Pool(workers)

    def _mcmc_rerun_cleanup(self):
        """ Deletes attributes that should be replaced if the MCMC is run again,
        either a brand new run or appending runs.

        Returns
        -------
        None.
        """
        # if this method is run again then __mcmc_samples__ will need to change (for plotting)
        if hasattr(self, "__mcmc_samples__"):
            del self.__mcmc_samples__
        if hasattr(self, "_lpc"):
            del self._lpc

    def run_mcmc(self,
                 code="emcee",
                 number_of_walkers=None,
                 walker_spread="mixed",
                 steps_per_walker=1200,
                 mp_workers=None,
                 append_runs=False,
                 **kwargs):
        """ Runs MCMC analysis on the data and model provided.

        Parameters
        ----------
        code : str
                Indicates the MCMC sampler being used. Eventually to make
                it easier to give user options.
                Default: "emcee"

        number_of_walkers : int
                The number of walkers to set for the MCMC run. Set to 2*`_ndim`
                if None is given.
                Default: None

        walker_spread : str
                Dictates how the walkers are spread out over the parameter
                space with respect to the starting values. If "mag_order"
                then the walkers are spread about an ordser of magnitude
                of the starting values, if "over_bounds" then the walkers
                will be spread randomly over the boundary range, and if
                "mixed" then half will be "mag_order" and half will be
                "over_bounds".
                Default: "mixed"

        steps_per_walker : int
                The number of steps each walker will take to sample the
                parameter space.
                Default: 1200

        mp_workers : int or None
                The number of parallel workers that split up the walker's
                steps for the MCMC.
                Default: None

        append_runs : bool
                Set to False to run new chains, set to True to start where the
                last run ended and append the runs.
                Default: False

        **kwargs :
                Passed to the MCMC sampler.

                Could pass `backend` object from `emcee` to save chains as they are
                running to a HDF5 file[1].
                If the `backend` kwarg is given it takes priority over `append_runs`.

                The `pool` arg is overwritten if `mp_workers` is provided.

        [1] https://emcee.readthedocs.io/en/stable/tutorials/monitor/

        Returns
        -------
        A 2d array of the MCMC samples after burning has taken place (output of
        `_run_mcmc_post()` method with 0 burned samples).
        """

        mcmc_setups, probability_args, walker_pos, orig_free_param_len = self._run_mcmc_setup(number_of_walkers=number_of_walkers,
                                                                                              walker_spread=walker_spread)

        if type(mp_workers)!=type(None):
            self._pickle_reason = "mcmc_parallelize"
            kwargs["pool"] = self._multiprocessing_setup(workers=mp_workers)

        # if user wants to append runs, an explicit sampler backend isn't given, and an MCMC run already exists then (start/append) new run (at the end of/to) previous
        if append_runs and ('backend' not in kwargs) and hasattr(self, 'mcmc_sampler'):
            kwargs['backend'], walker_pos = self.mcmc_sampler.backend, None

        # if this method is run again then some attrs need to be reset
        self._mcmc_rerun_cleanup()

        self.mcmc_sampler = self._run_mcmc_core(mcmc_setups, probability_args, walker_pos, steps_per_walker=steps_per_walker, **kwargs)

        self._pickle_reason = "normal"

        self._fpl = orig_free_param_len

        return self._run_mcmc_post(orig_free_param_len, discard_samples=0)

    @property
    def burn_mcmc(self):
        """ ***Property*** Allows the number of discarded samples to be set.

        Returns the number of discarded samples (`_discard_sample_number`).

        Returns
        -------
        Int.
        """
        if not hasattr(self, "_discard_sample_number"):
            self._discard_sample_number = 0
        return self._discard_sample_number

    @burn_mcmc.setter
    def burn_mcmc(self, burn):
        """ ***Property Setter*** Allows the number of discarded samples to be set.

        The samples are always discarded from the original sampler.

        Parameters
        ----------
        burn : int (>0)
                The burn-in for the MCMC chains.

        Returns
        -------
        None.

        Example
        -------
        # always discards from original sampler so to have a burn-in of 12
        s = LoadSpec(pha_file='filename1.pha',arf_file='filename1.arf',rmf_file='filename1.rmf')
        s.run_mcmc()
        s.burn_mcmc = 12

        # if the user then realises that they wanted have a burn-in of 35 then,
        s.burn_mcmc = 35

        # Can always undo the burn-in and return to every sample with,
        s.burn_mcmc = 0
        # or <equivalent>
        s.undo_burn_mcmc
        """

        self._discard_sample_number = int(burn)
        self.all_mcmc_samples = self._combine_samples_and_logProb(discard_samples=self._discard_sample_number)

        _ = self._run_mcmc_post(orig_free_param_len=self._fpl, discard_samples=self._discard_sample_number) # burns more self.all_mcmc_samples and updates the param tables

    @property
    def undo_burn_mcmc(self):
        """ ***Property*** Undoes any burn-in applied to the mcmc chains.

        Returns `all_mcmc_samples` back to every chain calculated.

        Returns
        -------
        None.
        """
        self.burn_mcmc = 0

    def plot_log_prob_chain(self):
        """ Produces a plot of the log-probability chain from all MCMC samples.

        The number of burned sampled, if any, is indicated with a shaded region.

        *** Note: Will update to include all parameter chains ***

        Returns
        -------
        Returns axis object of the axis that dictates the burned sample number.
        """
        # plots the original chain length with a shaded region indicating the burned samples
        ax = plt.gca()
        ax.plot(self._lpc)
        ax.set_xlim([0, len(self._lpc)])
        miny, maxy = np.min(self._lpc[np.isfinite(self._lpc[~np.isnan(self._lpc)])]), np.max(self._lpc[np.isfinite(self._lpc[~np.isnan(self._lpc)])])
        miny, maxy = miny-0.1*abs(miny), maxy+0.1*abs(maxy)
        ax.set_ylim([miny, maxy])
        ax_color = "dimgrey"
        ax.set_xlabel("chain [steps*walkers]", color=ax_color)
        ax.set_ylabel(self.loglikelihood, color=ax_color)
        ax.tick_params(axis='both', color=ax_color)

        ax2 = ax.twiny()
        ax2_color = "navy"
        ax2.set_xticks((np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1])*len(self._lpc)/self.nwalkers).astype(int))
        ax2.set_xlim([0, len(self._lpc)/self.nwalkers])
        ax2.set_ylim([miny, maxy])
        ax2.tick_params(axis='x', color=ax2_color)
        ax2.set_xlabel("chain [steps]\n(this dictates burned number)", color=ax2_color)

        already_discarded = self._discard_sample_number if hasattr(self, "_discard_sample_number") else 0

        if already_discarded>0:
            fill_color = "peru"
            ax2.fill_between([0, already_discarded], [miny, miny], [0,0], color=fill_color, alpha=0.1)#[maxy, maxy]
            ax2.axvline(x=already_discarded, color=fill_color, linestyle=':')
            ax.annotate("Burned", (0.05, 0.05), xycoords="axes fraction", color=fill_color)

        return ax2

    def _fix_corner_plot_titles(self, axes, titles, quantiles):
        """ Method to create corner plot titles.

        Defined by user quantiles instead of corner.py's default quantiles of [0.16, 0.5, 0.84].

        Parameters
        ----------
        axes : array of axes objects
                The array of axes from the corner plot.

        titles : list of strings
                List of the parameters for the titles.

        quantiles : list of floats
                List of the quantiles from the confidence range.

        Returns
        -------
        None.
        """
        for c,(t,s) in enumerate(zip(titles, self.all_mcmc_samples.T)):
            qs = np.percentile(s, np.array(quantiles)*100)
            qs_ext = np.diff(qs)
            title = t+" = {0:.1e}".format(qs[1])+"$^{{+{0:.1e}}}_{{-{1:.1e}}}$".format(qs_ext[-1], qs_ext[0])
            axes[c,c].set_title(title)

    def corner_mcmc(self, _fix_titles=True, **kwargs):
        """ Produces a corner plot of the MCMC run.

        Parameters
        ----------
        _fix_titles : True
                True to change the corner plot titles to the ones dictated by
                `self.error_confidence_range`.
                Default: True

        **kwargs :
                Passed to `corner.corner`.

        Returns
        -------
        Returns axis object of the corner plot.
        """
        # check there are MCMC samples to plot
        if not hasattr(self, "all_mcmc_samples"):#"mcmc_sampler"):
            print("The MCMC analysis has not been run yet. Please run run_mcmc(...) successfully first.")
            return

        cr = self.error_confidence_range
        quants = [0.5 - cr/2, 0.5, 0.5 + cr/2]

        kwargs["labels"] = self._free_model_param_names+self._free_rparam_names+["logProb"] if "labels" not in kwargs else kwargs["labels"]
        kwargs["levels"] = [cr] if "levels" not in kwargs else kwargs["levels"]
        kwargs["show_titles"] = False if "show_titles" not in kwargs else kwargs["show_titles"]
        kwargs["title_fmt"] = '.1e' if "title_fmt" not in kwargs else kwargs["title_fmt"]
        kwargs["quantiles"] = quants if "quantiles" not in kwargs else kwargs["quantiles"]

        # for some reason matplotlib contour.py can change a single value contour list into
        #    a list containing two of the same value (e.g., levels=[0.6] -> levels=[516, 516]).
        #    This, of course, throws a ValueError since levels need to be constantly increasing.
        #    This does not happen when >1 value in the levels are given.
        kwargs["levels"] = [0, *kwargs["levels"]]

        figure = corner.corner(self.all_mcmc_samples, **kwargs)

        # Extract the axes
        axes = np.array(figure.axes).reshape((len(kwargs["labels"]), len(kwargs["labels"])))

        if _fix_titles:
            self._fix_corner_plot_titles(axes, kwargs["labels"], quants)

        return axes

    def _prior_transform_nestle(self, *args):
        """ Creates the prior function used when running the nested sampling code.


        I.e., input will be a list of parameter values mapped to the unit hypercube
        with the output mapping them back to their corresponding value in the user
        defined bounds.

        Parameters
        ----------
        **args : floats
                Unit hypercube values for each free parameter.

        Returns
        -------
        The parameter unit hypercube value mapped back to its corresponding value
        in the user defined bounds.
        """

        bounds = np.array(self._free_model_param_bounds)

        # np.squeeze to avoid lists in lists
        return np.squeeze(np.array(args)*np.squeeze(np.diff(bounds)) + bounds[:,0])

    def run_nested(self,
                   code="nestle",
                   nlive = 10,
                   method = 'multi',
                   tol = 10,
                   **kwargs):
        """ Runs nested sampling on the data and provided model.

        [1] http://mattpitkin.github.io/samplers-demo/

        Parameters
        ----------
        code : str
                Indicates the nested sampling sampler being used. Eventually
                to make it easier to give user options.
                Default: "nestle"

        nlive : int
                Number of live points (default should be higher, e.g., 1024).
                Default: 10

        method : str
                Method for the nested sampling.
                Default: 'multi' (MutliNest algorithm)

        tol : float
                Stopping criterion.
                Default: 10

        **kwargs :
                Passed to the nested sampling method.

        Returns
        -------
        None.
        """

        mcmc_setups, probability_args, _, _ = self._run_mcmc_setup()

        ndims = mcmc_setups[1]       # number of parameters

        self.nestle = nestle.sample((lambda free_args: self._fit_stat_maximize(free_args, *probability_args)),
                                    self._prior_transform_nestle,
                                    ndims,
                                    method=method,
                                    npoints=nlive,
                                    dlogz=tol,
                                    **kwargs)

    def save(self, filename):
        """ Pickles data from the object in a way it can be loaded back in later.

        Parameters
        ----------
        filename : str
                The filename for the pickle file to be created. Should end in pickle
                but if not then it is added.

        Returns
        -------
        None.
        """
        filename = filename if filename.endswith("pickle") else filename+".pickle"
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __getstate__(self):
        """Tells pickle how this object should be pickled."""
        _model = {"_model":self._model.__name__}
        _user_fncs = {"usr_funcs":DYNAMIC_FUNCTION_SOURCE}
        _user_args = {"user_args":DYNAMIC_VARS}
        if self._pickle_reason=="mcmc_parallelize":
            _loaded_spec_data = {"loaded_spec_data":{s:{k:d for (k,d) in v.items() if k!="extras"} for (s,v) in self.loaded_spec_data.items()}} # don't need anything in "extras"
            _atts = {"params":self.params,
                     "rParams":self.rParams,
                     "loglikelihood":self.loglikelihood,
                     "_param_groups":self._param_groups,
                     "_free_model_param_bounds":self._free_model_param_bounds,
                     "_orig_params":self._orig_params}
            return {**_loaded_spec_data, **_model, **_atts, **_user_fncs, **_user_args}
        else:
            dict_copy = self.__dict__.copy()

            # delete attributes that rely on non-picklable objects (dynamic functions)
            if hasattr(self, '_model'):
                del dict_copy['_model']
            if hasattr(self, '_submod_functions'):
                del dict_copy['_submod_functions']
            if hasattr(self, 'all_models'):
                for mod in self.all_models.keys():
                    self.all_models[mod]["function"] = inspect.getsource(self.all_models[mod]["function"])

            # _model is a function in dict_copy (likely not picklable) but the **_model dict will replace this in dict_copy
            return {**dict_copy, **_model, **_user_fncs, **_user_args}

    def __setstate__(self, d):
        """Tells pickle how this object should be loaded."""
        add_var(**d["user_args"], quiet=True)
        for f,c in d["usr_funcs"].items():
            function_creator(function_name=f, function_text=c)
        del d["usr_funcs"], d["user_args"]
        self.__dict__ = d
        self._model = globals()[d["_model"]] if d["_model"] in globals() else None

    def __repr__(self):
        """Provide a representation to construct the class from scratch."""
        return self._construction_string_sunxspex

    def __str__(self):
        """Provide a printable, user friendly representation of what the class contains."""
        _loaded_spec = ""
        plural = ["Spectrum", "is"] if len(self.loaded_spec_data.keys())==1 else ["Spectra", "are"]
        tag = f"{plural[0]} Loaded {plural[1]}: "
        for s in self.loaded_spec_data.keys():
            _loaded_spec += str(self.loaded_spec_data[s]["extras"]["pha.file"])+"\n"+" "*len(tag)

        _loaded_spec += "\rLikelihood: "+str(self.loglikelihood)
        _loaded_spec += "\nModel: "+str(self._model)
        _loaded_spec += "\nModel Parameters: "+str(self.params.param_name)
        _loaded_spec += "\nModel Parameter Values: "+str(self.params.param_value)
        _loaded_spec += "\nModel Parameter Bounds: "+str(self.params.param_bounds)
        _loaded_spec += "\nFitting Range(s): "+str(self.energy_fitting_range)

        return f"No. of Spectra Loaded: {len(self.loaded_spec_data.keys())} \n{tag}{_loaded_spec}"

def load(filename):
    """ Loads in a saved instance of the SunXspex class.

    Parameters
    ----------
    filename : str
            Filename for the pickled fitting class.

    Returns
    -------
    Loaded in fitting class.
    """
    with open(filename, 'rb') as f:
        loaded = pickle.load(f)
    return loaded

# The following functions allows SunXspex.model take lambda functions and strings as inputs then convert them to named functions

def _func_self_contained_check(function_name, function_text):
    """ Checks that the function can be run only from its source code in any directory.

    Takes a user defined function name for a NAMED function and a string that
    produces the NAMED function and executes the string as code. This checks to
    make sure the function is completely self-contatined; i.e., able to be
    reconstructed from source to allow for smooth pickling and loading into a new
    environment to the one the original function was defined in.

    If an exception occurs here then the user is informed with a warning and
    (hopefully) a helpful message.

    The test inputs to check a function works are 1s for all arguments and
    energy bins of np.array([[1.60,1.64],[1.64,1.68],...,[4.96,5.00]]).

    Parameters
    ----------
    function_name : str
            The name of the function.

    function_text : str
            Code for the function as a string.

    Returns
    -------
    None.
    """
    exec(function_text, globals())
    params, _ = get_func_inputs(globals()[function_name])
    _test_e_range = np.arange(1.6,5.01, 0.04)[:,None]
    _test_params, _test_energies = np.ones(len(params)), np.concatenate((_test_e_range[:-1], _test_e_range[1:]), axis=1) # one 5 for each param, 2 column array of e-bins
    try:
        _func_to_test = globals()[function_name]
        del globals()[function_name] # this is a check function, don't want it just adding things to globals
        _func_to_test(*_test_params, energies=_test_energies)
    except NameError as e:
        raise NameError(str(e)+f"\nA user defined function should be completely self-contained and should be able to be created from its source code,\nrelying entirely on its local scope. E.g., modules not imported in the fitter module need to be imported in the fuction (fitter imported modules:\n{imports()}).")
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e)+"\nPlease check that absolute file/directory paths are given for the user defined function to be completely self-contained.")
    # except ValueError as e:
    #     warnings.warn(str(e)+"\nFunction failed self-contained check; however, this may be due to conflict in test inputs used to the model.")

def function_creator(function_name, function_text, _orig_func=None):
    """ Creates a named function from its name and source code.

    Takes a user defined function name for a NAMED function and a string that
    produces the NAMED function and executes the string as code. Replicates the
    user coding thier function in this module directly.

    Parameters
    ----------
    function_name : str
            The name of the function.

    function_text : str
            Code for the function as a string.

    _orig_func : function or None
            The original function provided to be broken down and recreated.
            Default: None

    Returns
    -------
    Returns the function that has just been created and executed into globals().
    """
    DYNAMIC_FUNCTION_SOURCE[function_name] = function_text
    try:
        _func_self_contained_check(function_name, function_text)
        # given the code for a NAMED function (not lambda) as a string, this will execute the code and return that function
        exec(function_text, globals())
        return globals()[function_name]
    except (NameError, FileNotFoundError) as e:
        warnings.warn(str(e) + "\nHi there! it looks like your model is not self-contained. This will mean that any method that includes\npickling (save, load, parallelisation, etc.) will not act as expected since if the model is loaded\ninto another session the namespace will not be the same.")
        return _orig_func

def deconstruct_lambda(function, add_underscore=True):
    """ Takes in a lambda function and returns it as a NAMED function

    Parameters
    ----------
    function : function
            The lambda function with function.__name__=="<lambda>".

    add_underscore : bool
            Add the underscore the start of the lambda function name. E.g., a=lambda x:x -> func.__name__=_a
            Default: True

    Returns
    -------
    Returns the NAMED function created from the lambda function provided.
    """
    # put in a lambda function and convert it to a named function and return it
    if function.__name__=="<lambda>":
        # get the text of the function being passed
        lambda_source_code = inspect.getsource(function)

        # split up the text using the general form of a lambda function
        regex = r'(.*)=(.*)lambda(.*):(.*)'
        x = re.search(regex, lambda_source_code)

        # get function name from the first (.*) in regex, remove spaces, and if there is a period replace with _
        fun_name = np.array(x.group(1).split(" "))[np.array(x.group(1).split(" "))!=""][0].replace(".", "") # incase set like self.model=lambda x:x then name would be self.model, now selfmodel
        input_params = np.array(x.group(3).split(" "))[np.array(x.group(3).split(" "))!=""]

        # build up the string for a named function, e.g., def f(x): \n return x
        _underscore = "_" if add_underscore else ""
        def_line = "".join(["def "+_underscore+fun_name, "(", *input_params, "):\n"]) # change function_name to _function_name so the original isn't overwritten
        return_line = " ".join(["    return ", *np.array(x.group(4).split(" "))[np.array(x.group(4).split(" "))!=""], "\n"])
        func_info = {"function_name":_underscore+fun_name, "function_text":"".join([def_line, return_line])}
    else:
        func_info = {"function_name":function.__name__, "function_text":inspect.getsource(function)}

    return function_creator(**func_info, _orig_func=function)#execute the function to be used here

def get_all_words(model_string):
    """ Find any groups of non-maths characters in a string.

    Parameters
    ----------
    model_string : string
            The string defining the model.

    Returns
    -------
    The groups of non-maths characters.
    """
    # https://stackoverflow.com/questions/44256638/python-regular-expression-to-find-letters-and-numbers
    regex4words = r"(?<![\"=\w])(?:[^\W]+)(?![\"=\w])"
    return re.findall(regex4words, model_string)

def get_nonsubmodel_params(model_string, _defined_photon_models):
    """ From the non-maths character groups in a string, remove the ones that refer to any defined model name.

    The others should be other parameters for the fit. E.g., "C*f_vth" means this function
    will pick out the "C" as a fitting parameter.

    Parameters
    ----------
    model_string : string
            The string defining the model.

    _defined_photon_models : dict
            Dictionary (should be from photon_models_for_fitting module) where the keys are the already
            defined functions with values representing that functions parameters.

    Returns
    -------
    Return parameters that are not inputs to the functions already defined in _defined_photon_models.
    """
    # return other parameters outside the defined models from the string as long as they're valid Python identifiers
    all_words = get_all_words(model_string)
    nonsubmodel_params = list(set(all_words) - set(_defined_photon_models.keys()))
    return [n for n in nonsubmodel_params if ((n.isidentifier()) and (not iskeyword(n)))]

def check_allowed_names(model_string):
    """ Check if the model string's non-maths characters are all viable Python identifiers.

    Parameters
    ----------
    model_string : string
            The string defining the model.

    Returns
    -------
    Boolean.
    """
    # check everything outside a mathematical expression and constants are all valid Python identifiers, if not print the ones that need to be changed
    all_words = get_all_words(model_string)
    all_allowed_words = set([n for n in all_words if ((n.isidentifier()) and (not iskeyword(n))) or (isnumber(n))])
    words_are_allowed = (set(all_words)==all_allowed_words)
    if words_are_allowed:
        return True
    else:
        print(set(all_words) - set([n for n in all_words if ((n.isidentifier()) and (not iskeyword(n))) or (isnumber(n))]))
        return False

def get_func_inputs(function):
    """ Get the inputs to a given function.

    Parameters
    ----------
    function : function
            The function to be inspected.

    Returns
    -------
    A list of the positional arguments and a list with all other arguments.
    """
    param_inputs = []
    other_inputs = {}
    for param, actual_input in inspect.signature(function).parameters.items():
        # for def fn(step=3):..., param would be step but actual_input would be step=3
        if (actual_input.kind==actual_input.POSITIONAL_OR_KEYWORD) and (actual_input.default is actual_input.empty):
            # variable param (make all free to begin with)
#                             self.model_param_names[param+"_spectrum"+str(s+1)] = "free"
            param_inputs.append(param) # self._model_param_names, pg, self._orig_params
        else:
            # then fixed arguments
            other_inputs[param] = [actual_input.kind,actual_input.default]# self._other_model_inputs
    return param_inputs, other_inputs

def imports():
    """ Lists the imports from other modules into this one.

    This is usedful when defining user made functions since these modules can
    be used in them normally. If a package the user uses is not in this list
    then they must include it in the defined model to make the model function
    self-contained.

    Parameters
    ----------

    Returns
    -------
    A list of the positional arguments and a list with all other arguments.
    """
    _imps = ""
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # check for modules and their names being used to refer to them
            _imps += "* import "+val.__name__ +" as "+ name + "\n"
        elif (isinstance(val, (types.FunctionType, types.BuiltinFunctionType)) and not isinstance(inspect.getmodule(val), type(None))) and (inspect.getmodule(val).__name__ not in ("__main__", __name__)):
            # check for functions and their names being used to refer to them
            _imps += "* from "+str(inspect.getmodule(val).__name__)+" import "+val.__name__+" as "+name + "\n"
    return _imps


def make_model(energies=None, photon_model=None, parameters=None, srm=None):
    """ Takes a photon model array ( or function if you provide the pinputs with parameters), the spectral response matrix and returns a model count spectrum.

    Parameters
    ----------
    energies : array/list
            List of energies.
            Default : None

    photon_model : function/array/list
            Array -OR- function representing the photon model (if it's a function, provide the parameters of the function as a list, e.g. paramters = [energies, const, power]).
            Default : None

    parameters : list
            List representing the inputs a photon model function, if a function is provided, excludeing the energies the spectrum is over.
            Default : None

    srm : matrix/array
            Spectral response matrix.
            Default : None

    Returns
    -------
    A model count spectrum.
    """

    ## if parameters is None then assume the photon_model input is already a spectrum to test, else make the model spectrum from the funciton and parameters
    if type(parameters) == type(None):
        photon_spec = photon_model
    else:
        photon_spec = photon_model(energies, *parameters)

    model_cts_spectrum = np.matmul(photon_spec, srm)

    return model_cts_spectrum
