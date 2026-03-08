"""
This module contains functions to carry out astropy fitting with spectral models
"""

import astropy.units as u
from astropy.modeling import fitting
from astropy.modeling import models
from matplotlib import pyplot as plt

import numpy as np

from sunkit_spex.models.physical.thermal import ThermalEmission
from sunkit_spex.models.physical.nonthermal import ThickTarget
from sunkit_spex.models.physical.albedo import Albedo
from sunkit_spex.models.scaling import InverseSquareFluxScaling
from sunkit_spex.models.instrument_response import MatrixModel

__all__ = ["fitter"]


class Fitter:

    def __init__(
            self,
            model,
            spectrum_object,
            fitting_method = fitting.TRFLSQFitter(calc_uncertainties=True),
            fit_range=None):

        self._model = model
        self._spectrum_object = spectrum_object
        self._fitting_method = fitting_method
        self._fit_range = fit_range
        self._fitted_model = None
        # self._PIPELINE_COMPONENTS = {'SRM', 'Albedo', 'InverseSquareFluxScaling'}

    @property
    def model(self):
        return self._model
    

            
    
    def _set_abledo_angle(self):

        if 'Albedo' in self.model.submodel_names:

            print(len(self._spectrum_object.meta['ph_axis']))

            replacement_albedo = Albedo(energy_edges=self._spectrum_object.meta['ph_axis'],
                                                                       theta=self._spectrum_object.meta['angle'])
            replacement_albedo.theta.fixed = True

            self._model = self._model.replace_submodel('Albedo',replacement_albedo)    

    def _set_observer_distance(self):

        match = np.where(np.array(self._model.submodel_names)=='InverseSquareFluxScaling')[0]
        
        if np.shape(match) != 0:
            param_names = [f'observer_distance_{str(ind)}' for ind in match]

            for param_name in param_names:
                setattr(self._model, param_name, self._spectrum_object.meta['distance'])
                getattr(self._model, param_name).fixed = True        

    def _set_srm(self):
        
        if 'SRM' in self.model.submodel_names:
            
            self._model = self._model.replace_submodel('SRM',MatrixModel(matrix= np.array(self._spectrum_object.meta['srm']),   
                                                                         spectrum_object=self._spectrum_object, 
                                                                         model_spec_units=u.ph * u.keV**-1 * u.s**-1 * u.cm**-2))            
    @property
    def fitting_method(self):
        return self._fitting_method

    @fitting_method.setter
    def fitting_method(self, value):
        self._fitting_method = value

    @property
    def fitted_model(self):
        """Return the fitted model. None until do_fit() has been called."""
        if self._fitted_model is None:
            raise RuntimeError("No fitted model available — call do_fit() first.")
        return self._fitted_model

    @property
    def fit_range(self):
        return self._fit_range


    @fit_range.setter
    def fit_range(self, value):
        """
        value : tuple
            (emin, emax) in same units as spectral_axis
        """
        
        if value is None:
            self._fit_range = None
            return

        emin, emax = value
        edges = self._spectrum_object.spectral_axis.bin_edges

        # Determine bins fully inside range
        lower = edges[:-1]
        upper = edges[1:]

        indices = np.where((lower >= emin) & (upper <= emax))[0]        

        self._fit_range = value
        self._fit_mask = indices

    def _apply_fit_range(self):

        if self._fit_range is None:
            return

        mask = self._fit_mask

        self._spectrum_object = self._spectrum_object[mask[0]:mask[-1]+1]

        print(self._spectrum_object.spectral_axis.bin_edges.shape)

        self._spectrum_object.spectral_axis._bin_edges = np.array(self._spectrum_object.spectral_axis.bin_edges[mask[0]:mask[-1]+2])


        print(self._spectrum_object.spectral_axis.bin_edges.shape)

        if 'srm' in self._spectrum_object.meta:
            self._spectrum_object.meta['srm'] = \
                self._spectrum_object.meta['srm'][:,mask[0]:mask[-1]+1]


    def _fit_prep(self):

        self._apply_fit_range()

        self._set_abledo_angle()
        self._set_observer_distance()
        self._set_srm()


    
    def do_fit(self):
        

        self._fit_prep()


        w =  np.array(1/self._spectrum_object.uncertainty.array) << self._spectrum_object.uncertainty.unit
        data = np.array(self._spectrum_object.data) << self._spectrum_object.unit


        # Store on the instance; access via the fitted_model property
        self._fitted_model = self._fitting_method(
            model=self._model,
            x=self._spectrum_object.meta['ph_axis'],
            y=data,
            weights=w,
            estimate_jacobian=True)        
        
        # return fitted_model

        
    # def _decompose_model(self, model):
    #     """Walk the fitted CompoundModel in post-order and separate leaf
    #     submodels into source components and pipeline components.

    #     traverse_postorder() visits leaves before their parent operator
    #     nodes, so we see every primitive submodel exactly once before any
    #     intermediate CompoundModel node that combines them. We skip those
    #     intermediate nodes by checking for the presence of `submodel_names`
    #     (only CompoundModel has that attribute).

    #     Returns
    #     -------
    #     source_names : list[str]
    #         Leaf model names that are genuine emission sources, in the
    #         left-to-right post-order they appear in the expression tree.
    #     pipeline : dict[str, Model]
    #         Mapping of pipeline-role names ('SRM', 'Albedo',
    #         'InverseSquareFluxScaling') to their fitted submodel instances.
    #     """
    #     source_names = []
    #     pipeline = {}
    #     seen = set()

    #     for m in model.traverse_postorder():
    #         # Skip intermediate CompoundModel operator nodes —
    #         # they carry combined state, not individual components.
    #         if hasattr(m, 'submodel_names'):
    #             continue

    #         name = m.name
    #         if name in seen:
    #             continue
    #         seen.add(name)

    #         if name in self._PIPELINE_COMPONENTS:
    #             pipeline[name] = model[name]   # use fitted instance
    #         else:
    #             source_names.append(name)

    #     return source_names, pipeline

    # def _build_source_pipeline(self, model, source_name, pipeline):
    #     """Reconstruct the forward pipeline for a single source component,
    #     applying scaling and SRM but *not* Albedo.

    #     The chain is:  source  [* InverseSquareFluxScaling]  [| SRM]
    #     """
    #     component = model[source_name]

    #     if 'InverseSquareFluxScaling' in pipeline:
    #         chain = component * pipeline['InverseSquareFluxScaling']
    #     else:
    #         chain = component

    #     if 'SRM' in pipeline:
    #         chain = chain | pipeline['SRM']

    #     return chain

    # def _build_full_source_sum_pipeline(self, model, source_names, pipeline,
    #                                     include_albedo=False):
    #     """Sum all source components, then thread through the pipeline.

    #     Parameters
    #     ----------
    #     include_albedo : bool
    #         When True the Albedo kernel is inserted before the SRM.
    #     """
    #     # Generic sum of all source leaf models
    #     total_source = model[source_names[0]]
    #     for name in source_names[1:]:
    #         total_source = total_source + model[name]

    #     if 'InverseSquareFluxScaling' in pipeline:
    #         chain = total_source * pipeline['InverseSquareFluxScaling']
    #     else:
    #         chain = total_source

    #     if include_albedo and 'Albedo' in pipeline:
    #         chain = chain | pipeline['Albedo']

    #     if 'SRM' in pipeline:
    #         chain = chain | pipeline['SRM']

    #     return chain

    # # Map bare astropy unit strings → LaTeX strings shown on the plot.
    # _UNIT_LATEX = {
    #     'MK'                          : r'MK',
    #     'keV'                         : r'keV',
    #     '1e49 cm-3'                   : r'$\times\,10^{49}$~cm$^{-3}$',
    #     '1e49 / cm3'                  : r'$\times\,10^{49}$~cm$^{-3}$',
    #     '10^49 cm^-3'                 : r'$\times\,10^{49}$~cm$^{-3}$',
    #     '1e35 electron / s'           : r'$\times\,10^{35}$~e~s$^{-1}$',
    #     '1e35 1 / s'                  : r'$\times\,10^{35}$~e~s$^{-1}$',
    #     'electron / (cm2 keV s)'      : r'e~cm$^{-2}$~keV$^{-1}$~s$^{-1}$',
    #     'ph / (cm2 keV s)'            : r'ph~cm$^{-2}$~keV$^{-1}$~s$^{-1}$',
    #     'AU'                          : r'AU',
    #     'deg'                         : r'deg',
    #     ''                            : r'',
    # }

    # # Map raw parameter names → (LaTeX display name, decimal places)
    # _PARAM_LATEX = {
    #     'temperature'     : (r'$T$',              1),
    #     'emission_measure': (r'$\mathrm{EM}$',    3),
    #     'p'               : (r'$\delta$',          2),
    #     'low_e_cutoff'    : (r'$E_c$',             1),
    #     'total_eflux'     : (r'Electron Flux',     2),
    #     'norm'            : (r'Norm',              3),
    #     'index'           : (r'$\Gamma$',          2),
    #     'e_break'         : (r'$E_{\rm break}$',   1),
    #     'observer_distance': (r'$d$',              3),
    # }

    # def _format_param_label(self, param_name, param):
    #     """Return a nicely LaTeX-formatted annotation string for one parameter.

    #     Falls back gracefully if the parameter name or unit is not in the
    #     lookup tables.
    #     """
    #     # Strip any trailing index added by CompoundModel (e.g. 'temperature_0')
    #     base_name = '_'.join(param_name.split('_')[:-1]) \
    #         if param_name[-1].isdigit() else param_name

    #     display_name, decimals = self._PARAM_LATEX.get(
    #         base_name, (param_name.replace('_', r'\_'), 3))

    #     value_str = f'{param.value:.{decimals}f}'

    #     # Resolve unit
    #     unit_raw  = str(param.unit) if param.unit else ''
    #     unit_latex = self._UNIT_LATEX.get(unit_raw, unit_raw)

    #     if unit_latex:
    #         return rf'{display_name} $=$ {value_str} {unit_latex}'
    #     return rf'{display_name} $=$ {value_str}'

    # def plot_fit_results(self, save_name, fit_times):
    #     """Plot the fitted spectrum, individual model components, the albedo
    #     contribution, and a delta-chi residual panel.

    #     The compound model is deconstructed generically via
    #     traverse_postorder() so no component names are hardcoded.
    #     """
    #     model    = self.fitted_model          # raises if do_fit() not called
    #     spec_obj = self._spectrum_object

    #     photon_edges = spec_obj.meta['ph_axis']

    #     # Use _bin_edges as the single source of truth for count-space edges
    #     # so that count_centers, count_bin_widths, and norm are all consistent.
    #     _bin_edges   = spec_obj.spectral_axis._bin_edges
    #     count_edges  = _bin_edges << spec_obj.spectral_axis.bin_edges.unit
    #     norm = np.diff(_bin_edges) * spec_obj.meta['exposure_time'].value

    #     observed_counts     = np.array(spec_obj.data) << spec_obj.unit
    #     observed_counts_err = spec_obj.uncertainty.array << spec_obj.uncertainty.unit

    #     count_centers    = count_edges[:-1] + 0.5 * np.diff(count_edges)
    #     count_bin_widths = 0.5 * np.diff(count_edges)
    #     x_unit = count_edges.unit
    #     y_unit = observed_counts.unit

    #     # ── Decompose model via traverse_postorder ───────────────────────
    #     source_names, pipeline = self._decompose_model(model)
    #     has_albedo  = 'Albedo' in pipeline
    #     n_sources   = len(source_names)

    #     # Assign colours: one per source + one reserved for albedo
    #     source_colours = [f'C{i}' for i in range(n_sources)]
    #     albedo_colour  = f'C{n_sources}'

    #     # ── Evaluate total model (unchanged compound model call) ─────────
    #     compound_model_evaluation = model(photon_edges)

    #     # ── Evaluate each source individually (no albedo) ────────────────
    #     source_evals = {
    #         name: self._build_source_pipeline(model, name, pipeline)(photon_edges)
    #         for name in source_names
    #     }

    #     # ── Evaluate albedo as the difference (with − without) ──────────
    #     if has_albedo:
    #         eval_no_albedo   = self._build_full_source_sum_pipeline(
    #             model, source_names, pipeline, include_albedo=False)(photon_edges)
    #         eval_with_albedo = self._build_full_source_sum_pipeline(
    #             model, source_names, pipeline, include_albedo=True)(photon_edges)
    #         albedo_eval = eval_with_albedo - eval_no_albedo

    #     # ── Figure layout ───────────────────────────────────────────────
    #     plt.rcParams.update({
    #         'xtick.labelsize': 14,
    #         'ytick.labelsize': 14,
    #         'axes.labelsize':  16,
    #     })

    #     fig     = plt.figure(figsize=(14, 10))
    #     spacing = 0.005
    #     ax_dat  = plt.axes([0, 0.25 + spacing, 1, 0.75])
    #     ax_rat  = plt.axes([0, 0, 1, 0.25], sharex=ax_dat)

    #     # ── Observed data ────────────────────────────────────────────────
    #     ax_dat.errorbar(
    #         count_centers, observed_counts / norm,
    #         yerr=observed_counts_err, xerr=count_bin_widths,
    #         label='Observed Data', marker='None', linestyle='None',
    #         color='grey', elinewidth=2)

    #     # ── Total model ──────────────────────────────────────────────────
    #     ax_dat.stairs(
    #         compound_model_evaluation.value / norm, count_edges.value,
    #         baseline=None, label='Total', linewidth=2,
    #         alpha=0.75, zorder=10000, color='k')

    #     # ── Individual source components (no albedo each) ────────────────
    #     for name, colour in zip(source_names, source_colours):
    #         ax_dat.stairs(
    #             source_evals[name].value / norm, count_edges.value,
    #             label=name, baseline=None, linewidth=2,
    #             alpha=0.9, zorder=10000, color=colour)

    #     # ── Albedo contribution ──────────────────────────────────────────
    #     if has_albedo:
    #         ax_dat.stairs(
    #             albedo_eval.value / norm, count_edges.value,
    #             label='Albedo', baseline=None, linewidth=2,
    #             alpha=0.9, zorder=10000, color=albedo_colour)

    #     # ── Free parameter annotations (one block per source) ────────────
    #     tfs   = 18
    #     y_pos = 0.75
    #     for name, colour in zip(source_names, source_colours):
    #         submodel = model[name]
    #         for param_name in submodel.param_names:
    #             param = getattr(submodel, param_name)
    #             if not param.fixed:
    #                 label = self._format_param_label(param_name, param)
    #                 ax_dat.text(
    #                     0.77, y_pos, label,
    #                     transform=ax_dat.transAxes,
    #                     fontsize=tfs, color=colour)
    #                 y_pos -= 0.055

    #     # ── Chi-squared / residuals ──────────────────────────────────────
    #     params_free = {k: v for k, v in model.fixed.items() if not v}
    #     dof = len(observed_counts) - len(params_free)

    #     print('model evaluation :', compound_model_evaluation)
    #     print('observed counts  :', observed_counts)

    #     delchi  = (observed_counts - compound_model_evaluation) / observed_counts_err
    #     chi     = np.sum((observed_counts - compound_model_evaluation)**2
    #                      / observed_counts_err**2)
    #     chi_red = np.round(chi / dof, 1).value

    #     ax_rat.stairs(delchi.value, count_edges.value,
    #                   baseline=None, linewidth=2, color='k')
    #     ax_rat.axhline(0, linestyle='--', linewidth=2, color='k')
    #     ax_rat.text(0.8, 0.06,
    #                 r'$\chi^{2}_{\mathrm{red}} = $' + str(chi_red),
    #                 transform=ax_rat.transAxes, fontsize=18)

    #     # ── Labels / formatting ──────────────────────────────────────────
    #     ax_dat.set_ylim(
    #         0.6 * np.min(observed_counts.value / norm),
    #         2.0 * np.max(observed_counts.value / norm))
    #     ax_dat.legend(frameon=False, fontsize=18)
    #     ax_dat.loglog()
    #     ax_dat.set_ylabel(r'Counts Spectrum (cts s$^{-1}$ keV$^{-1}$)', fontsize=16)

    #     ax_rat.set_xlabel(r'Energy (keV)', fontsize=16)
    #     ax_rat.set_ylabel(r'$(D - M)\,/\,\sigma$', fontsize=16)

    #     for ax in fig.axes:
    #         ax.tick_params(axis='both', which='both',
    #                        top=True, bottom=True, left=True, right=True,
    #                        direction='in', length=6)
    #         ax.tick_params(axis='both', which='minor',
    #                        top=True, bottom=True, left=True, right=True,
    #                        length=3)
    #         ax.minorticks_on()

    #     fig.suptitle(fit_times, fontsize=16, x=0.5, y=1.035)
    #     fig.savefig(str(save_name), bbox_inches='tight', dpi=300)
    #     plt.show()
    # def _albedo_angle(self):
    #     self.model.theta = angle
    #     'here we set the angles'

    # def _distance_scale(model,distance):
    #     model.distance = distance
    #     'here we set the distance'
    
    # def do_fit(self):
    
    #     data = get_data(self)


    #     'here we perform the fitting'

    # def plot_fit_results(self):
    #     'here we plot the fitting results'

    # def chi_squared(self):
    #     'here we calculate the chi^2'
    
    # def get_fit_results(self):
    #     'here we return fit results and uncertainties'

    # def get_fit_components(self):
    #      'here we return the fitted components'       

    # def run_mcmc(self):
    #      'run_mcmc'  