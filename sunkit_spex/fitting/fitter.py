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
from sunkit_spex.visualisation.plotter import plot

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

        
    def plot_fit_results(self,save=True):


        if save:
            plot(self._spectrum_object.spectral_axis._bin_edges*u.keV,
                self._spectrum_object.meta['ph_axis'], 
                self._spectrum_object.data << self._spectrum_object.unit, 
                self._spectrum_object.uncertainty.array << self._spectrum_object.unit,
                self.fitted_model,
                f'{self._spectrum_object.meta['time_range'][0]}_{self._spectrum_object.meta['time_range'][1]}_sunkit_spex_fit.png',
                f'{self._spectrum_object.meta['time_range'][0]} - {self._spectrum_object.meta['time_range'][1]}',
                self.fitting_method.fit_info['param_cov'],
                self._spectrum_object)
        else:
            plot(self._spectrum_object.spectral_axis._bin_edges*u.keV,
                self._spectrum_object.meta['ph_axis'], 
                self._spectrum_object.data << self._spectrum_object.unit, 
                self._spectrum_object.uncertainty.array << self._spectrum_object.unit,
                self.fitted_model,
                False,
                f'{self._spectrum_object.meta['time_range'][0]} - {self._spectrum_object.meta['time_range'][1]}',
                self.fitting_method.fit_info['param_cov'],
                self._spectrum_object)




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