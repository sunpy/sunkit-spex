"""
This module contains functions to carry out astropy fitting with spectral models
"""

import astropy.units as u
from astropy.modeling import fitting
from astropy.modeling import models

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
            fitting_method = fitting.TRFLSQFitter(calc_uncertainties=True)):
        
        self._model = model
        self._spectrum_object = spectrum_object
        self._fitting_method = fitting_method


    @property
    def model(self):
        return self._model
            
    
    def _set_abledo_angle(self):

        match = np.where(np.array(self._model.submodel_names)=='Albedo')[0]
        
        # if np.shape(match) != 0:
        #     param_names = [f'theta_{str(ind)}' for ind in match]

        #     for param_name in param_names:
        #         setattr(self._model, param_name, self._spectrum_object.meta['angle'])
        #         getattr(self._model, param_name).fixed = True 

        if 'Albedo' in self.model.submodel_names:

            print(len(self._spectrum_object.meta['ph_axis']))

            self._model = self._model.replace_submodel('Albedo',Albedo(energy_edges=self._spectrum_object.meta['ph_axis'],
                                                                       theta=self._spectrum_object.meta['angle']))    

    def _set_observer_distance(self):

        match = np.where(np.array(self._model.submodel_names)=='InverseSquareFluxScaling')[0]
        
        if np.shape(match) != 0:
            param_names = [f'observer_distance_{str(ind)}' for ind in match]

            for param_name in param_names:
                setattr(self._model, param_name, self._spectrum_object.meta['distance'])
                getattr(self._model, param_name).fixed = True        

    def _set_srm(self):
        
        if 'SRM' in self.model.submodel_names:

            self._model = self._model.replace_submodel('SRM',MatrixModel(matrix= self._spectrum_object.meta['srm'],   
                                                                         spectrum_object=self._spectrum_object, 
                                                                         model_spec_units=u.ph * u.keV**-1 * u.s**-1 * u.cm**-2))            
                                                        
    def _fit_prep(self):

        self._set_abledo_angle()
        self._set_observer_distance()
        self._set_srm()
    
    def do_fit(self):

        self._fit_prep()

        w = (1/self._spectrum_object.uncertainty.array) << self._spectrum_object.uncertainty.unit
        data = self._spectrum_object.data << self._spectrum_object.unit

        print('spec_axis = ',self._spectrum_object.spectral_axis.bin_edges)
        print('spec_data = ',data)

        fitted_model = self._fitting_method(model=self._model,
                                            x=self._spectrum_object.meta['ph_axis'],
                                            y=data,
                                            weights=w,
                                            estimate_jacobian=True)
        
        return fitted_model

        

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