"""
This module contains functions to carry out astropy fitting with spectral models
"""

import astropy.units as u
from astropy.modeling import fitting
from astropy.modeling import models

import re

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


    def get_data(self):

        count_rate = self._spectrum_object['count_rate'] * u.cts * u.keV**-1 * u.s**-1 
        count_rate_error = self._spectrum_object['count_rate_error'] * u.cts * u.keV**-1 * u.s**-1 

        photon_energy_edges = self._spectrum_object['photon_energy_edges'] * u.keV
        photon_bin_widths = np.diff(photon_energy_edges)

        effective_livetime = self._spectrum_object['effective_livetime'] * u.s
        
        angle = self._spectrum_object['angle'] * u.deg
        distance = self._spectrum_object['distance'] * u.AU.to(u.cm)


        counts = count_rate * photon_bin_widths *  effective_livetime
        counts_error = count_rate_error * photon_bin_widths *  effective_livetime

        return  (
            count_rate, 
                count_rate_error,
                counts, 
                counts_error,
                photon_energy_edges, 
                photon_bin_widths,
                effective_livetime,
                angle, distance 
                )

    @property
    def model(self):
        return self._model


            
    
    def _set_abledo_angle(self):
        
        pattern = re.compile(r"^albedo_(\d+)$")

        for name in self.model.submodel_names:
            match = pattern.match(name.lower())
            if match:
                x = match.group(1)
                param_name = f"theta_{x}"
                setattr(self.model, param_name, self.data.angle)
                getattr(self.model, param_name).fixed = True

    def _set_distance_value(self):
        
        pattern = re.compile(r"^InverseSquareFluxScaling_(\d+)$")

        for name in self.model.submodel_names:
            match = pattern.match(name.lower())
            if match:
                x = match.group(1)
                param_name = f"distance_{x}"
                setattr(self.model, param_name, self.data.distance)
                getattr(self.model, param_name).fixed = True


    def _fit_prep(self):

        self._set_abledo_angle()
        self._set_distance_value()

        

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