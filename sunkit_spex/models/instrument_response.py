"""Module for model components required for instrument response models."""

import numpy as np

from astropy.modeling import Fittable1DModel, Parameter
import astropy.units as u

__all__ = ["MatrixModel"]


class MatrixModel(Fittable1DModel):
    name = "SRM"
    conversion_factor = Parameter(fixed=True)
    _input_units_allow_dimensionless = True

    def __init__(self, matrix=None, 
                 model_spec_units=u.dimensionless_unscaled, 
                 data_spec_units=u.dimensionless_unscaled, 
                 conversion_factor=1*u.dimensionless_unscaled,
                 spectrum_object= None,
                 spectral_model=True):
        
        self.spectral_model = spectral_model

        if not self.spectral_model:

            self.model_spec_units = model_spec_units
            self.data_spec_units = data_spec_units
            self.matrix = matrix
            conversion_factor = 1 << (data_spec_units / model_spec_units)
        else:
            # self.matrix = matrix
            self.spectrum_object = spectrum_object
            self.model_spec_units = model_spec_units

            if spectrum_object:
                self.data_spec_units = spectrum_object.unit
                conversion_factor = 1* u.ct / u.ph
                # conversion_factor = 1 << (self.data_spec_units / self.model_spec_units)
                # print(conversion_factor)
            else:
                self.data_spec_units = data_spec_units
                # conversion_factor = 1 << (data_spec_units / model_spec_units)
                conversion_factor = 1* u.ct / u.ph

        super().__init__(conversion_factor=conversion_factor)

    def evaluate(self, x, conversion_factor):

        # matrix = self.matrix

        if self.spectral_model:

            matrix = self.spectrum_object.meta['srm']
            input_axis = np.array(self.spectrum_object.spectral_axis.bin_edges)
            input_widths = np.diff(input_axis)
            output_widths = np.diff(self.spectrum_object.meta['ph_axis'])

            # print('IR SRM = ',self.spectrum_object.meta['srm'].shape)
            # print('IR SRM = ',self.spectrum_object.spectral_axis.bin_edges.shape)

            geo_area = self.spectrum_object.meta['geo_area'] 
            exposure_time = self.spectrum_object.meta['exposure_time'] 
            norm = input_widths * exposure_time * geo_area

            # print('input_widths = ',input_widths)
            # print('exposure_time = ',exposure_time)
            # print('geo_area = ',geo_area)

            # print(x.unit)
            # print(conversion_factor.unit)
            # print(norm.unit)

            # flux =  (x @ matrix) * conversion_factor * norm
            flux =  (((x*output_widths*exposure_time)@ (matrix*geo_area*u.cm**2)) * conversion_factor ) 

        else:
            flux =  x  @ matrix * conversion_factor * (geo_area*u.cm**2)

        # print('HHEERRREEEE')

        if hasattr(conversion_factor,"unit"):
            return flux
        else:
            return flux.value

    def set_spectrum_object(self, new_spectrum_object):
        self.spectrum_object = new_spectrum_object
    
    # @property
    # def model_spec_units(self):
    #     return self._model_spec_units

    # @model_spec_units.setter
    # def model_spec_units(self, new_unit):
    #     self._model_spec_units = new_unit

    #     if hasattr(self,"data_spec_units"):

    #         if self.data_spec_units != u.dimensionless_unscaled:

    #             new_param_unit = self.data_spec_units / new_unit

    #             self.conversion_factor = self.conversion_factor.value * new_param_unit

    #         else:
                
    #             self.conversion_factor = self.conversion_factor * u.dimensionless_unscaled


    # @property
    # def data_spec_units(self):
    #     return self._data_spec_units

    # @data_spec_units.setter
    # def data_spec_units(self, new_unit):
    #     self._data_spec_units = new_unit

    #     if hasattr(self,"model_spec_units"):


    #         if self.data_spec_units != u.dimensionless_unscaled:

    #             new_param_unit =  new_unit / self.model_spec_units 

    #             self.conversion_factor = self.conversion_factor.value * new_param_unit

    #         else:
                
    #             self.conversion_factor = self.conversion_factor * u.dimensionless_unscaled

    @property
    def input_units(self):
        # return {"x": self.model_spec_units }SS
        return {"x": u.ph * u.keV**-1 * u.s**-1 * u.cm**-2 }

    @property
    def return_units(self):
        # return {"y": self.data_spec_units}
        return {"y": u.ct}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {"conversion_factor": self.conversion_factor.unit}

    # @property
    # def input_units(self):
    #     # return {"x": self.model_spec_units }SS
    #     return {"x": u.ph * u.keV**-1 * u.s**-1 * u.cm**-2 }

    # @property
    # def return_units(self):
    #     # return {"y": self.data_spec_units}
    #     return {"y": u.ct* u.keV**-1 * u.s**-1}

    # def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
    #     return {"conversion_factor": self.conversion_factor.unit}