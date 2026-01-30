"""Module for model components required for instrument response models."""

import numpy as np

from astropy.modeling import Fittable1DModel, Parameter

__all__ = ["MatrixModel"]


class MatrixModel(Fittable1DModel):
    name = "SRM"
    conversion_factor = Parameter(fixed=True)
    _input_units_allow_dimensionless = True

    def __init__(self, matrix, input_axis, output_axis, input_spec_units, output_spec_units):
        self.input_spec_units = input_spec_units
        self.output_spec_units = output_spec_units
        self.input_axis = input_axis
        self.output_axis = output_axis
        self.matrix = matrix

        # Unit hack to allow output units to be different from input units
        conversion_factor = 1 << (output_spec_units / input_spec_units)
        super().__init__(conversion_factor)

    def evaluate(self, x, conversion_factor):
        input_widths = np.diff(self.input_axis)
        output_widths = np.diff(self.output_axis)
        return ((x * input_widths) @ self.matrix * conversion_factor) / output_widths

    @property
    def input_units(self):
        return {"x": self.input_spec_units}

    @property
    def return_units(self):
        return {"y": self.output_spec_units}

    def _parameter_units_for_data_units(self):
        return {"conversion_factor": self.output_spec_units / self.input_spec_units}
