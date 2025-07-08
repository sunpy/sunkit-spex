"""Module for model components required for instrument response models."""

import numpy as np

from astropy.modeling import Fittable1DModel, Parameter

__all__ = ["MatrixModel"]


class MatrixModel(Fittable1DModel):
    name = "SRM"

    c = Parameter(fixed=True)

    def __init__(self, matrix, input_axis, output_axis, _input_units, _output_units, c):
        self._input_units = _input_units
        self._output_units = _output_units
        self.input_axis = input_axis
        self.output_axis = output_axis
        self.matrix = matrix
        super().__init__(c)

    _input_units_allow_dimensionless = True

    def evaluate(self, x, c):
        # Requires input must have a specific dimensionality

        input_widths = np.diff(self.input_axis)
        output_widths = np.diff(self.output_axis)


        flux = ((x * input_widths) @ self.matrix * c) / output_widths

        if hasattr(c, "unit"):
            return flux
        return flux.value

    @property
    def input_units(self):
        return self._input_units

    @property
    def return_units(self):
        return self._output_units

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {"c": self._output_units["y"] / self._input_units["x"]}