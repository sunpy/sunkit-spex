"""Module for model components required for instrument response models."""

import astropy.units as u
from astropy.modeling import Fittable1DModel, Parameter

__all__ = ["MatrixModel"]


class MatrixModel(Fittable1DModel):
    input_units = {"x": u.ph}
    c = Parameter(fixed=True)

    def __init__(self, matrix, input_axis, output_axis, c):
        self.inputs_axis = input_axis
        self.output_axis = output_axis
        self.matrix = matrix
        super().__init__(c)
        # self.matrix.value = self.matrix.value.flatten()

    def evaluate(self, x, c):
        # Requires input must have a specific dimensionality
        return x @ self.matrix * c

    # @property
    # def input_units(self):
    #     return {"x": u.ph}
    #
    # @property
    # def return_units(self):
    #     return {"y": u.ct}

    @staticmethod
    def _parameter_units_for_data_units(inputs_unit, outputs_unit):
        return {"c": outputs_unit["y"] / inputs_unit["x"]}
