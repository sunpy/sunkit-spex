"""Module for model components required for instrument response models."""

import astropy.units as u
from astropy.modeling import Fittable1DModel, Parameter

__all__ = ["MatrixModel"]


class MatrixModel(Fittable1DModel):
    matrix = Parameter(description="The matrix with which to multiply the input.", fixed=True)

    def __init__(self, matrix, input_axis, output_axis):
        # self.matrix = Parameter(default=matrix, description="The matrix with which to multiply the input.", fixed=True)
        self.inputs_axis = input_axis
        self.output_axis = output_axis
        super().__init__(matrix)

    @staticmethod
    def evaluate(x, matrix):
        # Requires input must have a specific dimensionality
        return x @ matrix.reshape(1961, 1961)

    # @property
    # def input_units(self):
    #     return {"x": u.ph}
    #
    # @property
    # def return_units(self):
    #     return {"y": u.ct}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {"matrix": u.ct / u.ph}
