"""Module for model components required for instrument response models."""

from astropy.modeling import Fittable1DModel, Parameter

__all__ = ["MatrixModel"]


class MatrixModel(Fittable1DModel):
    def __init__(self, matrix, input_axis, output_axis):
        self.matrix = Parameter(default=matrix, description="The matrix with which to multiply the input.", fixed=True)
        self.inputs_axis = input_axis
        self.output_axis = output_axis
        super().__init__()

    def evaluate(self, model_y):
        # Requires input must have a specific dimensionality
        return model_y @ self.matrix
