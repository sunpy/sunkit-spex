"""Module for model components required for instrument response models."""

from astropy.modeling import Fittable1DModel, Parameter

__all__ = ["MatrixModel"]


class MatrixModel(Fittable1DModel):
    c = Parameter(fixed=True)

    def __init__(self, matrix, input_axis, output_axis, _input_units, _output_units, c):
        self._input_units = _input_units
        self._output_units = _output_units
        self.inputs_axis = input_axis
        self.output_axis = output_axis
        self.matrix = matrix
        super().__init__(c)
        # self.matrix.value = self.matrix.value.flatten()

    def evaluate(self, x, c):
        # Requires input must have a specific dimensionality

        return x @ self.matrix * c

    @property
    def input_units(self):
        return self._input_units

    # @input_units.setter
    # def input_units(self, units):
    #     self._input_units = units

    @property
    def return_units(self):
        return self._output_units

    # @return_units.setter
    # def return_units(self, units):
    #     self._output_units = units

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {"c": self._output_units["y"] / self._input_units["x"]}
