"""Module for generic mathematical models."""

import numpy as np

import astropy.units as u
from astropy.modeling import FittableModel, Parameter
from astropy.units import Quantity

__all__ = ["GaussianModel", "StraightLineModel"]


class StraightLineModel(FittableModel):
    n_inputs = 1
    n_outputs = 1

    _input_units_allow_dimensionless = True

    input_units_equivalencies = {"keV": u.spectral()}

    slope = Parameter(default=1, description="Gradient of a straight line model.")
    intercept = Parameter(default=0, description="Y-intercept of a straight line model.")

    def __init__(self, slope=slope, intercept=intercept, edges=True, **kwargs):
        self.edges = edges

        super().__init__(slope, intercept, **kwargs)

    def evaluate(self, x, slope, intercept):
        if self.edges:
            x = x[:-1] + 0.5 * np.diff(x)

        """Evaluate the straight line model at `x` with parameters `slope` and `intercept`."""
        return slope * x + intercept

    @property
    def input_units(self):
        if isinstance(self.slope, Quantity):
            return {"x": self.intercept.unit / self.slope.unit}
        return None

    @property
    def return_units(self):
        if isinstance(self.slope, Quantity):
            return {"y": self.intercept.unit}
        return None

    def _parameter_units_for_data_units(self, input_units, output_units):
        return {"slope": output_units["y"] / input_units["x"], "intercept": output_units["y"]}


class GaussianModel(FittableModel):
    n_inputs = 1
    n_outputs = 1

    _input_units_allow_dimensionless = True

    amplitude = Parameter(default=1, min=0, description="Scalar for Gaussian.")
    mean = Parameter(default=0, min=0, description="X-offset for Gaussian.")
    stddev = Parameter(default=1, description="Sigma for Gaussian.")

    def __init__(self, amplitude=amplitude, mean=mean, stddev=stddev, edges=True, **kwargs):
        self.edges = edges

        super().__init__(amplitude, mean, stddev, **kwargs)

    def evaluate(self, x, amplitude, mean, stddev):
        """Evaluate the Gaussian model at `x` with parameters `amplitude`, `mean`, and `stddev`."""

        if self.edges:
            x = x[:-1] + 0.5 * np.diff(x)

        return amplitude * np.e ** (-((x - mean) ** 2) / (2 * stddev**2))

    @property
    def input_units(self):
        if isinstance(self.mean, Quantity):
            return {"x": self.mean.unit}
        return None

    @property
    def return_units(self):
        if isinstance(self.amplitude, Quantity):
            return {"y": self.amplitude.unit}
        return None

    def _parameter_units_for_data_units(self, input_units, output_units):
        return {"mean": input_units["x"], "stddev": input_units["x"], "amplitude": output_units["y"]}
