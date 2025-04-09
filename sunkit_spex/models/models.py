"""Module for generic mathematical models."""

import numpy as np

from astropy.modeling import Fittable1DModel, Parameter
from astropy.units import Quantity

__all__ = ["GaussianModel", "StraightLineModel"]


class StraightLineModel(Fittable1DModel):
    slope = Parameter(default=1, description="Gradient of a straight line model.")
    intercept = Parameter(default=0, description="Y-intercept of a straight line model.")

    def __init__(self, *args, **kwargs):
        self.edges = kwargs.pop("edges")

        super().__init__(*args, **kwargs)

    def evaluate(self, x, slope, intercept):
        if self.edges:
            x = x[:-1] + 0.5 * np.diff(x)

        """Evaluate the straight line model at `x` with parameters `slope` and `intercept`."""
        return slope * x + intercept


class GaussianModel(Fittable1DModel):
    amplitude = Parameter(default=1, min=0, description="Scalar for Gaussian.")
    mean = Parameter(default=0, min=0, description="X-offset for Gaussian.")
    stddev = Parameter(default=1, description="Sigma for Gaussian.")

    def __init__(self, *args, **kwargs):
        self.edges = kwargs.pop("edges")

        super().__init__(*args, **kwargs)

    def evaluate(self, x, amplitude, mean, stddev):
        """Evaluate the Gaussian model at `x` with parameters `amplitude`, `mean`, and `stddev`."""

        if self.edges:
            x = x[:-1] + 0.5 * np.diff(x)
        
        if isinstance(x, Quantity):
            y = amplitude * np.e ** (-((x.value - mean) ** 2) / (2 * stddev**2))
        else:
            y = amplitude * np.e ** (-((x - mean) ** 2) / (2 * stddev**2))

        return y
