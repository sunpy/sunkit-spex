"""Module for generic mathematical models."""

import numpy as np

from astropy.modeling import Fittable1DModel, Parameter

__all__ = ["StraightLineModel", "GaussianModel"]


class StraightLineModel(Fittable1DModel):
    slope = Parameter(default=1, description="Gradiant of a straight line model.")
    intercept = Parameter(default=0, description="Y-intercept of a straight line model.")

    @staticmethod
    def evaluate(x, slope, intercept):
        """Evaluate the straight line model at `x` with parameters `slope` and `intercept`."""
        return slope * x + intercept


class GaussianModel(Fittable1DModel):
    amplitude = Parameter(default=1, min=0, description="Scalar for Gaussian.")
    mean = Parameter(default=0, min=0, description="X-offset for Gaussian.")
    stddev = Parameter(default=1, description="Sigma for Gaussian.")

    @staticmethod
    def evaluate(x, amplitude, mean, stddev):
        """Evaluate the Gaussian model at `x` with parameters `amplitude`, `mean`, and `stddev`."""
        return amplitude * np.e ** (-((x - mean) ** 2) / (2 * stddev**2))
