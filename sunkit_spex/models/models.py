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

    def __init__(self, slope=slope.default, intercept=intercept.default, photon_model=True, **kwargs):
        self.photon_model = photon_model

        super().__init__(slope=slope, intercept=intercept, **kwargs)

    def evaluate(self, x, slope, intercept):
        if self.photon_model:
            x.to(u.keV)
            x = x[:-1] + 0.5 * np.diff(x)

        if isinstance(x, Quantity):
            x = x.value

        """Evaluate the straight line model at `x` with parameters `slope` and `intercept`."""
        return slope * x + intercept

    @property
    def input_units(self):
        if not self.photon_model:
            return None
        return {"x": u.keV}

    @property
    def return_units(self):
        if not self.photon_model:
            return None
        return {"y": u.ph * u.keV**-1 * u.s**-1}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        if not self.photon_model:
            return None
        return {"slope": outputs_unit["y"] / inputs_unit["x"], "intercept": outputs_unit["y"]}


class GaussianModel(FittableModel):
    n_inputs = 1
    n_outputs = 1

    _input_units_allow_dimensionless = True

    amplitude = Parameter(default=1, min=0, description="Scalar for Gaussian.")
    mean = Parameter(default=0, min=0, description="X-offset for Gaussian.")
    stddev = Parameter(default=1, description="Sigma for Gaussian.")

    def __init__(
        self, amplitude=amplitude.default, mean=mean.default, stddev=stddev.default, photon_model=True, **kwargs
    ):
        self.photon_model = photon_model

        super().__init__(amplitude=amplitude, mean=mean, stddev=stddev, **kwargs)

    def evaluate(self, x, amplitude, mean, stddev):
        """Evaluate the Gaussian model at `x` with parameters `amplitude`, `mean`, and `stddev`."""

        if self.photon_model:
            x = x[:-1] + 0.5 * np.diff(x)

        if isinstance(x, Quantity):
            y = amplitude * np.e ** (-((x.value - mean) ** 2) / (2 * stddev**2))
        else:
            y = amplitude * np.e ** (-((x - mean) ** 2) / (2 * stddev**2))

        return y

    @property
    def input_units(self):
        if not self.photon_model:
            return None
        return {"x": u.keV}

    @property
    def return_units(self):
        if not self.photon_model:
            return None
        return {"y": u.ph * u.keV**-1 * u.s**-1}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {"mean": inputs_unit["x"], "stddev": inputs_unit["x"], "amplitude": outputs_unit["y"]}
