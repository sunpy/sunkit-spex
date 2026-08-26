"""Module for generic mathematical models."""

from typing import Any

import numpy as np

import astropy.units as u
from astropy.modeling import FittableModel, Parameter
from astropy.units import Quantity

__all__ = ["GaussianModel", "StraightLineModel"]


class StraightLineModel(FittableModel):  # type: ignore[misc]  # astropy/ndcube/gwcs ship no type stubs
    n_inputs = 1
    n_outputs = 1

    _input_units_allow_dimensionless = True

    input_units_equivalencies = {"keV": u.spectral()}

    slope = Parameter(default=1, description="Gradient of a straight line model.")
    intercept = Parameter(default=0, description="Y-intercept of a straight line model.")

    def __init__(self, slope: Any = slope, intercept: Any = intercept, edges: bool = True, **kwargs: Any) -> None:
        self.edges = edges

        super().__init__(slope, intercept, **kwargs)

    # Parameters intentionally left unannotated: astropy's Model.input_units reads
    # evaluate.__annotations__ to infer per-input units, keyed by input name, so adding
    # unrelated (non-unit) annotations here breaks that introspection at runtime.
    def evaluate(self, x, slope, intercept):  # type: ignore[no-untyped-def]
        if self.edges:
            x = x[:-1] + 0.5 * np.diff(x)

        """Evaluate the straight line model at `x` with parameters `slope` and `intercept`."""
        return slope * x + intercept

    @property
    def input_units(self) -> dict[str, Any] | None:
        if isinstance(self.slope, Quantity):
            return {"x": self.intercept.unit / self.slope.unit}
        return None

    @property
    def return_units(self) -> dict[str, Any] | None:
        if isinstance(self.slope, Quantity):
            return {"y": self.intercept.unit}
        return None

    def _parameter_units_for_data_units(self, input_units: Any, output_units: Any) -> dict[str, Any]:
        return {"slope": output_units["y"] / input_units["x"], "intercept": output_units["y"]}


class GaussianModel(FittableModel):  # type: ignore[misc]  # astropy/ndcube/gwcs ship no type stubs
    n_inputs = 1
    n_outputs = 1

    _input_units_allow_dimensionless = True

    amplitude = Parameter(default=1, min=0, description="Scalar for Gaussian.")
    mean = Parameter(default=0, min=0, description="X-offset for Gaussian.")
    stddev = Parameter(default=1, description="Sigma for Gaussian.")

    def __init__(
        self, amplitude: Any = amplitude, mean: Any = mean, stddev: Any = stddev, edges: bool = True, **kwargs: Any
    ) -> None:
        self.edges = edges

        super().__init__(amplitude, mean, stddev, **kwargs)

    # Parameters intentionally left unannotated: astropy's Model.input_units reads
    # evaluate.__annotations__ to infer per-input units, keyed by input name, so adding
    # unrelated (non-unit) annotations here breaks that introspection at runtime.
    def evaluate(self, x, amplitude, mean, stddev):  # type: ignore[no-untyped-def]
        """Evaluate the Gaussian model at `x` with parameters `amplitude`, `mean`, and `stddev`."""

        if self.edges:
            x = x[:-1] + 0.5 * np.diff(x)

        return amplitude * np.e ** (-((x - mean) ** 2) / (2 * stddev**2))

    @property
    def input_units(self) -> dict[str, Any] | None:
        if isinstance(self.mean, Quantity):
            return {"x": self.mean.unit}
        return None

    @property
    def return_units(self) -> dict[str, Any] | None:
        if isinstance(self.amplitude, Quantity):
            return {"y": self.amplitude.unit}
        return None

    def _parameter_units_for_data_units(self, input_units: Any, output_units: Any) -> dict[str, Any]:
        return {"mean": input_units["x"], "stddev": input_units["x"], "amplitude": output_units["y"]}
