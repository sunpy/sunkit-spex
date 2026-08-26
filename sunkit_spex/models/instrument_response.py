"""Module for model components required for instrument response models."""

import numpy as np
from numpy.typing import NDArray

from astropy.modeling import Fittable1DModel, Parameter

__all__ = ["MatrixModel"]


class MatrixModel(Fittable1DModel):  # type: ignore[misc]  # astropy/ndcube/gwcs ship no type stubs
    def __init__(self, matrix: NDArray[np.float64]) -> None:
        self.matrix = Parameter(default=matrix, description="The matrix with which to multiply the input.", fixed=True)
        super().__init__()

    # Parameters intentionally left unannotated: astropy's Model.input_units reads
    # evaluate.__annotations__ to infer per-input units, keyed by input name, so adding
    # unrelated (non-unit) annotations here breaks that introspection at runtime.
    def evaluate(self, model_y):  # type: ignore[no-untyped-def]
        # Requires input must have a specific dimensionality
        return model_y @ self.matrix
