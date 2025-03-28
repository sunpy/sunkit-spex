import numpy as np

import astropy.units as u
from astropy.modeling import FittableModel, Parameter
from astropy.units import Quantity

__all__ = ["Constant", "InverseSquareFluxScaling"]


class InverseSquareFluxScaling(FittableModel):
    n_inputs = 1
    n_outputs = 1

    observer_distance = Parameter(
        name="observer_distance",
        default=1,
        unit=u.AU,
        description="Distance to the observer in AU",
        fixed=True,
    )

    _input_units_allow_dimensionless = True

    def evaluate(self, x, observer_distance):
        correction = distance_correction(observer_distance)
        dimension = np.shape(x)[0] - 1

        if isinstance(observer_distance, Quantity):
            return np.full(dimension, correction.value) * correction.unit
        return np.full(dimension, correction) * (1 / u.cm**2)

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {"observer_distance": u.AU}


class Constant(FittableModel):
    n_inputs = 1
    n_outputs = 1

    constant = Parameter(
        name="constant",
        default=1,
        description="Multiplicative Constant",
        fixed=True,
    )

    _input_units_allow_dimensionless = True

    def evaluate(self, x, constant):
        dimension = np.shape(x)[0] - 1

        return np.full(dimension, constant)


def distance_correction(observer_distance):
    if isinstance(observer_distance, Quantity):
        if observer_distance.unit.is_equivalent(u.AU):
            observer_distance_cm = observer_distance.to(u.cm)
        else:
            raise ValueError("Observer distance input must be an Astropy length convertible to AU.")

    else:
        AU_distance_cm = 1 * u.AU.to(u.cm).value
        observer_distance_cm = observer_distance * AU_distance_cm

    return 1 / (4 * np.pi * (observer_distance_cm**2))
