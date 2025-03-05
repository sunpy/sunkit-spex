import astropy.units as u
from astropy.modeling import FittableModel, Parameter
from astropy.units import Quantity

__all__ = ["Constant", "DistanceScale"]


class DistanceScale(FittableModel):
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

    def evaluate(self, spectrum, observer_distance):
        spectrum_distance_corrected = distance_correction(spectrum, observer_distance)

        return spectrum_distance_corrected

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

    def evaluate(self, spectrum, constant):
        return spectrum * constant


def distance_correction(spectrum, observer_distance):
    if isinstance(observer_distance, Quantity):
        AU_distance_cm = 1 * u.AU.to(u.cm)
        observer_distance_cm = observer_distance.to(u.cm)
    else:
        AU_distance_cm = 1.496e13
        observer_distance_cm = observer_distance * AU_distance_cm

    scale = AU_distance_cm**2 / observer_distance_cm**2

    flux = spectrum * scale

    return flux
