from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.io import readsav

import astropy.units as u
from astropy.modeling import FittableModel, Parameter
from astropy.units import Quantity

from sunpy.data import cache

__all__ = ["DistanceScale", "Constant"]


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

    # def __init__(self, *args, **kwargs):
    #     self.energy_edges = kwargs.pop("energy_edges")

    #     super().__init__(*args, **kwargs)

    def evaluate(spectrum, observer_distance):

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

    def evaluate(spectrum, constant):

        return spectrum * constant



def distance_correction(spectrum, observer_distance):

    AU_distance_cm = 1*u.AU.to(u.cm)
    observer_distance_cm = observer_distance.to(u.cm)

    scale = AU_distance_cm**2 / observer_distance_cm**2

    flux = spectrum * scale

    return flux

