import numpy as np

import astropy.units as u
from astropy.modeling import FittableModel, Parameter
from astropy.units import Quantity

__all__ = ["Constant", "InverseSquareFluxScaling"]


class InverseSquareFluxScaling(FittableModel):
    """
    InverseSqaureFluxScaling model converts luminosity output of physical models to a distance scaled flux.

    Parameters
    ==========
    energy_edges :
        Energy edges associated with input spectrum
    observer_distance:
        Distance of the observer from the source.




    Examples
    ========
    .. plot::
        :include-source:

        import astropy.units as u
        import numpy as np
        import matplotlib.pyplot as plt


        from sunkit_spex.models.scaling import InverseSquareFluxScaling
        from sunkit_spex.models.models import StraightLineModel

        y_units = u.ph*u.keV**-1*u.s**-1
        x_units = u.keV

        ph_energies = np.arange(4, 100, 0.5)*x_units
        ph_energies_centers = ph_energies[:-1] + 0.5*np.diff(ph_energies)

        sim_cont = {"slope": -2*y_units/x_units, "intercept": 100*y_units}
        source = StraightLineModel(**sim_cont)

        plt.figure()
        for i, d in enumerate([0.25,0.5,1]):
            distance =  InverseSquareFluxScaling(observer_distance=d*u.AU)
            observed = source * distance
            plt.plot(ph_energies_centers ,  observed(ph_energies), label='D = '+str(d)+' AU')
        plt.loglog()
        plt.legend()
        plt.show()
    """

    name = "InverseSquareFluxScaling"

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
        if isinstance(observer_distance, Quantity):
            if observer_distance.unit.is_equivalent(u.AU):
                observer_distance_cm = observer_distance.to(u.cm)
            else:
                raise ValueError("Observer distance input must be an Astropy length convertible to AU.")

        else:
            AU_distance_cm = (1 * u.AU).to_value(u.cm)
            observer_distance_cm = observer_distance * AU_distance_cm

        return 1 / (4 * np.pi * (observer_distance_cm**2))

    @property
    def return_units(self):
        return {"y": u.cm**-2}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {"observer_distance": u.AU}


class Constant(FittableModel):
    """
    A model which returns an array with dimensions n-1 of the input dimension populated with a constant value,
    of whichever units specified by the user.

    Parameters
    ==========
    energy_edges :
        Energy edges associated with input spectrum
    constant :
        A constant value which populates the output array


    Examples
    ========
    .. plot::
        :include-source:

        import astropy.units as u
        import numpy as np
        import matplotlib.pyplot as plt


        from sunkit_spex.models.scaling import Constant
        from sunkit_spex.models.models import StraightLineModel

        y_units = u.ph*u.keV**-1*u.s**-1
        x_units = u.keV

        ph_energies = np.arange(4, 100, 0.5)*x_units
        ph_energies_centers = ph_energies[:-1] + 0.5*np.diff(ph_energies)

        sim_cont = {"slope": -2*y_units/x_units, "intercept": 100*y_units}
        source = StraightLineModel(**sim_cont)


        plt.figure()
        for i, c in enumerate([0.25,0.5,1,2,4]):
            constant =  Constant(constant=c)
            observed = source * constant
            plt.plot(ph_energies_centers ,  observed(ph_energies), label='Const = '+str(c))
        plt.loglog()
        plt.legend()
        plt.show()
    """

    n_inputs = 1
    n_outputs = 1

    constant = Parameter(
        name="constant",
        default=1,
        description="Constant",
        fixed=True,
    )

    _input_units_allow_dimensionless = True

    name = "Constant"

    def __init__(self, constant=u.Quantity(constant.default)):
        super().__init__(constant=constant)

    def evaluate(self, x, constant):
        return constant

    @property
    def return_units(self):
        if isinstance(self.constant, Quantity):
            return {"y": self.constant.unit}
        return None

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        return {"constant": outputs_unit["y"]}
