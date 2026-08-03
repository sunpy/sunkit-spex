from typing import Any

import numpy as np

import astropy.units as u
from astropy.modeling import FittableModel, Parameter
from astropy.units import Quantity

__all__ = ["Constant", "InverseSquareFluxScaling"]


class InverseSquareFluxScaling(FittableModel):  # type: ignore[misc]  # astropy/ndcube/gwcs ship no type stubs
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

    # Parameters intentionally left unannotated: astropy's Model.input_units reads
    # evaluate.__annotations__ to infer per-input units, keyed by input name, so adding
    # unrelated (non-unit) annotations here breaks that introspection at runtime.
    def evaluate(self, x, observer_distance):  # type: ignore[no-untyped-def]
        if isinstance(observer_distance, Quantity):
            # A Quantity's .unit is never actually None; pyright's astropy inference is overly broad here.
            if observer_distance.unit.is_equivalent(u.AU):  # pyright: ignore[reportOptionalMemberAccess]
                observer_distance_cm = observer_distance.to(u.cm)
            else:
                raise ValueError("Observer distance input must be an Astropy length convertible to AU.")

        else:
            # `1 * u.AU` is a Quantity at runtime; pyright's overload resolution infers a
            # Unit-subclass union instead and misses the .to_value method that exists.
            AU_distance_cm = (1 * u.AU).to_value(u.cm)  # pyright: ignore[reportAttributeAccessIssue]
            observer_distance_cm = observer_distance * AU_distance_cm

        return 1 / (4 * np.pi * (observer_distance_cm**2))

    @property
    def return_units(self) -> dict[str, Any]:
        return {"y": u.cm**-2}

    def _parameter_units_for_data_units(self, inputs_unit: Any, outputs_unit: Any) -> dict[str, Any]:
        return {"observer_distance": u.AU}


class Constant(FittableModel):  # type: ignore[misc]  # astropy/ndcube/gwcs ship no type stubs
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

    # constant.default is always the concrete `1` set above; Parameter.default's declared
    # type is broader (Any | None), which is what pyright is actually reacting to here.
    def __init__(self, constant: Any = u.Quantity(constant.default)) -> None:  # pyright: ignore[reportArgumentType]
        super().__init__(constant=constant)

    # Parameters intentionally left unannotated: astropy's Model.input_units reads
    # evaluate.__annotations__ to infer per-input units, keyed by input name, so adding
    # unrelated (non-unit) annotations here breaks that introspection at runtime.
    def evaluate(self, x, constant):  # type: ignore[no-untyped-def]
        return constant

    @property
    def return_units(self) -> dict[str, Any] | None:
        if isinstance(self.constant, Quantity):
            return {"y": self.constant.unit}
        return None

    def _parameter_units_for_data_units(self, inputs_unit: Any, outputs_unit: Any) -> dict[str, Any]:
        return {"constant": outputs_unit["y"]}
