import numpy as np
import pytest

import astropy.units as u

from sunkit_spex.models import scaling


def inverse_square_flux_scaling():
    """Define expected inverse square flux scaling values for a distance of 1AU.

    Returns
    -------
    inputs: `list`
        The Python inputs required to produce the scaling values.

    expected_output: `astropy.units.Quantity`
        The output values expected from the InverseSquareFluxScaling model class.

    """
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    observer_distance = (1 * u.AU).to(u.cm)
    inputs = energy_edges
    expected_output = np.full(len(energy_edges) - 1, 3.55581626e-28) * (1 / u.cm**2)
    return [observer_distance, inputs], expected_output


def constant_scaling():
    """Define expected constant scaling values.

    Returns
    -------
    inputs: `list`
        The Python inputs required to produce the scaling values.

    expected_output: `numpy.array`
        The output values expected from the Constant model class.

    """
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    inputs = energy_edges
    constant_value = 0.5
    expected_output = np.full(len(energy_edges) - 1, 0.5)
    return [constant_value, inputs], expected_output


@pytest.mark.parametrize("distance", [inverse_square_flux_scaling])
def test_inverse_square_flux_scaling_class_against_expected_output(distance):
    input_args, expected = distance()
    output = scaling.InverseSquareFluxScaling(observer_distance=[input_args[0]])(input_args[1])
    np.testing.assert_allclose(output, expected, rtol=0.03)


@pytest.mark.parametrize("constant", [constant_scaling])
def test_constant_scaling_class_against_expected_output(constant):
    input_args, expected = constant()
    output = scaling.Constant(constant=[input_args[0]])(input_args[1])
    np.testing.assert_allclose(output, expected, rtol=0.03)


def test_input_units_observer_distance():
    with pytest.raises(ValueError, match="Observer distance input must be an Astropy length convertible to AU."):
        scaling.InverseSquareFluxScaling(observer_distance=1 * u.keV)(np.linspace(3, 28, 100))
