import numpy as np
import pytest

import astropy.units as u

from sunkit_spex.models import scaling


def inverse_square_scaling():
    """ """
    energy_edges = np.arange(3, 28.5, 0.5) * u.keV
    observer_distance = (1 * u.AU).to(u.cm)
    inputs = energy_edges
    expected_output = np.full(len(energy_edges) - 1, 3.55581626e-28) * (1 / u.cm**2)
    return [observer_distance, inputs], expected_output


@pytest.mark.parametrize("distance", [inverse_square_scaling])
def test_continuum_emission_against_ssw(distance):
    input_args, expected = distance()
    output = scaling.InverseSquareFluxScaling(observer_distance=[input_args[0]])(input_args[1])
    np.testing.assert_allclose(output, expected, rtol=0.03)


def test_input_units_observer_distance():
    with pytest.raises(ValueError, match="Observer distance input must be an Astropy length convertible to AU."):
        scaling.InverseSquareFluxScaling(observer_distance=1 * u.keV)(np.linspace(3, 28, 100))
