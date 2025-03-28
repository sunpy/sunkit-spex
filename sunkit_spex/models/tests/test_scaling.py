import numpy as np
import pytest

import astropy.units as u

from sunkit_spex.models import scaling


def test_input_units_observer_distance():
    with pytest.raises(ValueError, match="Observer distance input must be an Astropy length convertible to AU."):
        scaling.InverseSquareFluxScaling(observer_distance=1 * u.keV)(np.linspace(3, 28, 100))
