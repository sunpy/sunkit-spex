import numpy as np
import numpy.testing as npt

import astropy.units as u

import sys
sys.path.insert(0, '..')
import photon_power_law as ppl


def test_different_bins():
    ''' make sure that flux is conserved when using different energy bins '''
    eb1 = np.linspace(10, 100, num=30) << u.keV
    eb2 = np.linspace(10, 100, num=10) << u.keV

    shared_kw = dict(
        reference_energy=3 << u.keV,
        reference_flux=1 << ppl.PHOTON_RATE_UNIT,
        break_energy=20 << u.keV,
        lower_index=1,
        upper_index=3
    )

    dist1 = ppl.broken_power_law_binned_flux(energy_edges=eb1, **shared_kw)
    dist2 = ppl.broken_power_law_binned_flux(energy_edges=eb2, **shared_kw)

    flux1 = np.sum(dist1 * np.diff(eb1))
    flux2 = np.sum(dist2 * np.diff(eb2))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.stairs(dist1.value, eb1.value)
    ax.stairs(dist2.value, eb2.value)
    ax.set(xscale='log', yscale='log')
    plt.show()

    npt.assert_allclose(flux1, flux2)


def test_integration():
    for idx in range(10):
        n = 1 << ppl.PHOTON_RATE_UNIT
        norm_e = 3 << u.keV
        up_e, low_e = 4 << u.keV, 2 << u.keV

        if idx != 1:
            analytical_result = (
                n * norm_e / (1 - idx) * (
                    (up_e / norm_e)**(1 - idx) - (low_e / norm_e)**(1 - idx)
                )
            )
        else:
            analytical_result = n * norm_e * np.log(up_e / low_e)

        calc_res = ppl.power_law_integral(
            energy=up_e, norm_energy=norm_e, norm=n, index=idx)
        calc_res -= ppl.power_law_integral(
            energy=low_e, norm_energy=norm_e, norm=n, index=idx)

        npt.assert_allclose(calc_res, analytical_result)


if __name__ == '__main__':
    def all_tests():
        test_integration()
        test_different_bins()
    all_tests()
