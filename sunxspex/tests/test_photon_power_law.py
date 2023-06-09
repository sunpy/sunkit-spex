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

    npt.assert_allclose(flux1, flux2)


def test_integration():
    ''' test the actual power law integration '''
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


def test_empty_or_single_energy_edges():
    ''' edge case: not enough energy bins given '''
    for sz in (0, 1):
        edges = (np.arange(sz) + 10) << u.keV
        with npt.assert_raises(ValueError):
            ppl.broken_power_law_binned_flux(
                energy_edges=edges << u.keV, reference_energy=1 << u.keV,
                reference_flux=1 << ppl.PHOTON_RATE_UNIT, break_energy=3 << u.keV,
                lower_index=2, upper_index=3)


def test_one_big_bin():
    ''' edge case: single break energy in the middle of one big bin '''
    edges = [1, 10] << u.keV
    ref_eng = 1 << u.keV
    break_eng = 3 << u.keV
    ref_flux = 4 << ppl.PHOTON_RATE_UNIT
    li, ui = 2, 3

    ret = ppl.broken_power_law_binned_flux(
        energy_edges=edges, reference_energy=ref_eng,
        reference_flux=ref_flux, break_energy=break_eng,
        lower_index=li, upper_index=ui)

    un, ln = ppl.broken_power_law_normalizations(
        reference_flux=ref_flux,
        reference_energy=ref_eng,
        break_energy=break_eng,
        lower_index=li,
        upper_index=ui
    )

    manual_result = np.diff(
        ppl.power_law_integral(
            energy=u.Quantity([edges[0], break_eng]),
            norm_energy=ref_eng,
            norm=ln,
            index=li
        )
    )
    manual_result += np.diff(
        ppl.power_law_integral(
            energy=u.Quantity([break_eng, edges[1]]),
            norm_energy=ref_eng,
            norm=un,
            index=ui
        )
    )
    manual_result /= np.diff(edges)

    npt.assert_allclose(ret, manual_result)


def test_no_flux():
    ret = ppl.broken_power_law_binned_flux(
        energy_edges=[1, 2, 3] << u.keV, reference_energy=1 << u.keV,
        reference_flux=0 << ppl.PHOTON_RATE_UNIT, break_energy=3 << u.keV,
        lower_index=2, upper_index=3)
    npt.assert_allclose(ret, np.zeros_like(ret))


def test_single_power_law():
    eb1 = np.linspace(10, 100, num=30) << u.keV
    eb2 = np.linspace(10, 100, num=10) << u.keV

    shared_kw = dict(
        reference_energy=3 << u.keV,
        reference_flux=1 << ppl.PHOTON_RATE_UNIT,
        index=2,
    )

    dist1 = ppl.power_law_binned_flux(energy_edges=eb1, **shared_kw)
    dist2 = ppl.power_law_binned_flux(energy_edges=eb2, **shared_kw)

    flux1 = np.sum(dist1 * np.diff(eb1))
    flux2 = np.sum(dist2 * np.diff(eb2))

    npt.assert_allclose(flux1, flux2)


if __name__ == '__main__':
    def all_tests():
        test_integration()
        test_different_bins()

        test_empty_or_single_energy_edges()
        test_no_flux()
        test_one_big_bin()

        test_single_power_law()

    all_tests()
