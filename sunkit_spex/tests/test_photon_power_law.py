import numpy as np
import numpy.testing as npt

import astropy.units as u

from sunkit_spex import photon_power_law as ppl


def test_different_bins():
    ''' make sure that flux is conserved when using different energy bins '''
    energy_bins1 = np.linspace(10, 100, num=30) << u.keV
    energy_bins2 = np.linspace(10, 100, num=10) << u.keV

    shared_kw = dict(
        norm_energy=3 << u.keV,
        norm_flux=1 << ppl.PHOTON_RATE_UNIT,
        break_energy=20 << u.keV,
        lower_index=1,
        upper_index=3
    )

    dist1 = ppl.compute_broken_power_law(energy_edges=energy_bins1, **shared_kw)
    dist2 = ppl.compute_broken_power_law(energy_edges=energy_bins2, **shared_kw)

    flux1 = np.sum(dist1 * np.diff(energy_bins1))
    flux2 = np.sum(dist2 * np.diff(energy_bins2))

    npt.assert_allclose(flux1, flux2)


def test_integration():
    ''' test the actual power law integration '''
    for idx in range(10):
        norm = 1 << ppl.PHOTON_RATE_UNIT
        norm_e = 3 << u.keV
        up_e, low_e = 4 << u.keV, 2 << u.keV

        if idx != 1:
            analytical_result = (
                norm * norm_e / (1 - idx) * (
                    (up_e / norm_e)**(1 - idx) - (low_e / norm_e)**(1 - idx)
                )
            )
        else:
            analytical_result = norm * norm_e * np.log(up_e / low_e)

        calc_res = ppl.integrate_power_law(
            energy=up_e, norm_energy=norm_e, norm=norm, index=idx)
        calc_res -= ppl.integrate_power_law(
            energy=low_e, norm_energy=norm_e, norm=norm, index=idx)

        npt.assert_allclose(calc_res, analytical_result)


def test_empty_or_single_energy_edges():
    ''' edge case: not enough energy bins given '''
    for size in (0, 1):
        edges = (np.arange(size) + 10) << u.keV
        with npt.assert_raises(ValueError):
            ppl.compute_broken_power_law(
                energy_edges=edges << u.keV, norm_energy=1 << u.keV,
                norm_flux=1 << ppl.PHOTON_RATE_UNIT, break_energy=3 << u.keV,
                lower_index=2, upper_index=3)


def test_one_big_bin():
    ''' edge case: single break energy in the middle of one big bin '''
    edges = [1, 10] << u.keV
    ref_eng = 1 << u.keV
    break_eng = 3 << u.keV
    ref_flux = 4 << ppl.PHOTON_RATE_UNIT
    lower_index, upper_index = 2, 3

    ret = ppl.compute_broken_power_law(
        energy_edges=edges, norm_energy=ref_eng,
        norm_flux=ref_flux, break_energy=break_eng,
        lower_index=lower_index, upper_index=upper_index)

    upper_norm, lower_norm = ppl._compute_broken_power_law_normalizations(
        norm_flux=ref_flux,
        norm_energy=ref_eng,
        break_energy=break_eng,
        lower_index=lower_index,
        upper_index=upper_index
    )

    manual_result = np.diff(
        ppl.integrate_power_law(
            energy=u.Quantity([edges[0], break_eng]),
            norm_energy=ref_eng,
            norm=lower_norm,
            index=lower_index
        )
    )
    manual_result += np.diff(
        ppl.integrate_power_law(
            energy=u.Quantity([break_eng, edges[1]]),
            norm_energy=ref_eng,
            norm=upper_norm,
            index=upper_index
        )
    )
    manual_result /= np.diff(edges)

    npt.assert_allclose(ret, manual_result)


def test_no_flux():
    ret = ppl.compute_broken_power_law(
        energy_edges=[1, 2, 3] << u.keV, norm_energy=1 << u.keV,
        norm_flux=0 << ppl.PHOTON_RATE_UNIT, break_energy=3 << u.keV,
        lower_index=2, upper_index=3)
    npt.assert_allclose(ret, np.zeros_like(ret))


def test_single_power_law():
    energy_bins1 = np.linspace(10, 100, num=30) << u.keV
    energy_bins2 = np.linspace(10, 100, num=10) << u.keV

    shared_kw = dict(
        norm_energy=3 << u.keV,
        norm_flux=1 << ppl.PHOTON_RATE_UNIT,
        index=2,
    )

    dist1 = ppl.compute_power_law(energy_edges=energy_bins1, **shared_kw)
    dist2 = ppl.compute_power_law(energy_edges=energy_bins2, **shared_kw)

    flux1 = np.sum(dist1 * np.diff(energy_bins1))
    flux2 = np.sum(dist2 * np.diff(energy_bins2))

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
