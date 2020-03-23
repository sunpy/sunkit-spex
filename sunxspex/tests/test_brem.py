import numpy as np


from sunxspex import emission


def test_broken_powerlaw_dist():
    electron_energies = np.array([5, 10, 50, 150, 300, 500, 750, 1000])
    electron_distribution = emission.BrokenPowerLawElectronDistribution(p=3.0, q=5.0, eelow=3.0,
                                                                        eebrk=150.0, eehigh=10000.0)

    # brm2_distrn, [5, 10, 50, 150, 300, 500, 750, 1000], 3.0d, 5.0d, 3.0d, 150.0, 10000.0d, out
    # print, out
    res_idl_1 = [0.14402880576261082, 0.018003600720326352, 0.00014402880576261078,
                 5.3344002134300293e-06, 1.6670000666968841e-07, 1.2962592518634974e-08,
                 1.7070080682976095e-09, 4.0508101620734293e-10]
    np.array_equal(electron_distribution.flux(electron_energies), res_idl_1)

    # brm2_f_distrn, [5, 10, 50, 150, 300, 500, 750, 1000], 3.0d, 5.0d, 3.0d, 150.0, 10000.0d, out
    # print, out
    res_idl_2 = [0.35987197438839635, 0.089817963583501109, 0.0034006801259346183,
                 0.00020003999787660071, 1.2502490373201233e-05, 1.6203139378039724e-06,
                 3.2005388578040260e-07, 1.0126012702643657e-07]
    np.array_equal(electron_distribution.density(electron_energies), res_idl_2)


def test_brem_collisional_loss():
    photon_energies = np.array([1.0, 10.0, 100.0, 1000.0])
    res = emission.collisional_loss(photon_energies)
    # IDL code to generate values taken from dedt variable
    # Brm_ELoss, [1.0, 10.0, 100.0, 1000.0], dedt
    # print, dedt, format='(e0.116)'
    res_idl = [3.6275000635609979e+02, 1.2802678240553834e+02, 4.9735483535803510e+01,
               3.1420209156560023e+01]
    assert np.array_equal(res, res_idl)


def test_brem_cross_section():
    photon_energies = np.array([1.0, 10.0, 100.0, 1000.0])
    electron_energies = photon_energies+1
    res = emission.bremsstrahlung_cross_section(electron_energies, photon_energies)
    # IDL code to generate values taken from cross variable
    # Brm_BremCross, [1.0, 10.0, 100.0, 1000.0] + 1, [1.0, 10.0, 100.0, 1000.0], 1.2d, cross
    res_idl = [5.7397846333146957e-22, 4.3229680566443978e-24, 1.6053226238077600e-26,
               1.0327252400755748e-28]
    assert np.array_equal(res, res_idl)


def test_brem_thicktarget1():
    photon_energies = np.array([5, 10, 50, 150, 300, 500, 750, 1000], dtype=np.float64)
    res = emission.bremsstrahlung_thick_target(photon_energies, 5, 1000, 5, 10, 10000)
    assert np.all(res != 0)
    # IDL code to generate values taken from cross flux
    # flux = Brm2_ThickTarget([5, 10, 50, 150, 300, 500, 750, 1000], [1, 5,1000,5,10,10000])
    res_idl = [3.5282885128131238e-34, 4.7704601774538674e-35, 5.8706378555691385e-38,
               5.6778328842089976e-40, 3.1393035719304480e-41, 3.9809019377216963e-42,
               8.1224607804637566e-43, 2.6828147968651482e-43]
    assert np.allclose(res, res_idl, atol=0, rtol=1e-10)


def test_brem_thicktarget2():
    photon_energies = np.array([5, 10, 50, 150, 300, 500, 750, 1000], dtype=np.float64)
    res = emission.bremsstrahlung_thick_target(photon_energies, 3, 500, 6, 7, 10000)
    assert np.all(res != 0)
    # IDL code to generate values taken from cross flux
    #  flux = Brm2_ThickTarget([5, 10, 50, 150, 300, 500, 750, 1000], [1, 3, 500,6 , 7, 10000])
    res_idl = [4.5046333783173458e-34, 1.0601497794899769e-34, 2.7461522370645206e-36,
               1.4308656107380640e-37, 1.2640369923349261e-38, 1.1767820042098684e-39,
               1.5920587162709726e-40, 3.9685830064085143e-41]
    assert np.allclose(res, res_idl, atol=0, rtol=1e-10)


def test_brem_thintarget1():
    photon_energies = np.array([5, 10, 50, 150, 300, 500, 750, 1000], dtype=np.float64)
    res = emission.bremsstrahlung_thin_target(photon_energies, 5, 1000, 5, 10, 10000)
    assert np.all(res != 0)
    # IDL code to generate values taken from cross flux
    # flux = Brm2_ThinTarget([5, 10, 50, 150, 300, 500, 750, 1000], [1, 5, 1000, 5, 10, 10000])
    res_idl = [1.3792306669225426e-53, 3.2319324672606256e-54, 1.8906418622815277e-58,
               2.7707947605222644e-61, 5.3706858279023008e-63, 3.4603542191953094e-64,
               4.3847578461300751e-65, 1.0648152240531652e-65]
    assert np.allclose(res, res_idl, atol=0, rtol=1e-10)


def test_brem_thintarget2():
    photon_energies = np.array([5, 10, 50, 150, 300, 500, 750, 1000], dtype=np.float64)
    res = emission.bremsstrahlung_thin_target(photon_energies, 3, 200, 6, 7, 10000)
    assert np.all(res != 0)
    # IDL code to generate values taken from cross flux
    # flux = Brm2_ThinTarget([5, 10, 50, 150, 300, 500, 750, 1000], [1, 3, 200, 6, 7, 10000])
    res_idl = [1.410470406773663e-53, 1.631245131596281e-54, 2.494893311659408e-57,
               2.082487752231794e-59, 2.499983876763298e-61, 9.389452475896879e-63,
               7.805504370370804e-64, 1.414135608438244e-64]
    assert np.allclose(res, res_idl, atol=0, rtol=1e-10)
