import numpy as np


from sunxspex import emission


def test_broken_power_law_electron_distribution():
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


def test_get_integrand():
    photon_energies = np.array([1.0, 10.0, 100.0, 1000.0])
    electron_energies = photon_energies + 1
    params = {'electron_energy': electron_energies,
              'photon_energy': photon_energies,
              'eelow': 1.0,
              'eebrk': 150.0,
              'eehigh': 1000.0,
              'p': 5.0,
              'q': 7.0,
              'z': 1.2}

    res_thick = emission.get_integrand(model='thick-target', **params)
    res_thin_efd = emission.get_integrand(model='thin-target', **params)
    res_thin_noefd = emission.get_integrand(model='thin-target', **params, efd=False)
    # IDL code to generate values
    # Brm2_Fouter([1.0, 10.0, 100.0, 1000.0] + 1, [1.0, 10.0, 100.0, 1000.0], 10.0d,  150.0d,
    # 1000.0d, 5.0d, 7.0d, 1.2d)
    res_idl_thick = [6.1083381554006209e-24, 2.5108068464643281e-28, 8.1522892779571421e-34,
                     0.0000000000000000]
    assert np.array_equal(res_thick, res_idl_thick)
    # IDL code to generate values
    # Brm2_FThin([1.0, 10.0, 100.0, 1000.0]+1.0d, [1.0, 10.0, 100.0, 1000.0], 1.0d,  150.0d,
    # 1000.0d, 5.0d, 7.0d, 1.2d, 1)
    res_idl_thin_efd = [1.2229040135854787e-30, 1.8300600988388983e-36, 1.0413631578198003e-43,
                        0.0000000000000000]
    assert np.array_equal(res_thin_efd, res_idl_thin_efd)
    # IDL code to generate values
    # Brm2_FThin([1.0, 10.0, 100.0, 1000.0]+1.0d, [1.0, 10.0, 100.0, 1000.0], 1.0d,  150.0d,
    # 1000.0d, 5.0d, 7.0d, 1.2d, 0)
    res_idl_thin_noefd = [3.2341903200820362e-21, 1.1203835558833694e-26, 1.7180070908135551e-33,
                          0.0000000000000000]
    assert np.array_equal(res_thin_noefd, res_idl_thin_noefd)


def test_integrate_part():
    eph = np.array([10.0, 20.0, 40.0, 80.0, 150.0])
    params = {'model': 'thin-target',
              'maxfcn': 2048,
              'rerr': 1e-4,
              'z': 1.2,
              'p': 5.0,
              'q': 7.0,
              'eebrk': 200,
              'eelow': 1.0,
              'eehigh': 200.0,
              'photon_energies': eph,
              'a_lg': np.log10(eph),
              'b_lg': np.full_like(eph, np.log10(200)),
              'll': [0, 1, 2, 3, 4],
              'efd': True}

    res_thin, _ = emission.integrate_part(**params)
    # IDL code to generate values - constructed so it only cover a singe continuous part
    # brm2_dmlin([10.0d, 20.0d, 40.0d, 80.0d, 150.0], [200.0d, 200.0d, 200.0d, 200.0d, 200.0d],
    # 2048, 1e-4, [10.0d, 20.0d, 40.0d, 80.0d, 150.0], 1.0, 200.0d, 200.0d, 5.0d, 7.0d, 1.2d, 1)
    # Brm2_ThinTarget([10.0d, 20.0d, 40.0d, 80.0d, 150.0], [1.0d, 5.0d, 200.0d, 7.0d, 1.0d, 200.0d])
    res_idl_thin = [7.9163611801477292e-36, 1.1718303039579161e-37, 1.7710210625358297e-39,
                    2.6699438088131420e-41, 3.6688281208262375e-43]
    assert np.allclose(res_thin, res_idl_thin, atol=0, rtol=1e-10)

    params['model'] = 'thick-target'
    res_thick, _ = emission.integrate_part(**params)
    # IDL code to generate values - constructed so it only cover a singe continuous part
    # out = dblarr(5)
    # IDL> ier = dblarr(5)
    # IDL> brm2_dmlino_int, 2048, 1e-4, [10.0d, 20.0d, 40.0d, 80.0d, 150.0], 1.0, 200.0d, 200.0d,
    # 5.0d, 7.0d, 1.2d, alog10([10.0d, 20.0d, 40.0d, 80.0d, 150.0]), alog10([200.0d, 200.0d,
    # 200.0d, 200.0d, 200.0d]), [0, 1, 2, 3, 4], out, ier
    # print, out
    res_idl_thick = [1.7838076641732560e-27, 9.9894296899783751e-29, 5.2825655485310581e-30,
                     2.1347233135651843e-31, 2.9606798379782830e-33]
    assert np.allclose(res_thick, res_idl_thick, atol=0, rtol=1e-10)


def test_split_and_integrate():
    photon_energies = np.array([5, 10, 50, 150, 300, 500, 750, 1000], dtype=np.float64)
    params = {
        'model': 'thick-target',
        'photon_energies': photon_energies,
        'maxfcn': 2048,
        'rerr': 1e-4,
        'eelow': 10.0,
        'eebrk': 500.0,
        'eehigh': 1000.0,
        'p': 5.0,
        'q': 7.0,
        'z': 1.2,
        'efd': True
    }

    res_thick = emission.split_and_integrate(**params)
    params['model'] = 'thin-target'
    res_thin = emission.split_and_integrate(**params)
    # IDL code to generate values
    # Brm2_DmlinO, [5.0d, 10.0d, 50.0d, 150.0d, 300.0d, 500.0d, 750.0d, 1000.0d], $
    # [10000.0d, 10000.0d, 10000.0d, 10000.0d, 10000.0d, 10000.0d, 10000.0d, 10000.000], 2048, $
    # 1e-4, [5.0d, 10.0d, 50.0d, 150.0d, 300.0d, 500.0d, 750.0d, 1000.0d], $
    # 10.0d, 500.0d, 10000.000, 5.0d, 7.0d, 1.2d
    res_idl_thick = [2.2515963597937766e-30, 3.0443768835060635e-31, 3.7297617183535930e-34,
                     3.3249346002544473e-36, 1.2639820076797306e-37, 6.9618972578770903e-39,
                     6.2232553253478841e-40, 1.1626170413561901e-40]
    np.allclose(res_thick, res_idl_thick, atol=0, rtol=1e-10)
    # Brm2_Dmlin( [5.0d, 10.0d, 50.0d, 150.0d, 300.0d, 500.0d, 750.0d, 1000.0d], $
    # [10000.0d, 10000.0d, 10000.0d, 10000.0d, 10000.0d, 10000.0d, 10000.0d, 10000.000], 2048, $
    # 1e-4, [5.0d, 10.0d, 50.0d, 150.0d, 300.0d, 500.0d, 750.0d, 1000.0d], $
    # 10.0d, 500.0d, 10000.000, 5.0d, 7.0d, 1.2d, 1, ier2)
    res_idl_thin = [3.3783188543550312e-31, 7.9163900936296962e-32, 4.6308051717377875e-36,
                    6.7446378797836118e-39, 1.1835467661486555e-40, 4.3612071662677236e-42,
                    2.3851948333528724e-43, 3.2244753187594343e-44]
    np.allclose(res_thin, res_idl_thin, atol=0, rtol=1e-10)


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
