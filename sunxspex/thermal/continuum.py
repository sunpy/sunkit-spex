import astropy.units as u
import numpy as np
from scipy import stats
from scipy import interpolate
from sunpy.data import manager

from sunxspex.io import load_chianti_continuum, load_xray_abundances


def define_continuum_parameters(filename=None):
    """
    Define continuum intensities as a function of temperature.

    Intensities are set as global variables and used in
    calculation of spectra by other functions in this module. They are in
    units of per volume emission measure at source, i.e. they must be
    divided by 4 * pi R**2 to be converted to physical values where
    R**2 is observer distance.

    Intensities are derived from output from the CHIANTI atomic physics database.
    The default CHIANTI data used here is collected from
    `https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_cont_1_250_v71.sav`.
    This includes contributions from thermal bremsstrahlung and tw-photon interactions.
    To use a different file, provide the URL/file location via the filename kwarg,
    e.g. to include only thermal bremsstrahlung, set the filename kwarg to 
    'https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_cont_1_250_v70_no2photon.sav'

    Parameters
    ----------
    filename: `str` (optional)
        URL or file location of the CHIANTI IDL save file to be used.
    """
    global CONTINUUM_INTENSITY_PER_EM_AT_SOURCE
    if filename:
        with manager.override_file("chianti_continuum", uri=filename):
            CONTINUUM_INTENSITY_PER_EM_AT_SOURCE = load_chianti_continuum()
    else:
        CONTINUUM_INTENSITY_PER_EM_AT_SOURCE = load_chianti_continuum()


def define_default_abundances(filename=None):
    """
    Read default abundance values into global variable.

    By default, data is read from the following file:
    https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/xray_abun_file.genx
    To load data from a different file, see Notes section.

    Parameters
    ----------
    filename: `str` (optional)
        URL or file location of the .genx abundance file to be used.
    """
    global DEFAULT_ABUNDANCES
    if filename:
        with manager.override_file("xray_abundance", uri=filename):
            DEFAULT_ABUNDANCES = load_xray_abundances()
    else:
        DEFAULT_ABUNDANCES = load_xray_abundances()


define_continuum_parameters()
define_default_abundances()


@u.quantity_input(energy_edges=u.keV,
                  temperature=u.K,
                  emission_measure=(u.cm**(-3), u.cm**(-5)),
                  observer_distance=u.cm)
def continuum_emission(energy_edges,
                       temperature,
                       emission_measure,
                       abundance_type="sun_coronal",
                       relative_abundances=None,
                       observer_distance=(1*u.AU).to(u.cm)):
    # Define shortcuts to global variables and derive some useful parameters from them.
    T_grid = (CONTINUUM_INTENSITY_PER_EM_AT_SOURCE.temperature.data *
              CONTINUUM_INTENSITY_PER_EM_AT_SOURCE.attrs["units"]["temperature"])

    _LOGT = np.log10(CONTINUUM_INTENSITY_PER_EM_AT_SOURCE.temperature.data)
    _CONT_ELEMENT_IDX = CONTINUUM_INTENSITY_PER_EM_AT_SOURCE.element_index.data
    E_grid_keV = (CONTINUUM_INTENSITY_PER_EM_AT_SOURCE.wavelength.data *
                  CONTINUUM_INTENSITY_PER_EM_AT_SOURCE.attrs["units"]["wavelength"]).to(
                      u.keV, equivalencies=u.spectral()).value
    wave_edges_grid_AA = CONTINUUM_INTENSITY_PER_EM_AT_SOURCE.attrs["wavelength_edges"].to_value(u.AA)
    dwave_grid_AA = wave_edges_grid_AA[1:] - wave_edges_grid_AA[:-1]

    # Handle inputs and derive some useful parameters from them
    energy_edges_keV = np.zeros((2, len(energy_edges)-1))
    energy_edges_keV[0] = energy_edges[:-1].to_value(u.keV)
    energy_edges_keV[1] = energy_edges[1:].to_value(u.keV)
    energy_keV_gmean = stats.gmean(energy_edges_keV)  # Get energy bins centres based on geometric mean.
    logT_in = np.log10(temperature.to_value(u.K))

    # Find bin in temperature grid containing the input temperature and
    # the bins above and below it.
    selt = np.digitize(logT_in, _LOGT) - 1  #TODO: Extend this so function works for multiple temperatures
    indx = selt - 1 + np.arange(3)
    tband = _LOGT[indx]
    x0 = np.log(tband[0])
    x1 = np.log(tband[1])
    x2 = np.log(tband[2])
    # Extract continuum intensity grid for above temperature bins.
    tcdbase = CONTINUUM_INTENSITY_PER_EM_AT_SOURCE.data[:, indx, :]

    #####  Calculate Abundances #####
    # Calculate abundance of each desired element.
    default_abundances = DEFAULT_ABUNDANCES[abundance_type].data
    n_abundances = len(default_abundances)
    rel_abund_values = np.ones(n_abundances)
    if relative_abundances:
        # Convert input relative abundances to array where
        # first axis is atomic number, i.e == index + 1
        # Second axis is relative abundance value.
        rel_abund_array = np.array(relative_abundances).T
        rel_idx = np.rint(rel_abund_array[0]).astype(int) - 1
        rel_abund_values[rel_idx] = rel_abund_array[1]
    abundance_mask = np.zeros(n_abundances, dtype=bool)
    abundance_mask[_CONT_ELEMENT_IDX] = True
    abundances = default_abundances * rel_abund_values * abundance_mask

    # Calculate abundance-scaled continuum intensity as a function of temp. and energy
    tcdbase = CONTINUUM_INTENSITY_PER_EM_AT_SOURCE[:, indx].data
    tcd = np.zeros(tcdbase.shape[1:])
    sorted_element_index = np.sort(_CONT_ELEMENT_IDX)  #TODO: benchmark sort versus checking where abundace > 0
    for i in range(0,3):
        tcd[i] = np.matmul(abundances[sorted_element_index], tcdbase[:, i])

    repeat_E_grid_keV = np.repeat(E_grid_keV[np.newaxis, :], 3, axis=0)
    repeat_T_grid = np.repeat(T_grid[indx, np.newaxis], len(E_grid_keV), axis=1)
    exponential = repeat_E_grid_keV / repeat_T_grid.to_value(u.keV, equivalencies=u.temperature_energy())
    exponential = np.exp(np.clip(exponential, None, 80))

    # Calculate width of each spectral bin in energy using fraction of width in wavelength to central wavelength as a scaling factor.
    dE_grid_keV = E_grid_keV * dwave_grid_AA / CONTINUUM_INTENSITY_PER_EM_AT_SOURCE.wavelength.data
    dE_grid_keV = np.repeat(dE_grid_keV[np.newaxis, :], 3, axis=0)
    gaunt = tcd / dE_grid_keV * exponential

    tcont = np.zeros(energy_keV_gmean.shape)

    # Define valid range
    vrange, = np.where(gaunt[0] > 0)
    vrange1, = np.where(gaunt[1] > 0)
    if len(vrange) < len(vrange1):
        vrange = vrange1
    vrange1, = np.where(gaunt[2] > 0)
    if len(vrange) < len(vrange1):
        vrange = vrange1
    gaunt = gaunt[:, vrange]
    E_grid_keV = E_grid_keV[vrange]
    maxE = E_grid_keV[0]
    vgmean, = np.where(energy_keV_gmean < maxE)

    if len(vgmean) > 0:
        energy_keV_gmean = energy_keV_gmean[vgmean]
        cont0 = interpolate.interp1d(E_grid_keV, gaunt[0])(energy_keV_gmean)
        cont1 = interpolate.interp1d(E_grid_keV, gaunt[1])(energy_keV_gmean)
        cont2 = interpolate.interp1d(E_grid_keV, gaunt[2])(energy_keV_gmean)
        # Get the geometric weighted mean of the interpolated values between the three temperature bins.
        logeT_in = np.log(logT_in)
        cont0 = np.log(cont0)
        cont1 = np.log(cont1)
        cont2 = np.log(cont2)
        ynew  = np.exp(
            cont0 * (logeT_in - x1) * (logeT_in - x2) / ((x0 - x1) * (x0 - x2)) +
            cont1 * (logeT_in - x0) * (logeT_in - x2) / ((x1 - x0) * (x1 - x2)) +
            cont2 * (logeT_in - x0) * (logeT_in - x1) / ((x2 - x0) * (x2 - x1)) )

        tcont[vgmean] += ynew
        tcont *= np.exp(-(energy_keV_gmean / temperature.to_value(u.keV, equivalencies=u.temperature_energy())))
        spectrum = tcont * (energy_edges[1:] - energy_edges[:-1])

    dE_in = energy_edges[1:] - energy_edges[:-1]
    return (spectrum * CONTINUUM_INTENSITY_PER_EM_AT_SOURCE.attrs["units"]["data"] *
            emission_measure /
            (dE_in * observer_distance**2 / u.sr))
