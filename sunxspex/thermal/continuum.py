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
    global _CONT_T_K,_CONT_LOG10T, _CONT_T_KEV, _CONT_ELEMENT_IDX, _CONT_SORTED_ELEMENT_INDEX, wavelength, _CONT_E_KEV, _CONT_SPECTRAL_BIN_WIDTHS_KEV, _CONT_INTENSITY_PER_EM_AT_SOURCE, _CONT_INTENSITY_UNIT
    if filename:
        with manager.override_file("chianti_continuum", uri=filename):
            cont_info = load_chianti_continuum()
    else:
        cont_info = load_chianti_continuum()
    _CONT_ELEMENT_IDX = cont_info.element_index.data
    _CONT_SORTED_ELEMENT_INDEX = np.sort(_CONT_ELEMENT_IDX)

    T_grid = cont_info.temperature.data * cont_info.attrs["units"]["temperature"]
    _CONT_T_K = T_grid.to_value(u.K)
    _CONT_LOG10T = np.log10(_CONT_T_K)
    _CONT_T_KEV = T_grid.to_value(u.keV, equivalencies=u.temperature_energy())

    wavelength = cont_info.wavelength.data * cont_info.attrs["units"]["wavelength"]
    dwave_AA = (cont_info.attrs["wavelength_edges"][1:] -
                cont_info.attrs["wavelength_edges"][:-1]).to_value(u.AA)
    _CONT_E_KEV = wavelength.to_value(u.keV, equivalencies=u.spectral())
    _CONT_SPECTRAL_BIN_WIDTHS_KEV = _CONT_E_KEV * dwave_AA / wavelength.to_value(u.AA)

    _CONT_INTENSITY_PER_EM_AT_SOURCE = cont_info.data * 4 * np.pi  # Convert from per sterradian to emission at source.
    _CONT_INTENSITY_UNIT = cont_info.attrs["units"]["data"] * u.sr  # Remove per sterradian in accordance with above scaling.


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
    # Handle inputs and derive some useful parameters from them
    log10T_in = np.log10(temperature.to_value(u.K))
    T_in_keV = temperature.to_value(u.keV, equivalencies=u.temperature_energy())
    energy_edges_keV = energy_edges.to_value(u.keV)
    # Get energy bins centers based on geometric mean.
    energy_gmean_keV = stats.gmean(np.vstack((energy_edges_keV[:-1], energy_edges_keV[1:])))

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

    #####  Calculate Continuum Intensity Summed Over All ELements
    #####  As A Function of Temperature and Energy/Wavelength ######
    # Define a temperature band as the bin containing the input temperature and
    # the bins above and below it.  Find the indices of the temperature band.
    selt = np.digitize(log10T_in, _CONT_LOG10T) - 1  #TODO: Extend this so function works for multiple temperatures
    tband_idx = selt - 1 + np.arange(3)
    n_tband = len(tband_idx)
    # Calculate continuum intensity summed over all elements as a function of energy/wavelength
    # and temperature over the temperature band.
    element_intensities_per_em_at_source = _CONT_INTENSITY_PER_EM_AT_SOURCE[:, tband_idx]
    intensity_per_em_at_source = np.zeros(element_intensities_per_em_at_source.shape[1:])
    for i in range(0, n_tband):
        intensity_per_em_at_source[i] = np.matmul(
            abundances[_CONT_SORTED_ELEMENT_INDEX],
            element_intensities_per_em_at_source[:, i])

    ##### Calculate Continuum Intensity at Input Temperature  ######
    ##### Do this by interpolating the normalized temperature component
    ##### of the intensity grid to input temperature(s) and then rescaling.
    # Calculate normalized temperature component of the intensity grid.
    exponent = (np.repeat(_CONT_E_KEV[np.newaxis, :], n_tband, axis=0) /
                np.repeat(_CONT_T_KEV[tband_idx, np.newaxis], len(_CONT_E_KEV), axis=1))
    exponential = np.exp(np.clip(exponent, None, 80))
    dE_grid_keV = np.repeat(_CONT_SPECTRAL_BIN_WIDTHS_KEV[np.newaxis, :], n_tband, axis=0)
    gaunt = intensity_per_em_at_source / dE_grid_keV * exponential
    # Interpolate the normalized temperature component of the intensity grid the the
    # input temperature.
    spectrum = _interpolate_continuum_intensities(gaunt, _CONT_LOG10T[tband_idx], _CONT_E_KEV,
                                                  energy_gmean_keV, log10T_in)
    # Rescale the interpolated intensity.
    spectrum *= np.exp(-(energy_gmean_keV / T_in_keV))

    # Put intensity into correct units and scale by input emission measure and observer distance.
    return spectrum * _CONT_INTENSITY_UNIT * emission_measure / (4 * np.pi * observer_distance**2)


def _interpolate_continuum_intensities(data_grid, log10T_grid, energy_grid_keV, energy_keV, log10T):
    # Determine valid range based on limits of intensity grid's spectral extent
    # and the normalized temperature component of intensity.
    n_tband = len(log10T_grid)
    vrange, = np.where(data_grid[0] > 0)
    for i in range(1, n_tband):
        vrange_i, = np.where(data_grid[i] > 0)
        if len(vrange) < len(vrange_i):
            vrange = vrange_i
    data_grid = data_grid[:, vrange]
    energy_grid_keV = energy_grid_keV[vrange]
    energy_idx, = np.where(energy_keV < energy_grid_keV.max())

    # Interpolate temperature component of intensity and derive continuum intensity.
    spectrum = np.zeros(energy_keV.shape)
    if len(energy_idx) > 0:
        energy_keV = energy_keV[energy_idx]
        cont0 = interpolate.interp1d(energy_grid_keV, data_grid[0])(energy_keV)
        cont1 = interpolate.interp1d(energy_grid_keV, data_grid[1])(energy_keV)
        cont2 = interpolate.interp1d(energy_grid_keV, data_grid[2])(energy_keV)
        # Calculate the continuum intensity as the weighted geometric mean
        # of the interpolated values across the temperature band of the
        # temperature component of intensity.
        logelog10T = np.log(log10T)
        x0, x1, x2 = np.log(log10T_grid)
        spectrum[energy_idx]  = np.exp(
            np.log(cont0) * (logelog10T - x1) * (logelog10T - x2) / ((x0 - x1) * (x0 - x2)) +
            np.log(cont1) * (logelog10T - x0) * (logelog10T - x2) / ((x1 - x0) * (x1 - x2)) +
            np.log(cont2) * (logelog10T - x0) * (logelog10T - x1) / ((x2 - x0) * (x2 - x1)) )
    return spectrum
