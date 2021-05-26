import os.path
import copy

import numpy as np
import astropy.units as u
import scipy.interpolate
import scipy.stats
import sunpy.coordinates

import xarray
from astropy.table import Table

from sunxspex.io import load_chianti_lines_lite, load_chianti_continuum, load_xray_abundances

__all__ = ['ChiantiThermalSpectrum']


def define_line_parameters(filename=None):
    """Define line intensities as a function of temperature for calculating line emission.

    Line intensities are set as global variables and used in the
    calculation of spectra by other functions in this module. They are in
    units of per unit emission measure at source, i.e. they must be
    divided by 4 pi R**2 (where R is the observer distance) and
    multiplied by emission measure to be converted to physical values at the observer.

    Line intensities are derived from output from the CHIANTI atomic
    physics database. The default CHIANTI data used here is collected from
    `https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav`.
    To use a different file, provide the URL/file location via the filename kwarg.

    Parameters
    ----------
    filename: `str` (optional)
        URL or file location of the CHIANTI IDL save file to be used.
    """
    global _LINE_INTENSITY_PER_EM_AT_SOURCE, _LINE_INTENSITY_UNIT, _LINE_PEAKS_KEV, _LINE_LOGT, \
            _LINE_ELEMENT_IDX, _LINE_ATOMIC_NUMBERS
    if filename:
        with manager.override_file("chianti_lines", uri=filename):
            line_info = load_chianti_lines_lite()
    else:
        line_info = load_chianti_lines_lite()
    _LINE_INTENSITY_PER_EM_AT_SOURCE = np.array(line_info.data)
    _LINE_INTENSITY_UNIT = line_info.attrs["units"]["data"]
    _LINE_PEAKS_KEV = (line_info.peak_energy.data * line_info.attrs["units"]["peak_energy"]).to_value(
        u.keV, equivalencies=u.spectral())
    _LINE_LOGT = line_info.logT.data
    _LINE_ELEMENT_IDX = line_info.attrs["element_index"]
    _LINE_ATOMIC_NUMBERS = line_info.atomic_number.data


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


# Read line, continuum and abundance data into global variables.
define_line_parameters()
define_continuum_parameters()
define_default_abundances()


def line(energy_edges, temperature, emission_measure,
         abundance_type="sun_coronal", relative_abundances=None,
         observer_distance=(1*u.AU).to(u.cm)):
    """
    Calculate thermal line emission from the solar corona.

    The emission is calculated as a function of temperature and emission measure.

    Parameters
    ----------
    energy_edges: `astropy.units.Quantity`
        The edges of the energy bins in a 1D N+1 quantity.

    temperature: `astropy.units.Quantity`
        The temperature of the plasma.
        Can be scalar or 1D of any length. If not scalar, the flux for each temperature
        will be calculated. The first dimension of the output flux will correspond
        to temperature.

    emission_measure: `astropy.units.Quantity`
        The emission measure of the plasma at each temperature.
        Must be same length as temperature or scalar.

    abundance_type: `str` (optional)
        Abundance type to use.  Options are:
            1. cosmic
            2. sun_coronal - default abundance
            3. sun_coronal_ext
            4. sun_hybrid
            5. sun_hybrid_ext
            6. sun_photospheric
            7. mewe_cosmic
            8. mewe_solar
        The values for each abundance type is stored in the global
        variable DEFAULT_ABUNDANCES which is generated by `define_default_abundances`
        function. To load different default values for each abundance type,
        see the docstring of that function.

    relative_abundances: 2xN array-like
        The relative abundances of different elements as a fraction of their
        default abundances defined by abundance_type.
        The first axis represents the atomic number of the element.
        The second axis gives the factor by which to scale the element's abundance.

    observer_distance: `astropy.units.Quantity` (Optional)
        The distance between the source and the observer. Scales output to observer distance
        and unit by 1/length.
        Default=1 AU.

    Returns
    -------
    flux: `astropy.units.Quantity`
        The photon flux as a function of temperature and energy.

    Intensity Units

    The line intensities read from the CHIANTI file are in units of ph / cm**2 / s / sr.
    Therefore they are specific intensities, i.e. per steradian, or solid angle.
    Here, let us call these intensities, intensity_per_solid_angle.
    The solid angle is given by flare_area / observer_distance**2.
    Total integrated intensity can be rewritten in terms of volume EM and solid angle:
    intensity = intensity_per_solid_angle_per_volEM * volEM * solid_angle
              = intensity_per_solid_angle / (colEM * flare_area) * (flare_area / observer_dist**2) * volEM
              = intensity_per_solid_angle / colEM / observer_dist**2 * volEM
    i.e. flare area cancels. Therefore:
    intensity = intensity_per_solid_angle / colEM / observer_dist**2 * volEM,
    or, dividing both sides by volEM,
    intensity_per_volEM = intensity_per_solid_angle / colEM / observer_dist**2
    """
    # For ease of calculation, convert inputs to known units and structures.
    energy_edges_keV = energy_edges.to_value(u.keV)
    n_energy_bins = len(energy_edges_keV)-1
    temperature_K = temperature.to_value(u.K)
    if temperature.isscalar:
        temperature_K = np.array([temperature_K])
    n_temperatures = len(temperature_K)

    # Find indices of lines within user input energy range.
    energy_roi_indices = np.logical_and(_LINE_PEAKS_KEV >= energy_edges_keV.min(),
                                        _LINE_PEAKS_KEV <= energy_edges_keV.max())
    n_energy_roi_indices = energy_roi_indices.sum()
    # If there are line within energy range of interest, compile spectrum.
    if n_energy_roi_indices > 0:
        #####  Calculate Abundances #####
        # Calculate abundance of each desired element.
        default_abundances = DEFAULT_ABUNDANCES[abundance_type].data
        n_abundances = len(default_abundances)
        rel_abund_values = np.ones(n_abundances)
        if relative_abundances:
            # First axis of relative_abundances is atomic number, i.e == index + 1
            # Second axis is relative abundance value.
            rel_abund_values[relative_abundances[0]-1] = relative_abundances[1]
        abundance_mask = np.zeros(n_abundances, dtype=bool)
        abundance_mask[_LINE_ELEMENT_IDX] = True
        abundances = default_abundances * rel_abund_values * abundance_mask
        # Extract only lines within energy range of interest.
        line_abundances = abundances[_LINE_ATOMIC_NUMBERS[energy_roi_indices] - 2]
        # Above magic number of of -2 is comprised of:
        # a -1 to account for the fact that index is atomic number -1, and
        # another -1 because abundance index is offset from element index by 1.

        ##### Calculate Line Intensities in Input Energy Range #####
        # Calculate abundance-normalized intensity of each line in energy range of
        # interest as a function of energy and temperature.
        line_intensity_grid = _LINE_INTENSITY_PER_EM_AT_SOURCE[energy_roi_indices]
        line_intensities = _calculate_abundance_normalized_line_intensities(
            np.log10(temperature_K), line_intensity_grid, _LINE_LOGT)
        # Scale line intensities by abundances to get true line intensities.
        line_intensities *= line_abundances

        ##### Weight Line Emission So Peak Energies Maintained Within Input Energy Binning #####
        # Split emission of each line between nearest neighboring spectral bins in
        # proportion such that the line centroids appear at the correct energy
        # when average over neighboring bins.
        # This has the effect of appearing to double the number of lines as regards
        # the dimensionality of the line_intensities array.
        line_peaks_keV = _LINE_PEAKS_KEV[energy_roi_indices]
        split_line_intensities, line_spectrum_bins = _weight_emission_bins_to_line_centroid(
            line_peaks_keV, energy_edges_keV, line_intensities)
   
        #### Calculate Flux #####
        # Use binned_statistic to determine which spectral bins contain
        # components of line emission and sum over those line components
        # to get the total emission is each spectral bin.
        flux = scipy.stats.binned_statistic(line_spectrum_bins, split_line_intensities, "sum", n_energy_bins, (0, n_energy_bins-1)).statistic
    else:
        flux = np.zeros((n_temperatures, n_energy_bins))
    
    # Scale flux by observer distance, emission measure and spectral bin width
    # and put into correct units.
    energy_bin_widths = (energy_edges_keV[1:] - energy_edges_keV[:-1]) * u.keV
    flux = (flux * _LINE_INTENSITY_UNIT * emission_measure / (energy_bin_widths * 4 * np.pi * observer_distance**2))
    if temperature.isscalar and emission_measure.isscalar:
        flux = flux[0]

    return flux

def _calculate_abundance_normalized_line_intensities(logT, data_grid, line_logT_bins):
    """
    Calculates normalized line intensities at a given temperature using interpolation.

    Given a 2D array, say of line intensities, as a function of two parameters,
    say energy and log10(temperature), and a log10(temperature) value,
    interpolate the line intensities over the temperature axis and
    extract the intensities as a function of energy at the input temperature.

    Note that strictly speaking the code is agnostic to the physical properties
    of the axes and values in the array. All the matters is that data_grid
    is interpolated over the 2nd axis and the input value also corresponds to
    somewhere along that same axis. That value does not have exactly correspond to
    the value of a column in the grid. This is accounted for by the interpolation.

    Parameters
    ----------
    logT: 1D `numpy.ndarray` of `float`.
        The input value along the 2nd axis at which the line intensities are desired.
        If multiple values given, the calculation is done for each and the
        output array has an extra dimension.

    data_grid: 2D `numpy.ndarray`
        Some property, e.g. line intensity, as function two parameters,
        e.g. energy (0th dimension) and log10(temperature in kelvin) (1st dimension).

    line_logT_bins: 1D `numpy.ndarray`
        The value along the 2nd axis at which the data are required,
        say a value of log10(temperature in kelvin).

    Returns
    -------
    interpolated_data: 1D or 2D `numpy.ndarray`
        The line intensities as a function of energy (1st dimension) at
        each of the input temperatures (0th dimension).
        Note that unlike the input line intensity table, energy here is the 0th axis.
        If there is only one input temperature, interpolated_data is 1D.

    """
    # Ensure input temperatures are in an array to consistent manipulation.
    n_temperatures = len(logT)

    # Get bins in which input temperatures belong.
    temperature_bins = np.digitize(logT, line_logT_bins)-1

    # For each input "temperature", interpolate the grid over the 2nd axis
    # using the bins corresponding to the input "temperature" and the two neighboring bins.
    # This will result in a function giving the data as a function of the 1st axis,
    # say energy, at the input temperature to sub-temperature bin resolution.
    interpolated_data = np.zeros((n_temperatures, data_grid.shape[0]))
    for i in range(n_temperatures):
        # Indentify the "temperature" bin to which the input "temperature"
        # corresponds and its two nearest neighbors.
        indx = temperature_bins[i]-1+np.arange(3)
        # Interpolate the 2nd axis to produce a function that gives the data
        # as a function of 1st axis, say energy, at a given value along the 2nd axis,
        # say "temperature".
        get_intensities_at_logT = scipy.interpolate.interp1d(line_logT_bins[indx], data_grid[:, indx], kind="quadratic")
        # Use function to get interpolated_data as a function of the first axis at
        # the input value along the 2nd axis,
        # e.g. line intensities as a function of energy at a given temperature.
        interpolated_data[i, :] = get_intensities_at_logT(logT[i]).squeeze()[:]

    return interpolated_data


def _weight_emission_bins_to_line_centroid(line_peaks_keV, energy_edges_keV, line_intensities):
    """
    Split emission between neighboring energy bins such that averaged energy is the line peak.

    Given line peak energies and a set of the energy bin edges:
    1. Find the bins into which each of the lines belong.
    2. Calculate distance between the line peak energy and the
    center of the bin to which it corresponds as a fraction of the distance between
    the bin center the center of the next closest bin to the line peak energy.
    3. Assign the above fraction of the line intensity to the neighboring bin and
    the rest of the energy to the original bin.
    4. Add the neighboring bins to the array of bins containing positive emission.

    Parameters
    ----------
    line_peaks_keV: 1D `numpy.ndarray`
        The energy of the line peaks in keV.

    energy_peak_keV: 1D `numpy.ndarray`
        The edges of adjacent energy bins.
        Length must be n+1 where n is the number of energy bins.
        These energy bins may be referred to as 'spectrum energy bins' in comments.

    line_intensities: 2D `numpy.ndarray`
        The amplitude of the line peaks.
        The last dimension represents intensities of each line in line_peaks_keV while
        the first dimension represents the intensities as a function of another parameter,
        e.g. temperature.
        These intensities are the ones divided between neighboring bins as described above.

    Returns
    -------
    new_line_intensities: 2D `numpy.ndarray`
        The weighted line intensities including neigboring component for each line weighted
        such that total emission is the same, but the energy of each line averaged over the
        energy_edge_keV bins is the same as the actual line energy.

    new_iline: `numpy.ndarray`
        Indices of the spectrum energy bins to which emission from each line corresponds.
        This includes indices of the neighboring bin emission components.

    """
    # Get widths and centers of the spectrum energy bins.
    energy_bin_widths = energy_edges_keV[1:] - energy_edges_keV[:-1]
    energy_centers = energy_edges_keV[:-1] + energy_bin_widths/2
    energy_center_diffs = energy_centers[1:] - energy_centers[:-1]

    # For each line, find the index of the spectrum energy bin to which it corresponds.
    iline = np.digitize(line_peaks_keV, energy_edges_keV) - 1

    # Get the difference between each line energy and
    # the center of the spectrum energy bin to which is corresponds.
    line_deviations_keV = line_peaks_keV - energy_centers[iline]
    # Get the indices of the lines which are above and below their bin center.
    neg_deviation_indices, = np.where(line_deviations_keV < 0)
    pos_deviation_indices, = np.where(line_deviations_keV >= 0)
    # Discard bin indices at the edge of the spectral range if they should
    # be shared with a bin outside the energy range.
    neg_deviation_indices = neg_deviation_indices[np.where(iline[neg_deviation_indices] > 0)[0]]
    pos_deviation_indices = pos_deviation_indices[
        np.where(iline[pos_deviation_indices] <= (len(energy_edges_keV)-2))[0]]

    # Split line emission between the spectrum energy bin containing the line peak and
    # the nearest neighboring bin based on the proximity of the line energy to
    # the center of the spectrum bin.
    # Treat lines which are above and below the bin center separately as slightly
    # different indexing is required.
    new_line_intensities = copy.deepcopy(line_intensities)
    new_iline = copy.deepcopy(iline)
    if len(neg_deviation_indices) > 0:
        neg_line_intensities, neg_neighbor_intensities, neg_neighbor_iline = _weight_emission_bins(
            line_deviations_keV, neg_deviation_indices,
            energy_center_diffs, line_intensities, iline, negative_deviations=True)
        # Combine new line and neighboring bin intensities and indices into common arrays.
        new_line_intensities[:, neg_deviation_indices] = neg_line_intensities
        new_line_intensities = np.concatenate((new_line_intensities, neg_neighbor_intensities), axis=-1)
        new_iline = np.concatenate((new_iline, neg_neighbor_iline))

    if len(pos_deviation_indices) > 0:
        pos_line_intensities, pos_neighbor_intensities, pos_neighbor_iline = _weight_emission_bins(
            line_deviations_keV, pos_deviation_indices,
            energy_center_diffs, line_intensities, iline, negative_deviations=False)
        # Combine new line and neighboring bin intensities and indices into common arrays.
        new_line_intensities[:, pos_deviation_indices] = pos_line_intensities
        new_line_intensities = np.concatenate(
            (new_line_intensities, pos_neighbor_intensities), axis=-1)
        new_iline = np.concatenate((new_iline, pos_neighbor_iline))

    # Order new_line_intensities so neighboring intensities are next
    # to those containing the line peaks.
    ordd = np.argsort(new_iline)
    new_iline = new_iline[ordd]
    for i in range(new_line_intensities.shape[0]):
        new_line_intensities[i, :] = new_line_intensities[i, ordd]

    return new_line_intensities, new_iline


def _weight_emission_bins(line_deviations_keV, deviation_indices,
                          energy_center_diffs, line_intensities, iline,
                          negative_deviations=True):
    if negative_deviations is True:
        if not np.all(line_deviations_keV[deviation_indices] < 0):
            raise ValueError(
                "As negative_deviations is True, can only handle "
                "lines whose energy < energy bin center, "
                "i.e. all line_deviations_keV must be negative.")
        a = -1
        b = -1
    else:
        if not np.all(line_deviations_keV[deviation_indices] >= 0):
            raise ValueError(
                "As negative_deviations is not True, can only handle "
                "lines whose energy >= energy bin center, "
                "i.e. all line_deviations_keV must be positive.")
        a = 0
        b = 1

    # Calculate difference between line energy and the spectrum bin center as a
    # fraction of the distance between the spectrum bin center and the
    # center of the nearest neighboring bin.
    wghts = np.absolute(line_deviations_keV[deviation_indices]) / energy_center_diffs[iline[deviation_indices+a]]
    # Tile/replicate wghts through the other dimension of line_intensities.
    wghts = np.tile(wghts, tuple([line_intensities.shape[0]] + [1] * wghts.ndim))

    # Weight line intensitites.
    # Weight emission in the bin containing the line intensity by 1-wght,
    # since by definition wght < 0.5 and
    # add the line intensity weighted by wght to the nearest neighbor bin.
    # This will mean the intensity integrated over the two bins is the
    # same as the original intensity, but the intensity-weighted peak energy
    # is the same as the original line peak energy even with different spectrum energy binning.
    new_line_intensities = line_intensities[:, deviation_indices] * (1-wghts)
    neighbor_intensities = line_intensities[:, deviation_indices] * wghts
    neighbor_iline = iline[deviation_indices]+b

    return new_line_intensities, neighbor_intensities, neighbor_iline
