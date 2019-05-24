import os.path
import copy

import numpy as np
import astropy.units as u
from astropy.table import Table, Column
import scipy.interpolate
import sunpy.coordinates

from sunxspex.io import chianti_kev_line_common_load, load_xray_abundances
from sunxspex.utils import get_reverse_indices

SSWDB_XRAY_CHIANTI = os.path.expanduser(os.path.join("~", "ssw", "packages",
                                                     "xray", "dbase", "chianti"))
FILE_IN = "/Users/dnryan/ssw/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav"


class ChiantiKevLines():
    """
    Class for evaluating chianti_kev_lines while keeping certain variables common between methods.

    """
    def __init__(self):
        pass

@u.quantity_input(energy_edges=u.keV, temperature=u.K, emission_measure=1/u.cm**3,
                  observer_distance='length')
def chianti_kev_lines(energy_edges, temperature, emission_measure=1e44/u.cm**3,
                      relative_abundances=None, line_file=None,
                      observer_distance=None, earth=False, date=None,
                      **kwargs):
    """
    Returns a thermal spectrum (line + continuum) given temperature and emission measure.

    Uses a database of line and continua spectra obtained from the CHIANTI distribution

    Parameters
    ----------
    energy_edges: `astropy.units.Quantity`
        The edges of the energy bins in a 1D N+1 quantity.

    temperature: `astropy.units.Quantity`
        The electron temperature of the plasma.

    emission_measure: `astropy.units.Quantity`
        The emission measure of the emitting plasma.
        Default= 1e44 cm**-3

    relative_abundances: `list` of length 2 `tuple`s
        The relative abundances of different elements as a fraction of their
        nominal abundances which are read in by xr_rd_abundance().
        Each tuple represents an element.
        The first item in the tuple gives the atomic number of the element.
        The second item gives the factor by which to scale the element's abundance.

    observer_distance: `astropy.units.Quantity` (Optional)
        The distance between the source and the observer. Scales output to observer distance
        and unit by 1/length. If not set, output represents value at source and 
        unit will have an extra length component.
        Default=None

    earth: `bool` (Optional)
        Sets distance to Sun-Earth distance if not already set by user.
        If distance is set, earth is ignored. 
        If date kwarg is set (see below), Sun-Earth distance at that time is calculated.
        If date kwarg is not set, Sun_earth distance is set to 1 AU.
        Default=False

    date: `astropy.time.Time` for parseable by `sunpy.time.parse_time` (Optional)
        The date for which the Sun-Earth distance is to be calculated.
        Ignored if earth kwarg not set.
        Default=None.

    Returns
    -------
    Flux: `astropy.units.Quantity`

    Notes
    -----
    Explanation of Chianti units & emission measure (Ken Phillips, June 17, 2004):

    Output of Chianti ch_ss units are in photons (or ergs) cm-2 s-1 A-1 sr-1, i.e.
    the output is a specific intensity (not a flux as it is per solid angle).
    Suppose specific intensity at some wavelength is F_lam for a
    *surface* emission measure = 10^27 cm^-5 (this is the ch_ss default).

    Let a flare on the sun have area A (cm^2). Its solid angle at earth is A/(au)^2.

    Therefore flux_lam = F_lam * A / (au)^2

    The flare *volume* emission measure corresponding to the 10^27 surface EM is A * 10^27 cm^-3.

    So flux per unit volume EM (Ne^2 V = 1) is

    F_lam * A/(au)^2 * 1/(10^27 A) = F_lam / (10^27 [au]^2) = 4.44e-54 * F_lam
    (log(4.44e-54) = -53.35)

    Note the A's cancel out.

    So if you want to generate a *volume* EM = 10^49 cm^-3 from ch_ss,
    put in a value of log(*surface* EM) = 27.0 + 49.0 -  53.35 = 22.648

    The units of the spectrum will then be standard flux units, i.e. photons cm^2 s^-1 A^-1,
    without any steradian units.

    You can then convert fluxes to ph. Cm^2 s^-1 keV-1 by just multiplying
    by (lambda in A)^2 / 12.399 [or 12.399/(energy in keV)^2] and
    wavelength to energy by wavelength = 12.399/energy in keV.

    """
    # Set kwarg values from user inputs.
    if observer_distance is not None:
        if earth:
            raise ValueError(
                    "Conflicting inputs. Both distance and earth kwargs set. Can only set one.")
    else:
        if not earth:
            observer_distance = 1
        else:
            if date is None:
                observer_distance = (1 * u.AU).to(u.cm).value
            else:
                observer_distance = sunpy.coordinates.get_sunearth_distance(time=date).to(u.cm).value
    # Format relative abundances.
    if relative_abundances is not None:
        relative_abundances = Table(rows=relative_abundances,
                                    names=("atomic number", "relative abundance"),
                                    meta={"description": "relative abundances"},
                                    dtype=(int, float))

    # For ease of calculation, convert inputs to known units and
    # scale to manageable numbers.
    energy_edges_keV = energy_edges.to(u.keV).value
    temperature_K = temperature.to(u.K).value
    log_T = np.log10(temperature_K)
    emission_measure_cm = emission_measure.to(1/u.cm**3).value
    try:
        n_temperatures = len(temperature_K)
    except TypeError:
        n_temperatures = 1
    n_energy_bins = len(energy_edges_keV)-1

    # Define array to hold emission at each energy and temperature.
    spectrum = np.zeros((n_temperatures, n_energy_bins))

    # Load emission line data from CHIANTI file.
    zindex, line_meta, line_properties, line_intensities_grid = chianti_kev_line_common_load(linefile=FILE_IN)
    line_energy_bins = line_properties["ENERGY"].quantity.to(u.keV).value
    line_logT_bins = line_meta["LOGT_ISOTHERMAL"]
    line_element_indices = line_properties["IZ"].data

    # Find indices of line energy bins within user input energy range.
    energy_roi_indices = np.logical_and(line_energy_bins >= energy_edges_keV.min(),
                                        line_energy_bins <= energy_edges_keV.max())
    n_energy_roi_indices = energy_roi_indices.sum()
    energy_roi_indices = np.arange(len(line_energy_bins))[energy_roi_indices]

    # If there are line within energy range of interest, compile spectrum. 
    if n_energy_roi_indices > 0:
        # Restrict energy bins of lines to energy range of interest.
        line_energy_bins = line_energy_bins[energy_roi_indices]

        # Calculate abundance of each desired element.
        # First, load default abundances.
        default_abundances = load_xray_abundances(abundance_type=kwargs.get("abundance_type", None),
                                                  xray_abundance_file=kwargs.get("xray_abundance_file", None))
        len_abundances = len(default_abundances)
        # Second, define define ratios by which to scale relative abundances
        # which account for relative_abundances.
        abundance_ratios = np.ones(len_abundances)
        if relative_abundances is not None:
            abundance_ratios[relative_abundances["atomic number"]-1] = relative_abundances["relative abundance"]
        # Third, create mask to select only desired elements
        # that produce line emission as defined by zindex.
        abundances_mask = np.zeros(len_abundances)[zindex] = 1.0
        # Fourth, multiply default abundances by abundance ratios and mask to get true abundances.
        abundances = default_abundances * abundance_ratios * abundances_mask
        # Finally, extract only lines within energy range of interest.
        abundances = abundances[line_element_indices[energy_roi_indices]-1]

        # Calculate normalized emissivity of each line in energy range of interest
        # as a function of energy and temperature.
        line_intensities = _chianti_kev_getp(np.log10(temperature_K), line_intensities_grid[energy_roi_indices], line_logT_bins)
        # Scale line_intensities by abundances to get true line_intensities.
        line_intensities = line_intensities * abundances

        # Reweight the emission in bins around the line centroids
        # so they appear at the correct energy, despite the binning.
        line_intensities, iline = _weight_emission_bins_to_line_centroid(line_energy_bins, energy_edges_keV, n_temperatures, line_intensities)

        # Determine which spectrum energy bins contain components of line emission.
        # Sum over those line components to get total emission in each spectrum energy bin.
        spectrum_bins_line_energy_indices = get_reverse_indices(iline, nbins=n_energy_bins, min_range=0, max_range=n_energy_bins-1)[1]
        emitting_energy_bin_indices = np.where(np.histogram(iline, bins=n_energy_bins, range=(0, n_energy_bins-1))[0] > 0)[0]
        if len(emitting_energy_bin_indices) > 0:
            for j in range(n_temperatures):
                for i in emitting_energy_bin_indices:
                    spectrum[j, i] = sum(line_intensities[j, spectrum_bins_line_energy_indices[i]])

    # Eliminate redundant axes, scale units to observer distance and put into correct units.
    # When scaling to observer distance, don't divide by 4 pi. Unlike Mewe, CHIANTI is in units of steradian.
    energy_bin_widths = energy_edges_keV[1:] - energy_edges_keV[:-1]
    spectrum = spectrum.squeeze() / energy_bin_widths * emission_measure_cm / observer_distance**2

    return spectrum


def _chianti_kev_getp(logT, data_grid, line_logT_bins):
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
    logT: `float` or 1D `numpy.ndarray` of `float`.
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
    try:
        n_temperatures = len(logT)
    except TypeError:
        logT = np.array([logT])
        n_temperatures = 1

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


def _weight_emission_bins_to_line_centroid(line_energy_bins, energy_edges_keV, n_temperatures, line_intensities):
    """Weights emission in neighboring spectral bins to make centroid have correct spectral value."""
    n_energy_bins = len(energy_edges_keV)-1

    # Get widths and centers of energy bins. 
    energy_bin_widths = energy_edges_keV[1:] - energy_edges_keV[:-1]
    energy_centers = energy_edges_keV[:-1] + energy_bin_widths/2
    
    # For each line energy bin, find the index of the input energy bin to which it corresponds.
    iline = np.digitize(line_energy_bins, energy_edges_keV) - 1

    # Get reverse indices for each bin.
    rr = get_reverse_indices(line_energy_bins - energy_centers[iline], nbins=10, min_range=-10., max_range=10.)[1]
    # Extract bins with >0 counts.
    rr = tuple(np.array(rr)[np.where(np.array([len(ri) for ri in rr]) > 0)[0]])
    
    if len(rr[0]) >= 1:
        etst = rr[0]
        itst = np.where(iline[etst] > 0)[0]

        if len(itst) >= 1:
            etst = etst[itst]

            wght = (energy_centers[iline[etst]]-line_energy_bins[etst]) / (energy_centers[iline[etst]]-energy_centers[iline[etst]-1])
            wght = np.tile(wght, tuple([n_temperatures] + [1] * wght.ndim))

            temp = line_intensities[:, etst]
            line_intensities[:, etst] = temp * (1-wght)
            line_intensities = np.concatenate((line_intensities, temp*wght), axis=-1)

            iline = np.concatenate((iline, iline[etst]-1))

    if len(rr[1]) >= 1:

        etst = rr[1]
        itst = np.where( iline[etst] <= (n_energy_bins-2))[0]

        if len(itst) >= 1:
            etst = etst[itst]

            wght = (line_energy_bins[etst] - energy_centers[iline[etst]]) / (energy_centers[iline[etst]+1]-energy_centers[iline[etst]])
            wght = np.tile(wght, tuple([n_temperatures] + [1] * wght.ndim))

            temp = line_intensities[:, etst]
            line_intensities[:, etst] = temp * (1-wght)
            line_intensities = np.concatenate((line_intensities, temp*wght), axis=-1)
            iline = np.concatenate((iline, iline[etst]+1))

    ordd = np.argsort(iline)
    iline = iline[ordd]
    for i in range(n_temperatures):
        line_intensities[i, :] = line_intensities[i, ordd]

    return line_intensities, iline
