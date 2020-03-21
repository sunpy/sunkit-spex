import os.path
import copy

import numpy as np
import astropy.units as u
from astropy.table import Table
import scipy.interpolate
import sunpy.coordinates

from sunxspex.io import chianti_kev_line_common_load_light, load_xray_abundances
from sunxspex.utils import get_reverse_indices

__all__ = ['ChiantiThermalSpectrum']

class ChiantiThermalSpectrum:
    """
    Class for evaluating solar X-ray thermal spectrum using CHIANTI data.
    """
    @u.quantity_input(energy_edges=u.keV)
    def __init__(self, energy_edges,
                 abundance_type=None,
                 observer_distance=1*u.AU, date=None):
        """
        Read in data required by methods of this class.

        Parameters
        ----------
        energy_edges: `astropy.units.Quantity`
            The edges of the energy bins in a 1D N+1 quantity.

        abundance_type: `str`
            Type of abundance to be read from file.  Option are (From Chianti)
            1. cosmic
            2. sun_coronal - default abundance
            3. sun_coronal_ext
            4. sun_hybrid
            5. sun_hybrid_ext
            6. sun_photospheric
            7. mewe_cosmic
            8. mewe_solar - default for mewe_kev

        observer_distance: `astropy.units.Quantity` (Optional)
            The distance between the source and the observer. Scales output to observer distance
            and unit by 1/length.
            If None, output represents value at source and unit will have an extra length component.
            Default=1 AU

        date: `astropy.time.Time` for parseable by `sunpy.time.parse_time` (Optional)
            The date for which the Sun-Earth distance is to be calculated.
            Cannot be set is observer_distance kwarg also set.
            Default=None.

        """
        # Define energy bins on which spectrum will be calculated.
        self.energy_edges_keV = energy_edges.to(u.keV).value

        # Set observer_distance.
        if observer_distance is None:
            if date is None:
                self.observer_distance = 1
            else:
                self.observer_distance = sunpy.coordinates.get_sunearth_distance(time=date).to(u.cm)
        else:
            if date is not None:
                raise ValueError("Conflicting inputs. "
                                 "observer_distance and data kwargs cannot both be set.")
            self.observer_distance = observer_distance.to(u.cm)

        # Load emission line data from CHIANTI file.
        self.zindex, line_peak_energies, self.line_logT_bins, self.line_colEMs, \
            self.line_element_indices, self.line_intensities_per_solid_angle_grid = \
            chianti_kev_line_common_load_light()
        self.line_peaks_keV = line_peak_energies.to(u.keV).value

        # Load default abundances.
        self.default_abundances = load_xray_abundances(abundance_type=abundance_type)
        # Create mask to select only elements that produce line emission as defined by zindex.
        self.n_default_abundances = len(self.default_abundances)
        self.abundances_mask = np.zeros(self.n_default_abundances)[self.zindex] = 1.0

        # Calculate line intensities accounting for observer_distance.
        # The line intensities read from file are in units of ph / cm**2 / s / sr.
        # Therefore they are specific intensities, i.e. per steradian, or solid angle.
        # Here, let us call these intensities, intensity_per_solid_angle.
        # The solid angle is given by flare_area / observer_distance**2.
        # Total integrated intensity can be rewritten in terms of volume EM and solid angle:
        # intensity == intensity_per_solid_angle_per_volEM * volEM * solid_angle ==
        # == intensity_per_solid_angle / (colEM * flare_area) * (flare_area / observer_dist**2) * volEM ==
        # == intensity_per_solid_angle / colEM / observer_dist**2 * volEM
        # i.e. flare area cancels. Therefore:
        # intensity = intensity_per_solid_angle / colEM / observer_dist**2 * volEM,
        # or, dividing both sides by volEM,
        # intensity_per_volEM = intensity_per_solid_angle / colEM / observer_dist**2
        # Here, let us calculate intensity_per_volEM and
        # scale by the flare volume EM supplied by the user later.
        # Note that the column emission measure used by CHIANTI in calculating the intensities
        # is available from the file as self.line_colEMs.
        # Also noote that as part of this calculation, the steradian unit must be canceled manually.
        if isinstance(self.observer_distance, u.Quantity):
            self.line_intensities_per_volEM_grid = \
                self.line_intensities_per_solid_angle_grid / self.line_colEMs / \
                self.observer_distance**2 * u.sr
        else:
            self.line_intensities_per_volEM_grid = \
                self.line_intensities_per_solid_angle_grid / self.line_colEMs
        self.line_intensities_per_volEM_unit = self.line_intensities_per_volEM_grid.unit
        self.line_intensities_per_volEM_grid_values = self.line_intensities_per_volEM_grid.value

    @u.quantity_input(temperature=u.K, emission_measure=1/u.cm**3)
    def chianti_kev_lines(self, temperature, emission_measure=1e44/u.cm**3, relative_abundances=None, **kwargs):
        """
        Returns a thermal spectrum (line + continuum) given temperature and emission measure.

        Uses a database of line and continua spectra obtained from the CHIANTI distribution

        Parameters
        ----------
        temperature: `astropy.units.Quantity`
            The electron temperature of the plasma.

        emission_measure: `astropy.units.Quantity`
            The emission measure of the emitting plasma.
            Default= 1e44 cm**-3

        relative_abundances: `list` of length 2 `tuple`
            The relative abundances of different elements as a fraction of their
            nominal abundances which are read in by xr_rd_abundance().
            Each tuple represents an element.
            The first item in the tuple gives the atomic number of the element.
            The second item gives the factor by which to scale the element's abundance.

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
        # Format relative abundances.
        if relative_abundances is not None:
            relative_abundances = Table(rows=relative_abundances,
                                        names=("atomic number", "relative abundance"),
                                        meta={"description": "relative abundances"},
                                        dtype=(int, float)) # There should be a way to not initiate a Table here or should be a version of this method that doesn't

        # For ease of calculation, convert inputs to known units and
        # scale to manageable numbers.
        temperature_K = temperature.to(u.K).value
        em_unit = (1/u.cm**3).unit
        emission_measure_cm = emission_measure.to(em_unit).value

        try:
            n_temperatures = len(temperature_K)
        except TypeError:
            n_temperatures = 1
        n_energy_bins = len(self.energy_edges_keV)-1

        # Define array to hold emission at each energy and temperature.
        spectrum = np.zeros((n_temperatures, n_energy_bins))

        # Find indices of line energy bins within user input energy range.
        energy_roi_indices = np.logical_and(self.line_peaks_keV >= self.energy_edges_keV.min(),
                                            self.line_peaks_keV <= self.energy_edges_keV.max())
        n_energy_roi_indices = energy_roi_indices.sum()
        energy_roi_indices = np.arange(len(self.line_peaks_keV))[energy_roi_indices]

        # If there are line within energy range of interest, compile spectrum.
        if n_energy_roi_indices > 0:
            # Restrict energy bins of lines to energy range of interest.
            line_peaks_keV = self.line_peaks_keV[energy_roi_indices]

            # Calculate abundance of each desired element.
            # First, define ratios by which to scale relative abundances
            # which account for relative_abundances.
            abundance_ratios = np.ones(self.n_default_abundances)
            if relative_abundances is not None:
                abundance_ratios[relative_abundances["atomic number"]-1] = relative_abundances["relative abundance"]
            # Third, multiply default abundances by abundance ratios and mask to get true abundances.
            abundances = self.default_abundances * abundance_ratios * self.abundances_mask
            # Finally, extract only lines within energy range of interest.
            abundances = abundances[self.line_element_indices[energy_roi_indices]-1]

            # Calculate abundance-normalized intensity of each line in energy range of interest
            # as a function of energy and temperature.
            line_intensities = _chianti_kev_getp(
                    np.log10(temperature_K),
                    self.line_intensities_per_volEM_grid_values[energy_roi_indices],
                    self.line_logT_bins)
            # Scale line_intensities by abundances to get true line_intensities.
            line_intensities = line_intensities * abundances

            # Reweight the emission in bins around the line centroids
            # so they appear at the correct energy, despite the binning.
            line_intensities, iline = _weight_emission_bins_to_line_centroid(line_peaks_keV, self.energy_edges_keV, line_intensities)

            # Determine which spectrum energy bins contain components of line emission.
            # Sum over those line components to get total emission in each spectrum energy bin.
            spectrum_bins_line_energy_indices = get_reverse_indices(iline, nbins=n_energy_bins, min_range=0, max_range=n_energy_bins-1)[1]
            emitting_energy_bin_indices = np.where(np.histogram(iline, bins=n_energy_bins, range=(0, n_energy_bins-1))[0] > 0)[0]

            if len(emitting_energy_bin_indices) > 0:
                for i in emitting_energy_bin_indices:
                    spectrum[:, i] = np.sum(line_intensities[:, spectrum_bins_line_energy_indices[i]], axis=1)

        # Eliminate redundant axes, scale units to observer distance and put into correct units.
        energy_bin_widths = self.energy_edges_keV[1:] - self.energy_edges_keV[:-1]
        spectrum = (spectrum.squeeze() * self.line_intensities_per_volEM_unit) / \
            (energy_bin_widths * u.keV) * (emission_measure_cm * em_unit)

        return spectrum

    def chianti_kev_cont():
        raise NotImplementedError()


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
    line_deviation_bin_indices = get_reverse_indices(line_deviations_keV, nbins=10,
                                                     min_range=-10., max_range=10.)[1]
    neg_deviation_indices, pos_deviation_indices = tuple(np.array(line_deviation_bin_indices)[
        np.where(np.array([len(ri) for ri in line_deviation_bin_indices]) > 0)[0]])
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
