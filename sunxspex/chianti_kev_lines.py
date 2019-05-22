import os.path
import copy

import numpy as np
import astropy.units as u
from astropy.table import Table, Column
import scipy.interpolate
import sunpy.coordinates

from sunxspex.io import chianti_kev_line_common_load, xr_rd_abundance
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

#@u.quantity_input(energy_edges=u.keV, temperature=u.K, emission_measure=1/(u.cm**3),
#                  observer_distance='length')
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
        #relative_abundances = [(26, 1.), (28, 1.)]
        relative_abundances = Table(rows=relative_abundances,
                                    names=("atomic number", "relative abundance"),
                                    meta={"description": "relative abundances"},
                                    dtype=(int, float))

    # For ease of calculation, convert inputs to standard units and
    # scale to manageable numbers.
    em_factor = 1e44
    temp = temperature.to(u.MK).value
    emission_measure = emission_measure.to(u.cm**(-3)).value / em_factor
    energy_edges = energy_edges.to(u.keV).value
    energy = energy_edges

    mgtemp = temp * 1e6
    uu = np.log10(mgtemp)

    zindex, line_meta, line_properties, line_intensities = chianti_kev_line_common_load(linefile=FILE_IN)
    line_energies = line_properties["ENERGY"].quantity.to(u.keV).value
    log10_temp_K_range = line_meta["LOGT_ISOTHERMAL"]
    line_element_indices = line_iz = line_properties["IZ"].data

    # Load abundances
    abundance = xr_rd_abundance(abundance_type=kwargs.get("abundance_type", None),
                                xr_ab_file=kwargs.get("xr_ab_file", None))
    len_abundances = len(abundance)

    # Find energies within energy range of interest.
    line_indices = np.logical_and(line_energies >= energy.min(),
                                  line_energies <= energy.max())
    n_line_indices = line_indices.sum()
    line_indices = np.arange(len(line_energies))[line_indices]
    try:
        mtemp = len(temp)
    except TypeError:
        mtemp = 1
    nenrg = len(energy[:-1])
    spectrum = np.zeros((mtemp, nenrg))

    # Rename variables to IDL names for ease of comparison.
    eline = copy.copy(line_energies)
    logt = copy.copy(log10_temp_K_range)
    out_lines_iz = copy.copy(line_iz)
    sline = copy.copy(line_indices)
    nsline = copy.copy(n_line_indices)

    if n_line_indices > 0:
        eline = eline[sline]

        p = chianti_kev_getp(line_intensities, sline, logt, temp*1e6, nsline)
 
        abundance_ratio = np.ones(len_abundances)
        if relative_abundances is not None:
            abundance_ratio[relative_abundances["atomic number"]-1] = relative_abundances["relative abundance"]

        # We include default_abundance because it will have zeroes for elements not included
        # and ones for those included
        default_abundance = np.zeros(len_abundances)
        default_abundance[zindex] = 1.0
        abund = (default_abundance * abundance * abundance_ratio)[out_lines_iz[sline]-1]
        emiss = p * abund
        # Tested to here without rel_abund

        # energy products
        wedg = energy[1:] - energy[:-1]
        energm = energy[:-1] + wedg/2
        
        iline = np.digitize(eline, energy) - 1

        # Get reverse indices for each bin.
        rr = get_reverse_indices(eline - energm[iline], nbins=10, min_range=-10., max_range=10.)[1]
        # Extract bins with >0 counts.
        rr = tuple(np.array(rr)[np.where(np.array([len(ri) for ri in rr]) > 0)[0]])
        hhh = [len(rrr) for rrr in rr]

        # Reweight the emission in bins around the line centroids
        # so they appear at the correct energy, despite the binning.
        emiss, iline = _weight_emission_bins_to_line_centroid(hhh, rr, iline, eline, energm, mtemp, emiss, nenrg)

        fline = np.histogram(iline, bins=nenrg, range=(0, nenrg-1))[0]
        r = get_reverse_indices(iline, nbins=nenrg, min_range=0, max_range=nenrg-1)[1]

        select = np.where(fline > 0)[0]
        if len(select) > 0:
            for j in range(mtemp):
                for i in select:
                    spectrum[j, i] = sum(emiss[j, r[i]]) # Can this be vectorized with tuples of indices like np.where?
            # Put spectrum into correct units. This line is equivalent to chianti_kev_units.pro
            spectrum = spectrum / wedg * em_factor

    # Eliminate redundant axes and Scale units to observer distance.
    # Unlike Mewe, don't divide by 4 pi. Chianti is in units of steradian.
    spectrum = spectrum.squeeze() / observer_distance**2

    return spectrum


def chianti_kev_getp(line_intensities, sline, logt, mgtemp, nsline):
    try:
        mtemp = len(mgtemp)
    except TypeError:
        mgtemp = np.array([mgtemp])
        mtemp = 1
    nltemp = len(logt)
    selt = np.digitize( np.log10(mgtemp), logt)-1
    p = np.zeros((mtemp, nsline))
    for i in range(mtemp):
        indx = selt[i]-1+np.arange(3)
        indx = indx[np.logical_and(indx > 0, indx < (nltemp-1))]
        uu = np.log10(mgtemp[i])
        p[i, :] = scipy.interpolate.interp1d(
            logt[indx], line_intensities[sline][:, indx], kind="quadratic")(uu).squeeze()[:]
    return p


def _weight_emission_bins_to_line_centroid(hhh, rr, iline, eline, energm, mtemp, emiss, nenrg):
    """Weights emission in neighboring spectral bins to make centroid have correct spectral value."""
    if hhh[0] >= 1:
        etst = rr[0]
        itst = np.where(iline[etst] > 0)[0]

        if len(itst) >= 1:
            etst = etst[itst]

            wght = (energm[iline[etst]]-eline[etst]) / (energm[iline[etst]]-energm[iline[etst]-1])
            wght = np.tile(wght, tuple([mtemp] + [1] * wght.ndim))

            temp = emiss[:, etst]
            emiss[:, etst] = temp * (1-wght)
            emiss = np.concatenate((emiss, temp*wght), axis=-1)

            iline = np.concatenate((iline, iline[etst]-1))

    if hhh[1] >= 1:

        etst = rr[1]
        itst = np.where( iline[etst] <= (nenrg-2))[0]

        if len(itst) >= 1:
            etst = etst[itst]

            wght = (eline[etst] - energm[iline[etst]]) / (energm[iline[etst]+1]-energm[iline[etst]])
            wght = np.tile(wght, tuple([mtemp] + [1] * wght.ndim))

            temp = emiss[:, etst]
            emiss[:, etst] = temp * (1-wght)
            emiss = np.concatenate((emiss, temp*wght), axis=-1)
            iline = np.concatenate((iline, iline[etst]+1))

    ordd = np.argsort(iline)
    iline = iline[ordd]
    for i in range(mtemp):
        emiss[i, :] = emiss[i, ordd]

    return emiss, iline
