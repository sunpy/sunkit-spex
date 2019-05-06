import copy
import os.path
import glob
from collections import OrderedDict

import numpy as np
import astropy.units as u
import scipy.io
from sunpy.io.special.genx import read_genx
from scipy.sparse import csr_matrix

SSWDB_XRAY_CHIANTI = os.path.expanduser(os.path.join("~", "ssw", "packages",
                                                     "xray", "dbase", "chianti"))

class ChiantiKevLines():
    """
    Class for evaluating chianti_kev_lines while keeping certain variables common between methods.

    """
    def __init__(self):

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

    relative_abundances: `numpy.ndarray`
        A 2XN array, where the first index gives the atomic number
        of the element and the second gives its relative abundance
        to its nominal value given by ABUN.

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
        if earth is not False:
            warning.warn("distance and earth kwargs set. Ignoring earth and using distance.")
    else:
        if earth is False:
            observer_distance = 1
        else:
            if date is None:
                distance = 1 * u.AU
            else:
                distance = sunpy.coordinates.get_sunearth_distance(time=date)
    
    # For ease of calculation, convert inputs to standard units and
    # scale to manageable numbers.
    em_factor = 1e44
    temp = temperature.to(u.MK).value
    emission_measure = emission_measure.to(u.cm**(-3)).value / em_factor
    energy_edges = energy_edges.to(u.keV).value

    mgtemp = temp * 1e6
    uu = np.log10(mgtemp)

    zindex, out, totcont, totcont_lo, edge_str, ctemp, chianti_doc = chianti_kev_common_load(linefile=file_in)
    # Location of file giving intensity as a function of temperatures at all line energies:
    # https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav
    line_energies, log10_temp_K_range, line_intensities, line_element_indices, element_indices, \
      line_iz = _extract_from_chianti_lines_sav()
    energy = (np.linspace(3, 9, 1001) * u.keV).value

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
    zindex = copy.copy(element_indices)

    if n_line_indices > 0:
        eline = eline[sline]

        p = chianti_kev_getp(out, sline, logt, temp*1e6, nsline)
        # Tested to here

        abundance_ratio = np.ones(len_abundances)
        if rel_abun is not None:
            abundance_ratio[rel_abun[0,*]-1] = rel_abun[1,*]

        # We include default_abundance because it will have zeroes for elements not included
        # and ones for those included
        default_abundance = np.zeros(len_abundance)
        default_abundance[zindex] = 1.0
        abund = (default_abundance * abundance * abundance_ratio)[out_lines_iz[sline]-1]
        emiss = p * abund

        # energy products
        wedg = energy[1:] - energy[:-1]
        energm = energy[:-1] + wedg/2
        
        iline = np.digitize(energy, eline) - 1

        hhh = np.histogram(eline - energm[iline], bins=10, range=(-10., 10.))[0]
        rr = get_reverse_indices(eline - energm[iline], nbins=10, min_range=-10., max_range=10.)[1]

        ###### Ask Richard how wghtline works. I got None for line below. ######
        wghtline = True
        # look for wide bins next to line bins, if too wide x 2 eline bin width
        # then don't spread out lines
        wedg0 = wedg[iline]
        wedg0a = wedg[iline-1>0]
        wedg0b = wedg[iline+1<(n_elements(wedg)-1)]
        wghtline = wghtline and np.max([(wedg0a/wedg0).max(), (wedg0b/wedg0).max()]) < 2.) \
          and (wedg0.max() < 1.5)

        if wghtline:
            if hhh[0] >= 1:
                etst = rr[0]
                itst = np.where(iline[etst] > 0)[0]

                if len(itst) >= 1:
                    etst = etst[itst]

                    wght = (energm[iline[etst]]-eline[etst]) / (energm[iline[etst]]-energm[iline[etst]-1])
                    wght = np.tile(wght, tuple([mtemp] + [1] * wght.ndim))

                    temp = emiss[etst, :]
                    emiss[etst, :] = temp * (1-wght)
                    emiss = np.concatenate((emiss, temp*wght))

                    iline = np.concatenate((iline, iline[etst]-1))

            if hhh[1] >= 1:

                etst = rr[1]
                itst = np.where( iline[etst] <= (nenrg-2))[0]

                if len(itst) >= 1:
                    etst = etst[itst]

                    wght = (eline[etst] - energm[iline[etst]]) / (energm[iline[etst]+1]-energm[iline[etst]])
                    wght = np.tile(wght, tuple([mtemp] + [1] * wght.ndim))

                    temp = emiss[etst, :]
                    emiss[etst, :] = temp * (1-wght)
                    emiss = np.concatenate((emiss, temp*wght))
                    iline = np.concatenate((iline, iline[etst]+1))

            ordd   = np.sort(iline)
            iline = iline[ordd]
            for i in range(mtemp):
                emiss[i, :] = emiss[i, ordd]

        ##########################################################################

        fline = np.histogram(iline, min=0, max=nenrg-1)[0]
        r = get_reverse_indices(iline, nbins=nenrg, min_range=0, max_range=nenrg-1)[1]

        select = np.where(fline > 0)[0]
        nselect = len(select)
        if nselect > 0:
            for j in range(mtemp):
                for i in select:
                    rr = r[r[i]]
                    spectrum[j, i] = sum( emiss[j, rr:rr + fline[i]-1])

            # Put spectrum into correct units. This line is equivalent to chianti_kev_units.pro
            spectrum = spectrum * 1e44 / observer_distance / wedg

    return spectrum

def chianti_kev_common_load(linefile=None, contfile=None, _extra=None):
    """
    This procedure is called to load the common blocks that support the chianti_kev... functions.

    Parameters
    ----------
    linefile: 

    contfile:

    no_abund: Not used

    reload: Not used

    _extra:

    
    """
    zindex, out = chianti_kev_line_common_load(linefile, _extra=_extra))
    zindex, totcont, totcont_lo, edge_str, ctemp, chianti_doc = chianti_kev_cont_common_load(
            contfile, _extra=_extra)
    return zindex, out, totcont, totcont_lo, edge_str, ctemp, chianti_doc

def chianti_kev_line_common_load(file_in=None, _extra=None):
    line_energies, log10_temp_K_range, line_intensities, line_element_indices,\
      element_indices, line_element_indices = _extract_from_chianti_lines_sav()

    # Define defaults
    if file_in is None:
        file_in = os.path.join(SSWDB_XRAY_CHIANTI, "chianti_lines_1_10_v71.sav")
        file_check = glob.glob(file_in)
        if file_check == []:
            file_in = os.path.join(SSWDB_XRAY_CHIANTI, "chianti_lines.geny")
            file_check = glob.glob(file_in)
            if file_check = []:
                raise ValueError("line files not found: {0} or {1}".format(
                    os.path.join(SSWDB_XRAY_CHIANTI, "chianti_lines_1_10_v71.sav"), file_in))
            
    if file_in.split(".")[-1] == "sav":
        # Read file
        contents = scipy.io.readsav(file_in)
        zindex = contents["zindex"]
        out = contents["out"]
    elif file_in.split(".")[-1] == "geny":
        # Read file...
        raise NotImplementedError("Reading .geny file not yet implemented.")
    else:
        raise ValueError("unrecognized file type: .{0}. Must be .sav or .geny") 

    # Sort lines in ascending energy. Stored in file in ascending wavelength.
    out["lines"][0]["WVL"] = out["lines"][0]["WVL"][np.argsort(out["lines"][0]["WVL"])[::-1]]
    
    return zindex, out
    
    

def chianti_kev_cont_common_load(file_in, _extra=None):
    line_energies, log10_temp_K_range, line_intensities, line_element_indices,\
      element_indices, line_element_indices = _extract_from_chianti_lines_sav()

    # Define defaults
    if file_in is None:
        file_in = os.path.join(SSWDB_XRAY_CHIANTI, "chianti_cont_1_250_v71.sav")
        file_check = glob.glob(file_in)
        if file_check == []:
            file_in = os.path.join(SSWDB_XRAY_CHIANTI, "chianti_cont.geny")
            file_check = glob.glob(file_in)
            if file_check = []:
                raise ValueError("line files not found: {0}; {1}".format(
                    os.path.join(SSWDB_XRAY_CHIANTI, "chianti_cont_1_250_v71.sav"), file_in))
    # Read file
    if file_in.split(".")[-1] == "sav":
        contents = scipy.io.readsav(file_in)
        zindex = contents["zindex"]
        totcont = contents["totcont"]
        totcont_lo = contents["totcont_lo"]
        edge_str = contents["edge_str"]
        ctemp = contents["ctemp"]
        chianti_doc = contents["chianti_doc"]
    elif file_in.split(".")[-1] == "geny":
        # Read file...
        raise NotImplementedError("Reading .geny file not yet implemented.")
    else:
        raise ValueError("unrecognized file type: .{0}. Must be .sav or .geny")

    return zindex, totcont, totcont_lo, edge_str, ctemp, chianti_doc


def xr_rd_abundance(ab_type=None, xr_ab_file=None):
    """
    This returns the abundances written in the xray_abun_file.genx
    The abundances are taken from CHIANTI and MEWE.  The source filenames are:
    cosmic sun_coronal sun_coronal_ext sun_hybrid sun_hybrid_ext sun_photospheric mewe_cosmic mewe_solar
    The first six come fron Chianti, the last two from Mewe.  They are:
    cosmic sun_coronal sun_coronal_ext sun_hybrid sun_hybrid_ext sun_photospheric mewe_cosmic mewe_solar
    These abundances are used with CHIANTI_KEV.  MEWE_KEV can only use the two mewe sourced
    abundance distributions unless using a heavily modified rel_abun structure for all of the elements.

    Parameters
    ----------
    ab_type: `str`
        Type of abundance to be read from file.  Option are (From Chianti)
        1. cosmic
        2. sun_coronal - default abundance
        3. sun_coronal_ext
        4. sun_hybrid
        5. sun_hybrid_ext
        6. sun_photospheric
        7. mewe_cosmic
        8. mewe_solar - default for mewe_kev

    xr_ab_file: `str`
        Name and path to abundance file.
        Default= ~/ssw/packages/xray/dbase/chianti/xray_abun_file.genx

    Returns
    -------
    out:
        Array of 50 abundance levels for first 50 elements.

    """
    # If kwargs not set, set defaults
    if ab_type is None:
        ab_type = "sun_coronal"
    if xr_ab_file is None:
        xr_ab_file = os.path.expanduser(os.path.join(SSWDB_XRAY_CHIANTI,
                                                     "xray_abun_file.genx"))
    # Read file
    ab_sav = read_abundance_genx(xr_ab_file)
    # Return relevant abundance.
    return ab_sav[ab_type]


def read_abundance_genx(filename):
    # Read file.
    contents = read_genx(filename)
    # Combine data and keys from each entry in file.
    output = OrderedDict()
    for arr in contents["SAVEGEN0"]:
        output[arr["FILNAM"]] = arr["ABUND"]
    # Add header data
    output["header"] = contents["HEADER"]
    output["header"]["CHIANTI VERSION"] = float(contents["SAVEGEN1"][:3])

    return output


def _extract_from_chianti_lines_sav():
    """
    Extracts data from the CHIANTI lines file relevant for chianti_kev_lines function.

    Returns
    -------

    """
    # Read file.
    struct = scipy.io.readsav("chianti_lines_1_10_v71.sav")
    lines = struct["out"]["lines"][0]
    # Extract energy grid for line information.
    line_energies = (lines["wvl"] * u.angstrom).to(u.keV, equivalencies=u.spectral())
    # Extract log10 of temperature grid for line info.
    log10_temp_K_range = struct["out"]["logt_isothermal"][0]
    # Extract line intensities.
    line_intensities = np.empty((lines["int"].shape[0], lines["int"][0].shape[0]), dtype=float)
    for i in range(line_intensities.shape[0]):
        line_intensities[i, :] = struct["out"]["lines"][0]["int"][i]
    # line_intensities =* u.
    # Extract line IZs.
    line_element_indices = lines["iz"]
    # Extract the zindex
    element_indices = struct["zindex"]
    

    return line_energies, log10_temp_K_range, line_intensities, line_element_indices,\
      element_indices, line_element_indices


def chianti_kev_getp(line_intensities, sline, logt, mgtemp, nsline):
    nltemp = len(logt)
    selt = np.digitize( np.log10(mgtemp), logt)-1
    p = np.zeros((mtemp, nsline))
    for i in range(mtemp):
        indx = selt[i]-1+np.arange(3)
        indx = indx[np.logical_and(indx > 0, indx < (nltemp-1))]
        tband = 
        p[i, :] = scipy.interpolate.interp1d(
            logt[indx], line_intensities[sline][:, indx], kind="quadratic")(uu).squeeze()[:]

    return p

def get_reverse_indices(x, nbins, min_range=None, max_range=None):
    """
    For a set of contiguous equal sized 1D bins, generates index of lower edge of bin in which each element of x belongs and the indices of x in each bin.
    
    Parameters
    ----------
    x: array-like
        Values to be binned.

    nbins: `int`
        Number of bins to divide range into.
    
    min_range: `float` or `int` (Optional)
        Lower limit of range of bins. Default=min(x)

    max_range: `float` or `int` (Optional)
        Upper limit of range of bins. Default=max(x)

    Returns
    -------
    arrays_bin_indices: `np.ndarray`
        Index of lower edge of bin into which each element of x goes. Same length as x.
    
    bins_array_indices: `tuple` of `np.ndarray`s
        Indices of elements of x in each bin. One set of indices for each bin.

    bin_edges: `np.ndarray`
        Edges of bins. Length is nbins+1.

    """
    if min_range is None:
        min_range = min(x)
    if max_range is None:
        max_range = max(x)
    bin_edges = np.linspace(min_range, max_range, nbins+1)
    arrays_bin_indices = (float(nbins)/(max_range - min_range)*(x - min_range)).astype(int)
    bins_array_indices = tuple([np.where(arrays_bin_indices == i)[0] for i in range(nbins)])
    return arrays_bin_indices, bins_array_indices, bin_edges
