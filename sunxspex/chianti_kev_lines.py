import copy
import os.path
import glob
from collections import OrderedDict

import numpy as np
import astropy.units as u
from astropy.table import Table, Column
import scipy.io
from scipy.sparse import csr_matrix
from sunpy.io.special.genx import read_genx
import sunpy.coordinates
from sunpy.time import parse_time

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
        if earth is not False:
            warning.warn("distance and earth kwargs set. Ignoring earth and using distance.")
    else:
        if earth is False:
            observer_distance = 1
        else:
            if date is None:
                observer_distance = 1 * u.AU
            else:
                observer_distance = sunpy.coordinates.get_sunearth_distance(time=date)
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
        ###### Ask Richard how wghtline works. I got None for line below. ######
        wghtline = True
        # look for wide bins next to line bins, if too wide x 2 eline bin width
        # then don't spread out lines
        """
        wedg0 = wedg[iline]
        wedg0a = wedg[iline-1>0]
        wedg0b = wedg[iline+1<(n_elements(wedg)-1)]
        wghtline = wghtline and np.max([(wedg0a/wedg0).max(), (wedg0b/wedg0).max()]) < 2.) \
          and (wedg0.max() < 1.5)
        """

        if wghtline:
            if hhh[0] >= 1:
                etst = rr[0]
                itst = np.where(iline[etst] > 0)[0]

                if len(itst) >= 1:
                    etst = etst[itst]

                    wght = (energm[iline[etst]]-eline[etst]) / (energm[iline[etst]]-energm[iline[etst]-1])
                    #wght = np.tile(wght, tuple([mtemp] + [1] * wght.ndim))

                    #temp = emiss[etst, :]
                    #emiss[etst, :] = temp * (1-wght)
                    temp = emiss[etst]
                    emiss[etst] = temp * (1-wght)
                    emiss = np.concatenate((emiss, temp*wght))

                    iline = np.concatenate((iline, iline[etst]-1))

            if hhh[1] >= 1:

                etst = rr[1]
                itst = np.where( iline[etst] <= (nenrg-2))[0]

                if len(itst) >= 1:
                    etst = etst[itst]

                    wght = (eline[etst] - energm[iline[etst]]) / (energm[iline[etst]+1]-energm[iline[etst]])
                    #wght = np.tile(wght, tuple([mtemp] + [1] * wght.ndim))

                    #temp = emiss[etst, :]
                    #emiss[etst, :] = temp * (1-wght)
                    temp = emiss[etst]
                    emiss[etst] = temp * (1-wght)
                    emiss = np.concatenate((emiss, temp*wght))
                    iline = np.concatenate((iline, iline[etst]+1))

            ordd = np.argsort(iline)
            iline = iline[ordd]
            #for i in range(mtemp):
            #    emiss[i, :] = emiss[i, ordd]
            emiss = emiss[ordd]

        ##########################################################################

        fline = np.histogram(iline, bins=nenrg, range=(0, nenrg-1))[0]
        r = get_reverse_indices(iline, nbins=nenrg, min_range=0, max_range=nenrg-1)[1]

        select = np.where(fline > 0)[0]
        nselect = len(select)
        if nselect > 0:
            #for j in range(mtemp):
            j = 0
            for i in select:
                    #spectrum[j, i] = sum( emiss[j, r[i]])
                    spectrum[j, i] = sum( emiss[r[i]])
            # Put spectrum into correct units. This line is equivalent to chianti_kev_units.pro
            #spectrum = spectrum * 1e44 / observer_distance / wedg
            spectrum = spectrum / wedg * em_factor

    return spectrum


def chianti_kev_common_load(linefile=None, contfile=None):
    """
    This procedure is called to load the common blocks that support the chianti_kev... functions.
    Parameters
    ----------
    linefile: `str`
        Name and path of file containing the X-ray line info.
    contfile: `str`
        Name and path of file containing the X-ray continuum info.
    Returns
    -------
    Returns all outputs form chianti_kev_line_common_load() and chianti_kev_cont_common_load().
    """
    zindex_line, line_meta, line_properties, line_intensities = chianti_kev_line_common_load(linefile)
    zindex_cont, continuum_properties = chianti_kev_cont_common_load(contfile)
    if not all(zindex_line == zindex_cont):
        raise ValueError("Mismatch between zindex from line and continuum files.")
    return zindex_line, line_meta, line_properties, line_intensities, continuum_properties


def chianti_kev_line_common_load(linefile=None):
    """
    Read X-ray emission line info needed for the chianti_kev_... functions.
    Parameters
    ----------
    linefile: `str`
        Name of IDL save file containing line info.  If not given it is derived.
    Returns
    -------
    zindex: `numpy.ndarray`
        Indicies of elements as they appear in periodic table.
    line_meta: `dict`
        Various metadata associated with line properties.
    line_properties: `astropy.table.Table`
        Various properties of each lines.
    line_intensities: `astropy.units.Quantity`
        Intensities of each of the lines in line_properties over a temperature axis.
        The array is 2D with axes of (line, temperature axis)
    """

    # Define defaults
    if linefile is None:
        linefile = os.path.join(SSWDB_XRAY_CHIANTI, "chianti_lines_1_10_v71.sav")
        file_check = glob.glob(linefile)
        if file_check == []:
            linefile = os.path.join(SSWDB_XRAY_CHIANTI, "chianti_lines.geny")
            file_check = glob.glob(linefile)
            if file_check == []:
                raise ValueError("line files not found: {0} or {1}".format(
                    os.path.join(SSWDB_XRAY_CHIANTI, "chianti_lines_1_10_v71.sav"), linefile))
            
    if linefile.split(".")[-1] == "sav":
        # Read file
        contents = scipy.io.readsav(linefile)
        zindex = contents["zindex"]
        out = contents["out"]
    elif linefile.split(".")[-1] == "geny":
        # Read file...
        raise NotImplementedError("Reading .geny file not yet implemented.")
    else:
        raise ValueError("unrecognized file type: .{0}. Must be .sav or .geny")

    # Repackage metadata from file.
    date = []
    for date_byte in out["DATE"]:
        date_strings = str(date_byte[3:], 'utf-8').split()
        date.append(parse_time("{0}-{1}-{2} {3}".format(date_strings[3], date_strings[0],
                                                        date_strings[1], date_strings[2])))
    if len(date) == 1:
        date = date[0]
    line_meta = {
        "IONEQ_LOGT": _clean_array_dims(out["IONEQ_LOGT"]),
        "IONEQ_NAME": _clean_string_dims(out["IONEQ_NAME"]),
        "IONEQ_REF": _combine_strings(out["IONEQ_REF"]),
        "WVL_LIMITS": _clean_array_dims(out["WVL_LIMITS"]),
        "MODEL_FILE": _clean_string_dims(out["MODEL_FILE"]),
        "MODEL_NAME": _clean_string_dims(out["MODEL_NAME"]),
        "MODEL_NE": _clean_array_dims(out["MODEL_NE"]),
        "MODEL_PE": _clean_array_dims(out["MODEL_PE"]),
        "MODEL_TE": _clean_array_dims(out["MODEL_TE"]),
        "WVL_UNITS": _clean_units(out["WVL_UNITS"]),
        "INT_UNITS": _clean_units(out["INT_UNITS"]),
        "ADD_PROTONS": _clean_array_dims(out["ADD_PROTONS"], dtype=int),
        "DATE": date,
        "VERSION": _clean_string_dims(out['VERSION']),
        "PHOTOEXCITATION": _clean_array_dims(out["PHOTOEXCITATION"], dtype=int),
        "LOGT_ISOTHERMAL": _clean_array_dims(out["LOGT_ISOTHERMAL"]),
        "LOGEM_ISOTHERMAL": _clean_array_dims(out["LOGEM_ISOTHERMAL"]),
        "chianti_doc": _clean_chianti_doc(contents["chianti_doc"])
        }

    # Repackage out["line"] into a Table with appropriate units.
    # Create a list of tables to make sure all data in file is captured.
    # Although only one iteration is expected.
    line_properties = []
    line_intensities = []
    for lines in out["lines"]:
        line_props = Table()
        line_props["IZ"] = Column(lines["IZ"], description="Atomic number of ion element.")
        line_props["ION"] = Column(lines["ION"],
            description="Integer ionization state in astronomical notation, i.e. ION-1 = negative charge of ion.")
        line_props["IDENT"] = Column(lines["IDENT"])
        line_props["IDENT_LATEX"] = Column(lines["IDENT_LATEX"])
        line_props["SNOTE"] = Column(lines["SNOTE"],
            description="Ion label in astronomical (roman numeral) notation.")
        line_props["LVL1"] = Column(lines["LVL1"])
        line_props["LVL2"] = Column(lines["LVL2"])
        line_props["TMAX"] = Column(lines["TMAX"])
        line_props["WVL"] = Column(lines["WVL"], unit=line_meta["WVL_UNITS"])
        line_props["ENERGY"] = line_props["WVL"].quantity.to(u.keV, equivalencies=u.spectral())
        line_props["FLAG"] = Column(lines["FLAG"])

        # Sort lines in ascending energy.
        ordd = np.argsort(np.array(line_props["WVL"]))[::-1]
        line_props = line_props[ordd]

        # Extract line intensities.
        lines_int_sorted = lines["INT"][ordd]
        line_ints = np.empty((lines_int_sorted.shape[0], lines_int_sorted[0].shape[0]), dtype=float)
        for i in range(line_ints.shape[0]):
            line_ints[i, :] = lines_int_sorted[i]

        # Enter outputs from this iteration into list.
        line_properties.append(line_props)
        line_intensities.append(line_ints)

    # If there is only one element in the line properties, unpack values.
    if len(out["lines"]) == 1:
        line_properties = line_properties[0]
        line_intensities = line_intensities[0]
    
    return zindex, line_meta, line_properties, line_intensities
    

def chianti_kev_cont_common_load(contfile, _extra=None):
    """
    Read X-ray continuum emission info needed for the chianti_kev_... functions.
    Parameters
    ----------
    contfile: `str`
        Name of IDL save file containing continuum info.  If not given it is derived.
    Returns
    -------
    zindex: `numpy.ndarray`
        Indicies of elements as they appear in periodic table.
    continuum_properties: `dict`
        Properties of continuum emission.
    """
    # Define defaults
    if contfile is None:
        contfile = os.path.join(SSWDB_XRAY_CHIANTI, "chianti_cont_1_250_v71.sav")
        file_check = glob.glob(contfile)
        if file_check == []:
            contfile = os.path.join(SSWDB_XRAY_CHIANTI, "chianti_cont.geny")
            file_check = glob.glob(contfile)
            if file_check == []:
                raise ValueError("line files not found: {0}; {1}".format(
                    os.path.join(SSWDB_XRAY_CHIANTI, "chianti_cont_1_250_v71.sav"), contfile))
    # Read file
    if contfile.split(".")[-1] == "sav":
        contents = scipy.io.readsav(contfile)
        zindex = contents["zindex"]
        edge_str = {
                "CONVERSION": _clean_array_dims(contents["edge_str"]["CONVERSION"]),
                "WVL": _clean_array_dims(contents["edge_str"]["WVL"]),
                "WVLEDGE": _clean_array_dims(contents["edge_str"]["WVLEDGE"])
                   }
        continuum_properties = {
                "totcont": contents["totcont"],
                "totcont_lo": contents["totcont_lo"],
                "edge_str": edge_str,
                "ctemp": contents["ctemp"],
                "chianti_doc": _clean_chianti_doc(contents["chianti_doc"])
                               }
    elif contfile.split(".")[-1] == "geny":
        # Read file...
        raise NotImplementedError("Reading .geny file not yet implemented.")
    else:
        raise ValueError("unrecognized file type: .{0}. Must be .sav or .geny")

    return zindex, continuum_properties


def xr_rd_abundance(abundance_type=None, xr_ab_file=None):
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

    xr_ab_file: `str`
        Name and path to abundance file.
        Default= ~/ssw/packages/xray/dbase/chianti/xray_abun_file.genx

    Returns
    -------
    out:
        Array of 50 abundance levels for first 50 elements.

    """
    # If kwargs not set, set defaults
    if abundance_type is None:
        abundance_type = "sun_coronal"
    if xr_ab_file is None:
        xr_ab_file = os.path.expanduser(os.path.join(SSWDB_XRAY_CHIANTI,
                                                     "xray_abun_file.genx"))
    # Read file
    ab_sav = read_abundance_genx(xr_ab_file)
    # Return relevant abundance.
    return ab_sav[abundance_type]


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


def chianti_kev_getp(line_intensities, sline, logt, mgtemp, nsline):
    """Currently only supports single mgtemp input.  IDL supports array."""
    nltemp = len(logt)
    selt = np.digitize( np.log10(mgtemp), logt)-1
    p = np.zeros(nsline)
    indx = selt-1+np.arange(3)
    indx = indx[np.logical_and(indx > 0, indx < (nltemp-1))]
    uu = np.log10(mgtemp)
    p[:] = scipy.interpolate.interp1d(
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


def _clean_array_dims(arr, dtype=None):
    # Initialize a single array to hold contents of input arr.
    result = np.empty(list(arr.shape) + list(arr[0].shape))
    # Combine arrays in arr into single array.
    for i in range(arr.shape[0]):
        result[i] = arr[i]
    # Remove redundant dimensions
    result = np.squeeze(result)
    # If result is now unsized, convert to scalar.
    if result.shape == ():
        result = result.item()
        if dtype is not None:
            dtype(result)
    return result


def _clean_string_dims(arr):
    result = [str(s, 'utf-8') for s in arr]
    if len(result) == 1:
        result = result[0]
    return result


def _combine_strings(arr):
    result = [".".join([str(ss, 'utf-8') for ss in s]) for s in arr]
    if len(result) == 1:
        result = result[0]
    return result


def _clean_units(arr):
    result = []
    for a in arr:
        unit = str(a, 'utf-8')
        unit_components = unit.split()
        for i, component in enumerate(unit_components):
            # Remove plurals
            if component in ["photons", "Angstroms"]:
                component = component[:-1]
            # Insert ** for indices.
            component_minus_split = component.split("-")
            if len(component_minus_split) > 1:
                "**-".join(component_minus_split)
            component_plus_split = component.split("+")
            if len(component_plus_split) > 1:
                "**-".join(component_plus_split)
            unit_components[i] = component
        result.append("*".join(unit_components))
    if len(result) == 1:
        result = result[0]

    return result


def _clean_chianti_doc(arr):
    chianti_doc = {}
    chianti_doc["ion_file"] = str(arr[0][0], 'utf-8')
    chianti_doc["ion_ref"] = "{0}.{1}.{2}".format(str(arr["ion_ref"][0][0], 'utf-8'),
                                                  str(arr["ion_ref"][0][1], 'utf-8'),
                                                  str(arr["ion_ref"][0][2], 'utf-8'))
    chianti_doc["version"] = str(arr[0][2], 'utf-8')
    return chianti_doc
