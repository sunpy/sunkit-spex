import os.path
import glob
from collections import OrderedDict

import numpy as np
import astropy.units as u
from astropy.table import Table, Column
import scipy.io
from sunpy.io.special.genx import read_genx
from sunpy.time import parse_time

SSWDB_XRAY_CHIANTI = os.path.expanduser(os.path.join("~", "ssw", "packages",
                                                     "xray", "dbase", "chianti"))
FILE_IN = "/Users/dnryan/ssw/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav"


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
