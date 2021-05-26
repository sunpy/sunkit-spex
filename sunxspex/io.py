import os.path
import glob
from collections import OrderedDict

import astropy.units as u
import numpy as np
import scipy.io
import xarray
from astropy.table import Table, Column
from sunpy.io.special.genx import read_genx
from sunpy.time import parse_time
from sunpy.data import manager

__all__  = ['load_chianti_lines_lite', 'chianti_kev_line_common_load_light', 'chianti_kev_cont_common_load',
            'read_abundance_genx', 'load_xray_abundances']


@manager.require('chianti_lines',
                 ['https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav'],
                  '2046d818efec207a83e9c5cc6ba4a5fa8574bf8c2bd8a6bb9801e4b8a2a0c677')
def load_chianti_lines_lite():
    """
    Read X-ray emission line info from an IDL sav file produced by CHIANTI.

    This function does not read all data in the file, but only that required to calculate the
    observed X-ray spectrum.

    Returns
    -------
    line_intensities_at_source: `xarray.DataArray`
        Intensities of each of each line as a function of temperature and
        associated metadata and coordinates.

    Notes
    -----
    By default, this function uses reads the file located at
    https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav.
    To read a different file call this function in the following way:
    >>> from sunpy.data import manager
    >>> with manager.override_file("chianti_lines", uri=filename):
            line_info = load_chianti_lines_light()

    where filename is the location of the file to be read.
    """
    # Read linefile
    linefile = manager.get('chianti_lines')
    contents = scipy.io.readsav(linefile)
    out = contents["out"]

    # Define units
    wvl_units = _clean_units(out["WVL_UNITS"])
    int_units = _clean_units(out["INT_UNITS"])
    energy_unit = u.keV

    # Extract line info and convert from wavelength to energy.
    line_intensities = []
    line_elements = []
    line_peak_energies = []
    for j, lines in enumerate(out["lines"]):
        # Extract line element index and peak energy.
        line_elements.append(lines["IZ"] + 1)  #TODO: Confirm lines["IZ"] is indeed atomic number - 1
        line_peak_energies.append(u.Quantity(lines["WVL"], unit=wvl_units).to(
            energy_unit, equivalencies=u.spectral()))
        # Sort line info in ascending energy.
        ordd = np.argsort(np.array(line_peak_energies[j]))
        line_elements[j] = line_elements[j][ordd]
        line_peak_energies[j] = line_peak_energies[j][ordd]
        # Extract line intensities.
        line_intensities.append(_extract_line_intensities(lines["INT"][ordd]) * int_units)

    # If there is only one element in the line properties, unpack values.
    if len(out["lines"]) == 1:
        line_elements = line_elements[0]
        line_peak_energies = line_peak_energies[0]
        line_intensities = line_intensities[0]

    # Normalize line intensities by EM and integrate over whole sky to get intensity at source.
    # This means that physical intensities can be calculated by dividing by
    # 4 * pi * R**2 where R is the observer distance.
    line_colEMs = 10.**_clean_array_dims(out["LOGEM_ISOTHERMAL"]) / u.cm**5
    line_intensities /= line_colEMs
    line_intensities *= 4 * np.pi * u.sr

    # Put data into intuitive structure and return it.
    line_intensities_at_source = xarray.DataArray(
        line_intensities.value,
        dims=["lines", "temperature"],
        coords={"logT": ("temperature", _clean_array_dims(out["LOGT_ISOTHERMAL"])),
                "peak_energy": ("lines", line_peak_energies),
                "atomic_number": ("lines", line_elements)},
        attrs={"units": {"data": line_intensities.unit,
                         "peak_energy": line_peak_energies.unit},
               "file": linefile,
               "element_index": contents["zindex"],
               "chianti_doc": _clean_chianti_doc(contents["chianti_doc"])})

    return line_intensities_at_source



def chianti_kev_line_common_load_light():
    """
    Read only X-ray emission line info needed for the chianti_kev functions.
    Unlike chianti_kev_line_common_load which formats and returns all in the file,
    this function only returns the data required by the ChiantiKevLines class.
    Returns
    -------
    zindex: `numpy.ndarray`
        Indicies of elements as they appear in periodic table.
    line_peak_energies: `astropy.units.Quantity` of `list` of `astropy.units.Quantity`
        The energies of the line peaks.
    line_logT_bins: `numpy.ndarray` or `list` of `numpy.ndarray`
        The log10 temperature bins over which the line intensities are known.
    line_colEMs: `astropy.units.Quantity` of `list` of `astropy.units.Quantity`
        The column emission measures used to calculate the intensities for each line.
    line_element_indices: `numpy.ndarray` or `list` of `numpy.ndarray`
        The atomic number of each line minus 1.
    line_intensities: `astropy.units.Quantity`
        Intensities of each of the lines in line_properties over a temperature axis.
        The array is 2D with axes of (line, temperature axis)
    """
    # Read linefile
    contents = _read_linefile()
    out = contents["out"]

    zindex = contents["zindex"]
    line_logT_bins = _clean_array_dims(out["LOGT_ISOTHERMAL"])
    line_colEMs = 10.**_clean_array_dims(out["LOGEM_ISOTHERMAL"]) / u.cm**5
    wvl_units = _clean_units(out["WVL_UNITS"])
    int_units = _clean_units(out["INT_UNITS"])

    line_intensities = []
    line_element_indices = []
    line_peak_energies = []
    for j, lines in enumerate(out["lines"]):
        line_element_indices.append(lines["IZ"])
        line_peak_energies.append(u.Quantity(lines["WVL"], unit=wvl_units).to(u.keV, equivalencies=u.spectral()))

        # Sort lines in ascending energy.
        ordd = np.argsort(np.array(line_peak_energies[j]))
        line_element_indices[j] = line_element_indices[j][ordd]
        line_peak_energies[j] = line_peak_energies[j][ordd]

        # Extract line intensities.
        line_intensities.append(_extract_line_intensities(lines["INT"][ordd]) * int_units)

    # If there is only one element in the line properties, unpack values.
    if len(out["lines"]) == 1:
        line_element_indices = line_element_indices[0]
        line_peak_energies = line_peak_energies[0]
        line_intensities = line_intensities[0]

    return zindex, line_peak_energies, line_logT_bins, line_colEMs, line_element_indices, \
        line_intensities


def load_chianti_lines():
    """
    Read file containing X-ray emission line info needed for the chianti_kev functions.

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
    # Read linefile.
    contents = _read_linefile()
    zindex = contents["zindex"]
    out = contents["out"]

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
        line_intensities.append(_extract_line_intensities(lines["INT"][ordd]))

        # Enter outputs from this iteration into list.
        line_properties.append(line_props)
        line_intensities.append(line_ints)

    # If there is only one element in the line properties, unpack values.
    if len(out["lines"]) == 1:
        line_properties = line_properties[0]
        line_intensities = line_intensities[0]

    return zindex, line_meta, line_properties, line_intensities * line_meta["INT_UNITS"]


@manager.require('chianti_continuum',
                 ['https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_cont_1_250_v71.sav'],
                 'aadf4355931b4c241ac2cd5669e89928615dc1b55c9fce49a155b70915a454dd')
def load_chianti_continuum():
    """
    Read X-ray continuum emission info from an IDL sav file produced by CHIANTI

    Returns
    -------
    continuum_intensities: `xarray.DataArray`
        Continuum intensity as a function of element, temperature and energy/wavelength
        and associated metadata and coordinates.

    Notes
    -----
    By default, this function uses reads the file located at
    https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_cont_1_250_v71.sav
    To read a different file call this function in the following way:
    >>> from sunpy.data import manager
    >>> with manager.override_file("chianti_continuum", uri=filename):
            line_info = load_chianti_lines_light()
    
    where filename is the location of the file to be read.
    """
    # Define units
    intensity_unit = u.ph / (u.cm**2 * u.s * u.sr)
    temperature_unit = u.K
    wave_unit = u.AA
    # Read file
    contfile = manager.get("chianti_continuum")
    contents = scipy.io.readsav(contfile)
    # Concatenate low and high wavelength intensity arrays.
    intensities = np.concatenate((contents["totcont"], contents["totcont_lo"]), axis=-1)
    # Put file data into intuitive structure and return data.
    continuum_intensities = xarray.DataArray(
        intensities,
        dims=["element_index", "temperature", "wavelength"],
        coords={"element_index": contents["zindex"],
                "temperature": contents["ctemp"],
                "wavelength": _clean_array_dims(contents["edge_str"]["WVL"])},
        attrs={"units": {"data": intensity_unit,
                         "temperature": temperature_unit,
                         "wavelength": wave_unit},
               "file": contfile,
               "wavelength_edges": _clean_array_dims(contents["edge_str"]["WVLEDGE"]) * wave_unit,
               "chianti_doc": _clean_chianti_doc(contents["chianti_doc"])
               })
    return continuum_intensities


@manager.require('xray_abundances',
                 ['https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/xray_abun_file.genx'],
                 '92c0e1f9a83da393cc38840752fda5a5b44c5b18a4946e5bf12c208771fe0fd3')
def load_xray_abundances(abundance_type=None):
    """
    Returns the abundances written in the xray_abun_file.genx

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

    Returns
    -------
    out:
        Array of 50 abundance levels for first 50 elements.

    """
    # If kwargs not set, set defaults
    if abundance_type is None:
        abundance_type = "sun_coronal"
    xray_abundance_file = manager.get("xray_abundances")
    # Read file
    contents = read_abundance_genx(xray_abundance_file)
    # Restructure data into an easier form.
    try:
        header = contents.pop("header")
    except KeyError:
        header = None
    n_elements = len(contents[list(contents.keys())[0]])
    columns = [np.arange(1, n_elements+1)] + list(contents.values())
    names = ["atomic number"] + list(contents.keys())
    abundances = Table(columns, names=names)

    return abundances


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


@manager.require('chianti_lines',
                 ['https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav'],
                  '2046d818efec207a83e9c5cc6ba4a5fa8574bf8c2bd8a6bb9801e4b8a2a0c677')
def _read_linefile():
    linefile = manager.get('chianti_lines')
    # Read file
    contents = scipy.io.readsav(linefile)
    return contents


def _extract_line_intensities(lines_int_sorted):
    line_ints = np.empty((lines_int_sorted.shape[0], lines_int_sorted[0].shape[0]), dtype=float)
    for i in range(line_ints.shape[0]):
        line_ints[i, :] = lines_int_sorted[i]
    return line_ints

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

    return u.Unit(result)


def _clean_chianti_doc(arr):
    chianti_doc = {}
    chianti_doc["ion_file"] = str(arr[0][0], 'utf-8')
    chianti_doc["ion_ref"] = "{0}.{1}.{2}".format(str(arr["ion_ref"][0][0], 'utf-8'),
                                                  str(arr["ion_ref"][0][1], 'utf-8'),
                                                  str(arr["ion_ref"][0][2], 'utf-8'))
    chianti_doc["version"] = str(arr[0][2], 'utf-8')
    return chianti_doc
