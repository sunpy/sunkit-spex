from collections import OrderedDict

import numpy as np
import scipy.io
import xarray

import astropy.units as u
from astropy.table import Table

from sunpy.data import manager
from sunpy.io.special.genx import read_genx

__all__ = ['load_chianti_lines_lite', 'load_chianti_continuum',
           'read_abundance_genx', 'load_xray_abundances']


@manager.require('chianti_lines',
                 ['https://soho.nascom.nasa.gov/solarsoft/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav',
                  'https://lmsal.com/solarsoft/ssw/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav'],
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
    CHIANTI File

    By default, this function uses the file located at
    https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav.
    To use a different file (created by CHIANTI and saved as a sav file) call this function in the following way:
    >>> from sunpy.data import manager  # doctest: +SKIP
    >>> with manager.override_file("chianti_lines", uri=filename): # doctest: +SKIP
    ...     line_info = load_chianti_lines_light() # doctest: +SKIP

    where filename is the location of the file to be read.

    Intensity Units

    The line intensities read from the CHIANTI file are in units of ph / cm**2 / s / sr.
    Therefore they are specific intensities, i.e. per steradian, or solid angle.
    Here, let us call these intensities, intensity_per_solid_angle.
    The solid angle is given by flare_area / observer_distance**2.
    Total integrated intensity can be rewritten in terms of volume EM and solid angle:

    intensity = intensity_per_solid_angle_per_volEM * volEM * solid_angle
    intensity = intensity_per_solid_angle / (colEM * flare_area) * (flare_area / observer_dist**2) * volEM
    intensity = intensity_per_solid_angle / colEM / observer_dist**2 * volEM

    i.e. flare area cancels. Therefore:

    intensity = intensity_per_solid_angle / colEM / observer_dist**2 * volEM,

    or, dividing both sides by volEM,

    intensity_per_EM = intensity_per_solid_angle / colEM / observer_dist**2

    In this function, we normalize the intensity by colEM and scale it to the source, i.e.
    intensity_out = intensity_per_solid_angle / colEM * 4 * pi
    Therefore the intensity values output by this function must be multiplied by EM
    and divided by 4 pi observer_dist**2 to get physical values at the observer.
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
        line_elements.append(lines["IZ"] + 1)  # TODO: Confirm lines["IZ"] is indeed atomic number - 1
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
    line_intensities_per_EM_at_source = xarray.DataArray(
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

    return line_intensities_per_EM_at_source


@manager.require('chianti_continuum',
                 ['https://soho.nascom.nasa.gov/solarsoft/packages/xray/dbase/chianti/chianti_cont_1_250_v71.sav',
                  'https://lmsal.com/solarsoft/ssw/packages/xray/dbase/chianti/chianti_cont_1_250_v71.sav'],
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
    By default, this function uses the file located at
    https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_cont_1_250_v71.sav
    To use a different file call this function in the following way:
    >>> from sunpy.data import manager # doctest: +SKIP
    >>> with manager.override_file("chianti_continuum", uri=filename): # doctest: +SKIP
    ...    line_info = load_chianti_lines_light() # doctest: +SKIP

    where filename is the location of the file to be read.
    """
    # Define units
    intensity_unit = u.ph * u.cm**3 / (u.s * u.keV * u.sr)
    temperature_unit = u.K
    wave_unit = u.AA
    # Read file
    contfile = manager.get("chianti_continuum")
    contents = scipy.io.readsav(contfile)
    # Concatenate low and high wavelength intensity arrays.
    intensities = np.concatenate((contents["totcont_lo"], contents["totcont"]), axis=-1)
    # Integrate over sphere surface of radius equal to observer distance
    # to get intensity at source. This means that physical intensities can
    # be calculated by dividing by 4 * pi * R**2 where R is the observer distance.
    intensities *= 4 * np.pi
    intensity_unit *= u.sr
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
                 ['https://soho.nascom.nasa.gov/solarsoft/packages/xray/dbase/chianti/xray_abun_file.genx',
                  'https://lmsal.com/solarsoft/ssw/packages/xray/dbase/chianti/xray_abun_file.genx'],
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
        contents.pop("header")
    except KeyError:
        pass
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
