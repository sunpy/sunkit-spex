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


define_continuum_parameters()
define_default_abundances()
