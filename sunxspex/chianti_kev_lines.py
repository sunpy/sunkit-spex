import copy

import numpy as np
import astropy.units as u
import scipy.io

def chianti_kev_lines(energy_edges, temperature, emission_measure=1e44/u.cm**3,
                      relative_abundances=None):
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

    Returns
    -------
    Flux: `astropy.units.Quantity`
        

    """
    # For ease of calculation, convert inputs to standard units and
    # scale to manageable numbers.
    em_factor = 1e44
    temp = temperature.to(u.MK).value
    emission_measure = emission_measure.to(u.cm**(-3)).value / em_factor
    energy_edges = energy_edges.to(u.keV).value

    mgtemp = temp * 1e6
    uu = np.log10(mgtemp)

    # Location of file giving intensity as a function of temperatures at all line energies:
    # https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav
    line_energies, log10_temp_K_range, line_intensities, line_element_indices, element_indices, \
      line_iz = _extract_from_chianti_lines_sav()
    energy = np.linspace(3, 9, 1001) * u.keV

    # Load abundances
    #abundance = xr_rd_abundance(_extra=_extra) ;controlled thru environ vars, XR_AB_FILE, XR_AB_TYPE
    #len_abundances = len(abundance)
    

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
    spectrum = np.zeros((nenrg, mtemp))

    # Rename variables to IDL names for ease of comparison.
    eline = copy.copy(line_energies)
    logt = copy.copy(log10_temp_K_range)
    out_lines_iz = copy.copy(line_iz)
    sline = copy.copy(line_indices)
    nsline = copy.copy(n_line_indices)
    zindex = copy.copy(element_indices)

    if n_line_indices > 0:
        eline = eline[sline]
        nltemp = len(logt)
        selt = np.digitize( np.log10(temp*1e6), logt)-1
        p = np.zeros((mtemp, nsline))
	    for i in range(mtemp):
            indx = selt[i]-1+np.arange(3)
            indx = indx[np.logical_and(indx > 0, indx < (nltemp-1))]
            tband = 
            p[i, :] = scipy.interpolate.interp1d(
                logt[indx], line_intensities[sline][:, indx], kind="quadratic")(uu).squeeze()[:]

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

        hhh   = histogram( eline - energm[iline] , min=-10., max=10., bin=10, rev=rr)
        wghtlineenv = getenv('wghtline')
        wghtline = wghtlineenv eq 'T' or not (max(wedg) lt .01) ;(keV)
        wghtline = wghtlineenv eq 'F' ? 0 : wghtline
        # look for wide bins next to line bins, if too wide x 2 eline bin width
        # then don't spread out lines
        wedg0 = wedg[iline]
        wedg0a= wedg[iline-1>0]
        wedg0b= wedg[iline+1<(n_elements(wedg)-1)]
        wghtline =wghtline and np.max([(wedg0a/wedg0).max(), (wedg0b/wedg0).max()]) < 2.) \
          and (wedg0.max() < 1.5)
        
        

def _extract_from_chianti_lines_sav():
    """
    Extracts data from the CHIANTI lines file relevant for chianti_kev_lines function.

    Returns
    -------

    """
    # Read file.
    struct = scipy.io.readsav("/Users/dnryan/ssw/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav")
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
