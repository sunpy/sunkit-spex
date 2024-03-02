import astropy.units as u
from astropy.utils import lazyproperty
from dataclasses import dataclass
import numpy as np


@dataclass
class XraySpectrum:
    '''
    Minimum required data for a fit
    '''
    counts: u.ct
    counts_error: u.ct
    count_energy_edges: u.keV
    photon_energy_edges: u.keV
    response_matrix: (u.cm**2 * u.ct / u.ph) # type: ignore
    effective_exposure: u.s

    @lazyproperty
    def counts_de(self) -> u.keV:
        return np.diff(self.count_energy_edges)

    @lazyproperty
    def photons_de(self) -> u.keV:
        return np.diff(self.photon_energy_edges)
