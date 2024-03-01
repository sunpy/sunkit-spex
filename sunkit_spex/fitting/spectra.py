import astropy.units as u
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


@dataclass
class StrippedSpectrum:
    ''' Take units off of a counts spectrum '''
    counts: np.ndarray
    counts_error: np.ndarray
    count_energy_edges: np.ndarray
    photon_energy_edges: np.ndarray
    response_matrix: np.ndarray
    effective_exposure: np.ndarray

    @classmethod
    def from_xray_spectrum(cls, with_units: XraySpectrum):
        return cls(
            with_units.counts.to_value(u.ct),
            with_units.counts_error.to_value(u.ct),
            with_units.count_energy_edges.to_value(u.keV),
            with_units.photon_energy_edges.to_value(u.keV),
            with_units.response_matrix.to_value(u.cm**2 * u.ct / u.ph),
            with_units.effective_exposure.to_value(u.s)
        )
