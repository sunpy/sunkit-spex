from dataclasses import dataclass
import numpy as np

@dataclass
class SpecFileInfo:
    '''
    Spectrum data for STIX and RHESSI classes.
    Please change eventually...
    This is a bodge to make the code coupling less painful
    '''
    count_bins: np.ndarray
    time_bins: np.ndarray
    livetime: np.ndarray

    counts: np.ndarray
    counts_error: np.ndarray

    count_rate: np.ndarray
    count_rate_error: np.ndarray


@dataclass
class SrmFileInfo:
    photon_bin_edges: np.ndarray
    count_bin_edges: np.ndarray
    srm: np.ndarray
