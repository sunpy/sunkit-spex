import astropy.units as u
import numpy as np
import scipy.stats as st

from . import fit_models
from . import spectra

def linear(slope, intercept):
    photon_bins = np.linspace(2, 100, num=400) << u.keV
    count_bins = photon_bins

    # Response matrix with good units
    # Keep it square so that we don't need to worry about scaling fake model properly
    response = np.eye(photon_bins.size-1) * 0.5
    response = response << u.cm**2 * u.ct / u.ph

    # 'observing time'
    livetime = 0.98
    exposure = (2 << u.s) * livetime

    # data from model which we will fit
    mod = fit_models.Line()
    mod.slope = slope
    mod.intercept = intercept
    model = mod.evaluate(photon_bins)
    model *= np.diff(photon_bins) * exposure
    perfect = response @ model 

    # add Poisson noise
    observed = st.poisson.rvs(perfect.to_value(u.ct))
    error = np.sqrt(observed)

    return spectra.XraySpectrum(
        counts=observed << u.ct,
        counts_error=error << u.ct,
        count_energy_edges=count_bins,
        photon_energy_edges=photon_bins,
        response_matrix=response,
        effective_exposure=exposure
    )
