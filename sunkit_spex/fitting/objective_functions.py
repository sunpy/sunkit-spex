import astropy.units as u
import numpy as np

from . import spectra
from . import fit_models

'''
Various objective functions that may be mixed/matched for fitting

An objective function should be LARGE for GOOD results (--> posinf)
and SMALL (--> neginf) for BAD results
'''

class XrayObjectiveFunction:
    def __init__(self, data: spectra.XraySpectrum, model: fit_models.PhotonModel):
        self.data = data
        self.model = model

    def evaluate(self, new_params: list[float]) -> float:
        raise NotImplementedError
        _ = new_params


class XrayChi2(XrayObjectiveFunction):
    def __init__(self, data: spectra.XraySpectrum, model: fit_models.PhotonModel):
        super().__init__(data, model)

    def evaluate(self, new_params: list[float]) -> float:
        orig_params = [p.value.value for p in self.model.flat_parameters()]
        self.model.update_parameters(*new_params)
        priors = sum(
            p.evaluate_prior_logpdf()
            for p in self.model.flat_parameters()
        )
        if np.isneginf(priors) or np.isnan(priors):
            self.model.update_parameters(*orig_params)
            return -np.inf

        model = self.evaluate_model()
        errs = self.data.counts_error
        data = self.data.counts
        chi = ((model - data) / errs).to_value(u.one)
        chi[((data - model) == 0) | (errs == 0)] = 0
        return -np.sum(chi*chi) + priors

    def evaluate_model(self) -> u.ct:
        ret = (
            self.data.response_matrix @
            (
                self.model.evaluate(self.data.photon_energy_edges) *
                self.data.photons_de
            )
        )
        ret *= self.data.effective_exposure
        return ret.to(u.ct)
