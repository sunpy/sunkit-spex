import copy

import astropy.units as u
import dill
import emcee
import numpy as np

from . import spectra
from . import fit_models


class Fitter:
    def __init__(self):
        raise NotImplementedError

    def evaluate_model(self):
        raise NotImplementedError

    def perform_fit(self):
        raise NotImplementedError

    def save(self, fn: str='fitter.dill'):
        with open(fn, 'wb') as f:
            # Dill = better pickle
            # Used for multiprocessing
            dill.dump(self, f)


class MonteCarloChi2Fitter(Fitter):
    ''' Fit a model to data using emcee and chi2 minimization '''
    def __init__(
        self,
        spectral_data: spectra.XraySpectrum,
        photon_model: fit_models.PhotonModel
    ):
        self.data = copy.deepcopy(spectral_data)
        self.photon_model = photon_model
        self.emcee_sampler = None

    def evaluate_model(self) -> u.ct:
        ret = (
            self.data.response_matrix @
            self.photon_model.evaluate(self.data.photon_energy_edges)
        )
        ret *= np.diff(self.data.photon_energy_edges) * self.data.effective_exposure
        return ret.to(u.ct)

    def perform_fit(self, num_steps: int, emcee_kwargs: dict=None) -> None:
        def chi2():
            model = self.evaluate_model()
            errs = self.data.counts_error
            data = self.data.counts
            chi = ((model - data) / errs).to(u.one)
            # Use multiplication instead of exponentiation to avoid
            # weird floating point stuff
            return np.sum(chi*chi)

        def prob_func(parameters):
            self.photon_model.update_parameters(*parameters)
            # emcee maximizes a probability,
            # but we want to minimize chi2,
            # so return a negative
            return -chi2()

        emcee_kwargs = emcee_kwargs or dict()
        nwalkers = emcee_kwargs.pop('nwalkers', 4)
        no_units = self.model_parameters_no_units()
        ndim = len(no_units)

        self.emcee_sampler = emcee.EnsembleSampler(
            nwalkers=nwalkers,
            ndim=ndim,
            log_prob_fn=prob_func,
            **emcee_kwargs
        )

        # Give emcee the shape of parameters it expects
        initial = np.tile(
            no_units, nwalkers
        ).reshape(nwalkers, ndim)
        initial *= np.random.random(size=ndim*nwalkers).reshape(nwalkers, ndim)
        self.emcee_sampler.run_mcmc(initial, num_steps)

    def model_parameters(self) -> tuple[u.Quantity]:
        return self.photon_model.current_parameters()

    def model_parameters_no_units(self) -> tuple[float]:
        return [p.value for p in self.model_parameters().values()]
