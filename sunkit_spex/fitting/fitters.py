import copy

import astropy.units as u
from astropy.utils import lazyproperty
import dill
import emcee
from scipy import optimize
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


class PhotonFitter(Fitter):
    def __init__(
        self,
        spectral_data: spectra.XraySpectrum,
        photon_model: fit_models.PhotonModel
    ):
        self.data = copy.deepcopy(spectral_data)
        self.photon_model = photon_model
        self.emcee_sampler = None

    @lazyproperty
    def _count_de(self):
        return np.diff(self.data.count_energy_edges)

    def evaluate_model(self) -> u.ct:
        ret = (
            self.data.response_matrix @
            (
                self.photon_model.evaluate(self.data.photon_energy_edges) *
                self.data.photons_de
            )
        )
        ret *= self.data.effective_exposure
        return ret.to(u.ct)

    def model_parameters(self) -> dict[str, fit_models.ModelParameter]:
        return self.photon_model.current_parameters()

    def model_parameters_no_units(self) -> tuple[float]:
        return [p.value.value for p in self.model_parameters().values()]


class Chi2PhotonFitter(PhotonFitter):
    ''' Photon fitter which computes a chi2 for the data including prior
        distribution info (bounds in the case of a uniform prior)
    '''
    def __init__(
        self,
        spectral_data: spectra.XraySpectrum,
        photon_model: fit_models.PhotonModel
    ):
        super().__init__(spectral_data, photon_model)

    def chi2(self, new_params):
        self.photon_model.update_parameters(*new_params)
        model = self.evaluate_model()
        errs = self.data.counts_error
        data = self.data.counts
        chi = np.nan_to_num(((model - data) / errs).to_value(u.one))

        priors = sum(
            p.evaluate_prior_logpdf()
            for p in self.photon_model.current_parameters().values()
        )
        # minimizing chi2, so the negative of the logpdf
        # is what we want
        return np.sum(chi**2) - priors


class MonteCarloChi2Fitter(Chi2PhotonFitter):
    ''' Fit a model to data using emcee and chi2 minimization '''
    def __init__(
        self,
        spectral_data: spectra.XraySpectrum,
        photon_model: fit_models.PhotonModel
    ):
        super().__init__(spectral_data, photon_model)

    def perform_fit(self, num_steps: int, emcee_kwargs: dict=None) -> None:
        emcee_kwargs = emcee_kwargs or dict()
        no_units = self.model_parameters_no_units()
        ndim = len(no_units)
        nwalkers = emcee_kwargs.pop('nwalkers', 2*ndim)

        self.emcee_sampler = emcee.EnsembleSampler(
            nwalkers=nwalkers,
            ndim=ndim,
            log_prob_fn=self.log_prob_func,
            **emcee_kwargs
        )

        # Give emcee the shape of parameters it expects
        initial = np.tile(
            no_units, nwalkers
        ).reshape(nwalkers, ndim)
        initial *= np.random.random(size=ndim*nwalkers).reshape(nwalkers, ndim)
        self.emcee_sampler.run_mcmc(initial, num_steps)

    def log_prob_func(self, parameters):
        # self.photon_model.update_parameters(*parameters)
        # emcee maximizes a probability,
        # but we want to minimize chi2,
        # so return a negative
        ret = self.chi2(parameters)
        if np.any(np.isnan(ret)):
            return -np.inf
        return -ret


class NonlinearMinimizer(Chi2PhotonFitter):
    def __init__(
        self,
        spectral_data: spectra.XraySpectrum,
        photon_model: fit_models.PhotonModel
    ):
        super().__init__(spectral_data, photon_model)
        self.optimize_result = None

    def perform_fit(self, **minimize_opts):
        defaults = dict(method='Nelder-Mead')
        defaults.update(minimize_opts)
        self.optimize_result = optimize.minimize(
            fun=self.chi2,
            x0=self.model_parameters_no_units(),
            **defaults
        )
