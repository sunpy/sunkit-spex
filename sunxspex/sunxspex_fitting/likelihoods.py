"""
The following code contains the default log-likelihoods/fit statistics used for model fitting.
"""

import numpy as np
from scipy.special import factorial

__all__ = ["LogLikelihoods"]

class LogLikelihoods:
    """
    This class's job is to hold all the log-likelihoods/fit statistics in the one place.

    All are to be maximised to find the best fit.

    Methods
    -------
    remove_non_numbers : _lhoods (array)
            Takes an array and removes Nans, infs, -infs.

    gaussian_loglikelihood : model_counts (array), observed_counts (array), observed_count_errors (array)
            Gaussian log-likelihood function. Access via `log_likelihoods` attribute.

    chi2 : model_counts (array), observed_counts (array), observed_count_errors (array)
            Chi-squared fit statistic function. Access via `log_likelihoods` attribute.

    poisson_loglikelihood : model_counts (array), observed_counts (array), observed_count_errors (array)
            Poissonian log-likelihood function. Access via `log_likelihoods` attribute.

    cash_loglikelihood : model_counts (array), observed_counts (array), observed_count_errors (array)
            Cash log-likelihood function. Access via `log_likelihoods` attribute.

    cstat_loglikelihood : model_counts (array), observed_counts (array), observed_count_errors (array)
            C-stat log-likelihood function. Access via `log_likelihoods` attribute.

    Attributes
    ----------
    log_likelihoods : dict
            Name of each log-likelihoods/fit statistic (dict. key) with the method object (dict. value).


    _construction_string : string
            String to be returned from __repr__() dunder method.

    Examples
    --------
    # calculate how well a model fits data with the c-stat statistic
    ll = LogLikelihoods()
    model_counts, observed_counts, observed_count_errors = modelled_counts, counts_data, count_error
    log_likelihoods_or_fit_statistic = ll.loglikelihoods["cstat"](model_counts, observed_counts, observed_count_errors)
    """
    def __init__(self):
        """Construct a string to show how the class was constructed (`_construction_string`) and set the `log_likelihoods` dictionary attribute."""

        self._construction_string = "LogLikelihoods()"
        # dictionary of all likelihood functions? E.g.,
        self.log_likelihoods = {"gaussian":self.gaussian_loglikelihood,
                                "chi2":self.chi2,
                                "poisson":self.poisson_loglikelihood,
                                "cash":self.cash_loglikelihood,
                                "cstat":self.cstat_loglikelihood}

    def _remove_nans(self, _lhoods):
        """ Removes Nans in the output array from any /0 data entries.

        This does not affect the fitting since this will be true for all models fitting the same data.
        This can only complicate things if your model gave out nans or infs for some reason which would
        break things anyway. If using certain statistics (e.g., cstat, etc.) and your model gives out 0s
        this can also be problematic; however, this shouldn't be the case when working with Poisson
        statistics anyway (i.e., the probability of seeing >0 events from a modelled system that produces
        0 events per time is 0).

        Parameters
        ----------
        _lhoods : 1d array
                The fit statistic for each bin comparing a model to the data.

        Returns
        -------
        Input array but with Nans removed.
        """
        return _lhoods[~np.isnan(_lhoods)]

    def remove_non_numbers(self, _lhoods):
        """ Removes Nans, infs, -infs in the output array from any /0 data entries.

        This does not affect the fitting since this will be true for all models fitting the same data.
        This can only complicate things if your model gave out nans or infs for some reason which would
        break things anyway. If using certain statistics (e.g., cstat, etc.) and your model gives out 0s
        this can also be problematic; however, this shouldn't be the case when working with Poisson
        statistics (i.e., the probability of seeing >0 events from a modelled system that produces
        0 events per time is 0).

        Parameters
        ----------
        _lhoods : 1d array
                The fit statistic for each bin comparing a model to the data.

        Returns
        -------
        Input array but with Nans, infs, -infs removed.
        """
        _lhoods = self._remove_nans(_lhoods)
        return _lhoods[np.isfinite(_lhoods)]

    def _check_numbers_left(self, remaining):
        """ Check if there are any numbers left after all Nans, infs, and -infs have been remove and return the sum.

        If no numbers left then return -np.inf since it is a rubbish fit.

        Parameters
        ----------
        remaining : 1d array
                The fit statistic for each bin comparing a model to the data after any Nans,
                infs, -infs have been removed.

        Returns
        -------
        Sum of the bin fit statistics, or -np.inf if nothing to sum.
        """
        if len(remaining)==0:
            return -np.inf
        else:
            return np.sum(remaining)

    def gaussian_loglikelihood(self, model_counts, observed_counts, observed_count_errors):
        """ Gaussian log-likelihood.

        .. math::
         ln(L_{Gauss}) = -\frac{N}{2} ln(2\pi D^{2}) + \frac{1}{2}\Chi^{2}

        where N is the number of observed bins, D is the data, and chi-squared is negative
        its usual, minimise version and is located in the (to be maximised() method.

        Parameters
        ----------
        model_counts, observed_counts, observed_count_errors : 1d array
                The model counts, observed counts, and observed count errors.

        Returns
        -------
        A float, the gaussian log-likelihood (to be maximised).
        """

        likelihoods = -(len(observed_counts)/2) * np.log(2*np.pi*np.array(observed_count_errors)**2) + (1/2)* self.chi2(model_counts, observed_counts, observed_count_errors)

        # best value is first whole term, if the chi squared section has any value then it is always subtracted
        return self._check_numbers_left(self.remove_non_numbers(likelihoods)) # =ln(L)

    def chi2(self, model_counts, observed_counts, observed_count_errors):
        """ Chi-squared fit statistic.

        .. math::
         \Chi^{2} = - (\frac{(D - \mu)^{2}}{\sigma})^{2}

        where D is the data, mu is the model counts, and sigma is the error on the observed data.

        Parameters
        ----------
        model_counts, observed_counts, observed_count_errors : 1d array
                The model counts, observed counts, and observed count errors.

        Returns
        -------
        A float, the chi-squared fit statistic (to be maximised).
        """

        likelihoods = -( (np.array(observed_counts)-np.array(model_counts).flatten()) / np.array(observed_count_errors) )**2

        # best value is 0, every other value is negative
        return self._check_numbers_left(self.remove_non_numbers(likelihoods)) #

    def poisson_loglikelihood(self, model_counts, observed_counts, observed_count_errors):
        """ Poissonian log-likelihood.

        .. math::
         ln(L_{Poisson}) = D ln(\mu) - \mu - ln(D!)

        where D is the data and mu is the model counts.

        Parameters
        ----------
        model_counts, observed_counts, observed_count_errors : 1d array
                The model counts, observed counts, and observed count errors.

        Returns
        -------
        A float, the poissonian log-likelihood (to be maximised).
        """

        # proper Poisson log-likelihood, factorial and all
        likelihoods = np.array(observed_counts) * np.log(np.array(model_counts)) - np.array(model_counts) - np.log(factorial(observed_counts))

        return self._check_numbers_left(self.remove_non_numbers(likelihoods))

    def cash_loglikelihood(self, model_counts, observed_counts, observed_count_errors):
        """ Cash log-likelihood.

        A simplification of the poissonian log-likelihood where the independent data term is
        neglected. Since the data term is neglected the absolute number does not say much
        about the absolute goodness of fit as more observations means a higher natural number.

        .. math::
         ln(L_{Poisson}) = D ln(\mu) - \mu

        where D is the data and mu is the model counts.

        [1] Cash, W., ApJ 228, 939, 1979 (http://articles.adsabs.harvard.edu/pdf/1979ApJ...228..939C)

        Parameters
        ----------
        model_counts, observed_counts, observed_count_errors : 1d array
                The model counts, observed counts, and observed count errors.

        Returns
        -------
        A float, the cash log-likelihood (to be maximised).
        """

        # needs to be based on counts so multiply out the livetime and keV^-1
        # Cash - more bins=higher natural value, i.e., the number doesn't say much about the absolute goodness of fit
        likelihoods = np.array(observed_counts) * np.log(np.array(model_counts)) - np.array(model_counts)

        return self._check_numbers_left(self.remove_non_numbers(likelihoods))

    def cstat_loglikelihood(self, model_counts, observed_counts, observed_count_errors):
        """ C-stat log-likelihood.

        A simplification of the poissonian log-likelihood where the independent data term
        is replaced using sterling's approximation. Used in the XSPEC fitting software.
        Unlike the cash statistic, the resulting value retains some meaning of the absolute
        goodness of fit.

        .. math::
         ln(L_{Poisson}) = D (ln(\frac{\mu}{D}) + 1) - \mu

        where D is the data and mu is the model counts.

        [1] https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html

        Parameters
        ----------
        model_counts, observed_counts, observed_count_errors : 1d array
                The model counts, observed counts, and observed count errors.

        Returns
        -------
        A float, the c-stat log-likelihood (to be maximised).
        """

        # C-stat (XSPEC) - has data term in it so the lower the -2*ln(L) number the better the fit, this is the ln(L) though
        # Best value is 0, if obs_counts>mod_counts or obs_counts<mod_counts then likelihoods<0
        # Obvious since e^0=1
        likelihoods = np.array(observed_counts) * ( np.log(np.array(model_counts)/np.array(observed_counts)) + 1 ) - np.array(model_counts)

        return self._check_numbers_left(self.remove_non_numbers(likelihoods))

    def __repr__(self):
        """Provide a representation to construct the class from scratch."""
        return self._construction_string

    def __str__(self):
        """Provide a printable, user friendly representation of what the class contains."""
        return f"Different log-likelihoods/fit statistics available are: {self.log_likelihoods}"
