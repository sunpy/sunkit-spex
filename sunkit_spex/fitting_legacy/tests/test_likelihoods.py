import numpy as np
from numpy.testing import assert_allclose

from sunkit_spex.fitting_legacy.likelihoods import LogLikelihoods


def test_cstat():

    # c-stat shouldn't depend on errors, assumes poissonian
    observed_count_errors = 0

    # fake counts, fake model, and their expected cstat log-likelihood output
    # these should default to Poisson log-likelihood as well
    fake_model1p, fake_counts1p, cstat1p = [0], [0], -np.inf
    fake_model2p, fake_counts2p, cstat2p = [0], [1], -np.inf
    fake_model3p, fake_counts3p, cstat3p = [1], [0], -1

    # calculate c-stat
    cstat1p_result = LogLikelihoods().cstat_loglikelihood(model_counts=fake_model1p, observed_counts=fake_counts1p, observed_count_errors=observed_count_errors)
    cstat2p_result = LogLikelihoods().cstat_loglikelihood(model_counts=fake_model2p, observed_counts=fake_counts2p, observed_count_errors=observed_count_errors)
    cstat3p_result = LogLikelihoods().cstat_loglikelihood(model_counts=fake_model3p, observed_counts=fake_counts3p, observed_count_errors=observed_count_errors)

    # check agreement
    assert_allclose(cstat1p_result, cstat1p, rtol=1e-3)
    assert_allclose(cstat2p_result, cstat2p, rtol=1e-3)
    assert_allclose(cstat3p_result, cstat3p, rtol=1e-3)

    # calculate the Poissonian values
    poisson1p_result = LogLikelihoods().poisson_loglikelihood(model_counts=fake_model1p, observed_counts=fake_counts1p, observed_count_errors=observed_count_errors)
    poisson2p_result = LogLikelihoods().poisson_loglikelihood(model_counts=fake_model2p, observed_counts=fake_counts2p, observed_count_errors=observed_count_errors)
    poisson3p_result = LogLikelihoods().poisson_loglikelihood(model_counts=fake_model3p, observed_counts=fake_counts3p, observed_count_errors=observed_count_errors)

    # compare where C-stat should produce same as Poissonian
    assert_allclose(cstat1p_result, poisson1p_result, rtol=1e-3)
    assert_allclose(cstat2p_result, poisson2p_result, rtol=1e-3)
    assert_allclose(cstat3p_result, poisson3p_result, rtol=1e-3)

    # other test cases
    fake_model1, fake_counts1, cstat1 = [10], [10], 0
    fake_model2, fake_counts2, cstat2 = [10], [50], -40.472
    fake_model3, fake_counts3, cstat3 = [10, 10], [10, 50], cstat1+cstat2
    fake_model4, fake_counts4, cstat4 = [10, 50], [10, 0], -50
    fake_model5, fake_counts5, cstat5 = [10, 10, 10, 50], [10, 50, 10, 0], cstat3+cstat4

    # calculate c-stat
    cstat1_result = LogLikelihoods().cstat_loglikelihood(model_counts=fake_model1, observed_counts=fake_counts1, observed_count_errors=observed_count_errors)
    cstat2_result = LogLikelihoods().cstat_loglikelihood(model_counts=fake_model2, observed_counts=fake_counts2, observed_count_errors=observed_count_errors)
    cstat3_result = LogLikelihoods().cstat_loglikelihood(model_counts=fake_model3, observed_counts=fake_counts3, observed_count_errors=observed_count_errors)
    cstat4_result = LogLikelihoods().cstat_loglikelihood(model_counts=fake_model4, observed_counts=fake_counts4, observed_count_errors=observed_count_errors)
    cstat5_result = LogLikelihoods().cstat_loglikelihood(model_counts=fake_model5, observed_counts=fake_counts5, observed_count_errors=observed_count_errors)

    # check agreement
    assert_allclose(cstat1_result, cstat1, rtol=1e-3)
    assert_allclose(cstat2_result, cstat2, rtol=1e-3)
    assert_allclose(cstat3_result, cstat3, rtol=1e-3)
    assert_allclose(cstat4_result, cstat4, rtol=1e-3)
    assert_allclose(cstat5_result, cstat5, rtol=1e-3)
