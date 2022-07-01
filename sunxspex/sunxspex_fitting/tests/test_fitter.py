import numpy as np
from numpy.testing import assert_allclose

from sunxspex.sunxspex_fitting.fitter import SunXspex, add_photon_model

rng = np.random.default_rng(2022)


# doesn't work if defined in the test function
def gauss(a, b, c, energies=None):
    """Gaussian function.

    Note: Parameters are args and energies (x-axis input) is a kwarg with None as default.

    Parameters
    ----------
    a,b,c : floats
            The scaler, mean, and standard deviation of the gaussian model output, respectively.

    energies : 2d array
            An array where each entry is the two energy bin edge values.

    Returns
    -------
    A 1d array output of the model.
    """
    mid_x = np.mean(energies, axis=1)
    return a * np.exp(-((mid_x - b) ** 2 / (2 * c ** 2)))


def test_fitter_custom():
    # add the model to be used in fitting
    add_photon_model(gauss)

    a = (1.3e4, 350, 60)
    b = (3e3, 600, 60)
    maxi, step = 1e3, 1
    chan_bins = np.stack((np.arange(0, maxi, step), np.arange(step, maxi + step, step)), axis=-1)
    gauss_mod1 = gauss(*a, energies=chan_bins)
    gauss_mod2 = gauss(*b, energies=chan_bins)
    noise = rng.integers(low=10, high=100, size=(len(chan_bins))) / 100 * 5

    fake_data = gauss_mod1 + gauss_mod2 + noise

    # create a simple dictionary with teh custom data information
    custom_dict = {"count_channel_bins": chan_bins,
                   "counts": fake_data
                   }  # counts with noise

    noise_constant = np.mean(noise)

    custom_spec = SunXspex(custom_dict)

    # assign the fitting code's active model to be a combination of ones you defined
    custom_spec.model = f"gauss+gauss+{noise_constant}"

    # define a large enough range to be sure the answer is in there somewhere
    custom_spec.params["a1_spectrum1"] = [1e4, (5e2, 5e4)]
    custom_spec.params["b1_spectrum1"] = [400, (2e2, 1e3)]
    custom_spec.params["c1_spectrum1"] = [100, (1e1, 2e2)]

    custom_spec.energy_fitting_range = [150, 380]
    custom_spec.params["a2_spectrum1"] = "freeze"
    custom_spec.params["b2_spectrum1"] = "freeze"
    custom_spec.params["c2_spectrum1"] = "freeze"

    # fit
    minimiser_results = custom_spec.fit()

    # roughly second Gaussian fitting range
    custom_spec.energy_fitting_range = [550, 800]
    custom_spec.params["a1_spectrum1"] = "freeze"
    custom_spec.params["b1_spectrum1"] = "freeze"
    custom_spec.params["c1_spectrum1"] = "freeze"
    custom_spec.params["a2_spectrum1"] = ["free", 5e3, (1e3, 1e4)]
    custom_spec.params["b2_spectrum1"] = ["free", 600, (2e2, 1e3)]
    custom_spec.params["c2_spectrum1"] = ["free", 50, (1e1, 1e2)]

    # fit
    minimiser_results = custom_spec.fit()

    # full range
    custom_spec.energy_fitting_range = [150, 800]
    custom_spec.params["a1_spectrum1"] = "free"
    custom_spec.params["b1_spectrum1"] = "free"
    custom_spec.params["c1_spectrum1"] = "free"

    # fit
    minimiser_results = custom_spec.fit()

    assert_allclose(minimiser_results[0], a[0], rtol=1e-4)
    assert_allclose(minimiser_results[1], a[1], rtol=1e-4)
    assert_allclose(minimiser_results[2], a[2], rtol=1e-3)
    assert_allclose(minimiser_results[3], b[0], rtol=1e-3)
    assert_allclose(minimiser_results[4], b[1], rtol=1e-3)
    assert_allclose(minimiser_results[5], b[2], rtol=1e-3)
