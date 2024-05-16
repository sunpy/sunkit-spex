import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sunkit_spex.fitting_legacy.fitter import Fitter
from sunkit_spex.fitting_legacy.instruments import CustomLoader, InstrumentBlueprint

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


@pytest.fixture
def custom_spec():
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

    custom_spec = Fitter(custom_dict)

    # add the model to be used in fitting
    custom_spec.add_photon_model(gauss)

    # assign the fitting code's active model to be a combination of ones you defined
    custom_spec.model = f"gauss+gauss+{noise_constant}"

    return custom_spec, a, b


def test_fitter_fit(custom_spec):
    custom_spec, a, b = custom_spec
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


def test_fitter_plot(custom_spec):
    custom_spec, a, b = custom_spec
    custom_spec.energy_fitting_range = [150, 800]
    custom_spec.fit()
    custom_spec.plot()


def test_fitter_load(custom_spec, tmp_path):
    custom_spec, a, b = custom_spec
    savefile = tmp_path / 'test'
    custom_spec.save(str(savefile))
    with open(tmp_path / 'test.pickle', 'rb') as d:
        cs = pickle.load(d)

    cs.params["a1_spectrum1"] = [1e4, (5e2, 5e4)]
    cs.params["b1_spectrum1"] = [400, (2e2, 1e3)]
    cs.params["c1_spectrum1"] = [100, (1e1, 2e2)]
    cs.params["a2_spectrum1"] = ["free", 5e3, (1e3, 1e4)]
    cs.params["b2_spectrum1"] = ["free", 600, (2e2, 1e3)]
    cs.params["c2_spectrum1"] = ["free", 50, (1e1, 1e2)]
    cs.fit()

    minimiser_results = cs.fit()

    assert_allclose(minimiser_results[0], a[0], rtol=1e-3)
    assert_allclose(minimiser_results[1], a[1], rtol=1e-3)
    assert_allclose(minimiser_results[2], a[2], rtol=1e-3)
    assert_allclose(minimiser_results[3], b[0], rtol=1e-3)
    assert_allclose(minimiser_results[4], b[1], rtol=1e-3)
    assert_allclose(minimiser_results[5], b[2], rtol=1e-2)


def _assert_in_range(bounds, values):
    """Raise assertion if the values given are within certain bounds.

    I.e., lower_bound <= values <= upper_bound
    """
    assert np.all((bounds[0] <= values) & (values <= bounds[1])), "Value(s) not within bounds given."


def test_mcmc_walker_boundary_spread():
    # test lower and upper boundaries, [(lower,upper),(lower,upper),...]
    list_of_tuple_bounds1, number_of_values1 = [(1, 10)], 1
    list_of_tuple_bounds2, number_of_values2 = [(0.005, 0.3)], 5
    list_of_tuple_bounds3, number_of_values3 = [(1, 10), (0.005, 0.3)], 10

    # get `number` entries between all bounds in `value_bounds`
    spread1 = Fitter()._boundary_spread(value_bounds=list_of_tuple_bounds1, number=number_of_values1)
    spread2 = Fitter()._boundary_spread(value_bounds=list_of_tuple_bounds2, number=number_of_values2)
    spread3 = Fitter()._boundary_spread(value_bounds=list_of_tuple_bounds3, number=number_of_values3)

    # check values produced are within the bounds
    _assert_in_range(list_of_tuple_bounds1[0], spread1[:, 0])
    _assert_in_range(list_of_tuple_bounds2[0], spread2[:, 0])
    _assert_in_range(list_of_tuple_bounds3[0], spread3[:, 0])
    _assert_in_range(list_of_tuple_bounds3[1], spread3[:, 1])


def test_passing_instrument_loader_direct():
    """Test behaviour when passing an instrument loader directly to the fitting infrastructure."""

    # create a new instrument class from the beginning with the `InstrumentBlueprint` class
    class UserCustomInst1(InstrumentBlueprint):
        def __init__(self, spec_data_dict):
            # set up default values, only need these fields to be populated
            _count_length_default = np.ones(len(spec_data_dict["count_channel_bins"]))
            _chan_mids_default = np.mean(spec_data_dict["count_channel_bins"], axis=1)
            _default_spec_data = {"photon_channel_bins": spec_data_dict["count_channel_bins"],
                                  "photon_channel_mids": _chan_mids_default,
                                  "photon_channel_binning": _count_length_default,
                                  "count_channel_mids": _chan_mids_default,
                                  "count_channel_binning": _count_length_default,
                                  "count_error": _count_length_default,
                                  "count_rate": spec_data_dict["counts"],
                                  "count_rate_error": _count_length_default,
                                  "effective_exposure": 1,
                                  "srm": np.identity(len(spec_data_dict["counts"])),
                                  "extras": {}}
            _default_spec_data.update(spec_data_dict)
            self._loaded_spec_data = _default_spec_data

    # create a class from existing loader class
    class UserCustomInst2(CustomLoader):
        def __init__(self, *args, **kwargs):
            CustomLoader.__init__(self, *args, **kwargs)

    # create fake data
    bin_edges = np.arange(2)
    chan_bins = np.concatenate((bin_edges[:-1, None], bin_edges[1:, None]), axis=1)
    fake_data = bin_edges[:-1]
    custom_dict = {"count_channel_bins": chan_bins, "counts": fake_data}

    # pass the fake data to the different intrument classes defined
    custom_user_inst1 = CustomLoader(custom_dict)
    custom_user_inst2 = UserCustomInst1(custom_dict)
    custom_user_inst3 = UserCustomInst2(custom_dict)

    # pass the classes themselves, not the data or files to the fitter
    fitter = Fitter(custom_user_inst1, custom_user_inst2, custom_user_inst3)

    # check everything has carried through to where it should be
    assert fitter.data.loaded_spec_data["spectrum1"]._loaded_spec_data == custom_user_inst1._loaded_spec_data, "Failed to pass instrument loader directly to fitter."
    assert fitter.data.loaded_spec_data["spectrum2"]._loaded_spec_data == custom_user_inst2._loaded_spec_data, "Failed to pass instrument loader directly to fitter."
    assert fitter.data.loaded_spec_data["spectrum3"]._loaded_spec_data == custom_user_inst3._loaded_spec_data, "Failed to pass instrument loader directly to fitter."
