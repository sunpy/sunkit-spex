"""
======================
Fitting Simulated Data
======================

This is a file to show a very basic fitting of data where the model are
generated in a different space (photon-space) which are converted using
a square response matrix to the data-space (count-space).

.. note::
    Caveats:

    * The response is square so the count and photon energy axes are identical.
    * No errors are included in the fitting statistic.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import astropy.units as u
from astropy.modeling import fitting
from astropy.modeling.functional_models import Gaussian1D, Linear1D
from astropy.visualization import quantity_support

from sunkit_spex.data.simulated_data import simulate_square_response_matrix
from sunkit_spex.fitting.objective_functions.optimising_functions import minimize_func
from sunkit_spex.fitting.optimizer_tools.minimizer_tools import scipy_minimize
from sunkit_spex.fitting.statistics.gaussian import chi_squared
from sunkit_spex.models.instrument_response import MatrixModel
from sunkit_spex.spectrum import Spectrum
from sunkit_spex.spectrum.spectrum import SpectralAxis

#####################################################
#
# Start by creating simulated data and instrument.
# This would all be provided by a given observation.
#
# Can define the photon energies

start, inc = 1.6, 0.04
stop = 80 + inc / 2
ph_energies = np.arange(start, stop, inc) * u.keV

#####################################################
#
# Let's start making a simulated photon spectrum

sim_cont = {"edges":False,"slope": -1 * u.ph / u.keV, "intercept": 100 * u.ph}
sim_line = {"edges":False,"amplitude": 100 * u.ph, "mean": 30 * u.keV, "stddev": 2 * u.keV}
# use a straight line model for a continuum, Gaussian for a line
ph_model = Linear1D(**sim_cont) + Gaussian1D(**sim_line)

with quantity_support():
    plt.figure()
    plt.plot(ph_energies, ph_model(ph_energies))
    plt.xlabel(f"Energy [{ph_energies.unit}]")
    plt.title("Simulated Photon Spectrum")
    plt.show()

#####################################################
#
# Now want a response matrix

srm = simulate_square_response_matrix(ph_energies.size)
srm_model = MatrixModel(
    matrix=srm, input_axis=SpectralAxis(ph_energies), output_axis=SpectralAxis(ph_energies), c=1 * u.ct / u.ph
)
srm_model.inputs_units = {"x": u.ph}

with quantity_support():
    plt.figure()
    plt.imshow(
        srm_model.matrix,
        origin="lower",
        extent=(
            srm_model.inputs_axis[0].value,
            srm_model.inputs_axis[-1].value,
            srm_model.output_axis[0].value,
            srm_model.output_axis[-1].value,
        ),
        norm=LogNorm(),
    )
    plt.ylabel(f"Photon Energies [{srm_model.inputs_axis.unit}]")
    plt.xlabel(f"Count Energies [{srm_model.output_axis.unit}]")
    plt.title("Simulated SRM")
    plt.show()

#####################################################
#
# Start work on a count model

sim_gauss = {"edges":False,"amplitude": 70 * u.ct, "mean": 40 * u.keV, "stddev": 2 * u.keV}
# the brackets are very necessary
ct_model = (ph_model | srm_model) + Gaussian1D(**sim_gauss)

#####################################################
#
# Generate simulated count data to (almost) fit

sim_count_model = ct_model(SpectralAxis(ph_energies))

#####################################################
#
# Add some noise
np_rand = np.random.default_rng(seed=10)
sim_count_model_wn = (
    sim_count_model + (2 * np_rand.random(sim_count_model.size) - 1) * np.sqrt(sim_count_model.value) * u.ct
)

obs_spec = Spectrum(sim_count_model_wn.reshape(-1), spectral_axis=ph_energies)

#####################################################
#
# Can plot all the different components in the simulated count spectrum

with quantity_support():
    plt.figure()
    plt.plot(ph_energies, (ph_model | srm_model)(ph_energies), label="photon model features")
    plt.plot(ph_energies, Gaussian1D(**sim_gauss)(ph_energies), label="gaussian feature")
    plt.plot(ph_energies, sim_count_model, label="total sim. spectrum")
    plt.plot(obs_spec._spectral_axis, obs_spec.data, label="total sim. spectrum + noise", lw=0.5)
    plt.xlabel(f"Energy [{ph_energies.unit}]")
    plt.title("Simulated Count Spectrum")
    plt.legend()

    plt.text(80, 170, "(ph_model(sl,in,am1,mn1,sd1) | srm)", ha="right", c="tab:blue", weight="bold")
    plt.text(80, 150, "+ Gaussian(am2,mn2,sd2)", ha="right", c="tab:orange", weight="bold")
    plt.show()

#####################################################
#
# Now we have the simulated data, let's start setting up to fit it
#
# Get some initial guesses that are off from the simulated data above

guess_cont = {"edges":False,"slope": -0.5 * u.ph / u.keV, "intercept": 80 * u.ph}
guess_line = {"edges":False,"amplitude": 150 * u.ph, "mean": 32 * u.keV, "stddev": 5 * u.keV}
guess_gauss = {"edges":False,"amplitude": 350 * u.ct, "mean": 39 * u.keV, "stddev": 0.5 * u.keV}

#####################################################
#
# Define a new model since we have a rough idea of the mode we should use

ph_mod_4fit = Linear1D(**guess_cont) + Gaussian1D(**guess_line)
count_model_4fit = (ph_mod_4fit | srm_model) + Gaussian1D(**guess_gauss)

#####################################################
#
# Let's fit the simulated data and plot the result

opt_res = scipy_minimize(minimize_func, count_model_4fit.parameters, (obs_spec, count_model_4fit, chi_squared))

with quantity_support():
    plt.figure()
    plt.plot(ph_energies, sim_count_model_wn, label="total sim. spectrum + noise")
    plt.plot(ph_energies, count_model_4fit.evaluate(ph_energies.value, *opt_res.x), ls=":", label="model fit")
    plt.xlabel(f"Energy [{ph_energies.unit}]")
    plt.title("Simulated Count Spectrum Fit with Scipy")
    plt.legend()
    plt.show()


#####################################################
#
# Now try and fit with Astropy native fitting infrastructure and plot the result
#
# Try and ensure we start fresh with new model definitions

ph_mod_4astropyfit = Linear1D(**guess_cont) + Gaussian1D(**guess_line)
for m in ph_mod_4astropyfit:
    m.output_units = {"y": u.ph}
cgauss = Gaussian1D(**guess_gauss)
cgauss.output_units = {"y": u.ct}
srm_model.output_units = {"y": u.ct}

count_model_4astropyfit = (ph_mod_4astropyfit | srm_model) + cgauss

astropy_fit = fitting.LevMarLSQFitter()
astropy_fitted_result = astropy_fit(count_model_4astropyfit, ph_energies, obs_spec.data << obs_spec.unit)

plt.figure()
plt.plot(ph_energies, sim_count_model_wn, label="total sim. spectrum + noise")
plt.plot(
    ph_energies,
    count_model_4astropyfit.evaluate(ph_energies.value, *astropy_fitted_result.parameters),
    ls=":",
    label="model fit",
)
plt.xlabel("Energy [keV]")
plt.ylabel("cts s$^{-1}$ keV$^{-1}$")
plt.title("Simulated Count Spectrum Fit with Astropy")
plt.legend()
plt.show()

#####################################################
#
# Display a table of the fitted results

<<<<<<< HEAD
<<<<<<< HEAD
plt.figure(layout="constrained")


row_labels = (
    tuple(sim_cont)[-2:] + tuple(f"{p}1" for p in tuple(sim_line)[-3:]) + tuple(f"{p}2" for p in tuple(sim_gauss)[-3:])
)
column_labels = ("True Values", "Guess Values", "Scipy Fit", "Astropy Fit")
true_vals = np.array(tuple(sim_cont.values())[-2:] + tuple(sim_line.values())[-3:] + tuple(sim_gauss.values())[-3:])
guess_vals = np.array(
    tuple(guess_cont.values())[-2:] + tuple(guess_line.values())[-3:] + tuple(guess_gauss.values())[-3:]
)
scipy_vals = opt_res.x
astropy_vals = astropy_fitted_result.parameters

print(np.shape(scipy_vals))
print(np.shape(astropy_vals))
print(np.shape(true_vals))
print(np.shape(guess_vals))

=======
plt.figure(layout="constrained")

row_labels = (
    tuple(sim_cont) + tuple(f"{p}1" for p in tuple(sim_line)) + ("C",) + tuple(f"{p}2" for p in tuple(sim_gauss))
)
column_labels = ("True Values", "Guess Values", "Scipy Fit", "Astropy Fit")
true_vals = tuple(sim_cont.values()) + tuple(sim_line.values()) + (1 * u.m,) + tuple(sim_gauss.values())
true_vals = [t.value for t in true_vals]
guess_vals = tuple(guess_cont.values()) + tuple(guess_line.values()) + (1 * u.m,) + tuple(guess_gauss.values())
guess_vals = [g.value for g in guess_vals]
scipy_vals = opt_res.x
astropy_vals = astropy_fitted_result.parameters
>>>>>>> 9be4e97 (WIP)
cell_vals = np.vstack((true_vals, guess_vals, scipy_vals, astropy_vals)).T
cell_text = np.round(np.vstack((true_vals, guess_vals, scipy_vals, astropy_vals)).T, 2).astype(str)

plt.axis("off")
plt.table(
    cellText=cell_text,
    cellColours=None,
    cellLoc="center",
    rowLabels=row_labels,
    rowColours=None,
    colLabels=column_labels,
    colColours=None,
    colLoc="center",
    bbox=[0, 0, 1, 1],
)

plt.show()
<<<<<<< HEAD
=======
# plt.figure(layout="constrained")
#
# row_labels = tuple(sim_cont) + tuple(f"{p}1" for p in tuple(sim_line)) + tuple(f"{p}2" for p in tuple(sim_gauss))
# column_labels = ("True Values", "Guess Values", "Scipy Fit", "Astropy Fit")
# true_vals = np.array(tuple(sim_cont.values()) + tuple(sim_line.values()) + tuple(sim_gauss.values()))
# guess_vals = np.array(tuple(guess_cont.values()) + tuple(guess_line.values()) + tuple(guess_gauss.values()))
# scipy_vals = opt_res.x
# astropy_vals = astropy_fitted_result.parameters
# cell_vals = np.vstack((true_vals, guess_vals, scipy_vals, astropy_vals)).T
# cell_text = np.round(np.vstack((true_vals, guess_vals, scipy_vals, astropy_vals)).T, 2).astype(str)
#
# plt.axis("off")
# plt.table(
#     cellText=cell_text,
#     cellColours=None,
#     cellLoc="center",
#     rowLabels=row_labels,
#     rowColours=None,
#     colLabels=column_labels,
#     colColours=None,
#     colLoc="center",
#     bbox=[0, 0, 1, 1],
# )
#
# plt.show()
>>>>>>> b5668da (Scipy fitting works can't get astropy fitting to work)
=======
>>>>>>> 9be4e97 (WIP)
