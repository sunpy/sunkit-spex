"""
=======================================
Joint Fitting Two Thermal Data Sources
=======================================
"""

import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u
# imports
from astropy.modeling.optimizers import SLSQP
from astropy.modeling.statistic import leastsquare

from sunkit_spex.fitting import fitters
from sunkit_spex.models.physical.thermal import ThermalEmission

#####################################################
#
# First, we need to define the thermal model. We will use it to generate
# synthetic data.
#


data_temp = 15 << u.MK
data_em1 = 5 << u.cm**-3  # measured in 1e49
data_gjf1 = ThermalEmission(
    temperature=data_temp,
    emission_measure=data_em1,
    fixed={"mg": True, "al": True, "si": True, "s": True, "ar": True, "ca": True, "fe": True},
)

data_em2 = 0.3 << u.cm**-3  # measured in 1e49
data_gjf2 = ThermalEmission(
    temperature=data_temp,
    emission_measure=data_em2,
    fixed={"mg": True, "al": True, "si": True, "s": True, "ar": True, "ca": True, "fe": True},
)

#####################################################
#
# Notice that we have made sure that the elemental abundances are fixed
# for the thermal model.
#
# This means we will only be fitting temperature and emission measure
# from this model.
#
# Our plan for the two data sources is that they are from plasma with
# the same temperature but different emission measures.
#
# Create and plot the synthetic data to see what we have.

# create synthetic data
energy_edges1 = np.arange(1.6, 30, 0.2) << u.keV
x1 = energy_edges1
mid_x1 = (x1[:-1] + x1[1:]) / 2
y1 = data_gjf1(energy_edges1)
rng = np.random.default_rng(147)
noise = 0.1
y1 *= rng.normal(1, noise, mid_x1.shape)

energy_edges2 = np.arange(4, 15, 0.1) << u.keV
x2 = energy_edges2
mid_x2 = (x2[:-1] + x2[1:]) / 2
y2 = data_gjf2(energy_edges2)
y2 *= rng.normal(1, noise, mid_x2.shape)

# Plot the data with the best-fit model
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(mid_x1, y1, "bo", alpha=0.4, label="data_gjf1")
ax.plot(mid_x2, y2, "go", alpha=0.4, label="data_gjf2")
ax.set(xlabel=f"Energy [{x1.unit:latex}]", ylabel=f"{y1.unit:latex}")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()

#####################################################
#
# Now we have the data, let's set up the model we want to fit with our
# best initial guesses for parameters.
#
# In addition to the initialisation values being used at the best
# guesses when it comes to the fitting (later), we can also set up other
# parameter properties (e.g., fixing them or giving them bounds).
#
# For fun, let's give the ``temperature`` value some bounds. Notice we
# only handle the temperature in the first model. This is because we
# know we will tie the second model's ``temperature`` value to the first
# anyway.

# get models
gjf1 = ThermalEmission(
    temperature=10 << u.MK,
    emission_measure=1 << u.cm**-3,
    bounds={"temperature": (5 << u.MK, 20 << u.MK)},
    fixed={"mg": True, "al": True, "si": True, "s": True, "ar": True, "ca": True, "fe": True},
)

gjf2 = ThermalEmission(
    emission_measure=0.1 << u.cm**-3,
    fixed={"mg": True, "al": True, "si": True, "s": True, "ar": True, "ca": True, "fe": True},
)

#####################################################
#
# With some forsight, these models will be passed to the fitter with
# the corresponding data-sets.
#
# The models will then be extracted and put into a list. This means
# parameters from one model can be referenced in another. This is done
# just using list indexing as is the models are already in the list in
# the order they will be passed to the fitter.

# tie temperatures together
gjf2.temperature.tied = lambda models: models[0].temperature

#####################################################
#
# Set up the ``JointFitter``:

# set up the base joint fitter
fit_joint = fitters.JointFitter(optimizer=SLSQP, statistic=leastsquare)

#####################################################
#
# Pass the models and data to the fitter as it is called. This is
# consistent with the general Astorpy fitting API.
#
# As long as we pass in the order of ``model0``, ``x0``, ``y0``, ...,
# ``modelN``, ``xN``, ``yN`` then we can pass as many model-data grouped
# as we want.
#
# Since the values of the photon spectra are huge (~1e30) which will
# really give the fitter a difficult time when optimising values on
# such a large scale over multiple orders of magnitude. For this we can
# add weights, or errors, for the fitting process.
#
# To help normalise the residuals, let's set the weights to just
# ``1/data``. These weights should keep the fit statistic value here
# to reasonable values.

# pass model and data to fitter
g12 = fit_joint(
    gjf1,
    x1,
    y1,
    gjf2,
    x2,
    y2,
    fkwarg={
        "weights": [
            1 / y1,
            1 / y2,
        ]
    },
)

#####################################################
#
# **Notice that ``fit_joint`` here would happily accept even a single
# model-data group and still work.**
#
# The fitter returns copies of the models with the parameter values
# changed to the fitted values.

print("-----------------------------------")
print("Thermal model gjf1:")
print(gjf1.param_names)
print(gjf1.parameters)
print("-----------------------------------")
print("Thermal model gjf2:")
print(gjf2.param_names)
print(gjf2.parameters)
print("-----------------------------------")
print("Joint fit")
print(g12[0].param_names)
print(g12[0].parameters)
print(g12[1].param_names)
print(g12[1].parameters)
print("-----------------------------------")

# Plot the data with the best-fit model
_, ax = plt.subplots(figsize=(8, 5))
ax.plot(mid_x1, y1, "bo", alpha=0.4, label="data_gjf1")
ax.plot(mid_x2, y2, "go", alpha=0.4, label="data_gjf2")
ax.plot(mid_x1, g12[0](x1), "b--", label="gjf1")
ax.plot(mid_x2, g12[1](x2), "g--", label="gjf2")
ax.set(xlabel=f"Energy [{x1.unit:latex}]", ylabel=f"{y1.unit:latex}")
plt.ylim([np.min([np.min(y1.value), np.min(y2.value)]), np.max([np.max(y1.value), np.max(y2.value)])])
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()

#####################################################
#
# Display a table of the fitted results

plt.figure(layout="constrained")


row_labels = [
    f"gjf1 temperature [{data_temp.unit:latex}]",
    f"gjf1 emission measure [1e49{data_em1.unit:latex}]",
    f"gjf2 emission measure [1e49{data_em2.unit:latex}]",
]
column_labels = ("True Values", "Guess Values", "``JointFitter`` Fit")

true_vals = np.array([data_temp.value, data_em1.value, data_em2.value])
guess_vals = np.array([gjf1.parameters[0], gjf1.parameters[1], gjf2.parameters[1]])
fit_vals = np.array([g12[0].parameters[0], g12[0].parameters[1], g12[1].parameters[1]])

cell_vals = np.vstack((true_vals, guess_vals, fit_vals)).T
cell_text = np.round(cell_vals, 1).astype(str)

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
