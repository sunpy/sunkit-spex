"""
=====================================
Fitting With Different Fit Statistics
=====================================

We've made the joint fitting API allows the fit statistic to be
changeable when setting up the fitting.

Let's show this when fitting some simulated data.


"""

import matplotlib.pyplot as plt
import numpy as np

from astropy.modeling.functional_models import Linear1D

from sunkit_spex.fitting import fitters
from sunkit_spex.fitting.metrics.statistics import cash

#####################################################
#
# Let's make some synthetic data from them.

# Generate fake data
slope = 6.5
intercept = 1.11
noise = 0.05

# synthetic data from ``Gaussian1D``
rng = np.random.default_rng(147)
x1 = np.linspace(1.0, 6.0, 40)
area_model = Linear1D(slope=slope, intercept=intercept)
y1 = area_model(x1)
y1 *= rng.normal(1.0, noise, x1.shape)

#####################################################
#
# Plot the synthetic data to see what we have.

# Plot the data with the best-fit model
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x1, y1, "bo", alpha=0.4, label="Linear data")
ax.set(xlabel="Wavelength", ylabel="Flux")
plt.legend()
plt.show()

#####################################################
#
# Now we have the data, let's set up the model we want to fit with our
# best initial guesses for parameters.
#

# get models
g1 = Linear1D()

#####################################################
#
# Sunkit-spex has several options for the ``statistic`` input but the
# main thing is that it works with the optimisation method.
#
# For example, the Scipy ``minimize`` function expects to optimise a
# single numerical value and so the statistic option should return a
# single value.
#
# For this example, we'll use the Cash statistic. We can use ``help``
# on the function to get a little insight into it.
#

help(cash)

#####################################################
#
# Set up the joint fitter.
#
# It's at this point we pass in the statistic of choice for the
# data-to-model comparison.

# set up the base joint fitter
fit_obj = fitters.ScipyMinimizeJointFitter(statistic=cash)

#####################################################
#
# .. note::
#     This assumes all the data being fitted (e.g., if we were performing
#     joint fitting) makes use of the same fitting metric. In the future,
#     each data-model set will have the ability to be matched with it's
#     own metric.
#
#     However, these should likelihoods so the metric has it's original
#     physical meaning and not just random staticstical metrics.
#
#     Additionally, the ``cash`` function is just that, a function. A user
#     could equally define their own statistic and pass it here.

#####################################################
#
# Pass the model and data to the fitter as it is called. This is
# consistent with the general Astorpy fitting API.
#
# As long as we pass in the order of ``model0``, ``x0``, ``y0``, ...,
# ``modelN``, ``xN``, ``yN`` then we can pass as many model-data groups
# as we want.

# pass model and data to fitter
new_g1 = fit_obj(g1, x1, y1)

#####################################################
#
# The fitter returns copies of the models with the parameter values
# changed to the fitted values.

print("-----------------------------------")
print("Gaussian1")
print(g1.param_names)
print(g1.parameters)
print("-----------------------------------")
print("Fit")
print(new_g1[0].param_names)
print(new_g1[0].parameters)
print("-----------------------------------")

# Plot the data with the best-fit model
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x1, y1, "bo", alpha=0.4, label="Linear data 1")
ax.plot(x1, new_g1[0](x1), "b--", label="g1 Linear1D model")
ax.set(xlabel="Wavelength", ylabel="Flux")
plt.legend()
plt.show()

#####################################################
#
# Display a table of the fitted results

plt.figure(layout="constrained")


row_labels = [f"new_g1 {new_g1pn}" for new_g1pn in new_g1[0].param_names]
column_labels = ("True Values", "Guess Values", "Fit")

true_vals = np.array([slope, intercept])
guess_vals = np.array(g1.parameters)
fit_vals = np.array(new_g1[0].parameters)

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
