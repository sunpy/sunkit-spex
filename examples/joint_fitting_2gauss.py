"""
=======================================
Joint Fitting Two Gaussian Data Sources
=======================================

Based on the `Astropy JointFitter example <https://docs.astropy.org/en/stable/modeling/jointfitter.html>`_.


"""

import matplotlib.pyplot as plt
import numpy as np

from astropy.modeling import Fittable1DModel
# imports
from astropy.modeling.functional_models import FLOAT_EPSILON, Gaussian1D
from astropy.modeling.optimizers import SLSQP
from astropy.modeling.parameters import Parameter
from astropy.modeling.statistic import leastsquare

from sunkit_spex.fitting import fitters

#####################################################
#
# Let's make it a little more complicated and simultaneously fit two different
# models to two different data-sets where there is at least one common
# parameter between the two.
#
# To do this, let's use Astropy's native ``Gaussian1D`` model for one set
# of synthetic data and then one using the ``AreaGaussian1D`` model
# defined in the Astropy joint fitting example.


class AreaGaussian1D(Fittable1DModel):
    """
    One dimensional Gaussian model with area as a parameter.

    Parameters
    ----------
    area : float or `~astropy.units.Quantity`.
        Integrated area
        Note: amplitude = area / (stddev * np.sqrt(2 * np.pi))
    mean : float or `~astropy.units.Quantity`.
        Mean of the Gaussian.
    stddev : float or `~astropy.units.Quantity`.
        Standard deviation of the Gaussian with FWHM = 2 * stddev * np.sqrt(2 * np.log(2)).
    """

    area = Parameter(default=1)
    mean = Parameter(default=0)

    # Ensure stddev makes sense if its bounds are not explicitly set.
    # stddev must be non-zero and positive.
    stddev = Parameter(default=1, bounds=(FLOAT_EPSILON, None))

    @staticmethod
    def evaluate(x, area, mean, stddev):
        """
        AreaGaussian1D model function.
        """
        return (area / (stddev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x - mean) ** 2 / stddev**2)


#####################################################
#
# We now have the models so no let's make some synthetic data from them.

# Generate fake data
amplitude = 5.6
area = 1.5
mean = 5.1
stddev1 = 0.4
stddev2 = 0.2
noise = 0.10

# synthetic data from ``Gaussian1D``
rng = np.random.default_rng(147)
x1 = np.linspace(1.0, 6.0, 200)
area_model = Gaussian1D(amplitude=amplitude, mean=mean, stddev=stddev1)
y1 = area_model(x1)
y1 += rng.normal(0.0, noise, x1.shape)

# synthetic data from ``AreaGaussian1D``
x2 = np.linspace(4.0, 10.0, 200)
area_model = AreaGaussian1D(area=area, mean=mean, stddev=stddev2)
y2 = area_model(x2)
y2 += rng.normal(0.0, noise, x2.shape)

#####################################################
#
# Plot the synthetic data to see what we have.

# Plot the data with the best-fit model
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x1, y1, "bo", alpha=0.4, label="Gaussian data 1")
ax.plot(x2, y2, "go", alpha=0.4, label="Gaussian data 2")
ax.set(xlabel="Wavelength", ylabel="Flux")
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
# For fun, let's give the ``mean`` value some bounds and assume that the
# ``stddev`` value for the first data-set is known.

# get models
gjf1 = Gaussian1D(
    amplitude=5,
    mean=5.5,
    stddev=0.4,
    bounds={
        "mean": (0, 10),
    },
    fixed={
        "stddev": True,
    },
)
gjf2 = AreaGaussian1D(area=1, mean=6, stddev=0.1)

#####################################################
#
# With some forsight, these models will be passed to the fitter with
# the corresponding data-sets.
#
# The models will then be extracted and put into a list. This means
# parameters from one model can be referenced in another. This is done
# just using list indexing as is the models are already in the list in
# the order they will be passed to the fitter.
#
# Let's assume, as a scientist, we expect the ``mean`` value of both
# model answers to be the same but the other parameters---the
# ``amplitude``, ``area``, and the ``AreaGaussian1D`` ``stddev``---are
# independent. This could be due to some instrument or sensitivity
# dependence.

# tie some parameters together
gjf2.mean.tied = lambda models: models[0].mean.value

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
# ``modelN``, ``xN``, ``yN`` then we can pass as many model-data groups
# as we want.

# pass model and data to fitter
g12 = fit_joint(gjf1, x1, y1, gjf2, x2, y2)

#####################################################
#
# The fitter returns copies of the models with the parameter values
# changed to the fitted values.

print("-----------------------------------")
print("Gaussian1")
print(gjf1.param_names)
print(gjf1.parameters)
print("-----------------------------------")
print("AreaGaussian2")
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
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x1, y1, "bo", alpha=0.4, label="Gaussian data 1")
ax.plot(x2, y2, "go", alpha=0.4, label="Gaussian data 2")
ax.plot(x1, g12[0](x1), "b--", label="gjf1 Gaussian1D model")
ax.plot(x2, g12[1](x2), "g--", label="gjf2 AreaGaussian1D model")
ax.set(xlabel="Wavelength", ylabel="Flux")
plt.legend()
plt.show()

#####################################################
#
# Display a table of the fitted results

plt.figure(layout="constrained")


row_labels = [f"gjf1 {g12pn}" for g12pn in g12[0].param_names] + [f"gjf2 {g12pn}" for g12pn in g12[1].param_names]
row_labels[1] += "\n(bounded)"
row_labels[2] += "\n(fixed)"
row_labels[4] += "\n(tied to `gjf1 mean`)"
column_labels = ("True Values", "Guess Values", "``JointFitter`` Fit")

true_vals = np.array([amplitude, mean, stddev1, area, mean, stddev2])
guess_vals = np.array(np.concatenate((gjf1.parameters, gjf2.parameters)))
fit_vals = np.array(np.concatenate((g12[0].parameters, g12[1].parameters)))

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
