"""
=========================================================
Fitting NuSTAR Spectra: Duncan *et al.* 2021 comparison
=========================================================

This spectrum corresponds to the may1618 microflare in [Duncan2021]_.

An example of fitting multiple spectra simultaneously with 2 models where each model is allowed to vary at different times

We also allow the gain slope response parameter to vary.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.exceptions import VisibleDeprecationWarning
from parfive import Downloader

from sunkit_spex.legacy.fitting.fitter import Fitter

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
except AttributeError:
    warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)


#####################################################
#
# Set up some plotting numbers

spec_single_plot_size = (8, 10)
spec_plot_size = (25, 10)
spec_font_size = 18
default_font_size = 10
x_limits, y_limits = [1.6, 8.5], [1e-1, 1e3]

# set up plotting info stuff

dunc_xlims, dunc_ylims = [2.5, 11], [1e0, 4e4]

#####################################################
#
# Download the example data

dl = Downloader()

base_url = "https://sky.dias.ie/public.php/dav/files/BHW6y6aXiGGosM6/nustar/Duncan2021/"
file_names = [
    "nu80410201001A06_1618_p_chu2_N_sr.pha",
    "nu80410201001A06_1618_p_chu2_N_sr.arf",
    "nu80410201001A06_1618_p_chu2_N_sr.rmf",
    "nu80410201001B06_1618_p_chu2_N_sr.pha",
    "nu80410201001B06_1618_p_chu2_N_sr.arf",
    "nu80410201001B06_1618_p_chu2_N_sr.rmf",
]

for fname in file_names:
    dl.enqueue_file(base_url + fname, path="./nustar/Duncan2021/")
files = dl.download()


#####################################################
#
# First, load in your data files, here we load in 2 spectra

_dir = "./nustar/Duncan2021/"
spec = Fitter(pha_file=[_dir + "nu80410201001A06_1618_p_chu2_N_sr.pha", _dir + "nu80410201001B06_1618_p_chu2_N_sr.pha"])

#####################################################
#
# Define model, here we go for 2 isothermal models
spec.model = "C*(f_vth + f_vth)"

#####################################################
#
# Check the parameter table
print(spec.params)

#####################################################
#
# freeze the ones we don't want to vary
spec.params["C_spectrum1"] = {"Status": "frozen"}

#####################################################
#
# Set initial values
spec.params["T1_spectrum1"] = {"Value": 4.1, "Bounds": (2.5, 6)}
spec.params["EM1_spectrum1"] = {"Value": 14, "Bounds": (1e0, 1e2)}
spec.params["T2_spectrum1"] = {"Value": 10, "Bounds": (5, 15)}
spec.params["EM2_spectrum1"] = {"Value": 0.46, "Bounds": (1e-4, 10)}
spec.params["C_spectrum2"] = {"Status": "free", "Bounds": (0.5, 2)}

#####################################################
#
# Fit lower energy range with the first thermal model first
spec.params["T2_spectrum1"] = "frozen"
spec.params["EM2_spectrum1"] = "frozen"
spec.energy_fitting_range = [2.5, 4]

spec.fit(tol=1e-6)
print(spec.params)
print(spec.rParams)

#####################################################
#
# Now fit higher energy range with the second thermal model

spec.params["T1_spectrum1"] = "frozen"
spec.params["EM1_spectrum1"] = "frozen"
spec.params["C_spectrum2"] = "frozen"
spec.params["T2_spectrum1"] = "free"
spec.params["EM2_spectrum1"] = "free"

#####################################################
#
# Need the gain slope to vary too for this microflare but only needed for the 6.7 keV line
print(spec.rParams)
spec.rParams["gain_slope_spectrum1"] = "free"
spec.rParams["gain_slope_spectrum2"] = spec.rParams["gain_slope_spectrum1"]

spec.energy_fitting_range = [4, 10.8]

spec.fit(tol=1e-6)
print(spec.params)
print(spec.rParams)

#####################################################
#
# Now free everything over full range
spec.params["T1_spectrum1"] = "free"
spec.params["EM1_spectrum1"] = "free"
spec.params["C_spectrum2"] = "free"

spec.energy_fitting_range = [3, 10.8]

spec.fit(tol=1e-10)
print(spec.params)
print(spec.rParams)

#####################################################
#
# Plot the result
plt.rcParams["font.size"] = spec_font_size
plt.figure(figsize=spec_plot_size)
axes, res_axes = spec.plot()
for a in axes:
    a.set_xlim(dunc_xlims)
    a.set_ylim(dunc_ylims)
plt.show()
plt.rcParams["font.size"] = default_font_size

#####################################################
#
# For the 2 thermal model fitting
#
# +----------------------------------------+-----------------------------------------------+------------------------------------------+
# | Model Parameter                        | XSPEC (Duncan et al. 2021) [*]_               | This Work                                |
# +========================================+===============================================+==========================================+
# | Temperature 1 [MK]                     | :math:`4.1^{+0.1}_{-0.1}`                     | 4.8\ |pm|\ 0.3                           |
# +----------------------------------------+-----------------------------------------------+------------------------------------------+
# | Emission Measure 1 [cm\ :sup:`-3`]     | :math:`1.4^{+0.6}_{-0.4}\times10^{47}`        | 7.6 |pm| 2.4\ |x|\ 10 :sup:`46`          |
# +----------------------------------------+-----------------------------------------------+------------------------------------------+
# | Temperature 2 [MK]                     | :math:`10.00^{+0.03}_{-0.03}`                 | 10.4\ |pm|\ 0.1                          |
# +----------------------------------------+-----------------------------------------------+------------------------------------------+
# | Emission Measure 2 [cm\ :sup:`-3`]     | :math:`4.6^{+0.1}_{-0.2}\times10^{45}`        | 4.3 |pm| 0.3\ |x|\ 10 :sup:`45`          |
# +----------------------------------------+-----------------------------------------------+------------------------------------------+
#
# .. |pm| replace:: :math:`\pm`
# .. |x| replace:: :math:`\times`
# .. [*] Duncan *et al.* 2021's may1618 microflare
# .. [Duncan2021] https://iopscience.iop.org/article/10.3847/1538-4357/abca3d

#####################################################
#
# For the gain parameters
#
# +----------------------------------------+-----------------------------------------------+------------------------------------------+
# | Model Parameter                        | XSPEC (Duncan et al. 2021)                    | This Work                                |
# +========================================+===============================================+==========================================+
# | Gain Slope                             | 0.977\ |pm|\ 0.002                            | 0.978\ |pm|\ 0.001                       |
# +----------------------------------------+-----------------------------------------------+------------------------------------------+

#####################################################
# Although these values are slightly different, it is important to note that XSPEC and sunkit-spex work from different atomic databases. We also note that for a similar isothermal fit the temperature can drop/rise if the emission measure rises/drops and so fitting not just one but two of these models allows for these to vary more. We do see that this work (for this microflare) produces higher temperatures but correspondingly lower emission measures.
