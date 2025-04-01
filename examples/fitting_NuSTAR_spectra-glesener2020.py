"""
=========================================================
Fitting NuSTAR Spectra: Glesener *et al.* 2020 comparison
=========================================================

A real example from [Glesener2020](https://iopscience.iop.org/article/10.3847/2041-8213/ab7341) of fitting two NuSTAR spectra simultaneously with gain correction.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.exceptions import VisibleDeprecationWarning
from parfive import Downloader

from sunkit_spex.legacy.fitting.fitter import Fitter, load

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


#####################################################
#
# # Let's fit a more realistic example
#
# Try fitting the spectra presented in [Glesener *et al.* 2020](https://iopscience.iop.org/article/10.3847/2041-8213/ab7341). The spectrum presented shows clear evidence of non-thermal emission in an A5.7 microflare.
#
# ## Let's recreate Figure 4 (left) where NuSTAR FPMB is fitted with a thermal+cold thick target model.
#
# set up plotting info stuff

gles_xlims, gles_ylims = [2, 12], [1e1, 1e4]

#####################################################
#
# Download data files

dl = Downloader()

base_url = (
    "https://sky.dias.ie/index.php/s/ekBWE57kC7rjeBF/download?path=%2Fexample_data%2Fnustar%2FGlesener2020&files="
)
file_names = [
    "nu20312001001B06_cl_grade0_sr_grp.pha",
    "nu20312001001B06_cl_grade0_sr.arf",
    "nu20312001001B06_cl_grade0_sr.rmf",
    "nu20312001001A06_cl_grade0_sr_grp.pha",
    "nu20312001001A06_cl_grade0_sr.arf",
    "nu20312001001A06_cl_grade0_sr.rmf",
]

for fname in file_names:
    dl.enqueue_file(base_url + fname, path="./nustar/Glesener2020/")
files = dl.download()


#####################################################
#
# First, load in your data files, here we load in 2 spectra
_dir = "./nustar/Glesener2020/"
# in the files I have, the ARF and RMF file have different names to the PHA files so cannot use the PHA file name to help find the others so...
spec = Fitter(
    pha_file=_dir + "nu20312001001B06_cl_grade0_sr_grp.pha",
    arf_file=_dir + "nu20312001001B06_cl_grade0_sr.arf",
    rmf_file=_dir + "nu20312001001B06_cl_grade0_sr.rmf",
)

#####################################################
#
# define model, here we go for a single isothermal model + cold thick model

spec.model = "f_vth + thick_fn"

#####################################################
#
# define fitting range
spec.energy_fitting_range = [2.8, 10.5]

#####################################################
#
# sort temperature param from f_vth
spec.params["T1_spectrum1"] = {"Value": 10.3, "Bounds": (1.1, 15)}
# emission measure param from f_vth
spec.params["EM1_spectrum1"] = {"Value": 0.5, "Bounds": (1e-2, 1e1)}
# electron flux param from thick_fn
spec.params["total_eflux1_spectrum1"] = {"Value": 2.1, "Bounds": (1e-3, 10)}  # units 1e35 e^-/s
# electron index param from thick_fn
spec.params["index1_spectrum1"] = {"Value": 6.2, "Bounds": (3, 10)}
# electron low energy cut-off param from thick_fn
spec.params["e_c1_spectrum1"] = {"Value": 6.2, "Bounds": (1, 12)}  # units keV

#####################################################
#
# **This fit requires altering the gain parameters**
#
# Gain parameters can be tweaked in the same way model parameters can.
#
# The difference is that gain parameters all have specific starting values (slope=1, offset=0) and are frozen by default.

# from Gles. 2020 which had a gain correction fixed at 0.95
spec.rParams["gain_slope_spectrum1"] = {"Status": "fixed", "Value": 0.95}

print(spec.rParams)

print(spec.show_rParams)

#####################################################
#
# fit the model to the spectrum
spec.fit(tol=1e-8)

# plot the result
plt.rcParams["font.size"] = spec_font_size
plt.figure(figsize=spec_single_plot_size)
axes, res_axes = spec.plot()
for a in axes:
    a.set_xlim(gles_xlims)
    a.set_ylim(gles_ylims)
plt.show()
plt.rcParams["font.size"] = default_font_size


#####################################################
#
# **Let's recreate Figure 3(c)** 
# 
# Both NuSTAR FPMs are fitted with a thermal+cold thick target model simultaneously.
#
# First, load in your data files, here we load in 2 spectra
_dir = "./nustar/Glesener2020/"
# in the files I have, the ARF and RMF file have different names to the PHA files so cannot use the PHA file name to help find the others so...
spec = Fitter(
    pha_file=[_dir + "nu20312001001A06_cl_grade0_sr_grp.pha", _dir + "nu20312001001B06_cl_grade0_sr_grp.pha"],
    arf_file=[_dir + "nu20312001001A06_cl_grade0_sr.arf", _dir + "nu20312001001B06_cl_grade0_sr.arf"],
    rmf_file=[_dir + "nu20312001001A06_cl_grade0_sr.rmf", _dir + "nu20312001001B06_cl_grade0_sr.rmf"],
)

#####################################################
#
# define model, here we go for a single isothermal model + cold thick model
spec.model = "C*(f_vth + thick_fn)"

#####################################################
#
# define fitting range
spec.energy_fitting_range = [2.8, 10.5]

#####################################################
#
# sort temperature param from f_vth
spec.params["T1_spectrum1"] = {"Value": 10.3, "Bounds": (1.1, 15)}
# emission measure param from f_vth
spec.params["EM1_spectrum1"] = {"Value": 0.5, "Bounds": (1e-2, 1e1)}
# electron flux param from thick_fn
spec.params["total_eflux1_spectrum1"] = {"Value": 2.1, "Bounds": (1e-3, 10)}  # units 1e35 e^-/s
# electron index param from thick_fn
spec.params["index1_spectrum1"] = {"Value": 6.2, "Bounds": (3, 10)}
# electron low energy cut-off param from thick_fn
spec.params["e_c1_spectrum1"] = {"Value": 6.2, "Bounds": (1, 12)}  # units keV
# constant for systematic offset between FPMs, found to be about 1.1
spec.params["C_spectrum1"] = "frozen"
# constant for systematic offset between FPMs, found to be about 1.1
spec.params["C_spectrum2"] = {"Status": "fixed", "Value": 1.1}
# from Gles. 2020 which had a gain correction fixed at 0.95
spec.rParams["gain_slope_spectrum1"] = {"Status": "fixed", "Value": 0.95}
spec.rParams["gain_slope_spectrum2"] = spec.rParams["gain_slope_spectrum1"]

#####################################################
#
# fit the model to the spectrum
spec.fit(tol=1e-8)

# plot the result
plt.rcParams["font.size"] = spec_font_size
plt.figure(figsize=spec_plot_size)
axes, res_axes = spec.plot()
for a in axes:
    a.set_xlim(gles_xlims)
    a.set_ylim(gles_ylims)
plt.show()
plt.rcParams["font.size"] = default_font_size

#####################################################
#
# For the thermal and cold thick target total model we compare::
#
#    | Model Parameter                  | OSPEX (Glesener et al. 2020, just FPMB)       |This Work (just FPMB)           | This Work (FPMA&B)                |
#    | :---                             |    :----:                                     |     :----:                     |                              ---: |
#    | Temperature [MK]                 | 10.3$^{+0.7}_{-0.7}$                          | 10.12$\pm$0.10                 | 9.77$\pm$0.06                     |
#    | Emission Measure [cm$^{-3}$]     | 5.0$^{+1.3}_{-1.3}\times$10$^{45}$            | 4.86$\pm$0.01$\times$10$^{45}$ | 4.66$\pm$0.05$\times$10$^{45}$    |
#    | Electron Flux [e$^{-}$ s$^{-1}$] | 2.1$^{+1.2}_{-1.2}\times$10$^{35}$            | 2.17$\pm$0.06$\times$10$^{35}$ | 2.25$\pm$0.02$\times$10$^{35}$    |
#    | Index                            | 6.2$^{+0.6}_{-0.6}$                           | 5.83$\pm$0.09                  | 6.09$\pm$0.05                     |
#    | Low Energy Cut-off [keV]         | 6.2$^{+0.9}_{-0.9}$                           | 6.66$\pm$0.05                  | 6.52$\pm$0.05                     |
#
# **Now let's recreate Figure 4 (right)** 
# 
# NuSTAR FPMB is fitted with a warm thick target model.
#
# The warm thick target model helps to constrain the non-thermal emission with observed values (e.g., loop length, etc) and ties it to the thermal emission parameters.
#
# First, load in your data files, here we load in 1 spectrum


_dir = "./nustar/Glesener2020/"
spec = Fitter(
    pha_file=_dir + "nu20312001001B06_cl_grade0_sr_grp.pha",
    arf_file=_dir + "nu20312001001B06_cl_grade0_sr.arf",
    rmf_file=_dir + "nu20312001001B06_cl_grade0_sr.rmf",
)

#####################################################
#
# define model, here we go for a single isothermal model + cold thick model
spec.model = "thick_warm"

#####################################################
#
# define fitting range
spec.energy_fitting_range = [2.8, 10.5]

#####################################################
#
# .. note:
#    Note that similar parameters in the warm thick target and cold thick target models have slightly different names

# electron flux param
spec.params["tot_eflux1_spectrum1"] = {"Value": 2, "Bounds": (1e-2, 10)}
# electron index param
spec.params["indx1_spectrum1"] = {"Value": 6, "Bounds": (3, 10)}
# electron low energy cut-off param
spec.params["ec1_spectrum1"] = {"Value": 7, "Bounds": (3, 12)}
# loop plasma temperature param
spec.params["loop_temp1_spectrum1"] = {"Value": 10, "Bounds": (5, 15)}
# plasma number density param
spec.params["plasma_d1_spectrum1"] = {"Value": 1, "Bounds": (1e-2, 1e1)}  # units 1e10 cm^-3
# loop length param
spec.params["length1_spectrum1"] = {"Status": "fixed", "Value": 15}  # units Mm
# from Gles. 2020 which had a gain correction fixed at 0.95
spec.rParams["gain_slope_spectrum1"] = {"Status": "fixed", "Value": 0.95}

#####################################################
#
# fit the model to the spectrum
spec.fit(tol=1e-10)

# plot the result
plt.rcParams["font.size"] = spec_font_size
plt.figure(figsize=spec_single_plot_size)
axes, res_axes = spec.plot()
for a in axes:
    a.set_xlim(gles_xlims)
    a.set_ylim(gles_ylims)
plt.show()
plt.rcParams["font.size"] = default_font_size

#####################################################
#
# **Fit the warm thick target model to both FPMs simultaneously**
#
#
# First, load in your data files, here we load in 2 spectra

_dir = "./nustar/Glesener2020/"
spec = Fitter(
    pha_file=[_dir + "nu20312001001A06_cl_grade0_sr_grp.pha", _dir + "nu20312001001B06_cl_grade0_sr_grp.pha"],
    arf_file=[_dir + "nu20312001001A06_cl_grade0_sr.arf", _dir + "nu20312001001B06_cl_grade0_sr.arf"],
    rmf_file=[_dir + "nu20312001001A06_cl_grade0_sr.rmf", _dir + "nu20312001001B06_cl_grade0_sr.rmf"],
)

#####################################################
#
# define model, here we go for a single isothermal model + cold thick model
spec.model = "C*thick_warm"

#####################################################
#
# define fitting range
spec.energy_fitting_range = [2.8, 10.5]

# Note that similar parameters in the warm thick target and cold thick target models have slightly different names
# electron flux param
spec.params["tot_eflux1_spectrum1"] = {"Value": 2, "Bounds": (1e-3, 10)}
# electron index param
spec.params["indx1_spectrum1"] = {"Value": 6, "Bounds": (3, 10)}
# electron low energy cut-off param
spec.params["ec1_spectrum1"] = {"Value": 7, "Bounds": (1, 12)}
# loop plasma temperature param
spec.params["loop_temp1_spectrum1"] = {"Value": 10, "Bounds": (1.1, 15)}
# plasma number density param
spec.params["plasma_d1_spectrum1"] = {"Value": 1, "Bounds": (1e-2, 1e1)}
# loop length param
spec.params["length1_spectrum1"] = {"Status": "fixed", "Value": 15}
# constant for systematic offset between FPMs, found to be about 1.1
spec.params["C_spectrum1"] = "frozen"
spec.params["C_spectrum2"] = {"Status": "fixed", "Value": 1.1}
# from Gles. 2020 which had a gain correction fixed at 0.95
spec.rParams["gain_slope_spectrum1"] = {"Status": "fixed", "Value": 0.95}
spec.rParams["gain_slope_spectrum2"] = spec.rParams["gain_slope_spectrum1"]

#####################################################
#
# fit the model to the spectrum
spec.fit(tol=1e-5)

# plot the result
plt.rcParams["font.size"] = spec_font_size
plt.figure(figsize=spec_plot_size)
axes, res_axes = spec.plot()
for a in axes:
    a.set_xlim(gles_xlims)
    a.set_ylim(gles_ylims)
plt.show()
plt.rcParams["font.size"] = default_font_size

#####################################################
#
# For the warm thick target total model::
#
#    | Model Parameter                  | OSPEX (Glesener et al. 2020, just FPMB) |This Work (just FPMB)            | This Work (FPMA&B)              |
#    | :---                             |    :----:                               |     :----:                      |                            ---: |
#    | Temperature [MK]                 | 10.2$^{+0.7}_{-0.7}$                    | 11.34$\pm$0.24                  | 11.27$\pm$0.39                  |
#    | Plasma Density [cm$^{-3}$]       | 6.0$^{+2.0}_{-2.0}\times$10$^{9}$       | 4.86$\pm$0.03$\times$10$^{9}$   | 4.71$\pm$0.14$\times$10$^{9}$   |
#    | Electron Flux [e$^{-}$ s$^{-1}$] | 1.8$^{+0.8}_{-0.8}\times$10$^{35}$      | 2.06$\pm$0.02$\times$10$^{35}$  | 2.04$\pm$0.05$\times$10$^{35}$  |
#    | Index                            | 6.3$^{+0.7}_{-0.7}$                     | 7.09$\pm$0.04                   | 7.02$\pm$0.26                   |
#    | Low Energy Cut-off [keV]         | 6.5$^{+0.9}_{-0.9}$                     | 7.00$\pm$0.04                   | 6.66$\pm$0.25                   |
#
# All parameter values appear to be within error margins (or extremely close). This is more impressive when the errors calculated in this work for the minimised values assumes the parameter's have a Gaussian and independent posterior distribution (which is clearly not the case) and so these errors are likely to be larger; to be investigated with an MCMC.
#
# The simultaneous fit of FPMA&B with the cold thick target model and the warm thick model is not able to be performed in OSPEX.
#