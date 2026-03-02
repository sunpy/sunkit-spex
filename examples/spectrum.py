"""
========
Spectrum
========

This example will demonstrate how to store spectral data in `~sunkit_spex.spectrum.Specutm` container
"""

#####################################################
#
# Imports

import numpy as np
from ndcube import NDMeta
from ndcube.extra_coords import QuantityTableCoordinate, TimeTableCoordinate

import astropy.units as u
from astropy.coordinates import SpectralCoord
from astropy.time import Time

from sunkit_spex.spectrum import Spectrum

rng = np.random.default_rng()
#####################################################
#
# 1D Spectrum
# -----------
# Let's being with the simplest case a single spectrum that is a series of measurements as function of wavelength or
# energy. We will start of by creating some synthetic data and corresponding energy bins as well as some important metadata
# in this case the exposure time.

data = rng.random(50) * u.ct
energy = np.linspace(1, 50, 50) * u.keV
time = Time("2025-02-18T15:08")

exposure_time = 5 * u.s

#####################################################
#
# Once we have our synthetic data we can create our metadata container `NDMeta` and `Spectrum` object.

meta = NDMeta()
meta.add("exposure_time", exposure_time)
meta.add("date-obs", time)

spec_1d = Spectrum(data, spectral_axis=energy, meta=meta)
spec_1d

#####################################################
#
# One of the key feature of the `Spectrum` object is the ability to slice, crop and perform other operations using
# standard sliceing methods:

spec_1d_sliced = spec_1d[10:20]
print(spec_1d_sliced.shape)
print(spec_1d_sliced.axis_world_coords_values())
print(spec_1d_sliced.meta)
print(spec_1d_sliced.spectral_axis)

#####################################################
#
# High level coordinate objects such as SkyCoord and SpectralCoord

spec_1d_crop = spec_1d.crop(SpectralCoord(10.5, unit=u.keV), SpectralCoord(20, unit=u.keV))
print(spec_1d_crop.shape)
print(spec_1d_crop.axis_world_coords_values())
print(spec_1d_crop.meta)
print(spec_1d_crop.spectral_axis)

#####################################################
#
# And Quantities

spec_1d_crop_value = spec_1d.crop_by_values((10.5 * u.keV), (20.5 * u.keV))
print(spec_1d_crop_value.shape)
print(spec_1d_crop_value.axis_world_coords_values())
print(spec_1d_crop_value.meta)
print(spec_1d_crop_value.spectral_axis)

#####################################################
#
# 2D Spectrum (spectrogram or time v energy)
# ------------------------------------------
# Let build on the previous example by increasing the dimensionality of the data in this case to a spectrogram or a
# series of spectra as a function of time. Here we will simulate a series of 10 spectra taken over 10 minutes. Again we
# begin by creating our synthetic data as before but additionally creating the time variable.

data = rng.random((10, 50)) * u.ct
energy = np.linspace(1, 50, 51) * u.keV
times = Time("2025-02-18T15:08") + np.arange(10) * u.min
exposure_time = np.arange(5, 15) * u.s

#####################################################
#
# We are also going to demonstrate the  power of the sliceable metadata, so in this example each of the individual
# spectra have different exposure times (this could be another important information regard the observation)

meta = NDMeta()
meta.add("exposure_time", exposure_time, axes=(0,))

time_coord = TimeTableCoordinate(times, names="time", physical_types="time")
energy_coord = QuantityTableCoordinate(energy, names="energy", physical_types="em.energy")
wcs = (energy_coord & time_coord).wcs

spec_2d_time_energy = Spectrum(data, spectral_axis=energy, wcs=wcs, spectral_axis_index=1, meta=meta)

######################################################
#
# Again all standard slicing works

spec_2d_time_energy[2:5]
spec_2d_time_energy[:, 10:20]
spec_2d_time_energy_sliced = spec_2d_time_energy[2:5, 10:20]

######################################################
#
# We can being to see the usefulness of the sliceable metadata notice how the exposure time entry has been sliced
# appropriately

print(spec_2d_time_energy_sliced.shape)
print(spec_2d_time_energy_sliced.axis_world_coords_values())
print(spec_2d_time_energy_sliced.meta)
print(spec_2d_time_energy_sliced.spectral_axis)

######################################################
#
# The same can be archived using height level coordinate objects
#

spec_2d_time_energy_crop = spec_2d_time_energy.crop(
    [SpectralCoord(10, unit=u.keV), Time("2025-02-18T15:10")], [SpectralCoord(20, unit=u.keV), Time("2025-02-18T15:12")]
)

print(spec_2d_time_energy_crop.shape)
print(spec_2d_time_energy_crop.axis_world_coords_values())
print(spec_2d_time_energy_crop.meta)
print(spec_2d_time_energy_crop.spectral_axis)

######################################################
#
# Or Quantities as before
spec_2d_time_energy_crop_values = spec_2d_time_energy.crop_by_values((10 * u.keV, 2 * u.min), (19.5 * u.keV, 4 * u.min))

print(spec_2d_time_energy_crop_values.shape)
print(spec_2d_time_energy_crop_values.axis_world_coords_values())
print(spec_2d_time_energy_crop_values.meta)
print(spec_2d_time_energy_crop_values.spectral_axis)

#####################################################
#
# 2D Spectrum ( e.g. detector v energy)
# -------------------------------------

data = rng.random((10, 50)) * u.ct
energy = np.linspace(1, 50, 50) * u.keV

exposure_time = np.arange(10) * u.s
labels = np.array([f"det_+{chr(97 + i)}" for i in range(10)])

meta = NDMeta()
meta.add("exposure_time", exposure_time, axes=0)
meta.add("detector", labels, axes=0)

spec_2d_det_time = Spectrum(data, spectral_axis=energy, spectral_axis_index=1, meta=meta)
spec_2d_det_time


#####################################################
#

# spec_2d_det_time.crop((SpectralCoord(10 * u.keV), None), (SpectralCoord(20 * u.keV), None))

#####################################################
#

# spec_2d_det_time.crop_by_values((10 * u.keV, 0), (20 * u.keV, 2))

#####################################################
#
# 3D Spectrum ( e.g. detector v energy v time)
# --------------------------------------------

# data = rng.random(10, 20, 30) * u.ct
# energy = np.linspace(1, 31, 31) * u.keV
#
# labels = np.array([chr(97 + i) for i in range(10)])
# exposure_time = np.arange(10 * 20).reshape(10, 20) * u.s
# times = Time.now() + np.arange(20) * u.s
#
# meta = NDMeta()
# meta.add("exposure_time", exposure_time, axes=(0, 1))
# meta.add("detector", labels, axes=(0,))
#
# spec_3d_det_energy_time = Spectrum(data, spectral_axis=energy, spectral_axis_index=2, meta=meta)
# spec_3d_det_energy_time.extra_coords.add("time", (0,), times)
#
# spec_3d_det_energy_time[:, 10:15, :].meta
# spec_3d_det_energy_time[2:3, 10:15, :].meta

#####################################################
#
# 4D Spectrum ( e.g. spatial v spatial v energy v time)
# -----------------------------------------------------

# import numpy as np
# from ndcube import NDMeta
#
# import astropy.units as u
# from astropy.time import Time
#
# data = np.random.rand(10, 10, 20, 30) * u.ct
# energy = np.linspace(1, 31, 31) * u.keV
# exposure_time = np.arange(20) * u.s
# times = Time.now() + np.arange(20) * u.s
#
# meta = NDMeta()
# meta.add("exposure_time", exposure_time, axes=(2,))
#
# wcs = astropy.wcs.WCS(naxis=2)
# wcs.wcs.ctype = "HPLT-TAN", "HPLN-TAN"
# wcs.wcs.cunit = "deg", "deg"
# wcs.wcs.cdelt = 0.5, 0.4
# wcs.wcs.crpix = 5, 6
# wcs.wcs.crval = 0.5, 1
# wcs.wcs.cname = "HPC lat", "HPC lon"
#
# cube = NDCube(data=data, wcs=wcs, meta=meta)
#
# # Now instantiate the NDCube
# spec_4d_lon_lat_time_energy = Spectrum(data, wcs=wcs, spectral_axis=energy, spectral_axis_index=3, meta=meta)
# spec_4d_lon_lat_time_energy.extra_coords.add("time", (2,), times)
#
# spec_4d_lon_lat_time_energy
