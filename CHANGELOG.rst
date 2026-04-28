0.5.0 (2026-04-21)
==================

Breaking Changes
----------------

- `sunkit_spex.sunxspex_fitting.fitter.Fitter` no longer inherits from `sunkit_spex.sunxspex_fitting.data_loader.LoadSpec`. (`#75 <https://github.com/sunpy/sunkit-spex/pull/75>`__)
- Move `add_photon_model`, `del_photon_model`, `add_var`, `del_var` to instance methods on `Fitter` class. (`#75 <https://github.com/sunpy/sunkit-spex/pull/75>`__)
- Update STIX spectrogram and srm loading and plotting functions for the new STIX file format. (`#98 <https://github.com/sunpy/sunkit-spex/pull/98>`__)
- Rename of package from 'sunxspex' to 'sunkit-spex' with module being 'sunkit_spex'. (`#114 <https://github.com/sunpy/sunkit-spex/pull/114>`__)
- Update minimum python version to python version to 3.9 and add testing for 3.10 and 3.11. (`#121 <https://github.com/sunpy/sunkit-spex/pull/121>`__)
- Renamed `sunxspex_fitting` module to `fitting_legacy` and added a new `fitting` module. (`#137 <https://github.com/sunpy/sunkit-spex/pull/137>`__)
- Remove redundant 'legacy' in module from `~sunkit_spex.legacy.fitting_legacy` to `~sunkit_spex.legacy.fitting`. (`#158 <https://github.com/sunpy/sunkit-spex/pull/158>`__)
- Bump minimum version dependencies:
   * python>=3.12
   * matplotlib>=3.9
   * numdifftools>=0.9.42
   * numpy>=1.26
   * parfive>=2.1
   * scipy>=1.12
   * sunpy>=7.0
   * xarray>=2023.12
   * ndcube>=2.3 (`#234 <https://github.com/sunpy/sunkit-spex/pull/234>`__)


Deprecations
------------

- Deprecate the `sunkit_spex.legacy` module. (`#250 <https://github.com/sunpy/sunkit-spex/pull/250>`__)


New Features
------------

- Add photon space broken power law :func:`sunxspex.photon_power_law.broken_power_law_binned_flux`. Behavior is equivalent to the OSPEX power law f_bpow.pro when the pivot option is given. Similar to XSPEC bknpower. (`#103 <https://github.com/sunpy/sunkit-spex/pull/103>`__)
- Update the RHESSI instrument loader

  - Move location to :py:class:`sunkit_spex.extern.rhessi.RhessiLoader`
  - Move all "I/O" RHESSI code into :py:mod:`sunkit_spex.extern.rhessi` too
  - Remove broken/codependent STIX loader.
  - Have RHESSI loader auto-pick the correct response based on the fitting times; warn if time range spans an attenuator state change. (`#143 <https://github.com/sunpy/sunkit-spex/pull/143>`__)
- Add code for using Astropy models in relation to a simple example of forward-fitting X-ray spectroscopy. (`#155 <https://github.com/sunpy/sunkit-spex/pull/155>`__)
- Add STIX instrument loader `~sunkit_spex.extern.stix.STIXLoader` (`#160 <https://github.com/sunpy/sunkit-spex/pull/160>`__)
- Add function `~sunkit_spex.models.physical.albedo.get_albedo_matrix` to calculate Albedo correction for given input spectrum and add model `~sunkit_spex.models.physical.albedo.Albedo` to correct modeled photon spectrum for albedo. (`#161 <https://github.com/sunpy/sunkit-spex/pull/161>`__)
- For thermal emission in `~sunkit_spex.legacy.thermal.thermal_emission`, the output flux is set to 0 for input energies above CHIANTI energy grid and raises an error if input energies are below the energy grid. (`#178 <https://github.com/sunpy/sunkit-spex/pull/178>`__)
- Add three Astropy model classes, :class:`sunkit_spex.models.physical.thermal.ThermalEmission`, :class:`sunkit_spex.models.physical.thermal.ContinuumEmission` and :class:`sunkit_spex.models.physical.thermal.LineEmission`. These Astropy model classes act as wrappers to the existing physical model functions. This provides the capability to fit using the Astropy fitting framework. (`#189 <https://github.com/sunpy/sunkit-spex/pull/189>`__)
- Allow automatic loading of the correct STIX SRM in `~sunkit_spex.extern.stix.STIXLoader` based on attenuation state within the selected time for spectral fitting. (`#193 <https://github.com/sunpy/sunkit-spex/pull/193>`__)
- Add two Astropy model classes, :class:`sunkit_spex.models.physical.nonthermal.ThickTarget` and :class:`sunkit_spex.models.physical.nonthermal.ThinTarget`. This provides the capability to fit using the Astropy fitting framework. (`#194 <https://github.com/sunpy/sunkit-spex/pull/194>`__)
- Add new two new Astropy model classes, :class:`sunkit_spex.models.scaling.DistanceScale` and :class:`sunkit_spex.models.scaling.Constant`. These classes can be used with physical models to scale the flux either by observer distance or by a constant multiplicative factor respectively. (`#195 <https://github.com/sunpy/sunkit-spex/pull/195>`__)
- Adds functionality to enable class:`StraightLineModel` and class:`GaussianModel` to be evaluated at energy edges and return an output at energy centers. (`#202 <https://github.com/sunpy/sunkit-spex/pull/202>`__)
- Add the possibility to perform albedo correction within the legacy code. The alebdo matrix is generated with :func:`sunkit_spex.legacy.fitting.albedo.get_albedo_matrix` and the albedo correction is performed in  :class:`sunkit_spex.legacy.fitting.fitter.Fitter`. (`#206 <https://github.com/sunpy/sunkit-spex/pull/206>`__)
- Adds an example notebook for fitting a single STIX spectrum and a joint fit with STIX imaging and background detectors when an attenuator is used. (`#217 <https://github.com/sunpy/sunkit-spex/pull/217>`__)
- Add a new `sunkit_spex.spectrum.spectrum.Spectrum` object to hold spectral data. `~sunkit_spex.spectrum.spectrum.Spectrum` is based on `NDCube` and butils on it coordinate aware methods and metadata handling. (`#239 <https://github.com/sunpy/sunkit-spex/pull/239>`__)


Bug Fixes
---------

- Added special case in C-stat likelihood function :func:`sunkit_spex.sunxspex_fitting.likelihoods.LogLikelihoods.cstat_loglikelihood` it defaults to the Poisson likelihood now when data is zero. (`#85 <https://github.com/sunpy/sunkit-spex/pull/85>`__)
- Removed cast to `int` when converting the model count rates (counts/second) to just counts since the model represents the average number of counts, there is not need for these to be integers. (`#87 <https://github.com/sunpy/sunkit-spex/pull/87>`__)
- Fix bug where MCMC random walker starter positions were not correctly calculated. (`#89 <https://github.com/sunpy/sunkit-spex/pull/89>`__)
- Fixed bug where passing a user defined instrument loader class to :class:`sunkit_spex/sunxspex_fitting/fitter.Fitter` caused an error. (`#108 <https://github.com/sunpy/sunkit-spex/pull/108>`__)
- Upper bound of thick target emission function in :func:`sunxspex.sunxspex_fitting.photon_models_for_fitting.thick_fn` is calculated from highest energy input. (`#115 <https://github.com/sunpy/sunkit-spex/pull/115>`__)
- Fixed bug where changing the Chianti file for `sunkit_spex.thermal` and using :class:`sunkit_spex/sunxspex_fitting/fitter.Fitter` caused an error. (`#134 <https://github.com/sunpy/sunkit-spex/pull/134>`__)
- Fixes bug which caused energy units to get squared; update the way units are assigned in `~sunkit_spex.legacy.thermal.thermal_emission`. (`#171 <https://github.com/sunpy/sunkit-spex/pull/171>`__)
- Add back legacy parameters/functionality relating to `~sunkit_spex.extern.stix.RhessiLoader`. (`#181 <https://github.com/sunpy/sunkit-spex/pull/181>`__)
- Gallery examples of fitting NuSTAR data now work and Numpy version dependent bug in `~sunkit_spex.legacy.fitting.photon_models_for_fitting.thick_fn` is fixed. (`#199 <https://github.com/sunpy/sunkit-spex/pull/199>`__)
- Fixes fitting by removing input_unit_equivalencies from thermal classes, class:ThermalEmission , class:ContinuumEmission and class:LineEmission . This allows compound emission to function. (`#210 <https://github.com/sunpy/sunkit-spex/pull/210>`__)
- Fix bug introduced in refactoring of `~sunkit_spex.models.physical.albedo.Albedo` model. Internally the angle theta
  given in degrees wasn't converted to radians before use. (`#212 <https://github.com/sunpy/sunkit-spex/pull/212>`__)
- Fixes bug in count_rate calculation in class:`StraightLineModel`. We now calculate the mean of count rates for a given time range rather than the sum. (`#214 <https://github.com/sunpy/sunkit-spex/pull/214>`__)
- Ensure that the module-level abundance tables don't change in the legacy and release thermal modules. (`#231 <https://github.com/sunpy/sunkit-spex/pull/231>`__)
- Fix time selection bugs in `~sunkit_spex.extern.stix.STIXLoader` (`#241 <https://github.com/sunpy/sunkit-spex/pull/241>`__)


Documentation
-------------

- Add 'sphinx-gallery' to how-to section. (`#150 <https://github.com/sunpy/sunkit-spex/pull/150>`__)
- Add example gallery and convert ipython notebooks to gallery format. (`#153 <https://github.com/sunpy/sunkit-spex/pull/153>`__)
- Add installation instructions in documentation for sunkit-spex development version. (`#236 <https://github.com/sunpy/sunkit-spex/pull/236>`__)
- Split the gallery into `Examples` and `Legacy Examples` sections. (`#250 <https://github.com/sunpy/sunkit-spex/pull/250>`__)


Internal Changes
----------------

- Configure giles bot to check for change log entries. (`#70 <https://github.com/sunpy/sunkit-spex/pull/70>`__)
- Update SSW data url to point to lmsal server. (`#90 <https://github.com/sunpy/sunkit-spex/pull/90>`__)
- Update references to new repository name `sunkit-spex`. (`#129 <https://github.com/sunpy/sunkit-spex/pull/129>`__)
- Move CI to GitHub actions. (`#138 <https://github.com/sunpy/sunkit-spex/pull/138>`__)
- Update license and copyright years. (`#145 <https://github.com/sunpy/sunkit-spex/pull/145>`__)
- Remove unused dependencies which started to cause issues. Update URLs in examples for server change. Downgrade python in RTD environment to 3.10. (`#200 <https://github.com/sunpy/sunkit-spex/pull/200>`__)
- Remove unused files from root directory; add `dev` target to pyproject (`#221 <https://github.com/sunpy/sunkit-spex/pull/221>`__)
- Removes commented out code from `~sunkit_spex.models.physical.thermal` (`#232 <https://github.com/sunpy/sunkit-spex/pull/232>`__)
