0.1.dev158+g658b502.d20231003 (2023-10-03)
==========================================

Breaking Changes
----------------

- `sunkit_spex.sunxspex_fitting.fitter.Fitter` no longer inherits from `sunkit_spex.sunxspex_fitting.data_loader.LoadSpec`. (`#75 <https://github.com/sunpy/sunxspex/pull/75>`__)
- Move `add_photon_model`, `del_photon_model`, `add_var`, `del_var` to instance methods on `Fitter` class. (`#75 <https://github.com/sunpy/sunxspex/pull/75>`__)
- Update STIX spectrogram and srm loading and plotting functions for the new STIX file format. (`#98 <https://github.com/sunpy/sunxspex/pull/98>`__)
- Rename of package from 'sunxspex' to 'sunkit-spex' with module being 'sunkit_spex'. (`#114 <https://github.com/sunpy/sunxspex/pull/114>`__)


New Features
------------

- Add photon space broken power law :func:`sunxspex.photon_power_law.broken_power_law_binned_flux`. Behavior is equivalent to the OSPEX power law f_bpow.pro when the pivot option is given. Similar to XSPEC bknpower. (`#103 <https://github.com/sunpy/sunxspex/pull/103>`__)


Bug Fixes
---------

- Added special case in C-stat likelihood function :func:`sunkit_pex.sunxspex_fitting.likelihoods.LogLikelihoods.cstat_loglikelihood` it defaults to the Poisson likelihood now when data is zero. (`#85 <https://github.com/sunpy/sunxspex/pull/85>`__)
- Removed cast to `int` when converting the model count rates (counts/second) to just counts since the model represents the average number of counts, there is not need for these to be integers. (`#87 <https://github.com/sunpy/sunxspex/pull/87>`__)
- Fix bug where MCMC random walker starter positions were not correctly calculated. (`#89 <https://github.com/sunpy/sunxspex/pull/89>`__)
- Fixed bug where passing a user defined instrument loader class to :class:`sunkit_spex/sunxspex_fitting/fitter.Fitter` caused an error. (`#108 <https://github.com/sunpy/sunxspex/pull/108>`__)


Internal Changes
----------------

- Configure giles bot to check for change log entries. (`#70 <https://github.com/sunpy/sunxspex/pull/70>`__)
- Update SSW data url to point to lmsal server. (`#90 <https://github.com/sunpy/sunxspex/pull/90>`__)


sunkit_spex 0.1.dev121+gbe82142 (2022-04-16)
============================================

No significant changes.
