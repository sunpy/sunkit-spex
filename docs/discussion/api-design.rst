**********
API Design
**********

The purpose of `sunkit-spex` is provide the community with the appropriate data containers, models and fitting
infrastructure to fit various kind of spectra to models with an initial a focus on solar X-ray spectra.


Existing Software
=================

There are a number existing software packages designed to allow models to be fit to specific or general spectra
spectra. To avoid reinventing the wheel and making the same mistakes again these should be used as both inspiration and
cautionary tails.

* SPEX - IDL
* OSPEX - IDL
* XSPEC - command line, python,
* Sherpa -
* spectutils - python
* gammpy


Data Containers
===============

Spectral Data
-------------

.. code-block:: python

    class Spectrum(NDCube):
        r"""
        Spectrum container for 1D spectral data.

        Note that "1D" in this case refers to the fact that there is only one
        spectral axis.  `Spectrum` can contain "vector 1D spectra" by having the
        ``flux`` have a shape with dimension greater than 1.

        Parameters
        ----------
        data : `~astropy.units.Quantity`
            The data for this spectrum. This can be a simple `~astropy.units.Quantity`,
            or an existing `~Spectrum1D` or `~ndcube.NDCube` object.
        uncertainty : `~astropy.nddata.NDUncertainty`
            Contains uncertainty information along with propagation rules for
            spectrum arithmetic. Can take a unit, but if none is given, will use
            the unit defined in the flux.
        spectral_axis : `~astropy.units.Quantity` or `~specutils.SpectralAxis`
            Dispersion information with the same shape as the dimension specified by spectral_dimension
            of shape plus one if specifying bin edges.
        spectral_dimension : `int` default 0
            The dimension of the data which represents the spectral information default to first dimension index 0.
        mask : `~numpy.ndarray`-like
            Array where values in the flux to be masked are those that
            ``astype(bool)`` converts to True. (For example, integer arrays are not
            masked where they are 0, and masked for any other value.)
        meta : dict
            Arbitrary container for any user-specific information to be carried
            around with the spectrum container object.
        """


.. code-block:: python

    class CountSpectrum(Spectrum):
        r"""
        Spectrum container for count spectral data.

        Specifically, the data must be supplied as counts or convertable to counts by multiplying by the provided
        normalisation.

        Parameters
        ----------
        data : `~astropy.units.Quantity`
            The data for this spectrum. This can be a simple `~astropy.units.Quantity`,
            or an existing `~Spectrum1D` or `~ndcube.NDCube` object.
        norm : `~astropy.units.Quantity`
            The normalisation if the data unit is not counts then the product the unit of data and norm must be counts.
        uncertainty : `~astropy.nddata.NDUncertainty`
            Contains uncertainty information along with propagation rules for spectrum arithmetic. Can take a unit, but if
            none is given, will use the unit defined in the flux.
        spectral_axis : `~astropy.units.Quantity` or `~specutils.SpectralAxis`
            Dispersion information with the same shape as the dimension specified by spectral_dimension
            of shape plus one if specifying bin edges.
        spectral_dimension : `int` default 0
            The dimension of the data which represents the spectral information default to first dimension index 0.
        mask : `~numpy.ndarray`-like
            Array where values in the flux to be masked are those that
            ``astype(bool)`` converts to True. (For example, integer arrays are not
            masked where they are 0, and masked for any other value.)
        meta : dict
            Arbitrary container for any user-specific information to be carried
            around with the spectrum container object.
        """


Instrument Response
-------------------
Or more specifically spectrometer response

DRM(matrix, spectral_axis_in, spectra_axis_out) - Detector Response Matix

SRM(area, attenuation, ... DRM, spectral_axis_in, spectra_axis_out) - Spectrometer Response Matrix


Modeling
========
Model

Parameters


Fitting
=======
Fitter
