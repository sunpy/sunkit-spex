import copy
from copy import deepcopy

import numpy as np
from gwcs import WCS as GWCS
from gwcs import coordinate_frames as cf
from ndcube import NDCube

import astropy.units as u
from astropy.coordinates import SpectralCoord
from astropy.modeling.mappings import Identity, Mapping
from astropy.modeling.tabular import Tabular1D
from astropy.utils import lazyproperty
from astropy.wcs.wcsapi import sanitize_slices

__all__ = ["SpectralAxis", "Spectrum", "gwcs_from_array"]


__doctest_requires__ = {"Spectrum": ["ndcube>=2.3"]}


class SpectralGWCS(GWCS):
    """
    This is a placeholder lookup-table GWCS created when a :class:`~specutils.Spectrum` is
    instantiated with a ``spectral_axis`` and no WCS.
    """

    def __init__(self, *args, **kwargs):
        self.original_unit = kwargs.pop("original_unit", "")
        super().__init__(*args, **kwargs)

    def copy(self):
        """
        Return a shallow copy of the object.

        Convenience method so user doesn't have to import the
        :mod:`copy` stdlib module.

        .. warning::
            Use `deepcopy` instead of `copy` unless you know why you need a
            shallow copy.
        """
        return copy.copy(self)

    def deepcopy(self):
        """
        Return a deep copy of the object.

        Convenience method so user doesn't have to import the
        :mod:`copy` stdlib module.
        """
        return copy.deepcopy(self)


def gwcs_from_array(array, flux_shape, spectral_axis_index=None):
    """
    Create a new WCS from provided tabular data. This defaults to being
    a GWCS object with a lookup table for the spectral axis and filler
    pixel to pixel identity conversions for spatial axes, if they exist.
    """
    orig_array = u.Quantity(array)
    naxes = len(flux_shape)

    if naxes > 1:
        if spectral_axis_index is None:
            raise ValueError("spectral_axis_index must be set for multidimensional flux arrays")
        # Axis order is reversed for WCS from numpy array
        spectral_axis_index = naxes - spectral_axis_index - 1
    elif naxes == 1:
        spectral_axis_index = 0

    axes_order = list(np.arange(naxes))
    axes_type = [
        "SPATIAL",
    ] * naxes
    axes_type[spectral_axis_index] = "SPECTRAL"

    detector_frame = cf.CoordinateFrame(
        naxes=naxes,
        name="detector",
        unit=[
            u.pix,
        ]
        * naxes,
        axes_order=axes_order,
        axes_type=axes_type,
    )

    if array.unit in ("", "pix", "pixel"):
        # Spectrum was initialized without a wcs or spectral axis
        spectral_frame = cf.CoordinateFrame(
            naxes=1,
            unit=[
                array.unit,
            ],
            axes_type=[
                "Spectral",
            ],
            axes_order=(spectral_axis_index,),
        )
    else:
        phys_types = None
        # Note that some units have multiple physical types, so we can't just set the
        # axis name to the physical type string.
        if array.unit.physical_type == "length":
            axes_names = [
                "wavelength",
            ]
        elif array.unit.physical_type == "frequency":
            axes_names = [
                "frequency",
            ]
        elif array.unit.physical_type == "velocity":
            axes_names = [
                "velocity",
            ]
            phys_types = [
                "spect.dopplerVeloc.optical",
            ]
        elif array.unit.physical_type == "wavenumber":
            axes_names = [
                "wavenumber",
            ]
        elif array.unit.physical_type == "energy":
            axes_names = [
                "energy",
            ]
        else:
            raise ValueError("Spectral axis units must be one of length,frequency, velocity, energy, or wavenumber")

        spectral_frame = cf.SpectralFrame(
            unit=array.unit, axes_order=(spectral_axis_index,), axes_names=axes_names, axis_physical_types=phys_types
        )

    if naxes > 1:
        axes_order.remove(spectral_axis_index)
        spatial_frame = cf.CoordinateFrame(
            naxes=naxes - 1,
            unit=[
                "",
            ]
            * (naxes - 1),
            axes_type=[
                "Spatial",
            ]
            * (naxes - 1),
            axes_order=axes_order,
        )
        output_frame = cf.CompositeFrame(frames=[spatial_frame, spectral_frame])
    else:
        output_frame = spectral_frame

    # In order for the world_to_pixel transformation to automatically convert
    # input units, the equivalencies in the look up table have to be extended
    # with spectral unit information.
    SpectralTabular1D = type(
        "SpectralTabular1D", (Tabular1D,), {"input_units_equivalencies": {"x0": u.spectral()}, "bounds_error": True}
    )

    # We pass through the pixel values of spatial axes with Identity and use a lookup
    # table for the spectral axis values. We use Mapping to pipe the values to the correct
    # model depending on which axis is the spectral axis
    if naxes == 1:
        forward_transform = SpectralTabular1D(np.arange(len(array)) * u.pix, lookup_table=array)
    else:
        axes_order.append(spectral_axis_index)
        # WCS axis order is reverse of numpy array order
        mapped_axes = axes_order
        out_mapping = np.ones(len(mapped_axes)).astype(int)
        for i in range(len(mapped_axes)):
            out_mapping[mapped_axes[i]] = i
        forward_transform = (
            Mapping(mapped_axes)
            | Identity(naxes - 1) & SpectralTabular1D(np.arange(len(array)) * u.pix, lookup_table=array)
            | Mapping(out_mapping)
        )

    # If our spectral axis is in descending order, we have to flip the lookup
    # table to be ascending in order for world_to_pixel to work.
    forward_transform.inverse = SpectralTabular1D(array, lookup_table=np.arange(len(array)) * u.pix)

    tabular_gwcs = SpectralGWCS(
        original_unit=orig_array.unit,
        forward_transform=forward_transform,
        input_frame=detector_frame,
        output_frame=output_frame,
    )
    tabular_gwcs.bounding_box = None

    # Store the intended unit from the origin input array
    #     tabular_gwcs._input_unit = orig_array.unit

    return tabular_gwcs


class SpectralAxis(SpectralCoord):
    r"""
    Coordinate object representing spectral values corresponding to a specific
    spectrum. Overloads SpectralCoord with additional information (currently
    only bin edges).

    Parameters
    ----------
    bin_specification: str, optional
        Must be "edges" or "centers". Determines whether specified axis values
        are interpreted as bin edges or bin centers. Defaults to "centers".
    """

    _equivalent_unit = (*SpectralCoord._equivalent_unit, u.pixel)

    def __new__(cls, value, *args, bin_specification="centers", **kwargs):
        # Enforce pixel axes are ascending
        if (
            (type(value) is u.quantity.Quantity)
            and (value.size > 1)
            and (value.unit is u.pix)
            and (value[-1] <= value[0])
        ):
            raise ValueError("u.pix spectral axes should always be ascending")

        if bin_specification == "edges" and value.size < 2:
            raise ValueError('If bin_specification="centers" have at least two bin edges.')

        # Convert to bin centers if bin edges were given, since SpectralCoord
        # only accepts centers
        if bin_specification == "edges":
            bin_edges = value
            value = SpectralAxis._centers_from_edges(value)

        obj = super().__new__(cls, value, *args, **kwargs)

        if bin_specification == "edges":
            obj._bin_edges = bin_edges

        return obj

    @staticmethod
    def _centers_from_edges(edges):
        r"""
        Calculates the bin centers as the average of each pair of edges
        """
        return (edges[1:] + edges[:-1]) / 2

    @lazyproperty
    def bin_edges(self):
        r"""
        Calculates bin edges if the spectral axis was created with centers
        specified.
        """
        if hasattr(self, "_bin_edges") and self._bin_edges is not None:
            return self._bin_edges
        return None

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self._bin_edges = getattr(obj, "_bin_edges", None)


class Spectrum(NDCube):
    r"""
    Spectrum container for data which share a common spectral axis.

    Note that "1D" in this case refers to the fact that there is only one
    spectral axis. `Spectrum` can contain ND data where
    ``data`` have a shape with dimension greater than 1.

    Notes
    -----
    A stripped down version of `Spectrum1D` from `specutils`.

    Parameters
    ----------
    data : `~astropy.units.Quantity`
        The data for this spectrum. This can be a simple `~astropy.units.Quantity`,
        or an existing `~Spectrum` or `~ndcube.NDCube` object.
    uncertainty : `~astropy.nddata.NDUncertainty`
        Contains uncertainty information along with propagation rules for
        spectrum arithmetic. Can take a unit, but if none is given, will use
        the unit defined in the data.
    spectral_axis : `~astropy.units.Quantity` or `~specutils.SpectralAxis`
        Dispersion information with the same shape as the dimension specified by spectral_axis_index
        or shape plus one if specifying bin edges.
    spectral_axis_index : `int` default 0
        The dimension of the data which represents the spectral information default to first dimension index 0.
    mask : `~numpy.ndarray`-like
        Array where values in the flux to be masked are those that
        ``astype(bool)`` converts to True. (For example, integer arrays are not
        masked where they are 0, and masked for any other value.)
    meta : dict
        Arbitrary container for any user-specific information to be carried
        around with the spectrum container object.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from sunkit_spex.spectrum import Spectrum
    >>> spec = Spectrum(np.arange(1, 11)*u.watt,spectral_axis=np.arange(1, 12)*u.keV)
    >>> spec
    <sunkit_spex.spectrum.spectrum.Spectrum object at ...
    NDCube
    ------
    Shape: (10,)
    Physical Types of Axes: [('em.energy',)]
    Unit: W
    Data Type: float64
    """

    def __init__(
        self,
        data,
        *,
        spectral_axis=None,
        spectral_axis_index=None,
        wcs=None,
        **kwargs,
    ):
        # If the data argument is already a Spectrum (as it would
        # be for internal arithmetic operations), avoid setup entirely.
        if isinstance(data, Spectrum):
            self._spectral_axis = data.spectral_axis
            self._spectral_axis_index = data.spectral_axis_index
            super().__init__(data)
            return

        # Check for pre-defined entries in the kwargs dictionary.
        unknown_kwargs = set(kwargs).difference(
            {"data", "unit", "uncertainty", "meta", "mask", "copy", "extra_coords", "global_coords", "psf"}
        )

        if len(unknown_kwargs) > 0:
            raise ValueError(
                "Initializer contains unknown arguments(s): {}.".format(", ".join(map(str, unknown_kwargs)))
            )

        # Handle initializing from NDCube objects
        if isinstance(data, NDCube):
            if data.unit is None:
                raise ValueError("Input NDCube missing unit parameter")

            if spectral_axis is None:
                raise ValueError("Spectral axis must be specified")

            # Change the data array from bare ndarray to a Quantity
            q_data = data.data << u.Unit(data.unit)

            self.__init__(
                q_data, wcs=data.wcs, mask=data.mask, uncertainty=data.uncertainty, spectral_axis=spectral_axis
            )
            return

        self._spectral_axis_index = spectral_axis_index
        # If here data should be an array or quantity
        if spectral_axis_index is None and data is not None:
            if data.ndim == 1:
                self._spectral_axis_index = 0
        elif data is None:
            self._spectral_axis_index = 0

        # Ensure that the unit information codified in the quantity object is
        # the One True Unit.
        kwargs.setdefault("unit", data.unit if isinstance(data, u.Quantity) else kwargs.get("unit"))

        # If a WCS is provided, determine which axis is the spectral axis
        if wcs is not None:
            if spectral_axis is None:
                raise ValueError("Spectral axis must be specified")

            naxis = None
            if hasattr(wcs, "naxis"):
                naxis = wcs.naxis
            # GWCS doesn't have naxis
            elif hasattr(wcs, "world_n_dim"):
                naxis = wcs.world_n_dim

            if naxis is not None and naxis > 1:
                temp_axes = []
                phys_axes = wcs.world_axis_physical_types
                if self._spectral_axis_index is None:
                    for i in range(len(phys_axes)):
                        if phys_axes[i] is None:
                            continue
                        if phys_axes[i][0:2] == "em" or phys_axes[i][0:5] == "spect" or phys_axes[i][7:12] == "Spect":
                            temp_axes.append(i)
                    if len(temp_axes) != 1:
                        raise ValueError(
                            f"Input WCS must have exactly one axis with spectral units, found {len(temp_axes)}"
                        )
                    # Due to FITS conventions, the WCS axes are listed in opposite
                    # order compared to the data array.
                    self._spectral_axis_index = len(data.shape) - temp_axes[0] - 1

            else:
                if data is not None and data.ndim == 1:
                    self._spectral_axis_index = 0
                else:
                    if self.spectral_axis_index is None:
                        raise ValueError("WCS is 1D but flux is multi-dimensional. Please specify spectral_axis_index.")

        # If data and spectral axis are both specified, check that their lengths
        # match or are off by one (implying the spectral axis stores bin edges)
        bin_specification = "centers"  # default value
        if data is not None and spectral_axis is not None:
            if spectral_axis.shape[0] == data.shape[self.spectral_axis_index]:
                bin_specification = "centers"
            elif spectral_axis.shape[0] == data.shape[self.spectral_axis_index] + 1:
                bin_specification = "edges"
            else:
                raise ValueError(
                    f"Spectral axis length ({spectral_axis.shape[0]}) must be the "
                    "same size or one greater (if specifying bin edges) than that "
                    f"of the corresponding data axis ({data.shape[self.spectral_axis_index]})"
                )

        # Attempt to parse the spectral axis. If none is given, try instead to
        # parse a given wcs. This is put into a GWCS object to
        # then be used behind-the-scenes for all operations.

        # Ensure that the spectral axis is an astropy Quantity or SpectralAxis
        if not isinstance(spectral_axis, (u.Quantity, SpectralAxis)):
            raise ValueError("Spectral axis must be a `Quantity` or `SpectralAxis` object.")

        # If spectral axis is provided as an astropy Quantity, convert it
        # to a specutils SpectralAxis object.
        if not isinstance(spectral_axis, SpectralAxis):
            self._spectral_axis = SpectralAxis(spectral_axis, bin_specification=bin_specification)
        # If a SpectralAxis object is provided, we assume it doesn't need
        # information from other keywords added
        else:
            self._spectral_axis = spectral_axis

        # Check the spectral_axis matches the wcs
        if wcs is not None:
            wsc_coords = None
            if hasattr(wcs, "spectral") and getattr(wcs, "is_spectral", False):
                wcs_coords = wcs.spectral.pixel_to_world(np.arange(data.shape[self.spectral_axis_index])).to("keV")
            elif wcs.pixel_n_dim == 1:
                wcs_coords = wcs.pixel_to_world(np.arange(data.shape[self.spectral_axis_index]))
            # else:
            #     array_index = wcs.pixel_n_dim - self._spectral_axis_index - 1
            #     pixels = [0] * wcs.pixel_n_dim
            #     pixels[array_index] = np.arange(data.shape[self.spectral_axis_index])
            #     wcs_coords = wcs.pixel_to_world(*pixels)[array_index]
            if wsc_coords is not None:
                if not u.allclose(self._spectral_axis, wcs_coords):
                    raise ValueError(
                        f"Spectral axis {self._spectral_axis} and wcs spectral axis {wcs_coords} must match."
                    )

        if wcs is None:
            wcs = gwcs_from_array(self._spectral_axis, data.shape, spectral_axis_index=self.spectral_axis_index)

        super().__init__(data=data.value if isinstance(data, u.Quantity) else data, wcs=wcs, **kwargs)

        # make sure that spectral axis is strictly increasing or strictly decreasing
        is_strictly_increasing = np.all(self._spectral_axis[1:] > self._spectral_axis[:-1])
        if len(self._spectral_axis) > 1 and not (is_strictly_increasing):
            raise ValueError("Spectral axis must be strictly increasing decreasing.")

        if hasattr(self, "uncertainty") and self.uncertainty is not None:
            if not data.shape == self.uncertainty.array.shape:
                raise ValueError(
                    f"Data axis ({data.shape}) and uncertainty ({self.uncertainty.array.shape}) shapes must be the "
                    "same."
                )

    def __getitem__(self, item):
        sliced_cube = super().__getitem__(item)
        item = tuple(sanitize_slices(item, len(self.shape)))
        sliced_spec_axis = self.spectral_axis[item[self.spectral_axis_index]]
        return Spectrum(sliced_cube, spectral_axis=sliced_spec_axis)

    def _slice(self, item):
        kwargs = super()._slice(item)
        item = tuple(sanitize_slices(item, len(self.shape)))

        kwargs["spectral_axis_index"] = self.spectral_axis_index
        kwargs["spectral_axis"] = self.spectral_axis[item[self.spectral_axis_index]]
        return kwargs

    def _new_instance(self, **kwargs):
        keys = ("unit", "wcs", "mask", "meta", "uncertainty", "psf", "spectral_axis")
        full_kwargs = {k: deepcopy(getattr(self, k, None)) for k in keys}
        # We Explicitly DO NOT deepcopy any data
        full_kwargs["data"] = self.data
        full_kwargs.update(kwargs)
        new_spectrum = type(self)(**full_kwargs)
        if self.extra_coords is not None:
            new_spectrum._extra_coords = deepcopy(self.extra_coords)
        if self.global_coords is not None:
            new_spectrum._global_coords = deepcopy(self.global_coords)
        return new_spectrum

    @property
    def spectral_axis(self):
        return self._spectral_axis

    @property
    def spectral_axis_index(self):
        return self._spectral_axis_index
