import numpy as np
from gwcs import WCS as GWCS
from gwcs import coordinate_frames as cf
from ndcube import NDCube

import astropy.units as u
from astropy.coordinates import SpectralCoord
from astropy.modeling.tabular import Tabular1D
from astropy.utils import lazyproperty

__all__ = ["SpectralAxis", "Spectrum", "gwcs_from_array"]
__doctest_requires__ = {"Spectrum": ["ndcube>=2.3"]}


__doctest_requires__ = {"Spectrum": ["ndcube>=2.3"]}


def gwcs_from_array(array):
    """
    Create a new WCS from provided tabular data. This defaults to being
    a GWCS object.
    """
    orig_array = u.Quantity(array)

    coord_frame = cf.CoordinateFrame(naxes=1, axes_type=("SPECTRAL",), axes_order=(0,))
    spec_frame = cf.SpectralFrame(unit=array.unit, axes_order=(0,))

    # In order for the world_to_pixel transformation to automatically convert
    # input units, the equivalencies in the lookup table have to be extended
    # with spectral unit information.
    SpectralTabular1D = type("SpectralTabular1D", (Tabular1D,), {"input_units_equivalencies": {"x0": u.spectral()}})

    forward_transform = SpectralTabular1D(np.arange(len(array)), lookup_table=array)
    # If our spectral axis is in descending order, we have to flip the lookup
    # table to be ascending in order for world_to_pixel to work.
    if len(array) == 0 or array[-1] > array[0]:
        forward_transform.inverse = SpectralTabular1D(array, lookup_table=np.arange(len(array)))
    else:
        forward_transform.inverse = SpectralTabular1D(array[::-1], lookup_table=np.arange(len(array))[::-1])

    class SpectralGWCS(GWCS):
        def pixel_to_world(self, *args, **kwargs):
            if orig_array.unit == "":
                return u.Quantity(super().pixel_to_world_values(*args, **kwargs))
            return super().pixel_to_world(*args, **kwargs).to(orig_array.unit, equivalencies=u.spectral())

    return SpectralGWCS(forward_transform=forward_transform, input_frame=coord_frame, output_frame=spec_frame)

    # Store the intended unit from the origin input array
    #     tabular_gwcs._input_unit = orig_array.unit


class SpectralAxis(SpectralCoord):
    """
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
    def _edges_from_centers(centers, unit):
        """
        Calculates interior bin edges based on the average of each pair of
        centers, with the two outer edges based on extrapolated centers added
        to the beginning and end of the spectral axis.
        """
        a = np.insert(centers, 0, 2 * centers[0] - centers[1])
        b = np.append(centers, 2 * centers[-1] - centers[-2])
        edges = (a + b) / 2
        return edges * unit

    @staticmethod
    def _centers_from_edges(edges):
        """
        Calculates the bin centers as the average of each pair of edges
        """
        return (edges[1:] + edges[:-1]) / 2

    @lazyproperty
    def bin_edges(self):
        """
        Calculates bin edges if the spectral axis was created with centers
        specified.
        """
        if hasattr(self, "_bin_edges"):
            return self._bin_edges
        return self._edges_from_centers(self.value, self.unit)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self._bin_edges = getattr(obj, "_bin_edges", None)


class Spectrum(NDCube):
    r"""
    Spectrum container for data with one spectral axis.

    Note that "1D" in this case refers to the fact that there is only one
    spectral axis.  `Spectrum` can contain "vector 1D spectra" by having the
    ``flux`` have a shape with dimension greater than 1.

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
        the unit defined in the flux.
    spectral_axis : `~astropy.units.Quantity` or `~specutils.SpectralAxis`
        Dispersion information with the same shape as the dimension specified by spectral_axis_index
        of shape plus one if specifying bin edges.
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
        uncertainty=None,
        spectral_axis=None,
        spectral_axis_index=None,
        wcs=None,
        mask=None,
        meta=None,
        **kwargs,
    ):
        # If the data argument is already a Spectrum (as it would
        # be for internal arithmetic operations), avoid setup entirely.
        if isinstance(data, Spectrum):
            self._spectral_axis = spectral_axis
            self._spectral_dimension = spectral_axis_index
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

            # Change the data array from bare ndarray to a Quantity
            q_data = data.data << u.Unit(data.unit)

            self.__init__(q_data, uncertainty=data.uncertainty, mask=data.mask, wcs=data.wcs)
            return

        self._spectral_axis_index = spectral_axis_index
        # If here data is should be an array or quantity
        if spectral_axis_index is None and data is not None:
            if data.ndim == 1:
                self._spectral_axis_index = 0
        elif data is None:
            self._spectral_axis_index = 0

        # Ensure that the data argument is an astropy quantity
        # if data is not None:
        #     if not isinstance(data, u.Quantity):
        #         raise ValueError("Data must be a `Quantity` object.")
        #     if data.isscalar:
        #         data = u.Quantity([data])

        # Ensure that the unit information codified in the quantity object is
        # the One True Unit.
        kwargs.setdefault("unit", data.unit if isinstance(data, u.Quantity) else kwargs.get("unit"))

        # If flux and spectral axis are both specified, check that their lengths
        # match or are off by one (implying the spectral axis stores bin edges)
        if data is not None and spectral_axis is not None:
            if spectral_axis.shape[0] == data.shape[self.spectral_axis_index]:
                bin_specification = "centers"
            elif spectral_axis.shape[0] == data.shape[self.spectral_axis_index] + 1:
                bin_specification = "edges"
            else:
                raise ValueError(
                    f"Spectral axis length ({spectral_axis.shape[0]}) must be the "
                    "same size or one greater (if specifying bin edges) than that "
                    f"of the corresponding flux axis ({data.shape[self.spectral_axis_index]})"
                )

        # If a WCS is provided, determine which axis is the spectral axis
        if wcs is not None:
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

        # Attempt to parse the spectral axis. If none is given, try instead to
        # parse a given wcs. This is put into a GWCS object to
        # then be used behind-the-scenes for all operations.
        if spectral_axis is not None:
            # Ensure that the spectral axis is an astropy Quantity
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

            if wcs is None:
                wcs = gwcs_from_array(self._spectral_axis)

        elif wcs is None:
            # If no spectral axis or wcs information is provided, initialize
            # with an empty gwcs based on the data.
            if self.spectral_axis_index is None:
                if data.ndim == 1:
                    self._spectral_axis_index = 0
                else:
                    raise ValueError("Must specify spectral_axis_index if no WCS or spectral axis is input.")
            size = data.shape[self.spectral_axis_index] if not data.isscalar else 1
            wcs = gwcs_from_array(
                np.arange(size) * u.Unit("pixel"), data.shape, spectral_axis_index=self.spectral_axis_index
            )

        super().__init__(data=data.value if isinstance(data, u.Quantity) else data, wcs=wcs, **kwargs)

        # If no spectral_axis was provided, create a SpectralCoord based on
        # the WCS
        if spectral_axis is None:
            # If the WCS doesn't have a spectral attribute, we assume it's the
            # dummy GWCS we created or a solely spectral WCS
            if hasattr(self.wcs, "spectral"):
                # Handle generated 1D WCS that aren't set to spectral
                if not self.wcs.is_spectral and self.wcs.naxis == 1:
                    spec_axis = self.wcs.pixel_to_world(np.arange(self.data.shape[self.spectral_axis_index]))
                else:
                    spec_axis = self.wcs.spectral.pixel_to_world(np.arange(self.data.shape[self.spectral_axis_index]))
            else:
                # We now keep the entire GWCS, including spatial information, so we need to include
                # all axes in the pixel_to_world call. Note that this assumes/requires that the
                # dispersion is the same at all spatial locations.
                wcs_args = []
                for i in range(len(self.data.shape)):
                    wcs_args.append(np.zeros(self.data.shape[self.spectral_axis_index]))
                # Replace with arange for the spectral axis
                wcs_args[self.spectral_axis_index] = np.arange(self.data.shape[self.spectral_axis_index])
                wcs_args.reverse()
                temp_coords = self.wcs.pixel_to_world(*wcs_args)
                # If there are spatial axes, temp_coords will have a SkyCoord and a SpectralCoord
                if isinstance(temp_coords, list):
                    for coords in temp_coords:
                        if isinstance(coords, SpectralCoord):
                            spec_axis = coords
                            break
                    else:
                        # WCS axis ordering is reverse of numpy
                        spec_axis = temp_coords[len(temp_coords) - self.spectral_axis_index - 1]
                else:
                    spec_axis = temp_coords

            try:
                if spec_axis.unit.is_equivalent(u.one):
                    spec_axis = spec_axis * u.pixel
            except AttributeError:
                raise AttributeError(f"spec_axis does not have unit: {type(spec_axis)} {spec_axis}")

            self._spectral_axis = SpectralAxis(spec_axis)

        # make sure that spectral axis is strictly increasing
        if not np.all(self._spectral_axis[1:] >= self._spectral_axis[:-1]):
            raise ValueError("Spectral axis must be strictly increasing or decreasing.")

        if hasattr(self, "uncertainty") and self.uncertainty is not None:
            if not data.shape == self.uncertainty.array.shape:
                raise ValueError(
                    f"Data axis ({data.shape}) and uncertainty ({self.uncertainty.array.shape}) shapes must be the "
                    "same."
                )

    # @property
    # def data(self):
    #     return u.Quantity(self._data, unit=self.unit, copy=False)

    @property
    def spectral_axis(self):
        return self._spectral_axis

    @property
    def spectral_axis_index(self):
        return self._spectral_axis_index
