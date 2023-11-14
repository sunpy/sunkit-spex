from types import SimpleNamespace
from inspect import signature
from functools import partial

import numpy as np
from gwcs import WCS as GWCS
from gwcs import coordinate_frames as cf
from ndcube import NDCube
from scipy.optimize import curve_fit

import astropy.units as u
from astropy.coordinates import SpectralCoord
from astropy.modeling.tabular import Tabular1D
from astropy.utils import lazyproperty


def gwcs_from_array(array):
    """
    Create a new WCS from provided tabular data. This defaults to being
    a GWCS object.
    """
    orig_array = u.Quantity(array)

    coord_frame = cf.CoordinateFrame(naxes=1,
                                     axes_type=('SPECTRAL',),
                                     axes_order=(0,))
    spec_frame = cf.SpectralFrame(unit=array.unit, axes_order=(0,))

    # In order for the world_to_pixel transformation to automatically convert
    # input units, the equivalencies in the look up table have to be extended
    # with spectral unit information.
    SpectralTabular1D = type("SpectralTabular1D", (Tabular1D,),
                             {'input_units_equivalencies': {'x0': u.spectral()}})

    forward_transform = SpectralTabular1D(np.arange(len(array)),
                                          lookup_table=array)
    # If our spectral axis is in descending order, we have to flip the lookup
    # table to be ascending in order for world_to_pixel to work.
    if len(array) == 0 or array[-1] > array[0]:
        forward_transform.inverse = SpectralTabular1D(
            array, lookup_table=np.arange(len(array)))
    else:
        forward_transform.inverse = SpectralTabular1D(
            array[::-1], lookup_table=np.arange(len(array))[::-1])

    class SpectralGWCS(GWCS):
        def pixel_to_world(self, *args, **kwargs):
            if orig_array.unit == '':
                return u.Quantity(super().pixel_to_world_values(*args, **kwargs))
            return super().pixel_to_world(*args, **kwargs).to(
                orig_array.unit, equivalencies=u.spectral())

    tabular_gwcs = SpectralGWCS(forward_transform=forward_transform,
                                input_frame=coord_frame,
                                output_frame=spec_frame)

    # Store the intended unit from the origin input array
    #     tabular_gwcs._input_unit = orig_array.unit

    return tabular_gwcs


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

    _equivalent_unit = SpectralCoord._equivalent_unit + (u.pixel,)

    def __new__(cls, value, *args, bin_specification="centers", **kwargs):

        # Enforce pixel axes are ascending
        if ((type(value) is u.quantity.Quantity) and
                (value.size > 1) and
                (value.unit is u.pix) and
                (value[-1] <= value[0])):
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
        a = np.insert(centers, 0, 2*centers[0] - centers[1])
        b = np.append(centers, 2*centers[-1] - centers[-2])
        edges = (a + b) / 2
        return edges*unit

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
        if hasattr(self, '_bin_edges'):
            return self._bin_edges
        else:
            return self._edges_from_centers(self.value, self.unit)


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

    def __init__(self, data, *, uncertainty=None, spectral_axis=None,
                 spectral_dimension=0, mask=None, meta=None, **kwargs):

        # If the flux (data) argument is already a Spectrum (as it would
        # be for internal arithmetic operations), avoid setup entirely.
        if isinstance(data, Spectrum):
            return data

        # Ensure that the flux argument is an astropy quantity
        if data is not None:
            if not isinstance(data, u.Quantity):
                raise ValueError("Flux must be a `Quantity` object.")
            elif data.isscalar:
                data = u.Quantity([data])

        # Ensure that the unit information codified in the quantity object is
        # the One True Unit.
        kwargs.setdefault('unit', data.unit if isinstance(data, u.Quantity)
                          else kwargs.get('unit'))

        # If flux and spectral axis are both specified, check that their lengths
        # match or are off by one (implying the spectral axis stores bin edges)
        if data is not None and spectral_axis is not None:
            if spectral_axis.shape[0] == data.shape[spectral_dimension]:
                bin_specification = "centers"
            elif spectral_axis.shape[0] == data.shape[spectral_dimension]+1:
                bin_specification = "edges"
            else:
                raise ValueError(
                    "Spectral axis length ({}) must be the same size or one "
                    "greater (if specifying bin edges) than that of the spextral"
                    "axis ({})".format(spectral_axis.shape[0],
                                       data.shape[spectral_dimension]))

        # Attempt to parse the spectral axis. If none is given, try instead to
        # parse a given wcs. This is put into a GWCS object to
        # then be used behind-the-scenes for all operations.
        if spectral_axis is not None:
            # Ensure that the spectral axis is an astropy Quantity
            if not isinstance(spectral_axis, u.Quantity):
                raise ValueError("Spectral axis must be a `Quantity` or "
                                 "`SpectralAxis` object.")

            # If a spectral axis is provided as an astropy Quantity, convert it
            # to a SpectralAxis object.
            if not isinstance(spectral_axis, SpectralAxis):
                if spectral_axis.shape[0] == data.shape[spectral_dimension] + 1:
                    bin_specification = "edges"
                else:
                    bin_specification = "centers"
                self._spectral_axis = SpectralAxis(
                    spectral_axis,
                    bin_specification=bin_specification)

            wcs = gwcs_from_array(self._spectral_axis)

            super().__init__(
                data=data.value if isinstance(data, u.Quantity) else data,
                wcs=wcs, **kwargs)


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

    def __init__(self, data, norm=None, **kwargs):
        if data.unit != u.ct:
            data_norm_unit = (self.data.unit * norm.unit).decompose.unit
            if data_norm_unit != u.ct:
                raise ValueError('Data must be supplied in counts or the product of the norm and data has units of counts')
            data = data * norm
        self.norm = norm
        super().__init__(data, **kwargs)


class Response:
    r"""
    Model an instruments response
    """

    def __int__(self, callable, args):
        self.callable = callable


class SpectrometerResponseMatrix(Response):
    def __init__(self, matrix):
        self.matrix = matrix

    def forward(self, data):
        return np.matmul(data, self.matrix)


class Fitter:
    def __init__(self, data, model, response=None):
        self.data = data
        self.model = model
        if isinstance(data, CountSpectrum) and response is None:
            raise ValueError('When fitting a CountSpectrum a Response must be specified')
        self.response = response

    def fit(self):
        model_sig = signature(self.model)
        parameter_names = [param.name for name, param in model_sig.parameters.items()]
        parameter_units = [param.annotation for name, param in model_sig.parameters.items()]

        xunit = self.data._spectral_axis.unit
        munit = self.model(self.data._spectral_axis, 1 << parameter_units[1], 1 << parameter_units[2]).unit

        if self.response is None:

            if munit != self.data.unit:
                raise ValueError(f'Output from from model: {munit}, '
                                 f'does not match spectrum unit: {self.data.unit}.')

            def fmodel(x, slope, intercept):
                m = self.model(x << xunit, slope << parameter_units[1], intercept << parameter_units[2])
                return m.value

            popt, pcov = curve_fit(fmodel, self.data._spectral_axis, self.data.data)
        else:

            outunit = munit * self.response.matrix.unit

            if outunit != self.data.unit:
                raise ValueError(f'Output from from model and response unit: {outunit}, '
                                 f'do not match spectrum unit {self.data.unit}.')

            def rmodel(srm, model, x, slope, intercept):
                m = model(x << xunit, slope << parameter_units[1], intercept << parameter_units[2])
                d = srm.forward(m)
                return d.value

            fmodel = partial(rmodel, self.response, self.model)
            popt, pcov = curve_fit(fmodel, self.data._spectral_axis, self.data.data << self.data.unit)

        results = {name: value for name, value in zip(parameter_names[1:], popt)}
        results['covariance'] = pcov
        return SimpleNamespace(**results)
