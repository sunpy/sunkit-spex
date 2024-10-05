import numpy as np

from astropy.nddata.nduncertainty import (
    IncompatibleUncertaintiesException,
    NDUncertainty,
    VarianceUncertainty,
    _VariancePropagationMixin,
)

__all__ = ["PoissonUncertainty"]


class PoissonUncertainty(_VariancePropagationMixin, NDUncertainty):
    """Poissonian uncertainty assuming first order error propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division`` with other instances
    of `Poisson`. The class can handle if the uncertainty has a
    unit that differs from (but is convertible to) the parents `NDData` unit.
    The unit of the resulting uncertainty will have the same unit as the
    resulting data. Also support for correlation is possible but requires the
    correlation as input. It cannot handle correlation determination itself.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`

    Examples
    --------
    `Poisson` should always be associated with an `NDData`-like
    instance, either by creating it during initialization::

        >>> from astropy.nddata import NDData
        >>> from sunkit_spex.spectrum.uncertainty import PoissonUncertainty
        >>> ndd = NDData([1,2,3], unit='m',
        ...              uncertainty=PoissonUncertainty([0.1, 0.1, 0.1]))
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        PoissonUncertainty([0.1, 0.1, 0.1])

    or by setting it manually on the `NDData` instance::

        >>> ndd.uncertainty = PoissonUncertainty([0.2], unit='m', copy=True)
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        PoissonUncertainty([0.2])

    the uncertainty ``array`` can also be set directly::

        >>> ndd.uncertainty.array = 2
        >>> ndd.uncertainty
        PoissonUncertainty(2)

    .. note::
        The unit will not be displayed.
    """

    @property
    def supports_correlated(self):
        """`True` : `StdDevUncertainty` allows to propagate correlated \
                    uncertainties.

        ``correlation`` must be given, this class does not implement computing
        it by itself.
        """
        return True

    @property
    def uncertainty_type(self):
        """``"poisson"`` : `PoissonUncertainty` implements Poisson uncertainty."""
        return "poisson"

    def _convert_uncertainty(self, other_uncert):
        if isinstance(other_uncert, PoissonUncertainty):
            return other_uncert
        else:
            raise IncompatibleUncertaintiesException

    def _propagate_add(self, other_uncert, result_data, correlation):
        return super()._propagate_add_sub(
            other_uncert,
            result_data,
            correlation,
            subtract=False,
            to_variance=np.square,
            from_variance=np.sqrt,
        )

    def _propagate_subtract(self, other_uncert, result_data, correlation):
        return super()._propagate_add_sub(
            other_uncert,
            result_data,
            correlation,
            subtract=True,
            to_variance=np.square,
            from_variance=np.sqrt,
        )

    def _propagate_multiply(self, other_uncert, result_data, correlation):
        return super()._propagate_multiply_divide(
            other_uncert,
            result_data,
            correlation,
            divide=False,
            to_variance=np.square,
            from_variance=np.sqrt,
        )

    def _propagate_divide(self, other_uncert, result_data, correlation):
        return super()._propagate_multiply_divide(
            other_uncert,
            result_data,
            correlation,
            divide=True,
            to_variance=np.square,
            from_variance=np.sqrt,
        )

    def _propagate_collapse(self, numpy_operation, axis):
        # defer to _VariancePropagationMixin
        return super()._propagate_collapse(numpy_operation, axis=axis)

    def _data_unit_to_uncertainty_unit(self, value):
        return value

    def _convert_to_variance(self):
        new_array = None if self.array is None else self.array**2
        new_unit = None if self.unit is None else self.unit**2
        return VarianceUncertainty(new_array, unit=new_unit)

    @classmethod
    def _convert_from_variance(cls, var_uncert):
        new_array = None if var_uncert.array is None else var_uncert.array ** (1 / 2)
        new_unit = None if var_uncert.unit is None else var_uncert.unit ** (1 / 2)
        return cls(new_array, unit=new_unit)
