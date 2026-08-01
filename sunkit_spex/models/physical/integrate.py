from functools import lru_cache

import numpy as np
from scipy.special import roots_legendre

__all__ = ["fixed_quad_batch", "gauss_legendre"]


@lru_cache
def _cached_roots_legendre(n):
    """
    Cache the Gauss-Legendre nodes and weights for a given order.

    Similar to the caching used internally in `~scipy.integrate.fixed_qaud` but using the public API
    """
    return roots_legendre(n)


def gauss_legendre(func, a, b, n=5, args=(), func_kwargs={}):
    """
    Compute a definite integral using fixed-order Gaussian quadrature.
    Integrate `func` from `a` to `b` using Gaussian quadrature of
    order `n`.

    Parameters
    ----------
    func : callable
        A Python function or method to integrate (must accept vector inputs).
        If integrating a vector-valued function, the returned array must have
        shape ``(..., len(x))``.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    n : int, optional
        Order of quadrature integration. Default is 5.
    args : tuple, optional
        Extra arguments to pass to function, if any.
    func_kwargs :
         Keyword arguments to the function `func` to be integrated.

    Returns
    -------
    integral : float
        Gaussian quadrature approximation to the integral

    Examples
    --------
    >>> from sunkit_spex.models.physical.integrate import gauss_legendre
    >>> f = lambda x: x**8
    >>> gauss_legendre(f,0.0,1.0,n=4)
    array([0.11108844])
    >>> gauss_legendre(f,0.0,1.0,n=5)
    array([0.11111111])
    >>> print(1/9.0)  # analytical result
    0.1111111111111111

    >>> gauss_legendre(f, [0, 1, 2], [1, 2, 3], n=5)
    array([1.11111111e-01, 5.67777778e+01, 2.13011111e+03])
    >>> 1/9, (2**9 - 1**9)/9, (3**9 - 2**9)/9 # analytical result
    (0.1111111111111111, 56.77777777777778, 2130.1111111111113)

    >>> gauss_legendre(np.cos,0.0,np.pi/2,n=4)
    array([0.99999998])
    >>> gauss_legendre(np.cos,0.0,np.pi/2,n=5)
    array([1.])
    >>> float(np.sin(np.pi/2)-np.sin(0))  # analytical result
    1.0
    """
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)

    # Nodes and weights of the standard n-point Gauss-Legendre rule on [-1, 1].
    standard_nodes, standard_weights = _cached_roots_legendre(n)

    # Map each interval's nodes/weights from [-1, 1] to [a, b] via the standard substitution
    # node = midpoint + half_width * standard_node, weight = half_width * standard_weight.
    midpoint = (0.5 * (a + b))[:, np.newaxis]
    half_width = (0.5 * (b - a))[:, np.newaxis]
    nodes = midpoint + half_width * standard_nodes[np.newaxis, :]
    weights = half_width * standard_weights[np.newaxis, :]

    return np.sum(weights * func(nodes, *args, **func_kwargs), axis=1)


def fixed_quad_batch(func, a, b, n=5, args=(), func_kwargs={}):
    """
    Compute a definite integral using fixed-order Gaussian quadrature.

    Integrate `func` from `a` to `b` using Gaussian quadrature of
    order `n`.

    This is a modified version of `scipy.integrate.fixed_qaud`

    Parameters
    ----------
    func : callable
        A Python function or method to integrate (must accept vector inputs).
        If integrating a vector-valued function, the returned array must have
        shape ``(..., len(x))``.
    a : float or `np.array`
        Lower limit of integration.
    b : float or `np.array`
        Upper limit of integration.
    n : int, optional
        Order of quadrature integration. Default is 5.
    args : tuple, optional
        Extra arguments to pass to function, if any.
    func_kwargs: dict, optional
        Keyword arguments to the function to be integrated

    Returns
    -------
    val : float
        Gaussian quadrature approximation to the integral

    Examples
    --------
    >>> from sunkit_spex.models.physical.integrate import fixed_quad_batch
    >>> f = lambda x: x**8
    >>> fixed_quad_batch(f,0.0,1.0,n=4)
    array(0.11108844)
    >>> fixed_quad_batch(f,0.0,1.0,n=5)
    array(0.11111111)
    >>> print(1/9.0)  # analytical result
    0.1111111111111111

    >>> fixed_quad_batch(f, [0, 1, 2], [1, 2, 3], n=5)
    array([1.11111111e-01, 5.67777778e+01, 2.13011111e+03])
    >>> 1/9, (2**9 - 1**9)/9, (3**9 - 2**9)/9 # analytical result
    (0.1111111111111111, 56.77777777777778, 2130.1111111111113)

    >>> fixed_quad_batch(np.cos,0.0,np.pi/2,n=4)
    array(0.99999998)
    >>> fixed_quad_batch(np.cos,0.0,np.pi/2,n=5)
     array(1.)
    >>> float(np.sin(np.pi/2)-np.sin(0))  # analytical result
    1.0

    """
    a = np.array(a)
    b = np.array(b)
    standard_nodes, standard_weights = _cached_roots_legendre(n)
    nodes = (b - a).reshape(-1, 1) * (standard_nodes + 1) / 2.0 + a.reshape(-1, 1)
    return np.squeeze(
        (b - a).reshape(1, -1) / 2.0 * np.sum(standard_weights * func(nodes, *args, **func_kwargs), axis=1)
    )
