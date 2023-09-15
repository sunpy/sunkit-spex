import numpy as np
from scipy.integrate._quadrature import _cached_roots_legendre

__all__ = ['gauss_legendre', 'fixed_quad']


def _legendre_roots(a, b, n=5):
    """
    Calculate the positions and weights for a Gauss-Legendre integration scheme with limits [a, b].

    Parameters
    ----------
    a : `numpy.array`
        Lower integration limits
    b : `numpy.array`
        Upper integration limits
    n : `int`, optional
        Order of quadrature integration. Default is 5.

    Returns
    -------
    `tuple` :
        (x, w) The positions and weights for the integration.

    """
    # Legendre points and weights over interval [-1, 1]
    x, w = _cached_roots_legendre(n)

    # map x, w to interval [a, b]
    xm = 0.5 * (b + a)
    xl = 0.5 * (b - a)
    xab = xm.reshape(a.shape[0], 1) + xl.reshape(a.shape[0], 1) * x.reshape(1, x.shape[0])
    wab = w.reshape(1, w.shape[0]) * xl.reshape(a.shape[0], 1)
    return xab, wab


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
    >>> from sunkit_spex.integrate  import gauss_legendre
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
    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
    1.0
    """
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    xi, wi, = _legendre_roots(a, b, n)
    integral = np.sum(wi * func(xi, *args, **func_kwargs), axis=1)

    return integral


def fixed_quad(func, a, b, n=5, args=(), func_kwargs={}):
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
    >>> from sunkit_spex.integrate  import fixed_quad
    >>> f = lambda x: x**8
    >>> fixed_quad(f,0.0,1.0,n=4)
    array(0.11108844)
    >>> fixed_quad(f,0.0,1.0,n=5)
    array(0.11111111)
    >>> print(1/9.0)  # analytical result
    0.1111111111111111

    >>> fixed_quad(f, [0, 1, 2], [1, 2, 3], n=5)
    array([1.11111111e-01, 5.67777778e+01, 2.13011111e+03])
    >>> 1/9, (2**9 - 1**9)/9, (3**9 - 2**9)/9 # analytical result
    (0.1111111111111111, 56.77777777777778, 2130.1111111111113)

    >>> fixed_quad(np.cos,0.0,np.pi/2,n=4)
    array(0.99999998)
    >>> fixed_quad(np.cos,0.0,np.pi/2,n=5)
     array(1.)
    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
    1.0

    """
    a = np.array(a)
    b = np.array(b)
    x, w = _cached_roots_legendre(n)
    x = np.real(x)
    if np.any(np.isinf(a)) or np.any(np.isinf(b)):
        raise ValueError("Gaussian quadrature is only available for "
                         "finite limits.")
    y = (b - a).reshape(-1, 1) * (x + 1) / 2.0 + a.reshape(-1, 1)
    return np.squeeze(
        (b - a).reshape(1, -1) / 2.0 * np.sum(w * func(y, *args, **func_kwargs), axis=1))
