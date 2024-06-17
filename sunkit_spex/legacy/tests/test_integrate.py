import numpy as np
from numpy.testing import assert_allclose

from sunkit_spex.integrate import fixed_quad, gauss_legendre


def test_scalar():
    n = 4
    def func(x): return x ** (2 * n - 1)
    expected = 1 / (2 * n)
    got_fq = fixed_quad(func, 0, 1, n=n)
    got_gl = gauss_legendre(func, 0, 1, n=n)
    # quadrature exact for this input
    assert_allclose(got_fq, expected, rtol=1e-12)
    assert_allclose(got_gl, expected, rtol=1e-12)


def test_vector_x():
    n = 4
    p = np.arange(1, 2 * n)
    def func(x): return x ** p[:, None]
    expected = 1 / (p + 1)
    got_fq = fixed_quad(func, 0, 1, n=n)
    got_gl = gauss_legendre(func, 0, 1, n=n)
    assert_allclose(got_fq, expected, rtol=1e-12)
    assert_allclose(got_gl, expected, rtol=1e-12)


def test_vector_lims():
    n = 10
    def func(x): return x**8
    a = [0, 1, 2]
    b = [1, 2, 3]
    expected = 1/9, (2**9 - 1)/9, (3**9 - 2**9)/9
    got_fq = fixed_quad(func, a, b, n=n)
    got_gl = gauss_legendre(func, a, b, n=n)
    assert_allclose(got_fq, expected, rtol=1e-12)
    assert_allclose(got_gl, expected, rtol=1e-12)
