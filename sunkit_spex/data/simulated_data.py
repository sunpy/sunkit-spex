"""
Module to store functions used to generate simulated data products.
"""

import numpy as np

__all__ = ["simulate_square_response_matrix"]


def simulate_square_response_matrix(size, random_seed=10):
    """Generate a square matrix with off-diagonal terms.

    Returns a product to mimic an instrument response matrix.

    Parameters
    ----------
    size : `int`
        The length of each side of the square response matrix.

    random_seed : `int`, optional
        The seed input for the random number generator. This will accept any value input accepted by `numpy.random.default_rng`.

    Returns
    -------
    `numpy.ndarray`
        The simulated 2D square response matrix.
    """
    np_rand = np.random.default_rng(seed=random_seed)

    # fake SRM
    fake_srm = np.identity(size)

    # add some off-diagonal terms
    for c, r in enumerate(fake_srm):
        # add some features into the fake SRM
        off_diag = np_rand.random(c) * 0.005

        # add a diagonal feature
        _x = 50
        if c >= _x:
            off_diag[-_x] = np_rand.random(1)[0]

        # add a vertical feature in
        _y = 200
        __y = 30
        if c > _y + 100:
            off_diag[_y - __y // 2 : _y + __y // 2] = np.arange(2 * (__y // 2)) * np_rand.random(2 * (__y // 2)) * 5e-4

        # put these features in the fake_srm row and normalize
        r[: off_diag.size] = off_diag
        r /= np.sum(r)

    return fake_srm
