"""Module to store functions used to generate simulated data products."""

import numpy as np

__all__ = ["simulate_square_response_matrix"]

def simulate_square_response_matrix(size):
    """Generate a square matrix with off-diagonal terms. 
    
    Returns a product to mimic an instrument response matrix.

    Parameters
    ----------
    size : `int`
        The length of each side of the square response matrix.

    Returns
    -------
    `numpy.ndarray`
        The simulated 2D square response matrix.
    """
    # fake SRM
    fake_srm = np.identity(size)

    # add some off-diagonal terms
    for c, r in enumerate(fake_srm):
        # add some features into the fake SRM
        off_diag = np.random.rand(c)*0.005

        # add a diagonal feature
        _x = 50
        if c >= _x:
            off_diag[-_x] = np.random.rand(1)[0]

        # add a vertical feature in
        _y = 200
        __y = 30
        if c > _y+100:
            off_diag[_y-__y//2:_y+__y//2] = (np.arange(2*(__y//2))
                                             * np.random.rand(2*(__y//2))
                                             * 5e-4)

        # put these features in the fake_srm row and normalize
        r[:off_diag.size] = off_diag
        r /= np.sum(r)

    return fake_srm
