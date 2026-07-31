"""
Host utility functions that are useful for the metric module.
"""

import numpy as np

__all__ = ["weights_to_error", "error_to_weights"]

def weights_to_error(weights):
    """Convert weights to errors.
    
    error = 1/sqrt(weight)
    """
    return 1/np.sqrt(weights)

def error_to_weights(error):
    """Convert weights to errors.
    
    weight = 1/error**2
    """
    return 1/(error**2)