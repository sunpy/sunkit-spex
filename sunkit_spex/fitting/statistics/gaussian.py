import numpy as np

def chi_squared(data_y, model_y):
    """ 
    The form to optimise while fitting. 
    
    * No error included here. *
    """
    return np.sum((data_y - model_y)**2)