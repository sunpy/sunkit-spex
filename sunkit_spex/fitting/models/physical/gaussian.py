from astropy.modeling import Fittable1DModel, Parameter
import numpy as np

class GaussianPhotonModel(Fittable1DModel):
    
    a = Parameter(default=1, min=0, description="Scalar for Gaussian.")
    b = Parameter(default=0, min=0, description="X-offset for Gaussian.")
    c = Parameter(default=1, description="Sigma for Gaussian.")

    @staticmethod
    def evaluate(x, a, b, c):
        """ Evaluate the Gaussian model at `x` with parameters `a`, `b`, and `c`. """
        return a * np.e**(-(x-b)**2/(2*c**2))