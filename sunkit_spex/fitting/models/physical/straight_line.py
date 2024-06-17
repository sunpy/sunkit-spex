from astropy.modeling import Fittable1DModel, Parameter


class StraightLinePhotonModel(Fittable1DModel):

    m = Parameter(default=1, description="Gradiant of a straight line model.")
    c = Parameter(default=0, description="Y-intercept of a straight line model.")

    @staticmethod
    def evaluate(x, m, c):
        """ Evaluate the straight line model at `x` with parameters `m` and `c`. """
        return m*x+c
