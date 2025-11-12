import numpy as np

from splipy import Curve

from variational.data_generation import generate_data
from variational.spline_estimation import get_BSpline_decomposition
from variational.optimization import find_mode

eps = 0.1

def non_param_lsvi(f, N=20, Constraint=None):
    
    init_mode = find_mode(f)
    X = np.random.normal(loc=init_mode.x, scale=1.0, size=N) #TBC : Change scale
    sorted_X = np.sort(X)

    At = sorted_X[0]
    Bt = sorted_X[-1]

    boolean = True

    while boolean:
        
        beta, BSpline_Basis_M, BSpline_Basis_M1, knots = get_BSpline_decomposition(f, X=X, order=4, Constraint=Constraint, At=At, Bt=Bt)[1]
        
        beta_deriv = None #TBC

        B = Curve(BSpline_Basis_M, beta.reshape(-1, 1)).evaluate
        B_Prime = Curve(BSpline_Basis_M1, beta_deriv.reshape(-1, 1)).evaluate

        X = generate_data(B= B, B_Prime=B_Prime, N=N, eps=eps)


        if beta[0] < eps and beta[-1] < eps:
            boolean = False


    return Curve(BSpline_Basis_M, beta.reshape(-1, 1)).evaluate