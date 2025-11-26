import numpy as np
import splipy
from scipy.optimize import lsq_linear

def aux_concavity_matrix(i,j):
    if i<j:
        return 0
    if j==1 and i>=1:
        return 1
    if j==2 and i>=j:
        return i-1
    if j>=3 and i>=j:
        return j-i-1

def get_BSpline_decomposition(f, X, order=4, Constraint="Concavity", a=None, b=None):

    N = len(X)
    M = order
    sorted_X = np.sort(X)

    if a is None or b is None:
        a = sorted_X[0]
        b = sorted_X[-1]

    knots = np.concatenate([[a for i in range(order)], sorted_X, [b for i in range(order)]])


    f_X = f(sorted_X)
    BSpline_Basis = splipy.BSplineBasis(order=order, knots=knots)
    BSpline_Basis_lower = splipy.BSplineBasis(order=order-1, knots=knots)

    X_Tilde = np.array([BSpline_Basis.evaluate(x)[0] for x in sorted_X])

    if Constraint == "Concavity":

        Sigma = np.array([[aux_concavity_matrix(i+1, j+1)
                           for j in range(N+M)] for i in range(N+M)])
        X_Tilde = np.matmul(X_Tilde, Sigma)



        lower_bounds = np.concatenate(([-np.inf], np.zeros(X_Tilde.shape[1] - 1)))
        upper_bounds = np.full(X_Tilde.shape[1], np.inf)

        res = lsq_linear(X_Tilde, f_X, bounds=(lower_bounds, upper_bounds))
        beta = np.matmul(Sigma, res.x)

    elif Constraint is None:
        beta = lsq_linear(X_Tilde, f_X)

    else:
        raise ValueError("Constraint not recognized")

    return beta.reshape(-1,1), BSpline_Basis, BSpline_Basis_lower