import numpy as np
import splipy
from scipy.optimize import lsq_linear


def f(x):
    return -0.5 * np.log(2 * np.pi) - 0.5 * x**2

def aux_concavity_matrix(i,j):
    if i<j:
        return 0
    if j==1 and i>=1:
        return 1
    if j==2 and i>=j:
        return i-1
    if j>=3 and i>=j:
        return j-i-1

def get_BSpline_decomposition(f, X, order=4, Constraint=None, At=None, Bt=None):

    N = len(X)
    M = order
    sorted_X = np.sort(X)

    if At is None or Bt is None:
        At = sorted_X[0]
        Bt = sorted_X[-1]

    knots = np.concatenate([[At for i in range(order)], sorted_X, [Bt for i in range(order)]])


    f_X = f(sorted_X)
    BSpline_Basis_M = splipy.BSplineBasis(order=order, knots=knots)
    BSpline_Basis_M1 = splipy.BSplineBasis(order=order-1, knots=knots)

    X_Tilde = np.array([BSpline_Basis_M.evaluate(x)[0] for x in sorted_X])

    if Constraint == "Concavity":
        Sigma = np.array([[aux_concavity_matrix(i+1, j+1)
                           for j in range(N+M)] for i in range(N+M)])
        X_Tilde = np.matmul(X_Tilde, Sigma)

        lower_bounds = np.concatenate(([-np.inf], np.zeros(X_Tilde.shape[1] - 1)))
        upper_bounds = np.full(X_Tilde.shape[1], np.inf)

        res = lsq_linear(X_Tilde, f_X, bounds=(lower_bounds, upper_bounds))
        beta = np.matmul(Sigma, res.x)

    else:
        beta = lsq_linear(X_Tilde, f_X)
    return beta, BSpline_Basis_M, BSpline_Basis_M1, knots
