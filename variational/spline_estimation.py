import numpy as np
import splipy
from sklearn.linear_model import LinearRegression
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

def get_BSpline_decomposition(f, X, order=4, Constraint=None):
    N = len(X)
    M = order
    sorted_X = np.sort(X)
    knots = np.concatenate([[sorted_X[0] for i in range(order)], sorted_X, [sorted_X[-1] for i in range(order)]])


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
        Beta = np.matmul(Sigma, res.x)

    else:
        model = LinearRegression(fit_intercept=False)
        model.fit(X_Tilde, f_X)
        Beta = model.coef_
    return Beta, BSpline_Basis_M, BSpline_Basis_M1, knots
