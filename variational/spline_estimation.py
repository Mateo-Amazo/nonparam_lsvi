import numpy as np
import splipy
from sklearn.linear_model import LinearRegression


def aux_concavity_matrix(i,j):
    if i<j:
        return 0
    if j==1 and i>=1:
        return 1
    if j==2 and i>=j:
        return i-1
    if j>=3 and i>=j:
        return i-j+1

def get_BSpline_decomposition(f, X, order=4, Constraint = None):

    N = len(X)
    M = order

    sorted_X = np.sort(X)
    knots = np.concatenate([[sorted_X[0] for i in range(order)],sorted_X, [sorted_X[-1] for i in range(order)]])
    f_knots = np.array(f(knots))

    BSpline_Basis_M = splipy.BSplineBasis(order=order, knots=knots)
    BSpline_Basis_M1 = splipy.BSplineBasis(order=order-1, knots=knots)

    X_Tilde = np.array([BSpline_Basis_M.evaluate(knot)[0] for knot in knots])
    
    if Constraint == "Concavity":

        Sigma = np.array([[aux_concavity_matrix(i+1,j+1) for j in range(N+M)] for i in range(N+M)])
        X_Tilde = np.matmul(X_Tilde, Sigma)
        model = LinearRegression(positive=True, fit_intercept=False)
        model.fit(X_Tilde, f_knots)
        Beta = model.coef_
    else:
        model = LinearRegression(fit_intercept=False)
        model.fit(X_Tilde, f_knots)

        Beta = model.coef_

    return Beta, BSpline_Basis_M, BSpline_Basis_M1, knots
