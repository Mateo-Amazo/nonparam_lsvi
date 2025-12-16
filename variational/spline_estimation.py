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

def get_BSpline_decomposition(f, X, order=4, Constraint="Concavity", a=None, b=None, lam = 1e-2):

    N = len(X)
    M = order
    sorted_X = np.sort(X)

    if a is None or b is None:
        a = sorted_X[0]
        b = sorted_X[-1]

    knots = np.concatenate([[a for i in range(order)], sorted_X, [b for i in range(order)]])


    f_X = f(knots)
    BSpline_Basis = splipy.BSplineBasis(order=order, knots=knots)
    BSpline_Basis_lower = splipy.BSplineBasis(order=order-1, knots=knots)

    X_Tilde = np.array([BSpline_Basis.evaluate(x)[0] for x in knots])


    if Constraint == "Concavity":

        Sigma = np.fromfunction(
            np.vectorize(lambda i, j: aux_concavity_matrix(i+1, j+1)),
            (N+M, N+M),
            dtype=int
        )

        A = X_Tilde @ Sigma

        D = np.eye(N+M, k=1) - np.eye(N+M)
        D = D[:-1]


        A_aug = np.vstack([
            A,
            np.sqrt(lam) * (D @ Sigma)
        ])

        y_aug = np.concatenate([
            f_X,
            np.zeros(D.shape[0])
        ])

        lower_bounds = np.concatenate(([-np.inf], np.zeros(A.shape[1] - 1)))
        upper_bounds = np.full(A.shape[1], np.inf)

        res = lsq_linear(A_aug, y_aug, bounds=(lower_bounds, upper_bounds))

        beta = Sigma @ res.x


    elif Constraint is None:
        beta = lsq_linear(X_Tilde, f_X)

    else:
        raise ValueError("Constraint not recognized")

    return beta.reshape(-1,1), BSpline_Basis