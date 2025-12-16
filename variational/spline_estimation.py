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

def get_BSpline_decomposition(f, X, order=4, Constraint="Concavity", knots=None, a=None, b=None, lam = 1e-2):

    sorted_X = np.sort(X)

    if a is None or b is None:
        a = sorted_X[0]
        b = sorted_X[-1]

    if knots is None:

        knots = np.concatenate([[a for i in range(order)], sorted_X, [b for i in range(order)]])

    K = len(knots) - order

    BSpline_Basis = splipy.BSplineBasis(order=order, knots=knots)

    f_X = f(X)
    X_Tilde = BSpline_Basis.evaluate(X)

    D = np.eye(K, k=1) - np.eye(K)
    D = D[:-1]
    y_aug = np.concatenate([
        f_X,
        np.zeros(D.shape[0])
    ])


    if Constraint == "Concavity":

        Sigma = np.fromfunction(
            np.vectorize(lambda i, j: aux_concavity_matrix(i+1, j+1)),
            (K, K),
            dtype=int
        )

        A = X_Tilde @ Sigma


        A_aug = np.vstack([
            A,
            np.sqrt(lam) * (D @ Sigma)
        ])


        lower_bounds = np.concatenate(([-np.inf], np.zeros(A.shape[1] - 1)))
        upper_bounds = np.full(A.shape[1], np.inf)

        res = lsq_linear(A_aug, y_aug, bounds=(lower_bounds, upper_bounds))

        beta = Sigma @ res.x


    elif Constraint is None:

        X_Tilde_aug = np.vstack([
            X_Tilde,
            np.sqrt(lam) * D 
            ])

        beta = lsq_linear(X_Tilde_aug, y_aug).x

    else:
        raise ValueError("Constraint not recognized")

    return beta.reshape(-1,1), BSpline_Basis