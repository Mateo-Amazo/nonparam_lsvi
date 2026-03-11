import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline
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

def get_bf_BSpline_decomposition(target_function, multivariate_samples, BSpline_list=None, d=None, order=4, Constraint="Concavity", lam=1e-2):

    samples = np.sort(multivariate_samples[:,d])
    N = len(samples)

    avg_diff = np.mean(np.diff(samples))
    knots = np.concatenate([[samples[0]-avg_diff]*order, samples, [samples[-1]+avg_diff]*order])

    K = len(knots) - order

    X_Tilde = np.zeros((N, K))
    for k in range(K):
        c = np.zeros(K)
        c[k] = 1
        spline = BSpline(knots, c, order-1, extrapolate=False)
        X_Tilde[:, k] = spline(samples)

    Y = np.array([float(target_function(multivariate_samples[i])-sum(B(multivariate_samples[i,k]) for k,B in enumerate(BSpline_list) if k !=d )) for i in range(N)])

    D = np.eye(K, k=1) - np.eye(K)
    D = D[:-1]

    y_aug = np.concatenate([Y, np.zeros(D.shape[0])])

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

        lower_bounds = np.concatenate(([-np.inf], np.zeros(A.shape[1]-1)))
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

    spline_final = BSpline(knots, beta, order-1, extrapolate=True)

    return spline_final

def get_BSpline_decomposition(target_function, samples, order=4, Constraint="Concavity", lam=1e-2):

    samples = np.sort(samples)
    N = len(samples)

    avg_diff = np.mean(np.diff(samples))
    knots = np.concatenate([[samples[0]-avg_diff]*order, samples, [samples[-1]+avg_diff]*order])

    K = len(knots) - order

    X_Tilde = BSpline.design_matrix(samples, knots, order-1).toarray()

    Y = np.array([float(target_function(samples[i])) for i in range(N)])

    D = np.eye(K, k=1) - np.eye(K)
    D = D[:-1]

    y_aug = np.concatenate([Y, np.zeros(D.shape[0])])

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

        lower_bounds = np.concatenate(([-np.inf], np.zeros(A.shape[1]-1)))
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

    spline_final = BSpline(knots, beta, order-1, extrapolate=True)

    return spline_final