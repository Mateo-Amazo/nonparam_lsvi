import numpy as np
import matplotlib.pyplot as plt

from variational.spline_estimation import get_bf_BSpline_decomposition
from variational.cv_regularization import get_bf_lambdas_cv

threshold = 1e-2
nb_iter = 20

def backfitting(g, multivariate_samples, dimension=None, order=4, Constraint="Concavity", a=None, b=None):
    dimension = multivariate_samples.shape[1]
    BSpline_list_last = [lambda x: 0.0 for _ in range(dimension)]
    BSpline_list = [lambda x: 0.0 for _ in range(dimension)]

    max_diff = np.inf
    j = 0

    while j < nb_iter and max_diff > threshold:

        lambdas,_ ,_ = get_bf_lambdas_cv(
            g,
            multivariate_samples=multivariate_samples,
            BSpline_list=BSpline_list,
            log_bounds=(-5, 5), 
            order=order, 
            Constraint=Constraint
        )

        for d in range(dimension):

            spline = get_bf_BSpline_decomposition(
                target_function=g,
                multivariate_samples=multivariate_samples, 
                BSpline_list=BSpline_list,
                d=d,
                order=order, 
                Constraint=Constraint, 
                lam=lambdas[d]
            )

            BSpline_list_last[d] = BSpline_list[d]
            BSpline_list[d] = spline

        max_diff = max(
            abs(BSpline_list[d](x[d]) - BSpline_list_last[d](x[d])) 
            for d in range(dimension) 
            for x in multivariate_samples
        )

        j += 1

    return BSpline_list
