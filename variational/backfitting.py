import numpy as np
import matplotlib.pyplot as plt

from variational.spline_estimation import get_BSpline_decomposition
from variational.cv_regularization import get_lambda_cv

threshold = 1e-3

def backfitting(g, multivariate_samples, dimension=None, order=4, Constraint="Concavity", a=None, b=None):
    dimension = multivariate_samples.shape[1]
    BSpline_list_last = [lambda x: 0.0 for _ in range(dimension)]
    BSpline_list = [lambda x: 0.0 for _ in range(dimension)]

    while True:

        lambdas = get_lambda_cv(
            g,
            multivariate_samples=multivariate_samples,
            BSpline_list=BSpline_list,
            log_bounds=(-5, 5), 
            order=order, 
            Constraint=Constraint, 
            a=a, 
            b=b
        )

        for d in range(dimension):

            spline = get_BSpline_decomposition(
                target_function=g,
                multivariate_samples=multivariate_samples, 
                BSpline_list=BSpline_list,
                d=d,
                order=order, 
                Constraint=Constraint, 
                lam=lambdas[d],
                a=a,
                b=b
            )

            BSpline_list_last[d] = BSpline_list[d]
            BSpline_list[d] = lambda x: spline(x)

        max_diff = max(
            abs(BSpline_list[d](x[d]) - BSpline_list_last[d](x[d])) 
            for d in range(dimension) 
            for x in multivariate_samples
        )
        if max_diff < threshold:
            break

    return BSpline_list
