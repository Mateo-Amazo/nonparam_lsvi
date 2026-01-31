import numpy as np
import matplotlib.pyplot as plt

from variational.spline_estimation import get_BSpline_decomposition
from variational.cv_regularization import get_lambda_cv

threshold = 1e-3

def backfitting(g, my_samples, order=4, Constraint="Concavity", a=None, b=None):
    D = my_samples.shape[0]
    BSpline_list_last = [lambda x: 0.0 for _ in range(D)]
    BSpline_list = [lambda x: 0.0 for _ in range(D)]

    while True:
        for d in range(D):

            lam = get_lambda_cv(
                g,
                my_samples=my_samples,
                BSpline_list=BSpline_list,
                d=d,
                log_bounds=(-5, 5), 
                order=order, 
                Constraint=Constraint, 
                a=a, 
                b=b
            )

            spline = get_BSpline_decomposition(
                g,
                X=my_samples, 
                BSpline_list=BSpline_list,
                d=d,
                order=order, 
                Constraint=Constraint, 
                lam=lam,
                a=a,
                b=b
            )

            BSpline_list_last[d] = BSpline_list[d]
            BSpline_list[d] = lambda x: spline(x)

        max_diff = max(
            abs(BSpline_list[d](x) - BSpline_list_last[d](x)) 
            for d in range(D) 
            for x in my_samples
        )
        if max_diff < threshold:
            break

    return BSpline_list
