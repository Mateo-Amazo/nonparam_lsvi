import numpy as np
import matplotlib.pyplot as plt
from splipy import Curve

from variational.spline_estimation import get_BSpline_decomposition
from experiments.problems import log_gaussian, log_shifted_gaussian, log_shifted_scaled_gaussian, log_mixture_of_gaussian, sin_sum

X = np.linspace(-5, 5, 500)
a = X[0]
b = X[-1]
order = 4

f = sin_sum

lambdas = np.logspace(-5, -2, 5, base=10)

for lam in lambdas:

    Beta, BSpline_Basis = get_BSpline_decomposition(
        f=lambda x: f(x),
        X=X,
        order=order,
        Constraint="Concavity",
        lam=lam,
        a=a,
        b=b
        )

    approx_curve = Curve(BSpline_Basis, Beta)

    Y = np.array([approx_curve.evaluate(x)[0] for x in X])
    f_axis = f(X)
    plt.plot(X, f_axis, label="f(x)")
    plt.plot(X, Y, label="Splines, λ="+str(lam), linestyle='--')
    plt.show()