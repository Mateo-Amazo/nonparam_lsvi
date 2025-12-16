import numpy as np
import matplotlib.pyplot as plt
from splipy import Curve

from variational.spline_estimation import get_BSpline_decomposition
from experiments.problems import log_gaussian, log_shifted_gaussian, log_shifted_scaled_gaussian, log_mixture_of_gaussian, piecewise_wavy, sin_sum

X = np.linspace(-5, 5, 150)
f = piecewise_wavy
knots = np.concatenate(([-5 for _ in range(4)], np.linspace(-5, 5, 50), [5 for _ in range(4)]))

lambdas = np.logspace(-1, 2, 4, base=10)

for lam in lambdas:

    Beta, BSpline_Basis = get_BSpline_decomposition(
        f=lambda x: f(x),
        X=X,
        order=4,
        Constraint="Concavity",
        lam=lam,
        knots=knots
        )

    approx_curve = Curve(BSpline_Basis, Beta)

    Y = np.array([approx_curve.evaluate(x)[0] for x in X])

    exp = int(np.log10(lam))
    plt.plot(X, Y, linestyle='--', label=r"$\lambda$"+rf"=$10^{{{exp}}}$")

f_axis = f(X)
plt.plot(X, f_axis, color='black', label='True function')
plt.legend()
plt.savefig(f'experiments/graphs/regularization_influence.png')