import numpy as np
import matplotlib.pyplot as plt

from variational.spline_estimation import get_BSpline_decomposition
from experiments.problems import (
    log_gaussian,
    log_shifted_gaussian,
    log_shifted_scaled_gaussian,
    log_mixture_of_gaussian,
    piecewise_wavy,
    sin_sum
)


X = np.linspace(-5, 5, 150)
f = piecewise_wavy

knots = np.concatenate((
    [-5 for _ in range(4)],
    np.linspace(-5, 5, 50),
    [5 for _ in range(4)]
))

lambdas = np.logspace(-1, 2, 4, base=10)


for lam in lambdas:

    spline = get_BSpline_decomposition(
        target_function=lambda x: f(x),
        samples=X,
        order=4,
        Constraint=None,
        lam=lam,
        knots=knots
    )

    Y = spline(X)

    exp = int(np.log10(lam))
    plt.plot(
        X,
        Y,
        linestyle="--",
        label=r"$\lambda$" + rf"=$10^{{{exp}}}$"
    )


f_axis = f(X)

plt.plot(X, f_axis, color="black", label="True function")
plt.legend()
plt.savefig("experiments/graphs/regularization_influence.png")
plt.show()
