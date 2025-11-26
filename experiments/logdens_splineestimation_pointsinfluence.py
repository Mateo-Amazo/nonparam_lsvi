import numpy as np
import matplotlib.pyplot as plt
from splipy import Curve

from variational.spline_estimation import get_BSpline_decomposition

N = 100
order = 4
rho = 0

def f(x):
    main = -0.2 * (np.abs(x))
    wiggle = 0.5 * np.sin(5*x)
    dens = np.exp(main + wiggle)
    return np.log(dens)

X = np.random.normal(0, 2, N)
X = np.sort(X)

Beta, BSpline_Basis, _ = get_BSpline_decomposition(f, X, order=order, Constraint="Concavity")
approx_curve = Curve(BSpline_Basis, Beta.reshape(-1, 1))
knots = BSpline_Basis.knots


def B_Prime(x):
    deriv_matrix = BSpline_Basis.evaluate(x, d=1)[0]
    return (deriv_matrix @ Beta)[0]

def B(x):
    matrix = BSpline_Basis.evaluate(x)[0]
    return (matrix @ Beta)[0]


x_axis = np.linspace(X[0], X[-1], 700)
y_axis1 = np.array([B(x) for x in x_axis])
y_axis1_deriv = np.array([B_Prime(x) for x in x_axis])
y_true1 = f(x_axis)

plt.figure(figsize=(8, 5))

for k in X:
    plt.axvline(k, color='gray', linestyle='--', alpha=1)

plt.plot(x_axis, y_axis1, label="B(x)")
plt.plot(x_axis, y_true1, "--", label="f(x)")
plt.plot(x_axis, y_axis1_deriv, label="B'(x)")
plt.ylim(-15, 5)
plt.xlim(-20, 20)
plt.legend()

plt.savefig("graphs/approximations_pointsinfluence_" + str(N) + "points_" + str(order) + "order_2.png")