import numpy as np
import matplotlib.pyplot as plt
from splipy import Curve
import splipy

from variational.spline_estimation import get_BSpline_decomposition, get_beta_derivative

N = 10
order = 4
epsilon = 1e-10

def f(x):
    main = -0.2 * (np.abs(x))
    wiggle = 0.5 * np.sin(5*x)
    dens = np.exp(main + wiggle)
    return np.log(dens)

X = np.random.normal(0, 3, size=N)
X = np.sort(X)

Beta, BSpline_Basis, _ = get_BSpline_decomposition(f, X, order=order, Constraint="Concavity")
knots = BSpline_Basis.knots
approx_curve = Curve(BSpline_Basis, Beta.reshape(-1, 1))

Beta_deriv = get_beta_derivative(Beta, BSpline_Basis.knots, N, order)
knots = BSpline_Basis.knots
BSpline_Basis_lower = splipy.BSplineBasis(order=order-1, knots=knots)
approx_curve_deriv = Curve(BSpline_Basis_lower, Beta_deriv.reshape(-1, 1))


def B_Prime(x):
    if x>knots[-1]:
        return approx_curve_deriv.evaluate(knots[-1]-epsilon)[0]
    elif x<knots[0]:
        return approx_curve_deriv.evaluate(knots[0]+epsilon)[0]
    return approx_curve_deriv.evaluate(x)[0]

def B(x):
    if x>knots[-1]:
        return approx_curve.evaluate(knots[-1]-epsilon)[0] + approx_curve_deriv.evaluate(knots[-1]-epsilon)[0]*(x - knots[-1])
    elif x<knots[0]:
        return approx_curve.evaluate(knots[0]+epsilon)[0] + approx_curve_deriv.evaluate(knots[0]+epsilon)[0]*(x - knots[0])
    return approx_curve.evaluate(x)[0]


x_axis = np.linspace(-20, 20, 1000)
y_axis1 = np.array([B(x) for x in x_axis])
y_axis1_deriv = np.array([B_Prime(x) for x in x_axis])
y_true1 = f(x_axis)

plt.figure(figsize=(8, 5))

for k in X:
    plt.axvline(k, color='gray', linestyle='--', alpha=1)

plt.plot(x_axis, y_axis1)
plt.plot(x_axis, y_true1, "--")
plt.plot(x_axis, y_axis1_deriv)
plt.ylim(-15, 5)
plt.xlim(-20, 20)
plt.legend()

plt.savefig("graphs/density_spline_approximations_pointsinfluence_" + str(N) + "points_" + str(order) + "order_2.png")