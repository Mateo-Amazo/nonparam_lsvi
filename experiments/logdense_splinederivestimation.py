import numpy as np
import matplotlib.pyplot as plt
import splipy
from splipy import Curve
from variational.spline_estimation import get_BSpline_decomposition, get_beta_derivative

N = 10
order = 4

def f(x):
    main = -0.2 * (np.abs(x))
    wiggle = 0.5 * np.sin(5*x)
    dens = np.exp(main + wiggle)
    return np.log(dens)

X = np.linspace(-10, 10, N)

Beta, BSpline_Basis1, _ = get_BSpline_decomposition(f, X, order=order, Constraint="Concavity")
Beta_derivative = get_beta_derivative(Beta, BSpline_Basis1.knots, N, order)
knots = BSpline_Basis1.knots
BSpline_Basis_lower = splipy.BSplineBasis(order=order-1, knots=knots)

approx_curve = Curve(BSpline_Basis1, Beta.reshape(-1, 1))
approx_curve_derivative = Curve(BSpline_Basis_lower, Beta_derivative.reshape(-1, 1))


x_axis = np.linspace(knots[0], knots[-1], 1000)
y_axis = np.array([approx_curve.evaluate(x) for x in x_axis])
y_axis_derivative = np.array([approx_curve_derivative.evaluate(x) for x in x_axis])
y_true = f(x_axis)

plt.figure(figsize=(8, 5))
plt.plot(x_axis, y_axis, label="Spline Approximation")
plt.plot(x_axis, y_axis_derivative, label="Spline Derivative")
plt.plot(x_axis, y_true, "--")
plt.legend()

plt.savefig("graphs/density_splinederivapproximation_" + str(N) + "points_" + str(order) + "order_2.png")