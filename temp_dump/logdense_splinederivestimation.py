import numpy as np
import matplotlib.pyplot as plt
from splipy import Curve
from variational.spline_estimation import get_BSpline_decomposition

N = 10
order = 4

def f(x):
    main = -0.2 * (np.abs(x))
    wiggle = 0.5 * np.sin(5*x)
    dens = np.exp(main + wiggle)
    return np.log(dens)

X = np.linspace(-10, 10, N)

Beta, BSpline_Basis, _ = get_BSpline_decomposition(f, X, order=order, Constraint="Concavity")
approx_curve = Curve(BSpline_Basis, Beta.reshape(-1, 1))
knots = BSpline_Basis.knots

def B_Prime(x):
    deriv_matrix = BSpline_Basis.evaluate(x, d=1)[0]
    return (deriv_matrix @ Beta)[0]

def B(x):
    return approx_curve.evaluate(x)[0]

x_axis = np.linspace(knots[0], knots[-1], 1000)
y_axis = np.array([approx_curve.evaluate(x) for x in x_axis])
y_axis_derivative = np.array([B_Prime(x) for x in x_axis])
y_true = f(x_axis)

plt.figure(figsize=(8, 5))
plt.plot(x_axis, y_axis, label="Spline Approximation")
plt.plot(x_axis, y_axis_derivative, label="Spline Derivative")
plt.plot(x_axis, y_true, "--")
plt.legend()

plt.savefig("graphs/logdensity_splineapproxderiv_" + str(N) + "points_" + str(order) + "order_2.png")