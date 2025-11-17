import numpy as np
import matplotlib.pyplot as plt
from splipy import Curve
from variational.spline_estimation import get_BSpline_decomposition

N = 10
order = 4

def f(x):
    comps = [
        (0.15, -6, 0.7),
        (0.20, -2, 0.4),
        (0.30,  0, 0.6),
        (0.20,  3, 0.5),
        (0.15,  6, 1.0),
    ]
    dens = sum(
        p / (np.sqrt(2*np.pi)*s) * np.exp(-0.5*((x-m)/s)**2)
        for p, m, s in comps
    )
    return np.log(dens)

def f2(x):
    main = -0.2 * (np.abs(x))
    wiggle = 0.5 * np.sin(5*x)
    dens = np.exp(main + wiggle)
    return np.log(dens)

X = np.linspace(-10, 10, N)

Beta1, BSpline_Basis1, _ = get_BSpline_decomposition(f, X, order=order, Constraint="Concavity")
knots1 = BSpline_Basis1.knots
approx_curve1 = Curve(BSpline_Basis1, Beta1.reshape(-1, 1))

x_axis = np.linspace(knots1[0], knots1[-1], 1000)
y_axis1 = np.array([approx_curve1.evaluate(x) for x in x_axis])
y_true1 = f(x_axis)

Beta2, BSpline_Basis2, _ = get_BSpline_decomposition(f2, X, order=order, Constraint="Concavity")
knots2 = BSpline_Basis2.knots
approx_curve2 = Curve(BSpline_Basis2, Beta2.reshape(-1, 1))

x_axis2 = np.linspace(knots2[0], knots2[-1], 1000)
y_axis2 = np.array([approx_curve2.evaluate(x) for x in x_axis2])
y_true2 = f2(x_axis2)

plt.figure(figsize=(8, 5))
plt.plot(x_axis, y_axis1)
plt.plot(x_axis, y_true1, "--")
plt.plot(x_axis2, y_axis2)
plt.plot(x_axis2, y_true2, "--")
plt.legend()

plt.savefig("graphs/density_spline_approximations_" + str(N) + "points_" + str(order) + "order.png")