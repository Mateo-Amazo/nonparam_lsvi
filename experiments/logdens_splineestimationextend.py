import numpy as np
import matplotlib.pyplot as plt
from splipy import Curve
import splipy

from variational.spline_estimation import get_BSpline_decomposition, get_beta_derivative

N = 10
order = 4
epsilon = 1e-10

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

u = np.linspace(-1, 1, N)
X = 10 * np.sign(u) * np.abs(u)**0.3

Beta, BSpline_Basis, _ = get_BSpline_decomposition(f, X, order=order, Constraint="Concavity")
knots1 = BSpline_Basis.knots
approx_curve1 = Curve(BSpline_Basis, Beta.reshape(-1, 1))

Beta_deriv = get_beta_derivative(Beta, BSpline_Basis.knots, N, order)
knots = BSpline_Basis.knots
BSpline_Basis_lower = splipy.BSplineBasis(order=order-1, knots=knots)
approx_curve1_deriv = Curve(BSpline_Basis_lower, Beta_deriv.reshape(-1, 1))


def B_Prime(x):
    if x>knots1[-1]:
        return approx_curve1_deriv.evaluate(knots1[-1]-epsilon)[0]
    elif x<knots1[0]:
        return approx_curve1_deriv.evaluate(knots1[0]+epsilon)[0]
    return approx_curve1_deriv.evaluate(x)[0]

def B(x):
    if x>knots1[-1]:
        return approx_curve1.evaluate(knots1[-1]-epsilon)[0] + approx_curve1_deriv.evaluate(knots1[-1]-epsilon)[0]*(x - knots1[-1])
    elif x<knots1[0]:
        return approx_curve1.evaluate(knots1[0]+epsilon)[0] + approx_curve1_deriv.evaluate(knots1[0]+epsilon)[0]*(x - knots1[0])
    return approx_curve1.evaluate(x)[0]


x_axis = np.linspace(knots1[0]-10, knots1[-1]+10, 1000)
y_axis1 = np.array([B(x) for x in x_axis])
y_true1 = f(x_axis)


for k in X:
    plt.axvline(k, color='gray', linestyle='--', alpha=1)

plt.figure(figsize=(8, 5))
plt.plot(x_axis, y_axis1)
plt.plot(x_axis, y_true1, "--")
plt.legend()

plt.savefig("graphs/density_spline_approximationsextend_" + str(N) + "points_" + str(order) + "order.png")