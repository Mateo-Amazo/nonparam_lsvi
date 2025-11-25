import numpy as np
import matplotlib.pyplot as plt
from splipy import Curve
import splipy

from variational.spline_estimation import get_BSpline_decomposition, get_beta_derivative

N = 100
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


def B(x):
    return approx_curve.evaluate(x)[0]

def is_concave(f, a, b, num_points=100):
    x = np.linspace(a, b, num_points)
    f_values = f(x)
    f_second = (f_values[:-2] - 2*f_values[1:-1] + f_values[2:]) / ((x[1]-x[0])**2)
    
    return np.all(f_second <= 0)

print(is_concave(B, X[0], X[-1], num_points=100000))