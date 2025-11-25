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

X = np.linspace(-20, 20, N)
X = np.sort(X)

Beta, BSpline_Basis, _ = get_BSpline_decomposition(f, X, order=order, Constraint="Concavity")
approx_curve = Curve(BSpline_Basis, Beta.reshape(-1, 1))

Beta_deriv = get_beta_derivative(Beta, BSpline_Basis.knots, order)

def B(x):
    return approx_curve.evaluate(x)[0]

def B_Prime(x):
    deriv_matrix = BSpline_Basis.evaluate(x, d=1)
    return (deriv_matrix @ Beta_deriv.reshape(-1,1)).flatten()[0]

def is_concave(f, a, b, num_points=100):
    x = np.linspace(a, b, num_points)
    f_values = f(x)
    f_second = (f_values[:-2] - 2*f_values[1:-1] + f_values[2:]) / ((x[1]-x[0])**2)
    
    return np.all(f_second <= 0)

x_axis = X
y_axis1 = np.array([B(x) for x in x_axis])
y_axis1_deriv = np.array([B_Prime(x) for x in x_axis])
y_true1 = f(x_axis)

plt.figure(figsize=(8, 5))
plt.plot(x_axis, y_axis1)
plt.plot(x_axis, y_true1, "--")
plt.plot(x_axis, y_axis1_deriv)
plt.ylim(-15, 5)
plt.xlim(-20, 20)
plt.show()

print(is_concave(B, X[0], X[-1], num_points=100000))