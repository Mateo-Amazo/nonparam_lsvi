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
knots = BSpline_Basis.knots

deriv_matrix = BSpline_Basis.evaluate(knots[order], d=1)[0]
Deriv_right = (deriv_matrix @ Beta.reshape(-1,1))[0]

deriv_matrix = BSpline_Basis.evaluate(knots[order+N], d=1)[0]
Deriv_left = (deriv_matrix @ Beta.reshape(-1,1))[0]

def B_Prime(x):
    if x>knots[order+N]:
        return Deriv_right
    elif x<knots[order]:
        return Deriv_left
    deriv_matrix = BSpline_Basis.evaluate(x, d=1)[0]
    return (deriv_matrix @ Beta.reshape(-1,1))[0]

def B(x):
    if x>knots[order+N]:
        return B_Prime(knots[order+N])*(x - knots[-1]) + approx_curve.evaluate(knots[order+N])[0]
    elif x<knots[order]:
        return B_Prime(knots[order])*(x - knots[0]) + approx_curve.evaluate(knots[order])[0]
    return approx_curve.evaluate(x)[0]

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