import numpy as np
import matplotlib.pyplot as plt

import splipy
from splipy import Curve

from variational.spline_estimation import get_BSpline_decomposition, get_beta_derivative
from variational.data_generation import generate_data

N = 10
order = 4
epsilon = 1e-10
rho = 0

def f(x):
    main = -0.2 * (np.abs(x))
    wiggle = 0.5 * np.sin(5*x)
    dens = np.exp(main + wiggle)
    return np.log(dens)

X = np.linspace(-10, 10, N)

Beta, BSpline_Basis, _ = get_BSpline_decomposition(f, X, order=order, Constraint="Concavity")
Beta_deriv = get_beta_derivative(Beta, BSpline_Basis.knots, N, order)
knots = BSpline_Basis.knots
BSpline_Basis_lower = splipy.BSplineBasis(order=order-1, knots=knots)

B_aux = Curve(BSpline_Basis, Beta.reshape(-1, 1)).evaluate
B_Prime_aux = Curve(BSpline_Basis_lower, Beta_deriv.reshape(-1, 1)).evaluate

def B(x):
    if x>knots[-1]:
        return B_aux(knots[-1]-epsilon)[0] + B_Prime_aux(knots[-1]-epsilon)[0]*(x - knots[-1])
    elif x<knots[0]:
        return B_aux(knots[0]+epsilon)[0] + B_Prime_aux(knots[0]+epsilon)[0]*(x - knots[0])
    return B_aux(x)[0]


def B_Prime(x):
    if x>knots[-1]:
        return B_Prime_aux(knots[-1]-epsilon)[0]
    elif x<knots[0]:
        return B_Prime_aux(knots[0]+epsilon)[0]
    return B_Prime_aux(x)[0]


X2 = generate_data(B, B_Prime, N, rho)

x_axis = np.linspace(X2[0], X2[-1], 100)
y_f = np.array([f(x) for x in x_axis])
y_B = np.array([B(x) for x in x_axis])
y_Bprime = np.array([B_Prime(x) for x in x_axis])

for k in X2:
    plt.axvline(k, color='gray', linestyle='--', alpha=0.8)

plt.plot(x_axis, y_f, label="f(x)")
plt.plot(x_axis, y_B, label="B(x)")
plt.plot(x_axis, y_Bprime, label="B'(x)")
plt.show()