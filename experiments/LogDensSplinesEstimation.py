import numpy as np
import matplotlib.pyplot as plt
from splipy import Curve
from variational.spline_estimation import get_BSpline_decomposition

def f(x):
    return -0.5 * np.log(2 * np.pi) - 0.5 * x**2

N = 10
order = 4
X = np.random.normal(loc=0, scale=10, size=N)

Beta, BSpline_Basis, _, knots = get_BSpline_decomposition(f, X, order=order, Constraint="Concavity")
approx_curve = Curve(BSpline_Basis, Beta.reshape(-1, 1))

x_axis = np.linspace(knots[0], knots[-1], 1000)
y_axis = np.array([approx_curve.evaluate(x) for x in x_axis])
y_axis2 = f(x_axis)

plt.plot(x_axis, y_axis, label='Spline curve')
plt.plot(x_axis, y_axis2, label='True log-density', linestyle='dashed')
plt.legend()
plt.show()