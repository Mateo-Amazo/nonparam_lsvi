import numpy as np
import splipy
from splipy import Curve
import matplotlib.pyplot as plt

seed = 0
N = 50
order = 4

def f(x):
    return -0.5 * np.log(2 * np.pi) - 0.5 * x**2

X = np.random.normal(loc = 0, scale=1, size=N)

sorted_X = np.sort(X)
knots = np.concatenate([[sorted_X[0] for i in range(order)],sorted_X, [sorted_X[-1] for i in range(order)]])
f_x = np.array(f(sorted_X))

BSpline_Basis = splipy.BSplineBasis(order=order, knots=knots)
B = np.array([BSpline_Basis.evaluate(x)[0] for x in sorted_X])

print(B)
print(f_x)

c, *_ = np.linalg.lstsq(B, f_x, rcond=None)

print(c)

approx_curve = Curve(BSpline_Basis, c.reshape(-1, 1))

x_axis = np.linspace(-0.5, 0.5, 100)
y_axis = np.array([approx_curve.evaluate(x) for x in x_axis])

plt.plot(x_axis, y_axis, label='Spline curve')
plt.legend()
plt.show()