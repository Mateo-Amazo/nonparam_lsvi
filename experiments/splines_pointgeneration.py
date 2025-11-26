import numpy as np
import matplotlib.pyplot as plt
from splipy import Curve

from variational.spline_estimation import get_BSpline_decomposition
from variational.data_generation import generate_data

N = 10
N2 = 100
order = 4
rho = 1/2

def f(x):
    main = -0.2 * (np.abs(x))
    wiggle = 0.5 * np.sin(5*x)
    dens = np.exp(main + wiggle)
    return np.log(dens)

X = np.linspace(-10,10,N)

Beta, BSpline_Basis, _ = get_BSpline_decomposition(f, X, order=order, Constraint="Concavity")
approx_curve = Curve(BSpline_Basis, Beta)
knots = BSpline_Basis.knots

def B_Prime(x):
    if x>knots[order+N]:
        return (BSpline_Basis.evaluate(knots[order+N], d=1)[0] @ Beta)[0]
    elif x<knots[order]:
        return (BSpline_Basis.evaluate(knots[order], d=1)[0] @ Beta)[0]
    return (BSpline_Basis.evaluate(x, d=1)[0] @ Beta)[0]

def B(x):
    if x>knots[order+N]:
        return B_Prime(knots[order+N])*(x - knots[order+N]) + approx_curve.evaluate(knots[order+N])[0]
    elif x<knots[order]:
        return B_Prime(knots[order])*(x - knots[order]) + approx_curve.evaluate(knots[order])[0]
    return approx_curve.evaluate(x)[0]

X2 = generate_data(B, B_Prime, N2, rho)

x_axis = np.linspace(-20, 20, 500)
y_f = np.array([f(x) for x in x_axis])
y_B = np.array([B(x) for x in x_axis])
y_Bprime = np.array([B_Prime(x) for x in x_axis])


for k in X2:
    plt.axvline(k, color='gray', linestyle='--', alpha=0.8)

plt.plot(x_axis, y_f, label="f(x)")
plt.plot(x_axis, y_B, label="B(x)")
plt.plot(x_axis, y_Bprime, label="B'(x)")
plt.legend()

plt.savefig("graphs/splineapprox_sampling_" + str(N) + "points_" + str(order) + "order.png")