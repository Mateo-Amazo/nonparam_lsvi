import numpy as np
import matplotlib.pyplot as plt
from splipy import Curve

from variational.data_generation import generate_data
from variational.spline_estimation import get_BSpline_decomposition
from variational.optimization import find_mode

epsilon = 0.5

def nonparam_lsvi(f, order=4, N=20, rho=0.5, eps=1e-1, Constraint="Concavity"):
    
    init_mode = find_mode(f)
    X = np.random.normal(loc=init_mode, scale=1.0, size=N)
    sorted_X = np.sort(X)

    a = sorted_X[0]
    b = sorted_X[-1]


    approxList = []

    while True:
        
        Beta, BSpline_Basis, _ = get_BSpline_decomposition(f=f, X=X, order=order, Constraint=Constraint)
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

        x_axis = np.linspace(knots[0]-5, knots[-1]+5, 700)
        y_f = np.array([f(x) for x in x_axis])
        y_B = np.array([B(x) for x in x_axis])
        y_Bprime = np.array([B_Prime(x) for x in x_axis])

        for k in knots:
            plt.axvline(k, color='gray', linestyle='--', alpha=0.8)

        plt.plot(x_axis, y_f, label="f(x)")
        plt.plot(x_axis, y_B, label="B(x)")
        plt.plot(x_axis, y_Bprime, label="B'(x)")
        plt.show()

        approxList.append(B)

        if np.abs(Beta[0]) < eps and np.abs(Beta[-1]) < eps:
            return approxList


        X = generate_data(B, B_Prime, N, rho)

        Delta = np.mean(np.diff(X))
        a = np.min([a, X[0]-Delta])
        b = np.max([b, X[-1]+Delta]) 