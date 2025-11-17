import numpy as np
import matplotlib.pyplot as plt
from splipy import Curve

from variational.data_generation import generate_data
from variational.spline_estimation import get_BSpline_decomposition, get_beta_derivative
from variational.optimization import find_mode

epsilon = 1e-8

def nonparam_lsvi(f, order=4, N=20, rho=0.5, eps=1e-1, Constraint="Concavity"):
    
    init_mode = find_mode(f)
    X = np.random.normal(loc=init_mode, scale=1.0, size=N)
    sorted_X = np.sort(X)

    a = sorted_X[0]
    b = sorted_X[-1]


    approxList = []

    while True:
        
        beta, BSpline_Basis, BSpline_Basis_lower = get_BSpline_decomposition(f, X=X, order=order, Constraint=Constraint, a=a, b=b)

        knots = BSpline_Basis.knots

        beta_deriv = get_beta_derivative(beta, knots, N=len(X), order=order)

        B_aux = Curve(BSpline_Basis, beta.reshape(-1, 1)).evaluate
        B_Prime_aux = Curve(BSpline_Basis_lower, beta_deriv.reshape(-1, 1)).evaluate

        def B(x):
            if x>knots[-1]:
                return B_aux(knots[-1])[0] + B_Prime_aux(knots[-1])[0]*(x - knots[-1])
            elif x<knots[0]:
                return B_aux(knots[0])[0] + B_Prime_aux(knots[0])[0]*(x - knots[0])
            return B_aux(x)[0]


        def B_Prime(x):
            if x>knots[-1]:
                print("B_Prime extrapolated at right")
                print(B_Prime_aux(knots[-1]-epsilon)[0])
                return B_Prime_aux(knots[-1]-epsilon)[0]
            elif x<knots[0]:
                print("B_Prime extrapolated at left")
                print(B_Prime_aux(knots[0]+epsilon)[0])
                return B_Prime_aux(knots[0]+epsilon)[0]
            return B_Prime_aux(x)[0]

        x_axis = np.linspace(knots[0], knots[-1], 500)
        y_B = np.array([B(x) for x in x_axis])
        y_Bprime = np.array([B_Prime(x) for x in x_axis])

        plt.plot(x_axis, y_B, label="B(x)")
        plt.plot(x_axis, y_Bprime, label="B'(x)")
        plt.show()

        approxList.append(B)

        if np.abs(beta[0]) < eps and np.abs(beta[-1]) < eps:
            return approxList


        X = generate_data(B, B_Prime, N, rho)

        Delta = np.mean(np.diff(X))

        a = np.min([a, X[0]-Delta])
        b = np.max([b, X[-1]+Delta]) 