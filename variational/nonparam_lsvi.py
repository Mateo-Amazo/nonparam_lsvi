import numpy as np
import matplotlib.pyplot as plt
from splipy import Curve

from variational.log_concave_sampler import spline_log_concave_sampler
from variational.spline_estimation import get_BSpline_decomposition
from variational.optimization import find_mode

epsilon = 0.5


def nonparam_lsvi(f, initial_mode, initial_sampler, order=4, N=20, rho=0.5, eps=1e-1, Constraint="Concavity",
                  max_iter=10):
    samples_across_iteration = np.zeros((N, max_iter))
    my_mode = initial_mode
    my_samples = np.sort(initial_sampler(N))[0]

    a = my_samples[0]
    b = my_samples[-1]

    j = 0

    while True and j < max_iter:
        Beta, BSpline_Basis, _ = get_BSpline_decomposition(f=f, X=my_samples, order=order, Constraint=Constraint)
        approx_curve = Curve(BSpline_Basis, Beta.reshape(-1, 1))
        knots = BSpline_Basis.knots

        deriv_matrix = BSpline_Basis.evaluate(knots[order], d=1)[0]
        Deriv_right = (deriv_matrix @ Beta.reshape(-1, 1))[0]

        deriv_matrix = BSpline_Basis.evaluate(knots[order + N], d=1)[0]
        Deriv_left = (deriv_matrix @ Beta.reshape(-1, 1))[0]

        def B_Prime(x):
            if x > knots[order + N]:
                return Deriv_right
            elif x < knots[order]:
                return Deriv_left
            deriv_matrix = BSpline_Basis.evaluate(x, d=1)[0]
            return (deriv_matrix @ Beta.reshape(-1, 1))[0]

        def B(x):
            if x > knots[order + N]:
                return B_Prime(knots[order + N]) * (x - knots[-1]) + approx_curve.evaluate(knots[order + N])[0]
            elif x < knots[order]:
                return B_Prime(knots[order]) * (x - knots[0]) + approx_curve.evaluate(knots[order])[0]
            return approx_curve.evaluate(x)[0]

        x_axis = np.linspace(a, b, 100)
        y_f = np.array([f(x) for x in x_axis])
        y_B = np.array([B(x) for x in x_axis])

        # for k in knots:
        #    plt.axvline(k, color='gray', linestyle='--', alpha=0.8)

        plt.plot(x_axis, np.exp(y_f), label="f(x)")
        plt.plot(x_axis, np.exp(y_B), label="B(x)")
        plt.legend()
        plt.savefig(f'experiments/graphs/{j}.png')
        plt.close()

        # if np.abs(Beta[0]) < eps and np.abs(Beta[-1]) < eps:
        #    return approxList
        my_new_mode = find_mode(B, warm_start=my_mode, bounds={(1.5 * a, 1.5 * b)})
        _, my_sampler = spline_log_concave_sampler(B, B_Prime, mode=my_mode,
                                                   interval_for_finding_sz=(2 * a - my_new_mode, 2 * b - my_new_mode),
                                                   rho=rho)
        my_samples = my_sampler(N)
        my_samples = np.sort(my_samples)
        samples_across_iteration[:, j] = my_samples

        Delta = np.mean(np.diff(my_samples))
        a = np.min([a, my_samples[0] - Delta])
        b = np.max([b, my_samples[-1] + Delta])
        my_mode = my_new_mode
        j += 1
    return my_samples
