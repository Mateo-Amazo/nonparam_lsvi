import numpy as np
import matplotlib.pyplot as plt

from variational.log_concave_sampler import spline_log_concave_sampler
from variational.metropolishasting_sampler import mh_sampler
from variational.optimization import find_mode
from variational.backfitting import backfitting
from variational.laplace import laplace_approximation

def nonparam_lsvi(log_density, dimension, order=4, N=20, rho=0.5, step_size=0.1, Constraint="Concavity",
                  max_iter=10):

    _, mode, hess_inv_at_mode = laplace_approximation(log_density=log_density, init=[0. for _ in range(dimension)])
    sampler = lambda n: np.random.multivariate_normal(mean=mode.flatten(), cov=hess_inv_at_mode, size=n)

    samples_across_iteration = np.array([])
    multivariate_samples = sampler(N)

    j = 0

    BSpline_list = [lambda x: 0.0 for _ in range(dimension)]
    mode_list = np.array([0.0 for _ in range(dimension)])

    while j < max_iter:


        def g(X):
            X = np.atleast_2d(X)
            return (1 - step_size) * log_density(X) + step_size * sum(BSpline_list[d](X[:, d]) for d in range(dimension))


        BSpline_list = backfitting(g, multivariate_samples, dimension=dimension, order=order, Constraint=Constraint)

        for d in range(dimension):

            B = BSpline_list[d]
            knots = B.t
            a, b = knots[0], knots[-1]
            mode_list[d] = find_mode(B, warm_start=mode_list[d], bounds={(1.5 * a, 1.5 * b)})
            _, my_sampler = spline_log_concave_sampler(B, B.derivative(), mode=mode_list[d],
                                                    interval_for_finding_sz=(2 * a - mode_list[d], 2 * b - mode_list[d]),
                                                    rho=rho)
            my_samples = my_sampler(N)
            multivariate_samples[:, d] = my_samples

        samples_across_iteration = np.append(samples_across_iteration, multivariate_samples)

        B_0 = BSpline_list[0]
        a, b = B_0.t[0], B_0.t[-1]

        x_axis = np.linspace(a, b, 100)
        #y_f = np.array([log_density(x) for x in x_axis])
        y_B = np.array([B_0(x) for x in x_axis])
        #plt.plot(x_axis, np.exp(y_f), label="f(x)")
        plt.plot(x_axis, np.exp(y_B), label="B_0(x)")
        plt.legend()
        plt.savefig(f'experiments/graphs/lsvi_steps/{j}.png')
        plt.close()

        j += 1

    return samples_across_iteration
