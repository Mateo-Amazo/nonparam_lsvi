import numpy as np
import matplotlib.pyplot as plt

from variational.log_concave_sampler import spline_log_concave_sampler
from variational.metropolishasting_sampler import mh_sampler
from variational.optimization import find_mode
from variational.backfitting import backfitting
from variational.laplace import laplace_approximation

def nonparam_lsvi(log_density, D, order=4, N=20, rho=0.5, step_size=0.1, Constraint="Concavity",
                  max_iter=10):

    _, mode, hess_inv_at_mode = laplace_approximation(log_density=log_density, init=[0. for _ in range(D)])
    sampler = lambda n: mode + np.linalg.cholesky(hess_inv_at_mode) @ np.random.multivariate_normal(mean=mode,
                                                                                                           cov=np.eye(
                                                                                                               mode.shape[
                                                                                                                   0]),
                                                                                                           size=n).T

    samples_across_iteration = np.array([])
    multivariate_samples = sampler(N)

    a = multivariate_samples.min(axis=0)
    b = multivariate_samples.max(axis=0)

    j = 0

    BSpline_list = [lambda x: 0.0 for _ in range(D)]

    while j < max_iter:

        g = lambda x: (1 - step_size) * log_density(x) + step_size * (sum(BSpline_list[d](x[d]) for d in range(D)))

        BSpline_list = backfitting(g, multivariate_samples, order=order, Constraint=Constraint, a=a, b=b)
        x_axis = np.linspace(a, b, 100)
        y_f = np.array([f(x) for x in x_axis])
        y_B = np.array([B(x) for x in x_axis])

        # for k in knots:
        #    plt.axvline(k, color='gray', linestyle='--', alpha=0.8)

        plt.plot(x_axis, np.exp(y_f), label="f(x)")
        plt.plot(x_axis, np.exp(y_B), label="B(x)")
        plt.legend()
        plt.savefig(f'experiments/graphs/lsvi_steps/{j}.png')
        plt.close()

        for d in range(D):
            B = BSpline_list[d]
            B_Prime = B.derivative()
            my_new_mode = find_mode(B, warm_start=my_mode, bounds={(1.5 * a, 1.5 * b)})
            _, my_sampler = spline_log_concave_sampler(B, B_Prime, mode=my_mode,
                                                    interval_for_finding_sz=(2 * a - my_new_mode, 2 * b - my_new_mode),
                                                    rho=rho)
            my_samples = my_sampler(N)
            multivariate_samples[:, d] = my_samples

        samples_across_iteration = np.append(samples_across_iteration, multivariate_samples)

        Delta = np.array([np.mean(np.diff(multivariate_samples[:, d])) for d in range(D)])
        a = np.min([a, my_samples[0] - Delta])
        b = np.max([b, my_samples[-1] + Delta])
        j += 1

    return samples_across_iteration
