import numpy as np
from variational.nonparam_lsvi import nonparam_lsvi
from experiments.problems import log_gaussian, log_shifted_scaled_gaussian, log_mixture_of_gaussian
from variational.laplace import laplace_approximation

N = 50
order = 4
rho = 0.5
eps = 1

my_log_density = log_mixture_of_gaussian
_, mode, hess_inv_at_mode = laplace_approximation(log_density=my_log_density, init=0.)
my_initial_sampler = lambda n: mode + np.linalg.cholesky(hess_inv_at_mode) @ np.random.multivariate_normal(mean=mode,
                                                                                                           cov=np.eye(
                                                                                                               mode.shape[
                                                                                                                   0]),
                                                                                                           size=n).T
samples = nonparam_lsvi(my_log_density, mode, my_initial_sampler, order=order, eps=eps, N=N, rho=rho)
