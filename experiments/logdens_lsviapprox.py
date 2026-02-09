from variational.nonparam_lsvi import nonparam_lsvi
from experiments.problems import log_gaussian, log_shifted_scaled_gaussian, log_mixture_of_gaussian


N = 500
order = 4
rho = 0.5
lam = 1e-0

my_log_density = log_mixture_of_gaussian

samples = nonparam_lsvi(my_log_density, dimension=2, order=order, N=N, rho=rho)
