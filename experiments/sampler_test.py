import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from variational.log_concave_sampler import log_concave_sampler


def psi(x):
    return -0.5 * x**2

def dpsi(x):
    return -x

psi_dpsi = (psi, dpsi)

rho = 1.0
interval_for_finding_sz = (-20.0, 20.0)

sampler = log_concave_sampler(
    psi_dpsi=psi_dpsi,
    rho=rho,
    interval_for_finding_sz=interval_for_finding_sz
)

n_samples = 1000000
samples = sampler(n_samples)

kde = gaussian_kde(samples)

x_grid = np.linspace(
    np.min(samples) - 1,
    np.max(samples) + 1,
    1000
)

kde_vals = kde(x_grid)

psi_vals = np.exp(np.array([psi(x) for x in x_grid]))
psi_vals /= np.trapezoid(psi_vals, x_grid)

plt.figure(figsize=(8, 5))

plt.plot(x_grid, kde_vals, linewidth=2)
plt.plot(x_grid, psi_vals, "--", linewidth=2)

plt.hist(
    samples,
    bins=200,
    density=True,
    alpha=0.3
)

plt.savefig('experiments/graphs/log_concave_sampler.png')

