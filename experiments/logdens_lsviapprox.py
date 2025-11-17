import numpy as np
import matplotlib.pyplot as plt

from variational.nonparam_lsvi import nonparam_lsvi


N = 30
order = 4
rho = 0
eps = 1

def f(x):
    p1, mu1, sigma1 = 0.5, -3, 1
    p2, mu2, sigma2 = 0.5, 3, 1
    dens = (
        p1 / (np.sqrt(2 * np.pi) * sigma1) * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
        + p2 / (np.sqrt(2 * np.pi) * sigma2) * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    )
    return np.log(dens)

approxList = nonparam_lsvi(f, order=order, eps=eps, N=N, rho=rho)
print(len(approxList))