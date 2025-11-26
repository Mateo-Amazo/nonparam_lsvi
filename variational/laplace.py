from typing import Callable
import scipy
import numpy as np

def bfgs_wrapper(loss: Callable, init: np.ndarray):
    res = scipy.optimize.minimize(loss, init, method="BFGS")
    return res.x, res.hess_inv


def laplace_approximation(log_density: Callable, init: np.ndarray, optimization_method=bfgs_wrapper):
    """
    Compute the Laplace approximation of a density.
    """

    def loss(theta):
        return -log_density(theta)

    x, hess_inv = optimization_method(loss, init)
    return -log_density(x), x, hess_inv
