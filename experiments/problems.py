import numpy as np


def log_gaussian(x):
    return -0.5 * x ** 2


def log_shifted_gaussian(x):
    return -0.5 * (x - 1) ** 2


def log_shifted_scaled_gaussian(x, mean=1, scale=0.5):
    return -0.5 * (x - mean) ** 2 / scale**2


def log_mixture_of_gaussian(x, means=np.array([-1, 2]), var=np.array([1, 1.5])):
    """
    Assuming the weights
    """
    weights = np.ones(means.shape[0])/means.shape[0]
    return np.log(np.sum(np.exp(-0.5*(x[...,None]-means)**2/var * weights), axis=-1))

def sin_sum(x):

    return 0.7*np.sin(2*x) + 0.3*np.sin(7*x)
