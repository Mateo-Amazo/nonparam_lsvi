import numpy as np


def log_gaussian(x):
    x = np.asarray(x)
    return -0.5 * np.sum(x**2, axis=-1)


def log_shifted_gaussian(x, mean=np.array([1.0, 1.0])):
    x = np.asarray(x)
    mean = np.asarray(mean)
    return -0.5 * np.sum((x - mean)**2, axis=-1)


def log_shifted_scaled_gaussian(x, mean=np.array([1.0, 1.0]), scale=0.5):
    x = np.asarray(x)
    mean = np.asarray(mean)
    return -0.5 * np.sum((x - mean)**2, axis=-1) / scale**2


def log_mixture_of_gaussian(
    x,
    means=np.array([[-1.0, -1.0], [2.0, 2.0]]),
    var=np.array([1.0, 1.5])
):
    x = np.asarray(x)

    weights = np.ones(means.shape[0]) / means.shape[0]

    diff = x[..., None, :] - means
    quad = np.sum(diff**2, axis=-1)

    return np.log(
        np.sum(
            weights * np.exp(-0.5 * quad / var),
            axis=-1
        )
    )


def sin_sum(x):
    x = np.asarray(x)
    return np.array(0.7*np.sin(2*x) + 0.3*np.sin(7*x))


def piecewise_wavy(x):
    x = np.asarray(x)

    left = 0.5 * x + 2
    right = -0.5 * x + 2

    envelope = np.minimum(left, right)

    return np.array(envelope + 0.3 * np.sin(8 * x))
