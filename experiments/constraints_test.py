import numpy as np
import matplotlib.pyplot as plt

from variational.spline_estimation import get_BSpline_decomposition
from experiments.problems import log_gaussian, log_shifted_scaled_gaussian, log_mixture_of_gaussian

X = np.linspace(-5, 5, 10)
f = lambda x: log_shifted_scaled_gaussian(x, mean=1, scale=0.5)

spline = get_BSpline_decomposition(samples=X, target_function=f, order=4, Constraint="Concavity")

x_axis = np.linspace(-10, 10, 100)
y_f = np.array([f(x) for x in x_axis])
y_B = np.array([spline(x) for x in x_axis])
plt.plot(x_axis, y_f, label="f(x)")
plt.plot(x_axis, y_B, label="B(x)")
plt.legend()
plt.savefig(f'experiments/graphs/constraint_test.png')