import numpy as np
import matplotlib.pyplot as plt

from variational.cv_regularization import regularization_cst_cv
from experiments.problems import log_gaussian, log_shifted_gaussian, log_shifted_scaled_gaussian, log_mixture_of_gaussian

X = np.linspace(-5, 5, 500)
f = lambda x: log_shifted_scaled_gaussian(x, mean=1, scale=0.5)

lambdas, MSE_list = regularization_cst_cv(f=f, X=X, num=20, log_bounds=(-10,-3), order=4, Constraint="Concavity", a=X[0], b=X[-1])

plt.figure(figsize=(6, 4))
plt.plot(lambdas, MSE_list, marker='o')
plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$RMSE$')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()