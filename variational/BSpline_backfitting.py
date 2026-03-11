import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear

def BSpline_decomposition(target_function):
    pass


def BSpline_backfitting(target_function, covariables):

    d = covariables.shape[1]

    responses = target_function(covariables)

    while True:
        for j in range(d):
            pass
