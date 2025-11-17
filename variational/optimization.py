import numpy as np
from scipy.optimize import minimize, fsolve

def find_mode(B):
    res = minimize(lambda x: -B(x), x0=0, method='Powell')
    return res.x[0]

def find_sz(B, rho):

    gs = lambda x: B(-np.abs(x)) + rho
    gz = lambda x: B(np.abs(x)) + rho

    s = np.abs(fsolve(gs, x0=0)[0])
    z = np.abs(fsolve(gz, x0=0)[0])

    return s, z
