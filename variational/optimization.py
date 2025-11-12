import numpy as np
from scipy.optimize import minimize, fsolve

def find_mode(B):
    res = minimize(lambda x: -B(x), x0=0, method='Powell')
    return res

def find_sz(B, rho):
    
    def gs(x):
        return B(-x)+rho
    s = fsolve(gs, x0=0)

    def gz(x):
        return B(x)+rho
    z = fsolve(gz, x0=0)

    return s, z