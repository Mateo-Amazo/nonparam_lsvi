import numpy as np
from scipy.optimize import minimize, fsolve, bisect


def find_mode(f, warm_start=0, bounds=None):
    res = minimize(lambda x: -f(x), x0=warm_start, method='L-BFGS-B', bounds=bounds)
    return res.x[0]


def find_sz(f, rho, interval_for_finding_sz):
    left, right = interval_for_finding_sz
    # need to check that
    try:
        assert left <= 0
        assert right >= 0
        assert f(right) * f(0) <= 0
        assert f(left) * f(0) <= 0
    except:
        return failback_find_sz(f, rho)

    g = lambda x: f(x) + rho
    try:
        s = -bisect(g, a=left, b=0)
        z = bisect(g, a=0, b=right)
    except:
        return failback_find_sz(f, rho)
    # implement a failback
    return s, z


def failback_find_sz(f, rho):
    gs = lambda x: f(-np.abs(x)) + rho
    gz = lambda x: f(np.abs(x)) + rho

    s = np.abs(fsolve(gs, x0=0)[0])
    z = np.abs(fsolve(gz, x0=0)[0])
    return s, z
