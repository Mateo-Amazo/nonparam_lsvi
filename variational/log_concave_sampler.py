import numpy as np
from variational.optimization import find_mode, find_sz


def log_concave_sampler(psi_dpsi, rho, interval_for_finding_sz):
    r"""
    This is an implementation of the generic algorithm proposed in
    Devroye, Random variate generation for the generalized inverse Gaussian distribution, 2014,
    for sampling from univariate log-concave distributions p\propto e^{\psi}
    """
    psi, dpsi = psi_dpsi

    def Chi(x, rho, s_tilde, z_tilde, dzeta, xi):
        if -s_tilde <= x <= z_tilde:
            return 1
        if x >= z_tilde:
            return np.exp(-rho - dzeta * (x - z_tilde))
        else:
            return np.exp(-rho + xi * (x + s_tilde))

    s, z = find_sz(psi, rho, interval_for_finding_sz)
    dzeta, xi = -dpsi(z), dpsi(-s)
    p, r = 1 / xi, 1 / dzeta
    z_tilde, s_tilde = z - r * rho, s - p * rho
    q = z_tilde + s_tilde

    def _sampler(n_samples):
        my_samples = np.zeros(n_samples, )
        for _ in range(n_samples):
            boolean = True

            while boolean:

                U = np.random.uniform(0, 1)
                V = np.random.uniform(0, 1)
                W = np.random.uniform(0, 1)

                if U * (q + p + r) < q:
                    x = -s_tilde + q * V

                elif U * (q + p + r) <= (q + r):
                    x = z_tilde - r * np.log(V)

                else:
                    x = -s_tilde + p * np.log(V)

                if W * Chi(x, rho, s_tilde, z_tilde, dzeta, xi) <= np.exp(psi(x)):
                    boolean = False
            my_samples[_] = x
        return my_samples

    return _sampler


def spline_log_concave_sampler(my_spline, my_dspline, interval_for_finding_sz, mode=0., rho=0.5, warm_start=0.):
    if mode is None:
        mode = find_mode(my_spline, warm_start)

    def psi(x):
        return my_spline(mode + x) - my_spline(mode)

    def dpsi(x):
        return my_dspline(mode + x)

    my_log_concave_sampler = log_concave_sampler((psi, dpsi), rho, interval_for_finding_sz)
    return mode, my_log_concave_sampler
