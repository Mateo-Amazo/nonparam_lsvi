import numpy as np
from variational.optimization import find_mode, find_sz

def Chi(x, rho, s_tilde, z_tilde, dzeta, ksi):
    return (1 if (-s_tilde <= x <= z_tilde) else 0) + (np.exp(-rho-dzeta*(x-z_tilde)) if x>=z_tilde else 0) + (np.exp(-rho+ksi*(x+s_tilde)) if x<=-s_tilde else 0)

def Phi(x, B, mode):
    return B(mode+x)-B(mode)

def Phi_prime(x, B_Prime, mode):
    return B_Prime(mode+x)

def generate_data(B, B_Prime, N, rho):

    mode = find_mode(B)
    s, z = find_sz(B, rho)

    dzeta, ksi = -Phi_prime(z, B_Prime, mode), Phi_prime(-s, B_Prime, mode)

    p,r = 1/ksi, 1/dzeta
    z_tilde, s_tilde = z - r*rho, s - p*rho
    q = z_tilde + s_tilde

    samples = []

    for k in range(N):

        x = None
        boolean = True

        while boolean:

            U = np.random.uniform(0,1)
            V = np.random.uniform(0,1)
            W = np.random.uniform(0,1)


            if U*(q + p + r) <= q:
                x = -s_tilde + q*V

            elif U*(q + p + r) <= (q + p):
                x = z_tilde - r*np.log(V)

            else:
                x = -s_tilde + p*np.log(V)
            
            if W*Chi(x, rho, s_tilde, z_tilde, dzeta, ksi) <= np.exp(Phi(x, B, mode)):
                boolean = False

        samples.append(x + mode)

    return np.sort(samples)

