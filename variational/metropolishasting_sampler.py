import numpy as np

def make_mh_sampler(log_density, proposal_std=1.0, x0=0.0, burn_in=0):
    def sampler(n_samples):
        samples = np.zeros(n_samples + burn_in)
        x_current = x0

        for i in range(n_samples + burn_in):
            x_proposal = np.random.normal(loc=x_current, scale=proposal_std)
            log_ratio = log_density(x_proposal) - log_density(x_current)
            alpha = min(1, np.exp(log_ratio))

            if np.random.rand() < alpha:
                x_current = x_proposal

            samples[i] = x_current

        return samples[burn_in:]

    return sampler

