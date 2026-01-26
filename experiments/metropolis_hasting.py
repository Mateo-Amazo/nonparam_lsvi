import numpy as np
import matplotlib.pyplot as plt


def target_density(x, mu=0, sigma=1):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def metropolis_hastings(
    target,
    n_samples,
    proposal_std=1.0,
    x0=0.0
):
    samples = np.zeros(n_samples)
    x_current = x0

    accepted = 0

    for i in range(n_samples):

        x_proposal = np.random.normal(
            loc=x_current,
            scale=proposal_std
        )

        ratio = target(x_proposal) / target(x_current)

        alpha = min(1, ratio)

        u = np.random.rand()

        if u < alpha:
            x_current = x_proposal
            accepted += 1

        samples[i] = x_current

    acceptance_rate = accepted / n_samples

    return samples, acceptance_rate


n = 1000000
mu = 2
sigma = 1
proposal_std = 0.8


target = lambda x: target_density(x, mu, sigma)


samples, acc_rate = metropolis_hastings(
    target,
    n,
    proposal_std,
    x0=0
)

print(f"Taux d'acceptation : {acc_rate:.3f}")


x = np.linspace(-2, 6, 300)
true_density = (
    1 / (np.sqrt(2 * np.pi) * sigma)
    * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
)

plt.hist(samples, bins=50, density=True, alpha=0.6)
plt.plot(x, true_density, "r", lw=2)
plt.title("Metropolis-Hastings")
plt.show()
