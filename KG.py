import os
import numpy as np
import scipy.stats
from scipy.special import logit
from dotenv import load_dotenv

load_dotenv()

M = int(os.getenv("N_FLAVORS", 12))  # Number of ice cream flavors
B = float(os.getenv("BUDGET", 12))  # Budget (each taste costs 1 unit)
n_pilot = int(os.getenv("NUM_PILOT_ROUNDS", 5))
prior_belief = float(os.getenv("PRIOR_BELIEF", 0.75))
prior_uncertainty = float(os.getenv("PRIOR_UNCERTAINTY", 1.5))
noise_std = float(os.getenv("NOISE_STD", 1.5))

# Functions

def taste(i, true_utilities, noise_std=noise_std):
    """Simulate tasting flavor i. Returns noisy observation in logit space."""
    y = logit(true_utilities[i]) + np.random.randn() * noise_std
    return y

def Update(theta, Sigma, a, b, r, i):
    """Bayesian update equations for our belief state after a taste.
       Updates theta and Sigma in place; returns updated (a, b)."""
    innovation = r - theta[i]
    h = 1 + Sigma[i, i]

    theta[i] = theta[i] + Sigma[i, i] * innovation / h
    Sigma[i, i] = Sigma[i, i] - Sigma[i, i]**2 / h
    a = a + 0.5
    b = b + innovation**2 / (2 * h)

    return a, b



# 1. Randomly initialize latent utilities
if M <= 20:
    # Evenly spaced quantiles — guarantees good spread for small flavor lists (deterministic)
    percentiles = np.linspace(0.05, 0.95, M)
    true_utilities = scipy.stats.beta.ppf(percentiles, 2, 5)
else:
    # Random draws — law of large numbers gives you a good spread for large flavor lists
    true_utilities = np.random.beta(2, 5, size=M)

with open("flavors.txt", "r") as f:
    flavors = np.array(
        [line.strip() for line in f.readlines()[:M]]
    )

# Print flavor + latent utility
print("\nFlavor latent utilities:\n")
for flavor, utility in zip(flavors, true_utilities):
    print(f"{flavor:20s} {utility:.4f}")

# 2. Initialize belief state for each flavor
theta = np.ones(M) * logit(prior_belief) # Everything operates in logit space
Sigma = np.eye(M) * prior_uncertainty
a = float(os.getenv("GAMMA_SHAPE", 3.0)) # Noise has same precision regardless of which flavor we taste
b = float(os.getenv("GAMMA_RATE", 1.0))


# 3. Pilot Rounds
for i in range(n_pilot):
    j = np.random.randint(0, M) # Select flavor randomly
    y = taste(j, true_utilities) # Taste selected flavor
    y = np.clip(y, logit(0.001), logit(0.999))  # clip in logit space
    a,b = Update(theta, Sigma, a, b, y, j)
    B = B - 1 # Each taste costs 1 unit of budget