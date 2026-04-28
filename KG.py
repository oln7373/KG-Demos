import os
import numpy as np
import scipy.stats
from scipy.special import logit, expit
from scipy.stats import norm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

M                 = int(os.getenv("N_FLAVORS", 12))
budget_list       = np.array([int(x) for x in os.getenv("BUDGETS", "25,50,100,150,200,500,1000").split(",")])
n_pilot           = int(os.getenv("NUM_PILOT_ROUNDS", 5))
n_trials          = int(os.getenv("N_TRIALS", 200))
prior_belief      = float(os.getenv("PRIOR_BELIEF", 0.75))
prior_uncertainty = float(os.getenv("PRIOR_UNCERTAINTY", 1.5))
noise_std         = float(os.getenv("NOISE_STD", 1.5))
gamma_shape       = float(os.getenv("GAMMA_SHAPE", 3.0))
gamma_rate        = float(os.getenv("GAMMA_RATE", 1.0))

# Read in flavor list
with open("flavors.txt", "r") as f:
    flavors = np.array([line.strip() for line in f.readlines()[:M]])


# --- Single-trial helper functions (kept for reference) ---

def taste(i, true_utilities, noise_std=noise_std):
    """Simulate tasting flavor i. Returns noisy observation in logit space."""
    return logit(true_utilities[i]) + np.random.randn() * noise_std

def Update(theta, Sigma, a, b, r, i):
    """Bayesian update equations for our belief state after a taste.
       Updates theta and Sigma in place; returns updated (a, b)."""
    innovation  = r - theta[i]
    h           = 1 + Sigma[i, i]
    theta[i]    = theta[i]    + Sigma[i, i] * innovation / h
    Sigma[i, i] = Sigma[i, i] - Sigma[i, i]**2 / h
    a = a + 0.5
    b = b + innovation**2 / (2 * h)
    return a, b

def compute_all_kg(theta, Sigma, a, b):
    """Compute KG for all M flavors at once. Returns array of length M."""
    diag        = np.diag(Sigma)
    h           = 1 + diag
    alpha       = diag / h
    sigma_tilde = alpha * np.sqrt(h * b / a)

    sorted_idx   = np.argsort(-theta)
    best_idx     = sorted_idx[0]
    second_idx   = sorted_idx[1]
    theta_others = np.full(len(theta), theta[best_idx])
    theta_others[best_idx] = theta[second_idx]

    z  = np.abs(theta - theta_others) / np.maximum(sigma_tilde, 1e-10)
    kg = sigma_tilde * (norm.pdf(z) - z * norm.sf(z))
    kg[sigma_tilde < 1e-10] = 0.0
    return kg

def select_flavor(theta, kg, B, B_0):
    """Use budget-adjusted score policy to select next flavor to taste."""
    lam    = B / B_0
    scores = theta + lam * kg
    return np.argmax(scores)


# --- Batched trial function ---

def run_trials(B, n_pilot, KG_selection=True):
    """Run n_trials experiments in parallel using numpy batching.

       Inputs:
       B            = Budget
       n_pilot      = Number of pilot rounds (KG policy only)
       KG_selection = True → KG policy, False → random policy

       Returns:
       Array of shape (n_trials,) with 1 = correct, 0 = incorrect
    """
    B_0    = float(B)
    budget = float(B)
    batch  = np.arange(n_trials)

    # 1. Initialize true utilities for all trials
    if M <= 20:
        # Evenly spaced quantiles — same ground truth for every trial (deterministic)
        true_utilities = np.tile(
            scipy.stats.beta.ppf(np.linspace(0.05, 0.95, M), 2, 5),
            (n_trials, 1)
        )
    else:
        # Independent random draw per trial
        true_utilities = np.random.beta(2, 5, size=(n_trials, M))

    # 2. Initialize belief state for all trials
    # sigma holds only the diagonal of Sigma — off-diagonals stay 0 throughout
    theta = np.full((n_trials, M), logit(prior_belief))
    sigma = np.full((n_trials, M), prior_uncertainty)
    a     = np.full(n_trials, gamma_shape)
    b     = np.full(n_trials, gamma_rate)

    def observe_and_update(j):
        """Taste flavor j[t] in each trial t and update belief state in place."""
        nonlocal a, b
        y          = logit(true_utilities[batch, j]) + np.random.randn(n_trials) * noise_std
        y          = np.clip(y, logit(0.001), logit(0.999))
        innovation = y - theta[batch, j]
        h          = 1 + sigma[batch, j]
        theta[batch, j] += sigma[batch, j] * innovation / h
        sigma[batch, j] -= sigma[batch, j]**2 / h
        a += 0.5
        b += innovation**2 / (2 * h)

    # 3. Pilot rounds (random selection, KG policy only)
    if KG_selection:
        for _ in range(min(n_pilot, int(budget))):
            observe_and_update(np.random.randint(0, M, size=n_trials))
            budget -= 1

    # 4. Main rounds
    while budget > 0:
        if KG_selection:
            h_all       = 1 + sigma                                          # (n_trials, M)
            sigma_tilde = (sigma / h_all) * np.sqrt(h_all * b[:, None] / a[:, None])

            sorted_idx   = np.argsort(-theta, axis=1)
            best_idx     = sorted_idx[:, 0]
            second_idx   = sorted_idx[:, 1]
            theta_others = np.broadcast_to(theta[batch, best_idx, None], (n_trials, M)).copy()
            theta_others[batch, best_idx] = theta[batch, second_idx]

            z  = np.abs(theta - theta_others) / np.maximum(sigma_tilde, 1e-10)
            kg = sigma_tilde * (norm.pdf(z) - z * norm.sf(z))
            kg[sigma_tilde < 1e-10] = 0.0

            j = np.argmax(theta + (budget / B_0) * kg, axis=1)
        else:
            j = np.random.randint(0, M, size=n_trials)

        observe_and_update(j)
        budget -= 1

    # 5. Results
    best_flavors = np.argmax(theta, axis=1)
    true_bests   = np.argmax(true_utilities, axis=1)
    return (best_flavors == true_bests).astype(int)


if __name__ == '__main__':

    kg_rates     = np.zeros(len(budget_list))
    random_rates = np.zeros(len(budget_list))

    for idx, B in enumerate(budget_list):
        kg_rates[idx]     = run_trials(B, n_pilot, KG_selection=True).mean()
        random_rates[idx] = run_trials(B, n_pilot, KG_selection=False).mean()

    # Plot
    x     = np.arange(len(budget_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, kg_rates,     width, label='KG',     color='steelblue')
    ax.bar(x + width / 2, random_rates, width, label='Random', color='salmon')

    ax.set_xlabel('Budget (B)')
    ax.set_ylabel('Success Rate')
    ax.set_title(f'KG vs. Random Selection — Success Rate by Budget (M={M}, {n_trials} trials)')
    ax.set_xticks(x)
    ax.set_xticklabels(budget_list)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.show()
