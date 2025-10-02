"""
Bayesian adaptive experiment utilities extracted from the notebook
`toy_bayesian_model.ipynb` for unit testing and reuse.

Functions:
- alpha_theory(B_total, Gamma)
- likelihood(alpha_meas, B_unknown, B_bias, sigma, Gamma)
- update_posterior(prior, measurements, B_grid, B_bias, sigma, Gamma)
- KL_divergence(post, prior, B_grid)
- expected_KL(B_bias, prior, B_grid, sigma, Gamma, n_samples)
- run_experiment(N_exp, B_true, Gamma, sigma, B_range, n_grid)
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from scipy.stats import norm
from scipy.integrate import simpson

__all__ = [
    "alpha_theory",
    "likelihood",
    "update_posterior",
    "KL_divergence",
    "expected_KL",
    "run_experiment",
]


# --- Signal Model: Dispersive Lorentzian ---
def alpha_theory(B_total: np.ndarray | float, Gamma: float = 1.0) -> np.ndarray | float:
    """Simplified dispersive Lorentzian form.

    alpha(B) = B / (B^2 + Gamma^2)
    """
    return B_total / (B_total ** 2 + Gamma ** 2)


# --- Likelihood for a measurement ---
def likelihood(
    alpha_meas: float | np.ndarray,
    B_unknown: float,
    B_bias: float,
    sigma: float = 0.05,
    Gamma: float = 1.0,
):
    """Gaussian likelihood of measuring alpha_meas given B_unknown and B_bias.

    Parameters
    - alpha_meas: measured alpha value(s)
    - B_unknown: unknown field value to evaluate likelihood at
    - B_bias: controllable bias field
    - sigma: measurement noise std
    - Gamma: Lorentzian width
    """
    B_total = B_unknown + B_bias
    alpha_exp = alpha_theory(B_total, Gamma)
    return norm.pdf(alpha_meas, loc=alpha_exp, scale=sigma)


# --- Posterior update ---
def update_posterior(
    prior: np.ndarray,
    measurements: np.ndarray,
    B_grid: np.ndarray,
    B_bias: float,
    sigma: float = 0.05,
    Gamma: float = 1.0,
) -> np.ndarray:
    """Update posterior over B_grid given measurements (array of alpha values).

    Returns a normalized posterior defined on B_grid.
    """
    post = prior.copy()
    for alpha_meas in np.atleast_1d(measurements):
        L = np.array([likelihood(alpha_meas, b, B_bias, sigma, Gamma) for b in B_grid])
        post *= L
    # normalize using Simpson integration over the grid
    norm_const = simpson(post, B_grid)
    if norm_const <= 0:
        # avoid divide-by-zero; fall back to prior if degenerate
        return prior / simpson(prior, B_grid)
    post /= norm_const
    return post


# --- KL Divergence Utility ---
def KL_divergence(post: np.ndarray, prior: np.ndarray, B_grid: np.ndarray) -> float:
    """Compute KL(post || prior) over B_grid using Simpson integration.

    Adds a small epsilon to the prior to avoid division-by-zero.
    """
    eps = 1e-12
    ratio = np.where(post > 0, post / (prior + eps), 1.0)
    return float(simpson(post * np.log(ratio), B_grid))


def expected_KL(
    B_bias: float,
    prior: np.ndarray,
    B_grid: np.ndarray,
    sigma: float = 0.05,
    Gamma: float = 1.0,
    n_samples: int = 30,
) -> float:
    """Compute expected information gain (KL) from a single future measurement.

    Approximates the integral by sampling hypothetical outcomes alpha over a grid.
    """
    # Sample possible outcomes for alpha
    alphas = np.linspace(-1.5, 1.5, n_samples)
    P_alpha = []
    KL_vals = []

    for alpha in alphas:
        # Marginal likelihood P(alpha | B_bias)
        integrand = np.array([
            likelihood(alpha, b, B_bias, sigma, Gamma) * prior[k]
            for k, b in enumerate(B_grid)
        ])
        P_alpha_val = simpson(integrand, B_grid)
        if P_alpha_val < 1e-18:
            continue

        # Posterior given alpha
        post = np.array([
            likelihood(alpha, b, B_bias, sigma, Gamma) * prior[k]
            for k, b in enumerate(B_grid)
        ])
        post /= simpson(post, B_grid)
        KL_vals.append(KL_divergence(post, prior, B_grid))
        P_alpha.append(P_alpha_val)

    if not P_alpha:
        return 0.0

    P_alpha_arr = np.array(P_alpha)
    KL_arr = np.array(KL_vals)
    return float(np.sum(P_alpha_arr * KL_arr) / np.sum(P_alpha_arr))


# --- Adaptive Bayesian Experiment Loop ---
def run_experiment(
    N_exp: int = 5,
    B_true: float = 0.2,
    Gamma: float = 1.0,
    sigma: float = 0.05,
    B_range: tuple[float, float] = (-1.0, 1.0),
    n_grid: int = 500,
    rng: Optional[np.random.Generator] = None,
):
    """Run an adaptive loop selecting bias by maximizing expected KL.

    Returns:
    - B_grid: np.ndarray
    - posteriors: list[np.ndarray], length N_exp+1 (including prior)
    - B_biases: list[float], one per iteration
    """
    rng_gen: np.random.Generator = rng or np.random.default_rng()

    B_grid = np.linspace(B_range[0], B_range[1], n_grid)
    prior = np.ones_like(B_grid) / (B_range[1] - B_range[0])  # uniform prior density
    B_biases: list[float] = []
    posteriors = [prior]

    for _ in range(N_exp):
        # choose bias by maximizing expected KL over candidate set
        candidates = np.linspace(-0.5, 0.5, 25)
        util_vals = [expected_KL(b, prior, B_grid, sigma, Gamma) for b in candidates]
        B_bias = float(candidates[int(np.argmax(util_vals))])
        B_biases.append(B_bias)

        # generate synthetic measurements
        B_total = B_true + B_bias
        alpha_true = alpha_theory(B_total, Gamma)
        alpha_meas = rng_gen.normal(alpha_true, sigma, size=10)  # 10 samples

        # update posterior
        posterior = update_posterior(prior, alpha_meas, B_grid, B_bias, sigma, Gamma)
        posteriors.append(posterior)
        prior = posterior

    return B_grid, posteriors, B_biases


if __name__ == "__main__":
    B_grid, posts, biases = run_experiment()
    print("Chosen biases:", biases)
