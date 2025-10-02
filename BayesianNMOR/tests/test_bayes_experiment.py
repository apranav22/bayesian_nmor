import numpy as np
import math
from scipy.integrate import simpson

from BayesianNMOR.bayes_experiment import (
    alpha_theory,
    likelihood,
    update_posterior,
    KL_divergence,
    expected_KL,
    run_experiment,
)


def test_alpha_theory_properties():
    Gamma = 2.0
    # odd function: f(-x) = -f(x)
    xs = np.linspace(-5, 5, 101)
    vals = alpha_theory(xs, Gamma)
    vals_neg = alpha_theory(-xs, Gamma)
    np.testing.assert_allclose(vals_neg, -vals, rtol=1e-12, atol=1e-12)

    # bounded by +- 1/(2*Gamma)
    max_bound = 1.0 / (2.0 * Gamma)
    assert np.all(np.abs(vals) <= max_bound + 1e-12)

    # maximum at B = Gamma gives exactly 1/(2*Gamma)
    assert math.isclose(alpha_theory(Gamma, Gamma), max_bound, rel_tol=1e-12, abs_tol=1e-12)


def test_likelihood_matches_gaussian_peak():
    B_unknown = 0.3
    B_bias = -0.1
    Gamma = 1.0
    sigma = 0.05
    alpha_exp = alpha_theory(B_unknown + B_bias, Gamma)
    # pdf at the mean should be the Gaussian peak
    from scipy.stats import norm

    expected_peak = norm.pdf(alpha_exp, loc=alpha_exp, scale=sigma)
    got = likelihood(alpha_exp, B_unknown, B_bias, sigma, Gamma)
    assert math.isclose(got, expected_peak, rel_tol=1e-12, abs_tol=1e-12)


def test_update_posterior_normalization_and_peaks_near_true():
    rng = np.random.default_rng(0)
    B_true = 0.2
    Gamma = 1.0
    sigma = 0.02
    B_grid = np.linspace(-1, 1, 1001)
    prior = np.ones_like(B_grid) / 2.0  # uniform over [-1, 1]

    B_bias = 0.0
    alpha_true = alpha_theory(B_true + B_bias, Gamma)
    measurements = rng.normal(alpha_true, sigma, size=50)

    posterior = update_posterior(prior, measurements, B_grid, B_bias, sigma, Gamma)

    # normalization check
    area = simpson(posterior, B_grid)
    assert math.isclose(area, 1.0, rel_tol=1e-4, abs_tol=1e-4)

    # mode proximity: posterior should peak near B_true (within grid resolution)
    mode_idx = int(np.argmax(posterior))
    B_mode = B_grid[mode_idx]
    assert abs(B_mode - B_true) < 0.05


def test_KL_divergence_nonnegative_and_zero_when_equal():
    B_grid = np.linspace(-1, 1, 1001)
    prior = np.ones_like(B_grid) / 2.0
    post_same = prior.copy()
    kl0 = KL_divergence(post_same, prior, B_grid)
    assert abs(kl0) < 1e-10

    # a slightly peaked posterior should have positive KL
    post_peak = prior * (1 + 0.2 * np.exp(-((B_grid - 0.1) ** 2) / 0.01))
    post_peak /= simpson(post_peak, B_grid)
    klp = KL_divergence(post_peak, prior, B_grid)
    assert klp >= 0.0


def test_expected_KL_higher_near_zero_bias_than_edge():
    B_grid = np.linspace(-1, 1, 1001)
    prior = np.ones_like(B_grid) / 2.0
    Gamma = 1.0
    sigma = 0.05
    kl0 = expected_KL(0.0, prior, B_grid, sigma, Gamma, n_samples=41)
    kl_edge = expected_KL(0.5, prior, B_grid, sigma, Gamma, n_samples=41)
    assert kl0 >= kl_edge - 1e-6


def test_run_experiment_shapes_and_normalization():
    B_grid, posts, biases = run_experiment(N_exp=3, B_true=0.15, n_grid=401, sigma=0.03)

    # number of posteriors equals N_exp + 1 (includes prior)
    assert len(posts) == 4
    assert len(biases) == 3
    assert posts[0].shape == B_grid.shape

    # each posterior normalizes to ~1
    for p in posts:
        area = simpson(p, B_grid)
        assert math.isclose(area, 1.0, rel_tol=5e-3, abs_tol=5e-3)

    # biases fall in candidate range [-0.5, 0.5]
    assert np.all(np.array(biases) >= -0.5 - 1e-12)
    assert np.all(np.array(biases) <= 0.5 + 1e-12)
