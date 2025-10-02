"""
Bayesian NMOR (Nonlinear Magneto Optical Magnetometry) Package

This package provides Bayesian inference tools for adaptive experimental design
in nonlinear magneto-optical rotation measurements.

Key modules:
- bayes_experiment: Core experimental utilities and algorithms
"""

from .bayes_experiment import (
    alpha_theory,
    likelihood,
    update_posterior,
    KL_divergence,
    expected_KL,
    run_experiment,
)

__version__ = "0.1.0"
__author__ = "Pranav Agrawal"

__all__ = [
    "alpha_theory",
    "likelihood", 
    "update_posterior",
    "KL_divergence",
    "expected_KL",
    "run_experiment",
]