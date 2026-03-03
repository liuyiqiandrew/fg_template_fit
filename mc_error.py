"""Backward-compatible exports for Monte Carlo uncertainty helpers.

Prefer importing from ``fg_template_fit.monte_carlo``.
"""

from fg_template_fit.monte_carlo import *  # noqa: F401,F403

__all__ = ["rademacher_blocks", "sign_flip_noise", "mc_uncertainty_ad_as"]
