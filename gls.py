"""Backward-compatible exports for GLS utilities.

Prefer importing from ``fg_template_fit.core``.
"""

from fg_template_fit.core import *  # noqa: F401,F403
from fg_template_fit.examples import example_driver
from fg_template_fit.filters import EllFilterSpec, MFilterSpec, make_ell_window, make_m_window

__all__ = [
    "make_mask_index",
    "pack_qu",
    "unpack_qu",
    "apply_N_blocks",
    "enforce_spd_blocks",
    "apply_P_qu",
    "FitMode",
    "make_C_operator",
    "make_Minv_operator",
    "solve_C",
    "cross_split_fit_operatorW",
    "EllFilterSpec",
    "MFilterSpec",
    "make_ell_window",
    "make_m_window",
    "example_driver",
]
