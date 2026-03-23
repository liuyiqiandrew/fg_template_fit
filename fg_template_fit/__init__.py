"""Foreground template fitting utilities for split-based GLS estimation."""

from .core import (
    apply_N_blocks,
    apply_P_qu,
    FitMode,
    cross_split_fit_operatorW,
    enforce_spd_blocks,
    make_C_operator,
    make_mask_index,
    make_Minv_operator,
    pack_qu,
    solve_C,
    unpack_qu,
)
from .examples import example_driver
from .filters import (
    EllFilterSpec,
    MFilterSpec,
    apply_m_window_inplace,
    make_ell_window,
    make_m_window,
)
from .monte_carlo import mc_uncertainty_ad_as, rademacher_blocks, sign_flip_noise

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
    "apply_m_window_inplace",
    "rademacher_blocks",
    "sign_flip_noise",
    "mc_uncertainty_ad_as",
    "example_driver",
]
