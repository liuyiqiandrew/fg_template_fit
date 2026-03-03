from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
FitFunction = Callable[[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray], tuple[float, float]]
NoiseDrawFunction = Callable[[np.random.Generator], FloatArray]


def rademacher_blocks(idx: npt.NDArray[np.int64], block_nside: int, seed: int = 0) -> FloatArray:
    """Placeholder for block-constant Rademacher signs on a masked domain.

    Parameters
    ----------
    idx : ndarray of int64
        Full-sky pixel indices retained by the mask at working ``nside``.
    block_nside : int
        ``nside`` of superpixels used to define constant-sign blocks.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    ndarray of float64
        Intended output is one sign value per retained pixel in ``idx``.

    Raises
    ------
    NotImplementedError
        Always raised because the block mapping logic is not implemented.
    """
    _ = (idx, block_nside, seed)
    raise NotImplementedError("Block-based Rademacher generation is not implemented yet.")


def sign_flip_noise(v_noise: npt.NDArray[np.floating], signs: npt.NDArray[np.floating]) -> FloatArray:
    """Apply shared per-pixel sign flips to packed Q/U noise vectors.

    Parameters
    ----------
    v_noise : ndarray of float
        Packed noise vector of size ``2 * n`` with ``[Q; U]`` ordering.
    signs : ndarray of float
        Sign vector of size ``n`` containing values near ``{-1, +1}``.

    Returns
    -------
    ndarray of float64
        Sign-flipped packed vector where the same sign is applied to Q and U
        at each kept pixel.
    """
    n = signs.size
    out = v_noise.astype(np.float64, copy=True)
    out[:n] *= signs
    out[n:] *= signs
    return out


def mc_uncertainty_ad_as(
    fit_fn: FitFunction,
    y: npt.NDArray[np.floating],
    d1: npt.NDArray[np.floating],
    d2: npt.NDArray[np.floating],
    s1: npt.NDArray[np.floating],
    s2: npt.NDArray[np.floating],
    n_mc: int = 200,
    seed: int = 0,
    y2: Optional[npt.NDArray[np.floating]] = None,
    draw_y_noise_fn: Optional[NoiseDrawFunction] = None,
    signs_per_pixel: bool = True,
) -> dict[str, object]:
    """Estimate uncertainty on ``(a_d, a_s)`` via split-based Monte Carlo.

    Parameters
    ----------
    fit_fn : callable
        Function that returns ``(a_d, a_s)`` for one realization:
        ``fit_fn(y_i, d1_i, d2_i, s1_i, s2_i)``.
    y : ndarray of float
        Packed target vector.
    d1 : ndarray of float
        First dust split.
    d2 : ndarray of float
        Second dust split.
    s1 : ndarray of float
        First synchrotron split.
    s2 : ndarray of float
        Second synchrotron split.
    n_mc : int, optional
        Number of Monte Carlo realizations.
    seed : int, optional
        Seed for the random number generator.
    y2 : ndarray of float, optional
        Second target split. If provided, target noise is sampled via sign
        flips on ``0.5 * (y - y2)``.
    draw_y_noise_fn : callable, optional
        Function to draw model target noise when ``y2`` is not supplied.
        It must return a packed QU noise vector.
    signs_per_pixel : bool, optional
        If ``True``, sign flips are generated independently per kept pixel.
        The current fallback for ``False`` still uses per-pixel signs.

    Returns
    -------
    dict
        Dictionary with sample means, standard deviations, covariance matrix,
        and full sample arrays. Keys are:
        ``ad_mean``, ``ad_std``, ``as_mean``, ``as_std``, ``cov``,
        ``ad_samps``, ``as_samps``.
    """
    rng = np.random.default_rng(seed)

    dbar = 0.5 * (d1 + d2)
    ddif = 0.5 * (d1 - d2)
    sbar = 0.5 * (s1 + s2)
    sdif = 0.5 * (s1 - s2)

    if y2 is not None:
        ybar = 0.5 * (y + y2)
        ydif = 0.5 * (y - y2)
    else:
        ybar = y.astype(np.float64, copy=False)
        ydif = None

    n = y.size // 2
    ad_samps = np.empty(n_mc, dtype=np.float64)
    as_samps = np.empty(n_mc, dtype=np.float64)

    for i in range(n_mc):
        if signs_per_pixel:
            signs = rng.choice([-1.0, 1.0], size=n)
        else:
            signs = rng.choice([-1.0, 1.0], size=n)

        nd = sign_flip_noise(ddif, signs)
        ns = sign_flip_noise(sdif, signs)

        d1_i = dbar + nd
        d2_i = dbar - nd
        s1_i = sbar + ns
        s2_i = sbar - ns

        if ydif is not None:
            ny = sign_flip_noise(ydif, signs)
            y_i = ybar + ny
        elif draw_y_noise_fn is None:
            y_i = ybar
        else:
            y_i = ybar + draw_y_noise_fn(rng)

        ad, a_s = fit_fn(
            y_i.astype(np.float64, copy=False),
            d1_i.astype(np.float64, copy=False),
            d2_i.astype(np.float64, copy=False),
            s1_i.astype(np.float64, copy=False),
            s2_i.astype(np.float64, copy=False),
        )
        ad_samps[i] = ad
        as_samps[i] = a_s

    return {
        "ad_mean": float(np.mean(ad_samps)),
        "ad_std": float(np.std(ad_samps, ddof=1)),
        "as_mean": float(np.mean(as_samps)),
        "as_std": float(np.std(as_samps, ddof=1)),
        "cov": np.cov(np.vstack([ad_samps, as_samps]), ddof=1),
        "ad_samps": ad_samps,
        "as_samps": as_samps,
    }


__all__ = ["rademacher_blocks", "sign_flip_noise", "mc_uncertainty_ad_as"]
