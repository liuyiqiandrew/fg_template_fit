from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import healpy as hp
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
FilterMode = Literal["C1", "C2"]
FilterKind = Literal["highpass", "lowpass"]


@dataclass(frozen=True)
class EllFilterSpec:
    """Specification for a smooth ell-space filter window.

    Parameters
    ----------
    ell0 : float
        Center multipole for the transition.
    dell : float, optional
        Half-width of the transition band in multipole space. Use ``0`` for
        a hard step.
    mode : {'C1', 'C2'}, optional
        NaMaster-style smooth transition family.
    kind : {'highpass', 'lowpass'}, optional
        Filter type. ``highpass`` suppresses low ell; ``lowpass`` suppresses
        high ell.
    """

    ell0: float
    dell: float = 0.0
    mode: FilterMode = "C2"
    kind: FilterKind = "highpass"


@dataclass(frozen=True)
class MFilterSpec:
    """Specification for a smooth m-space filter window.

    Parameters
    ----------
    m0 : float
        Center azimuthal mode for the transition.
    dm : float, optional
        Half-width of the transition band in ``m``. Use ``0`` for a hard step.
    mode : {'C1', 'C2'}, optional
        NaMaster-style smooth transition family.
    kind : {'highpass', 'lowpass'}, optional
        Filter type. ``highpass`` suppresses low ``m``; ``lowpass`` suppresses
        high ``m``.
    """

    m0: float
    dm: float = 0.0
    mode: FilterMode = "C2"
    kind: FilterKind = "highpass"


EllFilterInput = Union[npt.NDArray[np.floating], EllFilterSpec]
MFilterInput = Union[npt.NDArray[np.floating], MFilterSpec]


def namaster_c_window(x: npt.NDArray[np.floating], mode: FilterMode) -> FloatArray:
    """Evaluate NaMaster-style transition windows on ``x in [0, 1]``.

    Parameters
    ----------
    x : ndarray of float
        Input transition coordinate in ``[0, 1]``.
    mode : {'C1', 'C2'}
        Window family.

    Returns
    -------
    ndarray of float64
        Window values with the same shape as ``x``.
    """
    mode_u = mode.upper()
    x = np.asarray(x, dtype=np.float64)
    if mode_u == "C1":
        return x - np.sin(2.0 * np.pi * x) / (2.0 * np.pi)
    if mode_u == "C2":
        return 0.5 * (1.0 - np.cos(np.pi * x))
    raise ValueError("mode must be 'C1' or 'C2'.")


def make_ell_window(
    lmax: int,
    ell0: float,
    dell: float = 0.0,
    mode: FilterMode = "C2",
    kind: FilterKind = "highpass",
) -> FloatArray:
    """Build a smooth or hard window ``w(ell)`` over multipoles.

    Parameters
    ----------
    lmax : int
        Maximum multipole index.
    ell0 : float
        Transition center.
    dell : float, optional
        Transition half-width. Hard step if ``dell <= 0``.
    mode : {'C1', 'C2'}, optional
        Window family used for smooth transitions.
    kind : {'highpass', 'lowpass'}, optional
        High-pass or low-pass behavior.

    Returns
    -------
    ndarray of float64
        Window values of shape ``(lmax + 1,)``.
    """
    ell = np.arange(lmax + 1, dtype=np.float64)
    if dell <= 0.0:
        w = (ell > ell0).astype(np.float64)
    else:
        x = (ell - ell0) / float(dell)
        t = np.clip((x + 1.0) * 0.5, 0.0, 1.0)
        w = namaster_c_window(t, mode=mode)

    kind_l = kind.lower()
    if kind_l == "highpass":
        return w
    if kind_l == "lowpass":
        return 1.0 - w
    raise ValueError("kind must be 'highpass' or 'lowpass'.")


def make_m_window(
    lmax: int,
    m0: float,
    dm: float = 0.0,
    mode: FilterMode = "C2",
    kind: FilterKind = "highpass",
) -> FloatArray:
    """Build a smooth or hard window ``w(m)`` over azimuthal modes.

    Parameters
    ----------
    lmax : int
        Maximum multipole / mode index.
    m0 : float
        Transition center in ``m``.
    dm : float, optional
        Transition half-width. Hard step if ``dm <= 0``.
    mode : {'C1', 'C2'}, optional
        Window family used for smooth transitions.
    kind : {'highpass', 'lowpass'}, optional
        High-pass or low-pass behavior.

    Returns
    -------
    ndarray of float64
        Window values of shape ``(lmax + 1,)``.
    """
    m = np.arange(lmax + 1, dtype=np.float64)
    if dm <= 0.0:
        w = (m > m0).astype(np.float64)
    else:
        x = (m - m0) / float(dm)
        t = np.clip((x + 1.0) * 0.5, 0.0, 1.0)
        w = namaster_c_window(t, mode=mode)

    kind_l = kind.lower()
    if kind_l == "highpass":
        return w
    if kind_l == "lowpass":
        return 1.0 - w
    raise ValueError("kind must be 'highpass' or 'lowpass'.")


def apply_m_window_inplace(alm: npt.NDArray[np.complexfloating], w_m: FloatArray, lmax: int) -> None:
    """Apply an ``m``-window to a packed HEALPix ``alm`` array in place.

    Parameters
    ----------
    alm : ndarray of complex
        HEALPix-packed spherical harmonic coefficients.
    w_m : ndarray of float64
        Azimuthal window of shape ``(lmax + 1,)``.
    lmax : int
        Harmonic truncation used by ``alm``.
    """
    for m in range(lmax + 1):
        w = w_m[m]
        if w == 1.0:
            continue
        i0 = hp.Alm.getidx(lmax, m, m)
        i1 = hp.Alm.getidx(lmax, lmax, m)
        alm[i0 : i1 + 1] *= w


def resolve_ell_filter(
    lmax: int,
    ell_filter: Optional[EllFilterInput] = None,
    fl: Optional[npt.NDArray[np.floating]] = None,
) -> Optional[FloatArray]:
    """Resolve ell filter input into a concrete harmonic window.

    Parameters
    ----------
    lmax : int
        Harmonic truncation for the returned window.
    ell_filter : ndarray or EllFilterSpec, optional
        Explicit window or window specification.
    fl : ndarray, optional
        Legacy alias for an explicit ell window.

    Returns
    -------
    ndarray of float64 or None
        Resolved ell window of shape ``(lmax + 1,)``, or ``None`` if no
        ell-space filtering is requested.
    """
    if ell_filter is not None and fl is not None:
        raise ValueError("Pass only one of `ell_filter` or legacy `fl`, not both.")

    candidate: Optional[EllFilterInput]
    if ell_filter is not None:
        candidate = ell_filter
    else:
        candidate = fl

    if candidate is None:
        return None

    if isinstance(candidate, EllFilterSpec):
        return make_ell_window(
            lmax=lmax,
            ell0=candidate.ell0,
            dell=candidate.dell,
            mode=candidate.mode,
            kind=candidate.kind,
        )

    arr = np.asarray(candidate, dtype=np.float64)
    if arr.shape != (lmax + 1,):
        raise ValueError(f"ell filter must have shape ({lmax + 1},), got {arr.shape}.")
    return arr


def resolve_m_filter(lmax: int, m_filter: Optional[MFilterInput] = None) -> Optional[FloatArray]:
    """Resolve m filter input into a concrete harmonic window.

    Parameters
    ----------
    lmax : int
        Harmonic truncation for the returned window.
    m_filter : ndarray or MFilterSpec, optional
        Explicit window or window specification.

    Returns
    -------
    ndarray of float64 or None
        Resolved ``m`` window of shape ``(lmax + 1,)``, or ``None`` if no
        ``m``-space filtering is requested.
    """
    if m_filter is None:
        return None

    if isinstance(m_filter, MFilterSpec):
        return make_m_window(
            lmax=lmax,
            m0=m_filter.m0,
            dm=m_filter.dm,
            mode=m_filter.mode,
            kind=m_filter.kind,
        )

    arr = np.asarray(m_filter, dtype=np.float64)
    if arr.shape != (lmax + 1,):
        raise ValueError(f"m filter must have shape ({lmax + 1},), got {arr.shape}.")
    return arr


__all__ = [
    "EllFilterSpec",
    "MFilterSpec",
    "EllFilterInput",
    "MFilterInput",
    "namaster_c_window",
    "make_ell_window",
    "make_m_window",
    "apply_m_window_inplace",
    "resolve_ell_filter",
    "resolve_m_filter",
]
