from __future__ import annotations

from typing import Literal, Optional, overload

import healpy as hp
import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator, cg
from .filters import (
    EllFilterInput,
    MFilterInput,
    apply_m_window_inplace,
    resolve_ell_filter,
    resolve_m_filter,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
FitMode = Literal["both", "dust", "synchrotron"]


def make_mask_index(w: npt.NDArray[np.floating], thr: float = 1e-6) -> IntArray:
    """Build the pixel index list for a masked domain.

    Parameters
    ----------
    w : ndarray of float
        Full-sky apodization weights in ``[0, 1]``.
    thr : float, optional
        Threshold used to keep valid pixels. Pixels with ``w > thr`` are
        retained.

    Returns
    -------
    ndarray of int64
        Indices of retained pixels in the full-sky map.
    """
    idx = np.where(w > thr)[0]
    return idx.astype(np.int64, copy=False)


def pack_qu(
    q_full: npt.NDArray[np.floating],
    u_full: npt.NDArray[np.floating],
    idx: IntArray,
) -> FloatArray:
    """Pack full-sky Q/U maps into a compressed masked-domain vector.

    Parameters
    ----------
    q_full : ndarray of float
        Full-sky Q map.
    u_full : ndarray of float
        Full-sky U map.
    idx : ndarray of int64
        Full-sky pixel indices to keep.

    Returns
    -------
    ndarray of float64
        Concatenated vector ``[Q(idx); U(idx)]``.
    """
    return np.concatenate([q_full[idx], u_full[idx]]).astype(np.float64, copy=False)


def unpack_qu(v: npt.NDArray[np.floating], npix: int, idx: IntArray) -> tuple[FloatArray, FloatArray]:
    """Unpack a compressed masked-domain Q/U vector back to full-sky maps.

    Parameters
    ----------
    v : ndarray of float
        Compressed vector ordered as ``[Q(idx); U(idx)]``.
    npix : int
        Number of full-sky pixels at the working ``nside``.
    idx : ndarray of int64
        Full-sky pixel indices represented in ``v``.

    Returns
    -------
    q : ndarray of float64
        Full-sky Q map with zeros outside ``idx``.
    u : ndarray of float64
        Full-sky U map with zeros outside ``idx``.
    """
    n = idx.size
    q = np.zeros(npix, dtype=np.float64)
    u = np.zeros(npix, dtype=np.float64)
    q[idx] = v[:n]
    u[idx] = v[n:]
    return q, u


def apply_N_blocks(
    q: npt.NDArray[np.floating],
    u: npt.NDArray[np.floating],
    qq: npt.NDArray[np.floating],
    uu: npt.NDArray[np.floating],
    qu: npt.NDArray[np.floating],
) -> tuple[FloatArray, FloatArray]:
    """Apply per-pixel 2x2 noise blocks to full-sky Q/U maps.

    Parameters
    ----------
    q : ndarray of float
        Input full-sky Q map.
    u : ndarray of float
        Input full-sky U map.
    qq : ndarray of float
        ``N_qq`` element per pixel.
    uu : ndarray of float
        ``N_uu`` element per pixel.
    qu : ndarray of float
        ``N_qu`` element per pixel.

    Returns
    -------
    q2 : ndarray of float64
        Output full-sky Q map after block multiplication.
    u2 : ndarray of float64
        Output full-sky U map after block multiplication.

    Notes
    -----
    The block model is:

    ``N_p = [[qq_p, qu_p], [qu_p, uu_p]]``.
    """
    q2 = qq * q + qu * u
    u2 = qu * q + uu * u
    return q2.astype(np.float64, copy=False), u2.astype(np.float64, copy=False)


def enforce_spd_blocks(
    qq: npt.NDArray[np.floating],
    uu: npt.NDArray[np.floating],
    qu: npt.NDArray[np.floating],
    ridge_frac: float = 1e-6,
    eps: float = 1e-30,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Stabilize per-pixel 2x2 blocks to be positive definite.

    Parameters
    ----------
    qq : ndarray of float
        Per-pixel ``N_qq`` block components.
    uu : ndarray of float
        Per-pixel ``N_uu`` block components.
    qu : ndarray of float
        Per-pixel ``N_qu`` block components.
    ridge_frac : float, optional
        Fraction of a robust scale estimate used for diagonal ridge addition
        on problematic pixels.
    eps : float, optional
        Tiny additive floor for numerical safety.

    Returns
    -------
    qq_out : ndarray of float64
        Stabilized ``qq`` values.
    uu_out : ndarray of float64
        Stabilized ``uu`` values.
    qu_out : ndarray of float64
        Unchanged cross term (returned for convenience).

    Notes
    -----
    A pixel is treated as problematic if either diagonal entry is non-positive
    or the determinant ``qq * uu - qu**2`` is non-positive.
    """
    qq = qq.astype(np.float64, copy=False)
    uu = uu.astype(np.float64, copy=False)
    qu = qu.astype(np.float64, copy=False)

    det = qq * uu - qu * qu
    bad = (qq <= 0.0) | (uu <= 0.0) | (det <= 0.0)

    if np.any(bad):
        good = ~bad
        scale = np.median(0.5 * (qq[good] + uu[good])) if np.any(good) else 1.0
        ridge = ridge_frac * scale + eps
        qq = qq.copy()
        uu = uu.copy()
        qq[bad] += ridge
        uu[bad] += ridge

    return qq, uu, qu


def apply_P_qu(
    q: npt.NDArray[np.floating],
    u: npt.NDArray[np.floating],
    nside: int,
    fwhm_rad: Optional[float],
    fl: Optional[npt.NDArray[np.floating]] = None,
    ell_filter: Optional[EllFilterInput] = None,
    m_filter: Optional[MFilterInput] = None,
    lmax: Optional[int] = None,
) -> tuple[FloatArray, FloatArray]:
    """Apply the spin-2 processing operator ``P`` to full-sky Q/U maps.

    Parameters
    ----------
    q : ndarray of float
        Input full-sky Q map.
    u : ndarray of float
        Input full-sky U map.
    nside : int
        HEALPix ``nside`` for input and output maps.
    fwhm_rad : float or None
        Gaussian beam full width at half maximum in radians. If ``None`` or
        non-positive, beam smoothing is skipped.
    fl : ndarray of float, optional
        Legacy alias for an explicit ``ell`` filter array of shape
        ``(lmax + 1,)``.
    ell_filter : ndarray or EllFilterSpec, optional
        ``ell``-space filter provided either as an explicit window or a
        specification object. If both ``fl`` and ``ell_filter`` are passed, a
        ``ValueError`` is raised.
    m_filter : ndarray or MFilterSpec, optional
        ``m``-space filter provided either as an explicit window or a
        specification object.
    lmax : int, optional
        Harmonic truncation. Defaults to ``3 * nside - 1``.

    Returns
    -------
    q2 : ndarray of float64
        Processed full-sky Q map.
    u2 : ndarray of float64
        Processed full-sky U map.

    Notes
    -----
    The operator applies:

    1. spin-2 map-to-alm transform,
    2. optional ``ell``-space filter,
    3. optional ``m``-space filter,
    4. optional isotropic beam,
    5. spin-2 alm-to-map transform.
    """
    if lmax is None:
        lmax = 3 * nside - 1

    alm_e, alm_b = hp.map2alm_spin([q, u], spin=2, lmax=lmax)
    ell_window = resolve_ell_filter(lmax=lmax, ell_filter=ell_filter, fl=fl)
    m_window = resolve_m_filter(lmax=lmax, m_filter=m_filter)

    if ell_window is not None:
        hp.almxfl(alm_e, ell_window, inplace=True)
        hp.almxfl(alm_b, ell_window, inplace=True)

    if m_window is not None:
        apply_m_window_inplace(alm_e, m_window, lmax=lmax)
        apply_m_window_inplace(alm_b, m_window, lmax=lmax)

    if fwhm_rad is not None and fwhm_rad > 0.0:
        bl = hp.gauss_beam(fwhm=fwhm_rad, lmax=lmax)
        hp.almxfl(alm_e, bl, inplace=True)
        hp.almxfl(alm_b, bl, inplace=True)

    q2, u2 = hp.alm2map_spin([alm_e, alm_b], nside=nside, spin=2, lmax=lmax)
    return q2.astype(np.float64, copy=False), u2.astype(np.float64, copy=False)


def make_C_operator(
    nside: int,
    w_apod: npt.NDArray[np.floating],
    idx: IntArray,
    qq: npt.NDArray[np.floating],
    uu: npt.NDArray[np.floating],
    qu: npt.NDArray[np.floating],
    fwhm_rad: Optional[float],
    fl: Optional[npt.NDArray[np.floating]] = None,
    ell_filter: Optional[EllFilterInput] = None,
    m_filter: Optional[MFilterInput] = None,
    lmax: Optional[int] = None,
    reg_eps: float = 0.0,
) -> LinearOperator:
    """Construct the covariance operator ``C`` in compressed QU space.

    Parameters
    ----------
    nside : int
        HEALPix resolution.
    w_apod : ndarray of float
        Full-sky apodization weights.
    idx : ndarray of int64
        Full-sky pixel indices defining the compressed domain.
    qq : ndarray of float
        Per-pixel ``N_qq`` noise components at working ``nside``.
    uu : ndarray of float
        Per-pixel ``N_uu`` noise components at working ``nside``.
    qu : ndarray of float
        Per-pixel ``N_qu`` noise components at working ``nside``.
    fwhm_rad : float or None
        Beam smoothing FWHM in radians for operator ``P``.
    fl : ndarray of float, optional
        Legacy alias for an explicit ``ell`` filter window.
    ell_filter : ndarray or EllFilterSpec, optional
        ``ell``-space filter used inside ``P``.
    m_filter : ndarray or MFilterSpec, optional
        ``m``-space filter used inside ``P``.
    lmax : int, optional
        Harmonic truncation used inside ``P``.
    reg_eps : float, optional
        Diagonal regularization term added as ``reg_eps * v`` to keep the
        operator well conditioned if ``P`` has null modes.

    Returns
    -------
    scipy.sparse.linalg.LinearOperator
        Linear operator implementing ``C`` on vectors of size ``2 * len(idx)``.

    Notes
    -----
    The implemented action is approximately:

    ``C(v) = P( w * N( w * P(v) ) ) + reg_eps * v``.
    """
    npix = hp.nside2npix(nside)
    n = idx.size
    if lmax is None:
        lmax = 3 * nside - 1

    qq_spd, uu_spd, qu_spd = enforce_spd_blocks(qq, uu, qu)
    w = w_apod.astype(np.float64, copy=False)
    ell_window = resolve_ell_filter(lmax=lmax, ell_filter=ell_filter, fl=fl)
    m_window = resolve_m_filter(lmax=lmax, m_filter=m_filter)

    def matvec(v: npt.NDArray[np.floating]) -> FloatArray:
        q_full, u_full = unpack_qu(v.astype(np.float64, copy=False), npix=npix, idx=idx)

        q_full, u_full = apply_P_qu(
            q_full,
            u_full,
            nside=nside,
            fwhm_rad=fwhm_rad,
            ell_filter=ell_window,
            m_filter=m_window,
            lmax=lmax,
        )

        q_full *= w
        u_full *= w

        q_full, u_full = apply_N_blocks(q_full, u_full, qq=qq_spd, uu=uu_spd, qu=qu_spd)

        q_full *= w
        u_full *= w

        q_full, u_full = apply_P_qu(
            q_full,
            u_full,
            nside=nside,
            fwhm_rad=fwhm_rad,
            ell_filter=ell_window,
            m_filter=m_window,
            lmax=lmax,
        )

        out = pack_qu(q_full, u_full, idx=idx)

        if reg_eps != 0.0:
            out = out + reg_eps * v

        return out.astype(np.float64, copy=False)

    ndim = 2 * n
    return LinearOperator((ndim, ndim), matvec=matvec, dtype=np.float64)


def make_Minv_operator(
    w_apod: npt.NDArray[np.floating],
    idx: IntArray,
    qq_eff: npt.NDArray[np.floating],
    uu_eff: npt.NDArray[np.floating],
    qu_eff: npt.NDArray[np.floating],
    ridge: float = 1e-12,
) -> LinearOperator:
    """Build a block-Jacobi preconditioner in compressed QU space.

    Parameters
    ----------
    w_apod : ndarray of float
        Full-sky apodization weights.
    idx : ndarray of int64
        Full-sky pixel indices defining the compressed domain.
    qq_eff : ndarray of float
        Approximate per-pixel ``qq`` for ``C``.
    uu_eff : ndarray of float
        Approximate per-pixel ``uu`` for ``C``.
    qu_eff : ndarray of float
        Approximate per-pixel ``qu`` for ``C``.
    ridge : float, optional
        Small diagonal stabilization term.

    Returns
    -------
    scipy.sparse.linalg.LinearOperator
        Approximate inverse preconditioner for vectors of length
        ``2 * len(idx)``.
    """
    w = w_apod[idx].astype(np.float64, copy=False)
    qq = (w * w) * qq_eff[idx] + ridge
    uu = (w * w) * uu_eff[idx] + ridge
    qu = (w * w) * qu_eff[idx]

    det = qq * uu - qu * qu
    inv_det = 1.0 / det

    wqq = uu * inv_det
    wuu = qq * inv_det
    wqu = -qu * inv_det

    n = idx.size

    def matvec(v: npt.NDArray[np.floating]) -> FloatArray:
        q = v[:n]
        u = v[n:]
        q2 = wqq * q + wqu * u
        u2 = wqu * q + wuu * u
        return np.concatenate([q2, u2]).astype(np.float64, copy=False)

    ndim = 2 * n
    return LinearOperator((ndim, ndim), matvec=matvec, dtype=np.float64)


def solve_C(
    cop: LinearOperator,
    b: npt.NDArray[np.floating],
    mop: Optional[LinearOperator] = None,
    tol: float = 1e-6,
    maxiter: int = 200,
) -> FloatArray:
    """Solve ``C x = b`` via conjugate gradients.

    Parameters
    ----------
    cop : scipy.sparse.linalg.LinearOperator
        Covariance operator ``C``.
    b : ndarray of float
        Right-hand side vector.
    mop : scipy.sparse.linalg.LinearOperator, optional
        Preconditioner approximating ``C^{-1}``.
    tol : float, optional
        Relative CG tolerance.
    maxiter : int, optional
        Maximum CG iterations.

    Returns
    -------
    ndarray of float64
        CG solution ``x``.

    Raises
    ------
    RuntimeError
        If CG does not converge.
    """
    try:
        x, info = cg(cop, b, M=mop, rtol=tol, atol=0.0, maxiter=maxiter)
    except TypeError as exc:
        if "rtol" not in str(exc):
            raise
        x, info = cg(cop, b, M=mop, tol=tol, maxiter=maxiter)
    if info != 0:
        raise RuntimeError(
            f"CG did not converge (info={info}). Try better preconditioning, "
            "higher maxiter, or non-zero reg_eps."
        )
    return x.astype(np.float64, copy=False)


def _resolve_fit_mode(fit_mode: FitMode | str) -> FitMode:
    """Validate and normalize the requested fit mode.

    Parameters
    ----------
    fit_mode : {"both", "dust", "synchrotron"} or str
        Requested component selection for the template fit.

    Returns
    -------
    {"both", "dust", "synchrotron"}
        Normalized fit mode.

    Raises
    ------
    ValueError
        If ``fit_mode`` is not a supported option.
    """
    allowed_modes = ("both", "dust", "synchrotron")
    if fit_mode not in allowed_modes:
        raise ValueError(
            "fit_mode must be one of "
            f"{allowed_modes}, got {fit_mode!r}."
        )
    return fit_mode


def _require_template_pair(
    template1: npt.NDArray[np.floating] | None,
    template2: npt.NDArray[np.floating] | None,
    component_name: str,
    fit_mode: FitMode,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Validate that the requested component has both split templates.

    Parameters
    ----------
    template1 : ndarray of float or None
        First split template for the selected component.
    template2 : ndarray of float or None
        Second split template for the selected component.
    component_name : str
        Human-readable component label used in error messages.
    fit_mode : {"both", "dust", "synchrotron"}
        Active fit mode requesting the component.

    Returns
    -------
    template1_resolved : ndarray of float
        Validated first split template.
    template2_resolved : ndarray of float
        Validated second split template.

    Raises
    ------
    ValueError
        If either split template is missing.
    """
    if template1 is None or template2 is None:
        raise ValueError(
            f"fit_mode={fit_mode!r} requires both {component_name} split templates."
        )
    return template1, template2


@overload
def cross_split_fit_operatorW(
    y: npt.NDArray[np.floating],
    d1: npt.NDArray[np.floating],
    d2: npt.NDArray[np.floating],
    s1: npt.NDArray[np.floating],
    s2: npt.NDArray[np.floating],
    cop: LinearOperator,
    mop: Optional[LinearOperator] = None,
    tol: float = 1e-6,
    maxiter: int = 200,
    sym: bool = True,
    fit_mode: Literal["both"] = "both",
) -> tuple[float, float, FloatArray, FloatArray]:
    ...


@overload
def cross_split_fit_operatorW(
    y: npt.NDArray[np.floating],
    d1: npt.NDArray[np.floating],
    d2: npt.NDArray[np.floating],
    s1: None = None,
    s2: None = None,
    cop: LinearOperator | None = None,
    mop: Optional[LinearOperator] = None,
    tol: float = 1e-6,
    maxiter: int = 200,
    sym: bool = True,
    fit_mode: Literal["dust"] = "dust",
) -> tuple[float, None, FloatArray, FloatArray]:
    ...


@overload
def cross_split_fit_operatorW(
    y: npt.NDArray[np.floating],
    d1: None = None,
    d2: None = None,
    s1: npt.NDArray[np.floating] | None = None,
    s2: npt.NDArray[np.floating] | None = None,
    cop: LinearOperator | None = None,
    mop: Optional[LinearOperator] = None,
    tol: float = 1e-6,
    maxiter: int = 200,
    sym: bool = True,
    fit_mode: Literal["synchrotron"] = "synchrotron",
) -> tuple[None, float, FloatArray, FloatArray]:
    ...


def cross_split_fit_operatorW(
    y: npt.NDArray[np.floating],
    d1: npt.NDArray[np.floating] | None = None,
    d2: npt.NDArray[np.floating] | None = None,
    s1: npt.NDArray[np.floating] | None = None,
    s2: npt.NDArray[np.floating] | None = None,
    cop: Optional[LinearOperator] = None,
    mop: Optional[LinearOperator] = None,
    tol: float = 1e-6,
    maxiter: int = 200,
    sym: bool = True,
    fit_mode: FitMode = "both",
) -> tuple[float | None, float | None, FloatArray, FloatArray]:
    """Estimate dust/synch template amplitudes using ``W = C^{-1}``.

    Parameters
    ----------
    y : ndarray of float
        Compressed target vector ``[Q(idx); U(idx)]``.
    d1 : ndarray of float or None, optional
        First dust split template. Required when ``fit_mode`` uses dust.
    d2 : ndarray of float or None, optional
        Second dust split template. Required when ``fit_mode`` uses dust.
    s1 : ndarray of float or None, optional
        First synchrotron split template. Required when ``fit_mode`` uses
        synchrotron.
    s2 : ndarray of float or None, optional
        Second synchrotron split template. Required when ``fit_mode`` uses
        synchrotron.
    cop : scipy.sparse.linalg.LinearOperator or None, optional
        Covariance operator ``C``.
    mop : scipy.sparse.linalg.LinearOperator, optional
        Preconditioner used for each CG solve.
    tol : float, optional
        CG relative tolerance.
    maxiter : int, optional
        CG iteration cap.
    sym : bool, optional
        If ``True``, symmetrize the normal matrix and RHS using both split
        orderings.
    fit_mode : {"both", "dust", "synchrotron"}, optional
        Component subset to fit. ``"both"`` solves the coupled 2-parameter
        system, while ``"dust"`` and ``"synchrotron"`` perform the
        corresponding single-component fit. Inactive template arguments may be
        omitted as ``None``.

    Returns
    -------
    ad : float or None
        Estimated dust amplitude. Returns ``None`` when
        ``fit_mode="synchrotron"``.
    a_s : float or None
        Estimated synchrotron amplitude. Returns ``None`` when
        ``fit_mode="dust"``.
    G : ndarray of float64
        Normal matrix used in the solve. Its shape is ``(2, 2)`` for
        ``fit_mode="both"`` and ``(1, 1)`` otherwise.
    bvec : ndarray of float64
        Right-hand side vector. Its length is ``2`` for ``fit_mode="both"``
        and ``1`` otherwise.

    Raises
    ------
    ValueError
        If ``cop`` is missing or the active fit mode is missing its required
        split templates.

    Notes
    -----
    The estimator is:

    ``a = (Z^T W X)^{-1} (Z^T W y)``, where ``X`` and ``Z`` contain the
    selected split templates. For ``fit_mode="both"``, ``X=[d1,s1]`` and
    ``Z=[d2,s2]``. For the single-component modes, only the requested dust or
    synchrotron template is included.
    """
    fit_mode = _resolve_fit_mode(fit_mode)
    if cop is None:
        raise ValueError("cop must be provided.")

    wy = solve_C(cop, y, mop=mop, tol=tol, maxiter=maxiter)

    split1_templates: list[npt.NDArray[np.floating]]
    split2_templates: list[npt.NDArray[np.floating]]
    weighted_split1_templates: list[FloatArray]
    weighted_split2_templates: list[FloatArray] = []
    if fit_mode == "both":
        dust1, dust2 = _require_template_pair(d1, d2, component_name="dust", fit_mode=fit_mode)
        sync1, sync2 = _require_template_pair(s1, s2, component_name="synchrotron", fit_mode=fit_mode)
        split1_templates = [dust1, sync1]
        split2_templates = [dust2, sync2]
        weighted_split1_templates = [
            solve_C(cop, dust1, mop=mop, tol=tol, maxiter=maxiter),
            solve_C(cop, sync1, mop=mop, tol=tol, maxiter=maxiter),
        ]
        if sym:
            weighted_split2_templates = [
                solve_C(cop, dust2, mop=mop, tol=tol, maxiter=maxiter),
                solve_C(cop, sync2, mop=mop, tol=tol, maxiter=maxiter),
            ]
    elif fit_mode == "dust":
        dust1, dust2 = _require_template_pair(d1, d2, component_name="dust", fit_mode=fit_mode)
        split1_templates = [dust1]
        split2_templates = [dust2]
        weighted_split1_templates = [solve_C(cop, dust1, mop=mop, tol=tol, maxiter=maxiter)]
        if sym:
            weighted_split2_templates = [solve_C(cop, dust2, mop=mop, tol=tol, maxiter=maxiter)]
    else:
        sync1, sync2 = _require_template_pair(s1, s2, component_name="synchrotron", fit_mode=fit_mode)
        split1_templates = [sync1]
        split2_templates = [sync2]
        weighted_split1_templates = [solve_C(cop, sync1, mop=mop, tol=tol, maxiter=maxiter)]
        if sym:
            weighted_split2_templates = [solve_C(cop, sync2, mop=mop, tol=tol, maxiter=maxiter)]

    gmat = np.array(
        [
            [template2 @ weighted_template1 for weighted_template1 in weighted_split1_templates]
            for template2 in split2_templates
        ],
        dtype=np.float64,
    )
    bvec = np.array([template2 @ wy for template2 in split2_templates], dtype=np.float64)

    if sym:
        gmat2 = np.array(
            [
                [template1 @ weighted_template2 for weighted_template2 in weighted_split2_templates]
                for template1 in split1_templates
            ],
            dtype=np.float64,
        )
        bvec2 = np.array([template1 @ wy for template1 in split1_templates], dtype=np.float64)
        gmat = 0.5 * (gmat + gmat2)
        bvec = 0.5 * (bvec + bvec2)

    amps = np.linalg.solve(gmat, bvec)
    if fit_mode == "dust":
        return float(amps[0]), None, gmat, bvec
    if fit_mode == "synchrotron":
        return None, float(amps[0]), gmat, bvec
    return float(amps[0]), float(amps[1]), gmat, bvec


__all__ = [
    "make_mask_index",
    "pack_qu",
    "unpack_qu",
    "apply_N_blocks",
    "enforce_spd_blocks",
    "apply_P_qu",
    "make_C_operator",
    "make_Minv_operator",
    "solve_C",
    "FitMode",
    "cross_split_fit_operatorW",
]
