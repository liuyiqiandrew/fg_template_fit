from __future__ import annotations

import healpy as hp
import numpy as np
import numpy.typing as npt

from .core import (
    apply_P_qu,
    cross_split_fit_operatorW,
    make_C_operator,
    make_Minv_operator,
    make_mask_index,
    pack_qu,
)
from .filters import EllFilterSpec, MFilterInput

FloatArray = npt.NDArray[np.float64]


def _process_map(
    q: npt.NDArray[np.floating],
    u: npt.NDArray[np.floating],
    w_apod: npt.NDArray[np.floating],
    nside: int,
    fwhm_rad: float,
    ell_filter: EllFilterSpec,
    m_filter: MFilterInput | None,
    lmax: int,
) -> tuple[FloatArray, FloatArray]:
    """Apply the same processing chain used by the covariance operator.

    Parameters
    ----------
    q : ndarray of float
        Input Q map.
    u : ndarray of float
        Input U map.
    w_apod : ndarray of float
        Full-sky apodization weights.
    nside : int
        HEALPix map resolution.
    fwhm_rad : float
        Beam smoothing width in radians.
    ell_filter : EllFilterSpec
        Ell-space filter specification.
    m_filter : ndarray or MFilterSpec or None
        Optional m-space filter for azimuthal mode cuts.
    lmax : int
        Harmonic truncation.

    Returns
    -------
    q_out : ndarray of float64
        Processed Q map.
    u_out : ndarray of float64
        Processed U map.
    """
    q_out, u_out = apply_P_qu(
        q,
        u,
        nside=nside,
        fwhm_rad=fwhm_rad,
        ell_filter=ell_filter,
        m_filter=m_filter,
        lmax=lmax,
    )
    q_out *= w_apod
    u_out *= w_apod
    return q_out, u_out


def example_driver(nside: int = 64, seed: int = 0) -> tuple[float, float]:
    """Run a synthetic end-to-end example of the cross-split GLS fit.

    Parameters
    ----------
    nside : int, optional
        Working HEALPix ``nside``.
    seed : int, optional
        Random seed used for synthetic maps.

    Returns
    -------
    ad : float
        Estimated dust amplitude.
    a_s : float
        Estimated synchrotron amplitude.
    """
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)

    w_apod = np.ones(npix, dtype=np.float64)
    idx = make_mask_index(w_apod, thr=1e-3)

    qq_y = np.ones(npix, dtype=np.float64)
    uu_y = np.ones(npix, dtype=np.float64)
    qu_y = np.zeros(npix, dtype=np.float64)

    lmax = 3 * nside - 1
    ell_filter = EllFilterSpec(ell0=20.0, dell=10.0, mode="C2", kind="highpass")
    m_filter = None

    fwhm_rad = np.deg2rad(1.0)

    cop = make_C_operator(
        nside=nside,
        w_apod=w_apod,
        idx=idx,
        qq=qq_y,
        uu=uu_y,
        qu=qu_y,
        fwhm_rad=fwhm_rad,
        ell_filter=ell_filter,
        m_filter=m_filter,
        lmax=lmax,
        reg_eps=1e-6 * np.median(0.5 * (qq_y[idx] + uu_y[idx])),
    )
    mop = make_Minv_operator(w_apod, idx, qq_eff=qq_y, uu_eff=uu_y, qu_eff=qu_y)

    dust_q = rng.normal(0.0, 2.0, size=npix)
    dust_u = rng.normal(0.0, 2.0, size=npix)
    sync_q = rng.normal(0.0, 1.0, size=npix)
    sync_u = rng.normal(0.0, 1.0, size=npix)

    dsplit_q = rng.normal(0.0, 0.3, size=npix)
    dsplit_u = rng.normal(0.0, 0.3, size=npix)
    ssplit_q = rng.normal(0.0, 0.2, size=npix)
    ssplit_u = rng.normal(0.0, 0.2, size=npix)

    d1_q, d1_u = dust_q + dsplit_q, dust_u + dsplit_u
    d2_q, d2_u = dust_q - dsplit_q, dust_u - dsplit_u
    s1_q, s1_u = sync_q + ssplit_q, sync_u + ssplit_u
    s2_q, s2_u = sync_q - ssplit_q, sync_u - ssplit_u

    ad_true = 1.2
    as_true = -0.35
    ny_q = rng.normal(0.0, 0.3, size=npix)
    ny_u = rng.normal(0.0, 0.3, size=npix)

    y_q = ad_true * dust_q + as_true * sync_q + ny_q
    y_u = ad_true * dust_u + as_true * sync_u + ny_u

    y_q, y_u = _process_map(y_q, y_u, w_apod, nside, fwhm_rad, ell_filter, m_filter, lmax)
    d1_q, d1_u = _process_map(d1_q, d1_u, w_apod, nside, fwhm_rad, ell_filter, m_filter, lmax)
    d2_q, d2_u = _process_map(d2_q, d2_u, w_apod, nside, fwhm_rad, ell_filter, m_filter, lmax)
    s1_q, s1_u = _process_map(s1_q, s1_u, w_apod, nside, fwhm_rad, ell_filter, m_filter, lmax)
    s2_q, s2_u = _process_map(s2_q, s2_u, w_apod, nside, fwhm_rad, ell_filter, m_filter, lmax)

    y = pack_qu(y_q, y_u, idx)
    d1 = pack_qu(d1_q, d1_u, idx)
    d2 = pack_qu(d2_q, d2_u, idx)
    s1 = pack_qu(s1_q, s1_u, idx)
    s2 = pack_qu(s2_q, s2_u, idx)

    ad, a_s, _, _ = cross_split_fit_operatorW(
        y,
        d1,
        d2,
        s1,
        s2,
        cop,
        mop=mop,
        tol=1e-6,
        maxiter=200,
        sym=True,
    )
    return ad, a_s


__all__ = ["example_driver"]
