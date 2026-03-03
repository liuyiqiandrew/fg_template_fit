"""Unit tests for core masked-domain operators and GLS fitting helpers."""

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

pytest.importorskip("healpy")

import fg_template_fit.core as core


def test_make_mask_index_thresholding() -> None:
    """Keep only indices with mask weights strictly above threshold."""
    w = np.array([0.0, 1e-7, 2e-6, 0.5], dtype=np.float64)
    idx = core.make_mask_index(w, thr=1e-6)
    assert np.array_equal(idx, np.array([2, 3], dtype=np.int64))


def test_pack_unpack_round_trip_on_masked_domain() -> None:
    """Round-trip packed QU vectors and ensure outside-mask zeros on unpack."""
    npix = 8
    idx = np.array([1, 3, 7], dtype=np.int64)
    q = np.arange(npix, dtype=np.float64)
    u = np.arange(npix, dtype=np.float64) + 10.0

    packed = core.pack_qu(q, u, idx)
    q2, u2 = core.unpack_qu(packed, npix=npix, idx=idx)

    assert np.array_equal(q2[idx], q[idx])
    assert np.array_equal(u2[idx], u[idx])

    outside = np.setdiff1d(np.arange(npix), idx)
    assert np.array_equal(q2[outside], np.zeros_like(outside, dtype=np.float64))
    assert np.array_equal(u2[outside], np.zeros_like(outside, dtype=np.float64))


def test_apply_N_blocks_matches_manual_formula() -> None:
    """Match per-pixel 2x2 block multiplication against manual expressions."""
    q = np.array([1.0, 2.0], dtype=np.float64)
    u = np.array([3.0, 4.0], dtype=np.float64)
    qq = np.array([2.0, 5.0], dtype=np.float64)
    uu = np.array([7.0, 11.0], dtype=np.float64)
    qu = np.array([13.0, 17.0], dtype=np.float64)

    q2, u2 = core.apply_N_blocks(q, u, qq=qq, uu=uu, qu=qu)

    assert np.allclose(q2, qq * q + qu * u)
    assert np.allclose(u2, qu * q + uu * u)


def test_enforce_spd_blocks_applies_ridge_to_bad_pixels_only() -> None:
    """Add ridge only on invalid SPD blocks while leaving valid block intact."""
    qq = np.array([1.0, 0.1, -0.2], dtype=np.float64)
    uu = np.array([1.0, 0.1, 1.0], dtype=np.float64)
    qu = np.array([0.0, 0.2, 0.0], dtype=np.float64)

    qq2, uu2, qu2 = core.enforce_spd_blocks(qq, uu, qu, ridge_frac=1.0, eps=0.0)

    # Pixel 0 is good and sets the median scale to 1.0, so ridge=1.0.
    assert np.isclose(qq2[0], qq[0])
    assert np.isclose(uu2[0], uu[0])
    assert np.isclose(qq2[1], qq[1] + 1.0)
    assert np.isclose(uu2[1], uu[1] + 1.0)
    assert np.isclose(qq2[2], qq[2] + 1.0)
    assert np.isclose(uu2[2], uu[2] + 1.0)
    assert np.array_equal(qu2, qu)


def test_make_Minv_operator_matches_per_pixel_inverse_action() -> None:
    """Match preconditioner output against direct per-pixel block inversion."""
    w_apod = np.array([1.0, 2.0], dtype=np.float64)
    idx = np.array([0, 1], dtype=np.int64)
    qq_eff = np.array([2.0, 3.0], dtype=np.float64)
    uu_eff = np.array([4.0, 5.0], dtype=np.float64)
    qu_eff = np.array([1.0, 2.0], dtype=np.float64)

    op = core.make_Minv_operator(w_apod, idx, qq_eff=qq_eff, uu_eff=uu_eff, qu_eff=qu_eff, ridge=0.0)
    v = np.array([0.7, -1.2, 2.3, 0.9], dtype=np.float64)

    out = op @ v

    n = idx.size
    q = v[:n]
    u = v[n:]
    expected_q = np.zeros(n, dtype=np.float64)
    expected_u = np.zeros(n, dtype=np.float64)
    for i in range(n):
        scale = w_apod[idx[i]] ** 2
        block = np.array(
            [
                [scale * qq_eff[idx[i]], scale * qu_eff[idx[i]]],
                [scale * qu_eff[idx[i]], scale * uu_eff[idx[i]]],
            ],
            dtype=np.float64,
        )
        inv = np.linalg.inv(block)
        expected_q[i], expected_u[i] = inv @ np.array([q[i], u[i]], dtype=np.float64)

    expected = np.concatenate([expected_q, expected_u])
    assert np.allclose(out, expected)


def test_make_C_operator_matches_expected_when_P_is_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate C matvec against manual formula when P is patched to identity."""
    monkeypatch.setattr(
        core,
        "apply_P_qu",
        lambda q, u, nside, fwhm_rad, fl=None, ell_filter=None, m_filter=None, lmax=None: (
            q.astype(np.float64, copy=False),
            u.astype(np.float64, copy=False),
        ),
    )

    nside = 1
    npix = 12
    idx = np.array([0, 2, 5], dtype=np.int64)
    w = np.linspace(0.5, 1.5, npix, dtype=np.float64)
    qq = np.full(npix, 2.0, dtype=np.float64)
    uu = np.full(npix, 3.0, dtype=np.float64)
    qu = np.full(npix, 0.25, dtype=np.float64)
    reg_eps = 0.05

    cop = core.make_C_operator(
        nside=nside,
        w_apod=w,
        idx=idx,
        qq=qq,
        uu=uu,
        qu=qu,
        fwhm_rad=None,
        lmax=3 * nside - 1,
        reg_eps=reg_eps,
    )

    v = np.array([1.0, -0.5, 0.75, 0.2, -1.5, 0.3], dtype=np.float64)
    out = cop @ v

    q_full, u_full = core.unpack_qu(v, npix=npix, idx=idx)
    q_full *= w
    u_full *= w
    q_full, u_full = core.apply_N_blocks(q_full, u_full, qq=qq, uu=uu, qu=qu)
    q_full *= w
    u_full *= w
    expected = core.pack_qu(q_full, u_full, idx=idx) + reg_eps * v

    assert np.allclose(out, expected)


def test_solve_C_solves_identity_operator() -> None:
    """Solve identity linear system and recover rhs exactly."""
    n = 4
    eye = LinearOperator((n, n), matvec=lambda x: x, dtype=np.float64)
    b = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)

    x = core.solve_C(eye, b, tol=1e-12, maxiter=20)
    assert np.allclose(x, b)


def test_solve_C_raises_when_cg_does_not_converge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise RuntimeError when underlying CG reports non-zero info code."""
    def fake_cg(*args, **kwargs):  # type: ignore[no-untyped-def]
        b = args[1]
        return np.zeros_like(b), 1

    monkeypatch.setattr(core, "cg", fake_cg)

    n = 3
    eye = LinearOperator((n, n), matvec=lambda x: x, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)

    with pytest.raises(RuntimeError, match="did not converge"):
        core.solve_C(eye, b)


def test_cross_split_fit_operatorW_recovers_known_amplitudes_with_identity_covariance() -> None:
    """Recover known template amplitudes in a noiseless identity-covariance case."""
    n = 3
    d = np.array([1.0, -0.4, 2.2, 0.1, 0.7, -1.5], dtype=np.float64)
    s = np.array([-0.3, 1.1, 0.2, 2.0, -0.8, 0.4], dtype=np.float64)
    ad_true = 1.7
    as_true = -0.45
    y = ad_true * d + as_true * s

    eye = LinearOperator((2 * n, 2 * n), matvec=lambda x: x, dtype=np.float64)
    ad, a_s, gmat, bvec = core.cross_split_fit_operatorW(
        y=y,
        d1=d,
        d2=d,
        s1=s,
        s2=s,
        cop=eye,
        mop=None,
        tol=1e-12,
        maxiter=30,
        sym=True,
    )

    assert np.allclose([ad, a_s], [ad_true, as_true], atol=1e-10)
    assert gmat.shape == (2, 2)
    assert bvec.shape == (2,)
