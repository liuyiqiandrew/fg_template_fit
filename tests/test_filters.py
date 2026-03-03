"""Unit tests for filter/window utilities used by harmonic processing."""

import numpy as np
import pytest

pytest.importorskip("healpy")

from fg_template_fit.filters import (
    EllFilterSpec,
    MFilterSpec,
    apply_m_window_inplace,
    make_ell_window,
    make_m_window,
    namaster_c_window,
    resolve_ell_filter,
    resolve_m_filter,
)


def test_namaster_c_window_endpoints_and_midpoint() -> None:
    """Verify C1/C2 windows hit expected boundary values and C2 midpoint."""
    x = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    c1 = namaster_c_window(x, mode="C1")
    c2 = namaster_c_window(x, mode="C2")

    assert np.allclose(c1[[0, 2]], [0.0, 1.0])
    assert np.allclose(c2[[0, 2]], [0.0, 1.0])
    assert np.isclose(c2[1], 0.5)


def test_make_ell_window_hard_highpass_and_lowpass() -> None:
    """Check hard ell high-pass and low-pass are complementary step windows."""
    w_hp = make_ell_window(lmax=5, ell0=2.0, dell=0.0, kind="highpass")
    w_lp = make_ell_window(lmax=5, ell0=2.0, dell=0.0, kind="lowpass")

    assert np.array_equal(w_hp, np.array([0, 0, 0, 1, 1, 1], dtype=np.float64))
    assert np.array_equal(w_lp, 1.0 - w_hp)


def test_make_ell_window_smooth_is_bounded_and_monotonic() -> None:
    """Ensure smooth ell high-pass is bounded in [0,1] and monotonic."""
    w = make_ell_window(lmax=20, ell0=10.0, dell=3.0, mode="C2", kind="highpass")
    assert np.all((w >= 0.0) & (w <= 1.0))
    assert np.all(np.diff(w) >= -1e-12)


def test_make_m_window_hard_highpass_and_lowpass() -> None:
    """Check hard m high-pass and low-pass are complementary step windows."""
    w_hp = make_m_window(lmax=5, m0=1.0, dm=0.0, kind="highpass")
    w_lp = make_m_window(lmax=5, m0=1.0, dm=0.0, kind="lowpass")

    assert np.array_equal(w_hp, np.array([0, 0, 1, 1, 1, 1], dtype=np.float64))
    assert np.array_equal(w_lp, 1.0 - w_hp)


def test_resolve_ell_filter_supports_spec_and_array() -> None:
    """Resolve ell filters from both spec objects and explicit arrays."""
    lmax = 8
    spec = EllFilterSpec(ell0=4.0, dell=2.0, mode="C2", kind="highpass")
    resolved_from_spec = resolve_ell_filter(lmax=lmax, ell_filter=spec)
    expected = make_ell_window(lmax=lmax, ell0=4.0, dell=2.0, mode="C2", kind="highpass")
    assert np.allclose(resolved_from_spec, expected)

    arr = np.linspace(0.0, 1.0, lmax + 1)
    resolved_from_array = resolve_ell_filter(lmax=lmax, ell_filter=arr)
    assert np.allclose(resolved_from_array, arr)


def test_resolve_ell_filter_rejects_ambiguous_or_bad_shape() -> None:
    """Raise on ambiguous ell args or wrong-sized explicit ell arrays."""
    lmax = 7
    with pytest.raises(ValueError, match="Pass only one"):
        resolve_ell_filter(
            lmax=lmax,
            ell_filter=EllFilterSpec(ell0=2.0),
            fl=np.ones(lmax + 1, dtype=np.float64),
        )

    with pytest.raises(ValueError, match="shape"):
        resolve_ell_filter(lmax=lmax, ell_filter=np.ones(lmax, dtype=np.float64))


def test_resolve_m_filter_supports_spec_and_array_and_none() -> None:
    """Resolve m filters from spec/array and pass through None unchanged."""
    lmax = 9
    spec = MFilterSpec(m0=5.0, dm=2.0, mode="C1", kind="highpass")
    resolved_from_spec = resolve_m_filter(lmax=lmax, m_filter=spec)
    expected = make_m_window(lmax=lmax, m0=5.0, dm=2.0, mode="C1", kind="highpass")
    assert np.allclose(resolved_from_spec, expected)

    arr = np.linspace(1.0, 0.0, lmax + 1)
    resolved_from_array = resolve_m_filter(lmax=lmax, m_filter=arr)
    assert np.allclose(resolved_from_array, arr)

    assert resolve_m_filter(lmax=lmax, m_filter=None) is None


def test_resolve_m_filter_rejects_bad_shape() -> None:
    """Raise on wrong-sized explicit m filter arrays."""
    with pytest.raises(ValueError, match="shape"):
        resolve_m_filter(lmax=4, m_filter=np.ones(3, dtype=np.float64))


def test_apply_m_window_inplace_scales_each_m_block() -> None:
    """Apply m-window in-place and verify each packed alm m-block scaling."""
    import healpy as hp

    lmax = 4
    alm = np.ones(hp.Alm.getsize(lmax), dtype=np.complex128)
    w_m = np.array([1.0, 0.5, 0.0, 0.25, 1.0], dtype=np.float64)

    apply_m_window_inplace(alm, w_m, lmax=lmax)

    for m in range(lmax + 1):
        i0 = hp.Alm.getidx(lmax, m, m)
        i1 = hp.Alm.getidx(lmax, lmax, m)
        assert np.allclose(alm[i0 : i1 + 1], w_m[m])
