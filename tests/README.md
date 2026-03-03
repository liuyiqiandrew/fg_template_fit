# Test Coverage Notes

This test suite focuses on deterministic unit tests for `core` and `filters`.

## `tests/test_filters.py`

What is validated:

- NaMaster transition windows (`C1`, `C2`) hit expected endpoint behavior.
- `ell` windows:
  - hard high-pass/low-pass behavior,
  - smooth windows stay in `[0, 1]`,
  - smooth high-pass windows are monotonic non-decreasing.
- `m` windows:
  - hard high-pass/low-pass behavior.
- Filter resolvers:
  - accept spec objects (`EllFilterSpec`, `MFilterSpec`),
  - accept explicit arrays,
  - return `None` when no filter is provided,
  - reject incompatible/ambiguous arguments and bad shapes.
- `apply_m_window_inplace` scales each packed-`alm` block by the expected
  per-`m` factor.

## `tests/test_core.py`

What is validated:

- Mask-domain indexing, packing, and unpacking behavior.
- Per-pixel 2x2 noise-block multiplication (`apply_N_blocks`).
- SPD stabilization ridge application on bad blocks (`enforce_spd_blocks`).
- Block-Jacobi preconditioner action (`make_Minv_operator`) against direct
  per-pixel matrix inversion.
- Covariance operator construction (`make_C_operator`) against a manually
  computed reference when `P` is patched to identity.
- CG solver success path and failure path (`solve_C`).
- Cross-split amplitude recovery (`cross_split_fit_operatorW`) for a case with
  known true coefficients and identity covariance.

## Dependency Note

`core` and `filters` import `healpy`, so test modules use
`pytest.importorskip("healpy")`. If `healpy` is unavailable, these tests are
reported as skipped.
