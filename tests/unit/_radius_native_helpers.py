"""Oracle and data generators for the native radius engine tests.

The oracle applies the contract directly: the canonical float64 value (a
per-feature sequential NumPy sum) classified against the thresholds, with the
uncertain band decided by :func:`radius_contract.exact_le`.
"""

from __future__ import annotations

import numpy as np

from mlxlearn._kernels.radius_contract import exact_le, thresholds
from mlxlearn._kernels.radius_result import RadiusCSR


def sequential_d2(train, queries, rows, cols) -> np.ndarray:
    """Per-feature sequential float64 sums of squared differences."""
    X = np.asarray(train, dtype=np.float32).astype(np.float64)
    Q = np.asarray(queries, dtype=np.float32).astype(np.float64)
    total = np.zeros(len(rows), dtype=np.float64)
    for feature in range(X.shape[1]):
        delta = Q[rows, feature] - X[cols, feature]
        total = total + delta * delta
    return total


def oracle(train, queries, R2, *, exclude_self=False, q_offset=0, integer_exact=False,
           thr=None) -> RadiusCSR:
    """Exact neighborhoods of every query (all pairs; keep n * m small)."""
    train = np.asarray(train, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    n, d = train.shape
    m = len(queries)
    if thr is None:
        thr = thresholds(d, R2, integer_exact=integer_exact)
    rows = np.repeat(np.arange(m, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), m)
    s = sequential_d2(train, queries, rows, cols)
    if thr.everything:
        member = np.ones(len(rows), dtype=bool)
    else:
        member = s <= thr.f64_in
        band = np.flatnonzero((s > thr.f64_in) & (s <= thr.f64_out))
        for i in band.tolist():
            member[i] = exact_le(queries[rows[i]], train[cols[i]], thr.R2)
    if exclude_self:
        member &= rows + q_offset != cols
    rows, cols, s = rows[member], cols[member], s[member]
    counts = np.bincount(rows, minlength=m)
    indptr = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    distances = np.sqrt(np.minimum(s, thr.R2))
    return RadiusCSR(indptr, cols, distances, "oracle")


def assert_same_csr(actual: RadiusCSR, expected: RadiusCSR, *, distances=True, context=""):
    np.testing.assert_array_equal(actual.indptr, expected.indptr, err_msg=f"indptr {context}")
    np.testing.assert_array_equal(actual.indices, expected.indices, err_msg=f"indices {context}")
    if distances:
        assert actual.distances is not None and expected.distances is not None
        np.testing.assert_array_equal(actual.distances.view(np.uint64),
                                      expected.distances.view(np.uint64),
                                      err_msg=f"distances {context}")


def generators(rng, n, m, d):
    """Named (train, queries, R2, integer_exact) cases covering the adversarial shapes."""
    cases = {}
    X = rng.normal(size=(n, d)).astype(np.float32)
    Q = rng.normal(size=(m, d)).astype(np.float32)
    cases["gauss"] = (X, Q, 0.9 * d, False)
    U = rng.random((n, d), dtype=np.float32)
    V = rng.random((m, d), dtype=np.float32)
    cases["uniform"] = (U, V, 0.05 * d, False)
    dup = np.repeat(rng.normal(size=(max(1, n // 4), d)), 4, axis=0)[:n].astype(np.float32)
    cases["duplicates"] = (dup, dup[: m], 0.5 * d, False)
    same = np.full((n, d), 0.375, dtype=np.float32)
    cases["identical"] = (same, np.vstack([same[:2], same[:2] + 1.0])[:m], 0.25, False)
    lattice = rng.integers(-3, 4, size=(n, d)).astype(np.float32)
    lattice_q = rng.integers(-3, 4, size=(m, d)).astype(np.float32)
    cases["lattice_exact"] = (lattice, lattice_q, 4.0, True)
    cases["lattice_band"] = (lattice, lattice_q, 4.0, False)
    scale = np.float32(2.0**-140)
    tiny = (rng.integers(-40, 41, size=(n, d)) * scale).astype(np.float32)
    tiny_q = (rng.integers(-40, 41, size=(m, d)) * scale).astype(np.float32)
    cases["subnormal"] = (tiny, tiny_q, float(25.0 * float(scale) ** 2 * d), False)
    offset = (rng.normal(size=(n, d)) + 1e6).astype(np.float32)
    offset_q = (rng.normal(size=(m, d)) + 1e6).astype(np.float32)
    cases["offset"] = (offset, offset_q, 0.5 * d, False)
    huge = (rng.normal(size=(n, d)) * 2.0**100).astype(np.float32)
    huge_q = (rng.normal(size=(m, d)) * 2.0**100).astype(np.float32)
    cases["huge"] = (huge, huge_q, float(d) * 2.0**200, False)
    return cases
