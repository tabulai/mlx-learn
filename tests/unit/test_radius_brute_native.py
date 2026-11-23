"""Native brute force for small, latency and out-of-range calls."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

pytest.importorskip("mlxlearn._kernels._radius_cpu_native")

from mlxlearn._kernels import radius_cpu as rc  # noqa: E402
from mlxlearn._kernels import radius_host as rh  # noqa: E402
from mlxlearn._kernels.radius_contract import thresholds  # noqa: E402
from mlxlearn._kernels.radius_result import sort_rows  # noqa: E402

from ._radius_native_helpers import assert_same_csr, generators, oracle  # noqa: E402

ALL = os.cpu_count() or 1


@pytest.mark.parametrize("d", [1, 3, 8, 11, 16, 40])
def test_brute_matches_the_exact_oracle_on_every_generator(d, rng):
    for name, (X, Q, R2, integer_exact) in generators(rng, 500, 90, d).items():
        thr = thresholds(d, R2, integer_exact=integer_exact)
        expected = oracle(X, Q, R2, integer_exact=integer_exact)
        assert_same_csr(rc.brute_query(X, Q, thr).to_csr(), expected, context=f"{name} d={d}")
        assert_same_csr(rc.brute_query(X, Q, thr, sort=True).to_csr(), sort_rows(expected),
                        context=f"{name} d={d} sorted")


def test_brute_handles_float32_unsafe_rows(rng):
    d = 4
    X = rng.normal(size=(600, d)).astype(np.float32)
    X[::5] *= np.float32(2.0**-61)  # below the float32-safe range
    X[1::5] *= np.float32(2.0**57)  # above it
    X[2::5] *= np.float32(2.0**-140)  # subnormals
    Q = np.vstack([X[::3], rng.normal(size=(40, d)).astype(np.float32)])
    for R2 in (0.0, 2.0**-120, 1.0, 2.0**116, math.inf):
        thr = thresholds(d, R2)
        expected = oracle(X, Q, R2)
        assert_same_csr(rc.brute_query(X, Q, thr).to_csr(), expected, context=f"R2={R2}")
        # The NumPy engine agrees on membership.
        host = rh.brute_host(rh.BruteState(X), X, Q, thr, exclude_self=False, need_distance=True)
        np.testing.assert_array_equal(host.indptr, expected.indptr)
        np.testing.assert_array_equal(host.indices, expected.indices)


def test_brute_block_mode_equals_query_mode(rng):
    d = 6
    X = rng.normal(size=(50_000, d)).astype(np.float32)
    Q = rng.normal(size=(9, d)).astype(np.float32)  # fewer than 4 * threads: block mode
    thr = thresholds(d, 2.0)
    expected = oracle(X, Q, 2.0)
    for threads in (1, 2, ALL):
        for sort in (False, True):
            result = rc.brute_query(X, Q, thr, threads=threads, sort=sort).to_csr()
            assert_same_csr(result, sort_rows(expected) if sort else expected,
                            context=f"threads={threads} sort={sort}")
    without = rc.brute_query(X, Q, thr, threads=ALL, need_distance=False).to_csr()
    np.testing.assert_array_equal(without.indices, expected.indices)


def test_brute_self_queries_with_offset_and_duplicates(rng):
    base = rng.normal(size=(300, 3)).astype(np.float32)
    X = np.vstack([base, base])
    for start, stop in ((0, 600), (250, 420)):
        Q = X[start:stop]
        thr = thresholds(3, 0.4)
        result = rc.brute_query(X, Q, thr, exclude_self=True, q_offset=start).to_csr()
        assert_same_csr(result, oracle(X, Q, 0.4, exclude_self=True, q_offset=start))


def test_brute_thread_invariance_and_latency_shapes(rng):
    for n, m, d in ((10_000, 1, 3), (10_000, 16, 64), (2000, 3000, 5)):
        X = rng.normal(size=(n, d)).astype(np.float32)
        Q = rng.normal(size=(m, d)).astype(np.float32)
        thr = thresholds(d, 0.9 * d)
        reference = rc.brute_query(X, Q, thr, threads=1).to_csr()
        for threads in (2, ALL):
            assert_same_csr(rc.brute_query(X, Q, thr, threads=threads).to_csr(), reference)


def test_brute_with_empty_inputs(rng):
    thr = thresholds(2, 1.0)
    X = rng.normal(size=(5, 2)).astype(np.float32)
    assert rc.brute_query(X, np.empty((0, 2), np.float32), thr).sizes() == (0, 0)
    result = rc.brute_query(np.empty((0, 2), np.float32), X, thr)
    assert result.sizes() == (5, 0)
    assert result.to_csr().indptr.tolist() == [0] * 6


@pytest.mark.parametrize("d", [1, 2, 3, 4, 7, 16, 64])
def test_brute_float32_screen_does_not_change_the_output(d, rng):
    X = rng.normal(size=(3000, d)).astype(np.float32)
    X[::7] *= np.float32(2.0**-130)
    X[1::7] *= np.float32(2.0**40)
    Q = np.vstack([X[:40], rng.normal(size=(60, d)).astype(np.float32)])
    for R2 in (0.0, 0.3 * d, 2.0 * d, 2.0**100, 2.0**125):
        thr = thresholds(d, R2)
        for need_distance in (True, False):
            screened = rc.brute_query(X, Q, thr, need_distance=need_distance).to_csr()
            plain = rc.brute_query(X, Q, thr, need_distance=need_distance, screen=False).to_csr()
            assert_same_csr(screened, plain, distances=need_distance)
        assert_same_csr(rc.brute_query(X, Q, thr).to_csr(), oracle(X, Q, R2))


@pytest.mark.parametrize("need_distance", [False, True])
def test_wide_brute_prefix_screen_still_checks_the_suffix(need_distance):
    # Most rows fail on their first feature, including all eight probe rows.
    # The exceptional rows share a prefix and differ only in the final lane.
    X = np.full((513, 65), 2.0, dtype=np.float32)
    X[[1, 2, 3, 4, -1]] = 0
    X[1, -1] = 0.5
    X[2, -1] = 1.0
    X[3, -1] = np.nextafter(np.float32(1.0), np.float32(2.0))
    X[4, 0], X[4, -1] = 1.0, 2.0**-40  # exactly outside, rounds to 1 in float64
    Q = np.zeros((1, X.shape[1]), dtype=np.float32)
    thr = thresholds(X.shape[1], 1.0)
    result = rc.brute_query(X, Q, thr, need_distance=need_distance).to_csr()
    expected = oracle(X, Q, 1.0)
    np.testing.assert_array_equal(expected.indices, [1, 2, 512])
    assert_same_csr(result, expected, distances=need_distance)


def test_thread_policy():
    from mlxlearn._common.config import tuning_context

    cores = rc.cpu_count()
    assert rc.resolve_threads(None) == cores
    assert rc.resolve_threads(-1) == cores
    assert rc.resolve_threads(1) == 1
    assert rc.resolve_threads(4) == 4
    assert rc.resolve_threads(-2) == max(1, cores - 1)
    assert rc.resolve_threads(-10_000) == 1
    with tuning_context(radius_threads=3):
        assert rc.resolve_threads(None) == 3
        assert rc.resolve_threads(8) == 3
