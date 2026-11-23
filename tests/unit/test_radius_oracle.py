"""Validation of the exact radius oracle and generators in ``tests/_radius_oracle.py``.

The oracle decides the radius suites, so it is checked against references that
share none of its shortcuts: :class:`fractions.Fraction` arithmetic on tiny
inputs, its own pure-integer path on every generator, and scikit-learn's
float64 brute force away from the boundary.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors

from tests import _radius_oracle as oracle

FLT_MAX = float(np.finfo(np.float32).max)
TINY32 = 2.0**-149


def _same(a, b) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def _csr_rows(indptr):
    return np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))


# ----------------------------------------------------------------------------------
# against fractions on tiny inputs
# ----------------------------------------------------------------------------------


def _tiny_problem(rng, style):
    n, nq, d = int(rng.integers(1, 11)), int(rng.integers(1, 6)), int(rng.integers(1, 6))
    if style == "normal":
        X = rng.normal(size=(n, d))
    elif style == "lattice":
        X = rng.integers(-3, 4, size=(n, d)).astype(np.float64)
    elif style == "subnormal":
        X = rng.integers(-4, 5, size=(n, d)) * TINY32
    elif style == "huge":
        X = rng.choice([-1.0, 1.0], size=(n, d)) * FLT_MAX * rng.uniform(0.5, 1.0, size=(n, d))
    elif style == "mixed":
        mant = rng.integers(2**23, 2**24, size=(n, d)).astype(np.float64)
        with np.errstate(under="ignore"):
            X = mant * 2.0 ** (rng.integers(-149, 104, size=(n, d)) - 23)
    else:  # duplicates and one-ulp neighbors
        X = np.repeat(rng.normal(size=((n + 1) // 2, d)), 2, axis=0)[:n]
    X = X.astype(np.float32)
    Q = X[rng.integers(0, n, size=nq)].copy()
    nudge = rng.random(Q.shape) < 0.3
    Q[nudge] = np.nextafter(Q[nudge], np.float32(np.inf))
    return X, Q


def _tiny_radii(X, Q, rng):
    radii = [0.0, math.inf, TINY32, 1e200, float(rng.uniform(0, 3))]
    for _ in range(3):
        i, j = int(rng.integers(len(Q))), int(rng.integers(len(X)))
        S = oracle.oracle_distances(X, Q, [i], [j], kind="fraction")[0]
        exact = oracle.radius_hitting(float(S)) if Fraction(float(S)) == S else None
        if exact is not None:
            radii.append(exact)
        if S > 0:
            radii.append(oracle.radius_for_r2(S, side="below"))
            radii.append(oracle.radius_for_r2(S, side="at_or_above"))
    return radii


def test_oracle_matches_fractions_on_tiny_inputs():
    rng = np.random.default_rng(9101)
    styles = ("normal", "lattice", "subnormal", "huge", "mixed", "duplicates")
    checks = 0
    for trial in range(120):
        X, Q = _tiny_problem(rng, styles[trial % len(styles)])
        for radius in _tiny_radii(X, Q, rng):
            reference = oracle.fraction_radius(X, Q, radius)
            for method in ("auto", "exact", "band"):
                for use_grid in (True, False):
                    got = oracle.oracle_radius(X, Q, radius, method=method, use_grid=use_grid)
                    assert _same(got, reference), (trial, radius, method, use_grid)
            reference = oracle.fraction_radius(X, None, radius, exclude_self=True)
            for method in ("exact", "band"):
                got = oracle.oracle_radius(X, None, radius, exclude_self=True, method=method)
                assert _same(got, reference), (trial, radius, method, "self")
            checks += 1
    assert checks > 900


def test_oracle_distances_are_exact_and_correctly_rounded():
    rng = np.random.default_rng(9102)
    for case_id in ("lattice-z3-sqrt3", "offset-1e12-d3", "tiny-subnormal-r5", "huge-r6e38",
                    "gauss-d16", "mixed-exponents", "near-min-normal", "perm-tie"):
        case = oracle.case_by_id(case_id)
        rows = rng.integers(0, len(case.Q), size=60)
        cols = rng.integers(0, len(case.X), size=60)
        X32, Q32 = oracle.to_float32(case.X), oracle.to_float32(case.Q)
        reference = [sum((Fraction(float(a)) - Fraction(float(b))) ** 2
                         for a, b in zip(Q32[r], X32[c], strict=True))
                     for r, c in zip(rows.tolist(), cols.tolist(), strict=True)]
        floats = oracle.oracle_distances(case.X, case.Q, rows, cols)
        assert floats.dtype == np.float64
        assert floats.tolist() == [float(S) for S in reference], case_id  # float(Fraction) rounds correctly
        assert oracle.oracle_distances(case.X, case.Q, rows, cols, kind="fraction") == reference
        scaled = oracle.oracle_distances(case.X, case.Q, rows, cols, kind="scaled")
        assert scaled == [int(S * (1 << 298)) for S in reference]


# ----------------------------------------------------------------------------------
# the oracle's own paths agree on every generator
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("case", oracle.all_cases(), ids=lambda c: c.id)
def test_oracle_paths_agree_on_every_case(case):
    X32 = oracle.to_float32(case.X)
    n, d = X32.shape
    # Python-integer evaluation of every pair is slow; check a prefix of queries.
    k = max(1, min(len(case.Q), 120_000 // (n * (d + 24))))
    Q = oracle.to_float32(case.Q)[:k]
    pure = oracle.oracle_csr(case.X, Q, case.radius, method="exact", use_grid=False)
    for method in ("auto", "band", "exact"):
        got = oracle.oracle_csr(case.X, Q, case.radius, method=method)
        assert np.array_equal(got[0], pure[0]) and np.array_equal(got[1], pure[1]), method
    full_auto = oracle.oracle_csr(case.X, case.Q, case.radius)
    full_band = oracle.oracle_csr(case.X, case.Q, case.radius, method="band")
    assert np.array_equal(full_auto[0], full_band[0]) and np.array_equal(full_auto[1], full_band[1])
    self_auto = oracle.oracle_csr(case.X, None, case.radius, exclude_self=True)
    self_band = oracle.oracle_csr(case.X, None, case.radius, exclude_self=True, method="band")
    assert np.array_equal(self_auto[0], self_band[0]) and np.array_equal(self_auto[1], self_band[1])
    # Rows ascend and X=None drops exactly the diagonal.
    rows = _csr_rows(self_auto[0])
    assert not np.any(rows == self_auto[1])
    with_self = oracle.oracle_csr(case.X, None, case.radius)
    assert with_self[1].size == self_auto[1].size + n
    for indptr, indices in (full_auto, self_auto):
        r = _csr_rows(indptr)
        same_row = r[1:] == r[:-1]
        assert np.all(indices[1:][same_row] > indices[:-1][same_row])


# ----------------------------------------------------------------------------------
# against scikit-learn away from the boundary
# ----------------------------------------------------------------------------------


EASY = ["gauss-d1", "gauss-d3", "gauss-d16", "gauss-d64", "blobs-d2", "blobs-d8", "hubs-d2",
        "hubs-d8", "label-sorted", "dup-r0.75"]


@pytest.mark.parametrize("case_id", EASY)
@pytest.mark.parametrize("algorithm", ["brute", "kd_tree"])
def test_oracle_agrees_with_sklearn_away_from_the_boundary(case_id, algorithm):
    case = oracle.case_by_id(case_id)
    X32, Q32 = oracle.to_float32(case.X), oracle.to_float32(case.Q)
    X64, Q64 = X32.astype(np.float64), Q32.astype(np.float64)
    R2 = oracle.squared_radius(case.radius)
    sk = SkNearestNeighbors(radius=case.radius, algorithm=algorithm).fit(X64)
    _, sk_ind = sk.radius_neighbors(Q64)
    indptr, indices = oracle.oracle_csr(case.X, case.Q, case.radius)
    d = X32.shape[1]
    norms_q = np.einsum("ij,ij->i", Q64, Q64)
    norms_x = np.einsum("ij,ij->i", X64, X64)
    disagreements = 0
    for i in range(len(Q32)):
        mine = set(indices[indptr[i]:indptr[i + 1]].tolist())
        theirs = set(np.asarray(sk_ind[i]).tolist())
        for j in mine ^ theirs:
            S = oracle.oracle_distances(case.X, case.Q, [i], [j])[0]
            # sklearn's float64 error: the Gram expansion (brute) is relative to
            # the norms, the direct differences of the tree are relative to S.
            scale = norms_q[i] + norms_x[j] if algorithm == "brute" else S
            assert abs(S - R2) <= 64 * (d + 2) * 2.0**-52 * (scale + R2), (i, j, S, R2)
            disagreements += 1
    assert disagreements <= max(2, indices.size // 1000)


# ----------------------------------------------------------------------------------
# generators
# ----------------------------------------------------------------------------------


def test_generators_are_deterministic_and_well_formed():
    first = oracle.all_cases()
    ids = [case.id for case in first]
    assert len(ids) == len(set(ids)) >= 50
    rebuilt = [case for family in oracle.CASE_FAMILIES.values() for case in family()]
    assert [case.id for case in rebuilt] == ids
    for a, b in zip(first, rebuilt, strict=True):
        assert np.array_equal(a.X, b.X) and np.array_equal(a.Q, b.Q), a.id
        assert a.radius == b.radius or (math.isnan(a.radius) and math.isnan(b.radius)), a.id
        X32, Q32 = oracle.to_float32(a.X), oracle.to_float32(a.Q)  # finite after the cast
        assert X32.ndim == Q32.ndim == 2 and X32.shape[1] == Q32.shape[1], a.id
        assert len(X32) >= 1 and len(Q32) >= 1 and len(X32) * len(Q32) <= 400_000, a.id
        assert a.radius >= 0 and not math.isnan(a.radius), a.id
        if a.y is not None:
            assert len(a.y) == len(X32)


def _pairs_with_S(case, target: Fraction):
    """(row, col) pairs whose exact S equals ``target``."""
    indptr, indices = oracle.oracle_csr(case.X, case.Q, math.inf)
    rows = _csr_rows(indptr)
    # S == target implies fl(S) == fl(target): preselect in float64, confirm exactly.
    near = oracle.oracle_distances(case.X, case.Q, rows, indices) == float(target)
    rows, indices = rows[near], indices[near]
    S = oracle.oracle_distances(case.X, case.Q, rows, indices, kind="scaled")
    want = target * (1 << 298)
    assert want.denominator == 1
    hit = np.array([s == want.numerator for s in S], dtype=bool)
    return rows[hit], indices[hit]


def _members(case):
    indptr, indices = oracle.oracle_csr(case.X, case.Q, case.radius)
    return set(zip(_csr_rows(indptr).tolist(), indices.tolist(), strict=True))


@pytest.mark.parametrize("case_id, shell, inside", [
    ("lattice-z3-sqrt3", 3, False), ("lattice-z3-sqrt6", 6, False),
    ("lattice-z3-sqrt2", 2, True), ("lattice-z2-sqrt5", 5, True),
    ("lattice-z2-r2", 4, True), ("lattice-z4-r3", 9, True), ("lattice-z8-r2", 4, True),
    ("lattice-2^23-r5", 25, True), ("lattice-dyadic-r5", Fraction(25, 2**24), True),
    ("tiny-subnormal-r5", Fraction(25, 2**298), True), ("r2-2^-298", Fraction(1, 2**298), True),
    ("offset-1e12-tie", 9 * 65536**2, True), ("perm-tie", 2**56 + 32, True),
])
def test_shell_cases_have_exact_ties_on_the_intended_side(case_id, shell, inside):
    case = oracle.case_by_id(case_id)
    rows, cols = _pairs_with_S(case, Fraction(shell))
    assert rows.size >= 2, case_id
    members = _members(case)
    assert all(((r, c) in members) is inside for r, c in zip(rows.tolist(), cols.tolist(), strict=True))
    R2 = oracle.squared_radius(case.radius)
    assert (Fraction(R2) >= Fraction(shell)) is inside


@pytest.mark.parametrize("case_id", ["perm-falsepos", "perm-falseneg", "perm-tie",
                                     "perm-mixed-d16-below"])
def test_permuted_cases_defeat_naive_float64_comparisons(case_id):
    case = oracle.case_by_id(case_id)
    X, Q = oracle.to_float32(case.X).astype(np.float64), oracle.to_float32(case.Q).astype(np.float64)
    R2 = oracle.squared_radius(case.radius)
    squares = (Q[:, None, :] - X[None, :, :]) ** 2
    naive = np.cumsum(squares, axis=2)[..., -1] <= R2  # sequential float64 sum
    truth = np.zeros_like(naive)
    indptr, indices = oracle.oracle_csr(case.X, case.Q, case.radius)
    truth[_csr_rows(indptr), indices] = True
    assert np.count_nonzero(naive != truth) >= 5


def test_r2_sweep_covers_the_intended_squared_radii():
    expected = {"r2-zero": 0.0, "r2-2^-120": 2.0**-120, "r2-2^-298": 2.0**-298,
                "r2-subnormal": 1e-160 * 1e-160, "r2-underflow": 0.0, "r2-normal": 2.25,
                "r2-huge": 1e30 * 1e30, "r2-1e308": 1e154 * 1e154, "r2-overflow": math.inf,
                "r2-inf": math.inf}
    for case_id, R2 in expected.items():
        assert oracle.squared_radius(oracle.case_by_id(case_id).radius) == R2, case_id
    assert 0 < 1e-160 * 1e-160 < 2.0**-1022  # a subnormal R2
    assert oracle.case_by_id("r2-underflow").radius > 0


def test_extreme_cases_reach_the_float32_range_limits():
    tiny = oracle.to_float32(oracle.case_by_id("tiny-subnormal-r5").X)
    assert np.any((tiny != 0) & (np.abs(tiny) < 2.0**-126))
    huge = oracle.case_by_id("huge-r6e38")
    X32 = oracle.to_float32(huge.X)
    assert np.abs(X32).max() == FLT_MAX
    S = oracle.oracle_distances(huge.X, huge.Q, [0], [1])[0]
    assert S > FLT_MAX**2  # differences beyond the float32 range
    mixed = oracle.to_float32(oracle.case_by_id("mixed-exponents").X)
    magnitude = np.abs(mixed[mixed != 0])
    assert magnitude.min() < 2.0**-60 and magnitude.max() > 2.0**56
    for tag, offset in (("1e3", 1e3), ("1e6", 1e6), ("1e12", 1e12)):
        X = oracle.case_by_id(f"offset-{tag}-d3").X
        assert X.dtype == np.float64 and np.all(np.abs(X - offset) < 0.01 * offset)
    labels = oracle.case_by_id("label-sorted").y
    assert np.all(np.diff(labels) >= 0)
    for case_id in ("hubs-d2", "hubs-d8"):
        indptr, _ = oracle.oracle_csr(*oracle.case_by_id(case_id)[:3])
        counts = np.diff(indptr)
        assert counts.max() >= 10 * max(1, np.median(counts)) and np.any(counts == 0), case_id


def test_sqrt_of_rounded_square_is_the_radius():
    # The lemma behind dist <= radius: sqrt(fl64(r * r)) == r whenever r * r is
    # a normal float64 (it can fail once r * r is subnormal, where only S == 0
    # pairs can be members anyway).
    rng = np.random.default_rng(9103)
    r = np.exp2(rng.uniform(-500, 500, size=200_000)) * rng.uniform(1, 2, size=200_000)
    r = np.concatenate([r, [1.0, 2.0**-500, 2.0**500, 1.5, math.sqrt(2), math.sqrt(3)]])
    r = np.concatenate([r, np.nextafter(r, np.inf), np.nextafter(r, 0)])
    assert np.all(np.sqrt(r * r) == r)


def test_oracle_rejects_invalid_inputs():
    X = np.zeros((3, 2), dtype=np.float32)
    for bad in (math.nan, -1.0, -5e-324):
        with pytest.raises(ValueError):
            oracle.oracle_radius(X, X, bad)
    with pytest.raises(ValueError):
        oracle.oracle_radius(np.array([[np.inf, 0.0]]), X, 1.0)
    with pytest.raises(ValueError):
        oracle.oracle_radius(np.array([[1e39, 0.0]]), X, 1.0)  # beyond float32
    with pytest.raises(ValueError):
        oracle.oracle_radius(X, X[:2], 1.0, exclude_self=True)
    with pytest.raises(ValueError):
        oracle.oracle_radius(X, np.zeros((2, 3)), 1.0)
    assert _same(oracle.oracle_radius(X, X, -0.0), [np.arange(3)] * 3)
