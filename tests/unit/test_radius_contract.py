"""Property tests of the radius-membership contract (``_kernels/radius_contract``).

Every assertion about membership is checked with exact arithmetic: Python
integers on the ``2**-149`` grid that holds every float32 value, or
:class:`fractions.Fraction`. Nothing here trusts a float64 comparison.

The threshold-soundness test is the heart of the file. For each dimension it
builds groups of pairs that share one exact ``S`` (signed permutations of one
difference vector), computes float64 values in four summation orders
(sequential, reversed, einsum and pairwise ``np.sum``), and classifies them
against ``thresholds(d, R2)`` for ``R2`` placed on and around ``S``: exactly on
it when ``S`` is a float64 value, one ulp either side, and at multiples of the
error bound. Crafted groups (one huge term plus many small odd squares) drive
the sequential rounding error to most of the proven bound, so a band that was
too narrow would be caught.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from mlxlearn._kernels import radius_contract as rc

DIMS = (1, 2, 3, 8, 16, 64, 784)
PAIRS_PER_DIM = 200_000
FLT_MAX = float(np.finfo(np.float32).max)
TINY32 = 2.0**-149
U = 2.0**-53


# ----------------------------------------------------------------------------------
# exact helpers (independent of the module under test)
# ----------------------------------------------------------------------------------


def _scaled(values) -> list[int]:
    """float32 values as integers on the 2**-149 grid."""
    scaled = np.asarray(values, dtype=np.float32).astype(np.float64) * 2.0**149
    return [int(v) for v in scaled.tolist()]


def _exact_scaled_sum(q, x) -> int:
    """S(q, x) * 2**298 as a Python integer."""
    diff = [a - b for a, b in zip(_scaled(q), _scaled(x), strict=True)]
    return sum(d * d for d in diff)


def _floor_scaled(R2: float) -> int | None:
    if math.isinf(R2):
        return None
    numerator, denominator = R2.as_integer_ratio()
    return (numerator << 298) // denominator


def _exact_le(S_scaled: int, R2: float) -> bool:
    bound = _floor_scaled(R2)
    return True if bound is None else S_scaled <= bound


def _fraction_S(q, x) -> Fraction:
    q = np.asarray(q, dtype=np.float32).tolist()
    x = np.asarray(x, dtype=np.float32).tolist()
    return sum(((Fraction(a) - Fraction(b)) ** 2 for a, b in zip(q, x, strict=True)), Fraction(0))


def _f32_random(rng, size, low_exp, high_exp):
    """float32 values with full 24-bit significands, exponents in [low, high]."""
    mantissa = rng.integers(2**23, 2**24, size=size).astype(np.float64)
    exponent = rng.integers(low_exp, high_exp + 1, size=size)
    sign = rng.choice([-1.0, 1.0], size=size)
    with np.errstate(under="ignore", over="ignore"):
        return (sign * mantissa * 2.0 ** (exponent - 23)).astype(np.float32)


# ----------------------------------------------------------------------------------
# thresholds
# ----------------------------------------------------------------------------------

R2_SAMPLES = (0.0, 5e-324, 2.0**-1022, 2.0**-298, 2.0**-120, 1e-40, 0.5, 1.0, 3.0,
              2.0**56 + 32, 1e38, 1e76, 1e300, 1.7976931348623157e308, math.inf)


@pytest.mark.parametrize("d", DIMS)
def test_thresholds_are_ordered_and_bracket_R2(d):
    for R2 in R2_SAMPLES:
        thr = rc.thresholds(d, R2)
        assert thr.R2 == R2 and thr.d == d and thr.integer_exact is False
        assert 0.0 <= thr.bulk_below <= thr.f64_in <= R2 <= thr.f64_out <= thr.prune_above
        if R2 == 0.0:
            assert thr.f64_in == thr.f64_out == thr.prune_above == thr.bulk_below == 0.0
        elif math.isinf(R2):
            assert thr.everything
            assert thr.f64_in == thr.f64_out == math.inf
        else:
            assert not thr.everything
            # The band is not empty, and not absurdly wide either (a wide band
            # costs exact resolutions, not correctness).
            assert thr.f64_in < R2 < thr.f64_out
            slack = 4 * 2.0**-1074
            assert thr.f64_out <= R2 * (1 + 8 * (d + 2) * U) + slack or math.isinf(thr.f64_out)
            assert thr.f64_in >= R2 * (1 - 8 * (d + 2) * U) - slack


@pytest.mark.parametrize("d", (1, 3, 64))
def test_integer_exact_thresholds_have_no_band(d):
    for R2 in (0.0, 1.0, 2.0000000000000004, 9.0, 1e15):
        thr = rc.thresholds(d, R2, integer_exact=True)
        assert thr.integer_exact
        assert thr.f64_in == thr.f64_out == R2
        member, uncertain = rc.classify_f64(np.array([R2, np.nextafter(R2, np.inf)]), thr)
        assert member.tolist() == [True, False] and not uncertain.any()


def test_thresholds_reject_invalid_R2_and_are_frozen():
    for bad in (-1.0, -5e-324, math.nan):
        with pytest.raises(ValueError):
            rc.thresholds(3, bad)
    thr = rc.thresholds(3, 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        thr.R2 = 2.0  # type: ignore[misc]


def test_gram_cut_rounds_R2_up_to_float32():
    for R2 in (0.0, 5e-324, 1e-50, 2.0**-149, 1.0, 1.1, 2.0**56 + 32, 3e38, FLT_MAX, 1e39, math.inf):
        cut = rc.thresholds(16, R2).gram_cut
        assert cut is not None and cut >= R2
        assert math.isinf(cut) or float(np.float32(cut)) == cut
        if 0 < cut < math.inf:
            below = float(np.nextafter(np.float32(cut), np.float32(-np.inf)))
            assert below < R2
    assert rc.thresholds(600_000, 1.0).gram_cut is None  # factor >= 1/8


def test_classify_f64_semantics():
    thr = rc.thresholds(4, 1.0)
    values = np.array([0.0, thr.f64_in, np.nextafter(thr.f64_in, np.inf), 1.0,
                       thr.f64_out, np.nextafter(thr.f64_out, np.inf), np.inf])
    member, uncertain = rc.classify_f64(values, thr)
    assert member.tolist() == [True, True, False, False, False, False, False]
    assert uncertain.tolist() == [False, False, True, True, True, False, False]
    member, uncertain = rc.classify_f64(values, rc.thresholds(4, math.inf))
    assert member.all() and not uncertain.any()
    member, uncertain = rc.classify_f64(np.array([0.0, 5e-324, 1.0]), rc.thresholds(4, 0.0))
    assert member.tolist() == [True, False, False] and not uncertain.any()


# ----------------------------------------------------------------------------------
# threshold soundness
# ----------------------------------------------------------------------------------


def _base_pair(kind: str, d: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """One float32 pair (q, x); groups are its signed coordinate permutations."""
    if kind == "mixed":  # exponent gaps above 29 make the float64 difference round
        return _f32_random(rng, d, -30, 30), _f32_random(rng, d, -30, 30)
    if kind == "close":
        q = _f32_random(rng, d, -3, 3)
        x = (q.astype(np.float64) * (1 + rng.normal(scale=2.0**-10, size=d))).astype(np.float32)
        return q, x
    if kind in ("crafted-up", "crafted-down"):
        # A big-term-first sum rounds every small odd square up (9 = 16 - 7) or
        # every 4 down at 2**56, so the sequential error approaches the bound.
        small = 3.0 if kind == "crafted-up" else 2.0
        x = np.full(d, small)
        x[0] = 2.0**28
        return np.zeros(d, np.float32), x.astype(np.float32)
    if kind == "subnormal":
        q = rng.integers(-2**20, 2**20, size=d) * TINY32
        x = rng.integers(-2**20, 2**20, size=d) * TINY32
        return q.astype(np.float32), x.astype(np.float32)
    if kind == "huge":
        sign = rng.choice([-1.0, 1.0], size=(2, d))
        values = sign * FLT_MAX * rng.uniform(0.25, 1.0, size=(2, d))
        return values[0].astype(np.float32), values[1].astype(np.float32)
    if kind == "ints":  # S exact in float64: R2 == S exactly
        return (rng.integers(-40, 41, size=d).astype(np.float32),
                rng.integers(-40, 41, size=d).astype(np.float32))
    if kind == "zero":
        q = rng.normal(size=d).astype(np.float32)
        return q, q.copy()
    raise AssertionError(kind)


KINDS = ("mixed", "mixed", "close", "crafted-up", "crafted-down", "subnormal", "huge",
         "ints", "mixed", "zero")


def _candidates(S: Fraction, d: int) -> list[float]:
    """R2 on S (when S is a float64), one and two ulps either side, and at
    multiples of the error bound (where the band's edges sit)."""
    T = float(S)
    out = [T, math.nextafter(T, -math.inf), math.nextafter(T, math.inf),
           math.nextafter(math.nextafter(T, -math.inf), -math.inf),
           math.nextafter(math.nextafter(T, math.inf), math.inf)]
    g = (d + 2) * U
    for k in (0.5, 1.0, 2.0, 3.0, 4.0):
        out += [T * (1 - k * g), T * (1 + k * g)]
    if S == 0:
        out += [0.0, TINY32 * TINY32, 1.0]
    return [r for r in out if r >= 0]


def _four_orders(diff: np.ndarray) -> tuple[np.ndarray, ...]:
    """float64 sums of squares in four orders: sequential, reversed, einsum, pairwise."""
    squares = diff * diff
    sequential = np.cumsum(squares, axis=1)[:, -1]  # add.accumulate is left to right
    reversed_ = np.cumsum(squares[:, ::-1], axis=1)[:, -1]
    einsum = np.einsum("ij,ij->i", diff, diff)
    pairwise = squares.sum(axis=1)
    return sequential, reversed_, einsum, pairwise


@pytest.mark.parametrize("d", DIMS)
def test_classify_f64_is_sound_for_any_summation_order(d):
    rng = np.random.default_rng(7_000 + d)
    orders = math.factorial(d) if d <= 4 else None
    if orders is not None:
        # Small d has at most d! distinct orders: use many groups instead.
        group_size, groups = orders, 10_000
        all_orders = np.array(list(itertools.permutations(range(d))), dtype=np.int64)
    else:
        group_size = 800 if d >= 512 else 200
        groups = PAIRS_PER_DIM // group_size
    counts = {"member": 0, "uncertain": 0, "reject": 0, "tie": 0}
    worst = 0.0
    for g in range(groups):
        kind = KINDS[g % len(KINDS)]
        if kind.startswith("crafted") and d == 1:
            kind = "mixed"
        q, x = _base_pair(kind, d, rng)
        S_scaled = _exact_scaled_sum(q, x)
        S = Fraction(S_scaled, 1 << 298)
        # Every permutation shares S and the per-coordinate float64 roundings
        # of fl(q_k - x_k); only the summation order differs. Signs do not
        # change any float64 square, so permuting the difference suffices.
        base = q.astype(np.float64) - x.astype(np.float64)
        if orders is not None:
            perms = all_orders
        else:
            perms = rng.permuted(np.broadcast_to(np.arange(d), (group_size, d)), axis=1)
        values = np.concatenate(_four_orders(base[perms]))
        if S:
            worst = max(worst, float(np.max(np.abs(values - float(S))) / float(S)))
        else:
            assert not values.any(), "s == 0 iff S == 0"
        for R2 in _candidates(S, d):
            thr = rc.thresholds(d, R2)
            assert thr.prune_above >= thr.f64_out and thr.bulk_below <= thr.f64_in
            member, uncertain = rc.classify_f64(values, thr)
            reject = ~(member | uncertain)
            if _exact_le(S_scaled, R2):
                assert not reject.any(), (
                    f"d={d} kind={kind}: S <= R2={R2!r} but a float64 value was rejected")
                counts["tie"] += int(S == Fraction(R2))
            else:
                assert not member.any(), (
                    f"d={d} kind={kind}: S > R2={R2!r} but a float64 value was accepted")
            counts["member"] += int(member.sum())
            counts["uncertain"] += int(uncertain.sum())
            counts["reject"] += int(reject.sum())
    assert counts["member"] and counts["uncertain"] and counts["reject"] and counts["tie"], counts
    # The data reach a good fraction of the proven bound gamma_{d+2}, so a band
    # several times too narrow would have failed above.
    gamma = (d + 2) * U / (1 - (d + 2) * U)
    assert worst >= (0.25 if d > 1 else 0.1) * gamma, (worst, gamma)


# ----------------------------------------------------------------------------------
# exact resolution
# ----------------------------------------------------------------------------------

SPECIAL32 = np.array([0.0, -0.0, TINY32, -TINY32, 2 * TINY32, 3 * TINY32, 2.0**-127, 2.0**-126,
                      np.nextafter(np.float32(2.0**-126), np.float32(0)), 2.0**-60, 1.0, -1.5,
                      3.4e38, -3.4e38, FLT_MAX, -FLT_MAX, 2.0**56, 2.0**100], dtype=np.float32)


def _adversarial_pairs(rng, count):
    """Pairs mixing special values, subnormals, huge and normal coordinates."""
    pairs = []
    for i in range(count):
        d = (1, 2, 3, 5, 8)[i % 5]
        style = i % 4
        if style == 0:
            q = rng.choice(SPECIAL32, size=d)
            x = rng.choice(SPECIAL32, size=d)
        elif style == 1:
            q = _f32_random(rng, d, -149, 127)
            x = _f32_random(rng, d, -149, 127)
        elif style == 2:  # small integers on a scaled grid: S is a float64, ties possible
            scale = 2.0 ** int(rng.integers(-149, 100))
            q = (rng.integers(-6, 7, size=d) * scale).astype(np.float32)
            x = (rng.integers(-6, 7, size=d) * scale).astype(np.float32)
        else:  # one-ulp neighbors
            q = _f32_random(rng, d, -140, 120)
            x = q.copy()
            k = int(rng.integers(d))
            x[k] = np.nextafter(x[k], np.float32(np.inf) if rng.random() < 0.5 else np.float32(-np.inf))
        pairs.append((q.astype(np.float32), x.astype(np.float32)))
    return pairs


def _R2_choices(S: Fraction, rng) -> list[float]:
    T = float(S) if S < Fraction(10) ** 308 else math.inf
    out = [0.0, math.inf, TINY32 * TINY32, float(rng.uniform(0, 4))]
    if T < math.inf:
        out += [T, math.nextafter(T, -math.inf), math.nextafter(T, math.inf)]
    return out


def test_exact_le_matches_fractions():
    rng = np.random.default_rng(8101)
    checked = ties = 0
    for q, x in _adversarial_pairs(rng, 1500):
        S = _fraction_S(q, x)
        for R2 in _R2_choices(S, rng):
            expected = math.isinf(R2) or S <= Fraction(R2)
            assert rc.exact_le(q, x, R2) == expected, (q, x, R2)
            assert rc.exact_le(x, q, R2) == expected
            checked += 1
            ties += int(not math.isinf(R2) and S == Fraction(R2))
    assert checked >= 10_000 and ties >= 300, (checked, ties)


def test_exact_le_uses_the_float32_image_of_its_inputs():
    rng = np.random.default_rng(8102)
    for _ in range(300):
        q = rng.normal(size=4) * 10.0 ** rng.integers(-30, 30)
        # float64 inputs that are not float32 values: the contract is about
        # their float32 images.
        x = q + rng.normal(size=4) * 1e-6 * np.abs(q)
        S = _fraction_S(q.astype(np.float32), x.astype(np.float32))
        T = float(S)
        for R2 in (T, math.nextafter(T, math.inf), max(0.0, math.nextafter(T, -math.inf))):
            assert rc.exact_le(q, x, R2) == (S <= Fraction(R2))


def test_resolve_band_matches_fractions():
    rng = np.random.default_rng(8103)
    for d in (1, 2, 3, 5, 8):
        train = np.concatenate([rng.integers(-5, 6, size=(60, d)).astype(np.float32),
                                rng.choice(SPECIAL32, size=(20, d)),
                                _f32_random(rng, (20, d), -149, 120)])
        queries = np.concatenate([rng.integers(-5, 6, size=(30, d)).astype(np.float32),
                                  rng.choice(SPECIAL32, size=(10, d)),
                                  _f32_random(rng, (10, d), -149, 120)])
        rows = rng.integers(0, len(queries), size=400)
        cols = rng.integers(0, len(train), size=400)
        exact = [_fraction_S(queries[r], train[c]) for r, c in zip(rows.tolist(), cols.tolist(), strict=True)]
        for R2 in (0.0, 1.0, 4.0, 9.0, 25.0, 2.0000000000000004, 2.9999999999999996,
                   TINY32 * TINY32, 1e77, math.inf, float(exact[0]),
                   math.nextafter(float(exact[1]), -math.inf)):
            keep = rc.resolve_band(train, queries, rows, cols, R2)
            assert keep.dtype == np.bool_ and keep.shape == rows.shape
            expected = [math.isinf(R2) or S <= Fraction(R2) for S in exact]
            assert keep.tolist() == expected, (d, R2)
    empty = rc.resolve_band(train, queries, np.empty(0, np.int64), np.empty(0, np.int64), 1.0)
    assert empty.shape == (0,)


def test_resolve_band_handles_exact_ties_on_the_scaled_grid():
    # S == R2 exactly at the bottom of the float32 range and with huge values.
    train = np.array([[TINY32, 0.0], [0.0, 0.0], [FLT_MAX, -FLT_MAX]], dtype=np.float32)
    queries = np.array([[0.0, 0.0], [-FLT_MAX, FLT_MAX]], dtype=np.float32)
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 2, 1])
    S_big = 2 * (2 * Fraction(FLT_MAX)) ** 2
    R2_big = float(S_big)  # 8 * FLT_MAX**2 is a float64 value
    assert Fraction(R2_big) == S_big
    assert rc.resolve_band(train, queries, rows, cols, TINY32 * TINY32).tolist() == [True, True, False, False]
    assert rc.resolve_band(train, queries, rows, cols, R2_big).tolist() == [True, True, True, True]
    smaller = math.nextafter(R2_big, -math.inf)
    assert rc.resolve_band(train, queries, rows, cols, smaller).tolist() == [True, True, False, True]


# ----------------------------------------------------------------------------------
# data statistics
# ----------------------------------------------------------------------------------


def _f32_next(value, direction):
    return float(np.nextafter(np.float32(value), np.float32(direction)))


ROW_FLAG_CASES = [
    (0.0, True), (-0.0, True), (1.0, True), (-1.0, True),
    (2.0**-60, True), (-(2.0**-60), True), (_f32_next(2.0**-60, 1.0), True),
    (_f32_next(2.0**-60, 0.0), False), (2.0**-61, False), (-(2.0**-61), False),
    (2.0**56, True), (-(2.0**56), True), (_f32_next(2.0**56, 0.0), True),
    (_f32_next(2.0**56, np.inf), False), (2.0**57, False), (-(2.0**57), False),
    (2.0**-31, True), (2.0**-30, True), (2.0**30, True), (2.0**31, True),
    (TINY32, False), (-TINY32, False), (2.0**-127, False), (2.0**-126, False),
    (FLT_MAX, False), (math.inf, False), (-math.inf, False), (math.nan, False),
]


def test_row_flags_boundaries():
    values = np.array([v for v, _ in ROW_FLAG_CASES], dtype=np.float32)
    expected = np.array([ok for _, ok in ROW_FLAG_CASES])
    # One value per row, padded with safe coordinates in a second column.
    A = np.stack([values, np.ones_like(values)], axis=1)
    flags = rc.row_flags(A)
    assert flags.dtype == np.uint8 and flags.shape == (len(values),)
    np.testing.assert_array_equal((flags & rc.FLAG_FP32_SAFE).astype(bool), expected)
    # One unsafe coordinate anywhere makes the whole row unsafe.
    B = np.ones((len(values), 5), dtype=np.float32)
    B[np.arange(len(values)), np.arange(len(values)) % 5] = values
    np.testing.assert_array_equal((rc.row_flags(B) & rc.FLAG_FP32_SAFE).astype(bool), expected)


def test_row_flags_are_blocking_and_layout_independent():
    rng = np.random.default_rng(8201)
    A = rng.choice(np.array([v for v, _ in ROW_FLAG_CASES[:-3]], dtype=np.float32), size=(1000, 3))
    reference = rc.row_flags(A)
    for block in (1, 7, 64, 999, 1000, 1 << 16):
        np.testing.assert_array_equal(rc.row_flags(A, block_rows=block), reference)
    np.testing.assert_array_equal(rc.row_flags(np.asfortranarray(A)), reference)
    np.testing.assert_array_equal(rc.row_flags(A.astype(np.float64)), reference)
    wide = np.concatenate([A, A], axis=1)[:, ::2]  # a strided view
    np.testing.assert_array_equal(rc.row_flags(wide), rc.row_flags(np.ascontiguousarray(wide)))
    assert rc.row_flags(np.empty((0, 3), np.float32)).shape == (0,)


def test_integer_exact_cases():
    ints = np.array([[0, 1, -2], [3, -0.0, 5]], dtype=np.float32)
    assert rc.integer_exact(ints)
    assert rc.integer_exact(ints, ints[:1])
    assert rc.integer_exact(ints, np.empty((0, 3), np.float32))
    assert not rc.integer_exact(ints, np.array([[0.5, 0, 0]], np.float32))
    assert not rc.integer_exact(np.array([[0.5, 1, 2]], np.float32))
    assert not rc.integer_exact(np.array([[np.nan, 1, 2]], np.float64))
    assert not rc.integer_exact(np.array([[np.inf, 1, 2]], np.float64))
    assert not rc.integer_exact(np.array([[TINY32, 1, 2]], np.float32))
    # The bound d * (2 * max|v|)**2 < 2**53, at d = 2.
    assert rc.integer_exact(np.array([[2.0**25 - 1, 0.0]]))
    assert not rc.integer_exact(np.array([[2.0**25, 0.0]]))
    assert not rc.integer_exact(np.array([[0.0, 0.0]]), np.array([[2.0**25, 1.0]]))
    assert not rc.integer_exact(np.array([[1e10]]))


def test_integer_exact_data_have_exact_float64_sums():
    rng = np.random.default_rng(8202)
    for d in (1, 3, 16, 784):
        top = min(int(math.isqrt(2**53 // d) // 2) - 1, 2**24)  # float32 integers
        A = rng.integers(-top, top + 1, size=(40, d)).astype(np.float64)
        A[0, 0], A[1, 0] = top, -top
        assert rc.integer_exact(A)
        diff = A[:20] - A[20:]
        s = np.einsum("ij,ij->i", diff, diff)
        for row, value in zip(diff.astype(np.int64).tolist(), s.tolist(), strict=True):
            assert value == sum(v * v for v in row)


# ----------------------------------------------------------------------------------
# validate_radius
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    (0, 0.0), (0.0, 0.0), (1.5, 1.5), (math.inf, math.inf), (np.inf, math.inf),
    (np.float32(1.5), 1.5), (np.float64(2.5), 2.5), (np.float16(0.5), 0.5),
    (np.int64(3), 3.0), (np.uint8(7), 7.0), (Fraction(3, 2), 1.5), (1e300, 1e300),
    (5e-324, 5e-324),
])
def test_validate_radius_accepts_reals(value, expected):
    result = rc.validate_radius(value)
    assert type(result) is float and result == expected


def test_validate_radius_negative_zero_is_zero():
    result = rc.validate_radius(-0.0)
    assert type(result) is float and result == 0.0
    assert result * result == 0.0  # R2 = +0.0 whatever the sign


@pytest.mark.parametrize("value", [math.nan, np.nan, np.float32(np.nan), float("-nan")])
def test_validate_radius_rejects_nan(value):
    with pytest.raises(ValueError):
        rc.validate_radius(value)


@pytest.mark.parametrize("value", [-1, -1.0, -5e-324, -math.inf, np.float64(-0.5), np.int8(-1)])
def test_validate_radius_rejects_negative(value):
    with pytest.raises(ValueError):
        rc.validate_radius(value)


@pytest.mark.parametrize("value", [None, "abc", "", [1.0], (1.0,), {1.0}, np.array([1.0]),
                                   np.array([1.0, 2.0]), np.ones((2, 2)), 1 + 0j, object()])
def test_validate_radius_rejects_non_scalars_and_non_numbers(value):
    with pytest.raises(TypeError):
        rc.validate_radius(value)
