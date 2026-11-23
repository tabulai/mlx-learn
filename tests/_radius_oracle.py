"""Exact radius-neighbor oracle and adversarial data for the radius test suites.

The library contract (docs/radius-certificate.md, clause M) fixes membership
exactly:

* ``X^`` and ``Q^`` are the float32 images of the inputs;
* ``R2 = fl64(radius * radius)``, which may be ``+inf``;
* ``S(q, x)`` is the exact real sum of the squared differences of the float32
  coordinates;
* ``x_j`` is a neighbor of ``q_i`` iff ``S(q_i, x_j) <= R2``. ``X=None`` excludes
  only ``j == i``; duplicates stay.

Every float32 value is an integer multiple of ``2**-149``, so ``S * 2**298`` is
an integer and the comparison is ``S * 2**298 <= floor(R2 * 2**298)``, decided
here with Python integers. Large inputs are screened in float64 first: with
float32 coordinates, ``s = sum_k fl(fl(q_k - x_k)**2)`` in *any* summation order
satisfies ``|s - S| <= gamma_{d+2} * S`` (``gamma_k = k u / (1 - k u)``,
``u = 2**-53``; nonzero differences are at least ``2**-149``, so their squares are
normal and there is no absolute term), and ``s == 0`` iff ``S == 0``. Pairs with
``s`` inside a band four times that wide are resolved exactly; everything else
is decided by the screen. The band is the oracle's own, deliberately wider than
the library's, and nothing here imports mlxlearn.

Generators return :class:`RadiusCase` values with deterministic seeds. Each is
small enough that :func:`oracle_radius` runs in well under a second.
"""

from __future__ import annotations

import math
import operator
from fractions import Fraction
from typing import NamedTuple

import numpy as np

__all__ = [
    "CASE_FAMILIES",
    "RadiusCase",
    "all_cases",
    "case_by_id",
    "fraction_radius",
    "labeled_cases",
    "oracle_csr",
    "oracle_distances",
    "oracle_radius",
    "radius_for_r2",
    "radius_hitting",
    "squared_radius",
    "to_float32",
]

SCALE_BITS = 149  #: every float32 value is an integer multiple of 2**-149
_DEN = 1 << (2 * SCALE_BITS)
_U = 2.0**-53
FLT_MAX = float(np.finfo(np.float32).max)
#: ``method="auto"`` evaluates every pair exactly at or below this many pairs.
EXACT_PAIR_BUDGET = 200_000
#: float64 (or int64) elements per screening block.
_BLOCK_ELEMENTS = 1 << 22


# ----------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------


def to_float32(A) -> np.ndarray:
    """The owned float32 coordinates of a 2-d input; non-finite values raise."""
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-d array, got shape {A.shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        A32 = np.ascontiguousarray(A, dtype=np.float32)
    if not np.isfinite(A32).all():
        raise ValueError("coordinates must be finite after the float32 cast")
    return A32


def squared_radius(radius) -> float:
    """``R2 = fl64(radius * radius)``; NaN and negative radii raise."""
    r = float(radius)
    if math.isnan(r):
        raise ValueError("radius must not be NaN")
    if r < 0:
        raise ValueError(f"radius must be non-negative, got {r}")
    return r * r


def _scaled_bound(R2: float, shift: int) -> int:
    """``floor(R2 * 2**shift)`` for finite ``R2 >= 0``, exactly."""
    numerator, denominator = R2.as_integer_ratio()
    if shift >= 0:
        return (numerator << shift) // denominator
    return numerator // (denominator << -shift)


def _scaled_rows(A32: np.ndarray) -> list[list[int]]:
    """``A * 2**149`` as Python integers (the float64 product is exact)."""
    scaled = A32.astype(np.float64) * 2.0**SCALE_BITS
    return [list(map(int, row)) for row in scaled.tolist()]


class _Grid(NamedTuple):
    """Integer images ``A * 2**shift`` when every pair sum fits int64 exactly."""

    shift: int
    X: np.ndarray
    Q: np.ndarray


def _integer_grid(X32: np.ndarray, Q32: np.ndarray) -> _Grid | None:
    d = X32.shape[1]
    values = np.abs(np.concatenate([X32.ravel(), Q32.ravel()]).astype(np.float64))
    nonzero = values[values != 0]
    if nonzero.size == 0:
        shift = 0
    else:
        mantissa, exponent = np.frexp(nonzero)  # 0.5 <= mantissa < 1
        significand = (mantissa * 2.0**24).astype(np.int64)  # exact: 24-bit significands
        lowest = significand & -significand
        lsb = exponent.astype(np.int64) - 24 + np.round(np.log2(lowest)).astype(np.int64)
        shift = int(-lsb.min())
        top = float(nonzero.max()) * 2.0**shift  # exact: an integer below 2**31 if usable
        if not top < 2.0**40 or d * (2 * int(top)) ** 2 >= 1 << 62:
            return None
    scale = 2.0**shift
    return _Grid(shift, (X32.astype(np.float64) * scale).astype(np.int64),
                 (Q32.astype(np.float64) * scale).astype(np.int64))


def _sum_of_products(a, b) -> int:  # math.sumprod is exact for ints, but Python >= 3.12
    return sum(map(operator.mul, a, b))


_sumprod = getattr(math, "sumprod", _sum_of_products)


def _exact_pair_sums(Qs: list[list[int]], Xs: list[list[int]], rows, cols) -> list[int]:
    """``S * 2**298`` for each (row, col) pair, as Python integers."""
    sub = operator.sub
    sumprod = _sumprod
    out = []
    append = out.append
    for r, c in zip(rows, cols, strict=True):
        diff = list(map(sub, Qs[r], Xs[c]))
        append(sumprod(diff, diff))
    return out


def _exact_sums_for_pairs(X32: np.ndarray, Q32: np.ndarray, rows, cols) -> list[int]:
    """``S * 2**298`` for the pairs ``(Q32[rows], X32[cols])``, converting only
    the rows that occur."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if rows.size == 0:
        return []
    used_rows, row_pos = np.unique(rows, return_inverse=True)
    used_cols, col_pos = np.unique(cols, return_inverse=True)
    return _exact_pair_sums(_scaled_rows(Q32[used_rows]), _scaled_rows(X32[used_cols]),
                            row_pos.ravel().tolist(), col_pos.ravel().tolist())


# ----------------------------------------------------------------------------------
# membership
# ----------------------------------------------------------------------------------


def oracle_radius(X, Q=None, radius=1.0, *, exclude_self: bool = False,
                  method: str = "auto", use_grid: bool = True) -> list[np.ndarray]:
    """Exact radius neighbors: ascending int64 training indices for each query row.

    ``Q=None`` queries the training rows themselves (``exclude_self`` is still
    explicit: pass ``True`` for scikit-learn's ``X=None`` semantics).

    ``method`` is ``"exact"`` (integers for every pair), ``"band"`` (float64
    screen, integers only inside the band) or ``"auto"`` (exact when the data
    lie on a small integer grid or at most ``EXACT_PAIR_BUDGET`` pairs, band
    otherwise). All three return the same sets.
    """
    indptr, indices = oracle_csr(X, Q, radius, exclude_self=exclude_self, method=method,
                                 use_grid=use_grid)
    return [indices[indptr[i]:indptr[i + 1]].copy() for i in range(len(indptr) - 1)]


def oracle_csr(X, Q=None, radius=1.0, *, exclude_self: bool = False,
               method: str = "auto", use_grid: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """:func:`oracle_radius` as CSR ``(indptr int64[nq+1], indices int64[nnz])``.

    ``use_grid=False`` disables the int64 fast path for data on a small integer
    grid, so every exact decision uses Python integers (for self-validation).
    """
    if method not in ("auto", "exact", "band"):
        raise ValueError(f"unknown method {method!r}")
    X32 = to_float32(X)
    Q32 = X32 if Q is None else to_float32(Q)
    if Q32.shape[1] != X32.shape[1]:
        raise ValueError(f"feature mismatch: {Q32.shape[1]} != {X32.shape[1]}")
    if exclude_self and Q32.shape[0] != X32.shape[0]:
        raise ValueError("exclude_self needs the query rows to be the training rows")
    R2 = squared_radius(radius)
    n, nq = X32.shape[0], Q32.shape[0]

    if math.isinf(R2) or n == 0 or nq == 0:
        mask_rows = np.repeat(np.arange(nq, dtype=np.int64), n)
        mask_cols = np.tile(np.arange(n, dtype=np.int64), nq)
        rows, cols = mask_rows, mask_cols
    else:
        grid = _integer_grid(X32, Q32) if use_grid else None
        if method == "auto":
            method = "exact" if grid is not None or n * nq <= EXACT_PAIR_BUDGET else "band"
        if method == "exact":
            rows, cols = _members_exact(X32, Q32, R2, grid)
        else:
            rows, cols = _members_band(X32, Q32, R2, grid)

    if exclude_self and rows.size:
        keep = rows != cols
        rows, cols = rows[keep], cols[keep]
    counts = np.bincount(rows, minlength=nq) if rows.size else np.zeros(nq, dtype=np.int64)
    indptr = np.zeros(nq + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, cols.astype(np.int64, copy=False)


def _query_blocks(nq: int, n: int, d: int):
    rows = max(1, _BLOCK_ELEMENTS // max(1, n * d))
    for start in range(0, nq, rows):
        yield start, min(nq, start + rows)


def _members_exact(X32, Q32, R2, grid):
    """Every pair decided with integers (row-major output: rows, then ascending cols)."""
    n, d = X32.shape
    nq = Q32.shape[0]
    row_parts, col_parts = [], []
    if grid is not None:
        bound = _scaled_bound(R2, 2 * grid.shift)
        for q0, q1 in _query_blocks(nq, n, d):
            diff = grid.Q[q0:q1, None, :] - grid.X[None, :, :]
            sums = np.einsum("ijk,ijk->ij", diff, diff)
            member = sums <= bound if bound < (1 << 62) else np.ones(sums.shape, dtype=bool)
            r, c = np.nonzero(member)
            row_parts.append(r + q0)
            col_parts.append(c)
    else:
        bound = _scaled_bound(R2, 2 * SCALE_BITS)
        Xs, Qs = _scaled_rows(X32), _scaled_rows(Q32)
        cols_all = range(n)
        for i in range(nq):
            sums = _exact_pair_sums(Qs, Xs, [i] * n, cols_all)
            c = np.fromiter((j for j, s in enumerate(sums) if s <= bound), dtype=np.int64)
            row_parts.append(np.full(c.size, i, dtype=np.int64))
            col_parts.append(c)
    if not row_parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return (np.concatenate(row_parts).astype(np.int64, copy=False),
            np.concatenate(col_parts).astype(np.int64, copy=False))


def _members_band(X32, Q32, R2, grid):
    """Float64 screen with a provably safe band; band pairs decided with integers."""
    n, d = X32.shape
    nq = Q32.shape[0]
    X64 = X32.astype(np.float64)
    Q64 = Q32.astype(np.float64)
    # |s - S| <= gamma_{d+2} S for any summation order. With c = 4 (d+2) u,
    # s < lo implies S < R2 and s > hi implies S > R2, including the outward
    # rounding of lo and hi themselves (the analysis in the module docstring).
    c = 4.0 * (d + 2) * _U
    lo = math.nextafter(R2 * (1.0 - c), -math.inf)
    hi = math.nextafter(R2 * (1.0 + c), math.inf)
    row_parts, col_parts = [], []
    band_rows, band_cols = [], []
    for q0, q1 in _query_blocks(nq, n, d):
        diff = Q64[q0:q1, None, :] - X64[None, :, :]
        s = np.einsum("ijk,ijk->ij", diff, diff)
        member = (s < lo) | (s == 0.0)
        band = ~member & (s <= hi)
        r, cidx = np.nonzero(member)
        row_parts.append(r + q0)
        col_parts.append(cidx)
        r, cidx = np.nonzero(band)
        band_rows.append(r + q0)
        band_cols.append(cidx)
    rows = np.concatenate(row_parts).astype(np.int64, copy=False)
    cols = np.concatenate(col_parts).astype(np.int64, copy=False)
    brows = np.concatenate(band_rows).astype(np.int64, copy=False)
    bcols = np.concatenate(band_cols).astype(np.int64, copy=False)
    if brows.size:
        keep = _decide_pairs(X32, Q32, brows, bcols, R2, grid)
        rows = np.concatenate([rows, brows[keep]])
        cols = np.concatenate([cols, bcols[keep]])
        order = np.lexsort((cols, rows))
        rows, cols = rows[order], cols[order]
    return rows, cols


def _decide_pairs(X32, Q32, rows, cols, R2, grid) -> np.ndarray:
    """Exact ``S <= R2`` for the given pairs."""
    if grid is not None:
        bound = _scaled_bound(R2, 2 * grid.shift)
        if bound >= (1 << 62):
            return np.ones(rows.size, dtype=bool)
        diff = grid.Q[rows] - grid.X[cols]
        return np.einsum("ij,ij->i", diff, diff) <= bound
    bound = _scaled_bound(R2, 2 * SCALE_BITS)
    sums = _exact_sums_for_pairs(X32, Q32, rows, cols)
    return np.fromiter((s <= bound for s in sums), dtype=bool, count=len(sums))


# ----------------------------------------------------------------------------------
# distances
# ----------------------------------------------------------------------------------


def oracle_distances(X, Q, rows, cols, *, kind: str = "float"):
    """Exact squared distances ``S`` of the pairs ``(Q[rows], X[cols])``.

    ``kind="float"`` returns the correctly rounded float64 values of ``S`` (the
    nearest float64, ties to even), ``"scaled"`` a list of the Python integers
    ``S * 2**298`` and ``"fraction"`` a list of :class:`fractions.Fraction`.
    """
    X32 = to_float32(X)
    Q32 = X32 if Q is None else to_float32(Q)
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    if kind == "float":
        grid = _integer_grid(X32, Q32)
        if grid is not None:
            diff = grid.Q[rows] - grid.X[cols]
            sums = np.einsum("ij,ij->i", diff, diff)
            # int64 -> float64 rounds to nearest; scaling by a power of two in
            # the normal range is exact, so the result is correctly rounded.
            return sums.astype(np.float64) * 2.0 ** (-2 * grid.shift)
        sums = _exact_sums_for_pairs(X32, Q32, rows, cols)
        # int / int true division is correctly rounded in CPython.
        return np.array([s / _DEN for s in sums], dtype=np.float64)
    sums = _exact_sums_for_pairs(X32, Q32, rows, cols)
    if kind == "scaled":
        return sums
    if kind == "fraction":
        return [Fraction(s, _DEN) for s in sums]
    raise ValueError(f"unknown kind {kind!r}")


def fraction_radius(X, Q=None, radius=1.0, *, exclude_self: bool = False) -> list[np.ndarray]:
    """Reference membership with :class:`fractions.Fraction`. Tiny inputs only.

    Independent of every shortcut above (no scaling, no screen, no grid); used
    to validate the oracle itself.
    """
    X32 = to_float32(X)
    Q32 = X32 if Q is None else to_float32(Q)
    R2 = squared_radius(radius)
    bound = None if math.isinf(R2) else Fraction(R2)
    Xf = [[Fraction(float(v)) for v in row] for row in X32.tolist()]
    Qf = [[Fraction(float(v)) for v in row] for row in Q32.tolist()]
    out = []
    for i, q in enumerate(Qf):
        members = []
        for j, x in enumerate(Xf):
            if exclude_self and i == j:
                continue
            if bound is None or sum((a - b) ** 2 for a, b in zip(q, x, strict=True)) <= bound:
                members.append(j)
        out.append(np.asarray(members, dtype=np.int64))
    return out


# ----------------------------------------------------------------------------------
# radius helpers
# ----------------------------------------------------------------------------------


def radius_hitting(target: float, *, steps: int = 8) -> float | None:
    """A radius ``r`` with ``fl64(r * r) == target``, or ``None`` if none is near."""
    if target == 0:
        return 0.0
    r = math.sqrt(target)
    for _ in range(steps):
        r = math.nextafter(r, -math.inf)
    for _ in range(2 * steps + 1):
        if r * r == target:
            return r
        r = math.nextafter(r, math.inf)
    return None


def radius_for_r2(S: Fraction, *, side: str) -> float:
    """The radius whose ``R2`` is the largest float64 below ``S`` (``side="below"``),
    the smallest at or above ``S`` (``"at_or_above"``) or exactly ``S`` (``"equal"``,
    raising if ``S`` is not reachable as ``fl64(r * r)``)."""
    approx = float(S)
    r = math.sqrt(approx)
    candidates = []
    for k in range(-12, 13):
        rr = r
        step = math.inf if k > 0 else -math.inf
        for _ in range(abs(k)):
            rr = math.nextafter(rr, step)
        candidates.append(rr)
    pairs = sorted({(c * c, c) for c in candidates})
    if side == "equal":
        for R2, c in pairs:
            if Fraction(R2) == S:
                return c
        raise ValueError(f"{S} is not fl64(r * r) for any nearby r")
    if side == "below":
        below = [(R2, c) for R2, c in pairs if Fraction(R2) < S]
        return below[-1][1]
    if side == "at_or_above":
        above = [(R2, c) for R2, c in pairs if Fraction(R2) >= S]
        return above[0][1]
    raise ValueError(f"unknown side {side!r}")


def _radius_for_count(X32, Q32, target: int) -> float:
    """A radius giving roughly ``target`` neighbors per query (median over a sample)."""
    sample = Q32[:: max(1, len(Q32) // 64)].astype(np.float64)
    diff = sample[:, None, :] - X32.astype(np.float64)[None, :, :]
    s = np.sort(np.einsum("ijk,ijk->ij", diff, diff), axis=1)
    k = min(target, s.shape[1]) - 1
    return float(math.sqrt(float(np.median(s[:, k]))))


# ----------------------------------------------------------------------------------
# generators
# ----------------------------------------------------------------------------------


class RadiusCase(NamedTuple):
    """One adversarial input. ``y`` holds class labels for estimator tests."""

    X: np.ndarray
    Q: np.ndarray
    radius: float
    id: str
    y: np.ndarray | None = None


def _f32(values) -> np.ndarray:
    """float32 array of values that must already be exactly representable."""
    values = np.asarray(values, dtype=np.float64)
    out = values.astype(np.float32)
    if not np.array_equal(out.astype(np.float64), values):
        raise AssertionError("generator value is not exactly representable in float32")
    return out


def _grid_points(values, d):
    mesh = np.meshgrid(*([np.asarray(values)] * d), indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, d)


def lattice_cases() -> list[RadiusCase]:
    """Integer and dyadic lattices with radii on lattice shells: many ``S == R2``.

    ``fl64(sqrt(k)**2)`` is not always ``k``: ``sqrt(3)`` squares to just below 3
    (the S == 3 shell is out) and ``sqrt(2)`` and ``sqrt(5)`` to just above (in).
    """
    cases = []
    rng = np.random.default_rng(1101)
    X = _f32(_grid_points(np.arange(-6, 7), 2))
    Q = _f32(_grid_points(np.arange(-7, 8), 2))
    cases.append(RadiusCase(X, Q, 2.0, "lattice-z2-r2"))
    cases.append(RadiusCase(X, Q, math.sqrt(5.0), "lattice-z2-sqrt5"))

    X = _f32(_grid_points(np.arange(-4, 5), 3))
    Q = _f32(rng.integers(-5, 6, size=(150, 3)))
    cases.append(RadiusCase(X, Q, math.sqrt(3.0), "lattice-z3-sqrt3"))
    cases.append(RadiusCase(X, Q, math.sqrt(2.0), "lattice-z3-sqrt2"))
    cases.append(RadiusCase(X, Q, math.sqrt(6.0), "lattice-z3-sqrt6"))

    X = _f32(rng.integers(-3, 4, size=(1500, 4)))
    Q = _f32(rng.integers(-4, 5, size=(150, 4)))
    cases.append(RadiusCase(X, Q, 3.0, "lattice-z4-r3"))

    X = _f32(rng.integers(-1, 2, size=(1200, 8)))
    Q = _f32(rng.integers(-1, 2, size=(150, 8)))
    cases.append(RadiusCase(X, Q, 2.0, "lattice-z8-r2"))

    # Dyadic lattice: k * 2**-12; radius 5 * 2**-12 puts (3, 4, 0) on the shell.
    X = _f32(rng.integers(-24, 25, size=(900, 3)) * 2.0**-12)
    Q = _f32(rng.integers(-26, 27, size=(150, 3)) * 2.0**-12)
    cases.append(RadiusCase(X, Q, 5 * 2.0**-12, "lattice-dyadic-r5"))

    # Integers near 2**23, where float32 still holds every integer: 3-4-5 ties.
    X = _f32(2.0**23 + rng.integers(-30, 31, size=(900, 2)))
    Q = _f32(2.0**23 + rng.integers(-32, 33, size=(150, 2)))
    cases.append(RadiusCase(X, Q, 5.0, "lattice-2^23-r5"))
    return cases


def duplicate_cases() -> list[RadiusCase]:
    """Repeated rows, all-identical data and all-zero data."""
    cases = []
    rng = np.random.default_rng(1201)
    base = rng.normal(size=(150, 5)).astype(np.float32)
    X = np.repeat(base, rng.integers(1, 6, size=150), axis=0)
    X = X[rng.permutation(len(X))]
    Q = np.concatenate([base[:60], rng.normal(size=(40, 5)).astype(np.float32)])
    cases.append(RadiusCase(X, Q, 0.0, "dup-r0"))
    cases.append(RadiusCase(X, Q, 0.75, "dup-r0.75"))

    row = np.array([[1.5, -2.25, 3.0]], dtype=np.float32)
    X = np.repeat(row, 300, axis=0)
    nudged = row.copy()
    nudged[0, 2] = np.nextafter(np.float32(3.0), np.float32(4.0))
    Q = np.concatenate([X[:5], nudged, np.zeros((1, 3), np.float32),
                        rng.normal(size=(8, 3)).astype(np.float32)])
    cases.append(RadiusCase(X, Q, 0.0, "same-r0"))
    cases.append(RadiusCase(X, Q, 1.0, "same-r1"))

    X = np.zeros((200, 4), dtype=np.float32)
    Q = _f32([[0, 0, 0, 0], [2.0**-149, 0, 0, 0], [0, -2.0**-149, 0, 2.0**-149], [1, 1, 1, 1]])
    cases.append(RadiusCase(X, Q, 0.0, "zeros-r0"))
    cases.append(RadiusCase(X, Q, 2.0**-149, "zeros-rmin"))
    return cases


def _signed_permutations(delta, count, rng):
    d = len(delta)
    out = np.empty((count, d), dtype=np.float64)
    for i in range(count):
        out[i] = np.asarray(delta, dtype=np.float64)[rng.permutation(d)]
        out[i] *= rng.choice([-1.0, 1.0], size=d)
    return out


def permuted_cases() -> list[RadiusCase]:
    """Points whose coordinates are signed permutations of each other.

    Every permutation has the same exact ``S`` but float64 sums in different
    orders round differently, so a route that compares rounded sums with ``R2``
    splits the group.

    * ``perm-falsepos``: ``(2**27, 1, 1, 1, 1)``. ``S = 2**54 + 4 > R2 = 2**54``,
      but a big-term-first float64 sum is ``2**54``.
    * ``perm-falseneg``: ``(2**28, 3, 3, 3)``. ``S = 2**56 + 27 <= R2 = 2**56 + 32``,
      but a big-term-first sum is ``2**56 + 48``.
    * ``perm-tie``: ``(2**28, 3, 3, 3, 2, 1)``. ``S == R2 == 2**56 + 32`` exactly;
      big-term-first gives ``2**56 + 48``.
    * ``perm-mixed-d16-*``: a random mixed-exponent vector with ``R2`` the
      float64 neighbors just below and at-or-above ``S``.
    """
    cases = []
    rng = np.random.default_rng(1301)

    def build(delta, radius, case_id, offsets):
        delta = np.asarray(delta, dtype=np.float64)
        d = len(delta)
        rows, queries = [], []
        for offset in offsets:
            q = np.asarray(offset, dtype=np.float64)
            queries.append(q)
            rows.append(q + _signed_permutations(delta, 60, rng))
            # Exactly on (and just past) the big coordinate alone.
            big = np.zeros(d)
            big[rng.integers(d)] = delta.max()
            rows.append((q + big)[None, :])
        X = _f32(np.concatenate(rows))
        Q = _f32(np.stack(queries))
        cases.append(RadiusCase(X, Q, radius, case_id))

    # Offsets that keep q + delta exact in float32 (multiples of 32 near 2**28).
    offsets5 = [np.zeros(5), 32.0 * rng.integers(-20, 21, size=5)]
    build([2.0**27, 1, 1, 1, 1], 2.0**27, "perm-falsepos", offsets5)
    offsets4 = [np.zeros(4), 32.0 * rng.integers(-20, 21, size=4)]
    build([2.0**28, 3, 3, 3], 2.0**28 + 2.0**-24, "perm-falseneg", offsets4)
    offsets6 = [np.zeros(6), 32.0 * rng.integers(-20, 21, size=6)]
    build([2.0**28, 3, 3, 3, 2, 1], 2.0**28 + 2.0**-24, "perm-tie", offsets6)

    mantissa = rng.integers(2**23, 2**24, size=16).astype(np.float64)
    delta = mantissa * 2.0 ** (rng.integers(-20, 21, size=16) - 23)
    S = sum(Fraction(float(np.float32(v))) ** 2 for v in delta)
    X = _f32(_signed_permutations(delta, 300, rng))
    Q = np.zeros((3, 16), dtype=np.float32)
    Q[1] = X[0]
    Q[2] = X[1]
    cases.append(RadiusCase(X, Q, radius_for_r2(S, side="below"), "perm-mixed-d16-below"))
    cases.append(RadiusCase(X, Q, radius_for_r2(S, side="at_or_above"), "perm-mixed-d16-above"))
    return cases


def offset_cases() -> list[RadiusCase]:
    """Large offsets with small radii; float64 inputs quantized by the float32 cast."""
    cases = []
    rng = np.random.default_rng(1401)
    # sigma is tens of float32 ulps at each offset (2**-14, 2**-4 and 2**16).
    for offset, sigma, tag in [(1e3, 0.01, "1e3"), (1e6, 2.0, "1e6"), (1e12, 2e6, "1e12")]:
        for d in (3, 8):
            X = offset + sigma * rng.normal(size=(1000, d))
            Q = offset + sigma * rng.normal(size=(150, d))
            radius = _radius_for_count(to_float32(X), to_float32(Q), 15)
            cases.append(RadiusCase(X, Q, radius, f"offset-{tag}-d{d}"))
    # At 1e12 float32 spacing is 2**16 = 65536, so coordinates sit on a coarse
    # grid; a radius of 3 grid steps puts (3, 0, 0) and (2, 2, 1) on the shell.
    X = 1e12 + 1.5 * 65536.0 * rng.normal(size=(400, 3))
    Q = 1e12 + 1.5 * 65536.0 * rng.normal(size=(100, 3))
    cases.append(RadiusCase(X, Q, 3 * 65536.0, "offset-1e12-tie"))
    return cases


def exponent_cases() -> list[RadiusCase]:
    """Subnormal, near-minimum-normal, near-FLT_MAX and wildly mixed exponents."""
    cases = []
    rng = np.random.default_rng(1501)
    tiny = 2.0**-149
    X = _f32(rng.integers(-12, 13, size=(600, 3)) * tiny)
    Q = _f32(rng.integers(-14, 15, size=(120, 3)) * tiny)
    cases.append(RadiusCase(X, Q, 5 * tiny, "tiny-subnormal-r5"))

    def random_f32(shape, low_exp, high_exp):
        mant = rng.integers(2**23, 2**24, size=shape).astype(np.float64)
        expo = rng.integers(low_exp, high_exp + 1, size=shape)
        sign = rng.choice([-1.0, 1.0], size=shape)
        with np.errstate(under="ignore"):
            return (sign * mant * 2.0 ** (expo - 23)).astype(np.float32)

    X = random_f32((500, 4), -149, -110)
    Q = np.concatenate([X[:40], random_f32((60, 4), -149, -110)])
    Q[:40, 0] = np.nextafter(Q[:40, 0], np.float32(np.inf))
    cases.append(RadiusCase(X, Q, _radius_for_count(X, Q, 3), "tiny-mixed"))

    base = 2.0**-126
    X = _f32(base * (1 + rng.integers(-2**22, 2**22, size=(400, 2)) * 2.0**-23))
    Q = _f32(base * (1 + rng.integers(-2**22, 2**22, size=(100, 2)) * 2.0**-23))
    cases.append(RadiusCase(X, Q, _radius_for_count(X, Q, 5), "near-min-normal"))

    sign = rng.choice([-1.0, 1.0], size=(400, 3))
    X = (sign * FLT_MAX * rng.uniform(0.25, 1.0, size=(400, 3))).astype(np.float32)
    X[:4] = _f32([[FLT_MAX, -FLT_MAX, FLT_MAX], [-FLT_MAX, FLT_MAX, -FLT_MAX],
                  [FLT_MAX, FLT_MAX, FLT_MAX], [0, 0, 0]])
    Q = np.concatenate([X[:30], (rng.choice([-1.0, 1.0], size=(50, 3)) * FLT_MAX
                                 * rng.uniform(0.25, 1.0, size=(50, 3))).astype(np.float32)])
    cases.append(RadiusCase(X, Q, 1e38, "huge-r1e38"))
    cases.append(RadiusCase(X, Q, 6e38, "huge-r6e38"))

    X = random_f32((500, 8), -140, 125)
    Q = X[:60].copy()
    column = rng.integers(0, 8, size=60)
    Q[np.arange(60), column] = np.nextafter(Q[np.arange(60), column], np.float32(np.inf))
    Q = np.concatenate([Q, random_f32((40, 8), -140, 125)])
    nudge = np.abs(Q[np.arange(60), column].astype(np.float64)
                   - X[np.arange(60), column].astype(np.float64))
    cases.append(RadiusCase(X, Q, float(np.median(nudge)), "mixed-exponents"))
    return cases


def r2_sweep_cases() -> list[RadiusCase]:
    """One multi-scale dataset swept over R2 in {0, 2**-120, tiny, huge, inf}."""
    rng = np.random.default_rng(1601)
    tiny = 2.0**-149
    parts = [
        np.zeros((4, 3)),
        rng.integers(-2, 3, size=(40, 3)) * tiny,
        rng.integers(-3, 4, size=(30, 3)) * 2.0**-61,
        rng.normal(size=(60, 3)),
        rng.normal(size=(30, 3)) * 1e30,
    ]
    X = np.concatenate(parts).astype(np.float32)
    X = np.concatenate([X, X[::7]])  # duplicates
    X = X[rng.permutation(len(X))]
    Q = np.concatenate([X[::3], rng.normal(size=(10, 3)).astype(np.float32)])
    radii = [
        (0.0, "r2-zero"),
        (2.0**-60, "r2-2^-120"),
        (2.0**-149, "r2-2^-298"),
        (1e-160, "r2-subnormal"),
        (5e-324, "r2-underflow"),
        (1.5, "r2-normal"),
        (1e30, "r2-huge"),
        (1e154, "r2-1e308"),
        (1e200, "r2-overflow"),
        (math.inf, "r2-inf"),
    ]
    return [RadiusCase(X, Q, r, case_id) for r, case_id in radii]


def hub_cases() -> list[RadiusCase]:
    """Heavy-tailed density: Zipf cluster sizes and log-uniform spreads."""
    cases = []
    rng = np.random.default_rng(1701)
    for d in (2, 8):
        clusters = 40
        weights = 1.0 / (1.0 + np.arange(clusters)) ** 1.3
        sizes = np.maximum(1, np.round(1500 * weights / weights.sum())).astype(int)
        spreads = 10.0 ** rng.uniform(-3, 0, size=clusters)
        centers = rng.uniform(-10, 10, size=(clusters, d))
        X = np.concatenate([centers[c] + spreads[c] * rng.normal(size=(sizes[c], d))
                            for c in range(clusters)]).astype(np.float32)
        X = X[rng.permutation(len(X))]
        hubs = centers[rng.integers(0, 5, size=100)] + 0.01 * rng.normal(size=(100, d))
        Q = np.concatenate([hubs, rng.uniform(-10, 10, size=(50, d))]).astype(np.float32)
        cases.append(RadiusCase(X, Q, 0.05 if d == 2 else 0.2, f"hubs-d{d}"))
    return cases


def _blobs(rng, n, nq, d, n_classes=6, spread=4.0):
    centers = rng.normal(scale=spread, size=(n_classes, d))
    y = rng.integers(0, n_classes, size=n)
    X = (centers[y] + rng.normal(size=(n, d))).astype(np.float32)
    yq = rng.integers(0, n_classes, size=nq)
    Q = (centers[yq] + rng.normal(size=(nq, d))).astype(np.float32)
    return X, Q, y


def label_sorted_cases() -> list[RadiusCase]:
    """Training rows in label-sorted order (blocks of one class), with labels."""
    rng = np.random.default_rng(1801)
    X, Q, y = _blobs(rng, 1200, 200, 4, n_classes=5, spread=2.0)
    order = np.argsort(y, kind="stable")
    X, y = X[order], y[order]
    return [RadiusCase(X, Q, _radius_for_count(X, Q, 15), "label-sorted", y)]


def gaussian_cases() -> list[RadiusCase]:
    """Plain Gaussian and clustered blobs at d in {1, 2, 3, 8, 16, 64}.

    The ``-edge`` variants put ``R2`` within an ulp of an exact pair distance.
    """
    cases = []
    for d in (1, 2, 3, 8, 16, 64):
        rng = np.random.default_rng(1900 + d)
        X = rng.normal(size=(1500, d)).astype(np.float32)
        Q = rng.normal(size=(200, d)).astype(np.float32)
        cases.append(RadiusCase(X, Q, _radius_for_count(X, Q, 12), f"gauss-d{d}"))
        Xb, Qb, yb = _blobs(rng, 1500, 200, d)
        cases.append(RadiusCase(Xb, Qb, _radius_for_count(Xb, Qb, 15), f"blobs-d{d}", yb))
        if d in (3, 16, 64):
            diff = Q[0].astype(np.float64) - X.astype(np.float64)
            j = int(np.argsort(np.einsum("ij,ij->i", diff, diff))[10])
            S = oracle_distances(X, Q, [0], [j], kind="fraction")[0]
            cases.append(RadiusCase(X, Q, radius_for_r2(S, side="below"), f"gauss-d{d}-edge"))
    return cases


CASE_FAMILIES = {
    "lattice": lattice_cases,
    "duplicates": duplicate_cases,
    "permuted": permuted_cases,
    "offset": offset_cases,
    "exponent": exponent_cases,
    "r2": r2_sweep_cases,
    "hubs": hub_cases,
    "label_sorted": label_sorted_cases,
    "gaussian": gaussian_cases,
}

_CACHE: dict[str, RadiusCase] | None = None


def all_cases() -> list[RadiusCase]:
    """Every generator's cases, in a fixed order (built once per process)."""
    global _CACHE
    if _CACHE is None:
        built: dict[str, RadiusCase] = {}
        for family in CASE_FAMILIES.values():
            for case in family():
                if case.id in built:
                    raise AssertionError(f"duplicate case id {case.id}")
                built[case.id] = case
        _CACHE = built
    return list(_CACHE.values())


def case_by_id(case_id: str) -> RadiusCase:
    all_cases()
    return _CACHE[case_id]


def labeled_cases() -> list[RadiusCase]:
    """Cases that carry class labels, for classifier and regressor checks."""
    return [case for case in all_cases() if case.y is not None]
