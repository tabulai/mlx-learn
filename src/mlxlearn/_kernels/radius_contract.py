"""The radius-membership contract shared by every radius route.

Every route returns the same neighbor sets, decided exactly:

* ``X^`` and ``Q^`` are the owned float32 coordinates and ``R2 = fl64(r * r)``.
* ``S(q, x)`` is the exact real squared Euclidean distance of the float32
  coordinates. A training row ``x`` is a neighbor of ``q`` iff ``S <= R2``.

Routes compute an approximation of ``S`` (float64 on the host, float32 or
binary32 Gram values on Metal) and classify each candidate against thresholds
that are rounded outward here. A pair whose value falls inside the uncertain
band is decided by :func:`exact_le`, which compares integers. Membership
therefore does not depend on the route, the batch, the tiling or the machine.

Error model for host float64 values. Each float32 coordinate is exactly
representable in float64. ``s = sum_k fl(fl(q_k - x_k)**2)`` rounds at most
``d + 2`` times along any term's path, and every term is nonnegative, so
``|s - S| <= g64 * S`` with ``g64 = (d+2)u / (1 - (d+2)u)`` and ``u = 2**-53``.
Nonzero differences of float32 values are at least ``2**-149`` in magnitude, so
their squares are normal float64 numbers (``>= 2**-298``) and no absolute term
is needed; the largest possible square (``(2 * FLT_MAX)**2 ~ 4.6e77``) cannot
overflow. Hence ``s <= R2 * (1 - 2 g64)`` proves membership and
``s > R2 * (1 + 2 g64)`` proves non-membership. The factor two leaves room for
the outward rounding of the thresholds themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

__all__ = [
    "Thresholds",
    "classify_f64",
    "exact_le",
    "integer_exact",
    "resolve_band",
    "row_flags",
    "thresholds",
    "validate_radius",
]

_U64 = 2.0**-53
#: The repository's binary32 operation bound for Metal arithmetic, with one
#: guard bit (see docs/knn-compact-certificate.md).
_U32 = 2.0**-23
_FLT_MAX = float(np.finfo(np.float32).max)
#: Rows with a coordinate outside {0} U [2**-60, 2**56] are never sent to a
#: float32 Gram screen; their products can underflow or their norms overflow.
FP32_SAFE_MIN = 2.0**-60
FP32_SAFE_MAX = 2.0**56
FLAG_FP32_SAFE = 1


def _down(value: float) -> float:
    return math.nextafter(value, -math.inf)


def _up(value: float) -> float:
    return math.nextafter(value, math.inf)


def _round32(value: float, direction: int) -> float:
    """Round a float64 value to a float32 neighbor in the given direction."""
    if math.isinf(value) or math.isnan(value):
        return value
    nearest = float(np.float32(value)) if abs(value) <= _FLT_MAX else math.copysign(math.inf, value)
    if direction > 0 and nearest < value:
        return float(np.nextafter(np.float32(nearest), np.float32(np.inf)))
    if direction < 0 and nearest > value:
        return float(np.nextafter(np.float32(nearest), np.float32(-np.inf)))
    return nearest


def validate_radius(radius) -> float:
    """Return a radius as a float, rejecting NaN and negative values.

    Accepts real Python and NumPy scalars (and 0-d real arrays); ``+inf`` is
    allowed and makes every training row a neighbor. Booleans, strings,
    complex numbers and arrays with dimensions are rejected.
    """
    if isinstance(radius, np.ndarray):
        if radius.ndim != 0:
            raise TypeError("radius must be a scalar, got an array of shape "
                            f"{radius.shape}")
        radius = radius[()]
    if (isinstance(radius, (bool, np.bool_))
            or not isinstance(radius, (Real, np.floating, np.integer))):
        raise TypeError(f"radius must be a real number, got {type(radius).__name__}")
    value = float(radius)
    if math.isnan(value):
        raise ValueError("radius must not be NaN")
    if value < 0:
        raise ValueError(f"radius must be non-negative, got {value}")
    return value + 0.0  # normalizes -0.0


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Outward-rounded classification thresholds for one radius and dimension.

    ``f64_in``/``f64_out`` bracket the band for host float64 values: a value at
    or below ``f64_in`` is a member, above ``f64_out`` a non-member, anything
    in between goes to :func:`exact_le`. ``prune_above`` and ``bulk_below`` are
    the wider margins for tree box tests. ``gram_cut`` is the float32 cut for
    the certified Gram lower bound used by the Metal screen (``None`` when that
    screen cannot be certified).
    """

    R2: float
    d: int
    f64_in: float
    f64_out: float
    prune_above: float
    bulk_below: float
    gram_cut: float | None
    gram_factor: float
    gram_floor: float
    integer_exact: bool

    @property
    def everything(self) -> bool:
        return math.isinf(self.R2)


def thresholds(d: int, R2: float, *, integer_exact: bool = False) -> Thresholds:
    """Build the thresholds for squared radius ``R2`` in ``d`` dimensions."""
    d = int(d)
    R2 = float(R2)
    if not R2 >= 0:
        raise ValueError(f"R2 must be a non-negative number, got {R2}")
    gram_factor = (2 * d + 16) * _U32
    gram_floor = d * 1e-30
    gram_cut = _round32(R2, +1) if gram_factor < 0.125 else None
    if math.isinf(R2):
        inf = math.inf
        return Thresholds(R2, d, inf, inf, inf, inf, inf, gram_factor, gram_floor, integer_exact)
    if R2 == 0.0 or integer_exact:
        # s == 0 iff S == 0 for float32 coordinates, and integer-exact sums
        # equal S exactly, so the comparison needs no band.
        return Thresholds(R2, d, R2, R2, R2, R2, gram_cut, gram_factor, gram_floor, integer_exact)
    g = (d + 2) * _U64 / (1.0 - (d + 2) * _U64)
    # A float64 value of exactly 0 means S == 0, so a cut below zero (R2 so
    # small that the product underflows) can be raised to zero.
    f64_in = max(_down(_down(R2 * _down(1.0 - 2.0 * g))), 0.0)
    f64_out = _up(_up(R2 * _up(1.0 + 2.0 * g)))
    prune_above = _up(_up(R2 * _up(1.0 + 4.0 * g)))
    bulk_below = max(_down(_down(R2 * _down(1.0 - 4.0 * g))), 0.0)
    return Thresholds(R2, d, f64_in, f64_out, prune_above, bulk_below, gram_cut,
                      gram_factor, gram_floor, False)


def classify_f64(squared: np.ndarray, thr: Thresholds) -> tuple[np.ndarray, np.ndarray]:
    """Split float64 values into certain members and uncertain band entries."""
    member = squared <= thr.f64_in
    if thr.f64_out == thr.f64_in:
        return member, np.zeros(squared.shape, dtype=bool)
    uncertain = (squared > thr.f64_in) & (squared <= thr.f64_out)
    return member, uncertain


# ----------------------------------------------------------------------------------
# exact resolution
# ----------------------------------------------------------------------------------

_SCALE_BITS = 149  # float32 values are integer multiples of 2**-149


def _scaled_integers(values: np.ndarray) -> list[int]:
    # Multiplying a float32 value by 2**149 in float64 is exact (no overflow:
    # FLT_MAX * 2**149 ~ 2.4e83), and the product is an integer.
    scaled = np.asarray(values, dtype=np.float32).astype(np.float64) * 2.0**_SCALE_BITS
    return [int(v) for v in scaled.tolist()]


def _r2_floor_scaled(R2: float) -> int | None:
    """``floor(R2 * 2**298)``, or ``None`` when R2 is infinite."""
    if math.isinf(R2):
        return None
    numerator, denominator = R2.as_integer_ratio()
    return (numerator << (2 * _SCALE_BITS)) // denominator


def exact_le(q: np.ndarray, x: np.ndarray, R2: float) -> bool:
    """Decide ``S(q, x) <= R2`` exactly with integer arithmetic."""
    bound = _r2_floor_scaled(float(R2))
    if bound is None:
        return True
    total = 0
    for a, b in zip(_scaled_integers(q), _scaled_integers(x), strict=True):
        delta = a - b
        total += delta * delta
        if total > bound:
            return False
    return True


def resolve_band(train: np.ndarray, queries: np.ndarray, rows: np.ndarray,
                 cols: np.ndarray, R2: float) -> np.ndarray:
    """Exactly decide membership for the given (query row, training column) pairs."""
    keep = np.empty(len(rows), dtype=bool)
    bound = _r2_floor_scaled(float(R2))
    if bound is None:
        keep[:] = True
        return keep
    for i, (row, col) in enumerate(zip(np.asarray(rows).tolist(), np.asarray(cols).tolist(),
                                       strict=True)):
        total = 0
        for a, b in zip(_scaled_integers(queries[row]), _scaled_integers(train[col]),
                        strict=True):
            delta = a - b
            total += delta * delta
        keep[i] = total <= bound
    return keep


# ----------------------------------------------------------------------------------
# data statistics
# ----------------------------------------------------------------------------------


def row_flags(A: np.ndarray, block_rows: int = 1 << 16) -> np.ndarray:
    """Per-row range flags; bit 0 set when every coordinate is 0 or in [2**-60, 2**56]."""
    A = np.asarray(A, dtype=np.float32)
    flags = np.zeros(A.shape[0], dtype=np.uint8)
    low_bits = np.uint32(np.float32(FP32_SAFE_MIN).view(np.uint32))
    high_bits = np.uint32(np.float32(FP32_SAFE_MAX).view(np.uint32))
    for start in range(0, A.shape[0], block_rows):
        block = A[start:start + block_rows].view(np.uint32) & np.uint32(0x7FFFFFFF)
        ok = (block == 0) | ((block >= low_bits) & (block <= high_bits))
        flags[start:start + block_rows] = np.all(ok, axis=1).astype(np.uint8) * FLAG_FP32_SAFE
    return flags


def integer_exact(train: np.ndarray, queries: np.ndarray | None = None) -> bool:
    """True when every coordinate is integral and float64 sums of squared
    differences are exact: ``d * (2 * max|v|)**2 < 2**53``."""
    arrays = [np.asarray(train)] + ([] if queries is None else [np.asarray(queries)])
    largest = 0.0
    for array in arrays:
        if array.size == 0:
            continue
        if not np.all(np.isfinite(array)) or not np.array_equal(array, np.round(array)):
            return False
        largest = max(largest, float(np.max(np.abs(array))))
    d = arrays[0].shape[1] if arrays[0].ndim == 2 else 1
    return d * (2.0 * largest) ** 2 < 2.0**53
