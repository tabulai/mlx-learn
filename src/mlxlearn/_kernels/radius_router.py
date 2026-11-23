"""State-free route selection for radius queries.

``plan_radius`` is a pure function of the call context: the same shapes,
data statistics, configuration and machine always give the same plan, and the
history of earlier calls is never consulted. Every route returns identical
neighbor sets (see :mod:`radius_contract`), so routing only affects speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["DensitySample", "RouteContext", "RoutePlan", "TrainSample", "density_sample",
           "plan_radius"]

ROUTES = ("auto", "tree", "brute", "metal", "stock")

#: Calls at most this large (n * n_query * d) run the host brute engine.
SMALL_WORK = 1 << 24
#: Gram screening needs these feature counts and float32-safe rows.
METAL_MAX_FEATURES = 4096


@dataclass(frozen=True, slots=True)
class DensitySample:
    m_hat: float  # estimated neighbors per query
    m_ub: float  # a generous upper estimate, for capacities
    skew: float  # densest sampled row / mean, >= 1
    c_hat: float = 0.0  # estimated Metal Gram-screen candidates per query


class TrainSample:
    """Evenly spread training rows, centred on the training mean in float64."""

    __slots__ = ("index", "rows", "norms", "center")

    def __init__(self, index, rows, norms, center):
        self.index, self.rows, self.norms, self.center = index, rows, norms, center

    @classmethod
    def from_train(cls, train: np.ndarray, center: np.ndarray) -> TrainSample:
        n, d = train.shape
        # At most 1024 rows and about 2**19 elements, spread over all of X.
        # Capacities only size a first pass (overflow is rescanned) and a
        # flooded Gram screen shows up plainly, so a small sample suffices and
        # keeps planning cheap for small calls.
        count = min(n, 1024, max(64, (1 << 19) // max(d, 1)))
        index = np.unique(np.linspace(0, n - 1, count).astype(np.int64)) if n else np.empty(0, np.int64)
        rows = np.asarray(train[index], dtype=np.float64) - center
        return cls(index, rows, np.einsum("ij,ij->i", rows, rows), center)


@dataclass(frozen=True, slots=True)
class RouteContext:
    n: int
    d: int
    n_query: int
    R2: float
    algorithm: str
    n_jobs: int | None
    patched: bool
    gpu: bool
    metal_ok: bool  # Gram bound valid and every row float32-safe
    native: bool
    sample: DensitySample | None
    forced: str = "auto"


@dataclass(frozen=True, slots=True)
class RoutePlan:
    route: str
    reason: str
    cap_hint: float = 0.0


def density_sample(sample: TrainSample, queries: np.ndarray, R2: float, *,
                   exclude_self: bool, q_offset: int = 0, n: int,
                   small: bool = False) -> DensitySample:
    """Estimate neighbors (and Metal Gram-screen candidates) per query.

    At most 32 evenly spread queries are compared, in float64, against the
    cached training sample; self-pairs are excluded for self-queries.
    """
    nq = len(queries)
    if n == 0 or nq == 0 or len(sample.index) == 0:
        return DensitySample(0.0, 0.0, 1.0, 0.0)
    if math.isinf(R2):
        return DensitySample(float(n), float(n), 1.0, float(n))
    # Small calls (where planning time shows) use a quarter of the sample:
    # their capacities are generous anyway, and the rules that read the
    # estimate only need to tell sparse rows from dense ones.
    rows, norms, x_index = sample.rows, sample.norms, sample.index
    if small and len(x_index) > 512:
        step = -(-len(x_index) // 512)
        rows, norms, x_index = rows[::step], norms[::step], x_index[::step]
    q_index = np.unique(np.linspace(0, nq - 1, min(nq, 16 if small else 32)).astype(np.int64))
    q = np.asarray(queries[q_index], dtype=np.float64) - sample.center
    q_norms = np.einsum("ij,ij->i", q, q)
    d = q.shape[1]
    # In place, to keep temporaries (and cold page faults) to a minimum.
    squared = q @ rows.T
    squared *= -2.0
    squared += q_norms[:, None]
    squared += norms[None, :]
    hits = squared <= R2
    # The Metal screen admits pairs whose certified lower bound reaches R2:
    # squared - f (|q|^2 + |x|^2) - floor <= R2.
    factor = (2 * d + 16) * 2.0**-23
    squared -= factor * q_norms[:, None]
    squared -= factor * norms[None, :]
    candidates = squared <= R2 + d * 1e-30
    if exclude_self:
        own = (q_index[:, None] + q_offset) == x_index[None, :]
        hits &= ~own
        candidates &= ~own
    per_row = hits.sum(axis=1)
    total = float(per_row.sum())
    scale = n / (len(q_index) * len(x_index))
    mean = total / len(q_index)
    skew = float(per_row.max()) / mean if mean > 0 else 1.0
    return DensitySample(total * scale, (total + 3.0) * scale * 2.0, max(1.0, skew),
                         float(candidates.sum()) * scale)


def _forced(ctx: RouteContext) -> RoutePlan | None:
    route = ctx.forced
    if route in ("", "auto"):
        return None
    if route not in ROUTES:
        raise ValueError(f"radius_route must be one of {ROUTES}, got {route!r}")
    if route == "metal" and not ctx.gpu:
        raise ValueError("radius_route='metal' requires a GPU")
    if route == "metal" and (not ctx.metal_ok or math.isinf(ctx.R2)):
        # Rows outside the float32-safe range are never screened on Metal;
        # the exact host engine serves them instead.
        return RoutePlan("brute", "forced metal: data outside the float32 screen")
    if route == "stock" and not ctx.patched:
        raise ValueError("radius_route='stock' is available only under patch_sklearn()")
    return RoutePlan(route, "forced", _cap_hint(ctx))


def _cap_hint(ctx: RouteContext) -> float:
    return 0.0 if ctx.sample is None else ctx.sample.m_ub


def _screen_selective(sample: DensitySample | None) -> bool:
    """False when the Gram screen would admit far more candidates than neighbors."""
    if sample is None:
        return True
    return sample.c_hat <= 8.0 * max(sample.m_hat, 1.0) + 64.0


def plan_radius(ctx: RouteContext) -> RoutePlan:
    """Choose an engine for one radius call."""
    forced = _forced(ctx)
    if forced is not None:
        return forced
    work = ctx.n * ctx.n_query * ctx.d
    if math.isinf(ctx.R2):
        return RoutePlan("brute", "infinite radius")
    if work <= SMALL_WORK:
        return RoutePlan("brute", "small call")
    if ctx.native:
        # The native multithreaded tree is fast through d = 8 everywhere, and
        # through d = 16 when no Metal screen is available.
        tree_dims = 8 if ctx.gpu and ctx.metal_ok else 16
    else:
        tree_dims = 8 if ctx.gpu and ctx.metal_ok else 15
    if ctx.algorithm != "brute" and ctx.d <= tree_dims:
        plan = RoutePlan("tree", f"d <= {tree_dims}", _cap_hint(ctx))
    elif ctx.gpu and ctx.metal_ok and _screen_selective(ctx.sample):
        plan = RoutePlan("metal", "gpu screen", _cap_hint(ctx))
    elif ctx.gpu and ctx.metal_ok and ctx.algorithm != "brute" and ctx.d <= 16:
        plan = RoutePlan("tree", "gram screen not selective", _cap_hint(ctx))
    elif ctx.gpu and ctx.metal_ok:
        plan = RoutePlan("brute", "gram screen not selective")
    else:
        plan = RoutePlan("brute", "no gpu screen")
    if ctx.patched:
        stock = _patched_stock_reason(ctx, plan)
        if stock is not None:
            return RoutePlan("stock", stock)
    return plan


def _patched_stock_reason(ctx: RouteContext, plan: RoutePlan) -> str | None:
    """State-free rules that keep a patched call on scikit-learn."""
    m_hat = 0.0 if ctx.sample is None else ctx.sample.m_hat
    pairs = m_hat * ctx.n_query
    heavy = ctx.n * ctx.n_query * ctx.d
    if not ctx.native and plan.route == "tree" and pairs >= 2**18 and ctx.n_jobs not in (None, 1):
        return "dense low-d output and a parallel scikit-learn tree"
    if not ctx.native and plan.route == "metal" and pairs >= 2**20 and ctx.d >= 9:
        # Without the native finisher every Metal candidate is refined in
        # NumPy, which loses to scikit-learn on dense outputs.
        return "dense high-d output"
    if plan.route == "metal" and m_hat >= 512 and ctx.n * ctx.n_query <= 2**27:
        # For moderately dense rows the single-tile bitmap is faster even
        # than stock's batched search. Very dense rows spend more time in
        # exact distance calculation and sorting.
        if (ctx.native and ctx.d <= 64 and ctx.n <= 2**17 and ctx.n * ctx.n_query <= 2**24
                and ctx.n / 64 <= m_hat <= ctx.n / 4):
            return None
        # Dense rows on a small screen: per-pair host work and the GPU's
        # cold-start cost outweigh the screen, and scikit-learn's batched
        # float64 GEMM wins (measured on M4 Max and M5 Ultra).
        return "dense output on a small screen"
    if plan.route == "brute" and ctx.d >= 9 and heavy >= 2**26:
        # scikit-learn's own brute force (d > 15) or tree (d <= 15) is at
        # least as fast as host brute force on heavy calls.
        return "heavy host brute force"
    return None
