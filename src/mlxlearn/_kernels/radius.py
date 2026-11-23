"""Radius search orchestration: per-index caches, routing and engine dispatch.

Every engine returns batches whose neighbor sets satisfy the exact contract in
:mod:`radius_contract`; the router only chooses the fastest one. Caches are
built lazily under one lock per index.
"""

from __future__ import annotations

import math
import os
import threading

import numpy as np

from .._common.arrays import INDEX_DTYPE, NUMPY_FLOAT
from .radius_contract import thresholds
from .radius_result import RadiusCSR, empty_result, sort_rows
from .radius_router import (
    METAL_MAX_FEATURES,
    SMALL_WORK,
    RouteContext,
    RoutePlan,
    TrainSample,
    density_sample,
    plan_radius,
)

__all__ = ["RadiusSearch", "StockRoute", "native_engine", "resolve_threads"]

#: Expected result pairs per cKDTree batch; bounds the Python lists it builds.
_TREE_BATCH_PAIRS = 1 << 23
#: Host brute-force calls above this much work (n * n_query * d) at d > 16 use
#: the float64 GEMM screen rather than per-pair distances.
_GEMM_SCREEN_WORK = 1 << 26

_UNSET = object()
_NATIVE = _UNSET
_FORKED = False


def _mark_forked() -> None:
    # Grand Central Dispatch is not fork-safe; a forked child runs the native
    # engines on its calling thread only.
    global _FORKED
    _FORKED = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_mark_forked)


def native_engine():
    """The native radius module when its extension is built, else ``None``.

    The probe runs once per process; a failed import is not retried.
    """
    global _NATIVE
    if _NATIVE is _UNSET:
        try:
            from . import radius_cpu
        except ImportError:  # pragma: no cover - depends on the build
            radius_cpu = None
        _NATIVE = radius_cpu if radius_cpu is not None and radius_cpu.AVAILABLE else None
    return _NATIVE


class StockRoute(Exception):
    """Raised to tell a patched estimator to delegate the whole call to scikit-learn."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def resolve_threads(n_jobs) -> int:
    """Worker threads for host engines.

    ``Tuning.radius_threads`` when positive, else ``n_jobs`` with joblib's
    semantics (``None`` and ``-1`` mean all cores, ``-2`` all but one, ...).
    A process forked after the engines started uses one thread.
    """
    from .._common.config import get_tuning

    if _FORKED:
        return 1
    configured = getattr(get_tuning(), "radius_threads", 0)
    if configured and configured > 0:
        return int(configured)
    cores = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 0:
        return cores
    if n_jobs < 0:
        return max(1, cores + 1 + int(n_jobs))
    return int(n_jobs)


def _aligned(values) -> np.ndarray:
    """Owned-dtype, C-contiguous and aligned (native code requires all three)."""
    return np.require(values, dtype=NUMPY_FLOAT, requirements=("C", "A"))


class _TrainStats:
    """Facts about the owned training rows, computed once."""

    def __init__(self, train: np.ndarray):
        self.n, self.d = train.shape
        self.max_abs = float(np.max(np.abs(train))) if train.size else 0.0
        self.integral = bool(train.size == 0 or np.array_equal(train, np.round(train)))
        self.center = (train.mean(axis=0, dtype=np.float64) if train.size
                       else np.zeros(self.d, dtype=np.float64))
        self.center_norm = float(np.sqrt(np.dot(self.center, self.center)))
        # Largest squared distance from the training mean: the Metal Gram
        # screen admits every pair once centred norms near 2**108.
        largest = 0.0
        block_rows = max(1, (1 << 20) // max(self.d, 1))
        for start in range(0, self.n, block_rows):
            block = train[start:start + block_rows].astype(np.float64)
            block -= self.center
            largest = max(largest, float(np.einsum("ij,ij->i", block, block).max(initial=0.0)))
        self.centred_norm_max = largest
        self._sample = None

    def sample(self, train: np.ndarray) -> TrainSample:
        if self._sample is None:
            self._sample = TrainSample.from_train(train, self.center)
        return self._sample


def _batch_integer_exact(stats: _TrainStats, queries: np.ndarray, *, self_query: bool) -> bool:
    if not stats.integral:
        return False
    if self_query:
        largest = stats.max_abs
    else:
        if not np.array_equal(queries, np.round(queries)):
            return False
        largest = max(stats.max_abs, float(np.max(np.abs(queries))) if queries.size else 0.0)
    return stats.d * (2.0 * largest) ** 2 < 2.0**53


def _query_norm_bound(queries: np.ndarray, stats: _TrainStats, *, self_query: bool) -> float:
    """An upper bound on the queries' squared distances from the training mean."""
    if self_query:
        return stats.centred_norm_max
    if not len(queries):
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        largest = float(np.max(np.einsum("ij,ij->i", queries, queries)))
    if not math.isfinite(largest):
        return math.inf
    return (math.sqrt(largest) * (1 + 1e-6) + stats.center_norm) ** 2


class RadiusSearch:
    """Lazy radius state over a :class:`NeighborIndex`'s owned float32 rows."""

    def __init__(self, index):
        self._index = index
        self._lock = threading.Lock()
        self._train = None
        self._stats = None
        self._tree = None
        self._brute = None
        self.last_plan: RoutePlan | None = None

    # ------------------------------------------------------------------ caches

    @property
    def train(self) -> np.ndarray:
        """Host float32 view of the owned training rows (zero-copy from MLX)."""
        train = self._train
        if train is None:
            with self._lock:
                if self._train is None:
                    self._train = _aligned(self._index.training_data)
                train = self._train
        return train

    @property
    def stats(self) -> _TrainStats:
        stats = self._stats
        if stats is None:
            train = self.train
            with self._lock:
                if self._stats is None:
                    self._stats = _TrainStats(train)
                stats = self._stats
        return stats

    def _ensure_tree(self, threads=None):
        tree = self._tree
        if tree is None:
            native = native_engine()
            train = self.train
            with self._lock:
                if self._tree is None:
                    if native is not None:
                        self._tree = native.build_tree(train, threads=threads)
                    else:
                        from .radius_host import build_tree

                        self._tree = build_tree(train)
                tree = self._tree
        return tree

    def _ensure_brute(self):
        brute = self._brute
        if brute is None:
            from .radius_host import BruteState

            train = self.train
            with self._lock:
                if self._brute is None:
                    self._brute = BruteState(train)
                brute = self._brute
        return brute

    # ----------------------------------------------------------------- routing

    def plan(self, queries: np.ndarray, radius: float, *, q_offset: int = 0,
             exclude_self: bool, algorithm: str, n_jobs, patched: bool,
             fixed: str | None, has_gpu, forced: str) -> RoutePlan:
        stats = self.stats
        n, d = stats.n, stats.d
        nq = len(queries)
        R2 = radius * radius
        small = n * nq * d <= SMALL_WORK or math.isinf(R2) or nq == 0
        gpu = False
        if fixed != "cpu" and (forced == "metal" or fixed == "mlx"
                               or (not small and forced in ("", "auto"))):
            gpu = bool(has_gpu())
        metal_ok = False
        if gpu:
            # The certified Gram screen is exact on all finite data (its floor
            # covers underflow and unsafe entries are admitted); it is only
            # useless when centred norms or the radius approach float32's limit.
            metal_ok = ((2 * d + 16) * 2.0**-23 < 0.125 and d <= METAL_MAX_FEATURES
                        and stats.centred_norm_max <= 2.0**100
                        and (math.isinf(R2) or R2 <= 2.0**100)
                        and _query_norm_bound(queries, stats, self_query=exclude_self)
                        <= 2.0**100)
        if fixed == "mlx" and forced in ("", "auto") and gpu and metal_ok and nq:
            # An explicit GPU fit keeps every certifiable query on Metal, like
            # k-neighbor search.
            forced = "metal"
        native = native_engine() is not None
        ctx = RouteContext(
            n=n, d=d, n_query=nq, R2=R2, algorithm=algorithm, n_jobs=n_jobs,
            patched=patched, gpu=gpu, metal_ok=metal_ok, native=native, sample=None,
            forced=forced,
        )
        plan = plan_radius(ctx)
        # The density sample is taken only when the plan reads it: Metal
        # capacity and screen selectivity, cKDTree batching, and the patched
        # rules of NumPy-only installations.
        wants_sample = not small and (
            plan.route == "metal" or (plan.route == "tree" and not native)
            or (patched and not native))
        if wants_sample:
            sample = density_sample(stats.sample(self.train), queries, R2,
                                    exclude_self=exclude_self, q_offset=q_offset, n=n,
                                    small=n * nq <= 2**27)
            plan = plan_radius(_with_sample(ctx, sample))
        return plan

    # ----------------------------------------------------------------- queries

    def query(self, queries, radius: float, **options) -> list:
        """Result batches for one call (see :meth:`query_with_plan`)."""
        return self.query_with_plan(queries, radius, **options)[0]

    def query_with_plan(self, queries, radius: float, *, q_offset: int = 0,
                        exclude_self: bool, need_distance: bool, sort_results: bool,
                        algorithm: str = "auto", n_jobs=None, patched: bool = False,
                        fixed: str | None = None, forced: str = "auto",
                        has_gpu=lambda: True, device_index=None,
                        raw: bool = False) -> tuple[list, RoutePlan]:
        """Return ``(batches, plan)``, or raise :class:`StockRoute` for a patched stock plan.

        Batches are :class:`RadiusCSR`; with ``raw=True`` native engines may
        return their own result objects instead (for fused reductions).
        """
        queries = _aligned(queries)
        nq = len(queries)
        need_distance = need_distance or sort_results
        plan = self.plan(queries, radius, q_offset=q_offset, exclude_self=exclude_self,
                         algorithm=algorithm, n_jobs=n_jobs, patched=patched, fixed=fixed,
                         has_gpu=has_gpu, forced=forced)
        self.last_plan = plan
        if plan.route == "stock":
            raise StockRoute(plan.reason)
        if nq == 0:
            return [empty_result(0, need_distance=need_distance, route=plan.route)], plan
        threads = resolve_threads(n_jobs)
        stats = self.stats
        R2 = radius * radius
        thr = thresholds(stats.d, R2, integer_exact=_batch_integer_exact(
            stats, queries, self_query=exclude_self))
        train = self.train
        native = native_engine()
        sorted_natively = native is not None
        if plan.route == "tree":
            parts = self._tree_parts(train, queries, thr, plan, q_offset=q_offset,
                                     exclude_self=exclude_self, need_distance=need_distance,
                                     sort_results=sort_results, threads=threads)
        elif plan.route == "metal":
            index = device_index() if callable(device_index) else device_index
            parts = [self._metal(index, train, queries, thr, plan, q_offset=q_offset,
                                 exclude_self=exclude_self, need_distance=need_distance,
                                 sort_results=sort_results, threads=threads)]
        elif native is not None and (stats.d <= 16
                                     or stats.n * nq * stats.d <= _GEMM_SCREEN_WORK):
            parts = [native.brute_query(train, queries, thr, exclude_self=exclude_self,
                                        q_offset=q_offset, need_distance=need_distance,
                                        sort=sort_results, threads=threads)]
        else:
            from .radius_host import brute_host

            parts = [brute_host(self._ensure_brute(), train, queries, thr,
                                exclude_self=exclude_self, q_offset=q_offset,
                                need_distance=need_distance, sort=sort_results,
                                threads=threads)]
        out = []
        for part in parts:
            if not isinstance(part, RadiusCSR):
                if raw:
                    out.append(part)
                    continue
                part = part.to_csr(need_distance=need_distance, route=plan.route)
            elif sort_results and not (sorted_natively and part.route in ("metal", "brute")):
                part = sort_rows(part)
            out.append(part)
        return out, plan

    def _tree_parts(self, train, queries, thr, plan, *, q_offset, exclude_self,
                    need_distance, sort_results, threads):
        native = native_engine()
        tree = self._ensure_tree(threads)
        nq = len(queries)
        if native is not None:
            # The native result holds at most ~32 B per pair; one call.
            return [native.tree_query(tree, train, queries, thr, exclude_self=exclude_self,
                                      q_offset=q_offset, need_distance=need_distance,
                                      sort=sort_results, threads=threads)]
        from .radius_host import tree_query

        per_query = max(1.0, plan.cap_hint or 1.0)
        batch = int(min(nq, max(256, _TREE_BATCH_PAIRS // per_query)))
        return [tree_query(tree, train, queries[start:start + batch], thr,
                           exclude_self=exclude_self, q_offset=q_offset + start,
                           need_distance=need_distance, threads=threads)
                for start in range(0, nq, batch)]

    def _metal(self, device_index, train, queries, thr, plan, *, q_offset, exclude_self,
               need_distance, sort_results, threads):
        from .radius_metal_v1 import metal_query

        result = metal_query(device_index, train, queries, thr, exclude_self=exclude_self,
                             q_offset=q_offset, need_distance=need_distance,
                             cap_hint=plan.cap_hint, threads=threads, sort=sort_results)
        if result is None:
            from .radius_host import brute_host

            result = brute_host(self._ensure_brute(), train, queries, thr,
                                exclude_self=exclude_self, q_offset=q_offset,
                                need_distance=need_distance, sort=sort_results,
                                threads=threads)
        return result


def _with_sample(ctx: RouteContext, sample) -> RouteContext:
    from dataclasses import replace

    return replace(ctx, sample=sample)


def lengths_of(parts: list[RadiusCSR]) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=INDEX_DTYPE)
    return np.concatenate([part.lengths() for part in parts])
