"""Radius search orchestration: per-index caches, routing and engine dispatch.

Every engine returns batches whose neighbor sets satisfy the exact contract in
:mod:`radius_contract`; the router selects an engine for the workload. Each index
constructs its search structures on demand.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import replace
from functools import cached_property

import numpy as np

from .._common.arrays import INDEX_DTYPE, NUMPY_FLOAT
from .radius_contract import thresholds
from .radius_result import RadiusCSR, all_indices, empty_result, sort_rows
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
    """Compute only the facts used by this index's chosen engines."""

    def __init__(self, train: np.ndarray):
        self.train = train
        self.n, self.d = train.shape

    @cached_property
    def integral(self):
        return _all_integral(self.train)

    @cached_property
    def max_abs(self):
        return _max_abs(self.train)

    @cached_property
    def integer_norms(self):
        return np.einsum("ij,ij->i", self.train, self.train)

    @cached_property
    def center(self):
        return self.train.mean(axis=0, dtype=np.float64)

    @cached_property
    def sample(self) -> TrainSample:
        return TrainSample.from_train(self.train, self.center)

    @cached_property
    def metal_range_ok(self):
        if self.max_abs <= 2.0**49 / math.sqrt(self.d):
            # Each centered coordinate is bounded by twice the largest input.
            return True
        return _centered_norms_fit(self.train, self.center)


def _all_integral(values):
    flat = values.reshape(-1)
    # A short prefix typically disproves this optional certificate on real data.
    prefix = flat[:128]
    if not np.array_equal(prefix.view(np.uint32), np.rint(prefix).view(np.uint32)):
        return False
    for start in range(len(prefix), flat.size, 1 << 20):
        block = flat[start:start + (1 << 20)]
        if not np.array_equal(block.view(np.uint32), np.rint(block).view(np.uint32)):
            return False
    return True


def _max_abs(values):
    return max(-float(values.min(initial=0.0)), float(values.max(initial=0.0)))


def _integer_distance_bound(stats: _TrainStats, queries: np.ndarray, *, self_query=False) -> float:
    """Bound every integer partial sum in the Gram identity, or decline with infinity."""
    if not stats.integral:
        return math.inf
    largest = stats.max_abs
    if not self_query:
        if not _all_integral(queries):
            return math.inf
        largest = max(largest, _max_abs(queries))
    return stats.d * (2.0 * largest) ** 2


def prefer_integer_host_storage(train):
    """A cheap fit-time hint. The query path independently certifies all rows."""
    n, d = train.shape
    if not (512 <= n <= 131072 and 9 <= d <= 128):
        return False
    prefix = train.reshape(-1)[:128]
    return _all_integral(prefix) and 4 * d * _max_abs(prefix) ** 2 <= 2.0**24


def _centered_norms_fit(values, center):
    block_rows = max(1, (1 << 20) // len(center))
    for start in range(0, len(values), block_rows):
        block = values[start:start + block_rows].astype(np.float64)
        block -= center
        if np.any(np.einsum("ij,ij->i", block, block) > 2.0**100):
            return False
    return True


def _metal_range_ok(queries, stats, self_query):
    if not stats.metal_range_ok:
        return False
    if self_query or max(stats.max_abs, _max_abs(queries)) <= 2.0**49 / math.sqrt(stats.d):
        return True
    return _centered_norms_fit(queries, stats.center)


class RadiusSearch:
    """Lazy radius state over a :class:`NeighborIndex`'s owned float32 rows."""

    def __init__(self, index):
        self._index = index
        self._lock = threading.Lock()
        self._train = None
        self._stats = None
        self._trees = {}
        self._pilot_tree = None
        self._brute = None
        self._lookup = None
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

    def _ensure_lookup(self):
        if self._lookup is None:
            from .radius_lookup import RowLookup

            train = self.train
            with self._lock:
                if self._lookup is None:
                    self._lookup = RowLookup(train)
        return self._lookup

    def _ensure_tree(self, threads=None, leaf_size=0):
        tree = self._trees.get(leaf_size)
        if tree is None:
            native = native_engine()
            train = self.train
            with self._lock:
                tree = self._trees.get(leaf_size)
                if tree is None:
                    if native is not None:
                        tree = native.build_tree(train, leaf_size=leaf_size, threads=threads)
                    else:
                        from .radius_host import build_tree

                        tree = build_tree(train)
                    self._trees[leaf_size] = tree
        return tree

    def _coarse_tree_selective(self, queries, R2, native):
        # Sampled cell visits predict work only. The full tree tests every
        # potentially intersecting box and refines candidates on the owned rows.
        pilot = self._pilot_tree
        if pilot is None:
            train = self.train
            with self._lock:
                if self._pilot_tree is None:
                    # Deterministic sampling also works for interleaved clusters.
                    sample = np.random.default_rng(0).choice(len(train), 1024, replace=False)
                    self._pilot_tree = native.build_tree(train[sample], leaf_size=16, threads=1)
                pilot = self._pilot_tree
        sample = np.linspace(0, len(queries) - 1, 8, dtype=np.intp)
        visits = native.tree_pilot(pilot, queries[sample], R2, visit_cap=128, threads=1)[:, 1]
        # Reject the route if any probe hit the visit cap. This bounds planning
        # on diffuse inputs while allowing inexpensive scans of compact cells.
        return visits.max() < 128 and visits.sum() <= 8 * 96

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
        # Distinct binary32 coordinates differ by at least 2**-149. Amortize
        # the stable row sort only across large batches of exact-match queries.
        if (R2 < 2.0**-298 and algorithm == "auto" and forced in ("", "auto")
                and fixed != "mlx" and nq >= 512 and nq * 16 >= n):
            return RoutePlan("row-lookup", "exact coordinate matches")
        if (forced in ("", "auto") and fixed != "mlx" and 0 < R2 < math.inf
                and 9 <= d <= 128 and 512 <= n <= 131072 and 16 <= nq
                and 2**16 <= n * nq <= 2**26 and native_engine() is not None
                and _integer_distance_bound(stats, queries) <= 2.0**24):
            # Wide, multi-billion-term batches use Metal's higher matrix
            # throughput when available. Host-only fits still use the certificate.
            if fixed == "cpu" or n * nq * d <= 2**30 or not has_gpu():
                return RoutePlan("integer-gram", "exact float32 integer dot products")
        small = n * nq * d <= SMALL_WORK or math.isinf(R2) or nq == 0
        # In higher dimensions a coarse tree is worthwhile only when the data
        # separate into a few cells per query. Limit full-index build cost and
        # amortize its construction and the bounded pilot over a substantial batch.
        if (not small and 16 < d <= 128 and n >= 8192 and nq >= 64
                and n * nq >= 2**21 and n * d <= 2**25 and algorithm == "auto"
                and fixed != "mlx" and forced in ("", "auto")):
            native = native_engine()
            if native is not None and self._coarse_tree_selective(queries, R2, native):
                return RoutePlan("tree", "selective coarse cells", leaf_size=128)
        gpu = False
        if fixed != "cpu" and (forced == "metal" or fixed == "mlx"
                               or (not small and forced in ("", "auto")
                                   and (algorithm == "brute" or d > 8))):
            gpu = bool(has_gpu())
        metal_ok = False
        if gpu:
            # The certified Gram screen is exact on all finite data (its floor
            # covers underflow and unsafe entries are admitted); it is only
            # useless when centred norms or the radius approach float32's limit.
            metal_ok = ((2 * d + 16) * 2.0**-23 < 0.125 and d <= METAL_MAX_FEATURES
                        and (math.isinf(R2) or R2 <= 2.0**100)
                        and _metal_range_ok(queries, stats, exclude_self))
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
            sample = density_sample(stats.sample, queries, R2,
                                    exclude_self=exclude_self, q_offset=q_offset, n=n,
                                    small=n * nq <= 2**27)
            ctx = replace(ctx, sample=sample)
            plan = plan_radius(ctx)
        if plan.route == "stock" and forced != "stock":
            from sklearn import get_config as sklearn_config

            # Stock's uncentered float64 Gram calculation can lose even an
            # entire interior distance at a large offset. A centered dataset
            # can also round an exterior point onto the cutoff. Delegate only
            # when stock uses brute force and its intermediate sums are exact.
            # Above 15 features both supported fit algorithms use stock brute.
            # Stock's non-Cython paths may square the radius in its input
            # dtype or compare a rounded square root against it (metric="l2").
            if (d <= 15 or not sklearn_config()["enable_cython_pairwise_dist"]
                    or _integer_distance_bound(stats, queries, self_query=exclude_self) >= 2.0**53):
                plan = plan_radius(replace(ctx, patched=False))
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
        if plan.route == "row-lookup":
            part = self._ensure_lookup().query(
                queries, exclude_self=exclude_self, q_offset=q_offset, need_distance=need_distance)
            return [part], plan
        R2 = radius * radius
        if math.isinf(R2) and not need_distance and not raw and forced in ("", "auto"):
            return [all_indices(n_train=self.train.shape[0], n_query=nq,
                                q_offset=q_offset, exclude_self=exclude_self)], plan
        threads = resolve_threads(n_jobs)
        stats = self.stats
        integer_exact = plan.route == "integer-gram" or (
            0 < R2 < math.inf and _integer_distance_bound(stats, queries, self_query=exclude_self) < 2.0**53)
        thr = thresholds(stats.d, R2, integer_exact=integer_exact)
        train = self.train
        native = native_engine()
        sorted_natively = native is not None
        if plan.route == "integer-gram":
            parts = native.integer_gram_parts(
                train, queries, stats.integer_norms, thr, q_offset=q_offset, exclude_self=exclude_self,
                need_distance=need_distance, sort=sort_results, threads=threads)
        elif plan.route == "tree":
            parts = self._tree_parts(train, queries, thr, plan, q_offset=q_offset,
                                     exclude_self=exclude_self, need_distance=need_distance,
                                     sort_results=sort_results, threads=threads)
        elif plan.route == "metal":
            index = device_index() if callable(device_index) else device_index
            parts = self._metal_parts(index, train, queries, thr, plan, q_offset=q_offset,
                                      exclude_self=exclude_self, need_distance=need_distance,
                                      sort_results=sort_results, threads=threads, raw=raw)
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
        tree = self._ensure_tree(threads, plan.leaf_size)
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

    def _metal_parts(self, device_index, train, queries, thr, plan, *, q_offset, exclude_self,
                     need_distance, sort_results, threads, raw):
        from .radius_metal_v1 import metal_query

        result = metal_query(device_index, train, queries, thr, exclude_self=exclude_self,
                             q_offset=q_offset, need_distance=need_distance,
                             cap_hint=plan.cap_hint, threads=threads, sort=sort_results, raw=raw)
        if result is None:
            from .radius_host import brute_host

            result = brute_host(self._ensure_brute(), train, queries, thr,
                                exclude_self=exclude_self, q_offset=q_offset,
                                need_distance=need_distance, sort=sort_results,
                                threads=threads)
        return result if isinstance(result, list) else [result]


def lengths_of(parts: list[RadiusCSR]) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=INDEX_DTYPE)
    return np.concatenate([part.lengths() for part in parts])
