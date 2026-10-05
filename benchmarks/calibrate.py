"""Calibration grids: the measurements every dispatch threshold rests on.

Usage::

    python -m benchmarks.calibrate list
    python -m benchmarks.calibrate run knn --output bench/calibration/knn-1.9.0.jsonl
    python -m benchmarks.calibrate table knn bench/calibration/knn-*.jsonl
    python -m benchmarks.calibrate update docs/benchmarks.md bench/calibration/*.jsonl

``run`` measures one named grid and appends a JSON line per measurement to
``--output``; a run that is interrupted, or split into pieces, picks up where the
file left off, because a measurement already in it is not taken again. ``table``
renders the Markdown tables of ``docs/benchmarks.md`` from one or more such files —
typically one per scikit-learn version, each cell then reading ``1.9.0 / 1.7.2`` —
with a ``dispatch`` column computed from the rules and ``Tuning`` of the mlxlearn
that renders it, so a table can be checked against the thresholds it justifies.
``update`` re-renders, in place, every such table a document already holds. The
results the published tables were rendered from are ``benchmarks/results/calibration/``.

**What is compared.** The mlxlearn side is the *patched* class — what
``patch_sklearn()`` installs — forced onto its MLX path
(``MLXLEARN_FORCE_BACKEND=mlx``), which is the path patched dispatch runs when a
threshold sends a problem to MLX; the stock side is scikit-learn's class with the
same parameters. Forcing, rather than lowering the thresholds, keeps the grids
measurable whatever the rules become: the linear and sigmoid ``SVC`` kernels, which
no threshold sends to MLX, are measured the same way. For the neighbor estimators
the stock side uses ``algorithm="auto"`` unless a grid says otherwise, which is
brute force from 16 features and a KD-tree below.

**How.** With the benchmark harness, :func:`benchmarks._harness.compare`: warm-up
calls, then interleaved pairs whose order alternates, a garbage collection before
every timed call and a synchronization inside its window. The collection idles
the GPU, so every MLX call pays to wake it, as a call made after other work does;
the thresholds are set against that. A neighbor model is fitted once per shape on
each side and every query batch of the grid is timed against that pair of fits;
the fit is timed separately where the grid asks for it. ``SVC`` and
``LogisticRegression`` time the fit.

**Reading a result.** ``speedup`` is stock time over mlxlearn time, medians, so
above 1 mlxlearn is faster; ``significant`` is the gate's test (three standard
errors of the median apart); a *loss* in the tables is what the gate would fail —
significant and more than 5% slower. The grids are named for the thresholds they
calibrate; ``docs/benchmarks.md`` cites each table by grid.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._harness import compare

__all__ = ["GRIDS", "Point", "main"]

#: A loss, as the gate counts one: significant, and more than this much slower.
_LOSS_TOLERANCE = 0.05

#: Neighbors, k.
_K = 10

#: Untimed calls per side before a comparison (the harness's default).
_WARMUP = 2


# --------------------------------------------------------------------------- points


@dataclass(frozen=True)
class Point:
    """One shape of one estimator, and what to measure there.

    A neighbor point is a pair of fits and one timed query batch per entry of
    ``queries`` — plus the fit itself when ``fit`` is set. An ``SVC`` or
    ``LogisticRegression`` point is one timed fit; an ``SVC`` point with
    ``queries`` is instead a pair of fits and one timed ``predict`` per batch size.
    """

    estimator: str  # "knn" | "svc" | "logreg"
    n: int
    d: int
    queries: tuple[int, ...] = ()
    fit: bool = False
    algorithm: str = "auto"  # knn: the scikit-learn algorithm, given to both sides
    # knn: "gauss", "lowrank" or another law of _knn_data; svc and logreg: "binary"
    # (benchmarks/run.py's binary target), "noisy", "random", "xor", "mixed", "blobs",
    # "separated-blobs", or a structured law of _fit_data. The estimator's first, by
    # default.
    data: str = ""
    kernel: str = "rbf"
    C: float = 1.0  # svc and logreg
    n_classes: int = 2
    k: int = 10  # knn: n_neighbors
    # svc: gamma as a multiple of scikit-learn's "scale" value, 1 / (d · Var X);
    # 0 means gamma="scale" itself.
    gamma: float = 0.0
    degree: int = 3
    coef0: float = 0.0
    tol: float = 0.0  # svc and logreg; 0 means the estimator's default
    max_iter: int = 0  # logreg; 0 means the grids' 200
    class_weight: str = ""  # svc: "" (none) or "balanced"
    cache_size: float = 0.0  # svc; 0 means scikit-learn's default, 200 MB
    # knn: the k each query asks for, kneighbors(n_neighbors=...); 0 means the fitted k.
    query_k: int = 0
    dtype: str = "float64"  # knn: of the training rows and the queries
    # knn with query_k: "" times the query as patched dispatch routes it (a fit it sends
    # to MLX; above knn_max_neighbors, scikit-learn's brute force over a float64 copy
    # built on the first such query and kept), "first" the same with that copy dropped
    # before every call — the first such query — and "mlx" the query on MLX (a forced
    # fit, whose queries stay there).
    path: str = ""
    # knn: another draw of the same law — its rows, and the centres or basis they share
    # with the queries. 0 is the draw every other grid measures.
    data_seed: int = 0
    # knn: also record what a fit-time probe of this many rows finds, beside the one of
    # Tuning.knn_probe_rows rows that dispatch reads (0: none).
    probe_also: int = 0

    def __post_init__(self) -> None:
        if not self.data:
            object.__setattr__(self, "data", "gauss" if self.estimator == "knn" else "binary")

    def keys(self) -> list[str]:
        """The keys of the measurements this point produces, one per record."""
        if self.estimator == "knn":
            base = f"knn/{self.n}x{self.d}/{self.algorithm}/{self.data}"
            if self.k != _K:
                base = f"{base}/k={self.k}"
            if self.query_k:
                base = f"{base}/query_k={self.query_k}"
            if self.dtype != "float64":
                base = f"{base}/{self.dtype}"
            if self.path:
                base = f"{base}/{self.path}"
            if self.data_seed:
                base = f"{base}/seed={self.data_seed}"
            ops = (["fit"] if self.fit else []) + [f"q{q}" for q in self.queries]
            return [f"{base}/{op}" for op in ops]
        if self.estimator == "svc":
            key = f"svc/{self.n}x{self.d}/{self.kernel}/C={self.C:g}/K={self.n_classes}"
            if self.gamma:
                key = f"{key}/gamma={self.gamma:g}xscale"
            if self.degree != 3:
                key = f"{key}/degree={self.degree}"
            if self.coef0:
                key = f"{key}/coef0={self.coef0:g}"
            if self.tol:
                key = f"{key}/tol={self.tol:g}"
            if self.class_weight:
                key = f"{key}/class_weight={self.class_weight}"
            if self.cache_size:
                key = f"{key}/cache_size={self.cache_size:g}"
            if self.data not in ("binary", "separated-blobs"):
                key = f"{key}/{self.data}"
            if self.queries:
                return [f"{key}/predict/q{q}" for q in self.queries]
            return [key]
        key = f"logreg/{self.n}x{self.d}/K={self.n_classes}/{self.data}"
        if self.C != 1.0:
            key = f"{key}/C={self.C:g}"
        if self.tol:
            key = f"{key}/tol={self.tol:g}"
        if self.max_iter:
            key = f"{key}/max_iter={self.max_iter}"
        return [key]

    @property
    def seed(self) -> int:
        return self.n * 131 + self.d * 7 + self.n_classes + 7_919 * self.data_seed


@dataclass(frozen=True)
class Grid:
    """A named list of points, and how ``table`` lays its results out."""

    description: str
    points: tuple[Point, ...]
    layout: str  # "knn-q" | "knn-worst" | "svc" | "logreg"
    iterations: int = 9


def _knn_points(shapes: Iterable[tuple[int, int]], queries, *, fit=False, max_work=2.5e11, **kw):
    """Neighbor points, dropping query batches whose stock cost would take minutes."""
    points = []
    for n, d in shapes:
        qs = tuple(q for q in queries if n * d * q <= max_work)
        points.append(Point("knn", n, d, qs, fit=fit, **kw))
    return tuple(points)


#: Batch sizes the dense neighbor grid times on every shape: a fit cannot see which
#: will follow it, so a threshold has to hold for all of them.
DENSE_QUERIES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)

#: Rows of the dense grid. Not powers of two, apart from the old floor at 16
#: features, so that nothing about a block size lines up by construction.
_DENSE_ROWS = (
    12_000, 20_000, 33_000, 50_000, 80_000, 120_000, 180_000, 262_144,
    400_000, 600_000, 1_000_000, 1_600_000,
)
_DENSE_FEATURES = (16, 24, 32, 48, 64, 96, 128, 256)

#: Batch sizes of the `knn-batches` grid: every few queries from 8 to 64, where a
#: 2 048-4 096-feature batch crosses 256 KiB of float32 rows.
_HUMP_QUERIES = (8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 48, 64)

#: Draws of the `knn-probe` grid: of the clustered law it calibrates, and of each control.
_PROBE_DRAWS = 64
_PROBE_CONTROLS = 8

#: Batch sizes of the `knn-probe` grid: those a few uncertified rows in a hundred lose.
_PROBE_QUERIES = (16, 32, 64, 128, 256)

#: (query row, training row) pairs from which a neighbor query's first pass selects with
#: the fused kernel (mlxlearn._kernels.distance._FUSED_MIN_PAIRS); below, MLX's own
#: selection runs. The `knn-boundary` grid times the largest batch below it.
_FUSED_MIN_PAIRS = 1 << 22

#: Batch sizes of the `knn-query-k` grid, and those it times the first large-k query at.
_QUERY_K_BATCHES = (1, 16, 64, 256, 1_024)
_QUERY_K_FIRST = (1, 64, 1_024)


def _dense_shapes() -> list[tuple[int, int]]:
    """From ~2.5x below 0.1.0a3's work floor (4 194 304 elements) to 1.3e8 elements."""
    return [
        (n, d)
        for d in _DENSE_FEATURES
        for n in _DENSE_ROWS
        if n * d * 2.5 >= 4_194_304 and n * d <= 1.3e8
    ]


def _svc_points(shapes, kernels, Cs, n_classes=2, data="binary") -> tuple[Point, ...]:
    return tuple(
        Point("svc", n, d, kernel=kernel, C=C, n_classes=n_classes, data=data)
        for n, d in shapes
        for kernel in kernels
        for C in Cs
    )


GRIDS: dict[str, Grid] = {
    "knn": Grid(
        "Neighbor queries on MLX against stock brute force, 16-256 features, every batch "
        "size from 1 to 4096, rows from below the work floor to 1.6 million: the "
        "`knn_min_work` / `knn_wide_features` rule.",
        _knn_points(_dense_shapes(), DENSE_QUERIES, fit=True, max_work=1.2e11),
        layout="knn-worst",
    ),
    "knn-rows": Grid(
        "Neighbor queries on MLX against stock brute force on wide, short training sets, "
        "2 048-131 072 rows at 256-1 024 features, every batch size from 1 to 4096: the "
        "row floor, and how wide data must be for the work floor alone to decide.",
        _knn_points(
            [
                (n, d)
                for d in (256, 384, 512, 768, 1_024)
                for n in (2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072)
            ],
            DENSE_QUERIES,
            fit=True,
            max_work=1.2e11,
        ),
        layout="knn-worst",
    ),
    "knn-tree": Grid(
        "Neighbor queries on MLX against scikit-learn's KD-tree (`algorithm='auto'` below "
        "16 features), Gaussian data: the `knn_min_features` handoff.",
        _knn_points(
            [(4_096, 2), (65_536, 2), (1_048_576, 3), (65_536, 4), (1_048_576, 5),
             (65_536, 6), (262_144, 6), (4_096, 8), (65_536, 8), (262_144, 8),
             (1_048_576, 8), (4_096, 12), (65_536, 12), (262_144, 12), (1_048_576, 12),
             (16_384, 14), (262_144, 14), (65_536, 15)],
            (1, 16, 256, 4_096),
            fit=True,
            max_work=1.2e11,
        ),
        layout="knn-q",
    ),
    "knn-tree-lowrank": Grid(
        "The same against the KD-tree on data of low intrinsic dimension: three latent "
        "dimensions embedded in 8 or 12 features, plus 5% noise.",
        _knn_points(
            [(65_536, 8), (262_144, 8), (65_536, 12), (262_144, 12)],
            (1, 16, 256, 4_096),
            fit=True,
            data="lowrank",
        ),
        layout="knn-q",
    ),
    "knn-narrow-brute": Grid(
        "Neighbor queries on MLX against stock brute force (`algorithm='brute'`) at 2-8 "
        "features: what patched mode hands to scikit-learn's brute force below 16 features.",
        _knn_points(
            [(n, d) for d in (2, 8) for n in (8_192, 131_072, 524_288, 2_097_152)],
            (1, 16, 256, 4_096),
            algorithm="brute",
        ),
        layout="knn-q",
    ),
    "svc": Grid(
        "Binary `SVC` fits (y = x0 + noise > 0) on MLX against LIBSVM, rbf and poly at "
        "C = 1 and 10 everywhere, and at C = 0.1 and 100 around the floors: the width, "
        "sample and work floors.",
        _svc_points(
            [(4_000, 64), (1_000, 128), (2_000, 128), (2_500, 128), (3_000, 128),
             (4_000, 128), (1_000, 256), (1_500, 256), (2_000, 256), (4_000, 256),
             (1_000, 512), (2_000, 512)],
            ("rbf", "poly"),
            (0.1, 1.0, 10.0, 100.0),
        )
        + _svc_points(
            [(4_000, 16), (4_000, 32), (8_000, 32), (1_500, 128), (8_000, 128), (8_000, 256)],
            ("rbf", "poly"),
            (1.0, 10.0),
        ),
        layout="svc",
        iterations=5,
    ),
    "svc-sigmoid": Grid(
        "Binary `SVC` fits with the sigmoid kernel on MLX against LIBSVM, C = 0.1-100: "
        "why patched dispatch never sends it to MLX.",
        _svc_points(
            [(4_000, 16), (4_000, 64), (2_000, 128), (4_000, 128), (1_000, 256),
             (4_000, 256), (8_000, 256), (2_000, 512)],
            ("sigmoid",),
            (0.1, 1.0, 10.0, 100.0),
        ),
        layout="svc",
        iterations=5,
    ),
    "svc-linear": Grid(
        "Binary `SVC` fits with the linear kernel on MLX against LIBSVM.",
        _svc_points([(2_000, 20), (2_000, 256)], ("linear",), (1.0,)),
        layout="svc",
        iterations=5,
    ),
    "svc-narrow": Grid(
        "Binary rbf `SVC` fits at 2-8 features, past the kernel cache (about 7 240 rows "
        "at `cache_size=200`) and below it.",
        _svc_points(
            [(n, d) for d in (2, 4, 8) for n in (2_000, 4_000, 8_000, 16_000)], ("rbf",), (1.0,)
        ),
        layout="svc",
        iterations=5,
    ),
    "svc-large": Grid(
        "Binary rbf `SVC` fits past the kernel cache at 256 features. A stock fit at "
        "32 000 rows takes nearly two minutes, so three pairs, too few for the "
        "significance test: the ratio is the evidence.",
        _svc_points([(16_000, 256), (32_000, 256)], ("rbf",), (1.0,)),
        layout="svc",
        iterations=3,
    ),
    "svc-multi": Grid(
        "One-vs-one `SVC` fits, rbf, C = 1, on well-separated Gaussian blobs (the "
        "benchmark's multiclass data): the `svc_min_pair_work` floor.",
        tuple(
            Point("svc", n, d, n_classes=K, data="separated-blobs")
            for n, d in ((4_000, 32), (8_000, 32), (8_000, 256), (2_048, 512))
            for K in (4, 8, 16, 32, *((64,) if (n, d) == (2_048, 512) else ()))
        ),
        layout="svc",
        iterations=5,
    ),
    "logreg-binary": Grid(
        "Binary `LogisticRegression` fits (y = x0 + noise > 0, max_iter=200) on MLX "
        "against lbfgs: the binary row and width limits.",
        tuple(
            Point("logreg", n, d, data="binary")
            for n in (5_000, 20_000, 50_000, 100_000, 200_000, 1_000_000)
            for d in (1, 4, 16, 32, 64, 128, 256, 1_024)
            if n * d <= 2.6e8 and not (n == 5_000 and d > 32)
        ),
        layout="logreg",
    ),
    "logreg-multi": Grid(
        "Multinomial `LogisticRegression` fits (max_iter=200) on MLX against lbfgs: "
        "overlapping Gaussian blobs, one per class, whose separation does not grow with "
        "the width: the sample floor.",
        tuple(
            Point("logreg", n, d, n_classes=K, data="blobs")
            for n in (5_000, 20_000, 60_000, 200_000)
            for K, d in ((3, 1), (3, 4), (3, 16), (3, 64), (3, 512), (5, 64), (10, 16),
                         (10, 128))
        ),
        layout="logreg",
    ),
    "logreg-separable": Grid(
        "Multinomial `LogisticRegression` fits (max_iter=200) on well-separated blobs, "
        "benchmarks/run.py's multiclass data, whose separation grows with the width: "
        "from 256 features scikit-learn converges in one iteration.",
        tuple(
            Point("logreg", n, d, n_classes=K, data="separated-blobs")
            for n in (60_000, 200_000)
            for K, d in ((3, 16), (3, 64), (3, 128), (3, 256), (3, 512), (5, 64), (5, 256),
                         (10, 128), (10, 512))
        ),
        layout="logreg",
    ),
    "logreg-labels": Grid(
        "`LogisticRegression` fits (max_iter=200) whose labels carry little or no signal "
        "— pure noise, binary and multinomial, and x0 + N(0, 3²) > 0 — so both solvers "
        "stop within a few iterations and the fixed cost of the MLX path is most of the "
        "fit: the binary row floor and width limit, and the multinomial width floor.",
        tuple(
            Point("logreg", n, d, data=data)
            for data in ("random", "noisy")
            for n in (100_000, 200_000, 500_000, 1_000_000)
            for d in (1, 4, 16, 32, 64, 256)
        )
        + tuple(
            Point("logreg", n, d, n_classes=K, data="random")
            for n in (60_000, 100_000, 200_000)
            for K, d in ((3, 1), (3, 2), (5, 1), (5, 2), (10, 1), (3, 4), (3, 16), (3, 64),
                         (3, 256), (10, 64))
        )
        + tuple(
            Point("logreg", n, 1, n_classes=K, data="random")
            for n, K in ((60_000, 6), (60_000, 7), (60_000, 8), (60_000, 9), (120_000, 3),
                         (150_000, 3))
        ),
        layout="logreg-data",
    ),
    "logreg-params": Grid(
        "`LogisticRegression` fits away from the default `tol` and `max_iter`, on the "
        "benchmark's binary target, on labels with moderate multinomial-logit signal and "
        "on random labels, binary at 200 000 rows and three classes at 60 000: a looser "
        "`tol` or a smaller `max_iter` stops both solvers within an iteration or two, "
        "where the fit is mostly MLX's fixed cost — the rule that `tol` be at most the "
        "default, 1e-4, and `logreg_min_max_iter`. 21 pairs each: at 9 the ratios near 1 "
        "did not reproduce.",
        tuple(
            Point("logreg", n, d, n_classes=K, data=data, tol=tol, max_iter=max_iter)
            for data, K, shapes in (
                ("binary", 2, ((200_000, 1), (200_000, 4), (200_000, 16), (200_000, 64))),
                ("signal", 2, ((200_000, 1), (200_000, 4), (200_000, 16), (200_000, 64))),
                ("random", 2, ((200_000, 16), (200_000, 64))),
                ("signal", 3, ((60_000, 2), (60_000, 8), (60_000, 64))),
                ("blobs", 3, ((60_000, 16),)),
            )
            for n, d in shapes
            for tol, max_iter in ((0.0, 0), (1e-5, 0), (5e-4, 0), (1e-3, 0), (3e-3, 0),
                                  (1e-2, 0), (0.0, 1), (0.0, 2), (0.0, 3), (0.0, 5))
        ),
        layout="logreg-params",
        iterations=21,
    ),
    "svc-params": Grid(
        "Binary `SVC` fits on MLX against LIBSVM away from the default kernel parameters, "
        "at shapes the size floors send to MLX: rbf at gamma from 0.03 to 10 times "
        "scikit-learn's 'scale' value, poly at degree 2 to 4 and coef0 of 0 and 1, "
        "C = 1-100 — the poly rule, and the corners the `svc_max_c` rule covers.",
        tuple(
            Point("svc", n, d, kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0)
            for n, d in ((3_000, 128), (8_000, 128), (1_500, 256), (4_000, 256), (1_100, 512),
                         (4_000, 512))
            for kernel, gamma, degree, coef0 in (
                *(("rbf", g, 3, 0.0) for g in (0.03, 0.1, 0.3, 0.5, 0.8, 3.0, 10.0)),
                ("poly", 0.0, 2, 0.0),
                ("poly", 0.0, 2, 1.0),
                ("poly", 0.0, 3, 1.0),
                ("poly", 0.0, 4, 0.0),
            )
            for C in (1.0, 10.0, 100.0)
        ),
        layout="svc-params",
        iterations=5,
    ),
    "svc-multi-overlap": Grid(
        "One-vs-one `SVC` fits, rbf, on overlapping classes (the binary target cut into K "
        "quantile bands) at C = 1-100: one-vs-one pairs either side of the "
        "`svc_min_pair_work` floor.",
        tuple(
            Point("svc", n, d, C=C, n_classes=K, data="mixed")
            for n, d, K in ((3_000, 128, 3), (4_500, 128, 3), (4_000, 128, 4), (6_000, 128, 4),
                            (7_500, 128, 5), (15_000, 128, 10), (2_000, 256, 4), (3_000, 256, 4),
                            (1_536, 512, 3), (2_048, 512, 4))
            for C in (1.0, 10.0, 100.0)
        ),
        layout="svc-multi-c",
        iterations=5,
    ),
    "svc-predict": Grid(
        "`predict` on an rbf `SVC` fitted on MLX (C = 1) against one fitted by LIBSVM, "
        "every batch size from 1 to 4096 rows: the `svc_host_inference_work` threshold "
        "below which the batch is computed on the host.",
        tuple(
            Point("svc", n, d, (1, 2, 4, 8, 16, 32, 64, 128, 256, 1_024, 4_096), n_classes=K,
                  data="binary" if K == 2 else "mixed")
            for n, d, K in ((3_000, 128, 2), (1_500, 256, 2), (4_000, 256, 2), (1_100, 512, 2),
                            (6_000, 512, 2), (8_000, 256, 4))
        ),
        layout="svc-predict",
        iterations=15,
    ),
    "knn-data": Grid(
        "Neighbor queries on MLX against stock brute force on data other than Gaussian, "
        "at wide shapes the size rules send to MLX and at a million rows and 16 or 32 "
        "features: one column on a scale 10⁴ times the rest, "
        "rows duplicated about 20 times, 100 tight clusters, binary and count features, "
        "and three controls (Gaussian, looser clusters, uniform). `probe` is what the "
        "fit-time probe found: the `knn_probe_max_uncertified` and integer-data rules.",
        tuple(
            Point("knn", n, d, (1, 16, 64, 256, 1_024), data=data)
            for n, d in ((24_000, 256), (65_536, 256), (8_192, 1_024), (1_000_000, 16),
                         (1_000_000, 32))
            for data in ("gauss", "unscaled", "dups", "tight-blobs", "binary", "counts",
                         "blobs", "uniform")
        ),
        layout="knn-data",
    ),
    "knn-narrow": Grid(
        "Neighbor queries on MLX against stock brute force at a million rows and more "
        "below 256 features, on Gaussian data and two other laws the fit-time probe "
        "passes (100 clusters of std 1, uniform): the `knn_min_features` rule.",
        tuple(
            point
            for data in ("gauss", "blobs", "uniform")
            for point in _knn_points(
                [(1_000_000, 16), (1_000_000, 32), (1_000_000, 48), (1_000_000, 64),
                 (1_000_000, 96), (1_000_000, 128), (1_600_000, 32), (1_600_000, 64)],
                (1, 4, 16, 64, 256, 1_024),
                max_work=1.2e11,
                data=data,
            )
        ),
        layout="knn-data",
    ),
    "knn-boundary": Grid(
        "Neighbor queries on MLX against stock brute force between the `knn` grid's 262 144 "
        "and 1 000 000 rows, at 16, 48 and 128 features, Gaussian data: the largest batch "
        "whose query rows times training rows stay below 2²², which MLX's own selection "
        "answers instead of the fused kernel — 15, 10 and 6 queries — beside 8. At 262 144 "
        "rows the `knn` grid's batches of 8 and 16 straddle it: the "
        "`knn_narrow_min_samples` rule.",
        tuple(
            Point("knn", n, d, tuple(sorted({-(-_FUSED_MIN_PAIRS // n) - 1, 8})))
            for n in (262_144, 400_000, 600_000)
            for d in (16, 48, 128)
        ),
        layout="knn-q",
    ),
    "knn-k": Grid(
        "Neighbor queries on MLX against stock brute force at k = 1-200 (the other grids "
        "use k = 10), every batch size from 1 to 4096: at a million rows and more from 16 "
        "to 128 features, and on short, wide training sets up to 4 096 features — the "
        "`knn_max_neighbors` rule, and k for the `knn_min_features` rule.",
        tuple(
            point
            for shapes, ks in (
                (((1_000_000, 16), (1_000_000, 32), (1_000_000, 48), (1_600_000, 32),
                  (1_000_000, 64), (1_000_000, 96), (1_000_000, 128), (1_600_000, 64)),
                 (1, 5, 10, 50, 100, 200)),
                (((24_000, 256), (100_000, 256), (8_192, 1_024), (6_000, 2_048), (4_096, 2_048),
                  (4_096, 3_072), (4_096, 4_096)),
                 (5, 10, 50, 100, 200)),
                # Either side of the rows-per-neighbor floor, at the widest shapes.
                (((4_096, 3_072), (4_096, 4_096)), (40,)),
                (((5_000, 3_072), (5_000, 4_096)), (50,)),
                (((6_144, 4_096),), (40,)),
                (((7_500, 4_096),), (50,)),
            )
            for k in ks
            for point in _knn_points(shapes, DENSE_QUERIES, max_work=1.2e11, k=k)
        ),
        layout="knn-k-worst",
    ),
    "knn-structured": Grid(
        "Neighbor queries on MLX against stock brute force on data with mild structure — "
        "rows repeated about twice, one column on 100 times the scale of the rest, 100 "
        "clusters of std 0.4 and 0.5 — beside Gaussian data, at shapes on and near the "
        "size floors and at a million rows from 16 features, every batch size from 1 to "
        "4096: few enough rows go uncertified that most of a 16-64-query batch is not, and "
        "most batches still hold one. `probe` is what the fit-time probe found: the "
        "`knn_probe_max_uncertified` rule, and data for the `knn_min_features` rule.",
        tuple(
            point
            for data in ("gauss", "dup2", "scaled100", "blobs0.5", "blobs0.4")
            for point in _knn_points(
                [(24_000, 256), (30_000, 256), (65_536, 256), (100_000, 256), (12_000, 512),
                 (50_000, 512), (16_384, 768), (8_192, 1_024), (4_096, 2_048),
                 (1_000_000, 16), (1_000_000, 32), (1_000_000, 48), (1_000_000, 64)],
                DENSE_QUERIES,
                max_work=1.2e11,
                data=data,
            )
        ),
        layout="knn-data-worst",
    ),
    "svc-hard": Grid(
        "`SVC` fits on MLX against LIBSVM where nearly every row becomes a support vector: "
        "labels with little or no signal (x0 + N(0, 3²) > 0, random, XOR), binary and "
        "one-vs-one, and rbf kernels far narrower than scikit-learn's 'scale' (gamma 10 "
        "to 1 000 times it), C = 1-100, either side of the width floor: the "
        "`svc_min_features` rule, and corners the `svc_max_c` rule covers.",
        tuple(
            Point("svc", n, d, C=C, data=data)
            for n, d in ((3_000, 128), (6_000, 128), (8_000, 128), (1_500, 256), (2_500, 256),
                         (4_000, 256), (1_024, 512), (2_000, 512))
            for data in ("noisy", "random", "xor")
            for C in (1.0, 10.0, 100.0)
        )
        + tuple(
            Point("svc", n, d, kernel="poly", C=C, data="noisy")
            for n, d in ((3_000, 128), (1_500, 256), (4_000, 256), (1_024, 512))
            for C in (1.0, 10.0, 100.0)
        )
        + tuple(
            Point("svc", n, d, C=C, n_classes=K, data="random")
            for n, d, K in ((4_500, 128, 3), (6_000, 128, 4), (2_250, 256, 3), (3_000, 256, 4),
                            (7_500, 256, 10), (1_536, 512, 3), (2_048, 512, 4))
            for C in (1.0, 10.0, 100.0)
        )
        + tuple(
            Point("svc", n, d, C=C, gamma=gamma)
            for n, d in ((3_000, 128), (1_500, 256), (4_000, 256), (1_024, 512), (1_100, 512),
                         (2_000, 512))
            for gamma in (10.0, 30.0, 100.0, 1_000.0)
            for C in (1.0, 10.0)
        ),
        layout="svc-hard",
        iterations=5,
    ),
    "svc-structured": Grid(
        "Binary `SVC` fits on MLX against LIBSVM on data whose intrinsic dimension is not "
        "its width — five or ten latent directions plus noise, singular values 1/i, "
        "features smoothed over 16 neighbours, Student-t(2) features, column scales "
        "10^U(0, 2), and scikit-learn's digits upsampled to 16², 20² and 28² pixels — at "
        "C = 0.1-10, and at C = 1 away from the defaults (gamma 0.03-100 x 'scale', the "
        "poly kernel, tol = 1e-5, balanced class weights on 10% positives, 8 000 rows): "
        "the `svc_max_c` rule.",
        tuple(
            Point("svc", n, d, C=C, data=data)
            for data in ("lowrank", "lowrank10", "decay", "smooth", "t2", "colscale")
            for n, d in ((1_500, 256), (4_000, 256), (1_024, 512), (2_000, 512))
            for C in (0.1, 1.0, 10.0)
        )
        + tuple(
            Point("svc", 1_797, side * side, C=C, data=f"digits{side}")
            for side in (16, 20, 28)
            for C in (0.1, 1.0, 10.0)
        )
        + tuple(
            Point("svc", 1_797, 784, C=C, n_classes=3, data="digits28") for C in (1.0, 10.0)
        )
        + tuple(
            Point("svc", n, d, gamma=gamma, data=data)
            for data in ("lowrank", "decay")
            for n, d in ((1_500, 256), (1_024, 512))
            for gamma in (0.03, 0.3, 10.0, 100.0)
        )
        + tuple(
            Point("svc", n, d, kernel="poly", C=C, data=data)
            for data in ("lowrank", "decay")
            for n, d in ((1_500, 256), (1_024, 512))
            for C in (1.0, 10.0)
        )
        + tuple(
            Point("svc", n, d, tol=1e-5, data=data)
            for data in ("lowrank", "decay")
            for n, d in ((1_500, 256), (1_024, 512))
        )
        + tuple(
            Point("svc", n, d, C=C, class_weight=weight, data="lowrank-imb")
            for n, d in ((1_500, 256), (1_024, 512))
            for C, weight in ((1.0, ""), (0.1, "balanced"), (0.2, "balanced"),
                              (1.0, "balanced"))
        )
        + tuple(Point("svc", 8_000, 256, data=data) for data in ("lowrank", "decay")),
        layout="svc-structured",
        iterations=5,
    ),
    "knn-probe": Grid(
        "Neighbor queries on MLX against stock brute force on many draws of one law: 100 "
        "clusters of std 0.5, at the two shapes nearest the size floor that `knn-structured` "
        "measures it at, where about one query row in a hundred cannot be certified — few "
        "enough that a probe of 256 rows sometimes certifies every one — and Gaussian, "
        "low-rank and std-1 cluster draws beside them, there and at 8 192 × 1 024, batches "
        "of 16-256 queries. Each draw records what a probe of `knn_probe_rows` rows found "
        "and what one of 256 found: the `knn_probe_rows` rule.",
        tuple(
            Point("knn", n, d, _PROBE_QUERIES, data=data, data_seed=seed, probe_also=256)
            for data, draws, shapes in (
                ("blobs0.5", _PROBE_DRAWS, ((24_000, 256), (30_000, 256))),
                *((law, _PROBE_CONTROLS, ((24_000, 256), (30_000, 256), (8_192, 1_024)))
                  for law in ("gauss", "lowrank", "blobs")),
            )
            for n, d in shapes
            for seed in range(draws)
        ),
        layout="knn-probe",
    ),
    "knn-batches": Grid(
        "Neighbor queries on MLX against stock brute force at every batch size from 8 to "
        "64 on short, wide training sets, 4 096-8 192 rows at 2 048-4 096 features, k = 10 "
        "and at the most neighbors the rules allow: right after a stock query, whose "
        "OpenMP threads are still spinning, an MLX call of some batch sizes — about 256 KiB "
        "of float32 queries — runs up to twice as slow as its neighbours. The "
        "`knn_min_train_samples` row floor.",
        tuple(
            point
            for shapes, k in (
                (((4_096, 2_048), (4_096, 2_560), (4_096, 3_072), (4_500, 2_304),
                  (5_000, 2_048), (5_000, 4_096), (6_000, 2_048), (6_000, 3_072),
                  (6_000, 4_096), (7_500, 4_096), (8_192, 2_048), (8_192, 3_072),
                  (8_192, 4_096), (12_000, 2_048)), 10),
                (((6_000, 3_072), (6_000, 4_096)), 40),
                (((7_500, 4_096), (8_192, 2_048), (8_192, 4_096)), 50),
            )
            for point in _knn_points(shapes, _HUMP_QUERIES, k=k)
        ),
        layout="knn-k-worst",
    ),
    "knn-query-k": Grid(
        "`kneighbors(n_neighbors=k)` on a model fitted with k = 10, against the same query "
        "on a stock model, at shapes patched dispatch fits on MLX, float64 and float32 "
        "data: k = 50 on MLX, and k = 51-200 three ways — answered as patched dispatch "
        "answers it, by scikit-learn's brute force over a float64 copy of the training "
        "rows (steady state, and the first such query, which builds the copy), and on "
        "MLX. The `knn_max_neighbors` rule at query time. 21 pairs each: at 9 the "
        "handoff's ratios, all near 1, did not reproduce.",
        tuple(
            Point("knn", n, d, queries, dtype=dtype, query_k=k, path=path)
            for n, d in ((8_192, 1_024), (8_192, 2_048), (65_536, 256), (1_000_000, 64))
            for dtype in ("float64", "float32")
            for k, path, queries in (
                (50, "", _QUERY_K_BATCHES),
                *(
                    (k, path, _QUERY_K_FIRST if path == "first" else _QUERY_K_BATCHES)
                    for k in (51, 60, 100, 200)
                    for path in ("", "first", "mlx")
                ),
            )
        ),
        layout="knn-query-k",
        iterations=21,
    ),
    "svc-poly": Grid(
        "Binary and one-vs-one `SVC` fits with the poly kernel (degree 3, coef0 = 0) on "
        "MLX against LIBSVM at C = 0.2-1, on data the other grids do not hold — "
        "scikit-learn's digits upsampled to 16², 20² and 28² pixels (digit >= 5, its "
        "parity, digit mod 4), five latent directions plus noise with random labels, "
        "Cauchy features — and on five latent directions at gamma 1.25 and 1.5 times "
        "'scale'; the rbf kernel beside it at C = 1: why patched dispatch never sends "
        "poly to MLX.",
        tuple(
            Point("svc", 1_797, side * side, kernel=kernel, C=C, n_classes=K, data=data)
            for data, side, K in (("digits16", 16, 2), ("digits16-parity", 16, 2),
                                  ("digits20", 20, 2), ("digits28", 28, 4))
            for kernel, Cs in (("poly", (0.2, 0.5, 1.0)), ("rbf", (1.0,)))
            for C in Cs
        )
        + tuple(
            Point("svc", n, d, kernel=kernel, C=C, data=data)
            for data in ("lowrank-rand", "cauchy")
            for n, d in ((1_500, 256), (3_000, 256), (1_024, 512))
            for kernel, Cs in (("poly", (0.5, 1.0)), ("rbf", (1.0,)))
            for C in Cs
            # At 1 500 x 256 and 3 000 x 256 these Cauchy rows are too far from the
            # origin for the poly kernel's float32 inner product (UnsupportedInputError:
            # a capability limit, which patched mode hands to LIBSVM): nothing to time.
            if not (data == "cauchy" and kernel == "poly" and d == 256)
        )
        + tuple(
            Point("svc", n, d, kernel="poly", gamma=gamma, data="lowrank")
            for n, d in ((1_500, 256), (2_000, 512))
            for gamma in (1.25, 1.5)
        ),
        layout="svc-structured",
        iterations=7,
    ),
    "svc-cache": Grid(
        "Binary rbf `SVC` fits on MLX against LIBSVM at C = 1 with `cache_size` from 1 MB "
        "to 1 000 MB, both sides given the same, on five latent directions plus noise and "
        "on the benchmark's Gaussian target: the rule that `cache_size` be at least "
        "scikit-learn's default, 200 MB.",
        tuple(
            Point("svc", n, d, cache_size=cache, data=data)
            for data, n, d, caches in (
                ("lowrank", 4_000, 256, (1.0, 3.0, 10.0, 40.0, 100.0, 0.0, 1_000.0)),
                ("lowrank", 3_000, 512, (1.0, 10.0, 0.0)),
                ("lowrank", 8_000, 256, (1.0, 0.0, 1_000.0)),
                ("binary", 4_000, 256, (1.0, 10.0, 0.0, 1_000.0)),
            )
            for cache in caches
        ),
        layout="svc-cache",
        iterations=7,
    ),
}


# ----------------------------------------------------------------------------- data


def _knn_data(
    point: Point, q: int, rng: np.random.Generator, *, train: bool = False
) -> np.ndarray:
    """``q`` rows of ``point.data``; training rows and queries come from the same law.

    Except ``"dups"`` and ``"dup2"``, whose training rows repeat and whose queries are
    fresh rows.
    """
    d = point.d
    fixed = np.random.default_rng(point.seed + 1)  # what training and queries share
    if point.data == "lowrank":
        basis = fixed.normal(size=(3, d))
        return rng.normal(size=(q, 3)) @ basis + 0.05 * rng.normal(size=(q, d))
    if point.data == "unscaled":  # one column in raw units, 0-10 000, among N(0, 1)
        X = rng.normal(size=(q, d))
        X[:, 0] = rng.uniform(0.0, 1e4, size=q)
        return X
    if point.data == "scaled100":  # one column on 100 times the scale of the rest
        X = rng.normal(size=(q, d))
        X[:, 0] *= 100.0
        return X
    if point.data in _KNN_REPEATS and train:  # every row repeated about 20 or 2 times
        base = rng.normal(size=(max(q // _KNN_REPEATS[point.data], 1), d))
        return base[rng.integers(0, base.shape[0], size=q)]
    if point.data in _KNN_BLOBS:  # 100 clusters around N(0, 3²) centres
        centers = fixed.normal(scale=3.0, size=(100, d))
        std = _KNN_BLOBS[point.data]
        return centers[rng.integers(0, 100, size=q)] + std * rng.normal(size=(q, d))
    if point.data == "binary":
        return (rng.random(size=(q, d)) < 0.2).astype(np.float64)
    if point.data == "counts":
        return rng.poisson(2.0, size=(q, d)).astype(np.float64)
    if point.data == "uniform":
        return rng.random(size=(q, d))
    if point.data not in ("gauss", *_KNN_REPEATS):
        raise ValueError(f"unknown data {point.data!r}")
    return rng.normal(size=(q, d))


#: Neighbor data whose training rows repeat, and about how many times each.
_KNN_REPEATS = {"dups": 20, "dup2": 2}

#: Neighbor data drawn from 100 clusters, and the clusters' standard deviation.
_KNN_BLOBS = {"tight-blobs": 0.1, "blobs0.4": 0.4, "blobs0.5": 0.5, "blobs": 1.0}


def _fit_data(point: Point) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(point.seed)
    n, d, K = point.n, point.d, point.n_classes
    if point.data == "binary":  # benchmarks/run.py's binary target
        X = rng.normal(size=(n, d))
        return X, (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    if point.data == "noisy":  # the same rule, with six times the noise
        X = rng.normal(size=(n, d))
        return X, (X[:, 0] + rng.normal(scale=3.0, size=n) > 0).astype(int)
    if point.data == "random":  # labels independent of X
        return rng.normal(size=(n, d)), rng.integers(0, K, size=n)
    if point.data == "xor":  # the signs of two features, XORed: no linear signal
        X = rng.normal(size=(n, d))
        return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    if point.data == "mixed":  # the binary target's score cut into K quantile bands
        X = rng.normal(size=(n, d))
        score = X[:, 0] + rng.normal(scale=0.5, size=n)
        return X, np.digitize(score, np.quantile(score, np.linspace(0, 1, K + 1)[1:-1]))
    if point.data == "signal":  # moderate multinomial-logit signal, K classes
        X = rng.normal(size=(n, d))
        W = rng.normal(size=(d, K))
        return X, np.argmax(X @ W / np.sqrt(d) + rng.gumbel(size=(n, K)), axis=1)
    if point.data.startswith("digits"):
        return _digits(point)
    if point.data in _STRUCTURED:
        return _structured(point.data, n, d, rng)
    y = rng.integers(0, K, size=n)
    if point.data == "separated-blobs":  # benchmarks/run.py's n_classes target
        centers = rng.normal(scale=2.0, size=(K, d))
    elif point.data == "blobs":  # overlapping at every width
        centers = rng.normal(scale=2.0, size=(K, d)) / np.sqrt(d) * 3
    else:
        raise ValueError(f"unknown data {point.data!r}")
    return centers[y] + rng.normal(size=(n, d)), y


#: Binary fit data whose features are not isotropic Gaussian, and so whose intrinsic
#: dimension is not its width (_structured).
_STRUCTURED = ("lowrank", "lowrank10", "lowrank-imb", "lowrank-rand", "decay", "smooth",
               "t2", "colscale", "cauchy")


def _structured(law: str, n: int, d: int, rng: np.random.Generator):
    """Binary targets on data of lower intrinsic dimension, or heavier tails, than N(0, I).

    ``lowrank``/``lowrank10``: 5 or 10 latent Gaussian directions plus N(0, 0.3²) noise
    in every feature, labelled by the first latent plus N(0, 1) — ``lowrank-imb`` the
    same, cut so that about 10% are positive, ``lowrank-rand`` labelled at random.
    ``decay``: full rank with singular values 1/i, labelled by the top direction.
    ``smooth``: each feature a moving average of 16 adjacent white-noise columns.
    ``t2``: Student-t features with 2 degrees of freedom. ``colscale``: column scales
    10^U(0, 2). ``cauchy``: standard Cauchy features, labelled by tanh(x0) plus
    N(0, 0.5²).
    """
    if law.startswith("lowrank"):
        rank = 10 if law == "lowrank10" else 5
        Z = rng.normal(size=(n, rank))
        X = Z @ rng.normal(size=(rank, d)) + 0.3 * rng.normal(size=(n, d))
        if law == "lowrank-rand":
            return X, rng.integers(0, 2, size=n)
        cut = 1.8 if law == "lowrank-imb" else 0.0
        return X, (Z[:, 0] + rng.normal(size=n) > cut).astype(int)
    if law == "decay":
        scale = 1.0 / np.arange(1, d + 1)
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        Z = rng.normal(size=(n, d)) * scale
        return Z @ Q.T, (Z[:, 0] / scale[0] + rng.normal(size=n) > 0).astype(int)
    if law == "smooth":
        c = np.cumsum(rng.normal(size=(n, d + 15)), axis=1)
        X = (c[:, 15:] - np.concatenate([np.zeros((n, 1)), c[:, :-16]], axis=1)) / 4.0
        return X, (X[:, d // 2] + rng.normal(size=n) > 0).astype(int)
    if law == "t2":
        X = rng.standard_t(2.0, size=(n, d))
        return X, (X[:, 0] + rng.normal(size=n) > 0).astype(int)
    if law == "colscale":
        X = rng.normal(size=(n, d)) * 10.0 ** rng.uniform(0.0, 2.0, size=d)
        return X, (X[:, 0] / X[:, 0].std() + rng.normal(size=n) > 0).astype(int)
    if law == "cauchy":
        X = rng.standard_cauchy(size=(n, d))
        return X, (np.tanh(X[:, 0]) + 0.5 * rng.normal(size=n) > 0).astype(int)
    raise ValueError(f"unknown data {law!r}")


def _digits(point: Point):
    """scikit-learn's 8 x 8 digits, 1 797 images upsampled bilinearly to S x S.

    ``point.data`` is ``"digits<S>"``, binary ``digit >= 5`` and K classes ``digit % K``,
    or ``"digits<S>-parity"``, binary ``digit % 2``.
    """
    from scipy.ndimage import zoom
    from sklearn.datasets import load_digits

    name, _, labels = point.data.partition("-")
    side = int(name.removeprefix("digits"))
    digits = load_digits()
    X = np.stack([zoom(image / 16.0, side / 8.0, order=1) for image in digits.images])
    X = X.reshape(len(X), -1)
    if (point.n, point.d) != X.shape:
        raise ValueError(f"{point.data} is {X.shape[0]} x {X.shape[1]}, not {point.n} x {point.d}")
    K = point.n_classes
    if labels == "parity":
        return X, digits.target % 2
    return X, (digits.target >= 5).astype(int) if K == 2 else digits.target % K


# -------------------------------------------------------------------------- running


def _forced():
    from mlxlearn._common.config import tuning_context

    return tuning_context(force_backend="mlx")


def _summary(comparison) -> dict:
    return {
        "mlxlearn_ms": comparison.mlxlearn.median_s * 1e3,
        "mlxlearn_iqr_ms": comparison.mlxlearn.iqr_s * 1e3,
        "mlxlearn_min_ms": comparison.mlxlearn.min_s * 1e3,
        "sklearn_ms": comparison.sklearn.median_s * 1e3,
        "sklearn_iqr_ms": comparison.sklearn.iqr_s * 1e3,
        "sklearn_min_ms": comparison.sklearn.min_s * 1e3,
        "speedup": comparison.speedup,
        "speedup_best": comparison.speedup_best,
        "significant": comparison.significant,
        "iterations": comparison.mlxlearn.iterations,
    }


def _versions() -> dict:
    import mlx.core as mx
    import sklearn

    import mlxlearn

    return {
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mlx": getattr(mx, "__version__", "unknown"),
        "mlxlearn": mlxlearn.__version__,
    }


def _measure_knn(point: Point, iterations: int, todo: set[str]):
    if point.query_k:
        yield from _measure_query_k(point, iterations, todo)
        return
    import sklearn.neighbors

    from mlxlearn.patching import _estimators as patched

    rng = np.random.default_rng(point.seed)
    X = _knn_data(point, point.n, rng, train=True)

    def make_mlx():
        return patched.NearestNeighbors(n_neighbors=point.k, algorithm=point.algorithm)

    def make_sk():
        return sklearn.neighbors.NearestNeighbors(n_neighbors=point.k, algorithm=point.algorithm)

    def mlx_fit():
        with _forced():
            return make_mlx().fit(X)

    model, stock = mlx_fit(), make_sk().fit(X)
    assert model.execution_backend_ == "mlx", model.fit_event_
    # What patched dispatch's fit-time probe finds on this data: forcing the backend
    # skips the probe, so it is taken here, and dispatch() applies the thresholds.
    from mlxlearn._common.config import get_tuning

    found = model._index_.probe(point.k, rows=get_tuning().knn_probe_rows)
    extra = {
        "stock_method": stock._fit_method,
        "probe_rows": found["rows"],
        "probe_uncertified": found["uncertified"],
        "integer_path": found["integer"],
    }
    if point.probe_also:
        also = model._index_.probe(point.k, rows=point.probe_also)
        extra |= {"probe_also_rows": also["rows"], "probe_also_uncertified": also["uncertified"]}
    keys = point.keys()
    if point.fit:
        key = keys.pop(0)
        if key in todo:
            result = compare(mlx_fit, lambda: make_sk().fit(X), label=key,
                             shape=(point.n, point.d), iterations=min(iterations, 7))
            yield key, {"operation": "fit", **extra, **_summary(result)}
    for q, key in zip(point.queries, keys, strict=True):
        if key not in todo:
            continue
        if q == 1:
            # A different row for every pair, the same one on both sides: what one
            # row costs a tree depends on where it lands, so a single fixed row can
            # stand for the tree's best case or its worst.
            batches = _knn_data(point, _WARMUP + iterations, rng)[:, None, :]
        else:
            batches = _knn_data(point, q, rng)[None, :, :]
        result = compare(_queries(model, batches), _queries(stock, batches), label=key,
                         shape=(point.n, point.d), iterations=iterations, warmup=_WARMUP)
        yield key, {"operation": "kneighbors", "q": q, **extra, **_summary(result)}


def _queries(model, batches: np.ndarray, k: int | None = None) -> Callable[[], object]:
    """``model.kneighbors`` on the next of ``batches`` at each call, cycling."""
    position = iter(range(1 << 62))

    def query():
        return model.kneighbors(batches[next(position) % len(batches)], n_neighbors=k)

    return query


def _measure_query_k(point: Point, iterations: int, todo: set[str]):
    """``kneighbors(n_neighbors=point.query_k)`` on a model fitted with ``point.k``.

    The fit is patched dispatch's own, unforced, and must land on MLX — except for
    ``path="mlx"``, whose forced fit keeps its queries there too. ``path="first"``
    drops the float64 copy a large-k query builds, and the stock model built over it,
    before every call, outside the timed window.
    """
    import sklearn.neighbors

    from mlxlearn.patching import _estimators as patched

    rng = np.random.default_rng(point.seed)
    dtype = np.dtype(point.dtype)
    X = _knn_data(point, point.n, rng, train=True).astype(dtype)
    if point.path == "mlx":
        with _forced():
            model = patched.NearestNeighbors(n_neighbors=point.k).fit(X)
    else:
        model = patched.NearestNeighbors(n_neighbors=point.k).fit(X)
    assert model.execution_backend_ == "mlx", model.fit_event_
    stock = sklearn.neighbors.NearestNeighbors(n_neighbors=point.k).fit(X)
    setup = None
    if point.path == "first":

        def setup():
            model.__dict__.pop("_host_brute_", None)
            model._index_._host64 = None

    for q, key in zip(point.queries, point.keys(), strict=True):
        if key not in todo:
            continue
        if q == 1:
            batches = _knn_data(point, _WARMUP + iterations, rng)[:, None, :].astype(dtype)
        else:
            batches = _knn_data(point, q, rng)[None, :, :].astype(dtype)
        result = compare(_queries(model, batches, point.query_k),
                         _queries(stock, batches, point.query_k), label=key,
                         shape=(point.n, point.d), iterations=iterations, warmup=_WARMUP,
                         mlx_setup=setup)
        answered = "sklearn" if "_host_brute_" in model.__dict__ else "mlx"
        yield key, {"operation": "kneighbors", "q": q, "answered_by": answered,
                    **_summary(result)}


def _measure_fit(point: Point, iterations: int, todo: set[str]):
    import sklearn.linear_model
    import sklearn.svm

    from mlxlearn.patching import _estimators as patched

    keys = point.keys()
    if not set(keys) & todo:
        return
    X, y = _fit_data(point)
    if point.estimator == "svc":
        params = {"kernel": point.kernel, "C": point.C, "degree": point.degree,
                  "coef0": point.coef0}
        if point.gamma:
            # A multiple of the "scale" value: the ratio the dispatch rule reads.
            params["gamma"] = point.gamma / (point.d * float(X.var()))
        if point.tol:
            params["tol"] = point.tol
        if point.class_weight:
            params["class_weight"] = point.class_weight
        if point.cache_size:
            params["cache_size"] = point.cache_size
        make_mlx, make_sk = (lambda: patched.SVC(**params)), (lambda: sklearn.svm.SVC(**params))
        if point.queries:
            yield from _measure_predict(point, X, y, make_mlx, make_sk, iterations, todo)
            return
    else:
        params = {"max_iter": point.max_iter or 200, "C": point.C}
        if point.tol:
            params["tol"] = point.tol
        make_mlx = lambda: patched.LogisticRegression(**params)  # noqa: E731
        make_sk = lambda: sklearn.linear_model.LogisticRegression(**params)  # noqa: E731
    seen: dict = {}

    def mlx_fit():
        with _forced():
            model = make_mlx().fit(X, y)
        seen["backend"] = model.execution_backend_
        seen["n_iter"] = int(np.max(model.n_iter_))
        return model

    def sk_fit():
        model = make_sk().fit(X, y)
        seen["sklearn_n_iter"] = int(np.max(model.n_iter_))
        return model

    (key,) = keys
    start = time.perf_counter()
    mlx_fit()
    sk_fit()
    once = time.perf_counter() - start
    # A pair of fits that takes more than 20 s is timed fewer times: never under 5,
    # the fewest the significance test accepts, unless the grid asks for fewer.
    count = iterations if once < 20 else max(min(5, iterations), min(iterations, int(300 / once)))
    result = compare(mlx_fit, sk_fit, label=key, shape=(point.n, point.d), iterations=count,
                     warmup=1)
    assert seen["backend"] == "mlx", seen
    yield key, {"operation": "fit", **seen, **_summary(result)}


def _measure_predict(point: Point, X, y, make_mlx, make_sk, iterations: int, todo: set[str]):
    """``predict`` on a model fitted on MLX against one fitted by scikit-learn.

    The fit is forced onto MLX; ``predict`` then runs as dispatch configures it,
    on the host up to ``svc_host_inference_work`` and on the device above.
    """
    with _forced():
        model = make_mlx().fit(X, y)
    stock = make_sk().fit(X, y)
    assert model.execution_backend_ == "mlx", model.fit_event_
    rng = np.random.default_rng(point.seed + 2)
    n_sv = int(model.support_vectors_.shape[0])
    for q, key in zip(point.queries, point.keys(), strict=True):
        if key not in todo:
            continue
        count = _WARMUP + iterations if q <= 64 else 1
        picks = [rng.integers(0, point.n, size=q) for _ in range(count)]
        batches = np.stack([X[p] + 0.01 * rng.normal(size=(q, point.d)) for p in picks])
        result = compare(_calls(model, "predict", batches), _calls(stock, "predict", batches),
                         label=key, shape=(point.n, point.d), iterations=iterations,
                         warmup=_WARMUP)
        yield key, {"operation": "predict", "q": q, "n_sv": n_sv, **_summary(result)}


def _calls(model, method: str, batches: np.ndarray) -> Callable[[], object]:
    """``model.<method>`` on the next of ``batches`` at each call, cycling."""
    position = iter(range(1 << 62))

    def call():
        return getattr(model, method)(batches[next(position) % len(batches)])

    return call


def _done(path: Path, sklearn_version: str) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("sklearn") == sklearn_version:
                done.add(record["key"])
    return done


def run(name: str, output: Path, *, iterations: int | None = None, limit: int | None = None,
        echo: Callable[[str], None] = print) -> int:
    """Measure grid ``name`` into ``output``; return the number of new records."""
    warnings.simplefilter("ignore")
    grid = GRIDS[name]
    versions = _versions()
    iterations = iterations or grid.iterations
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _done(output, versions["sklearn"])
    written = 0
    for index, point in enumerate(grid.points, 1):
        todo = {key for key in point.keys() if key not in done}
        if not todo:
            continue
        measure = _measure_knn if point.estimator == "knn" else _measure_fit
        for key, record in measure(point, iterations, todo):
            record = {
                "grid": name, "key": key, "estimator": point.estimator, "n": point.n,
                "d": point.d, "algorithm": point.algorithm, "data": point.data,
                "kernel": point.kernel, "C": point.C, "n_classes": point.n_classes,
                "k": point.k, "gamma": point.gamma, "degree": point.degree,
                "coef0": point.coef0, "tol": point.tol, "max_iter": point.max_iter,
                "class_weight": point.class_weight, "cache_size": point.cache_size,
                "query_k": point.query_k, "dtype": point.dtype, "path": point.path,
                "data_seed": point.data_seed, **versions, "time": time.time(), **record,
            }
            with output.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            written += 1
            marker = "" if record["significant"] else " ~"
            echo(f"[{index}/{len(grid.points)}] {key}: {record['mlxlearn_ms']:.2f} / "
                 f"{record['sklearn_ms']:.2f} ms, {record['speedup']:.2f}x{marker}")
            if limit is not None and written >= limit:
                return written
    return written


# --------------------------------------------------------------------------- tables


def load(paths: Iterable[Path], grid: str | None = None) -> list[dict]:
    """The records in ``paths``, optionally only those of ``grid``."""
    records = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if grid is None or record.get("grid") == grid:
                    records.append(record)
    return records


def is_loss(record: dict) -> bool:
    """What the gate would fail: significant, and more than 5% slower."""
    return bool(record["significant"]) and record["speedup"] < 1.0 - _LOSS_TOLERANCE


def _cell(record: dict | None) -> str:
    if record is None:
        return "—"
    text = f"{record['speedup']:.2f}"
    if is_loss(record):
        return f"**{text}**"
    return text if record["significant"] else f"{text}~"


def dispatch(record: dict) -> str:
    """Where patched dispatch, as configured now, sends the fit a record measured."""
    from mlxlearn.patching import _estimators as patched

    n, d = record["n"], record["d"]
    if record["estimator"] == "knn":
        model = patched.NearestNeighbors(
            n_neighbors=record.get("k", _K), algorithm=record["algorithm"]
        )
        below = model._below_crossover(n, d)[0] or model._patched_handoff(n, d) is not None
        if not below and record.get("query_k"):
            # A query asking for more neighbors than the fit, on an index it fitted.
            below = model._query_neighbors_handoff(record["query_k"]) is not None
        if not below and "probe_rows" in record:
            # The data rule, on what the fit-time probe found when this was measured.
            found = {"rows": record["probe_rows"], "uncertified": record["probe_uncertified"],
                     "integer": record["integer_path"]}
            below = model._probe_handoff(n, d, found) is not None
    elif record["estimator"] == "svc":
        model = patched.SVC(kernel=record["kernel"], C=record["C"],
                            degree=record.get("degree", 3), coef0=record.get("coef0", 0.0),
                            cache_size=record.get("cache_size") or 200.0)
        below = (
            model._below_crossover(n, d, n_classes=record["n_classes"])[0]
            # The C rule reads the mean box, C x the mean class and sample weight: C
            # itself here, as the grids pass no sample weights and balanced class
            # weights average to one.
            or model._kernel_handoff(record["C"]) is not None
        )
        if not below and record["operation"] == "predict":
            work = record["q"] * record["n_sv"] * d
            from mlxlearn._common.config import get_tuning

            return "host" if work <= get_tuning().svc_host_inference_work else "MLX"
    else:
        K = record["n_classes"]
        model = patched.LogisticRegression(
            C=record.get("C", 1.0), tol=record.get("tol") or 1e-4,
            max_iter=record.get("max_iter") or 200,
        )
        below = (
            model._below_crossover(n, d, 1 if K == 2 else K)[0]
            or model._param_handoff() is not None
        )
    return "scikit-learn" if below else "MLX"


def _shape(n: int, d: int | None = None) -> str:
    rows = f"{n:,}".replace(",", " ")
    return rows if d is None else f"{rows} × {d}"


# Each layout maps a record to (row sort key, row label) and (column sort key,
# column label); a cell holds one record per scikit-learn version.
def _knn_row(r):
    return (r["d"], r["n"]), _shape(r["n"], r["d"])


def _knn_column(r):
    if r["operation"] == "fit":
        return (1, 0), "fit"
    return (0, r["q"]), f"{r['q']} {'query' if r['q'] == 1 else 'queries'}"


def _svc_row(r):
    return (r["d"], r["n"]), _shape(r["n"], r["d"])


_KERNEL_ORDER = {"rbf": 0, "poly": 1, "sigmoid": 2, "linear": 3}


def _svc_column(r):
    if r["n_classes"] != 2:
        return (r["n_classes"], 0, 0), f"K = {r['n_classes']}"
    kernel = _KERNEL_ORDER.get(r["kernel"], len(_KERNEL_ORDER))
    return (0, kernel, r["C"]), f"{r['kernel']}, C = {r['C']:g}"


def _logreg_row(r):
    return (r["n"],), _shape(r["n"])


def _logreg_column(r):
    if r["n_classes"] == 2:
        return (0, r["d"]), f"{r['d']} feature{'' if r['d'] == 1 else 's'}"
    return (r["n_classes"], r["d"]), f"{r['n_classes']} classes × {r['d']}"


_KNN_DATA_ORDER = {"gauss": 0, "blobs": 1, "uniform": 2, "dup2": 3, "scaled100": 4,
                   "blobs0.5": 5, "blobs0.4": 6, "unscaled": 7, "dups": 8, "tight-blobs": 9,
                   "binary": 10, "counts": 11}


def _knn_data_row(r):
    order = _KNN_DATA_ORDER.get(r["data"], len(_KNN_DATA_ORDER))
    return (r["d"], r["n"], order), f"{_shape(r['n'], r['d'])}, {r['data']}"


def _knn_k_row(r):
    return (r["d"], r["n"], r.get("k", _K)), f"{_shape(r['n'], r['d'])}, k = {r.get('k', _K)}"


def _svc_variant(r) -> tuple[tuple, str]:
    if r["kernel"] == "rbf":
        factor = r.get("gamma") or 1.0
        return (0, factor, 0, 0.0), f"rbf, gamma = {factor:g} × scale"
    degree, coef0 = r.get("degree", 3), r.get("coef0", 0.0)
    label = f"{r['kernel']}, degree {degree}, coef0 = {coef0:g}"
    if r.get("gamma"):
        label = f"{label}, gamma = {r['gamma']:g} × scale"
    return (1, r.get("gamma") or 1.0, degree, coef0), label


def _svc_params_row(r):
    variant_key, variant = _svc_variant(r)
    return (r["d"], r["n"], variant_key), f"{_shape(r['n'], r['d'])}, {variant}"


def _svc_c_column(r):
    return (r["C"],), f"C = {r['C']:g}"


def _svc_multi_row(r):
    pair = _shape(2 * r["n"] // r["n_classes"])
    return (r["d"], r["n"], r["n_classes"]), (
        f"{_shape(r['n'], r['d'])}, K = {r['n_classes']} (pairs of {pair})"
    )


def _svc_predict_row(r):
    label = _shape(r["n"], r["d"])
    if r["n_classes"] != 2:
        label = f"{label}, K = {r['n_classes']}"
    return (r["d"], r["n"], r["n_classes"]), f"{label} ({_shape(r['n_sv'])} SVs)"


def _q_column(r):
    return (r["q"],), f"{r['q']} {'row' if r['q'] == 1 else 'rows'}"


_SVC_LABELS = {"binary": "", "noisy": "noisy", "random": "random", "xor": "XOR"}


def _svc_hard_row(r):
    variant_key, variant = _svc_variant(r)
    labels = _SVC_LABELS.get(r["data"], r["data"])
    classes = "" if r["n_classes"] == 2 else f", K = {r['n_classes']}"
    parts = [_shape(r["n"], r["d"]) + classes, labels, variant]
    return (r["d"], r["n"], r["n_classes"], labels, variant_key), ", ".join(p for p in parts if p)


def _svc_structured_row(r):
    variant_key, variant = _svc_variant(r)
    classes = "" if r["n_classes"] == 2 else f", K = {r['n_classes']}"
    parts = [_shape(r["n"], r["d"]) + classes, r["data"], variant]
    if r.get("tol"):
        parts.append(f"tol = {r['tol']:g}")
    if r.get("class_weight"):
        parts.append(f"class_weight = {r['class_weight']}")
    key = (r["d"], r["n"], r["n_classes"], r["data"], variant_key, r.get("tol") or 0.0,
           r.get("class_weight") or "")
    return key, ", ".join(parts)


_QUERY_PATHS = {"": (0, "as dispatched"), "first": (1, "as dispatched, first query"),
                "mlx": (2, "on MLX")}


def _knn_query_k_row(r):
    order, path = _QUERY_PATHS[r.get("path", "")]
    if r["query_k"] <= 50:
        path = "on MLX (as dispatched)"
    key = (r["d"], r["n"], r.get("dtype", "float64"), r["query_k"], order)
    label = f"{_shape(r['n'], r['d'])}, {r.get('dtype', 'float64')}, k = {r['query_k']}, {path}"
    return key, label


def _svc_cache_row(r):
    data = "Gaussian" if r["data"] == "binary" else r["data"]
    return (r["d"], r["n"], data), f"{_shape(r['n'], r['d'])}, {data}"


def _cache_column(r):
    cache = r.get("cache_size") or 200.0
    return (cache,), f"cache_size = {cache:g}"


def _logreg_data_row(r):
    return (r["data"], r["n"]), f"{r['data']}, {_shape(r['n'])}"


def _logreg_params_row(r):
    K = r["n_classes"]
    return (r["data"], K, r["n"], r["d"]), f"{r['data']}, K = {K}, {_shape(r['n'], r['d'])}"


def _logreg_params_column(r):
    if r.get("tol"):
        return (1, r["tol"]), f"tol = {r['tol']:g}"
    if r.get("max_iter"):
        return (2, r["max_iter"]), f"max_iter = {r['max_iter']}"
    if r.get("C", 1.0) != 1.0:
        return (3, r["C"]), f"C = {r['C']:g}"
    return (0, 0), "defaults"


_LAYOUTS = {
    "knn-q": (_knn_row, _knn_column, "rows × features"),
    "knn-worst": (_knn_row, _knn_column, "rows × features"),
    "knn-data": (_knn_data_row, _knn_column, "rows × features, data"),
    "knn-data-worst": (_knn_data_row, _knn_column, "rows × features, data"),
    "knn-k": (_knn_k_row, _knn_column, "rows × features, k"),
    "knn-k-worst": (_knn_k_row, _knn_column, "rows × features, k"),
    "knn-query-k": (_knn_query_k_row, _knn_column, "rows × features, dtype, query k, path"),
    "svc-cache": (_svc_cache_row, _cache_column, "rows × features, data"),
    "svc": (_svc_row, _svc_column, "rows × features"),
    "svc-params": (_svc_params_row, _svc_c_column, "rows × features, kernel"),
    "svc-hard": (_svc_hard_row, _svc_c_column, "rows × features, labels, kernel"),
    "svc-structured": (_svc_structured_row, _svc_c_column, "rows × features, data, kernel"),
    "logreg-params": (_logreg_params_row, _logreg_params_column, "labels, rows × features"),
    "svc-multi-c": (_svc_multi_row, _svc_c_column, "rows × features, classes"),
    "svc-predict": (_svc_predict_row, _q_column, "fit (support vectors)"),
    "logreg": (_logreg_row, _logreg_column, "rows"),
    "logreg-data": (_logreg_data_row, _logreg_column, "labels, rows"),
}


def render(name: str, records: list[dict]) -> str:
    """The Markdown table for grid ``name`` from ``records`` (other grids' are ignored).

    One row per shape, one column per configuration. A cell is stock time over MLX
    time for each scikit-learn version in the records, newest first — ``1.9.0 /
    1.7.2`` — **bold** where that is a loss by the gate's test and ``~`` where it is
    within noise. The dense neighbor grids (layouts ending ``-worst``) have a dozen
    batch sizes, so each of their cells is the worse version's ratio instead: its
    loss, if either version lost, else the lower of the two. The
    ``dispatch`` column is where patched dispatch as configured now sends the fit;
    where it differs across a row's columns it names the columns sent to MLX.
    """
    grid = GRIDS[name]
    records = [r for r in records if r.get("grid") == name]
    if not records:
        return f"(no results for grid {name!r})\n"
    versions = sorted(
        {r["sklearn"] for r in records},
        key=lambda v: tuple(int(p) if p.isdigit() else -1 for p in v.split(".")),
        reverse=True,
    )
    if grid.layout == "knn-probe":
        return _render_draws(name, records, versions)
    row_of, column_of, first = _LAYOUTS[grid.layout]
    table: dict = {}
    rows: dict = {}
    columns: dict = {}
    for record in records:
        row_key, row_label = row_of(record)
        column_key, column_label = column_of(record)
        rows[row_key] = row_label
        columns[column_key] = column_label
        table.setdefault(row_key, {}).setdefault(column_key, {})[record["sklearn"]] = record
    order = sorted(columns)
    worst = grid.layout.endswith("-worst")
    # The data grids also show what the fit-time probe found, which dispatch reads.
    probe = grid.layout.startswith("knn-data")
    lines = [
        f"<!-- python -m benchmarks.calibrate table {name}: scikit-learn "
        f"{' / '.join(versions)}{', the worse of them' if worst and len(versions) > 1 else ''}"
        " -->",
        f"| {first} | " + " | ".join(columns[c] for c in order)
        + (" | probe" if probe else "") + " | dispatch |",
        "|---" * (len(order) + 2 + int(probe)) + "|",
    ]
    for row_key in sorted(rows):
        row = table[row_key]
        cells = []
        routes = {}
        for column in order:
            by_version = row.get(column, {})
            if not by_version:
                cells.append("—")
                continue
            routes[column] = dispatch(next(iter(by_version.values())))
            if worst:
                # A loss on either version shows, even beside a lower ratio within noise.
                shown = min(by_version.values(), key=lambda r: (not is_loss(r), r["speedup"]))
                cells.append(_cell(shown))
            else:
                cells.append(" / ".join(_cell(by_version.get(v)) for v in versions))
        if probe:
            cells.append(_probe_cell([r for c in row.values() for r in c.values()]))
        if len(set(routes.values())) == 1:
            route = next(iter(routes.values()))
        elif set(routes.values()) == {"host", "MLX"}:
            last = max(c for c in order if routes.get(c) == "host")
            route = f"host to {columns[last]}, MLX above"
        else:
            route = "MLX at " + ", ".join(columns[c] for c in order if routes.get(c) == "MLX")
        lines.append(f"| {rows[row_key]} | " + " | ".join(cells) + f" | {route} |")
    return "\n".join(lines) + "\n"


def _render_draws(name: str, records: list[dict], versions: list[str]) -> str:
    """The `knn-probe` table: one row per shape and law, summarizing its draws.

    For the draws a probe passed — every probed row certified — under the rule as
    configured now (``dispatch``) and with a probe of the other size the records hold,
    how many there were and the worst batch among them on each version: the lowest
    ratio, **bold** where a batch lost by the gate's test. Every batch of every draw is
    a measurement of the MLX path; the probe decides only whether patched dispatch runs
    it.
    """
    groups: dict = {}
    for r in records:
        groups.setdefault((r["d"], r["n"], _KNN_DATA_ORDER.get(r["data"], 99), r["data"]),
                          []).append(r)
    also = {r["probe_also_rows"] for r in records if "probe_also_rows" in r}
    rows = {r["probe_rows"] for r in records}
    configured = " / ".join(str(n) for n in sorted(rows))
    other = " / ".join(str(n) for n in sorted(also))

    def worst(draw_records) -> str:
        cells = []
        for version in versions:
            mine = [r for r in draw_records if r["sklearn"] == version]
            if not mine:
                cells.append("—")
                continue
            cells.append(_cell(min(mine, key=lambda r: (not is_loss(r), r["speedup"]))))
        return " / ".join(cells)

    lines = [
        f"<!-- python -m benchmarks.calibrate table {name}: scikit-learn "
        f"{' / '.join(versions)} -->",
        f"| rows × features, data | draws | passed by {other} rows | worst batch of those | "
        f"passed by {configured} rows | worst batch of those | dispatch |",
        "|---" * 7 + "|",
    ]
    for (d, n, _, data), group in sorted(groups.items()):
        draws = sorted({r["data_seed"] for r in group})
        passed_also = [s for s in draws if all(
            r.get("probe_also_uncertified", 1) == 0 for r in group if r["data_seed"] == s)]
        routed = [s for s in draws if all(
            dispatch(r) == "MLX" for r in group if r["data_seed"] == s)]
        lines.append(
            f"| {_shape(n, d)}, {data} | {len(draws)} | {len(passed_also)} | "
            f"{worst([r for r in group if r['data_seed'] in passed_also]) if passed_also else '—'}"
            f" | {len(routed)} | "
            f"{worst([r for r in group if r['data_seed'] in routed]) if routed else '—'}"
            f" | MLX for {len(routed)} of {len(draws)} |"
        )
    return "\n".join(lines) + "\n"


def _probe_cell(records: list[dict]) -> str:
    """What the fit-time probe found, across versions: ``integer``, or uncertified rows."""
    if any(r.get("integer_path") for r in records):
        return "integer"
    found = sorted({(r["probe_uncertified"], r["probe_rows"]) for r in records})
    return " / ".join(f"{u}/{n}" for u, n in found)


# ----------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the grids")
    run_parser = sub.add_parser("run", help="measure a grid, appending to --output")
    run_parser.add_argument("grid", choices=sorted(GRIDS))
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--iterations", type=int, default=None)
    run_parser.add_argument("--limit", type=int, default=None,
                            help="stop after this many new measurements (resume later)")
    table_parser = sub.add_parser("table", help="render a grid's Markdown table")
    table_parser.add_argument("grid", choices=sorted(GRIDS))
    table_parser.add_argument("results", type=Path, nargs="+")
    update_parser = sub.add_parser(
        "update", help="re-render, in place, every calibration table a Markdown file holds"
    )
    update_parser.add_argument("document", type=Path)
    update_parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args(argv)

    if args.command == "list":
        for name, grid in GRIDS.items():
            count = sum(len(p.keys()) for p in grid.points)
            print(f"{name:18} {count:5} measurements  {grid.description}")
        return 0
    if args.command == "run":
        print(json.dumps(_versions()), flush=True)
        run(args.grid, args.output, iterations=args.iterations, limit=args.limit,
            echo=lambda s: print(s, flush=True))
        return 0
    if args.command == "update":
        text = args.document.read_text(encoding="utf-8")
        updated, count = update_document(text, load(args.results))
        args.document.write_text(updated, encoding="utf-8")
        print(f"re-rendered {count} table(s) in {args.document}")
        return 0
    sys.stdout.write(render(args.grid, load(args.results)))
    return 0


#: A rendered table in a document: the comment line ``render`` starts it with, then
#: its rows.
_RENDERED = re.compile(
    r"^<!-- python -m benchmarks\.calibrate table ([\w-]+):[^\n]*-->\n(?:\|[^\n]*\n)+",
    re.MULTILINE,
)


def update_document(text: str, records: list[dict]) -> tuple[str, int]:
    """``text`` with every calibration table it holds re-rendered from ``records``.

    A table whose grid ``records`` has no results for is left as it is, so updating
    from one grid's results does not erase the others.
    """
    grids = {record.get("grid") for record in records}
    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        if match.group(1) not in grids:
            return match.group(0)
        count += 1
        return render(match.group(1), records)

    return _RENDERED.sub(replace, text), count


if __name__ == "__main__":
    raise SystemExit(main())
