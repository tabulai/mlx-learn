"""Configuration: a small public surface over a larger private one.

Two layers, on purpose.

:class:`Config` is public and deliberately tiny — six options that change what a
user observes: which device to use, what to do when mlxlearn cannot serve a
request, what array type comes back, whether results are reproducible, the seed,
and whether diagnostics are recorded.

:class:`Tuning` is private. Crossover thresholds, block sizes, cache fractions and
solver limits are implementation details that will move as the benchmarks improve;
promoting them to public options would freeze today's implementation into the API.
They are overridable through ``MLXLEARN_*`` environment variables for CI and
debugging, and that is the whole supported surface for them.

**Where a setting reaches.** Three scopes, and each one exists because a user
reasonably expects it:

* :func:`set_config` changes the *process-wide* default. Every thread sees it —
  including the worker threads scikit-learn's ``n_jobs`` machinery runs fits on
  under joblib's threading backend. A per-thread ``set_config`` made
  ``fallback_policy="raise"`` pass vacuously the moment a ``GridSearchCV`` went
  parallel, which is the one place an assertion of full acceleration is worth
  making.
* :func:`config_context` overrides the fields it names for the calling *context*
  only (a :class:`contextvars.ContextVar`), so a library that sets a policy inside
  it cannot leak it into a thread it did not start. ``asyncio`` tasks and
  ``asyncio.to_thread`` inherit it, as they inherit any context variable; plain
  threads and thread pools, joblib's included, do not.
* A separate *process* — joblib's default ``loky`` backend, ``multiprocessing`` —
  imports mlxlearn afresh and starts from the environment: ``MLXLEARN_DEVICE``,
  ``MLXLEARN_FALLBACK_POLICY``, ``MLXLEARN_DIAGNOSTICS`` and so on, one per
  :class:`Config` field. Nothing a parent sets at runtime can reach a worker that
  is already running, so to govern process workers set those variables before
  the workers start. (To *assert* that nothing falls back under ``n_jobs``, a
  ``warnings.filterwarnings("error", category=MLXLearnFallbackWarning)`` also
  works: scikit-learn forwards warning filters to its workers.)
"""

from __future__ import annotations

import numbers
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from typing import Any, Literal, get_args

import numpy as np

__all__ = [
    "Config",
    "Tuning",
    "get_config",
    "set_config",
    "config_context",
    "get_tuning",
    "tuning_context",
    "reset_config",
]

Device = Literal["auto", "gpu", "cpu"]
FallbackPolicy = Literal["warn", "raise", "silent"]
OutputType = Literal["numpy", "input", "mlx"]
ForceBackend = Literal["", "mlx", "cpu", "sklearn"]

_SEED_LIMIT = 2**32


# --------------------------------------------------------------------------------------
# public configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Config:
    """User-visible configuration. Immutable; replace it, do not mutate it.

    Attributes
    ----------
    device : {"auto", "gpu", "cpu"}
        Which MLX device runs the work that dispatch has sent to MLX. ``"auto"``
        prefers the GPU and falls back to the CPU when no Metal device is usable.
        It does not override size dispatch: a problem below the measured
        crossover still goes to mlxlearn's CPU path (Layer 1) or to scikit-learn
        (patched), whatever the device. ``device="cpu"`` is the same float32
        algorithm on the CPU, not scikit-learn's float64 one. The names are
        MLX's, not Metal's: there is deliberately no ``"mps"``.
    fallback_policy : {"warn", "raise", "silent"}
        What patched mode does on a *capability* mismatch. ``"warn"`` falls back
        and warns at most once per estimator class and reason; ``"raise"`` turns
        any fallback into an error, which is how you assert that a workload is
        fully accelerated; ``"silent"`` falls back quietly, for production and
        agent harnesses where a warning stream is just noise. Unexpected backend
        failures raise under every policy.
    output_type : {"numpy", "input", "mlx"}
        What array type estimators return. 0.1.0 supports ``"numpy"`` only;
        ``"input"`` and ``"mlx"`` are reserved for the 0.2.x experimental MLX
        boundary and are rejected today rather than silently ignored.
    deterministic : bool
        When true, mlxlearn takes reproducible paths — fixed reduction orders,
        seeded initialization, stable tie-breaking — even where a faster
        nondeterministic path exists. Every 0.1.x solver is deterministic and has
        no faster nondeterministic variant, so today this option has no
        observable effect: results are reproducible either way.
    random_state : int or None
        Seed for mlxlearn's own RNG, an int in ``[0, 2**32)``. Estimator-level
        ``random_state`` parameters take precedence over it. No 0.1.x estimator
        draws random numbers, so today it has no observable effect; it does not
        seed scikit-learn either, so a patched estimator that falls back uses its
        own ``random_state`` parameter.
    diagnostics : bool
        Whether backend events are recorded. Recording is cheap and it is what
        makes a fallback debuggable after the fact, so it defaults on.
    """

    device: Device = "auto"
    fallback_policy: FallbackPolicy = "warn"
    output_type: OutputType = "numpy"
    deterministic: bool = True
    random_state: int | None = None
    diagnostics: bool = True

    def __post_init__(self) -> None:
        _validate_choice("device", self.device, get_args(Device))
        _validate_choice("fallback_policy", self.fallback_policy, get_args(FallbackPolicy))
        _validate_choice("output_type", self.output_type, get_args(OutputType))
        if self.output_type != "numpy":
            raise ValueError(
                f"output_type={self.output_type!r} is reserved for the 0.2.x experimental "
                "MLX array boundary and is not implemented in this release. "
                "mlxlearn 0.1.0 takes NumPy in and returns NumPy out."
            )
        # Normalized, not merely checked: a NumPy bool or integer is the value the
        # user meant, and storing the Python type keeps equality and pickles plain.
        object.__setattr__(self, "deterministic", _as_bool("deterministic", self.deterministic))
        object.__setattr__(self, "diagnostics", _as_bool("diagnostics", self.diagnostics))
        object.__setattr__(self, "random_state", _as_seed(self.random_state))


def _validate_choice(name: str, value: object, allowed: tuple[Any, ...]) -> None:
    if value not in allowed:
        options = ", ".join(repr(a) for a in allowed)
        raise ValueError(f"{name} must be one of {options}; got {value!r}")


def _describe(value: object) -> str:
    kind = type(value)
    module = "" if kind.__module__ == "builtins" else f"{kind.__module__}."
    return f"{value!r} ({module}{kind.__qualname__})"


def _as_bool(name: str, value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{name} must be a bool, got {_describe(value)}")


def _as_seed(value: object) -> int | None:
    if value is None:
        return None
    # bool is an Integral; a seed of True is a bug, not a seed of 1.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"random_state must be an int or None, got {_describe(value)}")
    seed = int(value)
    if not 0 <= seed < _SEED_LIMIT:
        raise ValueError(f"random_state must be in [0, 2**32), got {seed}")
    return seed


# --------------------------------------------------------------------------------------
# private tuning
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tuning:
    """Private tuning knobs. Not public API; may change in any release.

    Every default here is a *measurement*, or a placeholder waiting for one. The
    crossover thresholds are the sizes below which stock scikit-learn wins, taken
    from ``docs/benchmarks.md``; dispatch reads them so that "we are never slower"
    is enforced by handing small problems away rather than by hoping.
    """

    # ---- dispatch crossovers: below these, stock scikit-learn is used ----------------
    #
    # Measured, not guessed: the MLX path patched dispatch would run against stock
    # scikit-learn in interleaved, order-alternated pairs (benchmarks/_harness.compare:
    # a garbage collection before every timed call, mx.synchronize inside it, so every
    # MLX call pays to wake an idled GPU), median of 5-9, on an Apple M4 Max (16 cores,
    # 128 GiB, macOS 26.6.2) with MLX 0.32.2, against scikit-learn 1.9.0 (Python
    # 3.13.3, NumPy 2.5.2) and 1.7.2 (Python 3.10.19, NumPy 2.2.6). A threshold routes
    # to MLX only where every configuration measured won, or tied within noise, on
    # both versions. Each value below names the calibration table in
    # docs/benchmarks.md that justifies it, and `python -m benchmarks.calibrate`
    # regenerates that table (the grid of the same name).
    #
    # KNN, k=10, Gaussian data. Both floors bind in both layers (see
    # MLXNeighborsMixin._below_crossover), because a fit cannot see how it will be
    # queried and each floor guards a different batch size (table `knn-rows`: 256 to
    # 1024 features, 2048-131072 rows, every batch of 1-4096 queries). Below 4096
    # rows a 4096-query batch lost at every width (0.44x-0.79x at 2048 rows). Below
    # 6e6 elements batches of 1-32 queries lost at 4096-8192 rows (0.70x-0.95x), and
    # at 4 194 304 elements, the floor this one replaced, single rows only tied
    # (0.90x-0.99x). Above both, at 256-1024 features, every batch won on both
    # versions or tied (1.02x-8.5x).
    knn_min_train_samples: int = 4_096
    knn_min_work: int = 6_000_000  # n_train * n_features
    #
    # Patched mode only (MLXNeighborsMixin._patched_handoff), three more rules. A fit
    # with fewer than knn_min_features features goes to scikit-learn at any size: to
    # its KD-tree where algorithm='auto' would build one (at most 15 features; reason
    # "tree-wins"; tables `knn-tree`, `knn-tree-lowrank`), else to its brute force
    # (table `knn-narrow-brute`: at 2 features some batch lost at every size measured,
    # up to 2 097 152 rows, and at 8 features up to 131 072 rows, 0.62x-0.91x). This
    # floor stood at 64 while MLX's own selection picked the first pass's candidates:
    # at a million rows 16-48 features lost batches of 16 queries (0.73x-0.84x) and
    # 1 024 (0.62x-0.79x). With the fused selection kernel (_kernels/_topk.py), every
    # configuration measured at 1 000 000-1 600 000 rows and 16-48 features won or
    # tied on both versions, each MLX call after an idle GPU: every batch of 1-4096
    # queries on Gaussian data (table `knn`, 1.55x at worst); Gaussian, 100 clusters
    # of std 1 and uniform data (`knn-narrow`, 1.55x; `knn-data`, 1.53x); clusters of
    # std 0.4 and 0.5, which the fit-time probe below passes (`knn-structured`,
    # 1.54x), and rows repeated about twice and one column on 100 times the scale of
    # the rest, which it hands off (0.98x~ at worst); and k = 1-50 (`knn-k`, 1.26x~).
    # What the move costs is the index build: a patched fit there, probe included,
    # took 47-132 ms against stock's 1.9-6.0 ms (0.04x-0.06x, after an idle gap,
    # medians of 9, both versions; see the `knn/fit` accepted regression), and a
    # float64 index holds three times the float32 data on the device. 0 disables the
    # rule.
    knn_min_features: int = 16
    #
    # And a fit with fewer than knn_wide_features features needs
    # knn_narrow_min_samples rows (table `knn`: 16-256 features, 12 000-1.6 million
    # rows, every batch of 1-4096 queries). An idled GPU runs a call of a few
    # milliseconds' work up to two or three times slower, so at 16-128 features
    # batches of 1-128 queries lost on one version or both at 27 of the 50 sizes
    # measured below a million rows (down to 0.55x), every one of them at 180 000
    # rows or fewer; the size floors alone send 180 000 x 32 to MLX, where 16-query
    # batches ran 0.67x-0.71x. From 262 144 rows every batch of that table won or
    # tied (1.00x~ at worst), but its batches are powers of two, and a query of fewer
    # than 2^22 (query row, training row) pairs runs MLX's own selection rather than
    # the fused kernel: the largest such batch, which the table's 8 and 16 queries
    # straddle at 262 144 rows, lost there at 16 features (15 queries, 0.90x-0.91x)
    # and only tied at 400 000 rows (10 queries, 0.99x~-1.02x~); at 600 000 it won or
    # tied (6 queries, 1.27x-3.1x; table `knn-boundary`) — all of it on Gaussian data at
    # k = 10, the only law and k measured below a million rows. From 1 000 000 rows
    # at 16-128 features every configuration the other rules route there won or tied
    # (tables `knn`, `knn-narrow`, `knn-structured`, `knn-k`). At 256 features every batch won or tied from
    # 33 000 rows, the first size of the table the work floor lets through (1.07x at
    # worst); at 12 000 and 20 000 rows single rows and small batches lost
    # (0.85x-0.93x).
    knn_wide_features: int = 256
    knn_narrow_min_samples: int = 1_000_000
    #
    # And a fit with knn_wide_features features or more needs knn_wide_min_samples
    # rows (table `knn-batches`: 4 096-12 000 rows at 2 048-4 096 features, every
    # batch of 8-64 queries). Right after a stock query, whose OpenMP threads are
    # still spinning, an MLX call of some batch sizes — about 256 KiB of float32
    # queries, 32 rows at 2 048 features and 16 at 4 096 — ran up to twice as slow as
    # its neighbours (with the spinning stopped, KMP_BLOCKTIME=0, it did not), and on
    # a short training set a stock search costs about as much: at 4 096-7 500 rows
    # batches of 16-32 queries lost on one version or both (0.77x-0.90x) or tied
    # (1.04x at 7 500 x 4 096, k = 50), while from 8 192 rows every batch won or tied
    # (1.20x at worst). The size floors alone send such fits to MLX from about 1 500
    # features. It gives up what won at 4 096-7 500 rows and 2 048-4 096 features —
    # most batches there, 1.02x-3.0x (tables `knn-batches`, `knn-k`, and
    # `knn-structured` at 4 096 x 2 048) — and every such fit at 6 000-8 191 rows and
    # 1 000-2 047 features, which no table measured.
    knn_wide_min_samples: int = 8_192
    #
    # Patched mode also reads k: a fit's, the constructor's n_neighbors
    # (MLXNeighborsMixin._neighbors_handoff), and a query's, when it asks for another
    # (MLXNeighborsMixin._query_neighbors_handoff; table `knn-k`: k = 1-200 at the
    # narrow corner and on short, wide training sets up to 4 096 features, every batch
    # of 1-4096 queries). Everything above was measured at k = 10, and MLX pays per
    # query for every candidate it keeps — through each tile's selection, and in
    # refinement, a gather of k rows of every feature — where stock brute force
    # pays almost nothing more. At k = 100 a 1 024-query batch lost at 1 000 000 x
    # 64 on both versions (0.80x), and at 4 096-6 000 rows and 2 048-4 096 features
    # batches of 32-4 096 queries lost (0.61x-0.89x); k = 200 lost there at nearly
    # every batch from 8 (down to 0.33x), at 8 192 x 1 024 (0.74x-0.82x, 128
    # queries) and at 1 000 000-1 600 000 x 64 (0.67x-0.83x, 512-1 024 queries); at
    # 16-48 features, measured with the fused selection kernel, k = 100 and 200 lost
    # batches of 8-4 096 queries (0.54x-0.94x) where k = 1-50 won every batch.
    # k = 50 won or tied everywhere, but with 82-125 rows per neighbor at 2 048-4 096
    # features — 4 096-5 000 rows at k = 40-50 — only tied at batches of 16-512
    # queries (0.84x-0.99x); from 150 rows per neighbor every batch won or tied, and
    # the row floors above leave more than 160 at every k this allows. So a fit needs
    # n_neighbors of at most knn_max_neighbors. A query on an MLX-fitted index that
    # asks for more is answered by scikit-learn's brute force over a float64 copy of
    # the training rows (table `knn-query-k`, 21 pairs a point: 8 192 x 1 024 and
    # 2 048, 65 536 x 256 and 1 000 000 x 64, float64 and float32, k = 50-200, 1-1 024
    # queries). On float64 data it kept up with a stock model's, 0.88x-1.09x with one
    # significant loss in 160 (0.88x); on float32 data at a million rows it lost by a
    # few percent (0.93x-0.95x at 16-1 024 queries), stock searching float32 rows and
    # the copy being float64. On MLX those queries lost at 8 192 x 2 048 with k = 200
    # (0.73x-0.86x) and at 1 000 000 x 64 with k = 100-200 and 1 024 queries
    # (0.77x-0.86x). The first such query builds the copy, 8 * n_samples * n_features
    # bytes: 2-29 ms at the three smaller shapes and 25-85 ms at a million x 64. What
    # the k rule gives up is k = 100-200 at 96-128 features and a million rows
    # (1.2x-3.0x), k = 100 at 100 000 x 256 (1.1x-4.3x) and at 8 192 x 1 024
    # (1.15x-1.9x) — and at query time MLX's wins at k = 51-200 (1.08x-4.5x).
    knn_max_neighbors: int = 50
    #
    # All of the above is calibrated on Gaussian data. Patched mode also reads the
    # data (MLXNeighborsMixin._patched_data_handoff): the fit searches
    # knn_probe_rows of its own rows with the first float32 pass alone and hands the
    # fit to scikit-learn when more than knn_probe_max_uncertified of them — at 0,
    # any of them — could not be certified. A query row like those is searched
    # again, wider and then exactly, and the whole batch waits for it: the cost is
    # per *batch*, so a few such rows in a hundred, which put one in most batches of
    # 16-64 queries, are enough to lose. Table `knn-structured`, every batch of
    # 1-4096 queries at shapes on and near the size floors: rows repeated about
    # twice, one column on 100 times the scale of the rest, and 100 clusters of std
    # 0.4 or 0.5 lost some batch at 32 of the 52 shapes and laws measured, on one
    # version or both, down to 0.36x — at 256 features batches of 4-128 queries
    # (0.50x-0.91x) — and the probe found 3-473 of 512 rows uncertified in every
    # one of the 32, 3 of 512 for clusters of std 0.5 at 30 000 x 256 (0.52x at 16
    # queries). Where it found none (clusters at a million rows and 16-64 features)
    # they won or tied; so did Gaussian data, with none at every shape of that table
    # and of `knn-data`, wherever the size rules route it. Table `knn-data`: one column in
    # raw units, rows repeated about 20
    # times and 100 clusters of std 0.1 left 478-512 of 512 at every wide shape and
    # lost everywhere, down to 0.22x (0.16x at a million rows and 32 features);
    # clusters of std 1 left none at 256 features and 14 at 8 192 x 1 024, where 64
    # queries only tied. The probe first searched 64 rows and allowed 8 uncertified:
    # with one row in a hundred uncertified, 64 rows see none about half the time,
    # 256 one time in thirteen and 512 one time in 170. Table `knn-probe`, 64 draws of
    # those std-0.5 clusters at each of 24 000 and 30 000 x 256, where about one row
    # in a hundred is: a probe of 256 rows passed 9 of the 128, 8 of which lost a
    # batch of 16-64 queries on one version or both (down to 0.56x); 512 rows passed
    # none. Of 72 draws of Gaussian, low-rank and std-1 cluster data beside them, 512
    # rows gave up 5 that 256 passed, which won or tied every batch (1.06x-3.7x), and
    # every draw they passed won or tied (1.01x at worst). The probe of 512 rows costs
    # a fit 5-17 ms after an idle gap at 256-4 096 features and 16-28 ms at a million
    # rows and more (k = 5); see the `knn/fit` accepted regression.
    # Integer-valued data takes an exact path with nothing to certify; narrower than
    # knn_wide_features it goes to scikit-learn too: at a million rows and 16 or 32
    # features it lost batches of 16 queries and, of counts, 256 and 1024 queries
    # (0.53x-0.85x), while from 256 features it kept up (33-255 features were not
    # measured on MLX).
    knn_probe_max_uncertified: float = 0.0
    knn_probe_rows: int = 512
    #
    # Logistic regression, max_iter=200 (see LogisticRegression._below_crossover).
    # Every fit needs logreg_min_samples rows. Multinomial fits won at every width
    # measured from 60000 rows, on overlapping blobs (table `logreg-multi`: 3, 5 and
    # 10 classes, 1 to 512 features, 1.50x-8.6x on both versions) and on
    # well-separated ones that scikit-learn fits in one iteration (table
    # `logreg-separable`: 1.00x-4.7x); at 20000 rows they were mixed (0.78x-2.3x),
    # and at 5000 lost or tied at every width but 512 (0.37x-0.91x). But a fit whose
    # labels carry no signal converges in one iteration, and is then mostly MLX's
    # fixed cost — at one feature, more of it, since MLX's GEMM with an output of one
    # column is slow: the MLX path cost about 10 ms at 60000 x 1 whatever the class
    # count, against 7 at 60000 x 2 (table `logreg-labels`). At 60000 x 1, three to
    # nine random classes lost or tied (0.70x-1.17x; six read 0.93x-0.94x on 1.9.0 in
    # two runs outside the table), and 120000-200000 x 1 with three stayed within
    # noise (1.07x-1.36x), while from two
    # features every multinomial fit won or tied (1.06x at worst). So a multinomial fit
    # needs logreg_multinomial_min_features features. It replaces a work floor of
    # 360 000 rows x features x classes, which 60000 x 1 with six classes met and lost
    # at, and it gives up the one-feature fits that won: five and ten random classes at
    # 100000-200000 rows (1.37x-2.2x), ten at 60000 (1.15x-1.74x), and 200000 x 1 with
    # three overlapping ones (`logreg-multi`, 2.67x). A binary fit has floors of its
    # own (below), and this one does not bind it.
    logreg_min_samples: int = 60_000
    logreg_multinomial_min_features: int = 2
    #
    # A binary fit also needs logreg_binary_min_samples rows and at most
    # logreg_binary_max_features features. On the benchmark's binary target MLX won
    # from 100000 rows at every width up to 1024 (table `logreg-binary`, 1.30x-6.2x),
    # but a fit that converges in an iteration or two is mostly MLX's fixed cost —
    # the float32 copy of X and the device round trips — and labels carrying no
    # signal do exactly that (table `logreg-labels`): at 100000 rows random and noisy
    # labels lost on 1.7.2 at 4-256 features (0.61x-0.88x), and random ones on 1.9.0
    # at 4 and 256 features (0.81x, 0.90x); from 200000 rows they won
    # or tied up to 64 features (0.97x at worst), and at 256 features random labels
    # stood at 0.96x-1.17x, too close to call.
    logreg_binary_min_samples: int = 200_000
    logreg_binary_max_features: int = 64
    #
    # Patched mode also reads tol and max_iter (LogisticRegression._param_handoff). A
    # fit that stops within an iteration or two is mostly MLX's fixed cost whatever its
    # labels, and a looser tol or a smaller max_iter stops one there (table
    # `logreg-params`, 21 pairs a point, every loss reproduced by a second run: the
    # benchmark's binary target, labels with moderate multinomial-logit signal and
    # random labels, binary at 200000 x 1-64, three classes at 60000 x 2-64). On random
    # labels tol = 3e-3 and 0.01 lost at 200000 x 16 and 64 on both versions
    # (0.50x-0.82x), and tol = 5e-4 and 1e-3 at 200000 x 64 on 1.7.2 (0.90x-0.93x);
    # at the default, 1e-4, and at 1e-5 every fit won (1.07x at worst). max_iter = 1
    # lost at 200000 x 64 on both versions and at 60000 x 16 with three classes on
    # 1.7.2 (0.76x-0.94x); max_iter = 2 tied at worst (0.98x), and from 3 every fit won
    # (1.05x at worst). So a patched fit needs tol of at most the estimator's default —
    # the tol every size rule was measured at, a rule with no knob — and max_iter of at
    # least logreg_min_max_iter. It gives up every looser tol that won (1.05x-3.5x)
    # and max_iter = 2 (1.05x-2.4x).
    logreg_min_max_iter: int = 3
    #
    # SVC: the axis that matters is *width* (see SVC._below_crossover). Per SMO
    # iteration mlxlearn costs about the same at any width while LIBSVM pays O(n*d)
    # per kernel row it misses. Table `svc`: binary rbf and poly fits on the
    # benchmark's target, default gamma and degree, at C = 1 and 10 everywhere, 0.1
    # and 100 around the floors. At 16-64 features both lost at C = 10 and 100 on
    # both versions (0.15x-0.85x), and poly at C = 1 up to 32 features; at 128
    # features they won from 3 000 rows (1.04x-4.1x). But how many rows become
    # support vectors is set by the labels, which a fit cannot see before it runs
    # (table `svc-hard`): with little or no signal in them — x0 + N(0, 3²) > 0,
    # random, XOR — nearly every row is one, and at 128 features 3 000-6 000 rows
    # lost at C = 10 and 100 on 1.7.2 (0.83x-0.94x) and 3 000 on 1.9.0 too
    # (0.91x-0.94x), as did one-vs-one pairs of 3 000 x 128 on random labels
    # (0.81x-0.91x, both versions). From 256 features every one of those won, at
    # every C, from 1 500 x 256 and 1 024 x 512 (1.21x-5.9x). So the width floor is
    # 256 features. It gives up everything at 128-255 features that won: the
    # benchmark's target from 3 000 x 128 (1.11x-4.1x), weak labels at 8 000 x 128
    # (1.61x-3.2x), one-vs-one pairs of 3 000 x 128 on overlapping classes
    # (1.09x-2.0x). The work floor, 1 500 x 256, keeps 1 000 x 256 on LIBSVM (a 1.04x
    # win at worst); svc_min_samples lets 1 024 x 512 through. Above all three floors
    # every binary fit measured at the default kernel parameters won on both
    # versions, 1.36x-8.4x, on this Gaussian data. The linear kernel (table
    # `svc-linear`) and the sigmoid kernel (table `svc-sigmoid`: won at C = 0.1, lost
    # at C = 10 and 100 nearly everywhere, down to 0.06x) are excluded at every size;
    # patched mode excludes poly too, and reads C and cache_size for rbf (below).
    svc_min_features: int = 256
    svc_min_samples: int = 1_024
    svc_min_work: int = 384_000  # n_samples * n_features
    #
    # One-vs-one fits K(K-1)/2 subproblems of 2n/K rows on average, each paying a
    # fixed Gram dispatch and its own Python SMO loop, so many small classes lose
    # even when the total clears the floors above. The mean pair must clear the
    # binary work floor itself: on separated blobs (table `svc-multi`) pairs of
    # 256 x 512 (K=16) ran 0.73x-0.81x and 500 x 256 (K=32) 0.70x-0.77x, while pairs
    # of 1 500 x 256 and 1 024 x 512 won on overlapping classes (table
    # `svc-multi-overlap`, 1.39x-3.5x) and on random labels at C = 1-100 (table
    # `svc-hard`, 1.23x-4.2x). What the floor gives up are pairs of 512 x 512 and
    # 1000 x 256, which won (on separated blobs 1.28x-1.39x and 1.17x-1.21x;
    # 1000 x 256 on overlapping classes 1.09x-2.0x). This floor is on the *mean*
    # pair's work, (2n/K) * n_features, which ignores class imbalance by
    # construction. Every figure here is the host solver's: a multiclass rbf fit on the
    # GPU now runs its pairs concurrently in one Metal kernel (SVC._solve_pairs_on_gpu),
    # and neither this floor nor the two above has been recalibrated for it, so they
    # route the fits they routed before.
    svc_min_pair_work: int = 384_000
    #
    # Patched mode sends only the rbf kernel to MLX, and reads C and cache_size
    # (SVC._kernel_handoff). The floors above were set on isotropic Gaussian data,
    # whose intrinsic dimension is its width. What decides is how many SMO iterations
    # a fit takes per row, and on data of lower intrinsic dimension C multiplies them
    # (table `svc-structured`: five or ten latent directions plus noise, singular
    # values 1/i, features smoothed over 16 neighbours, Student-t(2) features and
    # column scales 10^U(0, 2) at 1 024-4 000 x 256-512, and scikit-learn's digits
    # upsampled to 256-784 pixels). At C = 10 they lost at 1 024-1 797 rows on both
    # versions, down to 0.54x for rbf and 0.26x for poly, the digits at 256 and 400
    # pixels 0.57x-0.81x; at C <= 1 every rbf fit won, 1.06x-9.2x, at every gamma from
    # 0.03 to 100 times "scale", with tol = 1e-5, with balanced class weights and at
    # 8 000 rows. So an rbf fit needs a mean box — C times the mean class and sample
    # weight, which balanced class weights, or sample weights normalized to mean 1,
    # leave at C — of at most svc_max_c, compared with a relative tolerance of 1e-9 so
    # that the rounding of that mean cannot move a fit. It gives up every rbf fit above
    # C = 1 that won: the benchmark's target at C = 10 and 100 from 256 features
    # (table `svc`, 1.36x-3.8x), labels with little or no signal (`svc-hard`,
    # 1.10x-2.7x), overlapping one-vs-one classes (`svc-multi-overlap`, 1.39x-2.0x),
    # gamma of 0.8-10 times "scale" (`svc-params`, 1.20x-4.6x) and the structured data
    # at 2 000-4 000 rows (`svc-structured`, 1.04x-2.3x).
    #
    # The poly kernel joins the linear and sigmoid kernels in patched mode's
    # never-on-MLX set (table `svc-poly`): even at C <= 1 it lost on data the other
    # grids did not hold — at C = 1 on digits upsampled to 256-784 pixels and on five
    # latent directions with random labels at 1 500 x 256 (0.46x-0.73x), at C = 0.5 on
    # the digits (0.61x-0.87x), and at gamma 1.25-1.5 times "scale" (0.46x-0.89x) —
    # where rbf, on the same data, won everywhere (1.70x-6.3x). It gives up every poly
    # fit the previous rule (degree 3 or more, coef0 = 0, C <= 1) sent to MLX that won,
    # 1.07x-7.0x across the grids. And a fit whose cache_size is below scikit-learn's
    # default of 200 MB goes to LIBSVM too (table `svc-cache`): rows a smaller cache
    # cannot hold are evaluated on demand, and at 1 MB five latent directions at
    # 4 000 x 256 lost on 1.7.2 (0.92x; 0.95x on 1.9.0), while at 200 and 1 000 MB
    # every fit won (3.4x-6.6x). It gives up 3-100 MB where they won (1.03x-4.6x).
    # Neither rule has a knob.
    svc_max_c: float = 1.0
    #
    # Not a dispatch rule but where a fitted SVC computes (SVC._pairwise_decision): a
    # predict or decision_function of at most this much work — query rows x support
    # vectors x features — is evaluated on the host, in NumPy float32, from the same
    # operand; larger batches on the fitted device. What the device charges a batch
    # of a few rows is the call, 0.64-0.98 ms once the GPU has idled: with this at
    # 0, a single row lost to LIBSVM at five of the six fits of `svc-predict` on one
    # version or both (0.27x-0.78x), and 2-4 rows at four of them; only at 6000 x 512,
    # where LIBSVM's own predict is slowest, did the device tie or win. On the host
    # (table `svc-predict`) single rows tied or won
    # (1.06x-2.7x) and every larger batch won, on both versions. The host was faster
    # than an idle GPU up to about 5e7 of this work at 128 features and further out
    # at wider data; 2**25 is below the crossover at every shape measured.
    svc_host_inference_work: int = 33_554_432

    # ---- blocking ---------------------------------------------------------------------
    distance_block_bytes: int = 256 * 1024 * 1024
    distance_block_bytes_max: int = 1024 * 1024 * 1024
    query_block_rows: int = 1024
    query_block_align: int = 128
    cpu_distance_block_bytes: int = 128 * 1024 * 1024
    cpu_query_block_rows: int = 256

    # ---- solvers ----------------------------------------------------------------------
    lbfgs_history: int = 10
    lbfgs_max_line_search: int = 32
    #: LogisticRegression's dense L-BFGS preconditioner: the curvature at the zero start,
    #: used as every iteration's initial inverse Hessian where a gate on a sample of the
    #: rows finds correlated features and no near-separable classes
    #: (``_solvers/logistic.py``, ``docs/fp32_policy.md`` §2.3). Off, every fit takes
    #: the diagonal path, bit for bit as before it, and no sample is taken. For tests and
    #: bisection only: MLXLEARN_LBFGS_DENSE_PRECONDITIONER=0.
    lbfgs_dense_preconditioner: bool = True
    smo_kernel_cache_fraction: float = 0.05  # of physical memory
    smo_kernel_cache_min_bytes: int = 32 * 1024 * 1024
    smo_kernel_cache_max_bytes: int = 2 * 1024 * 1024 * 1024
    smo_max_iter_factor: int = 100  # max_iter=-1 -> factor * n_samples, bounded below
    smo_min_iter_budget: int = 10_000_000

    # ---- memory -----------------------------------------------------------------------
    #
    # MLX returns freed buffers to its own cache rather than to the OS, and its
    # default cache limit is the memory limit (~95% of RAM). One 200k x 128
    # kneighbors call left 10 GiB cached with nothing alive; fits whose n_train
    # varied kept adding, to 50 GiB after six. So a top-level MLX operation that
    # grew the cache by more than this fraction of physical memory releases it on
    # the way out (mx.clear_cache frees only idle buffers and changes no setting).
    # The cost is reallocation on the next call of the same shape: a repeated
    # 150k x 64 / 6k-query kneighbors measured 470 -> 618 ms with the release
    # forced every call (it grows the cache by 7.5 GiB). 1/16 of RAM leaves that
    # case alone on a 128 GB machine (8 GiB threshold) and releases it on a 16 GB
    # one (1 GiB), where keeping it would pin half the machine. 0 disables it.
    mlx_cache_release_fraction: float = 1 / 16

    # ---- behavior ---------------------------------------------------------------------
    #: Force a specific backend regardless of size. For tests and bisection only.
    force_backend: ForceBackend = ""
    #: Ring-buffer size for diagnostics. The buffer is shared by every thread, so
    #: this is read from the process-wide default only (MLXLEARN_DIAGNOSTICS_CAPACITY,
    #: at import and on reset_config); tuning_context cannot resize it.
    diagnostics_capacity: int = 512
    #: mlxlearn's own Metal kernels (``_kernels/_metal.py``): the neighbor search's fused
    #: distance-and-selection kernel and SVC's one-vs-one SMO. Off, the neighbor search
    #: runs on MLX's built-in operations, which give the same answers bit for bit on the
    #: MLX versions its kernel is validated on, and SVC solves its pairs one after another
    #: on the host solver, at the same optimum to within tol. For tests and bisection
    #: only: MLXLEARN_METAL_KERNELS=0.
    metal_kernels: bool = True

    def __post_init__(self) -> None:
        # Normalized like Config's booleans: a NumPy bool is the value meant, and an int
        # or a string is a mistake to name, not a truth value to guess.
        object.__setattr__(self, "metal_kernels", _as_bool("metal_kernels", self.metal_kernels))
        object.__setattr__(
            self,
            "lbfgs_dense_preconditioner",
            _as_bool("lbfgs_dense_preconditioner", self.lbfgs_dense_preconditioner),
        )
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, int) and not isinstance(value, bool) and value < 0:
                raise ValueError(f"tuning value {f.name} must be non-negative, got {value}")
        # Zero here is not "off", it is a division by zero or an empty block
        # somewhere deep in a kernel. The error belongs at configuration time,
        # where it can name the knob.
        for name in _AT_LEAST_ONE:
            if getattr(self, name) < 1:
                raise ValueError(
                    f"tuning value {name} must be at least 1, got {getattr(self, name)}"
                )
        if not 0.0 < self.smo_kernel_cache_fraction <= 1.0:
            raise ValueError(
                "smo_kernel_cache_fraction must be in (0, 1], got "
                f"{self.smo_kernel_cache_fraction}"
            )
        # Written so that NaN fails the comparison and is rejected too.
        for name in ("mlx_cache_release_fraction", "knn_probe_max_uncertified"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        if self.force_backend not in get_args(ForceBackend):
            # Not lower-cased: "MLX" is as much a typo as "gpu", and a forced
            # backend that silently meant something else defeats the bisection
            # the knob exists for. The GPU is chosen with Config.device.
            raise ValueError(
                "force_backend must be one of '', 'mlx', 'cpu', 'sklearn' (the MLX "
                "GPU backend is 'mlx'; pick the device with mlxlearn's device option); "
                f"got {self.force_backend!r}"
            )


_AT_LEAST_ONE = (
    "distance_block_bytes",
    "distance_block_bytes_max",
    "query_block_rows",
    "query_block_align",
    "cpu_distance_block_bytes",
    "cpu_query_block_rows",
    "lbfgs_history",
    "lbfgs_max_line_search",
    "knn_probe_rows",
)

_ENV_PREFIX = "MLXLEARN_"


def _tuning_from_env(base: Tuning) -> Tuning:
    """Apply ``MLXLEARN_*`` overrides on top of ``base``.

    A malformed value is an error, not a silent fallback to the default: an
    environment variable that quietly does nothing is worse than no override at
    all, because you cannot tell the two apart from the outside.
    """
    overrides: dict[str, Any] = {}
    for f in fields(Tuning):
        key = _ENV_PREFIX + f.name.upper()
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            if f.type is int or f.type == "int":
                overrides[f.name] = int(raw)
            elif f.type is float or f.type == "float":
                overrides[f.name] = float(raw)
            elif f.type is bool or f.type == "bool":
                overrides[f.name] = _parse_bool(raw)
            else:
                overrides[f.name] = raw
        except ValueError as exc:
            raise ValueError(f"{key}={raw!r} is not a valid value for {f.name}") from exc
    if not overrides:
        return base
    try:
        return replace(base, **overrides)
    except ValueError as exc:
        names = ", ".join(_ENV_PREFIX + name.upper() for name in sorted(overrides))
        raise ValueError(f"invalid tuning override in the environment ({names}): {exc}") from exc


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _parse_bool(raw: str) -> bool:
    """An environment variable's boolean, or ``ValueError`` for anything else."""
    text = raw.strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"{raw!r} is not a boolean (use 1/0, true/false)")


def _config_from_env() -> Config:
    """The process-wide default configuration: :class:`Config` plus ``MLXLEARN_*``.

    This is what a fresh process — a loky or multiprocessing worker included —
    starts from. Same rule as tuning: a malformed value is an error.
    """
    overrides: dict[str, Any] = {}
    for f in fields(Config):
        key = _ENV_PREFIX + f.name.upper()
        raw = os.environ.get(key)
        if raw is None:
            continue
        text = raw.strip()
        if f.name in ("deterministic", "diagnostics"):
            if text.lower() in _TRUE:
                overrides[f.name] = True
            elif text.lower() in _FALSE:
                overrides[f.name] = False
            else:
                raise ValueError(f"{key}={raw!r} is not a boolean (use 1/0, true/false)")
        elif f.name == "random_state":
            if text.lower() in ("", "none"):
                overrides[f.name] = None
            else:
                try:
                    overrides[f.name] = int(text)
                except ValueError as exc:
                    raise ValueError(f"{key}={raw!r} is not an integer seed") from exc
        else:
            overrides[f.name] = text
    try:
        return Config(**overrides)
    except (TypeError, ValueError) as exc:
        names = ", ".join(_ENV_PREFIX + name.upper() for name in sorted(overrides))
        raise ValueError(f"invalid configuration in the environment ({names}): {exc}") from exc


# --------------------------------------------------------------------------------------
# state: process-wide defaults, context-local overrides
# --------------------------------------------------------------------------------------
#
# Two immutable values and a lock. set_config swaps the default atomically; the
# overrides live in context variables, so they are per thread and per asyncio
# task without any per-thread copy of the defaults that could go stale (a pool
# thread that had copied the default on first use would never see a later
# set_config).


class _ConfigScope:
    """One active :func:`config_context`: the fields it overrides, merged with outer ones.

    The resolved :class:`Config` is cached against the default it was built on,
    so ``get_config`` stays a couple of attribute reads, and a ``set_config``
    made while the scope is active still shows through for every field the
    scope does not override.
    """

    __slots__ = ("overrides", "_cache")

    def __init__(self, overrides: dict[str, Any], default: Config) -> None:
        self.overrides = overrides
        self._cache = (default, replace(default, **overrides))

    def resolve(self, default: Config) -> Config:
        base, resolved = self._cache
        if base is not default:
            resolved = replace(default, **self.overrides)
            self._cache = (default, resolved)
        return resolved


_lock = threading.Lock()
_default_config: Config = _config_from_env()
_default_tuning: Tuning = _tuning_from_env(Tuning())
_config_scope: ContextVar[_ConfigScope | None] = ContextVar("mlxlearn_config", default=None)
_tuning_scope: ContextVar[Tuning | None] = ContextVar("mlxlearn_tuning", default=None)


def _check_config_names(kwargs: dict[str, Any]) -> None:
    known = {f.name for f in fields(Config)}
    unknown = set(kwargs) - known
    if unknown:
        raise TypeError(
            f"unknown configuration option(s): {', '.join(sorted(unknown))}. "
            f"Valid options are: {', '.join(sorted(known))}."
        )


def get_config() -> Config:
    """Return the configuration in effect here.

    That is the process-wide default set by :func:`set_config`, with the fields
    named by any enclosing :func:`config_context` overridden.
    """
    scope = _config_scope.get()
    default = _default_config
    return default if scope is None else scope.resolve(default)


def set_config(**kwargs: Any) -> Config:
    """Update the process-wide default configuration and return the configuration in effect.

    Only the fields named are changed. Unknown field names are an error rather
    than being stored and ignored. The new default reaches every thread,
    including scikit-learn's ``n_jobs`` worker threads; process workers start
    from the ``MLXLEARN_*`` environment instead (see the module docstring).
    Inside a :func:`config_context`, the fields that context overrides keep its
    value until it exits.

    Examples
    --------
    >>> import mlxlearn
    >>> _ = mlxlearn.set_config(fallback_policy="raise")
    >>> mlxlearn.get_config().fallback_policy
    'raise'
    >>> _ = mlxlearn.set_config(fallback_policy="warn")
    """
    global _default_config
    _check_config_names(kwargs)
    with _lock:
        _default_config = replace(_default_config, **kwargs)
    return get_config()


@contextmanager
def config_context(**kwargs: Any):
    """Temporarily override configuration for the current context.

    The override applies to this thread (and to asyncio tasks started from it)
    and nowhere else. The previous configuration is restored even if the body
    raises.

    Examples
    --------
    >>> import mlxlearn
    >>> with mlxlearn.config_context(fallback_policy="raise"):
    ...     mlxlearn.get_config().fallback_policy
    'raise'
    >>> mlxlearn.get_config().fallback_policy
    'warn'
    """
    _check_config_names(kwargs)
    outer = _config_scope.get()
    merged = {**outer.overrides, **kwargs} if outer is not None else dict(kwargs)
    # Built, and therefore validated, before anything changes.
    scope = _ConfigScope(merged, _default_config)
    token = _config_scope.set(scope)
    try:
        yield get_config()
    finally:
        _config_scope.reset(token)


def get_tuning() -> Tuning:
    """Return the private tuning values in effect here."""
    scope = _tuning_scope.get()
    return _default_tuning if scope is None else scope


def _process_tuning() -> Tuning:
    """The process-wide tuning default, ignoring any :func:`tuning_context`.

    For the few knobs that size something shared by every thread.
    """
    return _default_tuning


@contextmanager
def tuning_context(**kwargs: Any):
    """Temporarily override private tuning in the current context. Tests and debugging only."""
    known = {f.name for f in fields(Tuning)}
    unknown = set(kwargs) - known
    if unknown:
        raise TypeError(f"unknown tuning option(s): {', '.join(sorted(unknown))}")
    token = _tuning_scope.set(replace(get_tuning(), **kwargs))
    try:
        yield get_tuning()
    finally:
        _tuning_scope.reset(token)


def reset_config() -> None:
    """Restore configuration and tuning to their defaults.

    Resets the process-wide defaults to what the ``MLXLEARN_*`` environment says
    (re-reading it), and drops any override active in the current context.
    """
    global _default_config, _default_tuning
    config = _config_from_env()
    tuning = _tuning_from_env(Tuning())
    with _lock:
        _default_config = config
        _default_tuning = tuning
    _config_scope.set(None)
    _tuning_scope.set(None)


# Importable so a test can assert the public surface has not silently grown. The
# plan fixes this list at six entries; the 50 environment variables of the
# ancestor do not get to become 50 public options by accident.
PUBLIC_CONFIG_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Config))
