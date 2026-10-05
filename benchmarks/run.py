"""Benchmark runner: measures crossovers, writes results and a report.

Usage::

    python -m benchmarks.run --profile smoke
    python -m benchmarks.run --profile full --output bench/results.json --report bench/report.md
    python -m benchmarks.run --profile published --report bench/published.md
    python -m benchmarks.run --environment-only --output bench/environment.json

Size grids are **per algorithm**, not universal. Exact KNN over embeddings is
interesting into the millions of rows; exact SVC is quadratic in the number of
samples and is interesting at 10⁴–10⁵. Running both on one grid would measure the
wrong thing twice.

The output feeds two consumers: ``docs/benchmarks.md``, which humans read, and the
crossover thresholds in ``mlxlearn._common.config.Tuning``, which dispatch reads.
That second consumer is the point — a published crossover that dispatch ignores is
a claim, not a mechanism. Every result therefore records the backend the mlxlearn
side actually ran on, so the gate can judge the dispatch decisions the configured
``Tuning`` really makes rather than a crossover inferred from the run it is judging.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import gate
from ._harness import MIN_SIGNIFICANCE_SAMPLES, Comparison, compare, environment

__all__ = ["main", "PROFILES", "PROFILE_SETTINGS", "Case"]

#: The scikit-learn estimators' own defaults, which a case only overrides — and only
#: names in its label — when it means to.
_DEFAULT_OPTIONS = {"sklearn_algorithm": "auto", "kernel": "rbf", "C": 1.0, "n_neighbors": 10}

#: A case the gate would fail is timed again over this many times its iterations,
#: and the gate fails it only if the loss reproduces there: a false alarm near 1% per
#: case, over the hundred-odd cases a night gates, would otherwise fail most nights
#: (SIGNIFICANCE_Z in _harness.py). A real regression reproduces.
CONFIRMATION_FACTOR = 3


@dataclass(frozen=True)
class Case:
    """One benchmarked workload.

    ``options`` carries the knobs a case changes from the defaults above —
    ``sklearn_algorithm`` (the ``NearestNeighbors`` algorithm, which selects the
    stock baseline; the mlxlearn side is given it too), ``n_neighbors``,
    ``query_neighbors`` (the k a ``kneighbors`` call asks for, when not the fitted
    one), ``kernel``, ``C`` and ``cache_size`` (SVC), ``tol`` and ``max_iter``
    (LogisticRegression),
    ``n_classes`` (classification targets) — and each one appears in
    the label, so two differently configured cases can never share a name.
    ``iterations`` caps the run's iteration count for a case whose single call
    takes minutes; it never raises it, so ``--iterations 1`` still means one.
    """

    algorithm: str
    operation: str
    n_samples: int
    n_features: int
    n_queries: int = 0
    options: dict = field(default_factory=dict)
    iterations: int | None = None

    def option(self, name: str):
        return self.options.get(name, _DEFAULT_OPTIONS.get(name))

    @property
    def label(self) -> str:
        base = f"{self.algorithm}/{self.operation}/{self.n_samples}x{self.n_features}"
        if self.n_queries:
            base = f"{base}/q{self.n_queries}"
        for name in sorted(self.options):
            base = f"{base}/{name}={self.options[name]}"
        return base


def _knn_grid(sizes, features, queries, operations=("fit", "kneighbors"), **options) -> list[Case]:
    return [
        Case("knn", op, n, d, q, options=dict(options))
        for n in sizes
        for d in features
        for q in queries
        for op in operations
    ]


def _svc_grid(shapes, *, kernels=("rbf",), Cs=(1.0,), n_classes=None, iterations=None):
    cases = []
    for n, d in shapes:
        for kernel in kernels:
            for C in Cs:
                options: dict = {}
                if kernel != "rbf":
                    options["kernel"] = kernel
                if C != 1.0:
                    options["C"] = C
                if n_classes is not None:
                    options["n_classes"] = n_classes
                cases.append(Case("svc", "fit", n, d, options=options, iterations=iterations))
    return cases


PROFILES: dict[str, list[Case]] = {
    # Fast enough to run on a laptop, or in CI as a harness check, while still
    # landing on both sides of every Tuning threshold (tests/tooling checks it):
    # each estimator has cases patched dispatch hands to scikit-learn and cases it
    # sends to MLX, so the MLX fit and query paths are gated, not only the handoffs.
    "smoke": [
        # Neighbors: below the row floor; wide and above both floors (MLX); 16-255
        # features, short (scikit-learn) and tall (MLX, at 64 and at 16); fewer than
        # 16 features, tall, to scikit-learn's brute force where it would build no
        # tree; narrow data the KD-tree takes whatever its size ("tree-wins"), and
        # below the work floor.
        *_knn_grid([1_000], [32], [256]),
        *_knn_grid([65_536], [256], [256]),
        *_knn_grid([262_144], [64], [256], operations=("kneighbors",)),
        *_knn_grid([1_048_576], [64], [256], operations=("kneighbors",)),
        *_knn_grid([1_048_576], [16], [256], operations=("kneighbors",)),
        *_knn_grid([1_048_576], [8], [256], operations=("kneighbors",)),
        *_knn_grid(
            [1_048_576], [8], [256], operations=("kneighbors",), sklearn_algorithm="brute"
        ),
        *_knn_grid([50_000], [3], [256], operations=("kneighbors",)),
        # Data the size rules would send to MLX and the fit-time probe does not:
        # duplicated rows at a wide shape — about 20 copies each, and about 2, which
        # leaves only a few rows in a hundred uncertified — and integer features at a
        # tall narrow one.
        *_knn_grid([24_000], [256], [256], operations=("kneighbors",), data="dups"),
        *_knn_grid([30_000], [256], [64], operations=("kneighbors",), data="dup2"),
        *_knn_grid([1_048_576], [64], [256], operations=("kneighbors",), data="binary"),
        # More neighbors than the rules keep on MLX, at a shape they otherwise do: asked
        # of the fit, and asked of a query on an index fitted with the default k.
        *_knn_grid([1_048_576], [64], [1_024], operations=("kneighbors",), n_neighbors=100),
        *_knn_grid([65_536], [256], [256], operations=("kneighbors",), query_neighbors=100),
        # Wide but short: the size floors pass it, the wide-data row floor does not.
        *_knn_grid([4_096], [2_048], [32], operations=("kneighbors",)),
        # Logistic regression: below the sample floor; binary but too wide, and too
        # short; binary, tall and narrow (MLX); multinomial (MLX).
        Case("logreg", "fit", 5_000, 32),
        Case("logreg", "fit", 200_000, 128),
        Case("logreg", "fit", 100_000, 16),
        Case("logreg", "fit", 200_000, 16),
        Case("logreg", "fit", 60_000, 64, options={"n_classes": 5}),
        # Multinomial on one feature, three classes.
        Case("logreg", "fit", 60_000, 1, options={"n_classes": 3}),
        # A fit the solver parameters stop within a few iterations, at a shape the size
        # rules send to MLX: a tol looser than the default, a short max_iter.
        Case("logreg", "fit", 200_000, 16, options={"tol": 1e-3}),
        Case("logreg", "fit", 200_000, 16, options={"max_iter": 2}),
        # SVC: below every floor; enough rows but too narrow, at 64 and at 128
        # features; wide (MLX); and a many-class problem each of whose pairs is below
        # the pair floor.
        Case("svc", "fit", 1_000, 32),
        Case("svc", "fit", 4_000, 64),
        Case("svc", "fit", 3_000, 128),
        Case("svc", "fit", 4_000, 256),
        *_svc_grid([(2_048, 512)], n_classes=4),
        *_svc_grid([(2_048, 512)], n_classes=16),
        # SVC above the measured box, C = 10, the poly kernel at its defaults, and a
        # cache smaller than scikit-learn's default (scikit-learn), whatever the size.
        Case("svc", "fit", 1_500, 256, options={"C": 10.0}),
        Case("svc", "fit", 1_500, 256, options={"kernel": "poly"}),
        Case("svc", "fit", 1_500, 256, options={"cache_size": 1.0}),
        # SVC inference on a model fitted on MLX: one row, answered on the host, and a
        # thousand, on the device.
        Case("svc", "predict", 4_000, 256, 1),
        Case("svc", "predict", 4_000, 256, 1_000),
    ],
    # The regression grid. Hours, on fixed hardware.
    "full": [
        # The stock baseline is scikit-learn's default algorithm="auto", which uses
        # a KD-tree at n_features <= 15. Low dimensions are where that matters, and
        # where mlxlearn's brute force can lose to it by orders of magnitude.
        *_knn_grid(
            [500, 2_000, 8_000, 32_000, 128_000, 512_000],
            [2, 3, 8, 64, 768],
            [1, 256, 4_096],
        ),
        # Large query batches — t-SNE, UMAP and LOF query tens of thousands of points
        # at once — at low and moderate width.
        *_knn_grid(
            [50_000, 200_000], [2, 3, 8, 32], [10_000, 50_000], operations=("kneighbors",)
        ),
        # Brute against brute at low width: the kernel comparison, kept apart from
        # the default-against-default comparison above.
        *_knn_grid(
            [8_000, 128_000], [3, 8], [256], operations=("kneighbors",), sklearn_algorithm="brute"
        ),
        *[Case("logreg", "fit", n, d) for n in (1_000, 10_000, 100_000, 500_000) for d in (16, 128, 1_024)],
        *_svc_grid([(n, d) for n in (500, 2_000, 8_000, 32_000) for d in (16, 128)]),
        *[Case("svc", "predict", n, d) for n in (2_000, 8_000) for d in (16, 128)],
        # One-vs-one solves K(K-1)/2 pairs of about 2n/K rows each, so class count,
        # not total size, decides what each pair costs.
        *[
            case
            for k in (8, 32, 128)
            for case in _svc_grid([(2_048, 512)], n_classes=k)
        ],
    ],
    # The cases docs/benchmarks.md publishes, so its tables can be regenerated. Its
    # settings (PROFILE_SETTINGS below) match how those tables were measured: the
    # Layer 1 classes forced onto the MLX device, isolating the kernels from the
    # dispatch decision. That is a kernel comparison, not the patched claim.
    "published": [
        # Neighbors against scikit-learn's brute force, k=10: each fit, and 256-query
        # batches, from below the row floor to past the floors, wide and tall, and
        # tall but narrower than the width floor; then the batch sizes a fit-time
        # rule cannot see, 1, 64 and 4096 queries, on either side of them — 64
        # queries is where short data below 256 features loses once the GPU has
        # idled.
        *[
            Case("knn", op, n, d, 256, options={"sklearn_algorithm": "brute"})
            for n, d in (
                (1_000, 32),
                (4_096, 64),
                (32_768, 16),
                (50_000, 32),
                (8_192, 768),
                (131_072, 128),
                (128_000, 768),
                (1_000_000, 16),
                (1_000_000, 64),
            )
            for op in ("fit", "kneighbors")
        ],
        *[
            Case("knn", "kneighbors", n, d, q, options={"sklearn_algorithm": "brute"})
            for n, d in (
                (1_000, 32), (8_192, 768), (131_072, 128), (1_000_000, 16), (1_000_000, 64)
            )
            for q in (1, 64, 4_096)
        ],
        # SVC: width decides. Binary rbf and poly at C=1 and C=10 on either side of
        # the width and work floors, the sigmoid kernel at C=10 and the linear kernel
        # (slow on MLX, so fewer iterations), and one-vs-one pairs on either side of
        # the pair floor.
        *_svc_grid([(4_000, 32), (4_000, 64), (4_000, 256), (1_000, 512)]),
        *_svc_grid(
            [(4_000, 64), (2_000, 128), (4_000, 128), (4_000, 256)],
            kernels=("rbf", "poly"),
            Cs=(10.0,),
        ),
        *_svc_grid([(4_000, 256)], kernels=("sigmoid",), Cs=(10.0,)),
        *_svc_grid([(2_000, 20), (2_000, 256)], kernels=("linear",), iterations=5),
        *_svc_grid([(2_048, 512)], n_classes=4),
        *_svc_grid([(2_048, 512)], n_classes=16),
        *_svc_grid([(2_500, 20)], n_classes=4),
        # Logistic regression, fit with max_iter=200: binary fits narrow and wide,
        # short and tall, and multinomial fits either side of the sample floor.
        *[
            Case("logreg", "fit", n, d)
            for n, d in (
                (20_000, 16),
                (200_000, 16),
                (1_000_000, 16),
                (200_000, 128),
                (20_000, 1_024),
                (100_000, 1_024),
            )
        ],
        *[
            Case("logreg", "fit", n, 64, options={"n_classes": 5})
            for n in (20_000, 60_000, 200_000)
        ],
    ],
}

#: Per-profile defaults for the command-line settings; an explicit flag wins.
PROFILE_SETTINGS: dict[str, dict] = {
    "smoke": {},
    "full": {},
    "published": {"layer": "direct", "force_backend": "mlx", "iterations": 25},
}


def _make_data(case: Case, rng: np.random.Generator):
    X = rng.normal(size=(case.n_samples, case.n_features))
    if case.algorithm == "knn":
        Q = rng.normal(size=(case.n_queries, case.n_features))
        data = case.option("data")
        if data in ("dups", "dup2"):
            # Every training row repeated about 20 (or 2) times: the first float32
            # pass cannot certify a neighbor set that ties with copies it left out.
            base = max(case.n_samples // (20 if data == "dups" else 2), 1)
            X = X[:base][rng.integers(0, base, size=case.n_samples)]
        elif data == "binary":
            X = (rng.random(size=X.shape) < 0.2).astype(np.float64)
            Q = (rng.random(size=Q.shape) < 0.2).astype(np.float64)
        return X, None, Q
    n_classes = case.option("n_classes")
    if n_classes is None:
        y = (X[:, 0] + rng.normal(scale=0.5, size=case.n_samples) > 0).astype(int)
    else:
        # Overlapping Gaussian blobs, one per class, so every one-vs-one pair is a
        # real optimization problem rather than a trivially separable one.
        centers = rng.normal(scale=2.0, size=(n_classes, case.n_features))
        y = rng.integers(0, n_classes, size=case.n_samples)
        X = centers[y] + rng.normal(size=X.shape)
    return X, y, X[: case.n_queries or min(1_000, case.n_samples)]


def _estimators(case: Case, layer: str):
    """The (mlxlearn, scikit-learn) estimator factories for one case."""
    import sklearn.linear_model
    import sklearn.neighbors
    import sklearn.svm

    if layer == "patched":
        from mlxlearn.patching import _estimators as mlx_linear

        mlx_neighbors = mlx_svm = mlx_linear
    else:
        from mlxlearn import linear_model as mlx_linear
        from mlxlearn import neighbors as mlx_neighbors
        from mlxlearn import svm as mlx_svm

    if case.algorithm == "knn":
        # Both sides are built as the user built them: a patched NearestNeighbors
        # given algorithm="brute" is compared with stock brute force, not with the
        # KD-tree patched dispatch would otherwise hand the fit to.
        algorithm = case.option("sklearn_algorithm")
        k = case.option("n_neighbors")
        return (
            lambda: mlx_neighbors.NearestNeighbors(n_neighbors=k, algorithm=algorithm),
            lambda: sklearn.neighbors.NearestNeighbors(n_neighbors=k, algorithm=algorithm),
        )
    if case.algorithm == "logreg":
        params = {"max_iter": case.options.get("max_iter", 200)}
        if "tol" in case.options:
            params["tol"] = case.options["tol"]
        return (
            lambda: mlx_linear.LogisticRegression(**params),
            lambda: sklearn.linear_model.LogisticRegression(**params),
        )
    if case.algorithm == "svc":
        params = {"kernel": case.option("kernel"), "C": case.option("C")}
        # The data are standard normal, so scikit-learn's "scale" is 1 / n_features
        # and a numeric gamma is gamma * n_features times it.
        params.update(
            {
                k: case.options[k]
                for k in ("gamma", "degree", "coef0", "cache_size")
                if k in case.options
            }
        )
        return (lambda: mlx_svm.SVC(**params), lambda: sklearn.svm.SVC(**params))
    raise ValueError(f"unknown algorithm {case.algorithm!r}")


def _runner(
    case: Case, data, layer: str
) -> tuple[Callable, Callable, tuple[int, ...], Callable[[], str]]:
    """Build the paired (mlxlearn, scikit-learn) callables for one case.

    The fourth element reports the backend the mlxlearn side ran on — its
    ``execution_backend_`` — once the callable has run.

    ``layer="patched"`` is the default and is what the performance gate is about:
    the claim is that *patched dispatch* is never significantly slower than stock
    scikit-learn, and patched dispatch hands sub-crossover problems to scikit-learn
    outright.

    ``layer="direct"`` measures the Layer 1 classes, which never dispatch to
    scikit-learn and instead fall back to their own CPU path below the crossover.
    Useful for development — it shows what the MLX and CPU paths actually cost —
    but reporting it as though it were the patched number would understate small
    problems, since it measures a path patched users never execute.
    """
    make_mlx, make_sk = _estimators(case, layer)
    X, y, Q = data
    shape = (case.n_samples, case.n_features)
    seen: dict[str, str] = {}

    def backend_of() -> str:
        return seen["backend"]

    if case.operation == "fit":
        target = (X,) if y is None else (X, y)

        def mlx_fit():
            model = make_mlx().fit(*target)
            # A string, not the model: holding the last fitted model would keep its
            # device buffers alive through the next timed fit.
            seen["backend"] = model.execution_backend_
            return model

        return mlx_fit, lambda: make_sk().fit(*target), shape, backend_of

    target = (X,) if y is None else (X, y)
    mlx_model = make_mlx().fit(*target)
    sk_model = make_sk().fit(*target)
    seen["backend"] = mlx_model.execution_backend_
    if case.operation == "kneighbors":
        k = case.option("query_neighbors")
        handoff = getattr(mlx_model, "_query_neighbors_handoff", None)
        if k is not None and handoff is not None and handoff(k) is not None:
            # Patched dispatch answers this query with scikit-learn's brute force,
            # whichever backend fitted the index.
            seen["backend"] = "sklearn"
        return (
            lambda: mlx_model.kneighbors(Q, n_neighbors=k),
            lambda: sk_model.kneighbors(Q, n_neighbors=k),
            shape,
            backend_of,
        )
    if case.operation == "predict":
        return lambda: mlx_model.predict(Q), lambda: sk_model.predict(Q), shape, backend_of
    raise ValueError(f"unknown operation {case.operation!r} for {case.algorithm!r}")


def _dispatch_backend(case: Case, data) -> str:
    """What patched dispatch picks for this case under the configured ``Tuning``.

    Only needed when a run forces the backend, which hides the decision, so the probe
    lifts the force wherever it came from: the flag, or ``MLXLEARN_FORCE_BACKEND``
    in the process's ``Tuning``. Probed with the patched class because its
    below-threshold route is stock scikit-learn, which is always cheap to run; the
    Layer 1 class would take its CPU path instead. The thresholds are shared, so the
    answer — "mlx" or not — is the same for both.
    """
    from mlxlearn._common.config import tuning_context

    make_mlx, _ = _estimators(case, "patched")
    X, y, _Q = data
    with tuning_context(force_backend=""):
        model = make_mlx().fit(X) if y is None else make_mlx().fit(X, y)
    return model.execution_backend_


def _forcing(force_backend: str | None):
    if not force_backend:
        return contextlib.nullcontext()
    from mlxlearn._common.config import tuning_context

    return tuning_context(force_backend=force_backend)


def run_profile(
    profile: str,
    *,
    seed: int = 0,
    iterations: int = 7,
    layer: str = "patched",
    force_backend: str | None = None,
) -> list[Comparison]:
    cases = PROFILES[profile]
    rng = np.random.default_rng(seed)
    results: list[Comparison] = []

    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case.label}", flush=True)
        data = _make_data(case, rng)
        # A case's own count is a ceiling for slow cases, never a floor.
        n = min(case.iterations, iterations) if case.iterations else iterations
        try:
            with _forcing(force_backend):
                mlx_fn, sk_fn, shape, backend_of = _runner(case, data, layer)
                comparison = compare(
                    mlx_fn,
                    sk_fn,
                    label=case.label,
                    shape=shape,
                    iterations=n,
                    algorithm=case.algorithm,
                    operation=case.operation,
                    n_queries=case.n_queries,
                    options=dict(case.options),
                    layer=layer,
                    force_backend=force_backend or None,
                )
        except ImportError as exc:
            # An estimator that has not landed yet is skipped loudly. A silently
            # shorter benchmark report reads as "we measured everything".
            print(f"    skipped: {exc}", flush=True)
            continue
        comparison.extra["backend"] = backend_of()
        comparison.extra["dispatch"] = (
            _dispatch_backend(case, data) if force_backend else comparison.extra["backend"]
        )
        results.append(comparison)
        if comparison.significant:
            marker = ""
        elif comparison.mlxlearn.iterations < MIN_SIGNIFICANCE_SAMPLES:
            marker = "  (too few iterations to judge)"
        else:
            marker = "  (within noise)"
        print(f"    {comparison.speedup:.2f}x on {comparison.extra['backend']}{marker}", flush=True)

        if gate.evaluate({"results": [comparison.as_dict()]})[0]:
            with _forcing(force_backend):
                confirmation = compare(
                    mlx_fn, sk_fn, label=case.label, shape=shape, iterations=CONFIRMATION_FACTOR * n
                )
            comparison.extra["confirmation"] = confirmation.as_dict()
            print(
                f"    re-measured over {confirmation.mlxlearn.iterations} iterations: "
                f"{confirmation.speedup:.2f}x"
                f"{'' if confirmation.significant else '  (within noise)'}",
                flush=True,
            )

    return results


def find_crossovers(results: list[Comparison]) -> dict[str, dict]:
    """The smallest measured size at which mlxlearn significantly wins.

    Reported per (algorithm, operation), never averaged across them. An average
    over fit and predict describes a workload nobody runs. This is a *measurement*,
    for a human to compare against ``Tuning``; the gate does not use it, because a
    crossover inferred from the run being judged moves with any regression and
    would excuse it.
    """
    crossovers: dict[str, dict] = {}
    for result in sorted(results, key=lambda r: (r.shape[0], r.shape[1])):
        key = f"{result.extra['algorithm']}/{result.extra['operation']}"
        entry = crossovers.setdefault(
            key,
            {
                "crossover_n_samples": None,
                "max_speedup": 0.0,
                "max_speedup_best": 0.0,
                "samples_tested": [],
            },
        )
        entry["samples_tested"].append(result.shape[0])
        entry["max_speedup"] = max(entry["max_speedup"], result.speedup)
        entry["max_speedup_best"] = max(entry["max_speedup_best"], result.speedup_best)
        # A crossover requires a win on *both* statistics. Median alone would
        # place the crossover wherever scikit-learn's tail happened to be fat.
        if (
            entry["crossover_n_samples"] is None
            and result.significant
            and result.speedup > 1.0
            and result.speedup_best > 1.0
        ):
            entry["crossover_n_samples"] = result.shape[0]
    return crossovers


def write_report(
    path: Path,
    results: list[Comparison],
    env: dict,
    profile: str,
    layer: str = "patched",
    force_backend: str | None = None,
) -> None:
    crossovers = find_crossovers(results)
    forced = f", backend forced to `{force_backend}`" if force_backend else ""
    iterations = sorted({r.mlxlearn.iterations for r in results})
    lines = [
        "# Benchmark report",
        "",
        f"Profile: `{profile}`, layer: `{layer}`{forced}. Generated by `python -m benchmarks.run`.",
        "",
        "`patched` measures the classes `patch_sklearn()` installs — which is what the "
        "performance gate is about, since patched dispatch hands sub-crossover problems to "
        "scikit-learn outright. `direct` measures the Layer 1 classes, which never dispatch "
        "to scikit-learn and use their own CPU path instead. A forced backend overrides the "
        "dispatch decision to isolate a kernel; the `dispatch` column says what the configured "
        "`Tuning` would have picked.",
        "",
        "Timings are `mx.eval`-synchronized, warmed up, and reported as the median of "
        f"{', '.join(map(str, iterations)) or 0} interleaved iterations, with the interquartile "
        "range (IQR). A difference counts as **significant** only when the two medians are more "
        "than three standard errors apart, each median's standard error estimated from its IQR "
        "(`1.2533 · IQR/1.349 / √n`) and the two combined in quadrature; anything less is "
        "**within noise** and does not count as a win or a loss. A case the gate would fail is "
        f"re-measured over {CONFIRMATION_FACTOR}x the iterations, and fails only if the loss "
        "reproduces; the last column shows that re-measurement.",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(env, indent=2),
        "```",
        "",
        "## Measured crossovers",
        "",
        "The smallest size at which mlxlearn significantly beats stock scikit-learn, per "
        "algorithm and operation. Compare them with the thresholds in "
        "`mlxlearn._common.config.Tuning`, which is what dispatch reads; the gate judges the "
        "decisions `Tuning` actually made (the `backend` column below), not these numbers.",
        "",
        "| algorithm / operation | crossover (n_samples) | best speedup | sizes tested |",
        "|---|---|---|---|",
    ]
    for key, entry in sorted(crossovers.items()):
        crossover = entry["crossover_n_samples"]
        lines.append(
            f"| `{key}` | {crossover if crossover is not None else 'not reached'} | "
            f"{entry['max_speedup']:.2f}x | {sorted(set(entry['samples_tested']))} |"
        )

    lines += [
        "",
        "## All measurements",
        "",
        "Two speedup columns. **median** is what a typical call costs; **best** compares the "
        "fastest observed run on each side. They differ most at small sizes, where "
        "scikit-learn's timings are heavy-tailed and its median sits well above its own "
        "minimum — reporting only the median ratio there would flatter mlxlearn.",
        "",
        "| case | backend | dispatch | mlxlearn median (IQR), ms | sklearn median (IQR), ms "
        "| speedup (median) | speedup (best) | significant | re-measured |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        again = result.extra.get("confirmation")
        rechecked = (
            f"{again['speedup']:.2f}x, {'significant' if again['significant'] else 'within noise'}"
            if again
            else ""
        )
        lines.append(
            f"| `{result.label}` | {result.extra.get('backend', '?')} | "
            f"{result.extra.get('dispatch', '?')} | "
            f"{result.mlxlearn.median_s * 1e3:.2f} ({result.mlxlearn.iqr_s * 1e3:.2f}) | "
            f"{result.sklearn.median_s * 1e3:.2f} ({result.sklearn.iqr_s * 1e3:.2f}) | "
            f"{result.speedup:.2f}x | {result.speedup_best:.2f}x | "
            f"{'yes' if result.significant else 'no'} | {rechecked} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--output", type=Path, default=Path("bench/results.json"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="timed iterations per case (default: the profile's, else 7); fewer than 5 "
        "never reports a significant difference, so --iterations 1 checks the plumbing only",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layer",
        choices=("patched", "direct"),
        default=None,
        help=(
            "'patched' (the default, except for the 'published' profile) measures what "
            "patch_sklearn() users get, which is what the gate claims; 'direct' measures "
            "the Layer 1 classes, which never dispatch to scikit-learn"
        ),
    )
    parser.add_argument(
        "--force-backend",
        choices=("mlx", "cpu"),
        default=None,
        help=(
            "override the dispatch decision (MLXLEARN_FORCE_BACKEND) to isolate a kernel; "
            "the 'published' profile forces 'mlx'"
        ),
    )
    parser.add_argument(
        "--environment-only",
        action="store_true",
        help="record hardware and versions without running anything",
    )
    args = parser.parse_args(argv)

    env = environment()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.environment_only:
        args.output.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(env, indent=2))
        return 0

    from mlxlearn._common.config import get_tuning

    settings = PROFILE_SETTINGS.get(args.profile, {})
    layer = args.layer or settings.get("layer", "patched")
    # MLXLEARN_FORCE_BACKEND forces exactly as the flag does, so it is recorded as
    # one: otherwise the gate would judge a forced kernel run as Tuning's decision.
    force_backend = (
        args.force_backend or settings.get("force_backend") or get_tuning().force_backend or None
    )
    iterations = args.iterations or settings.get("iterations", 7)

    results = run_profile(
        args.profile,
        seed=args.seed,
        iterations=iterations,
        layer=layer,
        force_backend=force_backend,
    )
    payload = {
        "schema_version": 2,
        "profile": args.profile,
        "layer": layer,
        "force_backend": force_backend,
        "iterations": iterations,
        "environment": env,
        "results": [r.as_dict() for r in results],
        "crossovers": find_crossovers(results),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")

    if args.report:
        write_report(args.report, results, env, args.profile, layer, force_backend)
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
