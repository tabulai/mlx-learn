"""Benchmark runner: measures crossovers, writes results and a report.

Usage::

    python -m benchmarks.run --profile smoke
    python -m benchmarks.run --profile full --output bench/results.json --report bench/report.md
    python -m benchmarks.run --environment-only --output bench/environment.json

Size grids are **per algorithm**, not universal. Exact KNN over embeddings is
interesting into the millions of rows; exact SVC is quadratic in the number of
samples and is interesting at 10⁴–10⁵. Running both on one grid would measure the
wrong thing twice.

The output feeds two consumers: ``docs/benchmarks.md``, which humans read, and the
crossover thresholds in ``mlxlearn._common.config.Tuning``, which dispatch reads.
That second consumer is the point — a published crossover that dispatch ignores is
a claim, not a mechanism.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._harness import Comparison, compare, environment

__all__ = ["main", "PROFILES"]


@dataclass(frozen=True, slots=True)
class Case:
    algorithm: str
    operation: str
    n_samples: int
    n_features: int
    n_queries: int = 0
    extra: dict = None  # type: ignore[assignment]

    @property
    def label(self) -> str:
        base = f"{self.algorithm}/{self.operation}/{self.n_samples}x{self.n_features}"
        return f"{base}/q{self.n_queries}" if self.n_queries else base


def _knn_grid(sizes, features, queries) -> list[Case]:
    return [
        Case("knn", op, n, d, q)
        for n in sizes
        for d in features
        for q in queries
        for op in ("fit", "kneighbors")
    ]


PROFILES: dict[str, list[Case]] = {
    # Fast enough to run on a laptop while iterating; enough points to see a crossover.
    "smoke": [
        *_knn_grid([1_000, 8_000, 50_000], [32], [256]),
        Case("logreg", "fit", 5_000, 32),
        Case("logreg", "fit", 50_000, 64),
        Case("svc", "fit", 1_000, 32),
        Case("svc", "fit", 4_000, 32),
    ],
    # The published grid. Hours, on fixed hardware.
    "full": [
        *_knn_grid(
            [500, 2_000, 8_000, 32_000, 128_000, 512_000],
            [8, 64, 768],
            [1, 256, 4_096],
        ),
        *[Case("logreg", "fit", n, d) for n in (1_000, 10_000, 100_000, 500_000) for d in (16, 128, 1_024)],
        *[Case("svc", "fit", n, d) for n in (500, 2_000, 8_000, 32_000) for d in (16, 128)],
        *[Case("svc", "predict", n, d) for n in (2_000, 8_000) for d in (16, 128)],
    ],
}


def _make_data(case: Case, rng: np.random.Generator):
    X = rng.normal(size=(case.n_samples, case.n_features))
    if case.algorithm == "knn":
        return X, None, rng.normal(size=(case.n_queries, case.n_features))
    y = (X[:, 0] + rng.normal(scale=0.5, size=case.n_samples) > 0).astype(int)
    return X, y, X[: min(1_000, case.n_samples)]


def _runner(
    case: Case, rng: np.random.Generator, layer: str
) -> tuple[Callable, Callable, tuple[int, ...]]:
    """Build the paired (mlxlearn, scikit-learn) callables for one case.

    ``layer="patched"`` is the default and is what the performance gate is about:
    the refactoring plan's claim is that *patched dispatch* is never significantly
    slower than stock scikit-learn, and patched dispatch hands sub-crossover
    problems to scikit-learn outright.

    ``layer="direct"`` measures the Layer 1 classes, which never dispatch to
    scikit-learn and instead fall back to their own CPU path below the crossover.
    Useful for development — it shows what the MLX and CPU paths actually cost —
    but reporting it as though it were the patched number would understate small
    problems, since it measures a path patched users never execute.
    """
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

    X, y, Q = _make_data(case, rng)
    shape = (case.n_samples, case.n_features)

    if case.algorithm == "knn":
        if case.operation == "fit":
            return (
                lambda: mlx_neighbors.NearestNeighbors(n_neighbors=10).fit(X),
                lambda: sklearn.neighbors.NearestNeighbors(n_neighbors=10, algorithm="brute").fit(X),
                shape,
            )
        mlx_model = mlx_neighbors.NearestNeighbors(n_neighbors=10).fit(X)
        sk_model = sklearn.neighbors.NearestNeighbors(n_neighbors=10, algorithm="brute").fit(X)
        return (lambda: mlx_model.kneighbors(Q), lambda: sk_model.kneighbors(Q), shape)

    if case.algorithm == "logreg":
        return (
            lambda: mlx_linear.LogisticRegression(max_iter=200).fit(X, y),
            lambda: sklearn.linear_model.LogisticRegression(max_iter=200).fit(X, y),
            shape,
        )

    if case.algorithm == "svc":
        if case.operation == "fit":
            return (
                lambda: mlx_svm.SVC(kernel="rbf").fit(X, y),
                lambda: sklearn.svm.SVC(kernel="rbf").fit(X, y),
                shape,
            )
        mlx_model = mlx_svm.SVC(kernel="rbf").fit(X, y)
        sk_model = sklearn.svm.SVC(kernel="rbf").fit(X, y)
        return (lambda: mlx_model.predict(Q), lambda: sk_model.predict(Q), shape)

    raise ValueError(f"unknown algorithm {case.algorithm!r}")


def run_profile(
    profile: str, *, seed: int = 0, iterations: int = 7, layer: str = "patched"
) -> list[Comparison]:
    cases = PROFILES[profile]
    rng = np.random.default_rng(seed)
    results: list[Comparison] = []

    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case.label}", flush=True)
        try:
            mlx_fn, sk_fn, shape = _runner(case, rng, layer)
        except ImportError as exc:
            # An estimator that has not landed yet is skipped loudly. A silently
            # shorter benchmark report reads as "we measured everything".
            print(f"    skipped: {exc}", flush=True)
            continue
        results.append(
            compare(
                mlx_fn,
                sk_fn,
                label=case.label,
                shape=shape,
                iterations=iterations,
                algorithm=case.algorithm,
                operation=case.operation,
                n_queries=case.n_queries,
                layer=layer,
            )
        )
        last = results[-1]
        marker = "" if last.significant else "  (within noise)"
        print(f"    {last.speedup:.2f}x{marker}", flush=True)

    return results


def find_crossovers(results: list[Comparison]) -> dict[str, dict]:
    """The smallest measured size at which mlxlearn significantly wins.

    Reported per (algorithm, operation), never averaged across them. An average
    over fit and predict describes a workload nobody runs.
    """
    crossovers: dict[str, dict] = {}
    for result in sorted(results, key=lambda r: r.shape[0]):
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
    path: Path, results: list[Comparison], env: dict, profile: str, layer: str = "patched"
) -> None:
    crossovers = find_crossovers(results)
    lines = [
        "# Benchmark report",
        "",
        f"Profile: `{profile}`, layer: `{layer}`. Generated by `python -m benchmarks.run`.",
        "",
        "`patched` measures the classes `patch_sklearn()` installs — which is what the "
        "performance gate is about, since patched dispatch hands sub-crossover problems to "
        "scikit-learn outright. `direct` measures the Layer 1 classes, which never dispatch "
        "to scikit-learn and use their own CPU path instead.",
        "",
        "Timings are `mx.eval`-synchronized, warmed up, and reported as the median of "
        f"{results[0].mlxlearn.iterations if results else 0} interleaved iterations with the "
        "interquartile range. A difference smaller than the combined spread is reported as "
        "**within noise** and does not count as a win.",
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
        "algorithm and operation. These feed `mlxlearn._common.config.Tuning`; below them, "
        "dispatch hands the work to scikit-learn on purpose.",
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
        "| case | n_samples | n_features | mlxlearn (ms) | sklearn (ms) | speedup (median) | speedup (best) | significant |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            f"| `{result.label}` | {result.shape[0]} | {result.shape[1]} | "
            f"{result.mlxlearn.median_s * 1e3:.2f} | {result.sklearn.median_s * 1e3:.2f} | "
            f"{result.speedup:.2f}x | {result.speedup_best:.2f}x | "
            f"{'yes' if result.significant else 'no'} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--output", type=Path, default=Path("bench/results.json"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layer",
        choices=("patched", "direct"),
        default="patched",
        help=(
            "'patched' (default) measures what patch_sklearn() users get, which is what "
            "the performance gate claims; 'direct' measures the Layer 1 classes, which "
            "never dispatch to scikit-learn"
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

    results = run_profile(
        args.profile, seed=args.seed, iterations=args.iterations, layer=args.layer
    )
    payload = {
        "schema_version": 1,
        "profile": args.profile,
        "layer": args.layer,
        "environment": env,
        "results": [r.as_dict() for r in results],
        "crossovers": find_crossovers(results),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")

    if args.report:
        write_report(args.report, results, env, args.profile, args.layer)
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
