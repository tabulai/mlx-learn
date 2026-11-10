"""Measure where PCA's GPU routes overtake scikit-learn, one solver at a time.

The thresholds in ``mlxlearn._common.config.Tuning`` (``pca_covariance_min_work``,
``pca_covariance_min_features``, ``pca_randomized_min_work``) and the decision to
keep ``full`` on the host come from these panels. Every configuration times three
routes on the same array, rotating their order between rounds and reporting the
median: stock scikit-learn, mlxlearn forced onto the GPU, and mlxlearn forced onto
its host path.

Usage::

    python -m benchmarks.pca_crossover covariance OUT.jsonl
    python -m benchmarks.pca_crossover randomized-small OUT.jsonl
    python -m benchmarks.pca_crossover randomized-boundary OUT.jsonl
    python -m benchmarks.pca_crossover full OUT.jsonl
    python -m benchmarks.pca_crossover summary OUT.jsonl [MORE.jsonl ...]

A crossover is only as good as its worst case, so each panel varies what a caller
can vary without changing the shape: dtype, whether the mean dominates the spread,
whether the spectrum is flat or low-rank with a noise floor, ``n_components``,
``iterated_power``, and ``fit`` against ``fit_transform``.
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import statistics
import time
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

ROUTES = ("sklearn", "gpu", "host")


def _low_rank(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A rank ``min(n, d) / 8`` signal over a noise floor 1e-3 as large, off-center."""
    n, d = base.shape
    rank = max(2, min(n, d) // 8)
    signal = (base[:, :rank] * np.linspace(10.0, 1.0, rank)) @ rng.standard_normal((rank, d))
    return signal + 0.01 * base + 5.0


def _datasets(
    base: np.ndarray, kinds: tuple[str, ...], rng: np.random.Generator
) -> Iterator[tuple[str, np.ndarray]]:
    for kind in kinds:
        if kind == "iso":
            yield kind, base
        elif kind == "shift":
            yield kind, base + 100.0
        else:
            yield kind, _low_rank(base, rng)


def _fit(
    cls: type,
    *,
    X: np.ndarray,
    k: int,
    solver: str,
    operation: str,
    iterated_power: int | str,
) -> object:
    model = cls(n_components=k, svd_solver=solver, iterated_power=iterated_power, random_state=0)
    return model.fit(X) if operation == "fit" else model.fit_transform(X)


def _time_routes(fit: Callable[[type], object], repeats: int) -> dict[str, float]:
    """Median milliseconds per route; ``fit`` runs one configuration with a given class."""
    from sklearn.decomposition import PCA as SkPCA

    from mlxlearn._common.config import tuning_context
    from mlxlearn.decomposition import PCA

    run_sklearn = functools.partial(fit, SkPCA)
    run_mlx = functools.partial(fit, PCA)

    def forced(backend: str) -> Callable[[], None]:
        def call() -> None:
            with tuning_context(force_backend=backend):
                run_mlx()

        return call

    calls = {"sklearn": run_sklearn, "gpu": forced("mlx"), "host": forced("cpu")}
    for call in calls.values():
        call()
        call()
    samples: dict[str, list[float]] = {name: [] for name in ROUTES}
    for index in range(repeats):
        shift = index % len(ROUTES)
        for name in ROUTES[shift:] + ROUTES[:shift]:
            start = time.perf_counter()
            calls[name]()
            samples[name].append(time.perf_counter() - start)
    return {f"{name}_ms": round(statistics.median(values) * 1e3, 4) for name, values in samples.items()}


def _cases(panel: str) -> Iterator[dict]:
    """Shapes and hyperparameters of one panel."""
    if panel == "covariance":
        works = (250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000, 32_000_000)
        for d in (8, 16, 32, 64, 128, 256, 512, 1000):
            for work in works:
                n = work // d
                if n >= 10 * d:  # where svd_solver="auto" picks covariance_eigh
                    yield {"solver": "covariance_eigh", "n": n, "d": d, "k": min(16, d),
                           "data": ("iso", "shift"), "runs": (("fit", "auto"), ("fit_transform", "auto"))}
    elif panel == "randomized-small":
        for n in (300, 600, 1000, 2000, 4000, 8000, 16000):
            for d in (32, 64, 128, 256, 512, 1024, 2048, 4096):
                if n * d > 40_000_000:
                    continue
                for k in (4, 32):
                    if k < min(n, d):
                        yield {"solver": "randomized", "n": n, "d": d, "k": k,
                               "data": ("iso", "lowrank"),
                               "runs": (("fit", "auto"), ("fit_transform", "auto"))}
    elif panel == "randomized-boundary":
        for work in (8_000_000, 16_000_000):
            for d in (32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 32768):
                n = work // d
                if n < 250:
                    continue
                for k in (4, 32, 128):
                    if k < min(n, d):
                        yield {"solver": "randomized", "n": n, "d": d, "k": k,
                               "data": ("iso", "lowrank"),
                               "runs": (("fit", "auto"), ("fit_transform", "auto"),
                                        ("fit", 0), ("fit", 2))}
    elif panel == "full":
        shapes = ((200, 200), (500, 100), (100, 500), (500, 500), (300, 2000), (2000, 300),
                  (1000, 1000), (4000, 200), (200, 4000), (1500, 1500), (3000, 600))
        for n, d in shapes:
            yield {"solver": "full", "n": n, "d": d, "k": min(n, d) // 2,
                   "data": ("iso", "lowrank"), "runs": (("fit", "auto"), ("fit_transform", "auto"))}
    else:
        raise SystemExit(f"unknown panel {panel!r}")


def run_panel(panel: str, output: Path, repeats: int) -> None:
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    with output.open("w") as sink:
        for case in _cases(panel):
            n, d, k, solver = case["n"], case["d"], case["k"], case["solver"]
            base = rng.standard_normal((n, d))
            for kind, values in _datasets(base, case["data"], rng):
                for dtype in (np.float32, np.float64):
                    X = np.ascontiguousarray(values.astype(dtype))
                    for operation, iterated_power in case["runs"]:
                        fit = functools.partial(_fit, X=X, k=k, solver=solver, operation=operation,
                                                iterated_power=iterated_power)
                        record = {"panel": panel, "solver": solver, "dtype": np.dtype(dtype).name,
                                  "n": n, "d": d, "k": k, "data": kind, "operation": operation,
                                  "iterated_power": str(iterated_power)}
                        try:
                            record.update(_time_routes(fit, repeats))
                        except Exception as exc:  # a failing configuration is a result too
                            record["error"] = repr(exc)[:200]
                        sink.write(json.dumps(record) + "\n")
                        sink.flush()
            print(f"{panel}: {n} x {d} k={k} done", flush=True)


def _grid(rows: list[dict], route: str, row_key: str, column_key: str) -> None:
    worst: dict[tuple[int, int], float] = collections.defaultdict(lambda: float("inf"))
    for row in rows:
        key = (row[row_key], row[column_key])
        worst[key] = min(worst[key], row["sklearn_ms"] / row[f"{route}_ms"])
    columns = sorted({column for _, column in worst})
    print(f"      {row_key:>5} | " + " ".join(f"{column:>9}" for column in columns))
    for name in sorted({name for name, _ in worst}):
        cells = (f"{worst[(name, column)]:9.2f}" if (name, column) in worst else "        ."
                 for column in columns)
        print(f"      {name:>5} | " + " ".join(cells))


def summarize(paths: list[Path]) -> None:
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    failed = [row for row in rows if "error" in row]
    rows = [row for row in rows if "error" not in row]
    for row in rows:
        row["work_k"] = round(row["n"] * row["d"] / 250_000) * 250
    print(f"{len(rows)} configurations, {len(failed)} failed")
    for panel in sorted({row["panel"] for row in rows}):
        for route in ("gpu", "host"):
            for dtype in ("float32", "float64"):
                subset = [row for row in rows if row["panel"] == panel and row["dtype"] == dtype]
                if not subset:
                    continue
                print(f"\n{panel}, {dtype}: scikit-learn time / mlxlearn[{route}] time, "
                      "worst case over data, n_components, iterated_power and operation")
                if panel == "covariance":
                    print("    rows: features; columns: thousands of elements")
                    _grid(subset, route, "d", "work_k")
                elif panel == "randomized-small":
                    print("    rows: samples; columns: features")
                    _grid(subset, route, "n", "d")
                else:
                    for row in sorted({(r["n"], r["d"]) for r in subset}, key=lambda t: (t[0] * t[1], t[1])):
                        values = [r["sklearn_ms"] / r[f"{route}_ms"] for r in subset
                                  if (r["n"], r["d"]) == row]
                        by_data = {
                            kind: min(r["sklearn_ms"] / r[f"{route}_ms"] for r in subset
                                      if (r["n"], r["d"]) == row and r["data"] == kind)
                            for kind in sorted({r["data"] for r in subset})
                        }
                        detail = ", ".join(f"{kind} {value:.2f}" for kind, value in by_data.items())
                        print(f"    {row[0]:>7} x {row[1]:<6} worst {min(values):5.2f}  "
                              f"median {statistics.median(values):5.2f}  ({detail})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("panel", choices=("covariance", "randomized-small", "randomized-boundary",
                                          "full", "summary"))
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.panel == "summary":
        summarize(args.paths)
    else:
        if len(args.paths) != 1:
            parser.error("a measurement panel writes exactly one output file")
        run_panel(args.panel, args.paths[0], args.repeats)


if __name__ == "__main__":
    main()
