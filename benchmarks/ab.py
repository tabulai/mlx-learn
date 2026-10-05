"""Release-over-release A/B: one mlxlearn source tree against another, fresh processes.

Usage::

    python benchmarks/ab.py run --baseline ../mlxlearn-0.1.0a2/src --candidate src \\
        --output bench/ab.jsonl --rounds 3 --iterations 7 nnq:100000:32:20000:10 lr:20000:16
    python benchmarks/ab.py list

This is what the CHANGELOG's **Performance** entries were measured with: the raw
results are ``benchmarks/results/ab-0.1.0a2.jsonl``. Each case runs in a fresh
process per version and round, the versions alternating first and second, so an
import-time or first-call cost, a thermal ramp or a stray background process
lands on both. Inside a process every timed call is prepared as the benchmark
harness prepares one — a garbage collection before it, the collector off, an
``mx.synchronize()`` inside the window — after two untimed warm-up calls.

Run it as a file, not with ``-m``: the child process must import mlxlearn from the
tree under test and from nothing else, so it cannot import this package.

A case is ``name:arg:arg...``; ``list`` prints them. Most time the candidate and
the baseline on the same work; the ``patched...``, ``handoff`` and ``svcmany`` cases
time the patched class against stock scikit-learn inside each process instead, and
``nnpeak``, ``nnfitpeak`` report memory rather than time.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve()

#: case name -> (arguments, what it measures)
CASES = {
    "nnq": ("n:d:q:k", "Layer 1 kneighbors on the GPU, Gaussian data"),
    "nnbin": ("n:d:q", "Layer 1 kneighbors on binary features"),
    "nnfit": ("n:d:device[:dtype]", "Layer 1 NearestNeighbors fit on a device"),
    "nnpeak": ("n:d:q:k", "peak device memory of one float32 query, MiB"),
    "nnfitpeak": ("n:d", "device peak and resident memory of a float64 fit, over float32 X"),
    "nnoffset": ("n:q:offset", "kneighbors on 2-feature data a few units apart around offset"),
    "cert": ("kind:d", "docs/fp32_policy.md §2.1 data, 50 000 rows, 4096 queries"),
    "cacherel": ("n:d:q:fraction", "a repeated query with mlx_cache_release_fraction set"),
    "range": ("n:d", "the float32 range check on float64 input, and its peak memory"),
    "unpatch": ("", "unpatch_sklearn() after patch_sklearn()"),
    "radius": ("n:d:q", "radius_neighbors on an MLX-fitted model, float64 query"),
    "knnpredict": ("n:d:q:classes", "Layer 1 KNeighborsClassifier.predict"),
    "patchedknn": ("n:d:q", "patched KNeighborsClassifier fit+predict against stock"),
    "handoff": ("knn|logreg|svc", "a patched fit below the crossover against stock, 200 x 5"),
    "lr": ("n:d", "forced-MLX LogisticRegression fit, the benchmark's binary data"),
    "svcfit": ("n:d:kernel:C[:shrinking]", "forced-MLX SVC fit, the benchmark's binary data"),
    "svcpredict": ("n:d:rows", "forced-MLX SVC predict"),
    "svcmany": ("n:d:K", "patched many-class SVC fit against stock, separated blobs"),
}


# ------------------------------------------------------------------------------ child


def _child(case: str, iterations: int) -> dict:  # noqa: C901 - one branch per case
    import mlx.core as mx
    import numpy as np

    from mlxlearn._common.config import tuning_context

    warnings.simplefilter("ignore")

    def timed(fn):
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter()
            fn()
            mx.synchronize()
            return time.perf_counter() - start
        finally:
            gc.enable()

    def run(fn, n=iterations, warm=2):
        for _ in range(warm):
            fn()
        mx.synchronize()
        return [timed(fn) for _ in range(n)]

    def paired(fm, fs, n):
        """Patched against stock in one process, order alternating."""
        m_s, s_s = [], []
        for i in range(n):
            if i % 2:
                m_s.append(timed(fm))
                s_s.append(timed(fs))
            else:
                s_s.append(timed(fs))
                m_s.append(timed(fm))
        return m_s, s_s

    def gauss(n, d, seed=0):
        return np.random.default_rng(seed).normal(size=(n, d))

    def binary(n, d):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, d))
        return X, (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)

    forced = lambda **kw: tuning_context(force_backend="mlx", **kw)  # noqa: E731
    name, *args = case.split(":")

    if name in ("nnq", "nnbin", "nnoffset", "cert", "cacherel", "radius", "nnpeak"):
        from mlxlearn.neighbors import NearestNeighbors

        k, extra_tuning = 10, {}
        if name == "nnq":
            n, d, q, k = map(int, args)
            X, Q = gauss(n, d), gauss(q, d, 1)
        elif name == "nnpeak":
            n, d, q, k = map(int, args)
            X, Q = gauss(n, d).astype(np.float32), gauss(q, d, 1).astype(np.float32)
        elif name == "nnbin":
            n, d, q = map(int, args)
            rng = np.random.default_rng(0)
            X = (rng.random((n, d)) < 0.5).astype(np.float64)
            Q = (rng.random((q, d)) < 0.5).astype(np.float64)
        elif name == "nnoffset":
            n, q, offset = int(args[0]), int(args[1]), float(args[2])
            rng = np.random.default_rng(0)
            X = offset + rng.normal(scale=5.0, size=(n, 2))
            Q = offset + rng.normal(scale=5.0, size=(q, 2))
        elif name == "cacherel":
            n, d, q, fraction = int(args[0]), int(args[1]), int(args[2]), float(args[3])
            X, Q = gauss(n, d), gauss(q, d, 1)
            extra_tuning = {"mlx_cache_release_fraction": fraction}
        elif name == "radius":
            n, d, q = map(int, args)
            X, Q = gauss(n, d), gauss(q, d, 1)
        else:
            X, Q = _certificate_data(args[0], int(args[1]))
        with forced(**extra_tuning):
            model = NearestNeighbors(n_neighbors=k).fit(X)
            if name == "nnpeak":
                model.kneighbors(Q[:8])
                mx.synchronize()
                mx.clear_cache()
                mx.reset_peak_memory()
                base = mx.get_active_memory()
                model.kneighbors(Q)
                mx.synchronize()
                return {"samples": None, "peak_mib": (mx.get_peak_memory() - base) / 2**20}
            if name == "radius":
                gaps = np.linalg.norm(X[:2000, None, :] - Q[None, :20, :], axis=2)
                r = float(np.sort(gaps, axis=0)[10].mean())
                return {"samples": run(lambda: model.radius_neighbors(Q, radius=r))}
            return {"samples": run(lambda: model.kneighbors(Q))}
    if name == "nnfit":
        import mlxlearn
        from mlxlearn.neighbors import NearestNeighbors

        n, d, device = int(args[0]), int(args[1]), args[2]
        X = gauss(n, d).astype(args[3] if len(args) > 3 else "float64")
        mlxlearn.set_config(device=device)
        with forced():
            return {"samples": run(lambda: NearestNeighbors(n_neighbors=10).fit(X))}
    if name == "nnfitpeak":
        from mlxlearn.neighbors import NearestNeighbors

        n, d = map(int, args)
        X = gauss(n, d)
        with forced():
            NearestNeighbors(n_neighbors=10).fit(X[:64])
            mx.synchronize()
            mx.clear_cache()
            mx.reset_peak_memory()
            base = mx.get_active_memory()
            model = NearestNeighbors(n_neighbors=10).fit(X)  # noqa: F841 - held for the reading
            mx.synchronize()
        f32 = n * d * 4
        return {"samples": None,
                "peak_over_f32": (mx.get_peak_memory() - base) / f32,
                "resident_over_f32": (mx.get_active_memory() - base) / f32}
    if name == "range":
        import tracemalloc

        from mlxlearn._common.validation import check_float32_range

        n, d = map(int, args)
        X = gauss(n, d)
        samples = run(lambda: check_float32_range(X))
        tracemalloc.start()
        check_float32_range(X)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return {"samples": samples, "peak_mib": peak / 2**20}
    if name == "unpatch":
        import mlxlearn

        mlxlearn.patch_sklearn()
        start = time.perf_counter()
        mlxlearn.unpatch_sklearn()
        return {"samples": [time.perf_counter() - start]}
    if name == "knnpredict":
        from mlxlearn.neighbors import KNeighborsClassifier

        n, d, q, classes = map(int, args)
        X, Q = gauss(n, d), gauss(q, d, 1)
        y = np.random.default_rng(2).integers(0, classes, size=n)
        with forced():
            model = KNeighborsClassifier(n_neighbors=10).fit(X, y)
        return {"samples": run(lambda: model.predict(Q))}
    if name in ("patchedknn", "handoff", "svcmany"):
        import sklearn.linear_model
        import sklearn.neighbors
        import sklearn.svm

        from mlxlearn.patching import _estimators as patched

        if name == "patchedknn":
            n, d, q = map(int, args)
            X, Q = gauss(n, d), gauss(q, d, 1)
            y = (X[:, 0] > 0).astype(int)
            fm = lambda: patched.KNeighborsClassifier().fit(X, y).predict(Q)  # noqa: E731
            fs = lambda: sklearn.neighbors.KNeighborsClassifier().fit(X, y).predict(Q)  # noqa: E731
            backend = patched.KNeighborsClassifier().fit(X, y).execution_backend_
            warm, count = 2, iterations
        elif name == "handoff":
            X = gauss(200, 5)
            y = (X[:, 0] > 0).astype(int)
            est = {
                "knn": (patched.KNeighborsClassifier, sklearn.neighbors.KNeighborsClassifier),
                "logreg": (patched.LogisticRegression, sklearn.linear_model.LogisticRegression),
                "svc": (patched.SVC, sklearn.svm.SVC),
            }[args[0]]
            fm, fs = (lambda: est[0]().fit(X, y)), (lambda: est[1]().fit(X, y))
            backend = fm().execution_backend_
            warm, count = 3, max(iterations, 41)
        else:
            n, d, K = map(int, args)
            rng = np.random.default_rng(0)
            centers = rng.normal(scale=2.0, size=(K, d))
            y = rng.integers(0, K, size=n)
            X = centers[y] + rng.normal(size=(n, d))
            fm, fs = (lambda: patched.SVC().fit(X, y)), (lambda: sklearn.svm.SVC().fit(X, y))
            backend = fm().execution_backend_
            warm, count = 1, iterations
        for _ in range(warm):
            fm()
            fs()
        mlx_samples, stock = paired(fm, fs, count)
        return {"samples": mlx_samples, "stock": stock, "backend": backend}
    if name == "lr":
        from mlxlearn.linear_model import LogisticRegression

        X, y = binary(*map(int, args))
        seen = {}

        def fit():
            with forced():
                model = LogisticRegression(max_iter=200).fit(X, y)
            seen["n_iter"] = int(np.max(model.n_iter_))
            return model

        return {"samples": run(fit), "n_iter": seen["n_iter"]}
    if name in ("svcfit", "svcpredict"):
        from mlxlearn.svm import SVC

        if name == "svcpredict":
            n, d, rows = map(int, args)
            X, y = binary(n, d)
            with forced():
                model = SVC().fit(X, y)
            Q = X[:rows].copy()
            return {"samples": run(lambda: model.predict(Q), n=max(iterations, 21))}
        n, d, kernel, C = int(args[0]), int(args[1]), args[2], float(args[3])
        kw = {} if len(args) < 5 else {"shrinking": args[4] == "1"}
        X, y = binary(n, d)

        def fit():
            with forced():
                return SVC(kernel=kernel, C=C, **kw).fit(X, y)

        return {"samples": run(fit, warm=1)}
    raise SystemExit(f"unknown case {case!r}; see `python {HERE.name} list`")


def _certificate_data(kind: str, d: int):
    """The data of ``docs/fp32_policy.md`` §2.1: 50 000 rows, 4096 queries (2048 for two
    clusters)."""
    import numpy as np

    rng = np.random.default_rng(0)
    n, q = 50_000, 4096
    if kind == "clusters1000":  # 1000 clusters of spread 1e-3
        centers = rng.normal(size=(1000, d))
        X = centers[rng.integers(0, 1000, n)] + 1e-3 * rng.normal(size=(n, d))
        Q = centers[rng.integers(0, 1000, q)] + 1e-3 * rng.normal(size=(q, d))
        return X, Q
    if kind == "blobs":  # make_blobs(cluster_std=0.01)
        from sklearn.datasets import make_blobs

        X, _ = make_blobs(n_samples=n + q, n_features=d, cluster_std=0.01, random_state=0)
        return X[:n], X[n:]
    if kind == "column":  # one column spanning 0-1e4 among unit-scale ones
        X = rng.normal(size=(n + q, d))
        X[:, 0] = rng.uniform(0, 1e4, size=n + q)
        return X[:n], X[n:]
    if kind == "twoclusters":  # two clusters at 0 and 1000
        q = 2048
        X = rng.normal(size=(n + q, d)) + 1000.0 * (rng.random((n + q, 1)) < 0.5)
        return X[:n], X[n:]
    if kind == "gauss":
        return rng.normal(size=(n, d)), rng.normal(size=(q, d))
    if kind == "dup20":  # every row repeated 20 times; queries near the rows
        base = rng.normal(size=(n // 20, d))
        X = np.repeat(base, 20, axis=0)
        return X, base[rng.integers(0, n // 20, q)] + 1e-3 * rng.normal(size=(q, d))
    raise SystemExit(f"unknown certificate data {kind!r}")


# ----------------------------------------------------------------------------- driver


def _summary(samples: list[float]) -> str:
    return f"{statistics.median(samples) * 1e3:9.2f} ms (min {min(samples) * 1e3:8.2f}, n={len(samples)})"


def drive(cases, *, versions: dict[str, Path], output: Path, rounds: int, iterations: int):
    """Time each case in fresh processes, the versions alternating first; append JSON lines."""
    output.parent.mkdir(parents=True, exist_ok=True)
    for raw in cases:
        # "candidate-only/<case>" times one tree, for an A/B inside it (shrinking on
        # against off, a knob set two ways) rather than across releases.
        case = raw.removeprefix("candidate-only/")
        names = ["candidate"] if case != raw else list(versions)
        results = {name: {"samples": [], "extra": []} for name in names}
        for r in range(rounds):
            order = names if r % 2 == 0 else names[::-1]
            for name in order:
                env = dict(os.environ, PYTHONPATH=str(versions[name]))
                proc = subprocess.run(
                    [sys.executable, str(HERE), "child", case, str(iterations)],
                    env=env, capture_output=True, text=True,
                )
                if proc.returncode:
                    results[name]["error"] = proc.stderr[-800:]
                    print(f"{case} {name} FAILED\n{proc.stderr[-800:]}", flush=True)
                    continue
                record = json.loads(proc.stdout.strip().splitlines()[-1])
                results[name]["samples"] += record.pop("samples") or []
                results[name]["extra"].append(record)
        with output.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"case": case, "time": time.time(), "res": results}) + "\n")
        parts = []
        for name in names:
            samples = results[name]["samples"]
            parts.append(f"{name} {_summary(samples)}" if samples else f"{name} {results[name]['extra']}")
        if all(results[n]["samples"] for n in names) and len(names) == 2:
            a, b = (statistics.median(results[n]["samples"]) for n in names)
            parts.append(f"baseline / candidate = {a / b:.2f}x")
        print(f"{case:32s} " + " | ".join(parts), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the cases")
    child = sub.add_parser("child", help="one case in this process (internal)")
    child.add_argument("case")
    child.add_argument("iterations", type=int)
    runner = sub.add_parser("run", help="A/B the cases in fresh processes")
    runner.add_argument("cases", nargs="+")
    runner.add_argument("--baseline", type=Path, required=True,
                        help="the src/ directory of the release to compare against")
    runner.add_argument("--candidate", type=Path, default=HERE.parents[1] / "src")
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--rounds", type=int, default=3)
    runner.add_argument("--iterations", type=int, default=7)
    args = parser.parse_args(argv)

    if args.command == "list":
        for name, (arguments, what) in CASES.items():
            spelled = f"{name}:{arguments}" if arguments else name
            print(f"{spelled:36s} {what}")
        return 0
    if args.command == "child":
        print(json.dumps(_child(args.case, args.iterations)))
        return 0
    drive(
        args.cases,
        versions={"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()},
        output=args.output, rounds=args.rounds, iterations=args.iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
