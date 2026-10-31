"""Audited, isolated before/after KNN benchmarks with interleaved samples.

    python -m benchmarks.knn_pipeline --profile smoke --resident \
        --output benchmarks/results/knn-pipeline-2026-10-28.json

The immutable reference runs in its own persistent Python process. Every variant
reads identical serialized float32 inputs. Timed calls are synchronized; numerical
audits run after the timer, on every call. No neighbor-result cache is used.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "benchmarks/reference/knn-2026-10-28.zip"


@dataclass(frozen=True)
class Case:
    name: str
    n_train: int
    n_query: int
    n_features: int
    k: int = 5


CASES = (
    Case("single_8_k5", 8192, 1, 8),
    Case("batch_8_k5", 8192, 256, 8),
    Case("batch_32_k5", 8192, 256, 32),
    Case("batch_32_k32", 8192, 128, 32, 32),
    Case("batch_128_k5", 4096, 128, 128),
    Case("batch_1024_k5", 2048, 32, 1024),
    Case("batch_32_k64", 4096, 128, 32, 64),
    Case("batch_16_k5", 8192, 256, 16),
    Case("single_32_k5", 8192, 1, 32),
    Case("batch_8_k32", 8192, 128, 8, 32),
    Case("batch_50000_32_k5", 50_000, 256, 32),
)
SMOKE = {"single_8_k5", "batch_32_k5", "batch_32_k64"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_hashes(source_root):
    return {
        str(p.relative_to(source_root)): sha256(p.read_bytes())
        for package in ("mlxlearn", "mlxlearn_device")
        for p in sorted((source_root / package).rglob("*"))
        if p.is_file() and p.suffix in {".py", ".typed", ".metal", ".h", ".hpp", ".cpp", ".mm"}
    }


def extract_reference(archive, destination):
    """Verify exact inventory, paths, and hashes before writing source entries."""
    manifest = json.loads(archive.with_suffix(".manifest.json").read_text())
    if sha256(archive.read_bytes()) != manifest["archive"]["sha256"]:
        raise ValueError("Reference archive hash differs from manifest")
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(set(names)) != len(names) or set(names) != set(manifest["files"]):
            raise ValueError("Reference archive inventory differs from manifest")
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] not in {
                "mlxlearn", "mlxlearn_device"
            }:
                raise ValueError(f"Unsafe reference path: {name}")
            mode = source.getinfo(name).external_attr >> 16
            if mode & 0o170000 not in (0, 0o100000):
                raise ValueError(f"Non-regular reference entry: {name}")
            data = source.read(name)
            if sha256(data) != manifest["files"][name]:
                raise ValueError(f"Reference source hash mismatch: {name}")
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return manifest


def make_fixture(case, seed, destination):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(case.n_train, case.n_features)).astype(np.float32)
    Q = rng.normal(size=(case.n_query, case.n_features)).astype(np.float32)
    # Coincident rows exercise zero-distance weighting without creating ties.
    if case.n_query > 1:
        Q[:2] = X[:2]
    y = rng.integers(0, 4, size=case.n_train)
    targets = rng.normal(size=case.n_train)
    with threadpool_limits(limits=1):
        reference = NearestNeighbors(n_neighbors=case.k, algorithm="brute").fit(
            X.astype(np.float64)
        )
        distances, indices = reference.kneighbors(Q.astype(np.float64))
    # A copied point's exact Euclidean distance is zero; BLAS Gram evaluation
    # can leave a tiny residual even in the float64 reference.
    distances = np.sqrt(np.sum(
        (X[indices].astype(np.float64) - Q[:, None].astype(np.float64)) ** 2, axis=2
    ))
    order = np.lexsort((indices, distances), axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    distances = np.take_along_axis(distances, order, axis=1)
    expected = {}
    for weighting in ("uniform", "distance"):
        if weighting == "uniform":
            weights = np.ones_like(distances)
        else:
            zeros = distances == 0
            with np.errstate(divide="ignore"):
                weights = np.where(np.any(zeros, axis=1, keepdims=True), zeros, 1 / distances)
        scores = np.stack([
            np.sum(weights * (y[indices] == c), axis=1) for c in range(4)
        ], axis=1)
        expected[f"proba_{weighting}"] = scores / scores.sum(axis=1, keepdims=True)
        expected[f"predict_{weighting}"] = scores.argmax(axis=1)
        expected[f"regress_{weighting}"] = (
            np.sum(weights * targets[indices], axis=1) / weights.sum(axis=1)
        )
    path = destination / f"{case.name}.npz"
    np.savez(path, X=X, Q=Q, y=y, targets=targets, distances=distances, indices=indices, **expected)
    return {"case": asdict(case), "sha256": sha256(path.read_bytes())}


class Worker:
    def __init__(self, variant, source_root, fixtures):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(source_root)
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", variant,
             "--source-root", str(source_root), "--fixtures", str(fixtures)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env,
        )
        self.info = self.request({"op": "info"})

    def request(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Benchmark worker exited: {self.process.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)


def worker_main(args):
    import mlx.core as mx
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    variant = args.worker
    device = variant == "resident"
    is_sklearn = variant.startswith("sklearn")
    if is_sklearn:
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
        tuning = contextlib.nullcontext
    elif device:
        from mlxlearn_device import fit_neighbors
        tuning = contextlib.nullcontext
    else:
        import mlxlearn
        from mlxlearn._common.config import tuning_context
        from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
        imported = Path(mlxlearn.__file__).resolve()
        if not imported.is_relative_to(Path(args.source_root).resolve()):
            raise RuntimeError(f"Wrong package imported: {imported}")
        def tuning():
            return tuning_context(force_backend="mlx")
    policy = threadpool_limits(limits=1) if variant == "sklearn_1" else contextlib.nullcontext()
    cache = {}

    def sync(value):
        if device:
            arrays = value if isinstance(value, tuple) else (value,)
            mx.eval(*(a for a in arrays if isinstance(a, mx.array)))
        mx.synchronize()

    def timed(call):
        mx.synchronize()
        start = time.perf_counter()
        result = call()
        sync(result)
        return result, (time.perf_counter() - start) * 1000

    def fixture(case):
        name = case["name"]
        if name not in cache:
            data = dict(np.load(Path(args.fixtures) / f"{name}.npz"))
            if device:
                data["Xd"], data["Qd"] = mx.array(data["X"]), mx.array(data["Q"])
                mx.eval(data["Xd"], data["Qd"])
            cache[name] = data
        return cache[name]

    def fit(data, k):
        if device:
            return fit_neighbors(data["Xd"], device="gpu")
        return NearestNeighbors(n_neighbors=k, algorithm="brute").fit(data["X"])

    def query(model, data, k, distance=True):
        if device:
            return model.query(data["Qd"], k, return_distance=distance)
        return model.kneighbors(data["Q"], return_distance=distance)

    def fit_query(data, k):
        # Include all preparation deferred from fit until its first query.
        model = fit(data, k)
        data["last_fit_query_model"] = model
        return query(model, data, k)

    def audit(result, data, operation):
        if operation in {"cold", "fit", "fit_query", "kneighbors"}:
            actual_dist, actual_idx = result
            actual_dist, actual_idx = np.asarray(actual_dist), np.asarray(actual_idx)
            # Integer equality is stricter than a distance-only audit.
            index_ok = np.array_equal(actual_idx, data["indices"])
            distance_ok = np.allclose(actual_dist, data["distances"], atol=1e-6, rtol=1e-6)
            return {"pass": bool(index_ok and distance_ok), "indices_match": bool(index_ok),
                    "distance_max_abs": float(np.max(np.abs(actual_dist - data["distances"]))) }
        if operation == "indices":
            if device and isinstance(result, tuple):
                result = result[1]
            return {"pass": bool(np.array_equal(np.asarray(result), data["indices"]))}
        expected = data[operation]
        actual = np.asarray(result)
        equal = (np.array_equal(actual, expected) if operation.startswith("predict")
                 else np.allclose(actual, expected, atol=2e-6, rtol=2e-5))
        return {"pass": bool(equal), "max_abs": float(np.max(np.abs(actual - expected)))}

    def execute(operation, case):
        data = fixture(case)
        k = case["k"]
        with tuning():
            if operation == "cold":
                model, fit_ms = timed(partial(fit, data, k))
                data["model"] = model
                value, query_ms = timed(partial(query, model, data, k))
                result = {"fit_ms": fit_ms, "query_ms": query_ms,
                          "audit": audit(value, data, operation)}
            elif operation == "fit":
                model, elapsed = timed(partial(fit, data, k))
                value = query(model, data, k)
                sync(value)
                result = {"ms": elapsed, "audit": audit(value, data, operation)}
            elif operation == "fit_query":
                value, elapsed = timed(partial(fit_query, data, k))
                model = data["last_fit_query_model"]
                result = {"ms": elapsed, "audit": audit(value, data, operation)}
            elif operation in {"kneighbors", "indices"}:
                model = data["model"]
                value, elapsed = timed(partial(
                    query, model, data, k, distance=operation != "indices"
                ))
                result = {"ms": elapsed, "audit": audit(value, data, operation)}
            else:
                kind, weighting = operation.split("_")
                model_key = ("regressor" if kind == "regress" else "classifier") + weighting
                if model_key not in data:
                    cls = KNeighborsRegressor if kind == "regress" else KNeighborsClassifier
                    target = data["targets"] if kind == "regress" else data["y"]
                    data[model_key] = cls(
                        n_neighbors=k, weights=weighting, algorithm="brute"
                    ).fit(data["X"], target)
                model = data[model_key]
                method = model.predict_proba if kind == "proba" else model.predict
                value, elapsed = timed(partial(method, data["Q"]))
                result = {"ms": elapsed, "audit": audit(value, data, operation)}
            if not is_sklearn and not device:
                result["backend"] = model.execution_backend_
                if result["backend"] != "mlx":
                    raise RuntimeError("Benchmark silently left the requested GPU backend")
            if not device:
                result["effective_fit_method"] = getattr(model, "_fit_method", None)
        return result

    with policy:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                operation = request["op"]
                if operation == "info":
                    import sklearn
                    result = {"variant": variant, "python": sys.version,
                              "mlx": mx.__version__, "numpy": np.__version__,
                              "sklearn": sklearn.__version__, "threadpools": threadpool_info(),
                              "hardware": dict(mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()),
                              "source_root": str(Path(args.source_root).resolve()),
                              "source_hashes": source_hashes(Path(args.source_root))}
                elif operation == "integrity":
                    result = source_hashes(Path(args.source_root))
                elif operation == "block":
                    result = run_block(execute, request)
                else:
                    result = execute(operation, request["case"])
                print(json.dumps(result), flush=True)
            except Exception:
                print(json.dumps({"error": traceback.format_exc()}), flush=True)


def summarize(samples):
    values = [sample["ms"] for sample in samples]
    return {"median_ms": statistics.median(values), "min_ms": min(values),
            "max_ms": max(values), "samples": samples,
            "quality_pass": all(sample["audit"]["pass"] for sample in samples)}


def run_block(execute, request):
    """Execute audited consecutive calls with count- and optional time-based warmup.

    Existing requests omit warmup_seconds and retain their fixed-count behavior.
    The new duration policy is explicit and does not change any thread settings.
    """
    repeats, warmup = request["repeats"], request["warmup"]
    warmup_seconds = float(request.get("warmup_seconds", 0))
    if (repeats < 1 or warmup < 0 or not math.isfinite(warmup_seconds)
            or warmup_seconds < 0
            or request["operation"] in {"cold", "block", "info", "integrity"}):
        raise ValueError("Invalid steady-use block request")
    warmup_audits = []
    started = time.perf_counter()
    while len(warmup_audits) < warmup or time.perf_counter() - started < warmup_seconds:
        sample = execute(request["operation"], request["case"])
        warmup_audits.append(sample["audit"])
    warmup_elapsed = time.perf_counter() - started
    samples = [execute(request["operation"], request["case"]) for _ in range(repeats)]
    result = summarize(samples)
    result.update({
        "warmup_audits": warmup_audits,
        "warmup_count": len(warmup_audits),
        "warmup_elapsed_seconds": warmup_elapsed,
        "warmup_min_seconds": warmup_seconds,
        "warmup_quality_pass": all(a["pass"] for a in warmup_audits),
    })
    return result


def run(args):
    cases = [case for case in CASES if (not args.case or case.name in args.case)
             and (args.case or args.profile == "full" or case.name in SMOKE)]
    if args.case and set(args.case) - {case.name for case in CASES}:
        raise ValueError(f"Unknown case; choices: {[case.name for case in CASES]}")
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    report = {"schema_version": 1, "seed": args.seed, "profile": args.profile,
              "repeats": args.repeats, "warmup": args.warmup,
              "platform": platform.platform(),
              "timing_policy": "Persistent isolated processes; alternating variant order per sample; synchronized calls; every call audited outside timing; no output caching.",
              "cold_policy": "First fit and query for each case in that worker, after imports. Earlier cases may populate Metal compilation caches; process startup and input construction excluded.",
              "fit_policy": "fit measures eager work only; fit_query measures a fresh fit plus its first query, including any deferred training preparation. Report fit_query when comparing complete initial-use costs.",
              "query_tolerances": {"atol": 1e-6, "rtol": 1e-6, "indices": "exact equality"},
              "accuracy_reference": "sklearn brute float64 on serialized float32 inputs; selected distances recomputed by direct float64 subtraction; uniform and distance predictions from these neighbors.",
              "resident_boundary": "MLX inputs uploaded before timing; fit owns its data; returned arrays are evaluated before timer stops. Other variants include NumPy input/output boundaries.",
              "cases": []}
    with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-bench-") as temp:
        temp = Path(temp)
        reference = temp / "reference"
        manifest = extract_reference(args.reference_archive, reference)
        report["reference"] = manifest
        fixtures = temp / "fixtures"
        fixtures.mkdir()
        fixture_info = [make_fixture(case, args.seed + CASES.index(case), fixtures) for case in cases]
        report["fixtures"] = fixture_info
        variants = ["current", "reference"]
        if args.threads in {"1", "both"}:
            variants.append("sklearn_1")
        if args.threads in {"default", "both"}:
            variants.append("sklearn_default")
        if args.resident:
            variants.append("resident")
        workers = {}
        try:
            for variant in variants:
                workers[variant] = Worker(variant, reference if variant == "reference" else ROOT / "src", fixtures)
            report["workers"] = {name: worker.info for name, worker in workers.items()}
            for case in cases:
                entry = {"case": asdict(case), "variants": {name: {} for name in variants}}
                for variant in variants:
                    entry["variants"][variant]["cold"] = workers[variant].request({"op": "cold", "case": asdict(case)})
                operations = ("fit", "fit_query", "kneighbors", "indices", "predict_uniform", "proba_uniform",
                              "regress_uniform", "predict_distance", "proba_distance", "regress_distance")
                for operation in operations:
                    names = [name for name in variants if name != "resident" or operation in {"fit", "fit_query", "kneighbors", "indices"}]
                    samples = {name: [] for name in names}
                    warmup_audits = {name: [] for name in names}
                    for repeat in range(args.warmup + args.repeats):
                        for name in (names if repeat % 2 == 0 else list(reversed(names))):
                            sample = workers[name].request({"op": operation, "case": asdict(case)})
                            if repeat >= args.warmup:
                                samples[name].append(sample)
                            else:
                                warmup_audits[name].append(sample["audit"])
                    for name in names:
                        entry["variants"][name][operation] = summarize(samples[name])
                        entry["variants"][name][operation]["warmup_audits"] = warmup_audits[name]
                report["cases"].append(entry)
                current = entry["variants"]["current"]["kneighbors"]["median_ms"]
                before = entry["variants"]["reference"]["kneighbors"]["median_ms"]
                entry["query_speedup_vs_reference"] = before / current
                print(f"{case.name}: query {current:.3f} ms, reference {before:.3f} ms, {before/current:.2f}x", flush=True)
            report["source_integrity_pass"] = all(
                worker.request({"op": "integrity"}) == worker.info["source_hashes"]
                for worker in workers.values()
            )
            report["quality_by_variant"] = {
                variant_name: all(
                    case["variants"][variant_name]["cold"]["audit"]["pass"]
                    and all(
                        value["quality_pass"] and all(a["pass"] for a in value["warmup_audits"])
                        for name, value in case["variants"][variant_name].items()
                        if name != "cold"
                    )
                    for case in report["cases"]
                )
                for variant_name in variants
            }
            # Preserve all comparator audit failures in the report. A comparator
            # failure is distinct from a failure of the implementation being
            # measured, and must not silently weaken its accuracy tolerance.
            report["quality_pass"] = all(report["quality_by_variant"].values())
            report["implementation_quality_pass"] = all(
                passed for name, passed in report["quality_by_variant"].items()
                if name in {"current", "reference", "resident"}
            )
        finally:
            for worker in workers.values():
                worker.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"implementation_quality_pass={report['implementation_quality_pass']} "
        f"quality_pass={report['quality_pass']} "
        f"source_integrity_pass={report['source_integrity_pass']}: {args.output}"
    )
    return 0 if report["implementation_quality_pass"] and report["source_integrity_pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--case", action="append")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", choices=("1", "default", "both"), default="both")
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--reference-archive", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/knn-pipeline-2026-10-28.json")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", help=argparse.SUPPRESS)
    parser.add_argument("--fixtures", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker_main(args)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
