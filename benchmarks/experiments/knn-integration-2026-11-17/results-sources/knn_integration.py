"""Source-isolated, interleaved kNN integration benchmarks.

Run ``python -m benchmarks.knn_integration --profile full --output report.json``.
The baseline defaults to v0.1.3; ``--reference label=revision`` adds another
git snapshot. Current sources (including uncommitted edits) are frozen before
any worker starts. Source archives, hashes, all timings and failed audits are
retained beside the report. Native extensions are intentionally omitted: every
case explicitly uses GPU search, never automatic CPU dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import select
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    name: str
    train_rows: int
    features: int
    query_rows: int
    k: int
    distribution: str = "gaussian"


CASES = (
    Case("small_certified", 2048, 96, 64, 10),
    Case("small_compact", 16384, 16, 64, 8),
    Case("gaussian_256k_k5", 262144, 96, 256, 5),
    Case("gaussian_256k_k10", 262144, 96, 256, 10),
    Case("gaussian_256k_k32", 262144, 96, 256, 32),
    Case("gaussian_1m_k10", 1048576, 96, 512, 10),
    Case("gaussian_1m_k32", 1048576, 96, 512, 32),
    Case("integer_256k_k8", 262144, 16, 256, 8, "integer"),
    Case("integer_256k_k9", 262144, 16, 256, 9, "integer"),
    Case("clustered_32k", 32768, 256, 128, 10, "clustered"),
)
QUICK_CASES = {"small_certified", "gaussian_256k_k10", "integer_256k_k8"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def manifest(folder):
    return {str(p.relative_to(folder)): digest(p.read_bytes())
            for p in sorted(folder.rglob("*")) if p.is_file()}


def snapshot(root, revision, destination):
    """Freeze Python/kernel source, without borrowing installed extensions."""
    destination.mkdir(parents=True)
    if revision is None:
        for package in sorted((root / "src").glob("mlxlearn*")):
            if not (package / "__init__.py").is_file():
                continue
            shutil.copytree(package, destination / "src" / package.name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "*.dylib"))
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    else:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", revision], text=True).strip()
        archive = subprocess.check_output(["git", "-C", str(root), "archive", commit, "src"])
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(destination, filter="data")
    hashes = manifest(destination)
    return {"revision": revision or "working_tree", "commit": commit,
            "files": hashes, "sha256": digest(json.dumps(hashes, sort_keys=True).encode())}


def archive_snapshot(source, target):
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def make_fixture(case, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    shape = (case.train_rows, case.features)
    query_shape = (case.query_rows, case.features)
    if case.distribution == "integer":
        return (rng.integers(-2, 3, shape).astype(np.float32),
                rng.integers(-2, 3, query_shape).astype(np.float32))
    if case.distribution == "clustered":
        centers = rng.normal(size=(16, case.features)).astype(np.float32)
        train = centers[rng.integers(0, len(centers), case.train_rows)]
        queries = centers[rng.integers(0, len(centers), case.query_rows)]
        return (train + np.float32(1e-4) * rng.normal(size=shape).astype(np.float32),
                queries + np.float32(1e-4) * rng.normal(size=query_shape).astype(np.float32))
    return rng.normal(size=shape).astype(np.float32), rng.normal(size=query_shape).astype(np.float32)


def exact_oracle(train, queries, k):
    """Direct float64 differences with canonical ties and bounded workspace."""
    import numpy as np

    distances, indices = [], []
    for query in queries.astype(np.float64):
        squared = np.empty(len(train), dtype=np.float64)
        for start in range(0, len(train), 4096):
            delta = train[start:start + 4096].astype(np.float64) - query
            squared[start:start + len(delta)] = np.sum(delta * delta, axis=1)
        candidates = np.flatnonzero(squared <= np.partition(squared, k - 1)[k - 1])
        order = candidates[np.lexsort((candidates, squared[candidates]))][:k]
        indices.append(order)
        distances.append(np.sqrt(squared[order]))
    return np.asarray(distances), np.asarray(indices)


def loaded_sources(source):
    """Reject an installed/editable package silently leaking into a snapshot."""
    source = source.resolve()
    loaded = {}
    for name, module in sys.modules.items():
        if not name.split(".")[0].startswith("mlxlearn"):
            continue
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if not path.is_relative_to(source):
                raise RuntimeError(f"Mixed source: {name} loaded from {path}")
            loaded[name] = str(path.relative_to(source))
    return loaded


def worker_main(source):
    import numpy as np

    sys.path.insert(0, str(source / "src"))
    import mlx.core as mx

    from mlxlearn_device import fit_neighbors

    case = train = query = index = oracle = None
    api = "device"

    def evaluate():
        if api == "device":
            out = index.query(query, case.k)
            mx.eval(*out)
            return out
        return index.query(query, case.k)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request["op"]
            if operation == "info":
                result = {"python": sys.version, "platform": platform.platform(),
                          "packages": {name: importlib.metadata.version(name)
                                       for name in ("numpy", "mlx", "scikit-learn", "scipy")},
                          "device": mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info(),
                          "native_extensions": "omitted; explicit GPU path", "loaded": loaded_sources(source)}
            elif operation == "prepare":
                case, api = Case(**request["case"]), request["api"]
                # Drop the preceding fitted index before constructing the next fixture.
                index = query = train = oracle = None
                mx.clear_cache()
                train, host_query = make_fixture(case, request["seed"])
                oracle = exact_oracle(train, host_query[:request["oracle_rows"]], case.k)
                if api == "device":
                    query = mx.array(host_query)
                    index = fit_neighbors(mx.array(train))
                else:
                    from mlxlearn._kernels.distance import build_index

                    query = host_query
                    index = build_index(train, backend="mlx")
                result = {"fixture_sha256": [digest(train.tobytes()), digest(host_query.tobytes())],
                          "input_bytes": int(train.nbytes + host_query.nbytes),
                          "loaded": loaded_sources(source)}
            elif operation == "query":
                mx.reset_peak_memory()
                active_before = mx.get_active_memory()
                start = time.perf_counter()
                output = evaluate()
                elapsed = 1000 * (time.perf_counter() - start)
                peak_active = mx.get_peak_memory()
                distances, indices = (np.asarray(a) for a in output)
                count = len(oracle[0])
                indices_ok = np.array_equal(indices[:count], oracle[1])
                distances_ok = np.allclose(distances[:count], oracle[0], atol=1e-6, rtol=1e-6)
                result = {"ms": elapsed, "sha256": [digest(distances.tobytes()),
                                                   digest(indices.astype(np.int64).tobytes())],
                          "mlx_memory": {"active_before_bytes": active_before,
                                         "peak_active_bytes": peak_active,
                                         "peak_increment_bytes": max(0, peak_active - active_before)},
                          "audit": {"pass": bool(indices_ok and distances_ok),
                                    "indices_exact": bool(indices_ok),
                                    "distances_close": bool(distances_ok), "oracle_rows": count}}
            elif operation == "integrity":
                result = {"files": manifest(source), "loaded": loaded_sources(source)}
            elif operation == "close":
                break
            else:
                raise ValueError(f"Unknown worker operation: {operation}")
            print(json.dumps({"result": result}), flush=True)
        except Exception:
            print(json.dumps({"error": traceback.format_exc()}), flush=True)


class Worker:
    def __init__(self, source, timeout):
        self.timeout = timeout
        # Isolated mode ignores PYTHONPATH and the current directory. The only
        # package path subsequently added is this worker's frozen source tree.
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker", str(source)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def request(self, value):
        if self.process.poll() is not None:
            raise RuntimeError(f"Worker exited with {self.process.returncode}")
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], self.timeout)[0]:
            self.process.kill()
            self.process.wait()
            raise TimeoutError(f"Worker exceeded {self.timeout}s for {value['op']}")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Worker closed its output")
        answer = json.loads(line)
        if "error" in answer:
            raise RuntimeError(answer["error"])
        return answer["result"]

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()


def summarize_case(variants, baseline):
    """Failures stay visible and never receive a successful speedup claim."""
    reference = variants.get(baseline, {})
    expected = reference.get("samples", [{}])[0].get("sha256") if reference.get("samples") else None
    fixture = reference.get("preparation", {}).get("fixture_sha256")
    for entry in variants.values():
        samples = entry.get("samples", [])
        checks = entry.get("warmup", []) + samples
        entry["quality_pass"] = bool(expected and fixture and samples and not entry.get("error")
                                     and entry.get("preparation", {}).get("fixture_sha256") == fixture and all(
            sample["audit"]["pass"] and sample["sha256"] == expected for sample in checks))
        if samples:
            entry["median_ms"] = statistics.median(sample["ms"] for sample in samples)
    baseline_ms = reference.get("median_ms")
    for entry in variants.values():
        entry["speedup_vs_baseline"] = (baseline_ms / entry["median_ms"]
                                        if baseline_ms and entry["quality_pass"]
                                        and reference.get("quality_pass") else None)
    return all(entry["quality_pass"] for entry in variants.values())


def parse_reference(value):
    label, separator, revision = value.partition("=")
    if not separator or not label or not revision or not label.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError("references must be safe-label=git-revision")
    if label == "current":
        raise argparse.ArgumentTypeError("current is reserved for the working tree")
    return label, revision


def run(args):
    references = [("baseline", args.baseline), *args.reference, ("current", None)]
    if len({label for label, _ in references}) != len(references):
        raise ValueError("Reference labels must be distinct")
    selected = [case for case in CASES if (not args.cases or case.name in args.cases)
                and (args.cases or args.profile == "full" or case.name in QUICK_CASES)]
    if not selected or (args.cases and set(args.cases) - {case.name for case in CASES}):
        raise ValueError("Unknown or empty case selection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = args.output.with_suffix("").with_name(args.output.stem + "-sources")
    if args.output.exists() or artifacts.exists():
        raise ValueError("Output or source archive directory already exists; choose a new output path")
    artifacts.mkdir(exist_ok=True)
    harness = artifacts / "knn_integration.py"
    shutil.copyfile(Path(__file__), harness)
    report = {"schema_version": 1, "protocol": "isolated_interleaved_warm_queries",
              "seed": args.seed, "repeats": args.repeats, "warmup": args.warmup,
              "oracle": "Direct float64 differences, canonical (distance,index) ties; sample rows; rtol=atol=1e-6",
              "timing": "Fit, input creation, compilation warmup and output audits excluded; "
                        "public API includes NumPy output conversion, device API includes mx.eval; "
                        "randomized variant order in every repetition; no concurrent GPU requests",
              "harness_sha256": digest(harness.read_bytes()), "harness": str(harness.resolve()),
              "sources": {}, "cases": [], "quality_pass": False}
    workers = {}
    rng = random.Random(args.seed)
    with tempfile.TemporaryDirectory(prefix="mlxlearn-integration-") as folder:
        folder = Path(folder)
        try:
            for label, revision in references:
                source = folder / label
                metadata = snapshot(ROOT, revision, source)
                archive = artifacts / f"{label}.zip"
                archive_snapshot(source, archive)
                metadata.update(archive=str(archive.resolve()), archive_sha256=digest(archive.read_bytes()))
                workers[label] = Worker(source, args.timeout)
                metadata["environment"] = workers[label].request({"op": "info"})
                report["sources"][label] = metadata
            for case in selected:
                for api in args.apis:
                    row = {"case": asdict(case), "api": api, "variants": {}, "orders": []}
                    report["cases"].append(row)
                    fixture_seed = args.seed + list(CASES).index(case)
                    row["fixture_seed"] = fixture_seed
                    for label, worker in workers.items():
                        entry = row["variants"][label] = {"warmup": [], "samples": []}
                        try:
                            entry["preparation"] = worker.request({"op": "prepare", "case": asdict(case),
                                                                 "api": api, "seed": fixture_seed,
                                                                 "oracle_rows": min(args.oracle_rows, case.query_rows)})
                            for _ in range(args.warmup):
                                entry["warmup"].append(worker.request({"op": "query"}))
                        except Exception as exc:
                            entry["error"] = str(exc)
                    for _ in range(args.repeats):
                        order = list(workers)
                        rng.shuffle(order)
                        row["orders"].append(order)
                        for label in order:
                            entry = row["variants"][label]
                            if entry.get("error"):
                                continue
                            try:
                                entry["samples"].append(workers[label].request({"op": "query"}))
                            except Exception as exc:
                                entry["error"] = str(exc)
                    row["quality_pass"] = summarize_case(row["variants"], "baseline")
                    report["quality_pass"] = all(item["quality_pass"] for item in report["cases"])
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps({"case": case.name, "api": api, "quality_pass": row["quality_pass"],
                                      "median_ms": {label: entry.get("median_ms")
                                                    for label, entry in row["variants"].items()},
                                      "current_speedup": row["variants"]["current"]["speedup_vs_baseline"]}), flush=True)
            report["source_integrity"] = {}
            for label, worker in workers.items():
                try:
                    check = worker.request({"op": "integrity"})
                    report["source_integrity"][label] = {"pass": check["files"] == report["sources"][label]["files"],
                                                         "loaded": check["loaded"]}
                except Exception as exc:
                    report["source_integrity"][label] = {"pass": False, "error": str(exc)}
            report["source_integrity_pass"] = all(item["pass"] for item in report["source_integrity"].values())
            report["quality_pass"] &= report["source_integrity_pass"]
        except Exception as exc:
            report["quality_pass"] = False
            report["fatal_error"] = str(exc)
            raise
        finally:
            for worker in workers.values():
                worker.close()
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--baseline", default="v0.1.3")
    parser.add_argument("--reference", type=parse_reference, action="append", default=[])
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--apis", nargs="+", choices=("device", "numpy"), default=["device", "numpy"])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--oracle-rows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, default=ROOT / "output/knn-integration-2026-11-17/report.json")
    args = parser.parse_args()
    if args.worker:
        worker_main(args.worker.resolve())
        return
    if args.repeats < 1 or args.warmup < 1 or args.oracle_rows < 1 or not 0 < args.timeout < float("inf"):
        parser.error("repeats, warmup, oracle rows and finite timeout must be positive")
    result = run(args)
    if not result["quality_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
