"""Isolated SVC comparison against the source-frozen first native implementation.

Run alone, after building the current extensions::

    python -m benchmarks.svc_working_set --repeats 5 --seed 41 \
        --baseline-dir /tmp/mlxlearn-svc-previous \
        --output benchmarks/results/svc-working-set-seed41.json

Each case/variant runs in a separate process. A first fresh fit and its first and
repeat predictions are reported separately from subsequent fresh-model medians.
Imports and input generation are outside timing. These are first calls within a
worker, not a claim about cold hardware or operating-system caches.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "benchmarks/reference/svc-pre-working-set-2026-10-31.zip"
MODULE = "benchmarks.svc_working_set"
BUILD_MARKER = ".svc-baseline-build.json"
THREAD_VARIABLES = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
CASES = (
    ("rbf_narrow", "rbf", 2000, 20, 2, "cpu", 200.0),
    ("rbf_wide", "rbf", 4000, 256, 2, "mlx", 200.0),
    ("linear", "linear", 1200, 20, 2, "cpu", 200.0),
    ("poly", "poly", 2000, 32, 2, "cpu", 200.0),
    ("sigmoid", "sigmoid", 2000, 32, 2, "cpu", 200.0),
    ("rbf_multiclass", "rbf", 1600, 32, 4, "mlx", 200.0),
    ("rbf_bounded_cache", "rbf", 1600, 32, 2, "mlx", 0.05),
)
EXTRA_CASES = (("linear_medium", "linear", 2500, 32, 2, "cpu", 200.0),)
VARIANTS = ("current", "previous", "sklearn")
METRICS = ("fit", "first_predict", "repeat_predict")
SOURCE_SUFFIXES = {".py", ".cpp", ".mm", ".h"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def controlled_environment(source_root):
    """Do not inherit an import path that can leak the current package to a baseline."""
    env = os.environ.copy()
    env.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    env["PYTHONPATH"] = os.pathsep.join((str(Path(source_root).resolve() / "src"), str(ROOT)))
    env["PYTHONNOUSERSITE"] = "1"
    return env


def read_reference_manifest(archive=REFERENCE):
    archive = Path(archive)
    manifest = json.loads(archive.with_suffix(".manifest.json").read_text())
    if sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("Baseline archive hash does not match its manifest")
    return manifest


def verify_source_tree(directory, manifest):
    directory = Path(directory)
    for path, digest in manifest["files"].items():
        file = directory / path
        if not file.is_file() or sha256(file) != digest:
            raise ValueError(f"Baseline source hash mismatch: {path}")
    expected = {p for p in manifest["files"] if p.startswith("src/")}
    actual = {
        p.relative_to(directory).as_posix() for p in (directory / "src").rglob("*")
        if p.is_file() and (p.suffix in SOURCE_SUFFIXES or p.name == "py.typed")
    }
    if actual != expected:
        raise ValueError(f"Unexpected baseline source inventory: {sorted(actual ^ expected)}")


def extract_reference(directory, archive=REFERENCE):
    """Verify every member before extracting; existing files are never overwritten."""
    directory = Path(directory)
    manifest = read_reference_manifest(archive)
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len({m.filename for m in members}) != len(members):
            raise ValueError("Duplicate baseline archive member")
        if {m.filename for m in members} != set(manifest["files"]):
            raise ValueError("Baseline archive inventory does not match manifest")
        content = {}
        for member in members:
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
                raise ValueError("Unsafe baseline archive member")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Baseline archive cannot contain symbolic links")
            data = zipped.read(member)
            if hashlib.sha256(data).hexdigest() != manifest["files"][member.filename]:
                raise ValueError(f"Baseline member hash mismatch: {member.filename}")
            content[member.filename] = data
    for name, data in content.items():
        target = directory / name
        if target.exists():
            if not target.is_file() or target.read_bytes() != data:
                raise ValueError(f"Refusing to overwrite changed baseline source: {name}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    verify_source_tree(directory, manifest)
    return manifest


def extension_files(directory):
    return {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted((Path(directory) / "src").rglob("*.so"))
    }


def prepare_baseline(directory, archive=REFERENCE):
    """Build once with this interpreter; cached sources and binaries are reverified."""
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError("--baseline-dir must be outside the repository")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = extract_reference(directory, archive)
    env = controlled_environment(directory)
    identity = {
        "archive_sha256": manifest["archive_sha256"], "python_executable": sys.executable,
        "python": sys.version, "soabi": sysconfig.get_config_var("SOABI"),
        "platform": platform.platform(),
        "compiler_environment": {key: env.get(key) for key in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "ARCHFLAGS", "MACOSX_DEPLOYMENT_TARGET")},
    }
    marker = directory / BUILD_MARKER
    if marker.exists():
        cached = json.loads(marker.read_text())
        if cached.get("identity") == identity:
            if extension_files(directory) != cached.get("extension_sha256"):
                raise ValueError("Cached baseline extension hashes changed")
            return manifest, {**cached, "reused": True, "directory": str(directory)}
    command = [sys.executable, "setup.py", "build_ext", "--inplace", "--force"]
    build = subprocess.run(command, cwd=directory, env=env, text=True, capture_output=True, check=False)
    log = directory / "svc-baseline-build.log"
    log.write_text(build.stdout + build.stderr)
    if build.returncode:
        raise RuntimeError(f"Baseline build failed; see {log}")
    binaries = extension_files(directory)
    for name in ("_smo_native", "_svc_cpu_native"):
        if not any(Path(path).name.startswith(name + ".") for path in binaries):
            raise RuntimeError(f"Baseline optional extension {name} was not built; see {log}")
    info = {"identity": identity, "command": command, "extension_sha256": binaries, "build_log_sha256": sha256(log)}
    marker.write_text(json.dumps(info, indent=2) + "\n")
    return manifest, {**info, "reused": False, "directory": str(directory)}


def source_hashes(directory):
    directory = Path(directory)
    paths = [directory / p for p in ("setup.py", "MANIFEST.in", "pyproject.toml")]
    paths.extend(p for p in (directory / "src").rglob("*")
                 if p.is_file() and (p.suffix in SOURCE_SUFFIXES or p.name == "py.typed"))
    return {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(paths)}


def check_import_roots(source_root, modules=None):
    expected = (Path(source_root) / "src").resolve()
    modules = sys.modules if modules is None else modules
    origins = {}
    for name, module in list(modules.items()):
        if name == "mlxlearn" or name.startswith("mlxlearn.") or name.startswith("mlxlearn_device"):
            origin = getattr(module, "__file__", None)
            if origin is not None:
                file = Path(origin).resolve()
                if not file.is_relative_to(expected):
                    raise RuntimeError(f"Worker imported {name} from the wrong source: {file}")
                origins[name] = str(file)
    if "mlxlearn" not in origins:
        raise RuntimeError("Worker did not import the selected mlxlearn package")
    return origins


def model_metadata(model):
    def serializable(value):
        return value.tolist() if hasattr(value, "tolist") else value

    return {
        "fit_status": int(model.fit_status_), "n_iter": serializable(model.n_iter_),
        "n_support": serializable(model.n_support_), "support_count": int(len(model.support_)),
        "backend": getattr(model, "execution_backend_", "sklearn"),
        "pair_solver": serializable(getattr(model, "_pair_solver_", None)),
        "pair_kkt_violation": serializable(getattr(model, "_pair_kkt_violation_", None)),
    }


def timed(call, mx):
    mx.synchronize()
    start = time.perf_counter()
    result = call()
    mx.synchronize()
    return time.perf_counter() - start, result


def worker(args):
    # Deliberately delayed: the parent must never import either SVC implementation.
    import mlx.core as mx
    import numpy as np
    import scipy
    import sklearn
    from sklearn.svm import SVC as SkSVC
    from threadpoolctl import threadpool_info, threadpool_limits

    name, kernel, n, d, classes, backend, cache = {c[0]: c for c in CASES + EXTRA_CASES}[args.case]
    if any(os.environ.get(variable) != "1" for variable in THREAD_VARIABLES):
        raise RuntimeError("Worker thread limits must be set before numerical imports")
    with np.load(args.fixture) as fixture:
        train, labels, query = fixture["train"], fixture["labels"], fixture["query"]
    if train.dtype != np.float32 or query.dtype != np.float32 or train.shape != (n, d) or query.shape != (257, d):
        raise ValueError("Unexpected benchmark fixture shape or dtype")
    config = contextlib.nullcontext()
    if args.variant == "sklearn":
        estimator = SkSVC
    else:
        from mlxlearn._common.config import tuning_context
        from mlxlearn.svm import SVC

        estimator = SVC
        config = tuning_context(force_backend=backend)
        check_import_roots(args.source_root)
    trials = []
    with threadpool_limits(limits=1), config:
        for _ in range(args.repeats + 1):
            fit_s, model = timed(lambda: estimator(kernel=kernel, cache_size=cache, decision_function_shape="ovo").fit(train, labels), mx)
            first_predict_s, first_prediction = timed(lambda model=model: model.predict(query), mx)
            repeat_predict_s, prediction = timed(lambda model=model: model.predict(query), mx)
            # Margin calculation and all metadata follow both timed predictions.
            margin = model.decision_function(query)
            trials.append({
                "fit_s": fit_s, "first_predict_s": first_predict_s, "repeat_predict_s": repeat_predict_s,
                "model": model_metadata(model), "prediction": prediction.tolist(),
                "first_prediction": first_prediction.tolist(), "margin": margin.tolist(),
            })
            del model
        pool_info = threadpool_info()
    native_extensions, imported_modules = {}, {}
    if args.variant != "sklearn":
        names = ["mlxlearn._solvers._smo_native", "mlxlearn._kernels._svc_cpu_native"]
        if args.variant == "current":
            names.append("mlxlearn._solvers._smo_block_native")
        for name in names:
            module = importlib.import_module(name)
            native_extensions[name] = {"path": str(Path(module.__file__).resolve()), "sha256": sha256(module.__file__)}
        imported_modules = check_import_roots(args.source_root)
    result = {
        "variant": args.variant, "case": args.case, "seed": args.seed, "repeats": args.repeats,
        "fixture_sha256": sha256(args.fixture), "trials": trials,
        "environment": {
            "pid": os.getpid(), "python_executable": sys.executable, "python": sys.version,
            "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "mlx": mx.__version__, "mlx_device": mx.device_info(),
            "thread_environment": {key: os.environ.get(key) for key in THREAD_VARIABLES},
            "threadpools_inside_limit": pool_info, "thread_limit": 1,
            "source_root": None if args.variant == "sklearn" else str(args.source_root.resolve()),
            "source_sha256": {} if args.variant == "sklearn" else source_hashes(args.source_root),
            "native_extensions": native_extensions, "imported_mlxlearn_modules": imported_modules,
            "harness_sha256": sha256(__file__),
        },
    }
    args.worker_output.write_text(json.dumps(result) + "\n")


def make_fixture(case, seed, destination):
    import numpy as np
    from sklearn.datasets import make_classification

    _, _, n, d, classes, _, _ = case
    X, y = make_classification(n_samples=n + 257, n_features=d,
                               n_informative=min(12, d - 4), n_redundant=4,
                               n_classes=classes, random_state=seed)
    X = X.astype(np.float32)
    np.savez(destination, train=X[:n], labels=y[:n], query=X[n:], target=y[n:])
    return y[n:]


def launch_worker(case, variant, source_root, fixture, destination, repeats, seed):
    command = [sys.executable, "-m", MODULE, "--worker", "--variant", variant,
               "--source-root", str(source_root), "--case", case[0], "--fixture", str(fixture),
               "--worker-output", str(destination), "--repeats", str(repeats), "--seed", str(seed)]
    run = subprocess.run(command, cwd=destination.parent, env=controlled_environment(source_root),
                         text=True, capture_output=True, check=False)
    if run.returncode:
        log = destination.with_suffix(".log")
        log.write_text(run.stdout + run.stderr)
        raise RuntimeError(f"{variant}/{case[0]} worker failed: {run.stderr[-3000:]} (full log: {log})")
    result = json.loads(destination.read_text())
    result["worker_command"] = command
    if run.stderr:
        result["worker_stderr"] = run.stderr
    return result


def verify_worker_provenance(result, source_root, expected_sources, expected_extensions):
    environment = result["environment"]
    if environment["source_sha256"] != expected_sources:
        raise ValueError("Worker source hashes changed or do not match the selected snapshot")
    for extension in environment["native_extensions"].values():
        path = Path(extension["path"]).resolve()
        if not path.is_relative_to(Path(source_root).resolve()):
            raise ValueError("Worker extension originated outside its source snapshot")
        relative = path.relative_to(Path(source_root).resolve()).as_posix()
        if expected_extensions.get(relative) != extension["sha256"]:
            raise ValueError("Worker extension hash does not match its recorded build")


def summarize(samples):
    return {"median_ms": statistics.median(samples) * 1000, "samples_s": samples}


def compare_case(case, results, target):
    import numpy as np

    name, kernel, n, d, classes, backend, cache = case
    if set(results) != set(VARIANTS):
        raise ValueError("Every case requires current, previous and sklearn results")
    if len({result["fixture_sha256"] for result in results.values()}) != 1:
        raise ValueError("Workers used different input fixtures")
    trial_counts = {len(result["trials"]) for result in results.values()}
    if len(trial_counts) != 1 or min(trial_counts) < 2:
        raise ValueError("Workers must report first call plus matching repeated trials")
    for variant, result in results.items():
        if result["variant"] != variant or result["case"] != name:
            raise ValueError("Worker result identity does not match the requested case/variant")
        if len(result["trials"]) != result["repeats"] + 1:
            raise ValueError("Worker trial count does not match requested repeats")
        for trial in result["trials"]:
            if any(not np.isfinite(trial[metric + "_s"]) or trial[metric + "_s"] <= 0 for metric in METRICS):
                raise ValueError("Worker timings must be positive and finite")
    if len({r["seed"] for r in results.values()}) != 1:
        raise ValueError("Workers used different seeds")
    threshold = 0.95 if kernel == "sigmoid" else 0.99
    expected_margin_shape = target.shape if classes == 2 else (len(target), classes * (classes - 1) // 2)
    quality = {variant: [] for variant in VARIANTS}
    passed = True
    for index in range(next(iter(trial_counts))):
        predictions = {v: np.asarray(r["trials"][index]["prediction"]) for v, r in results.items()}
        margins = {v: np.asarray(r["trials"][index]["margin"], dtype=np.float64) for v, r in results.items()}
        for variant, result in results.items():
            trial = result["trials"][index]
            prediction, margin = predictions[variant], margins[variant]
            if prediction.shape != target.shape or margin.shape != expected_margin_shape:
                raise ValueError("Worker prediction or OVO margin shapes do not match")
            first_matches = bool(np.array_equal(prediction, trial["first_prediction"]))
            q = {
                **trial["model"], "accuracy": float(np.mean(prediction == target)),
                "first_repeat_predictions_equal": first_matches,
                "all_margins_finite": bool(np.isfinite(margin).all()),
                "agreement_with_previous": float(np.mean(prediction == predictions["previous"])),
                "agreement_with_sklearn": float(np.mean(prediction == predictions["sklearn"])),
                "max_margin_difference_from_previous": float(np.max(np.abs(margin - margins["previous"]))),
                "max_margin_difference_from_sklearn": float(np.max(np.abs(margin - margins["sklearn"]))),
                "margins_close_to_previous": bool(np.allclose(margin, margins["previous"], atol=0.02, rtol=0.005)),
                "margins_close_to_sklearn": bool(np.allclose(margin, margins["sklearn"], atol=0.02, rtol=0.005)),
            }
            quality[variant].append(q)
            passed &= q["fit_status"] == 0 and first_matches and q["all_margins_finite"]
        current = quality["current"][-1]
        passed &= current["agreement_with_previous"] >= threshold and current["agreement_with_sklearn"] >= threshold
        passed &= current["margins_close_to_previous"] and current["margins_close_to_sklearn"]
    timing = {
        metric: {variant: summarize([trial[metric + "_s"] for trial in result["trials"][1:]])
                 for variant, result in results.items()}
        for metric in METRICS
    }
    return {
        "name": name, "kernel": kernel, "shape": [n, d], "classes": classes,
        "backend": backend, "cache_size_mb": cache, "query_rows": 257,
        "decision_function_shape": "ovo", "seed": results["current"]["seed"],
        "fixture_sha256": results["current"]["fixture_sha256"],
        "first_call": {variant: {metric + "_s": result["trials"][0][metric + "_s"] for metric in METRICS}
                       for variant, result in results.items()},
        **timing, "quality": {variant: entries[-1] for variant, entries in quality.items()},
        "quality_samples_including_first": quality, "quality_passed": bool(passed),
        "quality_thresholds": {"prediction_agreement": threshold, "margin_atol": 0.02, "margin_rtol": 0.005},
        "speedup_vs_previous": {metric: timing[metric]["previous"]["median_ms"] / timing[metric]["current"]["median_ms"] for metric in METRICS},
        "speedup_vs_sklearn": {metric: timing[metric]["sklearn"]["median_ms"] / timing[metric]["current"]["median_ms"] for metric in METRICS},
        "workers": {variant: {key: value for key, value in result.items() if key != "trials"}
                    for variant, result in results.items()},
    }


def select_cases(requested, smoke):
    if requested and smoke:
        raise ValueError("Use either --cases or --smoke")
    if not requested:
        return CASES[:2] if smoke else CASES
    names = [name for group in requested for name in group.split(",") if name]
    available = {case[0]: case for case in CASES + EXTRA_CASES}
    if not names or len(names) != len(set(names)) or any(name not in available for name in names):
        raise ValueError(f"--cases must name unique cases from {', '.join(available)}")
    return tuple(available[name] for name in names)


def driver(args):
    cases = select_cases(args.cases, args.smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        if args.baseline_dir is None:
            baseline = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="svc-frozen-native-")))
        else:
            baseline = args.baseline_dir
        baseline = baseline.resolve()
        manifest, build = prepare_baseline(baseline)
        work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="svc-working-set-")))
        report = {
            "schema_version": 1, "seed": args.seed, "repeats": args.repeats,
            "harness_sha256": sha256(__file__), "baseline_manifest": manifest, "baseline_build": build,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "current_source_sha256": source_hashes(ROOT),
            "current_extension_sha256": extension_files(ROOT),
            "timing_scope": "Separate process per case/variant; imports and fixture generation excluded; startup fresh fit/first predict/repeat predict outside medians; subsequent complete fresh fits with first and repeat 257-row predictions; synchronized MLX; one CPU thread",
            "cases": [],
        }
        for case_index, case in enumerate(cases):
            fixture = work / (case[0] + "-fixture.npz")
            target = make_fixture(case, args.seed, fixture)
            results = {}
            order = VARIANTS if case_index % 2 == 0 else tuple(reversed(VARIANTS))
            for variant in order:
                source_root = baseline if variant == "previous" else ROOT
                destination = work / f"{case[0]}-{variant}.json"
                results[variant] = launch_worker(case, variant, source_root, fixture, destination, args.repeats, args.seed)
                if variant != "sklearn":
                    expected_sources = report["current_source_sha256"] if variant == "current" else {
                        p: digest for p, digest in manifest["files"].items()
                        if p.startswith("src/") or p in ("setup.py", "MANIFEST.in", "pyproject.toml")
                    }
                    expected_extensions = report["current_extension_sha256"] if variant == "current" else build["extension_sha256"]
                    verify_worker_provenance(results[variant], source_root, expected_sources, expected_extensions)
            compared = compare_case(case, results, target)
            compared["worker_order"] = list(order)
            report["cases"].append(compared)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            speed = compared["speedup_vs_previous"]
            print(f"{case[0]}: fit {speed['fit']:.2f}x; first predict {speed['first_predict']:.2f}x; repeat predict {speed['repeat_predict']:.2f}x; quality={compared['quality_passed']}", flush=True)
        if not all(case["quality_passed"] for case in report["cases"]):
            raise SystemExit("SVC working-set benchmark quality gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=[case[0] for case in CASES + EXTRA_CASES], help=argparse.SUPPRESS)
    parser.add_argument("--fixture", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.worker:
        if any(getattr(args, field) is None for field in ("variant", "source_root", "case", "fixture", "worker_output")):
            parser.error("worker arguments are incomplete")
        worker(args)
    else:
        if args.output is None:
            parser.error("--output is required")
        try:
            driver(args)
        except ValueError as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
