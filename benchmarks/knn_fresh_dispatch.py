"""Same-process current versus verified frozen-v3 query implementation.

    python -m benchmarks.knn_fresh_dispatch \
        --output benchmarks/results/knn-fresh-vs-v3-2026-10-29.json

Load v3 validation and the complete archived device-search dependency graph under
unique module names, then restore their canonical imports for each reference call.
Remaining shared wrappers and dependencies must match v3. Current/current A/A measures
same-process noise. Nothing changes the archived isolated benchmark protocol.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import inspect
import json
import math
import platform
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from benchmarks.knn_pipeline import ROOT, extract_reference, sha256, source_hashes
from benchmarks.knn_review_distributions import REFERENCE, make_fixture


def harness_hashes():
    return {
        str(path.relative_to(ROOT)): sha256(path.read_bytes())
        for path in (Path(__file__).resolve(), ROOT / "benchmarks/knn_pipeline.py",
                     ROOT / "benchmarks/knn_review_distributions.py")
    }


def load_frozen(name, path):
    """Register before execution so dataclass module lookup sees its module."""
    if name in sys.modules:
        raise RuntimeError(f"Frozen module name already in use: {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


DEVICE_MODULES = (
    "_neighbors_registers", "_neighbors_fused", "_neighbors_paired",
    "_neighbors_compact", "_neighbors",
)
PUBLIC_METHODS = ("_check_neighbors_capability", "_fit_neighbors", "_prepare_neighbor_query")


def hooks(neighbors, compact, public_base):
    """Capture all identities that can affect static or lazy neighbor imports."""
    package = sys.modules["mlxlearn_device"]
    return (
        tuple(getattr(public_base.MLXNeighborsMixin, name) for name in PUBLIC_METHODS),
        tuple(sys.modules.get(f"mlxlearn_device.{name}") for name in DEVICE_MODULES),
        tuple(getattr(package, name, None) for name in DEVICE_MODULES),
        package.DeviceNeighborIndex, package.fit_neighbors,
    )


def restore_hooks(neighbors, compact, public_base, values):
    methods, modules, attributes, index_type, fit = values
    package = sys.modules["mlxlearn_device"]
    for name, method in zip(PUBLIC_METHODS, methods, strict=True):
        setattr(public_base.MLXNeighborsMixin, name, method)
    for name, module, attribute in zip(DEVICE_MODULES, modules, attributes, strict=True):
        canonical = f"mlxlearn_device.{name}"
        if module is None:
            sys.modules.pop(canonical, None)
        else:
            sys.modules[canonical] = module
        if attribute is None:
            if hasattr(package, name):
                delattr(package, name)
        else:
            setattr(package, name, attribute)
    package.DeviceNeighborIndex, package.fit_neighbors = index_type, fit


@contextlib.contextmanager
def device_modules(modules):
    """Bind both import caches and package attributes, restoring every identity."""
    package = sys.modules["mlxlearn_device"]
    saved = {
        name: (sys.modules.get(f"mlxlearn_device.{name}"), getattr(package, name, None))
        for name in modules
    }
    exported = package.DeviceNeighborIndex, package.fit_neighbors
    try:
        for name, module in modules.items():
            sys.modules[f"mlxlearn_device.{name}"] = module
            setattr(package, name, module)
        if "_neighbors" in modules:
            package.DeviceNeighborIndex = modules["_neighbors"].DeviceNeighborIndex
            package.fit_neighbors = modules["_neighbors"].fit_neighbors
        yield
    finally:
        for name, (module, attribute) in saved.items():
            canonical = f"mlxlearn_device.{name}"
            if module is None:
                sys.modules.pop(canonical, None)
            else:
                sys.modules[canonical] = module
            if attribute is None:
                if hasattr(package, name):
                    delattr(package, name)
            else:
                setattr(package, name, attribute)
        package.DeviceNeighborIndex, package.fit_neighbors = exported


@contextlib.contextmanager
def query_implementation(
    neighbors, compact, public_base, frozen_devices, frozen_base, *, reference,
):
    """Select a complete frozen search graph and validation, even on failed calls."""
    previous = hooks(neighbors, compact, public_base)
    try:
        with device_modules(frozen_devices if reference else {}):
            if reference:
                for name in PUBLIC_METHODS:
                    setattr(public_base.MLXNeighborsMixin, name,
                            getattr(frozen_base.MLXNeighborsMixin, name))
            yield
    finally:
        restore_hooks(neighbors, compact, public_base, previous)


def shared_mixin_members(path):
    """Reject other modified mixin methods that a frozen method could call."""
    replaced = {"_check_neighbors_capability", "_fit_neighbors", "_prepare_neighbor_query"}
    tree = ast.parse(path.read_text())
    mixin = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "MLXNeighborsMixin")
    return [
        (node.name, ast.dump(node, include_attributes=False))
        for node in mixin.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name not in replaced
    ]


def summary(samples):
    values = [sample["ms"] for sample in samples if "ms" in sample]
    return {
        "median_ms": statistics.median(values) if values else None,
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "completed_samples": len(values),
        "quality_pass": all(sample.get("audit", {}).get("pass", False) for sample in samples),
        "samples": samples,
    }


def run(args):
    if args.pairs < 1 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds <= 0:
        raise ValueError("pairs and warmup_seconds must be positive")
    import numpy as np

    report = {
        "schema_version": 2,
        "status": "running",
        "diagnostic_only": True,
        "platform": platform.platform(),
        "seeds": args.seed or [151, 152, 153, 251, 252, 253],
        "distributions": args.distribution or ["gaussian", "anisotropic"],
        "pairs": args.pairs,
        "warmup_seconds_per_variant": args.warmup_seconds,
        "protocol": (
            "One process, sequential public NumPy estimator calls; GPU completion synchronized. "
            "Reference calls use verified frozen-v3 capability checking, fit validation, "
            "query validation, index constructor, and the entire archived fused/paired/register/compact "
            "search dependency graph within a restoring import context. "
            "Separate fitted query models; fresh fit_query includes deferred preparation. "
            "Each variant receives duration warmup, then paired AB/BA rounds alternate order. "
            "Starting order rotates by seed. Every call is audited outside timing."
        ),
        "purpose": (
            "Measure fresh-fit improvements against frozen-v3 input validation and query "
            "orchestration when current validation or compact builders change. All remaining "
            "shared dependencies must match the archive. This supplements the isolated benchmark."
        ),
        "query_tolerances": {"atol": 1e-6, "rtol": 1e-6, "indices": "exact equality"},
        "accuracy_reference": "Exhaustive direct float64 distance on rounded float32 input; distance/index tie order.",
        "harness_hashes_before": harness_hashes(),
        "cases": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-fresh-") as temporary:
        temporary = Path(temporary)
        report["reference"] = extract_reference(args.reference_archive, temporary / "reference")
        fixtures = temporary / "fixtures"
        fixtures.mkdir()
        prepared_cases = []
        for seed_index, seed in enumerate(report["seeds"]):
            for distribution in report["distributions"]:
                destination = fixtures / str(seed) / distribution
                destination.mkdir(parents=True)
                case, fixture = make_fixture(distribution, seed, destination)
                prepared_cases.append((seed_index, case, fixture, destination))

        import mlx.core as mx
        from threadpoolctl import threadpool_info

        sys.path.insert(0, str(ROOT / "src"))
        import mlxlearn
        import mlxlearn_device
        from mlxlearn._common.config import tuning_context
        from mlxlearn.neighbors import NearestNeighbors
        from mlxlearn.neighbors import _base as public_base
        from mlxlearn_device import _neighbors as neighbors
        from mlxlearn_device import _neighbors_compact as compact
        from mlxlearn_device import _neighbors_integer as integer

        current_devices = {
            name: importlib.import_module(f"mlxlearn_device.{name}") for name in DEVICE_MODULES
        }
        for module in (mlxlearn, mlxlearn_device, public_base, integer, *current_devices.values()):
            if not Path(module.__file__).resolve().is_relative_to((ROOT / "src").resolve()):
                raise RuntimeError(f"Wrong diagnostic package imported: {module.__file__}")
        report["source_hashes_before"] = source_hashes(ROOT / "src")
        before, frozen = report["source_hashes_before"], report["reference"]["files"]
        report["differences_from_frozen"] = {
            "modified": [name for name in sorted(before.keys() & frozen.keys()) if before[name] != frozen[name]],
            "added": sorted(before.keys() - frozen.keys()),
            "removed": sorted(frozen.keys() - before.keys()),
        }
        # Every shared public-package source and device dependency must remain
        # identical. Validation and all five device-search modules are replaced
        # from the archive;
        # new current-only sources such as integer screening are not v3 inputs.
        replaced = {"mlxlearn/neighbors/_base.py",
                    *(f"mlxlearn_device/{name}.py" for name in DEVICE_MODULES)}
        shared = {name: digest for name, digest in frozen.items() if name not in replaced}
        mismatches = [name for name, digest in shared.items() if before.get(name) != digest]
        report["shared_source_verification"] = {
            "policy": "All archived sources except public validation and the five replaced device-search modules must match current.",
            "hashes": shared,
            "mismatches": mismatches,
            "pass": not mismatches,
        }
        if mismatches:
            report["status"] = "incompatible_shared_sources"
            save()
            raise RuntimeError(f"Shared code differs from v3: {mismatches}")
        if mlxlearn_device.DeviceNeighborIndex is not neighbors.DeviceNeighborIndex:
            raise RuntimeError("Public device package does not expose the patched index class")
        if not issubclass(NearestNeighbors, public_base.MLXNeighborsMixin):
            raise RuntimeError("Public estimator does not inherit the patched mixin")
        report["shared_mixin_members_match"] = (
            shared_mixin_members(ROOT / "src/mlxlearn/neighbors/_base.py")
            == shared_mixin_members(temporary / "reference/mlxlearn/neighbors/_base.py")
        )
        if not report["shared_mixin_members_match"]:
            report["status"] = "incompatible_shared_mixin"
            save()
            raise RuntimeError("Unpatched public mixin methods differ from v3")
        report["runtime"] = {
            "python": sys.version,
            "numpy": np.__version__,
            "mlx": mx.__version__,
            "threadpools": threadpool_info(),
            "hardware": dict(mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()),
            "integer_supported": bool(integer.is_supported()),
        }
        if not report["runtime"]["integer_supported"]:
            report["status"] = "unsupported"
            save()
            return 2
        original_hooks = hooks(neighbors, compact, public_base)
        frozen_modules = []
        try:
            frozen_devices = {}
            for name in DEVICE_MODULES:
                # Static imports bind to earlier frozen dependencies; lazy imports
                # receive this same graph during each reference call below.
                with device_modules(frozen_devices):
                    module = load_frozen(
                        f"mlxlearn_device._frozen_knn_fresh{name}",
                        temporary / f"reference/mlxlearn_device/{name}.py",
                    )
                frozen_devices[name] = module
                frozen_modules.append(module)
            with device_modules(frozen_devices):
                frozen_base = load_frozen(
                    "mlxlearn.neighbors._frozen_knn_fresh_base",
                    temporary / "reference/mlxlearn/neighbors/_base.py",
                )
            frozen_modules.append(frozen_base)
            frozen_neighbors = frozen_devices["_neighbors"]
            frozen_compact = frozen_devices["_neighbors_compact"]
            frozen_fused = frozen_devices["_neighbors_fused"]
            frozen_paired = frozen_devices["_neighbors_paired"]
            frozen_registers = frozen_devices["_neighbors_registers"]
            report["frozen_modules"] = {
                module.__name__: {
                    "path": str(Path(module.__file__).resolve()),
                    "sha256": sha256(Path(module.__file__).read_bytes()),
                } for module in frozen_modules
            }
            report["frozen_static_dependencies_match"] = (
                frozen_compact._HEADER == frozen_fused._HEADER
                and frozen_paired._HEADER == frozen_fused._HEADER
                and frozen_paired._merge_kernel is frozen_fused._merge_kernel
                and frozen_paired.paired_scan_source is frozen_registers.paired_scan_source
            )
            if not report["frozen_static_dependencies_match"]:
                raise RuntimeError("Frozen search modules imported a current dependency")
            report["public_method_signatures_match"] = all(
                inspect.signature(getattr(public_base.MLXNeighborsMixin, name))
                == inspect.signature(getattr(frozen_base.MLXNeighborsMixin, name))
                for name in PUBLIC_METHODS
            )
            report["frozen_input_helper_globals_match"] = all(
                getattr(frozen_base.MLXNeighborsMixin, name).__globals__["_prepare_neighbors_input"]
                is frozen_base._prepare_neighbors_input
                for name in ("_check_neighbors_capability", "_prepare_neighbor_query")
            )
            if not (report["public_method_signatures_match"] and report["frozen_input_helper_globals_match"]):
                raise RuntimeError("Frozen input-validation methods have incompatible signatures/globals")
            frozen_hooks = (
                tuple(getattr(frozen_base.MLXNeighborsMixin, name) for name in PUBLIC_METHODS),
                tuple(frozen_devices[name] for name in DEVICE_MODULES),
                tuple(frozen_devices[name] for name in DEVICE_MODULES),
                frozen_neighbors.DeviceNeighborIndex, frozen_neighbors.fit_neighbors,
            )
            with tuning_context(force_backend="mlx"):
                for seed_index, case, fixture, destination in prepared_cases:
                    with np.load(destination / f"{case.name}.npz") as fixture_file:
                        data = {name: fixture_file[name] for name in ("X", "Q", "distances", "indices")}
                    entry = {"case": asdict(case), "fixture": fixture, "experiments": {}}

                    def audit(output, data=data):
                        distances, indices = output
                        index_ok = np.array_equal(indices, data["indices"])
                        distance_ok = np.allclose(distances, data["distances"], atol=1e-6, rtol=1e-6)
                        return {"pass": bool(index_ok and distance_ok), "indices_match": bool(index_ok),
                                "distances_match": bool(distance_ok),
                                "distance_max_abs": float(np.max(np.abs(distances - data["distances"])))}

                    for experiment, operation, routes in (
                        ("fit_query_current_vs_v3", "fit_query", {"current": False, "v3": True}),
                        ("query_current_vs_v3", "kneighbors", {"current": False, "v3": True}),
                        ("fit_query_current_aa", "fit_query", {"current_a": False, "current_b": False}),
                        ("query_current_aa", "kneighbors", {"current_a": False, "current_b": False}),
                    ):
                        names = list(routes)
                        models = {}
                        if operation == "kneighbors":
                            for name in names:
                                with query_implementation(
                                    neighbors, compact, public_base,
                                    frozen_devices, frozen_base,
                                    reference=routes[name],
                                ):
                                    models[name] = NearestNeighbors(
                                        n_neighbors=case.k, algorithm="brute",
                                    ).fit(data["X"])
                        last_fresh = {}

                        def call(name, routes=routes, operation=operation, case=case,
                                 data=data, last_fresh=last_fresh, models=models, audit=audit):
                            try:
                                if hooks(neighbors, compact, public_base) != original_hooks:
                                    raise RuntimeError("Query hooks were not restored before the call")
                                with query_implementation(
                                    neighbors, compact, public_base,
                                    frozen_devices, frozen_base,
                                    reference=routes[name],
                                ):
                                    mx.synchronize()
                                    started = time.perf_counter()
                                    if operation == "fit_query":
                                        model = NearestNeighbors(n_neighbors=case.k, algorithm="brute").fit(data["X"])
                                        last_fresh[name] = model
                                    else:
                                        model = models[name]
                                    result = model.kneighbors(data["Q"])
                                    mx.synchronize()
                                    elapsed = (time.perf_counter() - started) * 1000
                                    check = audit(result)
                                    index = model._index_.device_index
                                    route = "integer" if getattr(index, "_integer", None) is not None else (
                                        "compact" if index._compact is not None else "direct_or_general"
                                    )
                                    preference = getattr(index, "_integer_preferred", None)
                                    expected_route = "compact" if routes[name] or preference is False else "integer"
                                    check["integer_preferred"] = preference
                                    check["index_class_match"] = type(index) is (
                                        frozen_neighbors.DeviceNeighborIndex if routes[name]
                                        else neighbors.DeviceNeighborIndex
                                    )
                                    check["pass"] &= check["index_class_match"]
                                    check["route_match"] = route == expected_route
                                    check["pass"] = check["pass"] and check["route_match"]
                                    check["backend_match"] = model.execution_backend_ == "mlx"
                                    check["pass"] = check["pass"] and check["backend_match"]
                                    check["implementation_hooks_match"] = (
                                        hooks(neighbors, compact, public_base)
                                        == (frozen_hooks if routes[name] else original_hooks)
                                    )
                                    check["pass"] = check["pass"] and check["implementation_hooks_match"]
                                    sample = {"ms": elapsed, "audit": check, "route": route,
                                              "query_implementation": "frozen_v3" if routes[name] else "current",
                                              "backend": model.execution_backend_}
                                sample["audit"]["hooks_restored"] = hooks(neighbors, compact, public_base) == original_hooks
                                sample["audit"]["pass"] &= sample["audit"]["hooks_restored"]
                                return sample
                            except Exception:
                                return {"error": traceback.format_exc(), "audit": {"pass": False}}

                        starting_order = names if seed_index % 2 == 0 else list(reversed(names))
                        warmups = {name: [] for name in names}
                        for name in starting_order:
                            started = time.perf_counter()
                            while not warmups[name] or time.perf_counter() - started < args.warmup_seconds:
                                warmups[name].append(call(name))
                        samples, pairs = {name: [] for name in names}, []
                        for pair in range(args.pairs):
                            order = starting_order if pair % 2 == 0 else list(reversed(starting_order))
                            pair_samples = {}
                            for name in order:
                                sample = call(name)
                                samples[name].append(sample)
                                pair_samples[name] = sample
                            numerator, denominator = pair_samples[names[1]], pair_samples[names[0]]
                            pairs.append({
                                "pair": pair, "order": order,
                                "ratio_second_variant_over_first": (
                                    numerator["ms"] / denominator["ms"]
                                    if "ms" in numerator and "ms" in denominator else None
                                ),
                            })
                        ratios = [pair["ratio_second_variant_over_first"] for pair in pairs
                                  if pair["ratio_second_variant_over_first"] is not None]
                        entry["experiments"][experiment] = {
                            "operation": operation,
                            "ratio_definition": f"{names[1]} milliseconds / {names[0]} milliseconds",
                            "paired_ratio_median": statistics.median(ratios) if ratios else None,
                            "paired_ratio_min": min(ratios) if ratios else None,
                            "paired_ratio_max": max(ratios) if ratios else None,
                            "pairs": pairs,
                            "variants": {name: {"retained": summary(samples[name]),
                                                "warmup": summary(warmups[name])} for name in names},
                        }
                    report["cases"].append(entry)
                    save()
                    print(f"{fixture['distribution']} seed {fixture['seed']}: " + json.dumps({
                        name: value["paired_ratio_median"] for name, value in entry["experiments"].items()
                    }), flush=True)
            report["quality_pass"] = all(
                variant[phase]["quality_pass"]
                for case in report["cases"] for experiment in case["experiments"].values()
                for variant in experiment["variants"].values() for phase in ("warmup", "retained")
            )
            report["quality_by_variant"] = {
                name: all(
                    experiment["variants"][name][phase]["quality_pass"]
                    for case in report["cases"] for experiment in case["experiments"].values()
                    if name in experiment["variants"] for phase in ("warmup", "retained")
                ) for name in ("current", "v3", "current_a", "current_b")
            }
            report["current_quality_pass"] = all(
                passed for name, passed in report["quality_by_variant"].items() if name != "v3"
            )
            report["reference_quality_pass"] = report["quality_by_variant"]["v3"]
            report["acceptance_policy"] = (
                "Current accuracy, source integrity and restored imports determine acceptance. "
                "The aggregate quality_pass and reference_quality_pass retain frozen-v3 failures. "
                "Timings on a failing reference are not an accuracy-matched speed comparison."
            )
            report["source_hashes_after"] = source_hashes(ROOT / "src")
            report["source_integrity_pass"] = report["source_hashes_before"] == report["source_hashes_after"]
            report["harness_hashes_after"] = harness_hashes()
            report["harness_integrity_pass"] = report["harness_hashes_before"] == report["harness_hashes_after"]
            report["frozen_source_integrity_pass"] = source_hashes(temporary / "reference") == frozen
            report["hooks_restored"] = hooks(neighbors, compact, public_base) == original_hooks
            report["status"] = "complete"
        except Exception:
            report["status"] = "error"
            report["error"] = traceback.format_exc()
            raise
        finally:
            restore_hooks(neighbors, compact, public_base, original_hooks)
            report["hooks_restored_after_cleanup"] = hooks(neighbors, compact, public_base) == original_hooks
            for module in frozen_modules:
                if sys.modules.get(module.__name__) is module:
                    del sys.modules[module.__name__]
            report["frozen_modules_unloaded"] = all(module.__name__ not in sys.modules for module in frozen_modules)
            save()
    return 0 if all(report[key] for key in (
        "current_quality_pass", "source_integrity_pass", "harness_integrity_pass", "frozen_source_integrity_pass",
        "hooks_restored", "hooks_restored_after_cleanup", "frozen_modules_unloaded",
        "shared_mixin_members_match", "public_method_signatures_match", "frozen_input_helper_globals_match",
        "frozen_static_dependencies_match",
    )) and report["shared_source_verification"]["pass"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--distribution", choices=("gaussian", "anisotropic", "offset_1e4", "clusters"), action="append")
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--reference-archive", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/knn-fresh-vs-v3-2026-10-29.json")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
