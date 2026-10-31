"""Held-out distribution audit of current KNN against frozen v3 and sklearn.

    python -m benchmarks.knn_review_distributions \
        --output benchmarks/results/knn-review-distributions-2026-10-29.json

Uses the existing isolated public-API workers and their synchronized, audited
timing. This harness changes only the inputs and the independent accuracy oracle.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path

from benchmarks.knn_pipeline import (
    ROOT,
    Case,
    Worker,
    extract_reference,
    sha256,
)

DISTRIBUTIONS = ("gaussian", "clusters", "offset_1e4", "anisotropic")
VARIANTS = ("current", "reference", "sklearn_default")
OPERATIONS = ("fit_query", "kneighbors")
REFERENCE = ROOT / "benchmarks/reference/knn-pre-v4-2026-10-28.zip"


def harness_hashes():
    return {
        str(path.relative_to(ROOT)): sha256(path.read_bytes())
        for path in (Path(__file__).resolve(), ROOT / "benchmarks/knn_pipeline.py")
    }


def direct_reference(X, Q, k, *, query_block=4):
    """Search every original float32 point in direct float64 arithmetic.

    No Gram products or approximate candidate shortlist participate in the
    oracle. A full lexicographic sort resolves squared-distance ties by index.
    Query blocking bounds the largest float64 difference tensor at about 49 MiB
    for the standard shape; the squared temporary uses the same amount again.
    """
    import numpy as np

    train = X.astype(np.float64)
    indices = np.empty((len(Q), k), dtype=np.int64)
    distances = np.empty((len(Q), k), dtype=np.float64)
    ids = np.arange(len(X), dtype=np.int64)
    for start in range(0, len(Q), query_block):
        query = Q[start:start + query_block].astype(np.float64)
        delta = query[:, None, :] - train[None, :, :]
        squared = np.sum(delta * delta, axis=2, dtype=np.float64)
        order = np.lexsort((np.broadcast_to(ids, squared.shape), squared), axis=1)[:, :k]
        indices[start:start + len(query)] = order
        distances[start:start + len(query)] = np.sqrt(np.take_along_axis(squared, order, axis=1))
    return distances, indices


def make_fixture(distribution, seed, destination):
    import numpy as np

    case = Case(f"{distribution}_50000_32_k5", 50_000, 256, 32, 5)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(case.n_train, case.n_features))
    Q = rng.normal(size=(case.n_query, case.n_features))
    parameters = {"base_distribution": "independent standard normal"}
    if distribution == "clusters":
        centers = rng.normal(size=(16, case.n_features)) * 8.0
        train_groups = rng.integers(0, len(centers), size=case.n_train)
        query_groups = rng.integers(0, len(centers), size=case.n_query)
        X = centers[train_groups] + 0.2 * X
        Q = centers[query_groups] + 0.2 * Q
        parameters.update(clusters=16, center_std=8.0, within_cluster_std=0.2)
    elif distribution == "offset_1e4":
        X += 1e4
        Q += 1e4
        parameters["common_offset"] = 1e4
    elif distribution == "anisotropic":
        scales = np.logspace(-2, 2, case.n_features)
        X *= scales
        Q *= scales
        parameters["feature_scales"] = scales.tolist()
    elif distribution != "gaussian":
        raise ValueError(f"Unknown distribution: {distribution}")
    X, Q = X.astype(np.float32), Q.astype(np.float32)
    Q[:2] = X[:2]
    distances, indices = direct_reference(X, Q, case.k)
    path = destination / f"{case.name}.npz"
    np.savez(path, X=X, Q=Q, distances=distances, indices=indices)
    return case, {
        "case": asdict(case),
        "distribution": distribution,
        "seed": seed,
        "parameters": parameters,
        "copied_training_queries": 2,
        "fixture_sha256": sha256(path.read_bytes()),
        "array_sha256": {
            name: sha256(np.ascontiguousarray(array).tobytes())
            for name, array in {"X": X, "Q": Q, "distances": distances, "indices": indices}.items()
        },
        "input_dtype": str(X.dtype),
        "train_min": float(X.min()),
        "train_max": float(X.max()),
    }


def audited_request(worker, request):
    """Keep worker exceptions visible without dropping the remaining panel."""
    try:
        return worker.request(request)
    except Exception:
        return {"error": traceback.format_exc(), "audit": {"pass": False}}


def summarize(samples, warmup_samples):
    values = [sample["ms"] for sample in samples if "ms" in sample]
    return {
        "median_ms": statistics.median(values) if values else None,
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "completed_samples": len(values),
        "samples": samples,
        "warmup_samples": warmup_samples,
        "quality_pass": all(sample["audit"]["pass"] for sample in samples),
        "warmup_quality_pass": all(sample["audit"]["pass"] for sample in warmup_samples),
    }


def write_report(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


def certificate_diagnostics(cases, fixtures, report):
    """Explain fallback rates only after every public timing has completed.

    The compact helper is a valid diagnostic for the frozen implementation only
    when its actual source hash matches the archive. The comparison records that
    check explicitly, independently of numerical agreement on these fixtures.
    """
    diagnostic = {
        "timing_policy": "Untimed; run after all cold, warmup and retained public-API calls.",
        "cases": [],
    }
    try:
        # Importing MLX here for the first time avoids creating an additional
        # parent GPU context while the isolated workers collect timings.
        import mlx.core as mx
        import numpy as np

        sys.path.insert(0, str(ROOT / "src"))
        import mlxlearn_device
        from mlxlearn_device import _neighbors_compact as compact
        from mlxlearn_device import _neighbors_integer as integer

        source_root = (ROOT / "src").resolve()
        for module in (mlxlearn_device, compact, integer):
            if not Path(module.__file__).resolve().is_relative_to(source_root):
                raise RuntimeError(f"Wrong diagnostic package imported: {module.__file__}")
        diagnostic["module_paths"] = {
            module.__name__: str(Path(module.__file__).resolve())
            for module in (mlxlearn_device, compact, integer)
        }
        compact_key = "mlxlearn_device/_neighbors_compact.py"
        diagnostic["compact_source_hashes"] = {
            "loaded_current": sha256(Path(compact.__file__).read_bytes()),
            "current_worker_before": report["workers"]["current"]["source_hashes"].get(compact_key),
            "reference_manifest": report["reference"]["files"].get(compact_key),
        }
        compact_hashes = list(diagnostic["compact_source_hashes"].values())
        diagnostic["compact_matches_frozen_reference"] = (
            all(compact_hashes) and len(set(compact_hashes)) == 1
        )
        diagnostic["integer_source_sha256"] = sha256(Path(integer.__file__).read_bytes())
        diagnostic["integer_supported"] = bool(integer.is_supported())

        def audited_output(squared, indices, certificate, expected):
            mx.eval(squared, indices, certificate)
            squared = np.asarray(squared)
            indices = np.asarray(indices)
            certificate = np.asarray(certificate)
            distances = np.sqrt(squared.astype(np.float64))
            index_match = np.array_equal(indices, expected["indices"])
            distance_match = np.allclose(distances, expected["distances"], atol=1e-6, rtol=1e-6)
            count = int(np.count_nonzero(certificate))
            return {
                "certified_count": count,
                "query_count": len(certificate),
                "certified_fraction": count / len(certificate),
                "fallback_count": len(certificate) - count,
                "oracle_audit": {
                    "pass": bool(index_match and distance_match),
                    "indices_match": bool(index_match),
                    "distances_exact_match": bool(np.array_equal(distances, expected["distances"])),
                    "distances_match": bool(distance_match),
                    "distance_max_abs": float(np.max(np.abs(distances - expected["distances"]))),
                },
            }, squared, indices

        for case in cases:
            with np.load(fixtures / f"{case.name}.npz") as loaded:
                data = {name: loaded[name] for name in ("X", "Q", "distances", "indices")}
            train, queries = mx.array(data["X"]), mx.array(data["Q"])
            entry = {"case": asdict(case), "algorithms": {}}
            outputs = {}
            for name, module in (("compact", compact), ("integer", integer)):
                if name == "integer" and not diagnostic["integer_supported"]:
                    entry["algorithms"][name] = {"status": "unsupported"}
                    continue
                try:
                    prepare = getattr(module, f"prepare_{name}")
                    search = getattr(module, f"{name}_neighbors")
                    with mx.stream(mx.gpu):
                        result = search(train, queries, case.k, prepare(train), return_certificate=True)
                        summary, squared, indices = audited_output(*result, data)
                    entry["algorithms"][name] = {"status": "complete", **summary}
                    outputs[name] = (squared, indices)
                except Exception:
                    entry["algorithms"][name] = {"status": "error", "error": traceback.format_exc()}
            if len(outputs) == 2:
                compact_squared, compact_indices = outputs["compact"]
                integer_squared, integer_indices = outputs["integer"]
                entry["integer_vs_compact"] = {
                    "indices_exact_match": bool(np.array_equal(integer_indices, compact_indices)),
                    "squared_distances_exact_match": bool(np.array_equal(integer_squared, compact_squared)),
                    "squared_distance_max_abs": float(np.max(np.abs(
                        integer_squared.astype(np.float64) - compact_squared.astype(np.float64)
                    ))),
                }
            diagnostic["cases"].append(entry)
        diagnostic["quality_pass"] = all(
            all(value["status"] == "unsupported" or (
                value["status"] == "complete" and value["oracle_audit"]["pass"]
            ) for value in case["algorithms"].values())
            and all(value for key, value in case.get("integer_vs_compact", {}).items()
                    if key.endswith("exact_match"))
            for case in diagnostic["cases"]
        )
        diagnostic["status"] = "complete"
    except Exception:
        diagnostic["status"] = "error"
        diagnostic["error"] = traceback.format_exc()
    return diagnostic


def run(args):
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    selected = args.distribution or list(DISTRIBUTIONS)
    report = {
        "schema_version": 1,
        "status": "running",
        "seed": args.seed,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "platform": platform.platform(),
        "harness_hashes_before": harness_hashes(),
        "operations": list(OPERATIONS),
        "reference_label": "Frozen pre-v4 archive: accepted v3 compact-search implementation",
        "timing_policy": (
            "Persistent isolated public-API workers, alternating variant order per round; "
            "synchronized NumPy-to-NumPy calls; every call audited outside its timer. "
            "Three fixed-count warmups and nine measured samples by default; no duration warmup."
        ),
        "fit_query_policy": "Fresh estimator fit plus first query, including deferred preparation.",
        "cold_policy": (
            "First fit/query for each shape/distribution in its persistent worker after imports; "
            "prior distributions may populate compilation caches. Recorded separately from retained samples."
        ),
        "accuracy_reference": (
            "Exhaustive direct float64 subtraction and squared-sum distance over every training row "
            "on the serialized float32 inputs, with full (distance,index) lexicographic ordering. "
            "Query blocks of four bound temporary memory; no Gram selection is used."
        ),
        "query_tolerances": {"atol": 1e-6, "rtol": 1e-6, "indices": "exact equality"},
        "fixtures": [],
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-distributions-") as temporary:
        temporary = Path(temporary)
        reference, fixtures = temporary / "reference", temporary / "fixtures"
        report["reference"] = extract_reference(args.reference_archive, reference)
        fixtures.mkdir()
        cases = []
        for distribution in selected:
            case, info = make_fixture(distribution, args.seed, fixtures)
            cases.append(case)
            report["fixtures"].append(info)
        workers = {}
        try:
            for name in VARIANTS:
                workers[name] = Worker(name, reference if name == "reference" else ROOT / "src", fixtures)
            report["workers"] = {name: worker.info for name, worker in workers.items()}
            for case in cases:
                entry = {"case": asdict(case), "variants": {name: {} for name in VARIANTS}}
                for name in VARIANTS:
                    entry["variants"][name]["cold"] = audited_request(
                        workers[name], {"op": "cold", "case": asdict(case)},
                    )
                for operation in OPERATIONS:
                    samples, warmups, rounds = ({name: [] for name in VARIANTS},
                                                 {name: [] for name in VARIANTS}, [])
                    for repeat in range(args.warmup + args.repeats):
                        order = list(VARIANTS if repeat % 2 == 0 else reversed(VARIANTS))
                        rounds.append({"round": repeat, "warmup": repeat < args.warmup, "order": order})
                        for name in order:
                            sample = audited_request(workers[name], {"op": operation, "case": asdict(case)})
                            (warmups if repeat < args.warmup else samples)[name].append(sample)
                    entry.setdefault("rounds", {})[operation] = rounds
                    for name in VARIANTS:
                        entry["variants"][name][operation] = summarize(samples[name], warmups[name])
                entry["speedups"] = {}
                for operation in OPERATIONS:
                    current = entry["variants"]["current"][operation]["median_ms"]
                    entry["speedups"][operation] = {
                        f"vs_{name}": (
                            entry["variants"][name][operation]["median_ms"] / current
                            if current and entry["variants"][name][operation]["median_ms"] is not None else None
                        )
                        for name in ("reference", "sklearn_default")
                    }
                report["cases"].append(entry)
                write_report(args.output, report)
                print(f"{case.name}: {json.dumps(entry['speedups'])}", flush=True)
            report["certificate_diagnostics"] = certificate_diagnostics(cases, fixtures, report)
            report["source_hashes_after"] = {
                name: audited_request(worker, {"op": "integrity"}) for name, worker in workers.items()
            }
            report["source_integrity_pass"] = all(
                report["source_hashes_after"][name] == worker.info["source_hashes"]
                for name, worker in workers.items()
            )
            report["harness_hashes_after"] = harness_hashes()
            report["harness_integrity_pass"] = report["harness_hashes_before"] == report["harness_hashes_after"]
            report["quality_by_variant"] = {
                name: all(
                    case["variants"][name]["cold"]["audit"]["pass"]
                    and all(case["variants"][name][operation]["quality_pass"]
                            and case["variants"][name][operation]["warmup_quality_pass"]
                            for operation in OPERATIONS)
                    for case in report["cases"]
                ) for name in VARIANTS
            }
            report["quality_pass"] = all(report["quality_by_variant"].values())
            report["implementation_quality_pass"] = all(
                report["quality_by_variant"][name] for name in ("current", "reference")
            )
            report["status"] = "complete"
        except Exception:
            report["status"] = "error"
            report["error"] = traceback.format_exc()
            raise
        finally:
            for worker in workers.values():
                worker.close()
            write_report(args.output, report)
    print(f"quality_by_variant={report['quality_by_variant']}: {args.output}", flush=True)
    return 0 if all(report[key] for key in (
        "implementation_quality_pass", "source_integrity_pass", "harness_integrity_pass",
    )) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--distribution", choices=DISTRIBUTIONS, action="append")
    parser.add_argument("--reference-archive", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/knn-review-distributions-2026-10-29.json")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
