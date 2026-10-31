"""Compare the existing Gram pipeline with direct search without source edits."""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlxlearn._common.config import tuning_context
from mlxlearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from mlxlearn_device import _neighbors as core
from mlxlearn_device import fit_neighbors

from .knn_pipeline import CASES, ROOT, make_fixture, source_hashes


def main():
    case = CASES[-1]
    result = {
        "dataset_seed": 51, "warmup": 3, "repeats": 9,
        "policy": "One process, alternating direct/general order each round; synchronized calls; "
                  "all samples audited after timing; source untouched, dispatch monkeypatched only. "
                  "Resident core query excludes host boundary; public calls include it. Fresh "
                  "fit_query includes Gram preparation. This is a diagnostic, not the target audit.",
        "source_hashes": source_hashes(ROOT / "src"), "operations": {},
    }
    original = core._use_fused_search
    with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-general-") as folder:
        result["fixture"] = make_fixture(case, 51, Path(folder))
        data = dict(np.load(Path(folder) / f"{case.name}.npz"))
    X, Q = data["X"], data["Q"]
    Xd, Qd = mx.array(X), mx.array(Q)
    mx.eval(Xd, Qd)
    models = {}

    def audit(value, operation):
        if operation in ("core_query", "kneighbors", "fit_query"):
            distance, indices = (np.asarray(a) for a in value)
            return {"pass": bool(np.array_equal(indices, data["indices"])
                                 and np.allclose(distance, data["distances"], rtol=1e-6, atol=1e-6)),
                    "indices_exact": bool(np.array_equal(indices, data["indices"])),
                    "distance_max_abs": float(np.max(np.abs(distance - data["distances"])))}
        return {"pass": bool(np.allclose(value, data[operation], atol=2e-6, rtol=2e-5)),
                "max_abs": float(np.max(np.abs(value - data[operation])))}

    def call(variant, operation):
        core._use_fused_search = lambda *_: variant == "direct"
        model = models[variant]
        if operation == "core_query":
            return model["core"].query(Qd, 5, _validated=True)
        if operation == "kneighbors":
            return model["query"].kneighbors(Q)
        if operation == "fit_query":
            return NearestNeighbors(n_neighbors=5, algorithm="brute").fit(X).kneighbors(Q)
        kind, weighting = operation.split("_")
        estimator = model[operation]
        return estimator.predict_proba(Q) if kind == "proba" else estimator.predict(Q)

    try:
        with tuning_context(force_backend="mlx"):
            for variant in ("direct", "general"):
                models[variant] = {
                    "core": fit_neighbors(Xd, device="gpu"),
                    "query": NearestNeighbors(n_neighbors=5, algorithm="brute").fit(X),
                }
                for weighting in ("uniform", "distance"):
                    models[variant][f"proba_{weighting}"] = KNeighborsClassifier(
                        n_neighbors=5, weights=weighting, algorithm="brute"
                    ).fit(X, data["y"])
                    models[variant][f"regress_{weighting}"] = KNeighborsRegressor(
                        n_neighbors=5, weights=weighting, algorithm="brute"
                    ).fit(X, data["targets"])
            for operation in ("core_query", "kneighbors", "fit_query", "proba_uniform",
                              "regress_uniform", "proba_distance", "regress_distance"):
                entries = {variant: {"samples": [], "warmup_audits": []}
                           for variant in ("direct", "general")}
                for repeat in range(12):
                    for variant in (("direct", "general") if repeat % 2 == 0 else ("general", "direct")):
                        mx.synchronize()
                        start = time.perf_counter()
                        value = call(variant, operation)
                        if operation == "core_query":
                            mx.eval(*value)
                        mx.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000
                        check = audit(value, operation)
                        if repeat < 3:
                            entries[variant]["warmup_audits"].append(check)
                        else:
                            entries[variant]["samples"].append({"ms": elapsed, "audit": check})
                for entry in entries.values():
                    values = [sample["ms"] for sample in entry["samples"]]
                    entry.update({"median_ms": statistics.median(values), "min_ms": min(values),
                                  "max_ms": max(values), "quality_pass": (
                                      all(sample["audit"]["pass"] for sample in entry["samples"])
                                      and all(check["pass"] for check in entry["warmup_audits"])
                                  )})
                result["operations"][operation] = entries
                print(operation, {v: {k: round(e[k], 3) for k in ("median_ms", "min_ms", "max_ms")}
                                  for v, e in entries.items()}, flush=True)
    finally:
        core._use_fused_search = original
    result["source_integrity_pass"] = result["source_hashes"] == source_hashes(ROOT / "src")
    result["quality_pass"] = all(entry["quality_pass"] for variants in result["operations"].values()
                                 for entry in variants.values())
    output = ROOT / "benchmarks/results/knn-general-probe-2026-10-28.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("quality_pass=", result["quality_pass"], "source_integrity_pass=", result["source_integrity_pass"])


if __name__ == "__main__":
    main()
