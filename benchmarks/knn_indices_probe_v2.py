"""Three fixed paired blocks diagnose indices-only latency without replacing benchmarks."""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from pathlib import Path

from .knn_pipeline import CASES, ROOT, make_fixture, source_hashes


def main():
    import mlx.core as mx
    import numpy as np

    from mlxlearn._common.config import tuning_context
    from mlxlearn.neighbors import NearestNeighbors

    result = {
        "dataset_seed": 52,
        "source_hashes": source_hashes(ROOT / "src"),
        "policy": "Same public model/input; three predeclared blocks, each one second continuous audited warmup then25 paired calls of distances+indices and indices-only. Pair order alternates. No sklearn speed comparison; does not replace any failed benchmark sample.",
        "blocks": [],
    }
    with tempfile.TemporaryDirectory(prefix="knn-indices-v2-") as temp:
        fixture = make_fixture(CASES[-1], 52, Path(temp))
        result["fixture"] = fixture
        data = dict(np.load(Path(temp) / f"{CASES[-1].name}.npz"))
        with tuning_context(force_backend="mlx"):
            model = NearestNeighbors(n_neighbors=5, algorithm="brute").fit(data["X"])

            def call(distances):
                mx.synchronize()
                start = time.perf_counter_ns()
                value = model.kneighbors(data["Q"], return_distance=distances)
                mx.synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                indices = value[1] if distances else value
                audit = {"indices_match": bool(np.array_equal(indices, data["indices"]))}
                if distances:
                    audit["distance_match"] = bool(
                        np.allclose(value[0], data["distances"], atol=1e-6, rtol=1e-6)
                    )
                audit["pass"] = all(audit.values())
                return {"ms": elapsed, "audit": audit, "backend": model.execution_backend_}

            for block in range(3):
                warm = []
                started = time.perf_counter()
                while time.perf_counter() - started < 1:
                    warm.append(call(bool(len(warm) % 2))["audit"])
                elapsed = time.perf_counter() - started
                samples = {"kneighbors": [], "indices": []}
                for rep in range(25):
                    order = (True, False) if (rep + block) % 2 == 0 else (False, True)
                    for distance in order:
                        samples["kneighbors" if distance else "indices"].append(call(distance))
                row = {
                    "block": block,
                    "warmup_count": len(warm),
                    "warmup_elapsed_seconds": elapsed,
                    "warmup_audits": warm,
                    "samples": samples,
                    "median_ms": {
                        name: statistics.median(v["ms"] for v in values)
                        for name, values in samples.items()
                    },
                }
                row["indices_over_distance_ratio"] = (
                    row["median_ms"]["indices"] / row["median_ms"]["kneighbors"]
                )
                result["blocks"].append(row)
                print(
                    "block",
                    block,
                    row["median_ms"],
                    "ratio",
                    row["indices_over_distance_ratio"],
                    flush=True,
                )
    result["quality_pass"] = all(
        a["pass"] for b in result["blocks"] for a in b["warmup_audits"]
    ) and all(
        s["audit"]["pass"] and s["backend"] == "mlx"
        for b in result["blocks"]
        for values in b["samples"].values()
        for s in values
    )
    result["source_integrity_pass"] = source_hashes(ROOT / "src") == result["source_hashes"]
    output = ROOT / "benchmarks/results/knn-indices-probe-v2-2026-10-28.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
