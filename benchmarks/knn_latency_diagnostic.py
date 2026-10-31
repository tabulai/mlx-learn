"""Diagnose interference between persistent CPU/GPU KNN benchmark workers.

This is an experiment, not a replacement performance protocol. It never changes
the default sklearn thread count or OpenMP wait policy. Every measured call uses
the existing benchmark's timing and accuracy checks.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from .knn_pipeline import CASES, ROOT, Worker, make_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "benchmarks/results/knn-latency-diagnostic-2026-10-28.json")
    args = parser.parse_args()
    case = CASES[-1]
    assert case.name == "batch_50000_32_k5"
    request_case = asdict(case)
    report = {
        "case": request_case,
        "dataset_seed": 51,
        "warmup": 3,
        "repeats": 9,
        "policy": "Unmodified default sklearn threads/wait policy. Same audited timer as pipeline. "
                  "Quiescence before both compared variants is outside timers. Delay order rotates "
                  "between trials; current-only blocks precede CPU worker creation and follow closure.",
        "isolated_before": {}, "mixed": {}, "isolated_after": {},
    }

    def request(worker, operation):
        return worker.request({"op": operation, "case": request_case})

    def samples(worker, operation):
        values = [request(worker, operation) for _ in range(12)]
        result = {"warmup": values[:3], "samples": values[3:]}
        result["median_ms"] = statistics.median(v["ms"] for v in values[3:])
        result["quality_pass"] = all(v["audit"]["pass"] for v in values)
        return result

    with tempfile.TemporaryDirectory(prefix="mlxlearn-knn-latency-") as name:
        fixtures = Path(name)
        report["fixture"] = make_fixture(case, 51, fixtures)
        current = Worker("current", ROOT / "src", fixtures)
        sklearn = None
        try:
            report["current_worker"] = current.info
            report["current_cold"] = request(current, "cold")
            for operation in ("kneighbors", "fit_query", "proba_uniform"):
                report["isolated_before"][operation] = samples(current, operation)
            sklearn = Worker("sklearn_default", ROOT / "src", fixtures)
            report["sklearn_worker"] = sklearn.info
            report["sklearn_cold"] = request(sklearn, "cold")
            delays = (0, 20, 50, 200)
            for delay in delays:
                report["mixed"][str(delay)] = {"current": [], "sklearn_default": []}
            for trial in range(12):
                order = delays[trial % 4:] + delays[:trial % 4]
                for delay in order:
                    time.sleep(delay / 1000)
                    cpu = request(sklearn, "kneighbors")
                    time.sleep(delay / 1000)
                    gpu = request(current, "kneighbors")
                    if trial >= 3:
                        report["mixed"][str(delay)]["sklearn_default"].append(cpu)
                        report["mixed"][str(delay)]["current"].append(gpu)
            for entry in report["mixed"].values():
                entry["median_ms"] = {
                    variant: statistics.median(v["ms"] for v in entry[variant])
                    for variant in ("current", "sklearn_default")
                }
                entry["quality_pass"] = all(v["audit"]["pass"]
                                            for name in ("current", "sklearn_default")
                                            for v in entry[name])
            sklearn.close()
            sklearn = None
            for operation in ("kneighbors", "fit_query", "proba_uniform"):
                report["isolated_after"][operation] = samples(current, operation)
            report["source_integrity_pass"] = (
                current.request({"op": "integrity"}) == current.info["source_hashes"]
            )
        finally:
            current.close()
            if sklearn is not None:
                sklearn.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for phase in ("isolated_before", "isolated_after"):
        print(phase, {name: round(v["median_ms"], 3) for name, v in report[phase].items()})
    for delay, value in report["mixed"].items():
        print("equal quiescence", delay, "ms:", value["median_ms"])
    print(args.output)


if __name__ == "__main__":
    main()
