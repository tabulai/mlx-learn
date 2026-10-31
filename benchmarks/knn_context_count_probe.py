"""Hold one KNN worker fixed while adding/removing idle MLX GPU processes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from .knn_pipeline import CASES, ROOT, Worker, make_fixture


def main():
    case = CASES[-1]
    request_case = asdict(case)
    result = {
        "case": request_case,
        "dataset_seed": 51,
        "policy": "Same primary process and model across all phases. Added workers use identical current source, run cold query plus 5 warm queries, then block on stdin. No sklearn comparison worker or altered thread/wait/power policy.",
        "phases": [],
        "workers": [],
    }
    with tempfile.TemporaryDirectory(prefix="knn-context-count-") as temp:
        directory = Path(temp)
        result["fixture"] = make_fixture(case, 51, directory)
        primary = Worker("current", ROOT / "src", directory)
        result["workers"].append(primary.info)
        result["primary_cold"] = primary.request({"op": "cold", "case": request_case})
        extras = []

        def phase(label):
            row = {
                "label": label,
                "primary_pid": primary.process.pid,
                "worker_pids": [primary.process.pid] + [w.process.pid for w in extras],
            }
            for operation in ("kneighbors", "fit_query"):
                row[operation] = primary.request(
                    {
                        "op": "block",
                        "operation": operation,
                        "case": request_case,
                        "warmup": 5,
                        "repeats": 25,
                    }
                )
            row["process_snapshot"] = subprocess.check_output(
                ["ps", "-p", ",".join(map(str, row["worker_pids"])), "-o", "pid,pcpu,time,state"],
                text=True,
            )
            result["phases"].append(row)
            print(
                label,
                {op: round(row[op]["median_ms"], 3) for op in ("kneighbors", "fit_query")},
                flush=True,
            )

        def add_worker():
            w = Worker("current", ROOT / "src", directory)
            result["workers"].append(w.info)
            w.request({"op": "cold", "case": request_case})
            w.request(
                {
                    "op": "block",
                    "operation": "kneighbors",
                    "case": request_case,
                    "warmup": 0,
                    "repeats": 5,
                }
            )
            extras.append(w)

        try:
            for cycle in range(2):
                phase(f"cycle{cycle}_one")
                add_worker()
                phase(f"cycle{cycle}_two")
                add_worker()
                phase(f"cycle{cycle}_three")
                time.sleep(1)
                phase(f"cycle{cycle}_three_after_1s_idle")
                while extras:
                    extras.pop().close()
                phase(f"cycle{cycle}_one_restored")
            result["source_integrity_pass"] = (
                primary.request({"op": "integrity"}) == primary.info["source_hashes"]
            )
        finally:
            for w in extras:
                w.close()
            primary.close()
    result["quality_pass"] = all(
        p[op]["quality_pass"] and all(a["pass"] for a in p[op]["warmup_audits"])
        for p in result["phases"]
        for op in ("kneighbors", "fit_query")
    )
    result["identical_source_hashes"] = all(
        w["source_hashes"] == result["workers"][0]["source_hashes"] for w in result["workers"]
    )
    output = ROOT / "benchmarks/results/knn-context-count-probe-2026-10-28.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


def idle_ramp_probe():
    """Isolate idle time with one unchanged worker and chronologically retained calls."""
    case = CASES[-1]
    with tempfile.TemporaryDirectory(prefix="knn-idle-ramp-") as temp:
        directory = Path(temp)
        fixture = make_fixture(case, 51, directory)
        worker = Worker("current", ROOT / "src", directory)
        base = {"case": asdict(case)}
        result = {"fixture": fixture, "worker": worker.info, "phases": []}
        try:
            worker.request({"op": "cold", **base})
            for label, delay in (
                ("initial", 0),
                ("idle1", 1),
                ("active1", 0),
                ("idle2", 1),
                ("active2", 0),
            ):
                time.sleep(delay)
                measurement = worker.request(
                    {
                        "op": "block",
                        "operation": "kneighbors",
                        "warmup": 0,
                        "repeats": 200,
                        **base,
                    }
                )
                values = [sample["ms"] for sample in measurement["samples"]]
                measurement["windows20"] = [
                    statistics.median(values[start : start + 20]) for start in range(0, 200, 20)
                ]
                result["phases"].append(
                    {
                        "label": label,
                        "idle_seconds": delay,
                        "measurement": measurement,
                    }
                )
                print(label, [round(value, 3) for value in measurement["windows20"]], flush=True)
            result["source_integrity_pass"] = (
                worker.request({"op": "integrity"}) == worker.info["source_hashes"]
            )
        finally:
            worker.close()
    output = ROOT / "benchmarks/results/knn-idle-ramp-probe-2026-10-28.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--idle-ramp",
        action="store_true",
        help="Run the single-worker idle control instead of varying context count.",
    )
    options = parser.parse_args()
    idle_ramp_probe() if options.idle_ramp else main()
