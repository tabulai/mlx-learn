"""Compare two frozen revisions of the exact kNN planner at identical workspace budgets.

Run python -m benchmarks.knn_planner_budgets --baseline <main-revision>
--candidate <candidate-revision> --output report.json. Native extensions are
omitted from both snapshots. All queries explicitly run on the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import tempfile
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from benchmarks import knn_integration as bench

MIB = 1 << 20
BASELINE = "6d871ccd93bb8e6278e2683e8d449084c0c33c03"
ROUNDING_GUARD = bench.Case("rounding_2m_k10", 2**21 + 1, 40, 32, 10, "rounding_guard")
CASES = (*bench.CASES, bench.Case("tied_1m_k10", 1048576, 96, 128, 10, "integer"),
         bench.Case("odd_1m_k17", 1048583, 101, 257, 17),
         bench.Case("gaussian_1m_k5", 1048576, 96, 512, 5), ROUNDING_GUARD)
DEFAULT_CASES = ("gaussian_1m_k10", "gaussian_1m_k32", "gaussian_256k_k10",
                 "tied_1m_k10", "odd_1m_k17", "small_certified", "gaussian_1m_k5")
LABELS = ("main", "candidate")


def environment(budget_mib):
    # Start both workers with identical planner settings, including on smaller hosts.
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLXLEARN_")}
    for suffix in ("DISTANCE_BLOCK_BYTES", "DISTANCE_BLOCK_BYTES_MAX"):
        env[f"MLXLEARN_{suffix}"] = str(budget_mib * MIB)
    return env


def measure(workers, case, api, budget, args, rng):
    seed = args.seed + CASES.index(case)
    row = {"case": asdict(case), "fixture_seed": seed, "api": api, "budget_mib": budget,
           "variants": {}, "timed_order": [], "passed": False}
    outputs = []
    try:
        for label in LABELS:
            info = row["variants"][label] = {"warmup": [], "samples": []}
            prepare = workers[label].request(
                {"op": "prepare", "case": asdict(case), "api": api, "seed": seed, "oracle_rows": 3}
            )
            info["preparation"] = {"fixture_sha256": prepare["fixture_sha256"],
                                   "input_bytes": prepare["input_bytes"]}
            for i in range(args.warmup):
                sample = workers[label].request({"op": "query", "probe_plan": i == 0,
                                                 "describe_outputs": True})
                if i == 0:
                    info["plan"] = sample.pop("plan")
                    if info["plan"]["budget_bytes"] != budget * MIB:
                        raise RuntimeError(f"{label}: worker did not apply requested budget")
                outputs.append(sample.pop("outputs"))
                info["warmup"].append(sample)
        order = list(LABELS)
        rng.shuffle(order)
        for _ in range(args.repeats):
            row["timed_order"].append(order.copy())
            for label in order:
                sample = workers[label].request({"op": "query", "describe_outputs": True})
                outputs.append(sample.pop("outputs"))
                row["variants"][label]["samples"].append(sample)
            order.reverse()
        exact = bench.summarize_case(row["variants"], "main")
        row["output_dtypes_and_shapes"] = outputs[0]
        expected_shape = [case.query_rows, case.k]
        row["passed"] = (exact and all(description == outputs[0] for description in outputs)
                         and all(shape == expected_shape for _, shape in outputs[0]))
        for info in row["variants"].values():
            # Use maxima over measured repetitions for high-water marks. Warmups
            # include index preparation and first compilation and stay separate.
            for key in ("peak_active_bytes", "peak_increment_bytes"):
                info[key] = max(sample["mlx_memory"][key] for sample in info["samples"])
        if row["passed"]:
            row["paired_speedup_median"] = statistics.median(
                a["ms"] / b["ms"] for a, b in zip(row["variants"]["main"]["samples"],
                                                   row["variants"]["candidate"]["samples"], strict=True)
            )
    except Exception as exc:
        row["error"] = str(exc)
    return row


def save(path, report):
    pending = path.with_suffix(path.suffix + ".partial")
    pending.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    pending.replace(path)


def run(args):
    report = {
        "schema": 1, "warmup": args.warmup, "repeats": args.repeats, "seed": args.seed,
        "metric": "Synchronized warm query ms; peak active MLX bytes and peak minus active bytes "
                  "at query start. Peak summaries use maxima over measured repetitions. "
                  "Device queries include mx.eval; NumPy queries include host output conversion.",
        "audit": "Compare SHA-256 over every distance bit and every index value between "
                 "implementations, including warmups. Match input hashes and output dtypes/shapes. "
                 "Also check the first three rows against a direct float64 oracle.",
        "driver_sha256": bench.digest(Path(__file__).read_bytes()),
        "shared_driver_sha256": bench.digest(Path(bench.__file__).read_bytes()),
        "sources": {}, "environments": {}, "source_integrity": {}, "cases": [], "passed": False,
    }
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="knn-plan-") as temp:
        sources = {label: Path(temp) / label for label in LABELS}
        for label, revision in zip(LABELS, (args.baseline, args.candidate), strict=True):
            meta = bench.snapshot(bench.ROOT, revision, sources[label])
            report["sources"][label] = {"commit": meta["commit"], "sha256": meta["sha256"]}
        try:
            selected = [c for c in CASES if c.name in args.cases]
            panels = {budget: selected.copy() for budget in args.budgets}
            if args.include_64_controls:
                large = [c for c in selected if c.name in ("gaussian_1m_k10", "gaussian_1m_k32")]
                at_64 = panels.setdefault(64, large)
                if ROUNDING_GUARD not in at_64:
                    at_64.append(ROUNDING_GUARD)
            for budget, cases in panels.items():
                with ExitStack() as stack:
                    workers = {}
                    for label, source in sources.items():
                        w = bench.Worker(source, args.timeout, environment=environment(budget))
                        stack.callback(w.close)
                        workers[label] = w
                        report["environments"][label] = w.request({"op": "info"})
                    for case in cases:
                        for api in args.apis:
                            row = measure(workers, case, api, budget, args, rng)
                            report["cases"].append(row)
                            save(args.output, report)
                            print(budget, case.name, api, "PASS" if row["passed"] else "FAIL",
                                  {label: round(info["median_ms"], 3) for label, info
                                   in row["variants"].items() if "median_ms" in info}, flush=True)
                    report["source_integrity"][str(budget)] = {}
                    for label, w in workers.items():
                        checked = w.request({"op": "integrity"})
                        hashed = bench.digest(json.dumps(checked["files"], sort_keys=True).encode())
                        report["source_integrity"][str(budget)][label] = {
                            "passed": hashed == report["sources"][label]["sha256"],
                            "loaded_modules": checked["loaded"],
                        }
            report["passed"] = (all(c["passed"] for c in report["cases"]) and
                                all(v["passed"] for variants in report["source_integrity"].values()
                                    for v in variants.values()))
        except Exception as exc:
            report["error"] = str(exc)
            raise
        finally:
            save(args.output, report)
    return report["passed"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--candidate", default="HEAD")
    ap.add_argument("--cases", nargs="+", choices=[c.name for c in CASES], default=DEFAULT_CASES)
    ap.add_argument("--budgets", nargs="+", type=int, default=(128, 256, 512, 1024), help="MiB")
    ap.add_argument("--include-64-controls", action="store_true",
                    help="Add selected million-row Gaussian k10/k32 and the rounding control at 64 MiB")
    ap.add_argument("--apis", nargs="+", choices=("device", "numpy"), default=("device", "numpy"))
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if min(*args.budgets, args.repeats, args.warmup, args.timeout) <= 0:
        ap.error("Budgets, repeat counts, and timeout must be positive.")
    if args.output.exists():
        ap.error("Output already exists. Choose a new path for this campaign.")
    if not run(args):
        raise SystemExit("One or more comparison audits failed.")


if __name__ == "__main__":
    main()
