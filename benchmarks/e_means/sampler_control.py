"""Controlled fresh-process timing with/without external memory sampling.

Both conditions run the frozen native-BF16 runtime, using the unmodified H100
worker and timing boundaries. This diagnostic must run without other GPU jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap_bench as bench  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["vibe-agnews", "vibe-glove"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 11, 12])
    parser.add_argument("--out", type=Path, default=HERE / "results/sampler_control")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    commands = []
    for index, dataset in enumerate(args.datasets):
        for seed in args.seeds:
            order = [True, False] if (index + seed) % 2 == 0 else [False, True]
            for memory in order:
                command = [sys.executable, str(HERE / "gap_bench.py"), "--datasets", dataset,
                           "--modes", "previous", "--seeds", str(seed), "--out", str(args.out)]
                if not memory:
                    command.append("--no-memory")
                subprocess.run(command, check=True, cwd=bench.base.REPO)
                commands.append(command)
    groups = {}
    for path in args.out.glob("*.json"):
        row = json.loads(path.read_text())
        if row.get("kind") != bench.KIND:
            continue
        config = row["config"]
        if config["dataset"] not in args.datasets or config["seed"] not in args.seeds:
            continue
        if config["mode"] != "previous":
            raise RuntimeError("Sampler control must use the frozen previous runtime")
        with bench.adapter(config):
            bench.validate_result(row, config, row["memory_samples"])
        shared_config = {k: v for k, v in config.items() if k not in {"memory", "output", "worker_output"}}
        key = json.dumps({"config": shared_config, **{k: row[k] for k in
                         ["dataset", "initial_centroids_sha256", "recipe_resolved", "versions", "hardware"]}}, sort_keys=True)
        group = groups.setdefault(key, {})
        if config["memory"] in group:
            raise RuntimeError("Duplicate sampler-control condition")
        group[config["memory"]] = (row, path)
    pairs = []
    for group in groups.values():
        if set(group) != {True, False}:
            raise RuntimeError("Sampler control is missing its paired condition")
        enabled, disabled = group[True][0], group[False][0]
        pairs.append({"dataset": enabled["config"]["dataset"], "seed": enabled["config"]["seed"],
                      "enabled_seconds": enabled["train_seconds"], "disabled_seconds": disabled["train_seconds"],
                      "enabled_over_disabled": enabled["train_seconds"] / disabled["train_seconds"],
                      "same_centroids": enabled["final_centroids_sha256"] == disabled["final_centroids_sha256"],
                      "raw_evidence": {str(path.relative_to(bench.base.REPO)): bench.base.file_sha256(path)
                                       for _, path in group.values()}})
    if len(pairs) != len(args.datasets) * len(args.seeds):
        raise RuntimeError("Unexpected sampler-control coverage")
    summary = {dataset: {"median_enabled_seconds": median(p["enabled_seconds"] for p in pairs if p["dataset"] == dataset),
                         "median_disabled_seconds": median(p["disabled_seconds"] for p in pairs if p["dataset"] == dataset),
                         "median_enabled_over_disabled": median(p["enabled_over_disabled"] for p in pairs if p["dataset"] == dataset)}
               for dataset in args.datasets}
    bench.base.atomic_json(args.out / "summary.json", {
        "kind": "sampler_control", "commands": commands, "pairs": pairs, "summary": summary,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "Balanced-order fresh-process frozen-runtime fits; only memory sampler enabled/disabled changes. Small differences remain timing/run-state noise; three seed pairs are not a precise overhead estimate.",
    })
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
