#!/usr/bin/env python3
"""Capture historical correctness and benchmark baselines without failing fast."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "phase0" / "baseline"
PACKAGES = ("numpy", "scipy", "scikit-learn", "mlx", "pytest", "flake8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_log(path: Path) -> None:
    """Keep captured output readable while avoiding platform whitespace churn."""
    text = path.read_text(encoding="utf-8")
    normalized = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(normalized, encoding="utf-8")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def apple_hardware() -> dict[str, str]:
    if platform.system() != "Darwin":
        return {}
    keys = {
        "Chip": "chip",
        "Model Name": "model_name",
        "Model Identifier": "model_identifier",
        "Total Number of Cores": "cpu_cores",
        "Memory": "memory",
    }
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    values: dict[str, str] = {}
    for raw_line in result.stdout.splitlines():
        if ":" not in raw_line:
            continue
        key, value = (part.strip() for part in raw_line.split(":", 1))
        if key in keys:
            values[keys[key]] = value
    return values


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def run_and_log(
    *,
    name: str,
    command: list[str],
    cwd: Path,
    log_path: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "exit_code": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "log": log_path.name,
        "log_sha256": sha256(log_path),
    }


def benchmark_counts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    algorithms_summary = {}
    for profile, algorithms in data.get("profiles", {}).items():
        for algorithm, results in algorithms.items():
            cells.extend(results)
            speedups = [
                float(cell["speedup"])
                for cell in results
                if isinstance(cell.get("speedup"), (int, float))
            ]
            algorithms_summary[f"{profile}/{algorithm}"] = {
                "cells": len(results),
                "correct": sum(bool(cell.get("correct")) for cell in results),
                "speedup_min": min(speedups) if speedups else None,
                "speedup_max": max(speedups) if speedups else None,
            }
    return {
        "cells": len(cells),
        "correct": sum(bool(cell.get("correct")) for cell in cells),
        "incorrect": sum(not bool(cell.get("correct")) for cell in cells),
        "errors": sum(bool(cell.get("error")) for cell in cells),
        "algorithms": algorithms_summary,
    }


def pytest_details(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    failures = re.findall(r"^FAILED (.+)$", text, flags=re.MULTILINE)
    summary_matches = re.findall(
        r"=+\s+(.+?\b(?:passed|failed).+?)\s+=+\s*$",
        text,
        flags=re.MULTILINE,
    )
    return {
        "summary": summary_matches[-1] if summary_matches else "",
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--skip-tests", action="store_true", help="Record environment and benchmark only."
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true", help="Record environment and tests only."
    )
    parser.add_argument(
        "--require-green",
        action="store_true",
        help="Return nonzero when a historical command fails.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = output_dir / "benchmark_results.json"
    previous_manifest_path = output_dir / "manifest.json"
    previous_commands = []
    if previous_manifest_path.exists():
        previous_commands = json.loads(
            previous_manifest_path.read_text(encoding="utf-8")
        ).get("commands", [])

    environment_data = {
        "schema_version": 1,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
            "python_executable": args.python,
        },
        "hardware": apple_hardware(),
        "packages": package_versions(),
        "source": {
            "path": str(source),
            "revision": git(source, "rev-parse", "HEAD"),
            "working_tree_clean": not bool(git(source, "status", "--porcelain")),
        },
    }
    environment_path = output_dir / "environment.json"
    environment_path.write_text(
        json.dumps(environment_data, indent=2) + "\n", encoding="utf-8"
    )

    child_environment = dict(os.environ)
    child_environment.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    commands = []
    if args.skip_tests:
        commands.extend(
            command
            for command in previous_commands
            if command.get("name") == "historical-pytest"
            and (output_dir / command.get("log", "")).is_file()
        )
    else:
        commands.append(
            run_and_log(
                name="historical-pytest",
                command=[args.python, "-m", "pytest"],
                cwd=source,
                log_path=output_dir / "test.log",
                environment=child_environment,
            )
        )
    if args.skip_benchmark:
        commands.extend(
            command
            for command in previous_commands
            if command.get("name") == "historical-fidelity-benchmark"
            and (output_dir / command.get("log", "")).is_file()
            and benchmark_path.is_file()
        )
    else:
        commands.append(
            run_and_log(
                name="historical-fidelity-benchmark",
                command=[
                    args.python,
                    "-m",
                    "benchmarks.bench_runner",
                    "--warmup-iters",
                    "1",
                    "--measure-iters",
                    "1",
                    "--profiles",
                    "fidelity",
                    "--output",
                    str(benchmark_path),
                ],
                cwd=source,
                log_path=output_dir / "benchmark.log",
                environment=child_environment,
            )
        )

    for command in commands:
        log_path = output_dir / command["log"]
        normalize_log(log_path)
        command["log_sha256"] = sha256(log_path)

    artifacts = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "summary.md"}:
            artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    counts = benchmark_counts(benchmark_path)
    test_details = pytest_details(output_dir / "test.log")
    manifest = {
        "schema_version": 1,
        "source_revision": environment_data["source"]["revision"],
        "environment_sha256": sha256(environment_path),
        "commands": commands,
        "pytest": test_details,
        "benchmark_counts": counts,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    summary = [
        "# Historical baseline summary",
        "",
        f"- Source revision: `{manifest['source_revision']}`",
        f"- Hardware: `{environment_data['hardware'].get('chip', 'unknown')}`",
        f"- Python: `{platform.python_version()}`",
        f"- scikit-learn: `{environment_data['packages']['scikit-learn']}`",
        f"- MLX: `{environment_data['packages']['mlx']}`",
        "",
        "## Commands",
        "",
    ]
    for command in commands:
        state = "PASS" if command["exit_code"] == 0 else "FAIL"
        summary.append(
            f"- **{state}** `{command['name']}` — exit {command['exit_code']}, "
            f"{command['duration_seconds']:.3f}s; see `{command['log']}`."
        )
    if test_details:
        summary.extend(["", "## Correctness tests", ""])
        if test_details["summary"]:
            summary.append(f"- `{test_details['summary']}`")
        for failure in test_details["failures"]:
            summary.append(f"- Historical failure: `{failure}`")
    if counts:
        summary.extend(
            [
                "",
                "## Fidelity benchmark",
                "",
                f"- Cells: {counts['cells']}",
                f"- Correct: {counts['correct']}",
                f"- Incorrect: {counts['incorrect']}",
                f"- Runtime errors: {counts['errors']}",
                "",
                "### Observed speedup ranges",
                "",
            ]
        )
        for name, details in counts["algorithms"].items():
            low = details["speedup_min"]
            high = details["speedup_max"]
            speedup = (
                f"{low:.3f}×–{high:.3f}×"
                if isinstance(low, (int, float)) and isinstance(high, (int, float))
                else "not measured"
            )
            summary.append(
                f"- `{name}`: {details['correct']}/{details['cells']} correct; "
                f"{speedup}"
            )
        summary.extend(
            [
                "",
                "These are one-measure historical fidelity-smoke timings, not final "
                "per-operation performance gates or crossover thresholds.",
            ]
        )
    summary.extend(
        [
            "",
            "A failing historical command is evidence to document, not a reason to "
            "rewrite the baseline. Re-run with `--require-green` only when using this "
            "as a strict regression gate.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    failed = any(command["exit_code"] != 0 for command in commands)
    print(f"Wrote historical baseline artifacts to {output_dir}.")
    for command in commands:
        print(
            f"{command['name']}: exit={command['exit_code']} "
            f"duration={command['duration_seconds']}s"
        )
    return 1 if args.require_green and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
