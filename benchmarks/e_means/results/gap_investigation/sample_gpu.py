"""Bounded read-only telemetry using the already installed mactop executable."""

import argparse
import datetime
import hashlib
import json
import resource
import selectors
import shutil
import subprocess
import time
from pathlib import Path


def stamp():
    before = time.monotonic_ns()
    wall_ns = time.time_ns()
    after = time.monotonic_ns()
    return {"monotonic_ns": (before + after) // 2, "monotonic_bracket_ns": [before, after],
            "wall_time_ns": wall_ns,
            "wall_utc": datetime.datetime.fromtimestamp(wall_ns / 1e9, datetime.timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("gpu-telemetry-window.json"))
    args = parser.parse_args()
    if not 0 < args.duration <= 60 or args.interval < 500:
        parser.error("Use a duration in (0,60] seconds and interval >=500ms")
    binary = shutil.which("mactop")
    if not binary:
        raise RuntimeError("mactop is not installed; this script never installs software")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    command = [binary, "--headless", "--count", "0", "--interval", str(args.interval)]
    before_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = stamp()
    records, errors = [], []
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    selector = selectors.DefaultSelector()
    try:
        selector.register(proc.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + args.duration
        while proc.poll() is None and time.monotonic() < deadline:
            ready = selector.select(timeout=min(.25, max(0, deadline - time.monotonic())))
            if not ready:
                continue
            read_started = stamp()
            line = proc.stdout.readline()
            received = stamp()
            if not line:
                break
            try:
                raw = json.loads(line)
                soc = raw.get("soc_metrics", {})
                records.append({"source_timestamp": raw.get("timestamp"), "read_started": read_started, "received": received,
                                "gpu_freq_mhz": soc.get("gpu_freq_mhz"), "gpu_activity_percent": raw.get("gpu_usage"),
                                "gpu_power_watts": soc.get("gpu_power"), "gpu_sram_power_watts": soc.get("gpu_sram_power"),
                                "gpu_temperature_celsius": raw.get("gpu_temp"), "cpu_temperature_celsius": raw.get("cpu_temp"),
                                "thermal_state": raw.get("thermal_state")})
            except (json.JSONDecodeError, AttributeError) as error:
                errors.append(str(error))
    finally:
        selector.close()
        if proc.poll() is None:
            proc.terminate()
        try:
            _, stderr = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()
    end = stamp()
    after_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    elapsed = (end["monotonic_ns"] - start["monotonic_ns"]) / 1e9
    cpu = (after_usage.ru_utime + after_usage.ru_stime) - (before_usage.ru_utime + before_usage.ru_stime)
    result = {"kind": "unprivileged_gpu_telemetry", "tool_version": version, "command": command,
              "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
              "start": start, "end": end, "elapsed_wall_seconds": elapsed,
              "requested_duration_seconds": args.duration, "requested_interval_ms": args.interval,
              "recorded_samples": len(records), "records": records, "parse_errors": errors, "stderr": stderr,
              "return_code": proc.returncode, "terminated_at_requested_deadline": proc.returncode == -15,
              "monitor_child_cpu_seconds": cpu, "monitor_child_cpu_percent_of_one_core": cpu / elapsed * 100,
              "semantics": {
                  "frequency": "IOReport GPUPH active-state-residency-weighted frequency from IOKit frequency table, rounded to integer MHz; not instantaneous clock.",
                  "activity": "IOReport active/total performance-state residency percentage, not per-process attribution.",
                  "power": "IOReport GPU Energy delta converted to watts using requested sample duration; not measured whole-system electrical power.",
                  "timestamps": "received.monotonic_ns shares Python monotonic clock with memory samples; it timestamps JSON receipt, not the exact beginning/end of IOReport counter collection. source_timestamp has only whole-second resolution. Fields need not share identical collection windows.",
                  "overhead": "One existing monitor process plus a mostly blocked Python reader; child user+system CPU usage is recorded. No claim of zero benchmark interference.",
                  "scope": "Read-only; no root, installs, builds, power/fan-setting changes or GPU benchmark commands. Process/network details discarded."},
              "source_urls": ["https://github.com/metaspartan/mactop/blob/v2.0.0/internal/app/headless.go",
                              "https://github.com/metaspartan/mactop/blob/v2.0.0/internal/app/ioreport.m#L621-L668"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(args.output), "samples": len(records), "wall_seconds": elapsed,
                      "child_cpu_seconds": cpu, "cpu_percent_one_core": cpu / elapsed * 100}))


if __name__ == "__main__":
    main()
