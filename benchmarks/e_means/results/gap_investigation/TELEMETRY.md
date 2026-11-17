# Read-only GPU telemetry without root

The existing `/opt/homebrew/bin/mactop` **v2.0.0** can collect GPU telemetry without sudo. No installation, build, GPU workload, fan adjustment or power-setting change was needed. Its [headless implementation](https://github.com/metaspartan/mactop/blob/v2.0.0/internal/app/headless.go) emits one JSON object per line with `--count 0`.

Use the bounded wrapper:

```bash
.venv/bin/python benchmarks/e_means/results/gap_investigation/sample_gpu.py \
  --duration 30 --interval 1000 --output /tmp/gpu-telemetry.json
```

The wrapper starts one monitor, receives its streamed samples, and terminates that monitor at the requested deadline. It never installs anything or changes device settings. Raw process, memory and network fields are discarded. It records the binary hash/version, command, start/end anchors, errors, return code and child CPU consumption. The wrapper is independent of measured benchmark sources.

Each `records[]` entry includes:

- `source_timestamp`: the tool's RFC3339 wall timestamp, with whole-second resolution.
- `read_started` and `received`: `monotonic_ns`, `monotonic_bracket_ns`, `wall_time_ns` and `wall_utc` around receiving the JSON object. The monotonic clock matches Python's benchmark-memory sampler clock.
- `gpu_freq_mhz`, `gpu_activity_percent`, `gpu_power_watts`, `gpu_sram_power_watts`, GPU/CPU temperatures in Celsius and `thermal_state`.

Receipt time is **not the exact counter-window boundary**. Collection and other telemetry calls occur before JSON emission, so joins to fit windows need to retain that uncertainty. A requested 1000 ms interval is not a guarantee of one output per second.

The [v2.0.0 IOReport implementation](https://github.com/metaspartan/mactop/blob/v2.0.0/internal/app/ioreport.m#L621-L668) defines frequency as a residency-weighted average across active GPUPH performance states, using a frequency table read from IOKit. It excludes OFF/IDLE/DOWN states from the frequency denominator and rounds to integer MHz. Activity is active/total residency; it is not per-process activity. GPU power is an energy-counter delta converted to watts using the requested sampling duration. Temperature and the OS thermal-state flag are separate readings. These are private-API tool estimates, not an instantaneous clock trace or a thermal diagnosis.

The requested anchored run is saved as [gpu-telemetry-window.json](gpu-telemetry-window.json): **21 samples in 30.005 seconds**, received from 22:03:39.080 to 22:04:05.385 UTC on 2026-11-14. Reported frequency ranged **523–1327 MHz**, median **1035 MHz**; activity ranged **8.45–100%**, median **100%**. GPU power ranged **0.214–41.017 W**, median **22.820 W**, and temperature ranged **78.97–85.40°C**, median **80.90°C**. Every thermal-state label was `Normal`. These are concurrent observations only; they do not identify the cause of earlier timing differences.

[gpu-frequency-table.json](gpu-frequency-table.json) records the preferred `pmgr` properties `voltage-states9-sram` and `voltage-states9`, using the same lookup/layout as mactop. Both report a maximum table entry of **1578 MHz**. This is a registry table maximum, not an observed clock or a promise of sustained frequency. Headless mactop does not directly export a maximum-frequency field.

The monitor consumed **3.769 CPU seconds over 30.005 wall seconds**, approximately **12.56% of one core**, including startup and shutdown. The Python reader spends most of its time blocked. This is modest monitoring work, but **not zero measurement interference**; controlled comparisons should apply the same telemetry collection to both arms. The sampler was stopped after this window. The earlier [30-sample probe](gpu-telemetry-30s.json) took 41.293 wall seconds and has only tool-provided timestamps; its filename does not describe its actual elapsed duration. [Machine-readable observation summary](gpu-telemetry-summary.json).
