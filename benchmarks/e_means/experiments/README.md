# E-means gap investigation experiments

These programs reproduce assignment-kernel experiments on Apple Silicon. They
assume the repository's `.venv` and run GPU work sequentially. Keep other GPU
benchmarks idle while collecting comparable timings.

`gap_baseline.py` loads the BF16 kernel from the frozen
`../reference/metal-v2-source.zip`, verifies its pinned source SHA-256, and
verifies the extracted copy against the archive. It does not use the current
production kernel as the baseline. The pinned kernel hash is
`cbe903b6972ab2fcca06b940005a5d3ae4e700f9595460036f14880625f3e4ea`;
the archive hash is
`2ac3e346f7a59186631fac850aa1e9831f34288dcb4e93bbc74ebb432b9383e0`.

The raw JSON in `../results/gap_investigation/` preserves the exact source
snapshots, configurations, timings, and hashes from the original runs. Later
formatting and frozen-baseline-loader changes to these scripts do not change
those historical records. `gap_validation.py` intentionally compares the
current production implementation against the frozen baseline.

Run from a fresh output directory to preserve the recorded evidence. Starting
in the repository root:

```sh
emeans_repo="$PWD"
emeans_output=$(mktemp -d)
cd "$emeans_output"
export PYTHONPATH="$emeans_repo/src"

"$emeans_repo/.venv/bin/python" "$emeans_repo/benchmarks/e_means/experiments/gap_assignment.py" \
  --roofline --dims 96,1024,3072 --rows 8192 --clusters 16384 --repeats 5 \
  --output roofline.json

"$emeans_repo/.venv/bin/python" "$emeans_repo/benchmarks/e_means/experiments/gap_assignment.py" \
  --stage schedule --dims 96,1024,3072 --rows 8192 --clusters 16384 --repeats 5 \
  --output schedule.json

"$emeans_repo/.venv/bin/python" "$emeans_repo/benchmarks/e_means/experiments/gap_assignment.py" \
  --stage tiles --dims 96,1024 --rows 8192 --clusters 16384 --repeats 5 \
  --output tiles.json

"$emeans_repo/.venv/bin/python" "$emeans_repo/benchmarks/e_means/experiments/gap_lowdim.py"
"$emeans_repo/.venv/bin/python" "$emeans_repo/benchmarks/e_means/experiments/gap_coalescing.py"
"$emeans_repo/.venv/bin/python" "$emeans_repo/benchmarks/e_means/experiments/gap_validation.py"
"$emeans_repo/.venv/bin/python" "$emeans_repo/benchmarks/e_means/experiments/gap_dispatch_bandwidth.py"
"$emeans_repo/.venv/bin/python" "$emeans_repo/benchmarks/e_means/experiments/gap_queue.py" --output queue.json
```

The low-dimensional, coalescing, validation, and bandwidth programs create their
result hierarchy under the fresh working directory. Queue ablation writes to
the explicit output path and refuses to overwrite existing evidence.
Low-dimensional and coalescing comparisons use sustained warmup and
interleaved baseline/candidate measurements; production validation reverses
the order on alternate rounds. Short initial sweeps are candidate searches,
not the final performance evidence.

Roofline FP16-output and BF16-output matrix multiplication cases are throughput
diagnostics. Their output representations differ from the required BF16-input,
FP32-accumulation assignment and are not production replacements. Copy
bandwidth counts bytes read plus bytes written. The singleton-copy measurement
includes Python, MLX, driver, execution, and synchronization time; it is not a
measurement of hardware dispatch latency alone.

`gap_queue.py` measures the current production implementation on SPACEV and
AGNews, keeping the audited data, recipe, and seed fixed. It replaces every
Nth batch's asynchronous submission with synchronous evaluation for
N=0/2/4/8 (zero means no replacement), resets counters per fit, and records
one full warmup plus three interleaved cache-cleared measurements per period.
Its memory metric is the MLX active allocation peak including resident input.
`--self-test` checks submission scheduling and API restoration on the CPU
without importing MLX. Production runtime files are never modified.
