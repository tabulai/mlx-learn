# Radius-neighbor benchmarks

Three tools measure mlx-learn's radius search (`radius_neighbors`, `radius_neighbors_graph`, and
`RadiusNeighborsClassifier`/`Regressor` `predict`) against scikit-learn:

| File | Purpose |
|---|---|
| `bench_radius.py` | The validation campaign: grids of configs, every variant timed in interleaved rounds, correctness checks, JSONL records |
| `analyze.py` | Ratios, win counts and geometric means per region, acceptance bars, same-session control ratios, routes taken, Markdown output |
| `radius_calibrate.py` | The microbenchmark framework (hot and gapped timed callables in interleaved rounds) and the environment helpers both tools share. Work item 4.3 fills in its suites |

Run everything from the repository root, with `PYTHONPATH=src`. A worktree's editable install
points at the main checkout, so without `PYTHONPATH=src` you benchmark the wrong code. Each
record stores `meta.mlxlearn_file` and the source revision, so you can check which tree ran.

## Quick start

```bash
PY=.venv/bin/python   # or the venv you benchmark with
PYTHONPATH=src $PY benchmarks/radius/bench_radius.py smoke --out /tmp/radius-smoke.jsonl --label dev
$PY benchmarks/radius/analyze.py /tmp/radius-smoke.jsonl --md /tmp/radius-smoke.md
```

The smoke grid (2 configs) takes about 25 s. `--list` prints a grid's config keys without
running anything. `--only SUBSTR` keeps the configs whose key contains `SUBSTR`, and `--max N`
keeps the first N.

## Grids

| Grid | Configs | Contents |
|---|---|---|
| `micro` | 2 | n=2000, nq=50, d in {3, 32}: tests and plumbing checks only |
| `smoke` | 2 | gauss n=20k nq=1k, d in {3, 32} |
| `core` | 161 | uniform/blobs at d in {2, 3, 8}; gauss d=12; gauss/blobs at d in {16, 64, 128}; n in {100k, 1M}; latency cells at n=10k, nq in {1, 16} |
| `real` | 24 | MNIST (784-d), CLIP-512 normalized, SPACEV (100-d) |
| `ops` | 44 | blobs, n=100k: sorted results, distance graph, classifier and regressor with uniform and distance weights, self queries at n=50k; MNIST classifier |
| `extra` | 85 | offsets 1e3 and 1e6; label-sorted training rows; hubs (heavy-tailed density); self queries at n in {50k, 1M} and d in {3, 8, 16}; latency calls (nq in {1, 16, 64} x d in {3, 16, 64} x n in {10k, 100k}); connectivity graphs; distance weights with fresh and coincident queries |

The smoke, core, real and ops grids reproduce the review driver's configs, config keys and data
bit for bit, so their records compare directly with earlier campaigns. Each config is keyed by
its sorted `key=value` pairs. Real-data records may carry `d`, which the key ignores.

Six latency cells of `extra` (n=10k, nq in {1, 16}) are identical to `core` cells. With the same
label and output file, the second grid skips them.

## Variants

| Variant | What runs |
|---|---|
| `sk_auto`, `sk_brute`, `sk_auto_nj` | scikit-learn: default; `algorithm="brute"`; `n_jobs=-1` (only when d <= 15) |
| `mlx` | Layer 1 (`mlxlearn.neighbors`) |
| `mlx_patched` | the patched class for a default user; its user stock is `sk_auto` |
| `mlx_patched_nj` | a user who asked for `n_jobs=-1`; its user stock is `sk_auto_nj`, or `sk_auto` for d > 15, where scikit-learn's brute force ignores `n_jobs`. On by default in `extra` |
| `mlx_patched_brute` | a user who asked for `algorithm="brute"`; its user stock is `sk_brute` |
| `mlx_route=<r>` | Layer 1 with `Tuning.radius_route=r` forced through `tuning_context` (`--routes`); the patched class for `r=stock` |
| `ctl_<variant>` | the same variant run by an older source tree (`--control`) |

`--variants sk_auto mlx` selects exactly those variants. `--variants +mlx_patched_brute` adds to
the grid's defaults. Each record names the user configuration (`user_config`) and the stock
variant (`user_stock`) of every patched variant. `summary.<variant>.speedup_vs_user_stock_warm`
holds the ratio.

## Protocol

For each config, the driver:

1. Generates the data. Seeds are fixed, and values are representable in float32, so mlx-learn's
   float32 copy is exact. It then sets the radius to the median distance to the m-th neighbor
   over 256 sampled queries.
2. Runs the cold phase, `--cold` rounds (default 2). Each round constructs, fits and runs the
   first query of every variant, in rotated order, with a `--pause` idle before every call.
   Before the pause, the driver collects garbage and drains (see below).
3. Runs the warm phase on the fitted estimators: `--warm` timed rounds (default 5) after one
   untimed round. Round r rotates the variant order by r and reverses it on odd rounds. The
   untimed round runs the correctness checks against scikit-learn brute force in float64:
   membership (exact row sets; off-boundary disagreements are those more than 1e-5 relative away
   from the radius), distance accuracy, sortedness, graph patterns and predictions.
4. Caps the rounds for slow configs. When a cold query takes more than 4 s, the warm rounds are
   cut to fit about 40 s. A variant whose cold query exceeds `--slow` (60 s) gets only one timed
   round.

`--gap` controls the spacing of the warm calls:

* `gapped` (the default) idles `--pause` seconds (0.3 s) before every call. The GPU clocks wind
  down during the pause, so this is the regime a sporadic caller sees. It is the regime the
  router is calibrated for.
* `hot` has no pause. Each timed call immediately follows `--hot-primers` (default 1) untimed
  calls of the same variant, and variants are still interleaved from slot to slot. The primer
  brings the CPU and GPU clocks up, so the timed call sees a back-to-back caller's state. With
  `--hot-primers 0`, the timed call starts after whatever the previous variant left behind.

The cold phase always pauses `--pause` seconds, in both modes. The record's `protocol` field stores
the mode, the pause, the number of primers, the drain setting and the rounds actually run.

### Drain before every call

After each parallel region, scikit-learn's OpenMP threads spin on every core for about 200 ms (the
default blocktime). Under a fixed 0.3 s pause, a call that follows a scikit-learn call therefore
starts after only about 0.1 s of idle, with the clocks still high. On the M4 Max, a 3 ms query
took 4.3 ms after `sk_brute` and 9.6 ms after `mlx_patched`; with a 0.6 s pause, it took 8.3 ms
after either.

The rotation gives every variant the same two neighbors in every round, so this became a
systematic bias per variant. In an A/A control run, the in-process `mlx`, a neighbor of
`sk_brute`, measured 1.9x faster than the identical `ctl_mlx`.

The driver now drains before every call, in both modes. It waits until its own process (and the
control child, after a `ctl_` call) uses less than a quarter of a core over 20 ms windows. Only
then does it pause, or run the primers. With the drain, the same A/A cell gave 1.02-1.05, by
median and by paired ratio. Each record stores the time spent draining in `drain_s`.
`--no-drain` restores the review driver's fixed pause, for comparisons with its records.

### Thermal gate and foreign load

Before every call, the driver waits while the thermal pressure level
(`com.apple.system.thermalpressurelevel`, the state `notifyutil -g` prints) is above
`--thermal-max` (0 = nominal; 9 disables the gate), for at most `--thermal-limit` seconds per
wait. Each record stores the thermal level at the start and end and the gate's totals in
`thermal_gate` (`waits`, `wait_s`, `timeouts`). A laptop that throttles shows up there.

The gate reads the level in-process through libSystem's notify API. The review driver instead
spawned `notifyutil` before every call, which forks the benchmark process, and the copy-on-write
faults after each fork made the next in-process call slower. In an A/A run (the same tree in-process
and as `--control`), the in-process side measured 15-20% slower on a 3 ms query. The driver
forks only once per config (`ps`, for the load capture), before generating the data.

Each record also stores `load_start` and `load_end`, the busiest other processes and their total
CPU percentage (`*_total`), excluding the benchmark and its control child. Other sessions may be
benchmarking on the same machine. If the totals are high, rerun the affected cells instead of
trusting them.

On the M4 Max, gapped calls of a few milliseconds are bimodal. The same query takes either about
3 ms or 8-10 ms, and the slow phase holds every variant for several rounds at a time.
Interleaving puts every variant into the same phases, but a median over 5 rounds can still land on
different phases for different variants. For short cells, prefer more `--warm` rounds and
`analyze.py --metric paired`. Heavy foreign load tends to hold the fast phase. Check the A/A
spread on the M5 Ultra the same way before relying on it there.

## Same-session control (`--control`)

A before/after claim needs the old code timed in the same session, interleaved with the new code.
A record from an earlier day is not enough. Before trusting small differences, run an A/A check:
pass the current tree itself (`--control src`) and look at the spread of `new vs ctl` around 1.

`--control SRC` starts a child process with `PYTHONPATH=SRC`, where `SRC` is the `src` directory
of another mlx-learn tree. The child serves
the same script. It generates identical data, which a fingerprint confirms, and fits the in-process
`mlx` and patched variants with the old code. The parent puts these `ctl_<variant>` variants in the
same rotation, with the same pauses, primers and thermal gate. The child times its own calls, so
the pipe is never inside a timed region. Correctness of `ctl_` outputs is checked through
membership digests, compared against both scikit-learn and the in-process counterpart.

To prepare a control tree, for example PR #3's head, outside the repository:

```bash
git worktree add /tmp/radius-ctl b1446bc   # or: git archive b1446bc src | tar -x -C /tmp/radius-ctl
cd /tmp/radius-ctl && $PY setup.py build_ext --inplace   # native modules, same interpreter
cd - && PYTHONPATH=src $PY benchmarks/radius/bench_radius.py core --out runs/m1.jsonl \
    --label m1 --control /tmp/radius-ctl/src
```

The parent refuses a child whose `mlxlearn` or `mlxlearn_device` was imported from outside `SRC`.
It warns when the two trees have different native modules (`meta.native_modules` against
`meta.control.native_modules`), since a tree without its extension is an unfair control. If the
child dies, the affected `ctl_` variants record an error and the child restarts for the next
config. `--control-timeout` (900 s) bounds a single reply.

## Forced routes (`--routes`)

`--routes all` adds one `mlx_route=<r>` variant for every value of the installed
`Tuning.radius_route` except `auto`. `--routes tree,metal` adds only those. When the installed
mlx-learn has no such field (PR #3), the option is skipped with a message. A forced route that is
infeasible for a config raises an error, which is recorded in `variant_errors`, and the variant
leaves the rotation. `MLXLEARN_RADIUS_ROUTE` in the environment forces every mlx-learn variant,
and the driver warns about it.

## Routes recorded

After every mlx-learn call, the driver reads `est._index_._radius_search.last_plan` (`route`,
`reason`, `threads` and `predicted_ms` when present). It clears that field before each call, so a
stale plan is never recorded. Without `last_plan`, it falls back to a probe on
`RadiusSearch.route` (PR #3's `tree`, `cpu`, `mlx` and `sklearn`). The record stores:

* `routes` and `plans`, keyed by `<variant>:cold` and `<variant>:warm`;
* `route_trace`, the route of every timed call, which shows any dependence on call history;
* `backend`, the estimator's `last_query_backend_` or `execution_backend_`.

## Real data

The `real` grid and the MNIST cells of `ops` read arrays from the directories in
`RADIUS_BENCH_DATA` (os.pathsep-separated), then from `benchmarks/radius/data`:

* `mnist-X.npy` (float32 [60000, 784], values 0-255) and `mnist-y.npy`;
* `imagenet-clip-512-normalized-{train,test}.npy` and `spacev-1M-{train,test}.npy`.

The driver also accepts the e_means cache layout, `~/.cache/mlxlearn/e_means/arrays/<name>/{train,test}.npy`.

## Remote machines (M5 Ultra, M1 Ultra)

The protocol is the same on a remote Mac. Use a dedicated directory there, so other sessions'
checkouts never change under the run.

```bash
rsync -a --delete src benchmarks/radius setup.py pyproject.toml <host>:radius-bench/
ssh <host> 'cd radius-bench && $PY setup.py build_ext --inplace'   # in the remote venv
ssh <host> 'cd radius-bench && nohup env PYTHONPATH=src $PY benchmarks/radius/bench_radius.py \
    core --out core.jsonl --label m5-pr3 > core.log 2>&1 &'
rsync <host>:radius-bench/core.jsonl runs/ && $PY benchmarks/radius/analyze.py runs/core.jsonl
```

Rerunning the same command resumes, because configs already recorded under the label are skipped.
`--retry-errors` reruns the configs whose stored record is an error. Each record stores the GPU
generation (`meta.device`, for example `applegpu_g17d`), the core counts, and the library and
macOS versions. A machine without native modules records an empty `meta.native_modules`. Build
the extension first, because without it the CPU routes are the NumPy fallbacks. Run the M1 only
when its foreign CPU load is low, and check `load_start_total` afterwards.

## Analysis

```bash
$PY benchmarks/radius/analyze.py a.jsonl [b.jsonl ...] [--label L] \
    [--metric warm_med|warm_min|cold_med|e2e|paired] [--md summary.md]
```

All ratios are speedups (baseline time / variant time), so a value above 1 means mlx-learn is
faster. `--metric` sets the time compared: the warm median (default), the warm minimum, the cold
first query, `e2e` (fit + cold first query), or `paired`. `paired` is the median, over the warm
rounds both sides ran, of the per-round ratio; records store each sample's round in
`warm_round`. The report contains:

* **Acceptance bars.** Layer 1 at least 1.0x both `sk_auto` and `sk_brute`. Each patched variant
  at least 0.95x its user stock. Each `<variant>: new vs ctl` at most 5% slower than control.
  Regret at most 1.25 and at most 2.
* **Per region.** Low-d (d <= 8), mid-d (9-15), high-d (>= 16), real data, and ops (every op
  other than `rn`), plus `all`. For each comparison: `mlx` against `sk_auto`, `sk_brute`,
  `sk_auto_nj` and the best scikit-learn variant; each patched variant against its user stock
  and the best scikit-learn variant; new against control; and regret. Each row shows the cell
  count, wins (ratio >= 1), cells passing the gate, and the geometric mean, minimum and maximum.
* **Regret.** The automatic route's time divided by the best forced route's time. For
  `mlx_patched`, the user's stock variant also counts as an option.
* **Routes taken** per variant and region, with an example reason, and whether each forced route
  was honored.
* **Correctness flags** against scikit-learn brute force, and the `ctl_` outputs that differ from
  their in-process counterparts.
* **Errors**, for configs and for individual variants.
* **Per config**: times, ratios and routes for every config.

When a file holds the same (label, config) more than once, the last record wins.

## Calibration microbenchmarks

```bash
PYTHONPATH=src $PY benchmarks/radius/radius_calibrate.py --list
PYTHONPATH=src $PY benchmarks/radius/radius_calibrate.py round_trip \
    --protocols hot gapped:0.002 gapped:0.3 --rounds 15 --out /tmp/round_trip.json
```

A suite returns `Case`s, which are named setups that return a zero-argument callable, with tags and
a per-call work count. `run_interleaved` times every case under a `Protocol` (`hot[:primers]` or
`gapped[:seconds]`), using the same rotation and thermal gate as the campaign. MLX results are
forced with `mx.eval`, never with `mx.synchronize`. `fit_affine` fits the affine cost terms of the
routing model by least squares on relative residuals. `round_trip`, `host_distances` and `ragged`
work today. `--list` shows the suites planned for work item 4.3 (gpu_direct, gpu_gram, finisher,
tree build and pilot, brute force, output, cKDTree, and the scikit-learn models).
