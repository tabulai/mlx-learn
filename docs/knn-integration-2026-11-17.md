# Combined numerical fixes and exact kNN integration — 2026-11-17

The combined implementation retains exact neighbor ordering while integrating
large-sort safety, filtered selection, partitioned/bounded fallback, M5 binary32
products, wide-full PCA corrections, and shared general-query tile planning.
It is unreleased; package metadata remains 0.1.3.

## What changed

- Full float32 sorts are limited to 2**20 training columns per tile; exact tile
  merging still covers all training rows. This avoids the incorrect MLX orders
  observed with 2**21 or more columns.
- GPU-generation-specific selection and block-height choices are preserved.
  The filtered selector computes minima over disjoint chunks and uses their
  kth minimum as a safe upper bound on the global kth value. It rejects only
  larger values; exact membership, distance refinement and canonical ties remain.
- Uncertified rows are compacted and searched in parallel training chunks, then
  merged with exact comparisons. Fused, compact and integer screens additionally
  use valid candidate-distance bounds to avoid unnecessary exact comparisons.
- M5/unknown-generation float32 matrix products use the shared binary32 kernel
  where necessary. Earlier GPUs retain their original MLX expressions.
- Wide full PCA reorthogonalizes components, copies negative-stride input views,
  and retains meaningful tail directions instead of replacing them with unrelated
  basis vectors. The earlier individual measurements and numerical reasoning are
  retained in the Unreleased changelog; this report's fresh timings cover kNN.

## Shared planning and memory contract

`mlxlearn_common._neighbors_plan` owns dependency-free defaults, memory detection,
workspace estimation and tile selection. Both general-search interfaces use it.
It does not import MLX, NumPy, sklearn, or either estimator interface. The existing
fused/small-search batching policies remain unchanged.

The GPU workspace target is physical memory / 32, clamped by the default 256 MiB
floor and 1 GiB ceiling. The CPU target remains 128 MiB, with the direct CPU
subtraction kernel applying its additional 64 MiB bound. Sixteen bytes per
query/training pair estimate the general distance tile. This is a workspace
planning target, not a hard limit on total allocations: resident inputs, prepared
training buffers, output, selector scratch and fallback scratch also use memory.
Explicit dimensions and the minimum candidate width retain their historical
ability to exceed the target.

The NumPy interface keeps scoped `tuning_context` controls. The standalone device
interface reads the same defaults and `MLXLEARN_*` environment overrides. Already
planned host dimensions bypass standalone environment parsing. Explicit direct
training tiles smaller than k remain valid because per-tile exact merging handles
them. No host-library dependency was added to resident queries, and automatic
CPU-only startup still avoids importing MLX.

On the measured 128 GiB machine, the 1,048,576-row/512-query general case uses
64 query rows and the full training width. That is why resident queries now
benefit from the filtered selector instead of repeatedly selecting narrow tiles.
The tradeoff is higher memory: the k=10 device case's observed peak active MLX
memory grows from **869.5 to 1,821.3 MiB**, including its resident input/prepared
buffers. Peak query increment grows from 97.3 to 1,049.0 MiB. Smaller explicit
tiles or lower workspace tuning retain a memory/throughput tradeoff. These MLX
counters exclude host NumPy memory, the allocator cache and total process RSS.

## Reproducible comparison

Machine: Apple M4 Max, 128 GiB, macOS 26.6.2. Runtime: Python 3.14.2, MLX 0.32.4,
NumPy 2.5.4, scipy 1.18.1 and sklearn 1.9.1. This is a fresh M4 measurement;
no fresh M1/M5 timing or full Python/sklearn compatibility matrix was run here.
The custom binary32 route is forced in regression tests even on the M4.

Three frozen sources are compared:

1. **Release:** v0.1.3 (`07054ce`), before the reviewed changes.
2. **Pre-integration:** `61cfe3c`, the current selection commits plus the original
   uncommitted filtered-selector changes captured before integration.
3. **Combined:** `ab8137e`, the complete correctness/fallback/planner integration.

Each variant runs in its own isolated persistent Python worker. All `mlxlearn*`
source packages are frozen, archived and SHA-256 hashed before timing. Imports
outside that snapshot fail the audit. Native extensions are deliberately omitted:
these benchmark cases explicitly select GPU execution and do not measure CPU
fallback or automatic estimator dispatch. This omission does not apply to the
normal full test suite, which uses the installed development extensions.

For each workload and API, there are two untimed warmups and seven timed queries.
Variant order is randomized in every repetition, and GPU requests never overlap.
Device timings include `mx.eval`; NumPy timings include output conversion. Fit,
input preparation, compilation warmup, oracle calculations and output audits are
outside the timing. This is a warmed query comparison, not a fit/startup claim.

Every warmup and timed result must agree on all distance bits and neighbor indices
across variants and match a chunked direct float64 oracle on three query rows.
Input hashes must also match. Source hashes are rechecked after all runs. Failed
rows are retained and cannot receive a successful speedup claim. All **20 cells
× 3 variants × 9 queries = 540 query audits** passed, as did source integrity.
This sampled oracle plus full-output baseline comparison does not prove universal
correctness; the separate exactness/regression suite covers additional edge cases.

Fixtures use one deterministic seed per case. Gaussian 256k cases are
262,144 × 96 with 256 queries; Gaussian 1m cases are 1,048,576 × 96 with 512
queries. Integer cases are 262,144 × 16 with 256 queries sampled from [-2, 2].
The clustered case has 32,768 × 256, 128 queries, 16 tight clusters and k=10.
`small_certified` is 2,048 × 96, 64 queries, k=10; `small_compact` is 16,384 × 16,
64 queries, k=8.

All times below are median milliseconds. A speedup below 1 is slower.

| Case | API | Release | Pre-integration | Combined | vs release | vs pre-integration |
|---|---|---:|---:|---:|---:|---:|
| `small_certified` | device | 0.800 | 0.747 | 0.810 | 0.99× | 0.92× |
| `small_certified` | numpy | 0.610 | 0.576 | 0.624 | 0.98× | 0.92× |
| `small_compact` | device | 0.619 | 0.663 | 0.698 | 0.89× | 0.95× |
| `small_compact` | numpy | 0.437 | 0.413 | 0.497 | 0.88× | 0.83× |
| `gaussian_256k_k5` | device | 12.365 | 11.408 | 4.166 | 2.97× | 2.74× |
| `gaussian_256k_k5` | numpy | 5.345 | 4.045 | 4.047 | 1.32× | 1.00× |
| `gaussian_256k_k10` | device | 28.435 | 26.912 | 4.506 | 6.31× | 5.97× |
| `gaussian_256k_k10` | numpy | 29.523 | 4.132 | 4.186 | 7.05× | 0.99× |
| `gaussian_256k_k32` | device | 27.956 | 27.754 | 4.339 | 6.44× | 6.40× |
| `gaussian_256k_k32` | numpy | 29.795 | 4.576 | 4.549 | 6.55× | 1.01× |
| `gaussian_1m_k10` | device | 188.544 | 188.458 | 34.956 | 5.39× | 5.39× |
| `gaussian_1m_k10` | numpy | 280.418 | 36.058 | 36.139 | 7.76× | 1.00× |
| `gaussian_1m_k32` | device | 196.065 | 194.992 | 35.534 | 5.52× | 5.49× |
| `gaussian_1m_k32` | numpy | 279.996 | 36.681 | 36.376 | 7.70× | 1.01× |
| `integer_256k_k8` | device | 22.079 | 22.854 | 2.281 | 9.68× | 10.02× |
| `integer_256k_k8` | numpy | 22.072 | 22.673 | 2.195 | 10.06× | 10.33× |
| `integer_256k_k9` | device | 193.045 | 193.550 | 3.997 | 48.29× | 48.42× |
| `integer_256k_k9` | numpy | 192.718 | 193.166 | 3.843 | 50.14× | 50.26× |
| `clustered_32k` | device | 272.869 | 272.082 | 4.408 | 61.91× | 61.73× |
| `clustered_32k` | numpy | 272.536 | 271.436 | 4.208 | 64.76× | 64.50× |

The large Gaussian NumPy improvements were already present before integration;
the combined build preserves those gains and brings them to the resident API.
The tied and clustered workloads expose the fallback performance cliffs fixed
by this integration. Their 48–65× gains should not be generalized to ordinary
Gaussian searches.

Small workloads remain a tradeoff. The four small cells are 1–14% slower than
the release (about 10–80 microseconds), and 5–20% slower than pre-integration in
this run. Extra fallback dispatches are not free even when all rows certify.
These differences and the complete timing samples are retained, not excluded
from the panel.

## Validation and package checks

- Full repository suite: **4,264 passed, 8 skipped**. Seven skips require pandas;
  one requires unavailable Python framework/compiler headers. The run emitted
  97 warnings, retained in the log rather than suppressed.
- Benchmark harness suites: **83 passed**, including 18 new evidence tests.
- Combined-route regressions force binary32 products, the actual filtered
  kernel, partitioned fallback, multiple training tiles, duplicates and
  self-exclusion together, including a singleton final tile. These tests are
  included in the full count.
- Shared-planner tests cover dependency isolation, scoped tuning versus env
  overrides, workspace sizing, the million-row plan and explicit tiny tiles.
- Ruff, `git diff --check`, and mechanical compliance checks passed. The lint
  command excluded the unrelated, pre-existing untracked October 27 logistic
  experiment directory; all committed source/tests/benchmarks were checked.
- Both sdist and local CPython 3.14 arm64 wheel built successfully. The extracted
  wheel contained `mlxlearn_common`; isolated imports, shared planning, a GPU
  neighbor query and PCA smoke checks passed. No package was published.

[Raw results](../benchmarks/experiments/knn-integration-2026-11-17/results.json)
contain per-repeat times, memory counters, environment, import inventories,
source hashes and checks. [Frozen sources and the exact harness](../benchmarks/experiments/knn-integration-2026-11-17/results-sources/)
are retained with [validation logs](../benchmarks/experiments/knn-integration-2026-11-17/validation/).

Reproduce from this integration branch with the same dependencies:

```bash
python -m pytest -q --durations=25
python -m pytest -q benchmarks/test_knn_*.py tests/unit/test_knn_integration_benchmark.py
python -m benchmarks.knn_integration --profile full \
  --reference preintegration=61cfe3c --output /tmp/knn-combined/results.json
```

Use `--profile quick`, or `--cases small_certified`, for a short pipeline check.
The harness includes retained sources beside every output JSON; the committed
`results-sources/knn_integration.py` records the exact measured driver.
