# Clustered and offset kNN performance fix — October 30, 2026

The implementation avoids expensive screening failures on tight clusters and
large common offsets while preserving original-coordinate distances, precise
neighbor ordering and exhaustive fallback. The complete measured results are
in the [72-instance benchmark tables](knn-distribution-benchmarks-2026-10-30.md).

**The measured clustered/offset slowdowns against sklearn are eliminated.**
Across twelve dataset instances and nine operations, all **108/108 medians**
are faster than each of the three sklearn configurations. This is a median
result on the tested fixtures, not an every-call or universal-distribution claim.

| Workload | Query speedup vs preceding build | Fresh fit + query speedup vs preceding build |
| --- | ---: | ---: |
| Clusters, 50k × 32 | **2.59–2.70×** | **2.09–2.16×** |
| Offset +1e4, 50k × 32 | **4.39–4.44×** | **2.74–2.76×** |
| Clusters, 4096 × 128 | **2.15–2.41×** | **2.28–2.37×** |

These are median paired ratios from contemporaneous controlled comparisons,
with an identical-code A/A worker. All three implementations pass every control
audit. Offset 128-feature inputs retain the already-centered general route;
their measured query ratios versus the preceding build span 0.93–1.07×.

The full rerun measures the following query speedups against sklearn:

| Workload | Default-thread brute | Single-thread brute | Auto/default threads |
| --- | ---: | ---: | ---: |
| Clusters, 50k × 32 | 2.29–2.45× | 8.93–9.03× | 2.35–2.45× |
| Offset +1e4, 50k × 32 | 3.93–4.33× | 15.24–16.47× | 4.09–4.46× |
| Clusters, 4096 × 128 | 2.56–2.87× | 1.35–1.47× | 2.62–2.78× |
| Offset +1e4, 4096 × 128 | 3.27–3.48× | 1.71–1.87× | 3.18–3.58× |

**Gaussian gains remain:** the independent 50k × 32 distribution seeds measure
4.26–4.73× query and 3.50–3.64× fresh-fit speedups versus default-thread sklearn.
The controlled Gaussian query ratios versus the preceding build are 0.962–1.042×
at 32 features and 0.989–1.029× at 128 features. Small fresh-fit costs remain:
the corresponding median paired ratios are 0.981–0.997× and 0.965–0.991×
(about 0.3–1.9% and 0.9–3.7% longer). Ratio-of-median fresh timings span
0.974–1.034× and 0.945–0.991×, respectively. A/A fresh paired ratios span
0.996–1.023× and 0.959–1.047×. These measurements preserve the large Gaussian
advantage but do not establish zero per-fit overhead; all raw samples and both
ratio definitions remain in the controlled results.

| Full 72-instance panel baseline | Medians ≥2× | Individual pairs ≥2× |
| --- | ---: | ---: |
| Default-thread brute | 482/648 (74.4%) | 7,288/9,720 (75.0%) |
| Single-thread brute | 246/648 (38.0%) | 4,083/9,720 (42.0%) |
| Auto/default threads | 524/648 (80.9%) | 7,774/9,720 (80.0%) |

All **14,472 current-version broad-panel audits pass**, and all 72 fixture array
hash inventories match the previous broad run exactly. The complete panel still
contains workloads where sklearn wins, particularly small real datasets and
some single-query cases. Exact-distance-valid sklearn tie alternatives remain
separate from numerical tolerance failures; baseline quality flags are retained.

## Changes and numerical safeguards

Three production files differ from the preceding frozen
[`knn-128-final` build](../benchmarks/reference/knn-128-final-2026-10-30.manifest.json):

- [`_neighbors.py`](../src/mlxlearn_device/_neighbors.py) caches a small sample
  of owned training data to detect tight groups and large common offsets.
  Clustered batches use the existing precise direct scan, avoiding the failed
  integer or Gram shortlist followed by a slower complete scan. For eligible
  32-feature inputs, the feature-scale and distribution decisions share one
  sixteen-row sample. Wide inputs sample only eight coordinates of sixteen
  rows. The existing unequal-scale compact route remains available.
- [`_neighbors_integer.py`](../src/mlxlearn_device/_neighbors_integer.py)
  optionally subtracts a cached training mean before integer screening. Every
  coordinate must pass an exact-subtraction guard; otherwise the row receives
  a negative-infinity lower bound. Unsafe translated values are replaced by
  zero before integer conversion, without allowing that row to be discarded.
  The original training/query arrays still drive final distances, refinement,
  exact comparisons and fallback. Centered searches retain `k+19` candidates
  (at most 27), while ordinary Gaussian searches retain `k+11`.
- [`distance.py`](../src/mlxlearn/_kernels/distance.py) distinguishes an
  automatically chosen memory tile from an explicitly requested training tile.
  This lets public 128-feature clustered queries reach the direct route.
  Explicit tiling still forces the general route, and ordinary general searches
  retain their existing memory-budget plan.

The translation guard uses Sterbenz's exact-subtraction condition: nonzero
values and centers have the same sign, their magnitudes differ by at most a
factor of two, and both magnitudes lie in `[2^-30, 2^30]`. The exponent range
keeps nonzero differences out of Metal's subnormal-flushing range. A zero center
uses the guarded identity transformation. Consequently the filter sees an exact
translation of the original coordinates wherever it can certify a result. No
metric rescaling, distance-tolerance relaxation or approximate-neighbor
acceptance was introduced. See the [certificate argument](knn-integer-certificate.md)
and [original-coordinate ordering argument](knn-precision-ordering.md).

The routing sample is a performance heuristic, not a numerical certificate.
Unusual cluster arrangements or outliers can defeat the heuristic; correctness
still comes from the selected search path's bounds and exact fallback. Only
owned training preparation is cached, never query results.

## Validation

**677 relevant tests pass**, with five unrelated logistic/SVC cases deselected.
The [full test log](../benchmarks/results/knn-distribution-full-tests-2026-10-30.log)
includes the existing precision, parity, ownership, validation, patching and
sklearn estimator-compliance tests. Added coverage includes:

- Independent float64 and Python-integer references for positive, negative and
  mixed-sign offsets, close unequal neighbors, true ties, subnormal coordinates,
  overflow and guard boundaries.
- Unsafe training rows and later out-of-guard queries, copied rows, query/reference
  tails, self-exclusion, changed queries, refits and owned fit storage.
- Every raw integer-screening pair in both centered and uncentered modes,
  checking lower-bound validity and useful nonvacuous bounds.
- Actual public/native route execution and preservation of automatic memory
  budgets and explicit training tiles.

Tests and timing run serially on the GPU. Ruff and `git diff --check` pass.
The [execution record](../benchmarks/results/knn-distribution-final-2026-10-30-execution.json)
records commands, exit statuses and source/harness integrity. Final production
is frozen in the [source archive](../benchmarks/reference/knn-distribution-final-2026-10-30.zip)
with a [SHA-256 manifest](../benchmarks/reference/knn-distribution-final-2026-10-30.manifest.json).
The [benchmark/test archive](../benchmarks/reference/knn-distribution-final-2026-10-30-harnesses.zip)
has its own [manifest](../benchmarks/reference/knn-distribution-final-2026-10-30-harnesses.manifest.json).
The [final validation record](../benchmarks/results/knn-distribution-final-2026-10-30-validation.json)
checks both archives, unchanged source/harness files, identical fixture arrays,
all targeted speed counts and report links.

## Benchmark protocol

The controlled comparison uses the identical 24 distribution fixtures as the
broad panel: four distributions, two shapes and three seeds. Current, an
identical-code A/A worker, and the frozen preceding build run 31 interleaved
paired calls after one second of continuous warmup per worker and operation.
Both reusable query and fresh fit-plus-query are measured. The A/A results
expose process and timing variation rather than treating every small ratio
change as a code improvement.

The full panel repeats the previous 72 dataset instances in 24 families, nine
search/inference operations, 15 retained calls after five warmups, and three
sklearn configurations: brute/default threads, brute/one thread and
auto/default threads. Fit-only and first-query timings remain separate.
Public NumPy boundaries and synchronization are included. Every output is
audited outside timing against exhaustive direct float64 differences on the
stored float32 inputs, with canonical distance/index ordering. Baseline
precision failures and valid alternate tie choices remain visible in the
results instead of being dropped from timing counts.

These measurements cover float32 Euclidean search on Apple M4 Max, using the
forced MLX GPU backend. They do not establish universal speedups, performance
on other hardware or metrics, or the behavior of automatic CPU/GPU dispatch.

```sh
python3 -m benchmarks.knn_distribution_control \
  --source-manifest benchmarks/reference/knn-distribution-final-2026-10-30.manifest.json \
  --output benchmarks/results/knn-distribution-final-2026-10-30-control.json
python3 -m benchmarks.knn_broad \
  --source-manifest benchmarks/reference/knn-distribution-final-2026-10-30.manifest.json \
  --repeats 15 --warmup 5 \
  --output benchmarks/results/knn-distribution-final-2026-10-30-broad.json
```
