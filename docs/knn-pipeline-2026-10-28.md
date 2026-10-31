# KNN device pipeline and fused selection — 2026-10-28

The subsequent [acceleration report](knn-acceleration-2026-10-28.md) records further
GPU occupancy, query reuse and fit-pipeline improvements. The measurements below
describe the earlier implementation preserved by its source archive and hashes.

This implements the KNN recommendations in the
[earlier performance investigation](performance-investigation-2026-10-26.md#nearest-neighbors):
avoid dense distance/partition buffers where small-k specialization pays off,
retain candidates on-device, and benchmark both full estimator calls and resident
MLX calls against an immutable baseline.

## Implementation

`mlxlearn_device._neighbors` is the shared NumPy-free core. The experimental
`fit_neighbors(X, device="gpu")` accepts a finite float32 MLX matrix, owns its
storage and returns an index whose `query(Q, k)` produces MLX float32 distances
and int32 indices. `device="cpu"` explicitly uses MLX CPU operations. Importing
and running neighbor search does not import NumPy, sklearn, or the logistic
optimizer. Public queries validate their input; sklearn calls reuse the wrapper's
validated input.

The sklearn estimators retain their constructors, parameter checks, feature names,
NumPy result types, backend diagnostics and fallback policy. Float64 conversion
reuses logistic regression's checked input preparation. Self queries reuse the
resident training matrix. The host training copy remains for portable pickle
state and inherited radius queries. Fitted state owns its arrays and caches only
training preparation, never completed query results.

For GPU searches with at most 32 features and 16 neighbors, a fused kernel scans
training points directly and keeps a sorted top-k list per SIMD lane. Repeated
head reductions merge those lists, allowing every winning neighbor to belong to
the same lane. Small query batches split training scans across up to 32 partitions
and merge their short lists on the GPU. Feature count is specialized at Metal
compilation; distances use compensated accumulation.

Other shapes use centered matrix products. Small candidate lists (up to 16)
use a fused tile selector; larger lists use packed distance/index ordering keys.
Candidates are refined from original coordinates before tile merging. Fused GPU
refinement avoids the former query-by-k-by-feature gather. CPU refinement chunks
features when that gather would exceed its working budget. Intermediate tile
evaluation bounds the live lazy graph. Final ordering and self-exclusion remain
on-device.

Classifier labels are encoded once, retained on-device, and rebuilt after pickle.
Voting uses deterministic compensated sums; predictions compare class scores
before float32 probability normalization. Distance weighting handles coincident
points and retains square-root residuals from squared distances for near ties.
Uniform prediction skips distance output. The sklearn regressor deliberately
retains float64 target gathering and reduction: converting arbitrary targets to
float32 would lose precision or overflow values accepted by the existing API.

The general Gram path still has the documented fixed over-selection limitation:
refinement cannot recover candidates already excluded by Gram rounding. This
change does not introduce approximate search, tree metrics, or float64 GPU
arithmetic. See the [float32 policy](fp32_policy.md).

## mlx-geospatial review

The inspected local `mlx-spatial` checkout is
[`tabulai/mlx-geospatial`](https://github.com/tabulai/mlx-geospatial), revision
[`e7af5bac`](https://github.com/tabulai/mlx-geospatial/commit/e7af5bac).
Its relevant implementation is
[`mlx_geospatial/haversine.py`](https://github.com/tabulai/mlx-geospatial/blob/e7af5bac/mlx_geospatial/haversine.py).
That revision has fused nearest-neighbor routines, not KMeans or DBSCAN clustering.

The reusable ideas are streaming lane-local top-k lists, SIMD list merging,
training-side preparation reuse and shape-sensitive dispatch. The Euclidean
kernels here are independently implemented and credit that design. Spherical
chord-distance pruning and haversine-specific thresholds were not transferred;
they do not establish valid Euclidean bounds or performance thresholds.

## Measurement

The harness freezes the full starting Python packages at `5615cb0` in a
[reference archive](../benchmarks/reference/knn-2026-10-28.zip) with an adjacent
SHA256 manifest. Current, frozen, resident and sklearn variants run in separate
persistent processes over identical serialized float32 inputs. Timings alternate
variant order and synchronize GPU completion. Every call is audited outside the
timer against float64 neighbors and direct distances on the rounded input.

Both warmed query and fresh fit-plus-query are measured. Fit alone measures only
eager work: the new general path prepares Gram statistics on first query, so
fit-only gains must not be interpreted as end-to-end gains. Cold calls are the
first call for each shape after imports; earlier shapes can warm compilation.
The harness also records single-thread and default-thread sklearn, prediction
and probability calls, all samples, backend, versions, hardware, input hashes
and source hashes before and after timing.

The initial panel rejected an overly broad direct-search route: D32/k32 was
slower than the frozen implementation. Feature specialization, small-list tile
selection and the final routing threshold address that result. The initial and
tuned-probe JSON files preserve these intermediate measurements.

## Recorded results

Apple M4 Max, 128 GiB unified memory, MLX 0.32.2. Nine measured samples
follow three warmups. Speedups below compare NumPy-to-NumPy calls with the frozen
implementation; values below 1 mean a regression. These are local measurements,
not portable speed guarantees. The [complete results](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-bf2517e9d550b3a0099f)
include all samples, resident timings, and both sklearn thread policies.

| Training × features / queries / k | Query (ms) | Query speedup | Fresh fit + query speedup |
|---|---:|---:|---:|
| 8,192 × 8 / 1 / 5 | 0.322 | 1.65× | 1.75× |
| 8,192 × 8 / 256 / 5 | 0.757 | 3.45× | 2.90× |
| 8,192 × 32 / 256 / 5 | 1.448 | 1.81× | 1.76× |
| 8,192 × 32 / 128 / 32 | 1.613 | 1.11× | 0.99× |
| 4,096 × 128 / 128 / 5 | 0.801 | 1.39× | 1.12× |
| 2,048 × 1024 / 32 / 5 | 1.092 | 0.97× | 1.06× |
| 4,096 × 32 / 128 / 64 | 1.107 | 1.42× | 1.15× |
| 8,192 × 16 / 256 / 5 | 0.957 | 2.63× | 2.45× |
| 8,192 × 32 / 1 / 5 | 0.355 | 1.68× | 1.55× |
| 8,192 × 8 / 128 / 32 | 1.580 | 1.16× | 0.97× |

Query medians improve in nine of ten cases, by 1.11–3.45×. The 1,024-feature
query is 3.4% slower, while its fresh fit-plus-query improves by 1.07×. Fresh
fit-plus-query for D32/k32 and D8/k32 is respectively 0.9% and 3.1% slower.
The 1,024-feature case also has slower indices-only search (6.6%), uniform
prediction (4.9%) and uniform probabilities (5.0%). D8/k32 uniform prediction
is 0.7% slower. Every other measured operation improves against the frozen
implementation. Single-query calls remain slower than single-thread sklearn
even though they improve over the previous MLX implementation. Existing patched
crossover settings were not generalized from this small panel.

The final implementation also compiles the integer ordering/gather operations
to reduce dispatch overhead. Paired probes verified bit-identical outputs with
CPU/GPU ties and exclusion masks. Floating-point preparation was left uncompiled
because experimental fusion changed its rounding. The preceding full panel is
preserved as `knn-pipeline-before-compiled-ordering-2026-10-28.json`.

Current, frozen-reference and resident results pass every exact-index and
`atol=rtol=1e-6` distance audit; source integrity passes. Both sklearn float32
comparators miss the strict distance audit only on an exactly copied training
row in the 1,024-feature case: they report a residual of `2.47349e-6` rather
than zero, with correct indices. The JSON preserves that failure:
`implementation_quality_pass=true`, each implementation variant passes,
and the all-comparator `quality_pass=false`. Tolerances were not relaxed.

Reproduce with:

```bash
python3 -m benchmarks.knn_pipeline --profile full --resident \
  --repeats 9 --warmup 3 --threads both \
  --output benchmarks/results/knn-pipeline-2026-10-28.json
```

## Validation

337 tests passed across neighbor parity, adversarial float32 cases, new device
and kernel tests, voting, sklearn pipelines/invariants, patching, sklearn
compliance for both API layers, and the existing device logistic tests. Ruff,
`tools/compliance.py`, and `git diff --check` passed.

New tests cover selection-cutoff ties across tiles, all winners in one SIMD
lane/partition, k=24/32 kernel edge cases, CPU/GPU routing, original-coordinate
refinement, mixed feature scales, overflow-safe self-exclusion, strided/readonly
inputs, empty core queries, device-owned data, cached-label pickle/refit behavior,
float64 regression targets, deterministic weighted votes, and weighted near ties
below float32 probability or square-root resolution. A fresh-process import
blocker executes search and aggregation while rejecting NumPy and sklearn.


## Additional 50,000 × 32 benchmark

User-requested larger dataset: 50,000 training rows, 32 features, 256 query rows,
k=5. Inputs are synthetic standard-normal float32 values, with two copied training
rows in the query batch to exercise zero-distance weighting. The effective data
seed is 51. This is a batch query against 50,000 rows, not an all-training-row
self-query. Algorithm source hashes match the preceding final panel exactly.

Apple M4 Max, MLX 0.32.2, sklearn 1.9.0; nine measured repetitions after three
warmups. Values are medians in milliseconds; lower is better. Default sklearn
uses 16 OpenMP threads. The resident API excludes input upload.

| Operation | New | Previous | sklearn 1 thread | sklearn default | MLX resident |
|---|---:|---:|---:|---:|---:|
| Fit | 0.985 | 1.988 | 0.403 | 0.389 | 0.650 |
| Fresh fit + query | 4.577 | 10.501 | 26.224 | 4.769 | 4.264 |
| Neighbors + distances | 3.208 | 7.610 | 25.902 | 4.231 | 3.646 |
| Indices only | 3.901 | 8.876 | 26.017 | 4.109 | 3.638 |
| Classifier, uniform | 3.698 | 9.259 | 26.316 | 4.438 | — |
| Probabilities, uniform | 4.089 | 7.703 | 26.285 | 4.271 | — |
| Regressor, uniform | 6.220 | 8.606 | 26.212 | 4.476 | — |
| Classifier, distance | 3.619 | 8.462 | 26.322 | 4.508 | — |
| Probabilities, distance | 3.639 | 7.894 | 26.297 | 4.655 | — |
| Regressor, distance | 3.945 | 8.306 | 26.150 | 4.620 | — |

Neighbor query medians improve by 2.37× over the frozen implementation, 8.07×
over single-thread sklearn and 1.32× over default-thread sklearn. Fresh
fit-plus-query improves by 2.29× over the frozen implementation and 1.04× over
default sklearn. All ten warmed operations improve over the frozen version.
Uniform-weight regression remains slower than default sklearn (6.220 versus
4.476 ms), and sklearn retains faster fit-only performance.

GPU timing variability is substantial: current query samples range from 2.482
to 6.818 ms, compared with 3.959 to 4.758 ms for default sklearn. The 1.32×
median comparison should not be treated as a tight latency guarantee. First-call
fit/query observations are recorded separately below; they are not medians and
can include initialization and compilation.

| First call | New | Previous | sklearn 1 thread | sklearn default | MLX resident |
|---|---:|---:|---:|---:|---:|
| Fit | 12.467 | 10.407 | 0.622 | 0.704 | 8.402 |
| Query | 9.587 | 24.993 | 33.015 | 12.903 | 8.816 |

Every accuracy audit passes for all five variants: exact neighbor indices,
distance `atol=rtol=1e-6`, and classifier/regressor checks on every timed and
warmup call. The current implementation’s maximum absolute distance error is
`2.61e-7`. Source integrity passes. Unlike the earlier 1,024-feature case, both
sklearn variants also pass the strict distance audit here.

[Complete 50,000 × 32 results](../benchmarks/results/knn-pipeline-50000x32-2026-10-28.json)

```bash
python3 -m benchmarks.knn_pipeline --case batch_50000_32_k5 --resident \
  --repeats 9 --warmup 3 --threads both \
  --output benchmarks/results/knn-pipeline-50000x32-2026-10-28.json
```
