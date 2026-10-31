# kNN 128-feature regression fix — October 30, 2026

The remaining 128-feature regression is eliminated in the six-seed controlled comparison and the repeated full panel. Query speedups are **1.003–1.138× versus v3** and **1.113–1.290× versus the preceding precision-fixed build**. The longer full-panel rerun measures **0.584 ms versus v3’s 0.722 ms (1.238×)** for 4096 training rows, 128 queries, 128 features and k=5. The earlier 18.3% slowdown was 0.912 versus 0.771 ms in a separate run; the controlled simultaneous comparison is the evidence for the code improvement.

All **652 relevant tests pass**, including 21 new independent shortlist cases. Every current-implementation accuracy audit in the full rerun passes. Gaussian gains remain, and no distance tolerance, oracle or exact-ordering requirement was relaxed. Timings are measurements on Apple M4 Max, not a guarantee of a speedup on every possible shape or distribution.

## Implementation

Only two production files differ from the preceding [precision-fixed snapshot](../benchmarks/reference/knn-precision-verified-2026-10-30.manifest.json):

- [`_neighbors_general.py`](../src/mlxlearn_device/_neighbors_general.py) retains `k+2` candidates for 33–256 features and k<=8. Other general searches retain `k+8`. The same extra lower-bound candidate certifies all omissions; uncertainty still triggers a complete original-coordinate scan.
- [`_neighbors_shortlist.py`](../src/mlxlearn_device/_neighbors_shortlist.py) combines candidate refinement, precise ranking and omission certification in one GPU kernel. Each query owns one threadgroup, with one SIMD group per candidate. Shared high/low components feed the existing exact-capable comparator. Barriers synchronize refinement and the kth cutoff before checking the unchanged strict certificate.

This removes intermediate arrays, kernel launches and host-side graph construction. The 128-feature k=5 case refines seven candidates instead of thirteen. All coordinate-range guards, double-single error bounds, exact integer comparisons, invalid-index handling and the final exhaustive fallback remain intact. Per-tile certificates remain conjunctive; a later merge cannot discard a failed certificate. CPU search, native bit-preserving ownership, and the 32-feature Gaussian integer/compact path are unchanged. The [numerical argument](knn-precision-ordering.md) documents why candidate capacity does not determine correctness.

The [carry diagnostic](../benchmarks/results/knn-128-carry-probe-2026-10-30.json) measured 4–9% lower query time from the smaller margin alone across three seeds, with every output audited and no uncertified rows. The remaining saving comes from combining the separate refinement/ranking/certificate stages. The diagnostic is not substituted for end-to-end acceptance timings.

## Controlled 128-feature comparison

Six actual dataset seeds (45, 46, 47, 145, 245, 345) use the same 4096 × 128 training / 128-query / k=5 shape. Four persistent isolated workers load current source, a current-code A/A control, frozen v3, and the prior precision-fixed archive. Each variant/operation receives one second of continuous warmup followed by 31 interleaved paired rounds. Starting order rotates by seed and operation; alternate rounds reverse it. All cold, warmup, probe and retained outputs are audited outside timing against unchanged exact-index and direct-float64-distance references. Fresh fit-plus-query includes all deferred preparation.

| Seed | Current query, ms | v3 query, ms | Prior precision query, ms | Paired query speedup vs v3 | Paired query speedup vs prior precision | Paired fresh speedup vs v3 |
|---|---:|---:|---:|---:|---:|---:|
| 45 | 0.388 | 0.400 | 0.448 | 1.024× | 1.113× | 1.089× |
| 46 | 0.369 | 0.414 | 0.478 | 1.112× | 1.290× | 1.105× |
| 47 | 0.388 | 0.404 | 0.457 | 1.077× | 1.172× | 1.046× |
| 145 | 0.396 | 0.396 | 0.480 | 1.003× | 1.197× | 1.018× |
| 245 | 0.391 | 0.398 | 0.454 | 1.003× | 1.145× | 1.107× |
| 345 | 0.345 | 0.383 | 0.442 | 1.138× | 1.266× | 1.107× |

Identical-source A/A paired ratios range **0.926–1.107×**. This exposes meaningful process/timing variation: near-1.00× results should be read as parity, not statistically established small wins. The improvement over the preceding precision implementation is visible across all six datasets. Paired ratios are medians of per-round ratios and need not equal the ratio of displayed medians. No confidence interval or universal speed claim is inferred from these measurements.

[Complete controlled results](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-7028b19017fae0d91493) include every sample, warmup, source/fixture/harness hash and loaded-worker inventory. All four variants pass accuracy and integrity checks.

## Gaussian and unequal-scale gains

The existing 50,000 × 32 / 256-query / k=5 protocol was rerun unchanged on Gaussian and unequal-scale inputs, seeds 151–153 and 251–253. It uses one-second warmups, 31 alternating pairs and same-process current-code A/A controls.

| Workload | Operation | Current median range, ms | v3 median range, ms | Median paired speedup |
|---|---|---:|---:|---:|
| Gaussian | Fresh fit + query | 0.664–0.693 | 0.792–0.841 | **1.189–1.223×** |
| Gaussian | Query | 0.410–0.423 | 0.483–0.494 | **1.169–1.178×** |
| Unequal scale | Fresh fit + query | 0.746–0.765 | 0.823–0.840 | **1.087–1.114×** |
| Unequal scale | Query | 0.483–0.566 | 0.482–0.569 | **0.981–1.002×** |

[Complete Gaussian/scale results](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-54ee805031cf3fa0e8c8). All accuracy, frozen implementation, method/global identity and source/harness integrity checks pass. Unequal-scale query ratios are reported without claiming a query improvement; the fresh-fit saving and Gaussian gains are preserved.

## Full benchmark rerun

The original nine-operation, three-dataset benchmark retains **243/243 paired calls meeting the 2× default-thread sklearn target**; the minimum ratio is **2.333×**. This includes fresh fit-plus-query, neighbors, indices, classifier prediction/probability and regression with both uniform and distance weights. Fit-only and resident timings are reported separately. Current, v3, single-thread/default-thread sklearn and resident outputs all pass on these datasets.

| Operation | Median speedup vs v3, three seeds | Median speedup vs default sklearn |
|---|---:|---:|
| `fit_query` | 1.249–1.352× | 2.768–3.547× |
| `kneighbors` | 1.247–1.336× | 3.065–3.453× |
| `indices` | 1.264–1.475× | 2.789–3.444× |
| `predict_uniform` | 1.346–1.507× | 3.166–4.226× |
| `proba_uniform` | 1.226–1.498× | 2.599–3.282× |
| `regress_uniform` | 1.261–1.501× | 2.691–3.184× |
| `predict_distance` | 1.219–1.344× | 3.097–3.118× |
| `proba_distance` | 1.349–1.529× | 3.256–3.600× |
| `regress_distance` | 1.331–1.362× | 3.026–3.365× |

[Complete nine-operation results](../benchmarks/results/knn-128-final-original-vs-v3-2026-10-30.json).

The entire eleven-shape panel ran twice: the established five-call/two-warmup protocol and a longer 31-call/ten-warmup repeat matching the preceding regression investigation. Both are retained. Every current, v3 and resident accuracy audit passes. A ratio below one is a measured slowdown; the optimization does not imply a win on every shape.

| Shape | Short-panel query speedup | Longer current query, ms | Longer v3 query, ms | Longer query speedup | Longer fresh-fit speedup |
|---|---:|---:|---:|---:|---:|
| `single_8_k5` | 0.888× | 0.331 | 0.303 | 0.913× | 1.004× |
| `batch_8_k5` | 0.915× | 0.459 | 0.453 | 0.987× | 1.034× |
| `batch_32_k5` | 0.982× | 0.752 | 0.695 | 0.925× | 1.173× |
| `batch_32_k32` | 0.990× | 1.635 | 1.588 | 0.971× | 0.997× |
| `batch_128_k5` | 1.378× | 0.584 | 0.722 | 1.238× | 1.207× |
| `batch_1024_k5` | 0.978× | 0.804 | 0.904 | 1.124× | 1.270× |
| `batch_32_k64` | 0.937× | 1.144 | 1.015 | 0.887× | 1.035× |
| `batch_16_k5` | 0.980× | 0.518 | 0.642 | 1.240× | 1.150× |
| `single_32_k5` | 0.929× | 0.360 | 0.340 | 0.944× | 0.998× |
| `batch_8_k32` | 0.942× | 1.636 | 1.806 | 1.104× | 1.090× |
| `batch_50000_32_k5` | 1.314× | 1.268 | 1.677 | 1.323× | 1.252× |

[Short full panel](../benchmarks/results/knn-128-final-panel-vs-v3-2026-10-30.json); [longer full panel](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-9caa5f759c7a73b773b7).

The distribution suite reran Gaussian, clustered, offset-1e4 and unequal-scale data at seeds 151, 251 and 252, using nine calls after three warmups. Current accuracy passes all 12 dataset cases for query and fresh fit-plus-query, including the previously fixed offset ordering failure.

| Seed | Current | v3 | sklearn default |
|---|---|---|---|
| 151 | Pass | Pass | Pass |
| 251 | Pass | FAIL | FAIL |
| 252 | Pass | Pass | Pass |

Seed 251 still exits 1 because v3 fails the offset-order oracle and sklearn has an unequal-scale copied-row residual of approximately `5.048e-6`. Both full panels retain sklearn’s 1024-feature copied-row residual of approximately `2.473e-6`; their aggregate quality flags remain false while current/reference/resident quality is true. These comparison-baseline failures were not removed or reclassified as passes.

Raw distribution records: [seed 151](../benchmarks/results/knn-128-final-distributions-seed151-2026-10-30.json), [seed 251](../benchmarks/results/knn-128-final-distributions-seed251-2026-10-30.json), [seed 252](../benchmarks/results/knn-128-final-distributions-seed252-2026-10-30.json).

## Validation and reproducibility

The [652-test run](../benchmarks/results/knn-128-full-tests-2026-10-30.log) covers existing precision, parity, ownership, validation, benchmark-harness, patching and neighbor estimator-compliance checks. The [21 new shortlist tests](../tests/unit/test_neighbors_shortlist.py) use independent float64 and Python-integer oracles across launch shapes, the known rounded offset tie, true ties, missing/invalid candidates, subnormal coordinates, overflow, strict nextafter certificate boundaries, prior-certificate failure, and tiled self-exclusion with a final one-row tile. The [new-test run](../benchmarks/results/knn-128-shortlist-tests-2026-10-30.log) passes. Ruff and `git diff --check` pass.

All GPU workloads ran sequentially on Apple M4 Max, Python 3.13, MLX 0.32.2, NumPy 2.5.2 and sklearn 1.9.0. The final [60-source archive](../benchmarks/reference/knn-128-final-2026-10-30.zip) and [SHA-256 manifest](../benchmarks/reference/knn-128-final-2026-10-30.manifest.json) match the production sources used throughout these runs. The [harness/test archive](../benchmarks/reference/knn-128-final-harnesses-2026-10-30.zip) has its own [manifest](../benchmarks/reference/knn-128-final-harnesses-2026-10-30.manifest.json). v3 remains the frozen `knn-pre-v4-2026-10-28.zip`; the prior precision build is `knn-precision-verified-2026-10-30.zip`.

[Execution record](../benchmarks/results/knn-128-final-execution-2026-10-30.json) retains commands, elapsed times and exit statuses; [compact machine-readable summary](../benchmarks/results/knn-128-final-summary-2026-10-30.json) retains the reported calculations. Probe/initial failed-build logs remain separate from the final `knn-128-final-*` acceptance records.

```sh
python3 -m benchmarks.knn_128_control --output /tmp/knn-128-control.json
python3 -m benchmarks.knn_fresh_dispatch --output /tmp/knn-dispatch.json
python3 -m benchmarks.knn_acceleration \
  --reference-archive benchmarks/reference/knn-pre-v4-2026-10-28.zip \
  --output /tmp/knn-operations.json
python3 -m benchmarks.knn_pipeline --profile full --resident --warmup 10 --repeats 31 \
  --reference-archive benchmarks/reference/knn-pre-v4-2026-10-28.zip \
  --output /tmp/knn-panel.json
python3 -m benchmarks.knn_review_distributions --seed 251 --output /tmp/knn-distributions.json
```
