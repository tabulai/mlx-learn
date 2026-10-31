# kNN precision and ordering fix — October 30, 2026

The reproduced rounded-distance ordering and Gram-cancellation failures are fixed. GPU comparisons retain original float32 coordinate precision, and CPU search ranks direct float64 distances. The Gaussian speed gains and the unequal-scale fresh-fit improvement remain in the controlled comparison against frozen v3. All **631 relevant tests pass**, and every current-implementation numerical audit in the full benchmark rerun passes. This is not a claim that every shape is faster; the complete shape panel below retains slower cases.

## What changed

- Two unequal squared distances that round to the same float32 value no longer become an index tie. Conservative intervals settle ordinary comparisons; guarded double-single calculations resolve close pairs; unresolved cases compare exact dyadic integers decoded from the original coordinate bits. Only true equality uses the smaller training index.
- Direct, compact and integer searches certify omitted candidates against the precise cutoff. Direct partition merges retain an extra small-k candidate, so most close cutoffs need only shortlist ordering. Uncertified membership triggers a complete original-coordinate scan.
- General GPU search uses Gram values only as conservative lower bounds. It retains `k+8` candidates plus an omission barrier, refines original coordinates, and falls back when the omitted bound is insufficient. CPU selection uses bounded direct float64 tiles, removing the former unverified Gram shortlist.
- Candidate refinement shares its extra precision across GPU lanes and comparisons. Large lists sort first and verify adjacent precise order before doing a complete candidate ranking. Larger Gram shortlists use sorting instead of a long insertion-based top-k scan.
- Native `fit_neighbors` owns a bitwise copy of training data. Its former arithmetic copy flushed subnormal coordinates. The cached scale guard and large-array validation optimization remain in place.

The [numerical argument](knn-precision-ordering.md) documents the interval bounds, guarded precision filters, exact accumulators, and complete fallback. Returned squared distances remain float32; exact GPU neighbor ordering does not imply arbitrary-precision distance output. CPU ordering is float64, not arbitrary precision.

## Reproduced failures and validation

For offset data at seed 251, query 79, training indices 1902 and 43153 have exact squared distances `21.925426483154297` and `21.92542552947998`. Both round to the same float32 squared value. v3 puts 1902 first; the fixed implementation correctly selects **43153 first**, including k=4 membership. Tests cover both CPU and GPU, distance-free output, and classifier/regressor predictions at the disputed boundary.

The monotone matrix `arange(8192*32).reshape(8192,32)` formerly returned training row 3 for both copied queries 3 and 4 in the general path. A Gram shortlist discarded the correct candidate before refinement. Direct CPU selection and certified GPU screening now return all tested copied rows correctly, including forced tiling.

The [33 precision regression cases](../tests/unit/test_neighbors_precision.py) also exercise true ties, self-exclusion, one-row tile tails, exact Python-integer oracles across finite float32 exponents, close moderate-scale pairs through 1024 features, native subnormal ownership, and correction of a large sorted candidate list with invalid tails. The [main verification run](../benchmarks/results/knn-precision-verified-tests-2026-10-30.log) passes 629 tests; the [two final large-list cases](../benchmarks/results/knn-precision-verified-large-list-tests-2026-10-30.log) also pass. These include existing parity, ownership, validation, benchmark-harness, patching and neighbor estimator-compliance checks. Ruff passes for the affected neighbor modules, regression tests and benchmark harness.

## Controlled comparison against v3

This comparison uses 50,000 training rows, 32 features, 256 queries, k=5, and float32 inputs. Unequal feature scales are `logspace(-2,2,32)`. Seeds 151–153 and 251–253 each receive one second of warmup per variant and 31 alternating retained pairs, with starting order rotated across seeds. Public query timings include validation, transfer and synchronization; fresh fit-plus-query includes all preparation. Every result is checked outside timing against exhaustive direct float64 distances on the original float32 inputs. Exact-index and distance tolerances are unchanged.

| Workload | Operation | Current median range, ms | v3 median range, ms | Median paired speedup, v3/current |
|---|---|---:|---:|---:|
| Gaussian | Fresh fit + query | 0.661–0.703 | 0.797–0.854 | **1.193–1.224×** |
| Gaussian | Query | 0.407–0.425 | 0.482–0.497 | **1.166–1.198×** |
| Unequal scale | Fresh fit + query | 0.730–0.768 | 0.798–0.838 | **1.093–1.110×** |
| Unequal scale | Query | 0.486–0.505 | 0.484–0.500 | **0.987–1.002×** |

Each range spans six seeds; paired speedups are medians of per-pair ratios, not ratios of the displayed ranges. Identical-current-code A/A ratios span 0.989–1.012×. Unequal-scale fresh latency is 8.5–9.9% lower than v3. Unequal-scale query differences are small and are reported without claiming a query speed gain.

Full records: [six-seed paired results](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-2ab9f553ad4500f2bc1e). The harness loads the actual frozen v3 index, kernels and public validation methods. It checks static dependencies, globals, restored hooks, and source/harness hashes; all these integrity checks pass.

## Full rerun

The original nine-operation target runs on three datasets (base seeds 41–43, dataset seeds 51–53), with three warmups and nine retained pairs per operation. It includes fresh fit-plus-query, neighbors, indices-only output, classifier prediction/probability and regression, each with uniform and distance weights. Fit-only and resident timings remain separately available.

All 243/243 retained target pairs meet the 2× default-thread sklearn requirement; the slowest paired speedup is **2.190×**. Current, v3, both sklearn thread settings and resident numerical audits pass on these three datasets.

| Operation | Median speedup vs v3, three seeds | Median speedup vs default sklearn |
|---|---:|---:|
| `fit_query` | 1.122–1.441× | 2.519–3.153× |
| `kneighbors` | 1.391–1.640× | 3.387–5.851× |
| `indices` | 1.287–1.333× | 3.049–4.930× |
| `predict_uniform` | 1.235–1.499× | 2.586–3.024× |
| `proba_uniform` | 1.326–1.518× | 3.088–4.125× |
| `regress_uniform` | 1.271–1.355× | 2.487–3.070× |
| `predict_distance` | 1.295–1.470× | 2.974–3.286× |
| `proba_distance` | 1.288–1.335× | 2.860–3.627× |
| `regress_distance` | 1.376–1.421× | 3.274–3.654× |

[Complete original-operation results](../benchmarks/results/knn-precision-verified-original-vs-v3-2026-10-30.json).

The eleven-shape panel retains five timed calls after two warmups, interleaved across isolated current/v3/sklearn workers, plus resident results. These short intermittent measurements have more variability than the duration-warmed controlled comparison. Every current, frozen-v3 and resident numerical audit passes.

| Shape case | Current query, ms | v3 query, ms | Query speedup | Fresh fit + query speedup |
|---|---:|---:|---:|---:|
| `single_8_k5` | 0.328 | 0.316 | 0.964× | 0.907× |
| `batch_8_k5` | 0.515 | 0.455 | 0.884× | 0.894× |
| `batch_32_k5` | 0.448 | 0.417 | 0.932× | 0.934× |
| `batch_32_k32` | 1.838 | 1.515 | 0.824× | 1.026× |
| `batch_128_k5` | 0.861 | 0.692 | 0.803× | 0.918× |
| `batch_1024_k5` | 0.559 | 0.556 | 0.994× | 1.094× |
| `batch_32_k64` | 0.789 | 0.658 | 0.833× | 1.067× |
| `batch_16_k5` | 0.518 | 0.572 | 1.104× | 0.936× |
| `single_32_k5` | 0.389 | 0.353 | 0.905× | 0.957× |
| `batch_8_k32` | 1.574 | 1.562 | 0.992× | 1.010× |
| `batch_50000_32_k5` | 1.373 | 1.769 | 1.288× | 1.268× |

[Complete eleven-shape results](../benchmarks/results/knn-precision-verified-panel-vs-v3-2026-10-30.json). A speedup below 1 is a measured slowdown; no shape or failed sample is omitted.

A longer follow-up retains **31 calls after 10 warmups** on every shape, using the same frozen source and fixtures. It resolves the large-k concern, but confirms a remaining 128-feature cost: **18.3% higher query latency** (0.912 versus 0.771 ms). Single-query cases are about 7% slower. The main 50,000 × 32 Gaussian workload remains faster. Precision is fixed and the targeted gains remain; a uniform speed win across shapes has not been achieved.

| Shape case | Longer-run query speedup | Longer-run fresh fit + query speedup |
|---|---:|---:|
| `single_8_k5` | 0.936× | 0.959× |
| `batch_8_k5` | 1.033× | 1.097× |
| `batch_32_k5` | 0.959× | 0.976× |
| `batch_32_k32` | 0.961× | 1.014× |
| `batch_128_k5` | 0.845× | 0.946× |
| `batch_1024_k5` | 1.250× | 1.131× |
| `batch_32_k64` | 1.146× | 1.036× |
| `batch_16_k5` | 1.118× | 1.021× |
| `single_32_k5` | 0.933× | 1.032× |
| `batch_8_k32` | 1.087× | 0.950× |
| `batch_50000_32_k5` | 1.235× | 1.284× |

[Complete longer panel](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-d985a7266fe3eaf453fa). Current, v3 and resident quality checks and source integrity all pass; sklearn’s existing copied-row residual remains visible.

The distribution suite covers Gaussian, clustered, offset-1e4, and unequal-scale data for seeds 151, 251 and 252, with three warmups and nine retained calls per operation. Current output passes all 12 distribution/seed cases, for both query and fresh fit-plus-query, including the formerly failing offset case.

| Seed | Current | Frozen v3 | sklearn default |
|---|---|---|---|
| 151 | Pass | Pass | Pass |
| 251 | Pass | FAIL | FAIL |
| 252 | Pass | Pass | Pass |

The seed-251 distribution run intentionally retains exit status 1: v3 still fails the offset ordering oracle, and sklearn retains an unequal-scale copied-row residual of approximately `5.048e-6`. The full shape panel retains sklearn’s 1024-feature copied-row residual of approximately `2.473e-6`; its aggregate quality flag is false while current/reference/resident quality is true. These are comparison-baseline failures, not unresolved current-implementation failures. No oracle, tolerance or failed sample was changed to make them pass.

Raw distribution records: [seed 151](../benchmarks/results/knn-precision-verified-distributions-seed151-2026-10-30.json), [seed 251](../benchmarks/results/knn-precision-verified-distributions-seed251-2026-10-30.json), [seed 252](../benchmarks/results/knn-precision-verified-distributions-seed252-2026-10-30.json).

## Reproducibility

All GPU workloads ran sequentially on Apple M4 Max with Python 3.13, MLX 0.32.2, NumPy 2.5.2 and scikit-learn 1.9.0. The final 59-source [archive](../benchmarks/reference/knn-precision-verified-2026-10-30.zip) and [SHA-256 manifest](../benchmarks/reference/knn-precision-verified-2026-10-30.manifest.json) match the production files used throughout this rerun. The [frozen harnesses and regression tests](../benchmarks/reference/knn-precision-verified-harnesses-2026-10-30.zip) have a separate [manifest](../benchmarks/reference/knn-precision-verified-harnesses-2026-10-30.manifest.json). Frozen v3 is `knn-pre-v4-2026-10-28.zip`, matching the preceding comparisons.

[Execution record](../benchmarks/results/knn-precision-verified-execution-2026-10-30.json) records commands, timings and exit statuses; [compact machine-readable summary](../benchmarks/results/knn-precision-verified-summary-2026-10-30.json) contains the reported calculations. Intermediate `knn-precision-*` and `knn-precision-final-*` results remain in the repository, including pre-optimization slowdowns and speed-target misses. Only `knn-precision-verified-*` records describe the final source.

```sh
python3 -m benchmarks.knn_fresh_dispatch --output /tmp/knn-controlled.json
python3 -m benchmarks.knn_acceleration \
  --reference-archive benchmarks/reference/knn-pre-v4-2026-10-28.zip \
  --output /tmp/knn-operations.json
python3 -m benchmarks.knn_pipeline --profile full --resident \
  --reference-archive benchmarks/reference/knn-pre-v4-2026-10-28.zip \
  --output /tmp/knn-panel.json
python3 -m benchmarks.knn_review_distributions --seed 251 --output /tmp/knn-distributions.json
```

The subsequent [128-feature optimization](knn-128-feature-fix-2026-10-30.md)
addresses the remaining overhead recorded here while retaining these precision
fixes. This page and its benchmark records remain the historical result.
