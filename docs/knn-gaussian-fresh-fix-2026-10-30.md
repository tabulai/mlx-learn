# Gaussian fresh-fit overhead fix — October 30, 2026

Fresh general-path searches now evaluate reusable Gram preparation together
with their first query. The separate preparation submission and synchronization
were unnecessary because the arrays depend only on owned training data.
The [complete benchmark tables](knn-gaussian-fresh-benchmarks-2026-10-30.md)
include all measured workloads and both frozen implementation comparisons.

**The remaining 3.7% Gaussian fresh-fit regression is eliminated in all six
128-feature controlled datasets.** Current fresh fit-plus-query takes
0.647–0.708 ms. Both median paired ratios and ratios of per-worker medians
exceed 1× against both frozen implementations at every seed.

| Comparison, Gaussian 4096 × 128 | Median paired speedup | Ratio of timing medians |
| --- | ---: | ---: |
| Versus pre-regression `knn-128-final` | **1.196–1.281×** | **1.197–1.317×** |
| Versus preceding clustered/offset build | **1.220–1.330×** | **1.232–1.326×** |
| Identical current-code A/A worker | 0.962–1.010× | 0.960–1.033× |

The fresh-fit improvement is substantially larger than the observed A/A
variation. Query-only timings remain near the previous implementations; the
optimization targets setup rather than the distance-selection algorithm.
The independent 32-feature Gaussian control retains all six seeds and timing
variation rather than imposing an every-seed speedup claim on an unchanged
search path. Its fresh-fit ratios of medians span 0.981–1.021× versus the
pre-regression build and 0.988–1.039× versus the preceding build. All corresponding
paired ratios, including slower observations, are in the full tables.
Expanded 32-feature seed 552 takes 2.344–2.362 ms for queries and
2.454–2.480 ms for fresh fits across all four builds. This shared high-latency
case remains in the control; these measurements do not identify its cause.
It is part of the expanded control, outside the unchanged 72-instance sklearn panel.

Clustered and offset gains remain in the contemporaneous comparison:

| Workload | Query speedup vs pre-regression build | Fresh speedup vs pre-regression build |
| --- | ---: | ---: |
| Clusters, 50k × 32 | 2.61–2.68× | 2.06–2.10× |
| Offset +1e4, 50k × 32 | 4.36–4.77× | 2.60–2.71× |
| Clusters, 4096 × 128 | 2.37–2.49× | 2.26–2.42× |
| Offset +1e4, 4096 × 128 | 0.997–1.069× | 1.26–1.29× |

These are median paired speedups over three seeds. The narrow clustered and
offset search kernels are unchanged. Their timing differences versus the
preceding distribution-fixed build, as well as A/A measurements, remain
visible in the full tables. The 128-feature offset and unequal-scale paths
also benefit from deferred preparation. Every cold, warmup, probe and retained
audit passes in all four controlled workers; source and fixture integrity pass.

## Full sklearn rerun

The complete panel covers 72 dataset instances in 24 families, with nine
search/inference operations and 15 retained calls after five warmups. All
**14,472 current accuracy audits pass**, including cold, warmup and retained
outputs. All 72 fixture array inventories match the preceding full benchmark.

| sklearn configuration | Dataset-operation medians ≥2× | Individual paired calls ≥2× |
| --- | ---: | ---: |
| Brute, default threads | **492/648 (75.9%)** | 7,367/9,720 (75.8%) |
| Brute, one thread | **246/648 (38.0%)** | 4,052/9,720 (41.7%) |
| Auto, default threads | **528/648 (81.5%)** | 7,801/9,720 (80.3%) |

All **108 clustered/offset dataset-operation medians** beat every sklearn
configuration. Their minimum speedups are 1.920× against default-thread brute,
1.286× against single-thread brute and 1.912× against auto. The full panel
still does not support a universal 2× claim: against the fastest measured
sklearn configuration in each cell, 233/648 medians reach 2×. Small datasets,
single queries and some other workloads continue to favor sklearn.

Every baseline timing remains in the speed counts. The strict canonical audit
separately reports valid alternate exact-tie choices and numerical differences;
these are not interchangeable failure categories. The complete tables retain
all baseline audit details, raw record pointers and per-family misses.

## Implementation

Only [`_neighbors.py`](../src/mlxlearn_device/_neighbors.py) differs from the
preceding [clustered/offset build](../benchmarks/reference/knn-distribution-final-2026-10-30.manifest.json):

- `_prepare_gram_arrays` compiles the existing anchored, quarter-scaled mean,
  centering and squared-norm calculations. `_prepare_gram` stores their lazy
  outputs; the query's normal evaluation executes the complete graph. Later
  queries reuse the materialized training arrays. This removes an extra GPU
  submission/synchronization and reduces repeated Python graph construction.
- A bounded cache stores only sample row offsets and feature selectors derived
  from matrix shape. Contiguous float32 buffers read the exact same sampled
  coordinates through a flat view; strided inputs retain logical indexing.
  Sample values, classification arithmetic and routing thresholds are unchanged.
  Training values and query results never enter this global cache.

Clustered direct scans, offset integer screening, unequal-scale routing,
original-coordinate refinement, exact tie ordering, omitted-candidate bounds,
self-exclusion and exhaustive fallback are unchanged. Explicit training tiles
and automatic general-path memory budgets retain their existing behavior.
Each fit owns independent arrays, including when two fits have identical shapes
or multiple queries are constructed before evaluation.

An isolated diagnostic measured Gaussian 4096 × 128 fresh fit-plus-query at
approximately 0.693 ms with eager preparation, 0.559 ms with lazy preparation,
and 0.529 ms with compiled lazy preparation. This diagnostic motivated the
change; the frozen-worker results below are the acceptance evidence.
The [diagnostic measurements](../benchmarks/results/knn-gaussian-gram-lazy-probe-2026-10-30.json)
retain all original samples and other distributions.

## Validation and protocol

**714 relevant tests pass:** the [713-test full suite](../benchmarks/results/knn-gaussian-fresh-full-tests-2026-10-30.log)
and one subsequently added [independent-fit ownership test](../benchmarks/results/knn-gaussian-fresh-independent-fit-test-2026-10-30.log).
Five unrelated logistic/SVC cases are deselected. New regression coverage checks
deferred queries evaluated in reverse order, caller-buffer mutation, independent
same-shaped fits, 32/128/1024-feature paths, explicit tiles, self-exclusion and
duplicates, offset/rounded-close ties, huge constants and subnormal coordinates,
and CPU isolation. Metadata tests check strided layouts, shape-only caching,
different data with the same shape, in-place input changes and unavailable buffers.
Ruff and `git diff --check` pass.

The controlled run uses four persistent isolated workers: current, identical
current-code A/A, the pre-regression `knn-128-final` build, and the preceding
`knn-distribution-final` build. It covers 30 dataset instances, including six
Gaussian seeds per shape (151, 251, 252, 351, 451, 552), plus the unchanged
clustered, offset and unequal-scale fixtures at three seeds. Each query and
fresh-fit operation gets one second of continuous warmup and 31 alternating
paired rounds. Both median paired ratios and ratios of per-worker medians are
retained, alongside A/A variation and every numerical audit.

The full sklearn rerun uses the same 72 dataset instances, 24 families and nine
search/inference operations as before: 15 retained calls after five warmups,
against brute/default threads, brute/one thread and auto/default threads.
Fit-only and first-query records remain separate. Timings include public NumPy
boundaries, validation, conversions and synchronization. Inputs and numerical
tolerances are unchanged. Every output is audited outside timing against
exhaustive original-coordinate references. Baseline numerical differences and
valid alternate tie choices remain visible rather than being removed from the
speed counts.

All GPU workloads run serially on Apple M4 Max. Source and harness files remain
frozen throughout timing. These are float32 Euclidean measurements using the
forced MLX GPU backend; they do not establish universal speedups, results on
other hardware or metrics, or automatic CPU/GPU dispatch performance. Warmed
fresh-fit measurements include new model preparation on every call; one-time
compilation encountered by a fixture can affect its separately recorded first
query. Those records come from persistent workers and do not measure process
startup. Absolute times from the continuously warmed control and the five-warmup
sklearn panel should not be compared as though they used the same warmup protocol.

## Reproduction and artifacts

- [Controlled four-worker results](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-2c25c338eb8511a10104)
- [Complete three-baseline results](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-f8f82b657c035b95a37d)
- [Derived broad summary](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-e4c267f4b694d73c2156)
- [Execution record](../benchmarks/results/knn-gaussian-fresh-final-2026-10-30-execution.json)
- [Final integrity and acceptance checks](../benchmarks/results/knn-gaussian-fresh-final-2026-10-30-validation.json)
- [Frozen production archive](../benchmarks/reference/knn-gaussian-fresh-final-2026-10-30.zip) and [manifest](../benchmarks/reference/knn-gaussian-fresh-final-2026-10-30.manifest.json)
- [Frozen benchmark/test archive](../benchmarks/reference/knn-gaussian-fresh-final-2026-10-30-harnesses.zip) and [manifest](../benchmarks/reference/knn-gaussian-fresh-final-2026-10-30-harnesses.manifest.json)

```sh
python3 -m benchmarks.knn_gaussian_fresh_control \
  --source-manifest benchmarks/reference/knn-gaussian-fresh-final-2026-10-30.manifest.json \
  --output benchmarks/results/knn-gaussian-fresh-final-2026-10-30-control.json
python3 -m benchmarks.knn_broad \
  --source-manifest benchmarks/reference/knn-gaussian-fresh-final-2026-10-30.manifest.json \
  --repeats 15 --warmup 5 \
  --output benchmarks/results/knn-gaussian-fresh-final-2026-10-30-broad.json
```
