# Warmed Digits and cold-start KNN latency — October 30, 2026

This change improves the automatic CPU path and avoids initializing the GPU
for eligible host fits. The [full sklearn panel](knn-latency-benchmarks-2026-10-30.md),
[paired implementation comparison](knn-latency-control-2026-10-30.md), and
[fresh-process cold results](knn-latency-cold-2026-10-30.md) retain separate
measurements of steady operations, first queries, setup, and total startup.

## Measured results

**Digits is 1.82–2.45× faster than the preceding automatic version across all
27 continuously warmed seed/operation comparisons.** The duplicate current
worker improves by 1.84–2.53×. These are median paired speedups; the respective
ratio-of-medians ranges are 1.82–2.44× and 1.85–2.55×. The large CPU-dispatch
regression is fixed, but not every observation beats every GPU reference.

The full 72-case sklearn panel retains all 648 case/operation medians:

| sklearn configuration | Current faster | At least 2× faster |
| --- | ---: | ---: |
| Default-thread brute force | 625/648 | 594/648 |
| Single-thread brute force | 648/648 | 300/648 |
| Default-thread automatic algorithm | 648/648 | 583/648 |

All **162 small/single-query medians** beat each comparator. The minimum
speedups are 2.287×, 1.050× and 1.582× in the table's order. All **108
clustered/offset medians** also win, with minima of 2.130×, 1.348× and 2.150×.
The 23 default-brute misses remain in the 8-feature, 32-neighbor batch family.
There is no universal 2× speedup.

Across the 60 Gaussian/clustered/offset/unequal-scale preservation cells, the
median of paired speedups against the preceding automatic version is 1.000×,
with individual cells ranging from 0.919× to 1.066×. No preservation cell is
more than 5% slower in both current workers under both timing summaries.
That descriptive check is not a statistical significance test and does not
mean every cell improved. All 21 GPU device source files are byte-identical
to both previous snapshots.

The **450 fresh-process launches** establish that every one of the 18 fixture
medians wins against all three sklearn configurations for both first query
and first fit-plus-query. The smallest first-query speedup is 2.775×; the
smallest first-fit/query speedup is 3.211×. Against preceding automatic
dispatch, first queries improve 7.46–43.59× and first fit/query improves
63.39–221.10×. Those large API-only ratios include relocating native loading
to estimator imports. With imports and interpreter/input startup charged,
process-to-result improvement is **1.044–1.060×** versus preceding automatic
dispatch, with all 18 medians faster.

Total startup still is not uniformly faster than sklearn. Process-to-result
wins are **16/18**, **17/18** and **7/18** against default brute, single-thread
brute and auto, respectively. The worst ratio against auto is 0.984×,
approximately 1.6% slower. Estimator setup alone takes roughly 0.77–0.78 seconds
here and remains slower than sklearn auto in all 18 medians. The separate
[cold report](knn-latency-cold-2026-10-30.md) retains every phase and exception.

The broad panel's first-per-fixture observation is a different experiment:
its workers have already processed other fixtures. All 18 targeted combined
fit/query observations beat all three sklearn configurations, but Breast
Cancer seed 151's query alone remains 0.333 ms versus 0.322 ms for single-thread
sklearn, a 10.8 µs gap. That observation is retained alongside the repeated
fresh-process results; neither substitutes for the other.

The full implementation control also retains these limitations:

- Digits seed 252 distance-weighted probabilities are slower than the older
  GPU reference in both current workers: paired speedups 0.882× and 0.929×.
- Offset 4096 × 128, seed 252, fresh fit/query is slower in both workers than
  the current forced-GPU and older GPU references. Against the older GPU,
  paired speedups are 0.827× and 0.917×.
- Across all 222 controlled cells, 19 mostly small CPU cells are more than
  5% slower than preceding automatic dispatch in both current workers under
  both summaries. These do not include the 60 preservation cells. All targeted
  sklearn median comparisons still win.

Both the [independent audit](../benchmarks/results/knn-latency-final-2026-10-30-independent-audit.json)
and the [additional audit](../benchmarks/results/knn-latency-final-2026-10-30-additional-audit.json)
retain the raw pointers and source/runtime verification. All **14,472 current
broad accuracy audits** pass; every current/control cold, warmup, probe and
retained audit passes. Sklearn's canonical tie alternatives and precision
failures are listed separately and never removed from timing denominators.

### Focused repeat of warmed exceptions

After inspecting the full run, a separate frozen-source repeat covered all three
seeds of Digits, Breast Cancer and 4096 × 128 offset data: **nine fixtures,
60 case/operation cells, five workers and 31 paired rounds**, with continuous
warmup increased from 0.25 to one second. All APIs in the two small-data families
and both search/fresh-fit operations in the offset family remain included.
This longer-warmup repeat supplements the full run; it does not replace it,
and absolute times across the two protocols are not used to infer a change.

Digits again improves in all 27 cells versus preceding automatic dispatch:
paired speedups **1.842–2.534×**, or **1.806–2.463×** in the duplicate current
worker. None of the six selected preceding-auto misses, one current-GPU miss,
or two older-GPU misses from the full run repeats under the same descriptive
criterion of more than 5% runtime slowdown in both workers under both summaries.
The remaining 13 preceding-auto misses outside these selected families were not
retested, and the original observations remain in the full report.

Offset seed 252 fresh fit/query still has preceding-auto paired speedups of
0.869× and 0.907×; its ratio-of-medians results do not both meet that slowdown
criterion. These differing summaries remain visible and preclude a zero-overhead
claim.

A different older-GPU exception appears: Digits seed 252 distance-weighted
regression has paired speedups 0.884× and 0.882×, with ratio-of-medians values
0.899× and 0.887×. There is no such repeat miss against preceding automatic or
current forced GPU. The repeat therefore supports the large Digits improvement
while leaving a claim of uniformly faster performance against every historical
GPU observation unsupported. See [every repeat cell](knn-latency-warm-repeat-2026-10-30.md)
and its [independent audit](../benchmarks/results/knn-latency-final-2026-10-30-warm-exceptions-repeat-audit.json).

## Implementation

The preceding automatic build fixed most small-workload disadvantages against
sklearn, but its bounded-integer CPU search made several passes over the entire
distance matrix. On continuously warmed Digits workloads, the preceding GPU
implementation could be about 2.3× faster. The new native selector combines
the Gram distance expression and canonical top-k selection in one pass,
avoiding separate full-matrix norm additions and repeated NumPy minimum scans.
It changes the CPU implementation, not the data or metric. The dispatch
eligibility rules remain conservative; there is no dataset-name check, query
counter, timing-based estimator hook, or result cache.

The integer certificate requires integral original float32 coordinates and
`4 × D × M² ≤ 2²⁴`, where `M` bounds the absolute training and query values.
Every norm and absolute dot product is therefore at most `2²²`. All products,
dot partial sums, and intermediate terms of `norm(q) + norm(x) - 2 dot(q,x)`
are exactly representable integers in float32. The native selector validates
the bounded integer intermediates and scans references in original-index order.
A vector block can be skipped when its minimum is at least the current kth
score: a later equal-score reference cannot displace an earlier index. This
preserves true ties and membership at the kth boundary. Squared integer results
convert exactly to float64 before the public square root.

Coordinate integrality is checked by comparing original float32 bits with the
bits of their truncated values. A subnormal cannot pass as integer zero even
if floating comparison flushes tiny inputs. Native validation separately rejects
nonzero magnitudes below one before its floating checks. Buffer shape, dtype,
alignment, writable output and non-overlap checks remain explicit. Self identities
are excluded, and explicit training tiles retain the certified portable path.

A second native helper computes feature ranges in one pass. It compares ordered
integer representations of the original float32 bits and reconstructs extrema
in float64, preserving subnormals and extreme finite values. The existing
concentration decision, column permutation and matching training/query snapshot
remain unchanged. This reduces real first-query preparation; it does not move
that preparation into fit. The proposed `np.take` permutation alternative was
slower and was not retained.

Automatic fits with `N ≤ 2,048, D ≤ 64` or `N ≤ 16,384, D ≤ 32` now validate
and own host storage without probing the GPU. GPU availability remains unknown
until a query actually needs it. Such a query pays the probe and upload inside
its measured execution. Larger fits retain their established device preparation.
Explicit device choices and forced backends remain fixed. Unknown availability
also survives serialization, so an unprobed model can later use GPU rather than
being mistaken for a CPU-only model. CPU classifier predictions no longer
import the GPU voting implementation before returning through their host path.

The CPU search implementation is now an ordinary dependency of the neighbor
estimator API. Loading the native extension moves into API import instead of
the first query. **This relocates the library-loading cost; it does not eliminate
it.** Cold reports charge it to estimator setup and retain setup-plus-fit/query
and process-to-result totals. There is no data-dependent prewarm. The separate
GPU probe and device initialization really are avoided on these automatic host paths.

All `mlxlearn_device/*` source files remain unchanged from the preceding
automatic and small-GPU snapshots. The Gaussian, clustered, offset and GPU
precision kernels retain their exact preceding implementation. Public-boundary
costs are still checked by the paired measurements.

## Correctness and build validation

The [final relevant suite](../benchmarks/results/knn-latency-full-tests-final-2026-10-30.log)
passes **1,191 tests**; five unrelated LogisticRegression/SVC cases are
deselected. It includes exact ordering, true and rounded ties, weighted votes,
integer certificate boundaries, extreme coordinates and subnormals, self
exclusion, explicit tiling, malformed/aliased buffers, input ownership, concurrent
permutation preparation, pickle/refit, patching, validation, sklearn estimator
compliance, and benchmark integrity. Portable precision fixtures explicitly
disable all three optional native entry points. The existing distance and
prediction tolerances are unchanged.

Ruff and the wheel/source-distribution builds pass.
[Package validation](../benchmarks/results/knn-latency-package-validation-final-2026-10-30.json)
checks all 63 source files in both distributions and matches the wheel's native
binary to the measured runtime. The tested wheel targets
`cp313-cp313-macosx_11_0_arm64`. Native compilation happens at build time;
MLX may separately compile GPU kernels on first use. An absent or older optional
CPU extension retains the correct portable implementation, but its performance
is not described by the native results.

## Measurement protocol

The broad run preserves all 72 dataset instances, 24 families, nine headline
search/inference operations and three sklearn configurations: default-thread
brute force, one-thread brute force, and default-thread automatic algorithm.
It retains 15 calls after five warmups per operation and variant. Fit-only and
first-fit/query-per-fixture records remain separate. A fixture's first call in
a persistent worker is not a new interpreter; earlier fixtures can initialize
libraries and compiler caches.

The control keeps the same 48 fixtures and 222 case-operation cells: 162
small/single-query cells and 60 Gaussian, clustered, offset and unequal-scale
query/fresh-fit cells, including six Gaussian seeds per shape. Five workers
run current automatic dispatch, identical automatic A/A, current forced GPU,
the preceding automatic source with its exact archived native binary, and the
preceding small-workload GPU source. Each worker/operation gets 0.25 seconds
of continuous warmup followed by 31 interleaved rounds. Every cold, warmup,
probe and retained audit is saved, including failures. Both ratio-of-medians
and median-paired-ratio summaries remain visible; they are descriptive statistics,
not confidence intervals.

The new cold run starts **450 separate Python processes**: 18 target fixtures,
five repetitions, and current automatic, preceding automatic, and all three
sklearn variants. Each child performs exactly one first fit/query. No metadata
request, explicit MLX synchronization, estimator preflight, or workload warmup
precedes it. The child marks result availability before audit and metadata work.
Reports retain NumPy import, fixture loading, estimator setup, fit, query,
combined fit/query, estimator-to-result, and parent process-start-to-result
times. Interpreter and OS scheduling are part of the last measure. No OS,
filesystem, bytecode-cache, or Metal-disk-cache coldness is claimed.

All timed CPU/GPU work is serial. Public validation, owned data preparation,
NumPy boundaries, lazy search work, and required completion are inside the
appropriate timed operations. Source, runtime, tests, harnesses and fixtures
remain frozen during the final runs. Accuracy failures never remove timings
from comparator counts. Measurements concern float32 Euclidean workloads on
the tested Apple M4 Max; they do not guarantee a speedup on every machine,
metric, dataset, or individual call.

## Diagnostics and reproduction

The [integer-selector diagnostic](../benchmarks/results/knn-integer-selector-probe-2026-10-30.json)
measures all three Digits seeds against portable CPU selection.
The [range-helper diagnostic](../benchmarks/results/knn-breast-native-range-probe-2026-10-30.json)
retains all Breast seeds and original-coordinate accuracy checks.
An [isolated lazy-export experiment](../benchmarks/results/knn-lazy-export-comparison-2026-10-30.json)
kept only one initializer different and identical native binaries across both
arms. All nine NN startup comparisons were slower, so the experiment was
reverted. Its 28 process launches remain available; unused-class lazy exports
are not part of the final implementation.

```sh
python3 setup.py build_ext --inplace
python3 -m benchmarks.knn_broad \
  --current-mode auto \
  --source-manifest benchmarks/reference/knn-latency-final-2026-10-30.manifest.json \
  --repeats 15 --warmup 5 --target 2 \
  --output benchmarks/results/knn-latency-rerun-broad.json
python3 -m benchmarks.knn_auto_control \
  --source-manifest benchmarks/reference/knn-latency-final-2026-10-30.manifest.json \
  --previous-archive benchmarks/reference/knn-small-final-2026-10-30.zip \
  --previous-auto-archive benchmarks/reference/knn-auto-final-2026-10-30.zip \
  --previous-auto-runtime benchmarks/reference/knn-auto-final-2026-10-30-runtime.zip \
  --pairs 31 --warmup-seconds .25 \
  --output benchmarks/results/knn-latency-rerun-control.json
python3 -m benchmarks.knn_cold \
  --source-manifest benchmarks/reference/knn-latency-final-2026-10-30.manifest.json \
  --previous-auto-archive benchmarks/reference/knn-auto-final-2026-10-30.zip \
  --previous-auto-runtime benchmarks/reference/knn-auto-final-2026-10-30-runtime.zip \
  --repeats 5 --output benchmarks/results/knn-latency-rerun-cold.json
```

The [source archive](../benchmarks/reference/knn-latency-final-2026-10-30.zip),
[native runtime and build provenance](../benchmarks/reference/knn-latency-final-2026-10-30-runtime.zip),
and [138-file harness/test/build archive](../benchmarks/reference/knn-latency-final-2026-10-30-harnesses.zip)
identify the exact implementation and tools. Archived binaries require matching
architecture and Python ABI; rebuild when they differ.

The [execution record](../benchmarks/results/knn-latency-final-2026-10-30-execution.json)
retains commands, artifact hashes, package validation, environment, timing order,
and the five production files changed from preceding automatic dispatch.
