# KNN first-query and warmed-search follow-up — October 30, 2026

This is the intermediate CPU-kernel revision frozen before the user authorized
removing direct sklearn neighbor subclass identity. Its full broad, balanced
control, legacy control and fresh-model panels completed. Its fresh-process
panel was interrupted after that authorization, with eight cases recorded and
`acceptance_pass: false`; it is not a completed cold-start result. The subsequent
startup revision will be reported separately. These earlier archives and raw
measurements remain unchanged.

This revision improves serial native CPU search while preserving the existing
sklearn subclass relationship, numerical contract and GPU kernels. The startup
investigation and the search optimizations are separate: no unproven import
rearrangement is included.

## Full sklearn panel

All **14,472 current accuracy audits** pass. Across the 72 fixture instances and
648 case/operation medians, the results are:

| sklearn configuration | Current faster | At least 2× faster |
| --- | ---: | ---: |
| Default-thread brute force | 623/648 | 588/648 |
| Single-thread brute force | 648/648 | 299/648 |
| Default-thread automatic algorithm | 648/648 | 603/648 |

All **162 targeted small/single-query medians** beat every comparator; the
minimum speedups are 2.622×, 1.255× and 1.649× in the table's order. All **108
clustered/offset medians** also win, with minima 2.256×, 1.324× and 2.270×.
This still is not a universal 2× speedup or a win over every sklearn configuration
on every workload.

The broad panel's first-per-fixture observations now win on all eighteen target
fixtures against all three sklearn configurations, for both first query and
combined first fit/query. Their narrowest wins against single-thread sklearn
are 1.212× and 1.114×, respectively. This closes the previously observed 10.8 µs
first-query gap in this full run. These observations occur in persistent workers;
they are distinct from repeated fresh-model and fresh-process measurements.

## Implementation and precision

The bounded-integer selector processes sixteen references per block, with
independent validation accumulators. Every input is still validated, including
raw-bit subnormal rejection. Every score block is checked for negative values
before it can be skipped. The coordinate/integer-intermediate certificate and
exact float32 Gram expression are unchanged.

For `k=5`, five scalar score/index pairs replace a dynamically indexed selection
buffer. References still arrive in ascending original-index order, and an equal
score is inserted after existing equal scores. A block whose minimum is at
least the retained fifth score cannot displace an earlier equal-score index.
The general-k implementation remains available. There is no dataset-name check,
result cache, query counter, reduced candidate set or relaxed accuracy threshold.

The direct native scan is factored into compile-time variants. On completed
concentrated-feature permutation snapshots, a conservative multiplication bound
replaces division plus `nextafter` when updating the rejection cutoff. With
`e = 16(D+1) eps32`, screened dimensions give
`2^-18 <= e <= 65 * 2^-19`. The identity
`(1+2e)(1-e) = 1+e-2e²` leaves a margin greater than `2^-19`, far larger than
float64 multiplication rounding. Thus `upper * (1+2e)` is a safely looser cutoff
than `upper / (1-e)`. Existing range guards keep nonzero screened squared sums
normal and finite; zero remains zero. Exact dyadic comparison, ambiguous ties,
and reported-distance conversion are unchanged.

The optional new native entry has the existing seven-argument calling shape.
An older binary falls back to the original entry through a capability check.
The original query entry is retained, and portable CPU fallback remains tested.
Native execution stays serial; no thread or process scheduling policy is changed.
The one-coordinate rejection experiment was slower or unstable and was removed.

Targeted probes, preceding the full rerun, found roughly 21–22% faster public
Digits queries and 27–31% faster Breast Cancer queries. Iris, Wine and Gaussian
single-query engine checks also improved. These probes are diagnostics rather
than replacements for the full panels:

- [Final selector/scan probe](../benchmarks/results/knn-throughput-native-topfive-probe-2026-10-30.json)
- [Final-binary small/Gaussian checks](../benchmarks/results/knn-throughput-native-single-query-probe-2026-10-30.json)

## Startup compatibility boundary

The [startup investigation](knn-startup-investigation-2026-10-30.md) retains all
108 fresh-process launches from two rejected experiments. Lazy package exports
and earlier native loading did not establish a repeatable total-startup gain,
so both were reverted. No startup cost is hidden in background work or prewarm.

An instrumented trace attributes about 129 ms to the additional import chain
required by `sklearn.neighbors` beyond sklearn's root. Avoiding that chain would
require changing direct estimator inheritance. The current implementation
preserves genuine `isinstance` compatibility with stock sklearn neighbor
classes. The 129 ms attribution is not a measured speedup prediction for an
unimplemented standalone estimator. Process-to-result measurements remain
separate from API-only fit/query timings.

## Measurement design

The established broad panel retains 72 fixture instances in 24 families, nine
headline operations, three sklearn configurations, five warmups and fifteen
retained calls. Fixtures, original-coordinate oracles and tolerances are unchanged.
The three configurations are default-thread brute force, single-thread brute
force, and the default-thread automatic algorithm.

The full implementation control retains 48 fixtures and 222 operation cells,
including all 162 target cells and 60 Gaussian/clustered/offset/unequal-scale
preservation cells. Five isolated workers run current automatic dispatch,
identical automatic A/A, current forced GPU, the immediate preceding
`knn-latency-final` automatic source with its exact archived native runtime, and
the historical `knn-small-final` GPU source. Warmup is 0.25 seconds per worker
and operation, followed by 31 interleaved rounds.

The previous alternating order placed each worker in only two positions. Saved
results show substantial position and timing-phase sensitivity, without proving
its hardware cause. The new full control explicitly uses a ten-row Williams
schedule. Its first thirty rounds balance positions and within-round
predecessors; all thirty-one rounds remain in every headline summary. Across-round
previous requests, time since warmup, positions, and phase splits are recorded
separately. This protocol change is disclosed rather than silently applied to
historical results.

A predeclared legacy-order repeat retains all three seeds of Digits, Breast
Cancer and 4096 × 128 offset data: nine fixtures, sixty cells, five workers,
31 rounds and one second of warmup. It matches the previous focused protocol
and supplements the balanced full control. Neither run replaces the other, and
absolute times across different warmup protocols are not treated as causal proof.

The cold panel retains 450 fresh-process launches: eighteen target fixtures,
five variants and five launches per fixture/variant. The new initialized-process
panel adds 2,700 fresh models and 5,400 audited queries: thirty balanced rounds
on eighteen target fixtures across five workers. Every request constructs and
fits a new model, then separately times its first and second queries. No model
or query is prewarmed; round zero remains in the headline statistics. This
separates first-model preparation from process startup and repeated fitted-model
reuse. Imports, input loading, fit, first query and combined timings are explicitly
scoped; fresh processes do not imply cold OS/filesystem/bytecode/Metal caches.

All timed workloads run serially. Numerical audit failures never remove timings
from comparator denominators. Both ratio-of-medians and median-paired-ratio
summaries are retained, together with slower observations. They are descriptive
statistics, not confidence intervals or a universal speed guarantee.

## Validation and reproduction

The expanded production suite passes **1,263 tests**, with five unrelated
LogisticRegression/SVC cases deselected. A separate **59 benchmark-harness tests**
pass, for 1,322 distinct tests in total. Native precision coverage includes
minimum sample counts for the five-neighbor specialization, vector/scalar tails,
true and rounded ties, subnormals, self exclusion, malformed or overlapping
buffers, portable fallback, old native binaries and concurrent permutation
preparation. An independent static review found no precision or compatibility
defect. A stale diagnostics test was updated to verify the existing automatic
host-fit reason and shape rather than the superseded fit-only crossover detail.

Ruff, wheel and source-distribution builds pass. Package validation checks all
63 source files in both distributions and confirms that the wheel contains the
measured native binary. The wheel is `cp313-cp313-macosx_11_0_arm64`; optional
native compilation occurs at build time. Portable installations keep the same
precision contract, but their performance is not represented by native timings.

- [Production tests](../benchmarks/results/knn-followup-full-tests-candidate-2026-10-30.log)
- [Harness tests](../benchmarks/results/knn-followup-harness-tests-2026-10-30.log)
- [Package validation](../benchmarks/results/knn-followup-package-validation-2026-10-30.json)
- [Frozen source](../benchmarks/reference/knn-followup-final-2026-10-30.zip)
- [Native binary and build provenance](../benchmarks/reference/knn-followup-final-2026-10-30-runtime.zip)
- [144-file harness, test and build snapshot](../benchmarks/reference/knn-followup-final-2026-10-30-harnesses.zip)
- [Supplemental exploratory probe drivers](../benchmarks/reference/knn-followup-diagnostic-drivers-2026-10-30.zip)

The supplemental archive retains the four available exploratory CPU drivers.
Some temporary baseline paths must be restored or adjusted before rerunning
them. The exploratory single-query check used an unsaved inline driver; the
final full benchmark tools are retained independently in the harness snapshot.

The source and native runtime are frozen separately; restoring a native archive
requires matching architecture and Python ABI. Rebuild on a different platform.
Use new output paths when reproducing measurements so original evidence remains
intact. The full benchmark commands and final artifact hashes are recorded with
the results.
