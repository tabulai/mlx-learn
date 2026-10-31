# KNN Digits GPU-control revision — October 30, 2026

The remaining Digits GPU-control slowdown is eliminated in both benchmark
orderings. **All 432 comparisons pass**, covering all 27 Digits operation cells,
both GPU baselines, both automatic replicas and both ratio statistics. The
narrowest speedup is **1.047×**. All six final panels pass their numerical and
integrity gates, including **14,472/14,472** current broad accuracy audits.

The earlier gains against sklearn remain on startup, first queries, small and
single-query workloads, and Gaussian/clustered/offset data. This is not a claim
of zero regression against the preceding automatic implementation: repeated
Breast Cancer losses and smaller first-query differences remain in the measured
results, detailed below.

## Final performance results

The [full broad panel](knn-digits-benchmarks-2026-10-30.md) covers 72 fixtures and
648 headline operation/fixture medians, with separate fit-only timings:

| sklearn configuration | Current faster | At least 2× faster | Minimum speedup |
| --- | ---: | ---: | ---: |
| Default-thread brute force | 625/648 | 587/648 | 0.663× |
| Single-thread brute force | 648/648 | 300/648 | 1.309× |
| Default-thread automatic algorithm | 648/648 | 592/648 | 1.637× |

All **162 small/single-query cells**, **108 clustered/offset cells** and **81
explicit Gaussian-distribution cells** beat all three sklearn configurations.
The respective minimum speedups across the three comparators are 1.309×,
1.365× and 1.481×. All 23 default-thread misses belong to the previously known
eight-feature batch workload with `k=32`. The result is not universally 2×.

![Full KNN benchmark comparison](assets/knn-digits-benchmarks-2026-10-30.png)

Each chart cell is the minimum median-time ratio across seeds, not an average.
The tables preserve every seed, operation and miss; rounding does not determine
the color or acceptance decision.

### Digits GPU controls

The [balanced control](knn-digits-control-2026-10-30.md) and
[original-order repeat](knn-digits-legacy-control-2026-10-30.md) both pass every
Digits GPU comparison. Their minima across both replicas and both statistics
are 1.079× and 1.047× respectively.

The previously failing seed-252 distance-weighted regression cell now measures
0.444750 ms for current automatic, 0.429250 ms for its identical A/A replica and
0.480041 ms for historical GPU in the original-order repeat:

| Comparison against historical GPU | Ratio of medians | Median paired ratio |
| --- | ---: | ---: |
| Primary automatic replica | 1.079× | 1.051× |
| Identical automatic A/A replica | 1.118× | 1.145× |

Those replace the preceding revision's 0.964× / 0.979× primary ratios in the same
protocol. Absolute milliseconds from separate runs are not interchangeable;
each ratio compares implementations within its own run. The additional 1.10×
engineering-margin target passes **46/48** comparisons. Its two misses are the
primary replica's two ratios above, which remain faster than GPU but do not meet
that larger margin. The overall 1.047× minimum belongs to seed-252
distance-weighted classification probabilities in the A/A replica versus
same-source GPU.

### Startup and first queries

The [450-launch fresh-process panel](knn-digits-cold-2026-10-30.md) retains the
startup gains, with all 18 fixture medians faster than every sklearn setting:

| Comparator | Process-to-first-result speedup | Fixtures faster |
| --- | ---: | ---: |
| Default-thread brute | 1.191–1.224× | 18/18 |
| Single-thread brute | 1.181–1.222× | 18/18 |
| sklearn auto | 1.159–1.221× | 18/18 |
| Preceding automatic | 0.985–1.020× | 10/18 |

Current process-to-result medians span 679–711 ms, versus 679–711 ms for the
preceding automatic build. The largest matched startup loss is 1.54%, or
10.45 ms. Both versions leave `sklearn.neighbors` unimported before fit in all
90 measured launches. This revision does not claim another import-speed gain.

The [2,700-fresh-model panel](knn-digits-first-query-2026-10-30.md) also retains
all 18 first-query and fit-plus-first-query wins against all three sklearn
settings. First-query speedups are 6.121–41.293× versus default brute,
1.243–3.282× versus single-thread brute, and 2.693–24.899× versus sklearn auto.
All three Digits splits improve over preceding automatic: 1.055–1.107× for
first query and 1.045–1.090× for combined fit and first query.

Across all 18 fresh-model fixtures, however, first-query medians improve versus
preceding automatic on only 9/18, and combined fit/query on 11/18. The largest
first-query percentage loss is 8.3%; the largest absolute loss is 9.6 µs.
Combined fit/query loses at most 10.2%, with a largest absolute loss of about
10.3 µs. In fresh processes, the
corresponding wins are 11/18 and 13/18, with maximum losses of 4.3% / 5.7 µs and
1.9% / 2.9 µs. These subphase observations are retained alongside total startup.

### Remaining isolated differences

The balanced control has 18 greater-than-5% flags versus preceding automatic.
Nine are covered by the original-order repeat; three recur in both replicas
under both statistics, all on Breast Cancer:

| Fixture / operation | Legacy primary M/P | Legacy A/A M/P |
| --- | ---: | ---: |
| Seed 151, distance-weighted classification | 0.851 / 0.870× | 0.855 / 0.847× |
| Seed 251, uniform regression | 0.826 / 0.803× | 0.837 / 0.800× |
| Seed 252, uniform classification probabilities | 0.803 / 0.816× | 0.791 / 0.832× |

Primary median losses on these three cells are approximately 30.8, 29.7 and
34.8 µs. The repeat also has eight new prior-auto flags; every observation is
listed in its [complete tables](knn-digits-legacy-control-2026-10-30.md).

The [Gaussian/cluster preservation repeat](knn-digits-preservation-repeat-2026-10-30.md)
has no greater-than-5% flag versus preceding automatic. Gaussian 4096×128,
seed 552, has a same-source GPU flag in the balanced run that does not recur,
and a historical-GPU flag appearing in the repeat: primary M/P 0.940/0.947×,
A/A 0.945/0.943×. The primary repeat median is 0.39258 ms versus 0.36921 ms.
This isolated loss remains a limitation even though broad Gaussian gains hold.

The native-binary preservation investigation below checks whether the repeated
Breast Cancer losses are caused by this implementation change. It does not
replace these separate-worker observations or establish zero runtime overhead.

## Implementation and precision

The only production change from the
[preceding startup revision](knn-startup-fix-2026-10-30.md) is in the
[native CPU integer selector](../src/mlxlearn/_kernels/_neighbors_cpu_native.cpp).
For `k=5`, the selector now validates each dot-product value in the same pass
that computes exact squared distances and selects neighbors. The previous
implementation read the complete dot matrix once for validation and again for
selection. All other neighbor counts retain their preceding implementation.

The original-coordinate integer certificate, arithmetic expression and
validation predicate are unchanged. Every SIMD and scalar lane is checked,
including excluded self IDs and blocks that cannot enter the top five.
Malformed vector lanes are masked before arithmetic, while their failure stays
in the global validity mask. Each row's five results remain in private temporary
storage until every row passes. Invalid input therefore cannot expose partial
outputs. An invalid integral-domain value retains error priority over a negative
squared distance found earlier in the matrix.

References still arrive in ascending original training-index order. A later
equal-distance reference cannot displace an earlier one. The
[integer certificate](knn-integer-certificate.md#cpu-bounded-integer-selection)
documents the exactness and ordering argument. GPU kernels, direct CPU scans,
dispatch, imports, estimator adapters, distance weighting and float64 regression
target arithmetic are unchanged. No query-result cache, workload-name rule,
thread-count change or background prewarm was added.

## Causal development evidence

The [baseline stage profile](../benchmarks/results/knn-digits-gap-baseline-stages-2026-10-30.json)
identified selection as the main Digits cost: 133–149 µs, compared with about
50 µs for matrix multiplication and 12.7–14.0 µs for weighted prediction's final
aggregation. These isolated component timings are not directly comparable to
the absolute times in interleaved GPU controls.

The [interleaved native-binary ablation](../benchmarks/results/knn-digits-gap-fused-ablation-2026-10-30.json)
substituted the exact archived preceding native binary and the candidate into
the same Python wrapper. All three Digits splits improved: selection took
about 31% less time, warmed distance-weighted prediction 16–19% less, and fresh
fit-plus-prediction about 12% less. All retained observations and the precise
audit scope are described in the
[probe provenance](../benchmarks/results/knn-digits-gap-probe-provenance-2026-10-30.json)
and [candidate validation record](../benchmarks/results/knn-digits-gap-candidate-validation-2026-10-30.json).
That component probe checked selector squared distances and indices after each
call, and engine indices only. Public prediction calls were timed but were not
individually audited after each call. Separate correctness tests and the final
public benchmark panels supply those prediction audits.

The [development GPU control](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-cc96cf6c82004afd39f0)
retained all 27 Digits cells across both GPU controls, two automatic replicas and
both ratio statistics: all 216 comparisons exceeded 1×, with a minimum of
1.110×. Some comparisons with preceding automatic remained below 1×; those
observations are retained. The final full controls are separate runs and
do not replace this development evidence.

## Native preservation investigation

The recurring Breast Cancer observations prompted a separate
[exact-native A/B/A investigation](../benchmarks/results/knn-digits-breast-native-preservation-verification-2026-10-30.json).
Static disassembly found that the direct and concentrated query kernels moved
216 bytes within the new binary. Their symbolized instructions, registers,
stack layout and arithmetic were unchanged apart from relocated call targets.
Changed instruction placement was therefore a plausible hypothesis, not an
established cause.

The supplemental probe used the same Python wrappers and coordinates, the exact
old and new binaries, two current replicas, all six mode orderings and all three
Breast Cancer splits. All **10,800 results** passed independent neighbor or
prediction checks, including all 2,160 warmup and 8,640 retained calls. Every
timing remains in the [raw results](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-7fd99d2f5bf7650d7d2e).

Old/new median-time ratios for the concentrated native primitive were
0.9884–1.0021×; warmed public query, classification, regression and probability
ratios were 0.9876–1.0155× across both current replicas and all seeds. First-query
ratios were 0.9786–0.9940×, a maximum observed slowdown of about 2.2%; combined
fit/first-query ratios were 0.9853–1.0077×. The unpermuted direct primitive
improved 1.171–1.201×. No concentrated or warmed public greater-than-5% loss
reproduced in this shared-process test.

This evidence does not attribute the full-control regressions to the native
change, and it does not explain the separate-worker measurements. Those losses
remain valid and causally unresolved. No alignment change, source edit, rebuild,
sample deletion or replacement benchmark followed this investigation. The
supplemental verification rechecks all frozen source, harness and native bytes.

## Correctness validation

The relevant suite passes **1,474 tests on each of sklearn 1.7.2, 1.8.0 and
1.9.0**. Five unrelated LogisticRegression/SVC cases are deselected in each run.
The [1.7.2](../benchmarks/results/knn-digits-sk172-tests-2026-10-30.log),
[1.8.0](../benchmarks/results/knn-digits-sk180-tests-2026-10-30.log) and
[1.9.0](../benchmarks/results/knn-digits-sk190-tests-2026-10-30.log) logs retain all
warnings. The [65 harness checks](../benchmarks/results/knn-digits-harness-tests-2026-10-30.log)
also pass: **1,539 distinct checks** in the current environment, with the older
sklearn runs counted as compatibility reruns.

The 66 added adversarial cases cover late malformed rows, skipped blocks,
self-excluded IDs, 16-wide blocks, 4-wide remainders, scalar tails, NaN and
subnormal inputs, error precedence, and unchanged output buffers on rejection.
Existing exact-order and portable-fallback tests remain in the suite. The
native extension was built and tested on macOS arm64; a non-ARM native build
was not tested.

## Benchmark protocol

The [saved declaration](../benchmarks/results/knn-digits-final-predeclaration-2026-10-30.json)
predates the candidate GPU controls and all final panels. Component profiling
and native ablation were development evidence, not preregistered final results.
The final comparison uses the immediately preceding startup revision and its
exact archived native binary, alongside same-source forced GPU and the older
GPU build. The three sklearn settings are default-thread brute force,
single-thread brute force and default-thread automatic algorithm selection.

The full run preserves the previous six panels and inventories: 450 fresh
processes; 72 broad fixtures with nine headline operations plus separate fit-only
timings; 48 balanced-control fixtures; the original nine-fixture legacy control;
nine Gaussian/cluster preservation fixtures; and 2,700 fresh models with first
and second queries measured separately. All timed panels run serially, with no
competing tests, builds, profiles or artifact audits.

Ratios mean comparator time divided by current time; values below 1× are
slowdowns. Both ratio of medians and median paired ratio are retained. All 27
Digits cells must beat both GPU controls in both automatic replicas under both
statistics in both full-control orderings: 432 comparisons. A separate 1.10×
development-margin target covers the 48 distance-weighted regression
comparisons. Every smaller or inconsistent miss against preceding automatic
also remains visible. A repeated greater-than-5% runtime flag requires a ratio
below `1/1.05` in both statistics and both replicas; this is a descriptive
recurrence screen, not a significance test.

Fresh-process latency includes interpreter startup, input loading, imports,
native loading, fit, first query and result delivery. It does not imply an empty
OS filesystem or GPU disk cache. The fresh-model panel measures new models in
initialized processes without prewarming those models or their first queries.

The [serial runner record](../benchmarks/results/knn-digits-final-2026-10-30-runner-status.json)
retains actual commands, ordering, timestamps and logs. All unfavorable timing
observations remain in the raw results; accuracy failures never remove timings
from a denominator.

## Reproducibility and artifacts

Measurements used an Apple M4 Max with 128 GiB memory, CPython 3.13.3, MLX
0.32.2, NumPy 2.5.2, SciPy 1.18.1 and sklearn 1.9.0. Thread settings and workload
definitions are unchanged from the preceding panels. The native build retains
`-O3 -std=c++17 -fno-fast-math -ffp-contract=off`.

| Artifact | SHA-256 |
| --- | --- |
| [63-file source snapshot](../benchmarks/reference/knn-digits-final-2026-10-30.zip) | `5a4b3c5c130c043640b8755b3e26aaf53911729cd5e56acc37699e681da268db` |
| [Exact native runtime and build provenance](../benchmarks/reference/knn-digits-final-2026-10-30-runtime.zip) | `b92e7cb2dc879eaffbcb375c5f6ea9e9dc8427f0ca68f325afb9ca9ad37a5bae` |
| [148-file harness, tests and component probes](../benchmarks/reference/knn-digits-final-2026-10-30-harnesses.zip) | `8718e3cdfdadaca62ead81ac22faae1d673a6db4bef1c72237c04c185a8b96a3` |

The [execution record](../benchmarks/results/knn-digits-final-2026-10-30-execution.json)
contains exact commands, timestamps, versions, baseline identities and artifact
hashes. The [independent verifier](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-94ba3b57658c12377bb5),
[additional acceptance audit](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-88d3bb17ad5ad333ff56)
and [post-run integrity record](../benchmarks/results/knn-digits-final-2026-10-30-post-run-integrity.json)
all pass their numerical/evidence checks. Performance limits and the unmet
1.10× margin checks remain separate from numerical acceptance.

The [package validation](../benchmarks/results/knn-digits-final-2026-10-30-package-validation.json)
confirms that wheel and source distribution contain the exact 63 frozen source
files and the wheel contains the measured native binary. Only the native C++
source differs from the preceding startup snapshot; all 21 GPU device files,
Python CPU engine and estimator wrappers remain identical. The later
[preservation-probe verification](../benchmarks/results/knn-digits-breast-native-preservation-verification-2026-10-30.json)
is supplemental to the immutable six-panel execution record.
