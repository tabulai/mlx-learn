# Small-workload GPU overhead reduction — October 30, 2026

Small datasets and single-query searches now combine screening, selection,
certification and exact fallback in one GPU launch. Canonical float32 queries
also avoid repeated generic array conversion while retaining validation.

The [full sklearn tables](knn-small-benchmarks-2026-10-30.md) retain every dataset
and operation. The separate [controlled implementation comparison](knn-small-control-2026-10-30.md)
compares the current source with the preceding Gaussian fresh-fit build and
an identical-current-code A/A worker.

**The change reduces the slowdowns; it does not eliminate every disadvantage
against CPU sklearn.** All 162 targeted dataset-operation comparisons improve
against the preceding build: median paired speedups span **1.081–1.483×**, and
ratios of per-worker medians span **1.001–1.563×**. The two statistics describe
different summaries of the same retained samples. A/A paired ratios span
0.876–1.118×, so the smallest individual gains should not be treated as
statistically established speedups.
An [independent control audit](../benchmarks/results/knn-small-independent-control-audit-2026-10-30.json)
recomputed every statistic and verified all 48 fixture inventories. All
652,162 current, 658,284 A/A and 556,768 previous-build cold, warmup, probe and
retained audits pass. These are repeated-output checks over 48 fixtures, not
independent datasets.

## Full results and remaining slowdowns

The full run covers 72 dataset instances and 648 search/inference medians per
sklearn configuration. The targeted subset contains 18 instances and 162
medians: single-query Gaussian 8D/32D and raw Digits, Breast Cancer, Iris and Wine.

| sklearn configuration | Full-panel medians ≥2× | Targeted small/single medians faster than sklearn |
| --- | ---: | ---: |
| Brute, default threads | **526/648 (81.2%)** | **154/162** |
| Brute, one thread | **252/648 (38.9%)** | **5/162** |
| Auto, default threads | **551/648 (85.0%)** | **91/162** |

The remaining targeted misses are **8**, **157** and **71** medians respectively.
The minimum targeted speedup ratios are 0.742×, 0.256× and 0.399×. In particular,
single-thread sklearn still wins most small workloads, and auto retains strong
advantages for tiny low-dimensional data. These are substantial remaining
slowdowns, not a universal-parity result. No cases are removed from the counts.

The broad ≥2× counts in the preceding run were 492/648, 246/648 and 528/648.
Those separate-run totals are context, not an isolated estimate of code effects;
the contemporaneous controlled comparison is the evidence for source changes.

All **14,472 current accuracy audits pass**. All **108 clustered/offset medians**
beat all three sklearn configurations, with minimum speedups of **2.189×**
against default-thread brute, **1.323×** against single-thread brute and
**2.143×** against auto. Every one of those medians reaches 2× against the two
default-thread configurations. The 72 input/oracle array inventories are
identical to the preceding full run.

## Preservation compared with the preceding build

The control includes six Gaussian seeds per shape and three seeds per other
distribution. Across its 60 preservation cells, previous/current paired ratios
span 0.959–1.189×; ratios of medians span 0.961–1.162×. Eight paired medians and
nine ratios of medians are below parity. Corresponding A/A ranges are
0.925–1.081× and 0.886–1.072×. The results retain those slower observations and
do not establish an improvement at every seed. Search routes and numerical
certificates for these batch shapes are unchanged.

| Gaussian shape | Query paired speedup | Fresh paired speedup | Fresh ratio of medians | Fresh A/A paired ratio |
| --- | ---: | ---: | ---: | ---: |
| 50k × 32, six seeds | 1.034–1.089× | 1.004–1.068× | 0.992–1.063× | 0.961–1.041× |
| 4096 × 128, six seeds | 1.010–1.189× | 0.962–1.083× | 0.961–1.125× | 0.925–1.067× |

The fresh-fit gains of the preceding Gaussian build remain broadly intact;
small fluctuations overlap the observed A/A variation. Expanded 32-feature
Gaussian seed 552 remains a shared high-latency case in all workers and is
retained in the control. The full tables also retain higher-latency Digits
operation observations without filtering them or assigning an unverified cause.

## Implementation and scope

The new [`_neighbors_small.py`](../src/mlxlearn_device/_neighbors_small.py)
assigns one GPU threadgroup to each query. Its SIMDgroups scan disjoint
reference partitions using the existing compensated direct distances and
omission bounds. After one shared-memory barrier, the first SIMDgroup merges
the lists and checks the existing precision certificates. Certain results
return immediately; uncertain ordering uses the existing exact comparator.
Uncertain membership rescans the original coordinates in the same launch.

The rescan avoids expensive selection work for provably irrelevant references.
The fast list contains K distinct valid references. The conservative upper
bound of its largest computed distance therefore bounds the true kth distance,
even if fast membership was incorrect. A reference can be skipped only when
its conservative lower bound strictly exceeds that upper bound. Strict
inequality retains boundary ties, and unsafe/extreme ranges disable pruning.
The original-coordinate exact comparator and numerical bounds are unchanged.

[`DeviceNeighborIndex.query`](../src/mlxlearn_device/_neighbors.py) selects this
path only for k ≤ 8 and either of these shapes:

| Workload | Training rows | Features | Query rows |
| --- | ---: | ---: | ---: |
| Small matrix | ≤2,048 | ≤64 | ≤256 |
| Very small query batch | ≤16,384 | ≤32 | ≤4 |

Small matrices use four SIMDgroups per query; the second route uses sixteen
when training exceeds 2,048 rows. Explicit training tiles retain the general
path, and explicit query tiles preserve global self-exclusion offsets. CPU
search and the established Gaussian, clustered, offset and unequal-scale batch
routes remain unchanged. A native query's optional square root shares the
selection kernel; the sklearn boundary retains float64 output conversion.

The compiled graph takes training and query arrays as dynamic inputs. Only
kernel code and graph structure are cached. Independently owned fits and
deferred queries retain their own inputs; no query results or training values
are cached globally.

The [`_validate_neighbor_query`](../src/mlxlearn/neighbors/_base.py) shortcut
accepts only exact NumPy arrays with native float32 dtype, no dtype metadata,
C layout, two nonempty dimensions and finite coordinates. sklearn still checks
feature names and the fitted feature count. All other inputs retain the prior
conversion and error path. This saves about 12–15 µs on small queries in the
[CPU-only validation diagnostic](../benchmarks/results/knn-small-query-validation-probe-2026-10-30.json).

## Correctness and compatibility

**775 relevant tests pass:** the [754-test full suite](../benchmarks/results/knn-small-full-tests-2026-10-30.log)
plus [21 subsequently added ownership/routing checks](../benchmarks/results/knn-small-additional-tests-2026-10-30.log).
Five unrelated logistic/SVC cases are deselected. New GPU checks use independent
exhaustive dyadic and Python-integer oracles for small matrices, single queries,
rounded-but-unequal distances, true ties, overflow, subnormals, self exclusion,
explicit tiling, caller mutation and reverse evaluation of independent fits.
CPU checks cover dispatch boundaries and input-validation semantics.
Five additional [post-benchmark boundary checks](../benchmarks/results/knn-small-boundary-checks-2026-10-30.json)
pass at both maximum dispatch shapes, k=8, one training row, self exclusion and
strided native inputs. They use exact bounded-dyadic exhaustive references and
do not change the timed sources or fixtures.

The validation suite includes malformed and nonfinite inputs, changing query
values, strided and byte-swapped arrays, subclasses, read-only arrays, dtype
metadata, feature-count errors and DataFrame feature-name behavior.
[Compatibility evidence](../benchmarks/results/knn-small-query-validation-compatibility-2026-10-30.json)
also records successful 25-test host-only runs on sklearn 1.7.2 and 1.8.0 before
the final metadata-specific case was added. Ruff and `git diff --check` pass.

## Benchmark protocol and limitations

The controlled run uses 48 dataset instances and 222 dataset-operation cells:
18 small/single-query instances across all nine search/inference operations,
plus 30 Gaussian, clustered, offset and unequal-scale instances for query and
fresh fit-plus-query. Each current, A/A and frozen-previous worker receives
one second of continuous warmup, followed by 31 interleaved paired rounds.
The preceding source is `knn-gaussian-fresh-final-2026-10-30.zip`. Both median
paired ratios and ratios of timing medians are retained.

The broad run preserves all 72 fixtures, 24 families and nine search/inference
operations from the preceding report: 15 retained calls after five warmups,
against default-thread brute, single-thread brute and default-thread auto
sklearn. Fit-only and first-query records are separate. All outputs are audited
outside timing against exhaustive original-coordinate references. Baseline
numerical differences and valid alternate tie choices remain visible and do
not remove timings from speed counts.

Both runs use the public NumPy API and force the MLX GPU backend. Validation,
conversions, model preparation and synchronization are included. Timed calls
are serial; source, harnesses, fixtures and tests stay frozen during final runs.
No host search, sklearn fallback, estimator hooks or result caches are introduced.
First-query records come from persistent workers and do not measure process
startup. The continuously warmed control and five-warmup sklearn panel use
different warmup protocols, so their absolute times are not interchangeable.

These measurements describe float32 Euclidean workloads on an Apple M4 Max,
not other hardware, metrics, sparse data, float64 search or automatic backend
selection. A [diagnostic](../benchmarks/results/knn-small-latency-floor-inputs-2026-10-30.json)
measured a trivial synchronized GPU launch at about 106 µs; a warmed Iris auto
query in a [separate diagnostic](../benchmarks/results/knn-small-stages-probe-2026-10-30.json)
took about 101 µs. This illustrates fixed-cost
pressure at tiny sizes, not a universal hardware lower bound or an acceptance
test. The complete benchmark retains workloads that still favor sklearn.

## Reproduction and artifacts

- [Controlled raw results](../benchmarks/results/knn-small-final-2026-10-30-control.json)
- [Broad raw results](../benchmarks/results/knn-small-final-2026-10-30-broad.json)
- [Broad summary](../benchmarks/results/knn-small-final-2026-10-30-summary.json)
- [Execution record](../benchmarks/results/knn-small-final-2026-10-30-execution.json)
- [Final integrity and acceptance checks](../benchmarks/results/knn-small-final-2026-10-30-validation.json)
- [Independent control audit](../benchmarks/results/knn-small-independent-control-audit-2026-10-30.json)
- [Production archive](../benchmarks/reference/knn-small-final-2026-10-30.zip) and [manifest](../benchmarks/reference/knn-small-final-2026-10-30.manifest.json)
- [Benchmark/test archive](../benchmarks/reference/knn-small-final-2026-10-30-harnesses.zip) and [manifest](../benchmarks/reference/knn-small-final-2026-10-30-harnesses.manifest.json)

```sh
python3 -m benchmarks.knn_small_control \
  --source-manifest benchmarks/reference/knn-small-final-2026-10-30.manifest.json \
  --output benchmarks/results/knn-small-final-2026-10-30-control.json
python3 -m benchmarks.knn_broad \
  --source-manifest benchmarks/reference/knn-small-final-2026-10-30.manifest.json \
  --repeats 15 --warmup 5 \
  --output benchmarks/results/knn-small-final-2026-10-30-broad.json
```
