# Automatic CPU/GPU KNN dispatch — October 30, 2026

Automatic dispatch eliminates the small-dataset and single-query slowdowns in
the full sklearn panel's retained timing medians: **all 162 targeted
dataset-operation medians beat each of the three sklearn configurations**. The previous GPU-only
run won 154, 5 and 91 of those comparisons respectively. These separate-run
counts describe coverage; the paired implementation control isolates the code
change.

The narrowest targeted win is **1.010×**, against single-thread brute force.
That small margin is not a robust guarantee for every repetition or machine.
The result is also **not a universal 2× speedup**. The complete panel retains
21 comparisons where default-thread brute force wins.

## Full sklearn comparison

The [complete benchmark tables](knn-auto-benchmarks-2026-10-30.md) cover the
unchanged 72 dataset instances, 24 families and nine search/inference operations:
648 medians per comparator. The target subset contains 18 instances: single-query
Gaussian 8D/32D and raw Digits, Breast Cancer, Iris and Wine, with three seeds each.
Ratios are sklearn time divided by automatic-dispatch time; larger is better.

| sklearn configuration | Target medians faster | Target minimum | Full medians faster | Full medians ≥2× |
| --- | ---: | ---: | ---: | ---: |
| Brute, default threads | **162/162** | **2.361×** | **627/648** | **592/648** |
| Brute, one thread | **162/162** | **1.010×** | **648/648** | **267/648** |
| Auto, default threads | **162/162** | **1.532×** | **648/648** | **585/648** |

The corresponding target median speedups are 5.612×, 1.346× and 2.745×.
All **14,472 current-version accuracy audits pass**, including cold, warmup
and retained calls. The strict canonical oracle rejects some sklearn outputs;
baseline numerical differences and valid alternate tie choices remain visible
in the tables and do not remove their timings from speed counts.

All **108 clustered/offset medians** beat all three configurations. Their
minimum speedups are **2.161×**, **1.316×** and **2.181×**, respectively. All
21 full-panel misses against default-thread brute force belong to the
8-feature, k=32 batch family, outside the small/single-query target subset.

Cold behavior is separate. Target cold queries win 18/18, 16/18 and 17/18
comparisons; cold fit-plus-query sums win 17/18, 16/18 and 17/18. The first
8-feature single-query fixture takes 15.224 ms including fit, compared with
7.560 ms default brute, 8.425 ms single-thread brute and 2.240 ms sklearn auto.
Digits and Breast Cancer also have individual cold-query misses against
single-thread brute. The [additional audit](../benchmarks/results/knn-auto-final-2026-10-30-additional-audit.json)
retains each cold observation. Warmed-process medians do not establish a
universal first-use latency win.

## Paired preservation and remaining tradeoffs

The [four-worker control](knn-auto-control-2026-10-30.md) passes all **849,312**
cold, warmup, probe and retained output audits: 254,741 automatic, 252,045
automatic A/A, 167,951 current GPU and 174,575 previous GPU checks. These are
repeated-output checks over 48 fixtures, not independent datasets. The
[independent verifier](../benchmarks/results/knn-auto-final-2026-10-30-independent-audit.json)
recomputes timing summaries, verifies fixture identities and confirms that
all current workers used the archived native binary and source.

Across the 60 preservation query/fresh cells, previous-GPU/auto median paired
ratios span **0.893–1.055×**, with a median of **1.006×**; the independent auto
replica spans **0.931–1.067×**, with a median of **0.993×**. Current forced
GPU/auto spans **0.928–1.073×**, with a median of **1.001×**. The auto A/A
ratios span **0.968–1.095×**. These descriptive ranges retain slower results;
they are not confidence intervals and do not prove zero overhead at every seed.
Twenty-four previous-GPU/auto paired medians are below 1×; 17 preservation
cells are slower against the previous GPU in both automatic replicas.
No preservation cell is more than 5% slower in both automatic replicas under
both the ratio of medians and median paired ratio. This is a descriptive
cross-check, not a statistical noninferiority test. The worst Gaussian paired
ratio, 0.893× for 128D fresh seed 252, corresponds to a ratio of medians of
0.966×; the replica has 0.975× paired and 1.003× by medians. It does not
reproduce a stable 12% regression, but the slower observations remain.

| Gaussian preservation workload | Previous GPU/auto paired | Previous GPU/auto replica paired | Auto A/A paired |
| --- | ---: | ---: | ---: |
| 50k × 32 query, six seeds | 0.947–1.022× | 0.949–1.014× | 0.984–1.025× |
| 50k × 32 fresh fit+query, six seeds | 0.942–1.020× | 0.978–1.041× | 0.970–1.012× |
| 4096 × 128 query, six seeds | 0.994–1.018× | 0.949–1.040× | 0.979–1.005× |
| 4096 × 128 fresh fit+query, six seeds | 0.893–1.055× | 0.975–1.067× | 0.974–1.031× |

The Gaussian, clustered and offset gains against sklearn remain, and their
GPU kernels are unchanged. Individual paired observations nevertheless show
slowdowns; unchanged kernels alone do not guarantee identical public-API timing.
The expanded Gaussian 32D seed 552 remains a shared high-latency case and is
retained without filtering.

Automatic dispatch is also not always faster than forcing GPU after prolonged
warmup. Across the 162 target cells, median paired previous-GPU/auto speedup
is **2.079×**, but 19 cells favor the previous GPU, chiefly Digits. Its worst
paired ratio is **0.427×**; both auto replicas show that disadvantage. The
CPU route still beats all sklearn configurations under the full panel's
five-warmup protocol. This distinction is material: the dispatch policy fixes
the measured sklearn disadvantage without establishing the best possible
backend for every warmup regime.

## Implementation

The default `device="auto"` policy is captured at fit and chooses CPU or GPU
for each query batch. Explicit device choices remain fixed. Small automatic
fits initially own host training data; GPU preparation is deferred until a
query needs it. Large fits keep their established GPU preparation. Both paths
own their authoritative training values independently of caller arrays and
the mutable sklearn compatibility mirror.

```python
from mlxlearn import config_context
from mlxlearn.neighbors import NearestNeighbors

with config_context(device="auto"):
    model = NearestNeighbors(n_neighbors=5).fit(X)
distances, indices = model.kneighbors(Q)
print(model.execution_policy_)      # "auto"
print(model.last_query_backend_)    # actual "cpu" or "mlx" query backend
```

`execution_backend_` continues to describe fit/storage placement; it does not
change with each query. Classifier prediction, probability and regressor
prediction share the selected search path. Changing global configuration after
fit does not change the fitted policy. Radius queries retain their existing
behavior. Unexpected backend errors raise; they are not rerun through sklearn.

The current CPU regions in [`NeighborIndex`](../src/mlxlearn/_kernels/distance.py)
are deliberately conservative. For at most 32 features and 32 neighbors, CPU
search covers at most four queries with `N × Q ≤ 16,384`, matrices with at most
256 rows and 64 queries, and matrices with at most 512 rows and 32 queries.
A further 256–512-row, 9–32-feature region allows up to 128 queries when the
eight largest squared feature ranges account for at least 99% of their total.
Suitable sparse, bounded integer matrices with 33–64 features, at most 2,048
rows, 256 queries and eight neighbors use exact integer Gram arithmetic.
Training and every query must satisfy its arithmetic certificate; a sparse
sample can reject this optimization but cannot certify it. Other GPU-capable
workloads keep the GPU path. These thresholds are implementation details,
not a promise of the fastest backend for every possible distribution.

The new optional [C++ CPU extension](../src/mlxlearn/_kernels/_neighbors_cpu_native.cpp)
uses Neon scans and conservative distance bounds. Safe float32 inputs get a
fast scan; other inputs use float64 arithmetic with exact conversion of
subnormal coordinates. A partial-distance bound can skip a reference only
when its lower bound is strictly above the kth candidate's upper bound.
Overlapping distance intervals are resolved with exact 640-bit keys derived
from the original float32 coordinates. Only true equal distances use training
index order. Selected distances are recomputed in float64; cached exact keys
are rounded consistently so true ties receive identical distance weights.

For concentrated feature scales, an owned column permutation puts the most
selective coordinates first. It changes neither values nor the metric.
The [portable CPU implementation](../src/mlxlearn/_kernels/neighbors_cpu.py)
retains certified omission bounds, exhaustive fallback and exact ordering when
the extension is unavailable. No query results are cached. CPU/GPU transitions,
refit, serialization and concurrent preparation preserve training ownership.

**Every `mlxlearn_device/*` source hash is unchanged from the preceding
`knn-small-final-2026-10-30.zip`.** The Gaussian, clustered, offset and precision
GPU kernels therefore retain exactly their preceding implementation. The
controlled measurements below assess public-boundary overhead as well.

## Precision, compatibility and packaging

The [final relevant suite](../benchmarks/results/knn-auto-full-tests-final-2026-10-30.log)
passes **1,025 tests**, with five unrelated LogisticRegression/SVC cases
deselected. It covers CPU and GPU precision, exact tie ordering, unequal scales,
overflow and subnormals, canonical kth-boundary membership, distance-weighted
votes, self exclusion, tiling, input validation, patching, sklearn estimator
compliance, mutable inputs, pickle/refit, dispatch transitions and benchmark
integrity. Portable CPU precision tests explicitly disable the native extension.
The [float32 policy](fp32_policy.md) explains the arithmetic contract: inputs
are still converted to float32; host float64 arithmetic does not recover
precision lost during that conversion.

An additional [native boundary diagnostic](../benchmarks/results/knn-auto-native-endpoints-final-2026-10-30.json)
passes 48 checks over six wholly in-range fixtures at the inclusive float32
screening bounds, with 32 and 64 features. Exact dyadic ordering passes,
133 tied-neighbor pairs return identical distances, and the maximum distance
error against the rationally rounded reference is zero ULP. The diagnostic
does not change any frozen source or test file.

The native extension is compiled ahead of time, with fast-math and contraction
disabled. No C++ compilation occurs during fit or query. The
package adds an explicit SciPy dependency for portable CPU search. The
optional extension targets Apple arm64; if it is unavailable, the portable
implementation remains correct, but the native timings do not apply.
Wheel and source-distribution builds both pass. The native CPU module uses
the CPython ABI, so the tested wheel is `cp313-cp313-macosx_11_0_arm64`, rather
than advertising the Metal module's broader internal `abi3` compatibility.
The [package validation](../benchmarks/results/knn-auto-package-validation-final-2026-10-30.json)
checks all 63 source files in both distributions and the wheel's native binary
against the measured runtime. Production and targeted-test Ruff checks and
`git diff --check` pass. Nine nonfunctional style findings in the frozen
benchmark helpers remain recorded in the execution evidence; the helpers
were not reformatted after timing.

Independent shape diagnostics rejected broader CPU routing that slowed dense
Gaussian and integer batches. The final focused guard check passes all four
accuracy cases and keeps them on GPU. Its auto/GPU fresh latency ratios range
from 0.946 to 1.051; the remaining 5.1% observation is retained, not presented
as universal parity. These seven-repeat diagnostics have a different protocol
from the full final runs. An earlier experimental borrowed-buffer transition
stalled and was discarded; the final transition uses an independent GPU copy
and the subsequent 39-shape rerun completed.

## Protocol and reproduction

Measurements use public NumPy estimator APIs on an Apple M4 Max with 128 GiB,
Python 3.13.3, MLX 0.32.2, NumPy 2.5.2 and sklearn 1.9.0. CPU/GPU timed work is
serial. Validation, NumPy boundaries, lazy preparation and synchronization are
inside timed operations. Fresh fit-plus-query includes all per-fit work. Fit-only
and cold records are separate; cold means first fit/query for a fixture in a
persistent worker, excluding process startup. The C++ extension is built ahead
of time; MLX can compile GPU kernels on first use. There is no result caching,
input substitution or estimator hook in the benchmarks.

The broad run retains 15 calls after five warmups for each operation and
comparator. Its 72 input and oracle inventories match the preceding full run.
The paired control uses 48 fixtures and 222 case-operation cells: all 162 target
cells plus 60 Gaussian, clustered, offset and unequal-scale query/fresh cells.
It includes six Gaussian seeds per shape. Four isolated workers run current
auto, identical current auto (A/A), current forced GPU, and the frozen previous
GPU build, with 0.25 seconds of continuous warmup per worker/operation followed
by 31 interleaved rounds. All warmup, probe, cold and retained audits remain in
the raw evidence. This warmup differs from the preceding control's one second;
absolute timings across protocols are not interchangeable.

```sh
python3 setup.py build_ext --inplace
python3 -m benchmarks.knn_broad \
  --current-mode auto \
  --source-manifest benchmarks/reference/knn-auto-final-2026-10-30.manifest.json \
  --repeats 15 --warmup 5 --target 2 \
  --output benchmarks/results/knn-auto-rerun-broad.json
python3 -m benchmarks.knn_auto_control \
  --source-manifest benchmarks/reference/knn-auto-final-2026-10-30.manifest.json \
  --previous-archive benchmarks/reference/knn-small-final-2026-10-30.zip \
  --pairs 31 --warmup-seconds .25 \
  --output benchmarks/results/knn-auto-rerun-control.json
```

Frozen artifacts retain the exact [source](../benchmarks/reference/knn-auto-final-2026-10-30.zip),
[native binary and build provenance](../benchmarks/reference/knn-auto-final-2026-10-30-runtime.zip),
and [harnesses/tests](../benchmarks/reference/knn-auto-final-2026-10-30-harnesses.zip).
The binary archive requires the matching architecture and CPython ABI;
otherwise rebuild from source. This evidence concerns the tested float32
Euclidean workloads and native runtime, not every metric, device or dataset.

The [execution record](../benchmarks/results/knn-auto-final-2026-10-30-execution.json)
contains commands, validation counts, source differences, build results and
artifact hashes. The [broad raw results](../benchmarks/results/knn-auto-final-2026-10-30-broad.json),
[control raw results](../benchmarks/results/knn-auto-final-2026-10-30-control.json)
and [broad summary](../benchmarks/results/knn-auto-final-2026-10-30-summary.json)
retain every measured case.
