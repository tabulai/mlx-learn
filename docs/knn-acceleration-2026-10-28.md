# Further KNN acceleration — 2026-10-28

The [next pass](knn-v2-2026-10-28.md) adds packed reads, scalar selection and
conversion-buffer reuse. It also establishes that the short warmup used below,
including the report's 25-call sustained-use section, can remain inside an
idle-to-active performance ramp. The recorded timings remain unchanged; the
new report distinguishes them from fully warmed measurements.

**The requested consistent 2× speedup is not achieved.** Both the original interleaved audit and an additional sustained-use audit fail that target. The interleaved audit improves over the previous implementation; sustained-use queries regress by 3.6–13.5%. Both protocols and all measured operations are reported below.

This follow-up targets 50,000 training rows, 32 float32 features, 256 query rows
and k=5 against default-thread sklearn. The earlier device pipeline is the
baseline for this comparison, including its original fused KNN implementation.
The immutable [pre-acceleration archive](../benchmarks/reference/knn-pre-acceleration-2026-10-28.zip)
contains the exact 52 source files measured in the preceding 50,000-row report.
Its manifest records working-tree source hashes; the checkout commit alone does
not identify that uncommitted implementation.

## Changes

The direct scan now targets up to 4,096 independent SIMDgroups instead of 256,
with at most 32 reference partitions and at least 512 training rows per partition.
At the target shape this increases the partitions from one to sixteen. Each
partition retains its local top-k list; the GPU merges the short lists using the
same distance/index order. The less expensive value-only selector retains its
own launch geometry.

A second specialization processes two queries per lane, sharing training loads
and interleaving independent compensated sums. Each pair keeps the original
feature accumulation order. Odd query batches mask their final inactive query.
Batches below 32 queries retain the original scan. The paired route also requires
at least 8,192 training rows, at most 32 features and at most 16 neighbors. No
training transpose or additional fit preparation is needed, and no query results
are cached.

Overflow recovery moves to the end of each paired distance calculation. With
finite inputs, squared terms are nonnegative: once the original sum overflows it
stays infinite. An unguarded compensation can instead become NaN; mapping that
final result to positive infinity preserves the result while removing a finite
check from each feature iteration. Finite distances retain their bit patterns.

The sklearn bridge now constructs the resident index from `mx.array`'s owned
copy of its validated NumPy data. This removes a redundant arithmetic copy and
GPU synchronization. Host and device storage remain independent, including
before the first query. The public native `fit_neighbors` API retains its owned
copy for arbitrary MLX inputs. The constructor implementations in
[MLX 0.29.4](https://github.com/ml-explore/mlx/blob/v0.29.4/python/src/convert.cpp)
and [0.32.2](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/convert.cpp)
were checked; the potentially borrowing `mx.asarray` is deliberately not used.

## Experiments and rejected alternatives

These are synchronized kernel probes, not the public API performance claim.
The full estimator benchmark below includes validation, input/output conversion,
aggregation and fresh-fit preparation.

| Experiment at 50,000 × 32 / 256 queries / k=5 | Observed kernel median | Decision |
|---|---:|---|
| Original scan, one reference partition | 1.71 ms | Underutilizes the GPU |
| Original scan, sixteen partitions | 0.88–0.91 ms | Keep |
| Feature-major training transpose | 1.29 ms | Reject; also requires preparation and storage |
| Blocked feature-major layout with explicit unrolling | 1.04 ms | Reject |
| Two queries per lane, sixteen partitions | 0.73 ms | Keep |
| Four queries per lane | 0.77 ms | Prefer two |
| Fast candidate screening, exact refinement and conditional exact scan | 0.97 ms | Reject; additional stages erase arithmetic savings |
| Two references per lane | 1.23 ms | Reject |
| Scan and merge in one threadgroup | 0.84 ms | Reject |
| Independent Kahan buckets | 0.79 ms | Reject; rounding changes without a speed gain |
| Compensated pairwise summation tree | 0.81 ms | Reject; better error but slower |

The screening experiment was never enabled in production. Ties, zero distances,
overflow and uncertain cutoffs took its exact fallback; the additional numerical
proof and implementation complexity were not justified by its measured latency.
The retained implementation scans all references with the existing compensated
arithmetic.

## Measurement protocol

`python3 -m benchmarks.knn_acceleration` runs three independent standard-normal
datasets (effective seeds 51, 52 and 53), each with two copied training rows in the
query batch. It measures ten operations after three warmups, retaining nine
samples per operation. Current, previous, native resident, single-thread sklearn
and default-thread sklearn use identical serialized inputs in separate processes.
Variant order alternates; GPU results are synchronized before timers stop.
Default sklearn uses its unchanged sixteen OpenMP threads on this M4 Max.

Each timed and warmup result is audited outside the timer: exact neighbor indices,
distance `atol=rtol=1e-6`, and classifier/regressor output checks. Source hashes
must remain unchanged throughout all seeds. The actual estimator executing each
operation must report the MLX backend. Fit-only timing is separate from fresh
fit-plus-query, which includes all training preparation.

The target gate requires every search, prediction and fresh-fit/query median to
beat default-thread sklearn by at least 2× on every seed. Fit alone is reported
separately. Per-round target counts and first/second-half medians expose timing
drift; they are descriptive observations, not confidence bounds or latency
guarantees.

The [occupancy-only attempt](../benchmarks/results/knn-acceleration-occupancy-2026-10-28.json)
failed this gate despite favorable isolated probes. A separate
[latency diagnostic](../benchmarks/results/knn-latency-diagnostic-2026-10-28.json)
found that equal 20–200 ms pauses did not remove the discrepancy, and it occurred
without sklearn workers too. OpenMP spin-wait is therefore not an established
explanation. The final protocol keeps sklearn's default thread settings and does
not select only favorable samples.

The existing Gram path was also measured explicitly at this size. Public query
was 1.842 ms versus 0.748 ms for direct scanning in that isolated probe; fresh
fit-plus-query was 3.200 ms versus 1.658 ms. Its unchanged dispatch is supported
by the [full probe](../benchmarks/results/knn-general-probe-2026-10-28.json).

## Interleaved results: target not consistently met

The current implementation passes all accuracy checks, but the original target gate remains false. Across the three datasets, neighbor queries improve **2.36–2.72× over the immediately previous version** and **1.57–2.92× over default-thread sklearn**. Fresh fit-plus-query improves **2.20–2.30× over the previous version** and **1.39–1.48× over default-thread sklearn**. Only indices-only search, uniform probabilities and distance-weighted classification exceed 2× default sklearn on every dataset median. Fit alone remains slower than sklearn.

These are medians of nine calls after three warmups, in milliseconds. The native resident column excludes upload; the other columns include NumPy boundaries. The dataset has 50,000 training rows and **256 query rows**, not 50,000 self queries.

### Dataset seed 51

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default | Native resident ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.569 | 1.062 | 0.383 | 0.381 | 1.87× | 0.67× | 0.687 |
| Fresh fit + query | 3.248 | 7.145 | 26.055 | 4.500 | 2.20× | 1.39× | 3.490 |
| Neighbors + distances | 2.080 | 5.663 | 25.829 | 4.397 | 2.72× | 2.11× | 1.931 |
| Indices only | 2.179 | 5.307 | 25.850 | 4.469 | 2.44× | 2.05× | 2.867 |
| Classifier, uniform | 2.629 | 4.316 | 25.918 | 4.646 | 1.64× | 1.77× | — |
| Probabilities, uniform | 1.848 | 3.744 | 25.989 | 4.207 | 2.03× | 2.28× | — |
| Regressor, uniform | 1.858 | 4.053 | 25.944 | 4.246 | 2.18× | 2.29× | — |
| Classifier, distance | 1.974 | 4.355 | 26.135 | 4.661 | 2.21× | 2.36× | — |
| Probabilities, distance | 2.870 | 6.856 | 26.208 | 4.990 | 2.39× | 1.74× | — |
| Regressor, distance | 2.862 | 6.776 | 26.010 | 4.728 | 2.37× | 1.65× | — |

### Dataset seed 52

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default | Native resident ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.553 | 1.061 | 0.374 | 0.378 | 1.92× | 0.68× | 0.709 |
| Fresh fit + query | 3.179 | 7.317 | 25.987 | 4.690 | 2.30× | 1.48× | 3.535 |
| Neighbors + distances | 1.603 | 3.791 | 25.840 | 4.684 | 2.36× | 2.92× | 1.825 |
| Indices only | 1.649 | 3.828 | 25.827 | 4.351 | 2.32× | 2.64× | 2.576 |
| Classifier, uniform | 2.820 | 6.520 | 25.975 | 4.259 | 2.31× | 1.51× | — |
| Probabilities, uniform | 1.906 | 4.335 | 25.848 | 4.215 | 2.27× | 2.21× | — |
| Regressor, uniform | 1.648 | 4.188 | 25.870 | 4.393 | 2.54× | 2.67× | — |
| Classifier, distance | 1.935 | 4.858 | 25.964 | 4.294 | 2.51× | 2.22× | — |
| Probabilities, distance | 2.869 | 6.775 | 26.120 | 4.798 | 2.36× | 1.67× | — |
| Regressor, distance | 2.841 | 6.770 | 26.023 | 4.272 | 2.38× | 1.50× | — |

### Dataset seed 53

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default | Native resident ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.576 | 1.080 | 0.379 | 0.367 | 1.87× | 0.64× | 0.676 |
| Fresh fit + query | 3.296 | 7.351 | 26.194 | 4.652 | 2.23× | 1.41× | 3.538 |
| Neighbors + distances | 2.822 | 6.792 | 25.852 | 4.433 | 2.41× | 1.57× | 3.152 |
| Indices only | 1.477 | 3.689 | 25.849 | 4.301 | 2.50× | 2.91× | 1.946 |
| Classifier, uniform | 1.742 | 4.363 | 25.907 | 4.161 | 2.50× | 2.39× | — |
| Probabilities, uniform | 1.946 | 4.399 | 26.028 | 4.341 | 2.26× | 2.23× | — |
| Regressor, uniform | 2.559 | 6.527 | 25.887 | 4.130 | 2.55× | 1.61× | — |
| Classifier, distance | 1.654 | 4.259 | 26.067 | 4.560 | 2.57× | 2.76× | — |
| Probabilities, distance | 2.867 | 6.604 | 26.010 | 4.530 | 2.30× | 1.58× | — |
| Regressor, distance | 1.720 | 4.177 | 25.869 | 4.318 | 2.43× | 2.51× | — |


The paired retained rounds reaching 2× for neighbors-plus-distances are **6/9, 6/9 and 0/9**, respectively. Fresh fit-plus-query reaches 2× in **0/9** rounds on every seed. Median speedups are not latency guarantees. The [complete target report](../benchmarks/results/knn-acceleration-2026-10-28.json) retains all samples, per-round comparisons, half-run medians, source hashes and quality checks.

## Full earlier-workload regression table

The reference is the same immediately previous implementation. Query performance improves in nine of ten shapes. The single-query, 32-feature median is 3.0% slower. The largest measured operation regression is 8.5% for distance-weighted single-query classification at eight features; all individual operations are shown below. These short-call differences include runtime variability.

### single_8_k5

8,192 training rows × 8 features; 1 queries; k=5.

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default | Native resident ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.171 | 0.569 | 0.137 | 0.144 | 3.33× | 0.85× | 0.475 |
| Fresh fit + query | 0.417 | 0.823 | 0.346 | 1.021 | 1.98× | 2.45× | 0.661 |
| Neighbors + distances | 0.306 | 0.312 | 0.251 | 0.739 | 1.02× | 2.41× | 0.435 |
| Indices only | 0.302 | 0.310 | 0.251 | 0.671 | 1.03× | 2.22× | 0.431 |
| Classifier, uniform | 0.369 | 0.581 | 0.314 | 0.770 | 1.58× | 2.09× | — |
| Probabilities, uniform | 0.349 | 0.575 | 0.285 | 0.734 | 1.65× | 2.10× | — |
| Regressor, uniform | 0.306 | 0.351 | 0.244 | 0.533 | 1.15× | 1.74× | — |
| Classifier, distance | 0.399 | 0.368 | 0.329 | 1.160 | 0.92× | 2.91× | — |
| Probabilities, distance | 0.363 | 0.373 | 0.289 | 0.809 | 1.03× | 2.23× | — |
| Regressor, distance | 0.341 | 0.372 | 0.264 | 0.767 | 1.09× | 2.25× | — |

### batch_8_k5

8,192 training rows × 8 features; 256 queries; k=5.

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default | Native resident ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.200 | 0.613 | 0.165 | 0.158 | 3.07× | 0.79× | 0.275 |
| Fresh fit + query | 0.593 | 1.062 | 4.399 | 1.538 | 1.79× | 2.59× | 0.806 |
| Neighbors + distances | 0.522 | 0.861 | 4.324 | 1.906 | 1.65× | 3.65× | 0.620 |
| Indices only | 0.488 | 0.672 | 4.287 | 1.321 | 1.38× | 2.71× | 0.817 |
| Classifier, uniform | 0.519 | 0.719 | 4.375 | 1.572 | 1.39× | 3.03× | — |
| Probabilities, uniform | 0.538 | 0.923 | 4.354 | 1.223 | 1.72× | 2.27× | — |
| Regressor, uniform | 0.499 | 0.868 | 4.296 | 1.300 | 1.74× | 2.61× | — |
| Classifier, distance | 0.552 | 0.816 | 4.520 | 1.798 | 1.48× | 3.25× | — |
| Probabilities, distance | 0.543 | 1.047 | 4.387 | 1.857 | 1.93× | 3.42× | — |
| Regressor, distance | 0.525 | 0.927 | 4.353 | 1.630 | 1.77× | 3.10× | — |

### batch_32_k5

8,192 training rows × 32 features; 256 queries; k=5.

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default | Native resident ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.223 | 0.638 | 0.173 | 0.186 | 2.86× | 0.83× | 0.299 |
| Fresh fit + query | 0.846 | 1.869 | 4.573 | 1.833 | 2.21× | 2.17× | 1.089 |
| Neighbors + distances | 0.691 | 1.361 | 4.460 | 1.626 | 1.97× | 2.35× | 0.842 |
| Indices only | 0.681 | 1.534 | 4.442 | 1.466 | 2.25× | 2.15× | 0.806 |
| Classifier, uniform | 0.719 | 1.313 | 4.514 | 1.679 | 1.83× | 2.34× | — |
| Probabilities, uniform | 0.919 | 1.568 | 4.510 | 1.456 | 1.71× | 1.58× | — |
| Regressor, uniform | 0.770 | 1.566 | 4.501 | 1.838 | 2.03× | 2.39× | — |
| Classifier, distance | 0.744 | 1.613 | 4.549 | 1.807 | 2.17× | 2.43× | — |
| Probabilities, distance | 0.965 | 1.553 | 4.511 | 1.846 | 1.61× | 1.91× | — |
| Regressor, distance | 0.777 | 1.508 | 4.505 | 1.875 | 1.94× | 2.41× | — |

### batch_32_k32

8,192 training rows × 32 features; 128 queries; k=32.

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default | Native resident ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.264 | 0.662 | 0.211 | 0.209 | 2.51× | 0.79× | 0.342 |
| Fresh fit + query | 2.055 | 2.501 | 3.216 | 2.051 | 1.22× | 1.00× | 2.173 |
| Neighbors + distances | 1.596 | 1.858 | 3.071 | 1.885 | 1.16× | 1.18× | 1.701 |
| Indices only | 1.580 | 1.813 | 3.032 | 1.482 | 1.15× | 0.94× | 1.630 |
| Classifier, uniform | 1.625 | 1.833 | 3.182 | 1.649 | 1.13× | 1.02× | — |
| Probabilities, uniform | 1.619 | 1.855 | 3.128 | 1.605 | 1.15× | 0.99× | — |
| Regressor, uniform | 1.582 | 1.808 | 3.045 | 1.833 | 1.14× | 1.16× | — |
| Classifier, distance | 1.855 | 1.881 | 3.234 | 1.843 | 1.01× | 0.99× | — |
| Probabilities, distance | 1.862 | 1.846 | 3.152 | 1.960 | 0.99× | 1.05× | — |
| Regressor, distance | 1.604 | 1.787 | 3.082 | 1.785 | 1.11× | 1.11× | — |

### batch_128_k5

4,096 training rows × 128 features; 128 queries; k=5.

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default | Native resident ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.315 | 0.760 | 0.251 | 0.249 | 2.41× | 0.79× | 0.605 |
| Fresh fit + query | 1.318 | 1.604 | 1.887 | 1.715 | 1.22× | 1.30× | 1.780 |
| Neighbors + distances | 0.749 | 0.778 | 1.721 | 1.463 | 1.04× | 1.96× | 0.846 |
| Indices only | 0.767 | 0.799 | 1.721 | 1.212 | 1.04× | 1.58× | 0.811 |
| Classifier, uniform | 0.763 | 0.821 | 1.751 | 1.231 | 1.08× | 1.61× | — |
| Probabilities, uniform | 0.769 | 0.793 | 1.721 | 1.229 | 1.03× | 1.60× | — |
| Regressor, uniform | 0.723 | 0.960 | 1.691 | 1.260 | 1.33× | 1.74× | — |
| Classifier, distance | 0.780 | 0.857 | 1.798 | 1.563 | 1.10× | 2.00× | — |
| Probabilities, distance | 0.759 | 0.779 | 1.736 | 1.545 | 1.03× | 2.03× | — |
| Regressor, distance | 0.768 | 0.748 | 1.707 | 1.427 | 0.97× | 1.86× | — |

### batch_1024_k5

2,048 training rows × 1024 features; 32 queries; k=5.

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default | Native resident ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.673 | 0.969 | 0.428 | 0.430 | 1.44× | 0.64× | 0.506 |
| Fresh fit + query | 2.123 | 2.492 | 2.587 | 2.288 | 1.17× | 1.08× | 1.980 |
| Neighbors + distances | 0.957 | 1.021 | 2.237 | 1.785 | 1.07× | 1.86× | 1.034 |
| Indices only | 0.960 | 1.064 | 2.199 | 1.583 | 1.11× | 1.65× | 1.002 |
| Classifier, uniform | 1.019 | 1.328 | 2.259 | 1.554 | 1.30× | 1.52× | — |
| Probabilities, uniform | 1.028 | 1.294 | 2.286 | 1.585 | 1.26× | 1.54× | — |
| Regressor, uniform | 0.994 | 1.271 | 2.224 | 1.531 | 1.28× | 1.54× | — |
| Classifier, distance | 1.054 | 1.289 | 2.315 | 1.883 | 1.22× | 1.79× | — |
| Probabilities, distance | 1.047 | 1.161 | 2.285 | 1.883 | 1.11× | 1.80× | — |
| Regressor, distance | 1.041 | 1.056 | 2.277 | 1.951 | 1.01× | 1.87× | — |

### batch_32_k64

4,096 training rows × 32 features; 128 queries; k=64.

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default | Native resident ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.244 | 0.442 | 0.199 | 0.174 | 1.81× | 0.71× | 0.506 |
| Fresh fit + query | 1.457 | 1.923 | 2.632 | 1.974 | 1.32× | 1.36× | 1.673 |
| Neighbors + distances | 1.076 | 1.327 | 2.539 | 1.701 | 1.23× | 1.58× | 1.131 |
| Indices only | 1.065 | 1.058 | 2.513 | 1.878 | 0.99× | 1.76× | 1.402 |
| Classifier, uniform | 1.103 | 1.335 | 2.669 | 1.918 | 1.21× | 1.74× | — |
| Probabilities, uniform | 1.097 | 1.309 | 2.674 | 1.802 | 1.19× | 1.64× | — |
| Regressor, uniform | 1.064 | 1.265 | 2.547 | 1.529 | 1.19× | 1.44× | — |
| Classifier, distance | 1.138 | 1.164 | 2.832 | 2.542 | 1.02× | 2.23× | — |
| Probabilities, distance | 1.149 | 1.420 | 2.722 | 2.589 | 1.24× | 2.25× | — |
| Regressor, distance | 1.094 | 1.389 | 2.567 | 2.024 | 1.27× | 1.85× | — |

### batch_16_k5

8,192 training rows × 16 features; 256 queries; k=5.

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default | Native resident ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.211 | 0.588 | 0.170 | 0.161 | 2.79× | 0.76× | 0.521 |
| Fresh fit + query | 0.693 | 1.144 | 4.392 | 1.659 | 1.65× | 2.39× | 0.912 |
| Neighbors + distances | 0.539 | 1.094 | 4.288 | 1.519 | 2.03× | 2.82× | 0.649 |
| Indices only | 0.542 | 1.102 | 4.277 | 1.183 | 2.03× | 2.18× | 0.635 |
| Classifier, uniform | 0.594 | 1.108 | 4.375 | 1.386 | 1.87× | 2.33× | — |
| Probabilities, uniform | 0.558 | 1.115 | 4.344 | 1.313 | 2.00× | 2.35× | — |
| Regressor, uniform | 0.601 | 1.122 | 4.311 | 1.643 | 1.87× | 2.73× | — |
| Classifier, distance | 0.601 | 1.137 | 4.435 | 1.735 | 1.89× | 2.89× | — |
| Probabilities, distance | 0.588 | 0.903 | 4.329 | 1.886 | 1.54× | 3.21× | — |
| Regressor, distance | 0.610 | 0.894 | 4.356 | 1.856 | 1.46× | 3.04× | — |

### single_32_k5

8,192 training rows × 32 features; 1 queries; k=5.

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default | Native resident ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.205 | 0.394 | 0.170 | 0.165 | 1.93× | 0.81× | 0.297 |
| Fresh fit + query | 0.504 | 0.674 | 0.372 | 1.113 | 1.34× | 2.21× | 0.717 |
| Neighbors + distances | 0.361 | 0.351 | 0.255 | 0.756 | 0.97× | 2.09× | 0.477 |
| Indices only | 0.346 | 0.348 | 0.249 | 0.522 | 1.01× | 1.51× | 0.463 |
| Classifier, uniform | 0.380 | 0.595 | 0.290 | 0.586 | 1.57× | 1.54× | — |
| Probabilities, uniform | 0.364 | 0.368 | 0.273 | 0.474 | 1.01× | 1.30× | — |
| Regressor, uniform | 0.348 | 0.346 | 0.253 | 0.799 | 0.99× | 2.30× | — |
| Classifier, distance | 0.394 | 0.375 | 0.295 | 0.949 | 0.95× | 2.41× | — |
| Probabilities, distance | 0.377 | 0.571 | 0.278 | 0.843 | 1.52× | 2.24× | — |
| Regressor, distance | 0.374 | 0.383 | 0.268 | 0.980 | 1.02× | 2.62× | — |

### batch_8_k32

8,192 training rows × 8 features; 128 queries; k=32.

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default | Native resident ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fit | 0.228 | 0.648 | 0.197 | 0.197 | 2.84× | 0.87× | 0.255 |
| Fresh fit + query | 1.992 | 2.379 | 3.124 | 2.046 | 1.19× | 1.03× | 2.171 |
| Neighbors + distances | 1.558 | 1.786 | 2.965 | 1.674 | 1.15× | 1.07× | 1.852 |
| Indices only | 1.546 | 1.521 | 2.959 | 1.358 | 0.98× | 0.88× | 1.602 |
| Classifier, uniform | 1.593 | 1.663 | 3.024 | 1.455 | 1.04× | 0.91× | — |
| Probabilities, uniform | 1.563 | 1.682 | 3.024 | 1.518 | 1.08× | 0.97× | — |
| Regressor, uniform | 1.554 | 1.544 | 2.948 | 1.364 | 0.99× | 0.88× | — |
| Classifier, distance | 1.706 | 1.856 | 3.134 | 1.757 | 1.09× | 1.03× | — |
| Probabilities, distance | 1.612 | 1.827 | 3.004 | 1.809 | 1.13× | 1.12× | — |
| Regressor, distance | 1.604 | 1.812 | 2.990 | 1.660 | 1.13× | 1.04× | — |

Current, frozen and resident implementations pass all regression audits. Both sklearn variants retain the earlier strict-distance failure on an exactly copied row in the 1,024-feature case: the residual is 2.47349e-6, with matching neighbor indices. No tolerance was relaxed. The [regression JSON](../benchmarks/evidence/knn-evidence-2026-10-31.md#artifact-bd0d6ecae6c132993a5d) preserves that comparator failure.

## Validation and reproduction

366 tests passed across neighbor parity, float32 edge cases, device/core tests, query grouping, pipelines, voting, invariants, patching, sklearn compliance, existing device logistic tests and benchmark-report guards. The final routing change was additionally covered by 65 focused passing tests. The new tests compare batched grouped scans with independent original one-query scans and verify that host mutations cannot affect the device index before its first query. Ruff, mechanical compliance and `git diff --check` pass.

```bash
python3 -m benchmarks.knn_acceleration \
  --output benchmarks/results/knn-acceleration-2026-10-28.json
```

Exploratory timings, including rejected candidates, are preserved in the [kernel experiment record](../benchmarks/results/knn-kernel-experiments-2026-10-28.json). They were not substituted for the public API results.

## Sustained-use audit: target also not met

This supplementary protocol was added after the original mixed-workload audit, to investigate the difference from isolated probes. It does not replace that audit. Each variant executes five warmups and 25 synchronized, individually timed calls inside one worker process, without inter-process requests between samples. All validation and NumPy boundaries remain timed; accuracy auditing remains outside each timer. Variant blocks rotate by seed. Fresh fit-plus-query still creates a new model on every call. Default sklearn still uses sixteen threads.

All accuracy, backend and source checks pass. Queries are **1.60–1.77×** default-thread sklearn, while fresh fit-plus-query is **1.41–1.46×**. Relative to the previous implementation, queries are **3.6–13.5% slower**, and fresh fit-plus-query is **4.0–5.4% slower**. No inference operation achieves the 2× target on every dataset. This protocol-dependent behavior means the isolated kernel gains cannot be presented as a consistent application speedup.

### Sustained use, dataset seed 51

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default |
|---|---:|---:|---:|---:|---:|---:|
| Fit | 0.469 | 0.623 | 0.277 | 0.301 | 1.33× | 0.64× |
| Fresh fit + query | 3.151 | 3.031 | 25.164 | 4.610 | 0.96× | 1.46× |
| Neighbors + distances | 2.529 | 2.442 | 24.844 | 4.480 | 0.97× | 1.77× |
| Indices only | 2.710 | 2.209 | 24.795 | 4.001 | 0.82× | 1.48× |
| Classifier, uniform | 2.745 | 2.347 | 25.016 | 4.189 | 0.86× | 1.53× |
| Probabilities, uniform | 2.196 | 2.417 | 24.971 | 4.183 | 1.10× | 1.90× |
| Regressor, uniform | 2.735 | 1.627 | 24.904 | 4.032 | 0.59× | 1.47× |
| Classifier, distance | 2.570 | 2.404 | 24.998 | 4.398 | 0.94× | 1.71× |
| Probabilities, distance | 2.748 | 2.424 | 25.077 | 4.518 | 0.88× | 1.64× |
| Regressor, distance | 2.721 | 2.483 | 25.001 | 4.551 | 0.91× | 1.67× |

### Sustained use, dataset seed 52

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default |
|---|---:|---:|---:|---:|---:|---:|
| Fit | 0.462 | 0.837 | 0.265 | 0.298 | 1.81× | 0.65× |
| Fresh fit + query | 3.166 | 3.007 | 24.777 | 4.462 | 0.95× | 1.41× |
| Neighbors + distances | 2.678 | 2.380 | 24.615 | 4.298 | 0.89× | 1.60× |
| Indices only | 2.701 | 1.599 | 24.614 | 4.118 | 0.59× | 1.52× |
| Classifier, uniform | 2.752 | 2.418 | 25.081 | 4.377 | 0.88× | 1.59× |
| Probabilities, uniform | 1.337 | 2.326 | 24.861 | 4.180 | 1.74× | 3.13× |
| Regressor, uniform | 2.722 | 2.382 | 24.851 | 4.080 | 0.88× | 1.50× |
| Classifier, distance | 2.762 | 2.236 | 24.875 | 4.314 | 0.81× | 1.56× |
| Probabilities, distance | 2.770 | 2.209 | 24.745 | 4.247 | 0.80× | 1.53× |
| Regressor, distance | 2.749 | 1.663 | 24.998 | 4.613 | 0.60× | 1.68× |

### Sustained use, dataset seed 53

| Operation | Current ms | Previous ms | sklearn 1 thread ms | sklearn default ms | vs previous | vs default |
|---|---:|---:|---:|---:|---:|---:|
| Fit | 0.464 | 0.653 | 0.335 | 0.307 | 1.41× | 0.66× |
| Fresh fit + query | 3.151 | 2.995 | 25.243 | 4.454 | 0.95× | 1.41× |
| Neighbors + distances | 2.741 | 2.416 | 24.801 | 4.735 | 0.88× | 1.73× |
| Indices only | 2.710 | 2.438 | 24.995 | 4.067 | 0.90× | 1.50× |
| Classifier, uniform | 2.790 | 1.633 | 25.109 | 3.971 | 0.59× | 1.42× |
| Probabilities, uniform | 2.750 | 1.630 | 25.019 | 4.049 | 0.59× | 1.47× |
| Regressor, uniform | 2.720 | 2.196 | 24.977 | 3.999 | 0.81× | 1.47× |
| Classifier, distance | 2.553 | 2.233 | 25.080 | 4.508 | 0.87× | 1.77× |
| Probabilities, distance | 2.750 | 2.333 | 25.032 | 4.169 | 0.85× | 1.52× |
| Regressor, distance | 2.748 | 1.635 | 25.015 | 4.530 | 0.60× | 1.65× |

[Complete sustained-use results](../benchmarks/results/knn-steady-2026-10-28.json).

```bash
python3 -m benchmarks.knn_steady \
  --output benchmarks/results/knn-steady-2026-10-28.json
```
