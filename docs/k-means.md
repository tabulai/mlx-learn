# K-means on Apple Silicon

`mlxlearn.cluster.KMeans` implements Lloyd's algorithm for dense data on the
Apple Silicon GPU. Each iteration assigns every row to its nearest centroid,
then recomputes the centroid means. It uses FP32 arithmetic through MLX and
custom Metal kernels and requires a working Metal GPU.

```python
import numpy as np
from mlxlearn.cluster import KMeans

X = np.asarray(embeddings, dtype=np.float32)
model = KMeans(
    n_clusters=256,
    init="k-means++",
    n_init=1,
    max_iter=100,
    random_state=10,
).fit(X)

centers = model.cluster_centers_
labels = model.labels_
training_mse = model.inertia_ / len(X)
```

Returned labels describe the final centroids. Fitting includes a final
assignment and inertia calculation; `inertia_` is the sum of squared distances
to those centers. `n_iter_` records the number of Lloyd updates performed.
`predict(X)` returns nearest-center labels, `score(X)` returns negative inertia,
and `transform(X)` returns Euclidean distances to every center.

## Parameters and supported scope

| Parameter | Default | Meaning |
|---|---|---|
| `n_clusters` | `8` | Number of centroids; cannot exceed the number of training rows. |
| `init` | `"k-means++"` | `"k-means++"`, `"random"`, or an explicit `(n_clusters, n_features)` matrix. |
| `n_init` | `"auto"` | Number of initializations; automatic mode uses ten for random initialization and one for k-means++ or explicit centers. |
| `max_iter` | `300` | Maximum number of full Lloyd updates; must be an integer. |
| `tol` | `1e-4` | Convergence tolerance. Use `0` for comparisons that disable positive-tolerance stopping, and still inspect `n_iter_`. |
| `random_state` | `None` | Random seed or compatible random-state object. |
| `algorithm` | `"lloyd"` | Only Lloyd's algorithm is supported. |
| `copy_x` | `True` | Accepted for API compatibility; this implementation preserves the caller's input for either setting. |
| `assignment_batch_size` | `None` | Optional row limit for nearest-centroid assignment tiles. |
| `statistics_batch_size` | `65536` | Row chunk size for accumulating centroid statistics. |
| `verbose` | `False` | Print training progress. |

Explicit initial centers select one initialization. Requesting more than one
with an explicit matrix emits a `RuntimeWarning` and uses one. When several
initializations are requested, the returned model is selected by final inertia.
K-means++ initialization uses scikit-learn's public helper during preparation;
the Lloyd iterations run on Metal. `n_init_`, `best_restart_`,
`convergence_reason_` and `iteration_stats_` describe the selected run.

The convergence threshold is `tol * mean(feature_variances)`, applied to the
sum of squared center shifts. Unchanged labels are checked first. Even with
`tol=0`, unchanged labels or an exactly zero shift can terminate training.

Inputs must be finite, dense, real-valued arrays. Computation uses FP32, including
when inputs have another floating-point dtype. Sparse data, non-`None`
`sample_weight`, callable initializers, Elkan's algorithm and reduced-precision
KMeans are unsupported.
Fractional epoch budgets belong to [EMeans](e-means.md), not this estimator.

The API follows scikit-learn conventions but does not promise identical
centroids, random initialization, tie-breaking, or inertia bits. GPU reductions
and floating-point ordering differ. To compare implementations from exactly the
same starting centers, pass the same explicit FP32 initialization to each:

```python
seed, k = 10, 1024
initial = X[np.random.RandomState(seed).permutation(len(X))[:k]].copy()
model = KMeans(
    n_clusters=k,
    init=initial,
    n_init=1,
    max_iter=3,
    tol=0,
    random_state=seed,
).fit(X)
```

## Device execution and memory

The complete training matrix and centroids reside on the GPU. Assignment and
statistics use chunks, but this is not an out-of-core or distributed estimator.
The chunk parameters bound individual work buffers rather than total process
memory. A memory-mapped NumPy input does not make training stream from disk.
`transform` returns the complete `n_samples × n_clusters` distance matrix and
therefore needs memory proportional to that output; it is not appropriate for
materializing every distance in the 10M benchmark.

Empty clusters are relocated to farthest assigned points, with those points'
contributions removed from their donor clusters. Remaining zero-count centers
receive the largest occupied cluster's mean. Selection among tied farthest
points can differ from scikit-learn. This path uses per-row distances and
selected rows rather than a full matrix of each row's assigned center.

For controlled resident-data benchmarks, `prepare_device(X)` validates and
prepares the input and initialization outside the timed region.
`train_device(prepared)` then performs training, final assignments and inertia
on the device. This lower-level interface separates preparation from fit timing;
ordinary application code should use `fit`.
The prepared object contains owned FP32 data, a tuple of initializations, the
assignment origin and tolerance. The returned device result contains centers,
labels, inertia, iteration history and restart diagnostics. `train_device`
sets `fit_result_`; it does not populate the NumPy fitted attributes produced
by `fit`.
Prepared inputs are bound to the estimator configuration. Changing parameters
or replacing an explicit initialization array requires preparing again.

Retaining a full NumPy input while preparing a separate GPU representation can
temporarily require both allocations. Process physical footprint, resident set
size and MLX active allocation measure different things. Report preparation
peaks separately when a benchmark excludes preparation from its fit window.

## Comparison with e-means

KMeans computes centroid means after each complete assignment pass. EMeans
updates its moving centers after each mini-batch, with a scheduled exponential
moving average and revival of low-usage clusters. Its default closing centers
are means of the moving assignments from the final pass. Equal epoch budgets
therefore compare optimization behavior at a shared pass budget; they do not
guarantee equal clustering quality.

The five-dataset report
[KMeans and e-means at 10M](k-means-10m-comparison-2026-11-15.md)
records the campaign's completion status and measured results. Its four modes are Metal KMeans, scikit-learn
KMeans, Metal EMeans and original Faiss EMeans, all in FP32 with identical
explicit initialization. DEEP, SPACEV, SimSearchNet++ and DINO use two epochs;
SIFT uses three. All four modes are measured anew with final assignments and
inertia included. Earlier e-means measurements with different budgets or only
online final-pass labels remain historical results.

The scikit-learn benchmark uses unmodified Lloyd KMeans with `n_init=1`,
`tol=0` and `copy_x=False`. Scikit-learn centers and restores this input in place,
which can change FP32 bits, so the benchmark reloads exact cached data outside
the timer between warmup and timed fits. Its empty-cluster relocation can create
large full-matrix temporaries; disabling the initial input copy does not remove
those allocations. Native label dtypes and observed iteration counts are
recorded, while shared held-out and training-sample scoring provides the quality
comparison independently of each implementation's inertia reduction.
