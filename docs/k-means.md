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
K-means++ initialization selects the centers of scikit-learn's public helper
for the same `random_state`, with most of the work on Metal (see
[K-means++ initialization](#k-means-initialization)); the Lloyd iterations run
on Metal. `n_init_`, `best_restart_`, `convergence_reason_` and
`iteration_stats_` describe the selected run.

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
and floating-point ordering differ. K-means++ starting centers are those of
scikit-learn's `kmeans_plusplus` helper on the float32 input; scikit-learn's
own `KMeans` seeds from mean-centered data and can start elsewhere. To compare
implementations from exactly the same starting centers, pass the same explicit
FP32 initialization to each:

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

## K-means++ initialization

`init="k-means++"` selects the rows that
`sklearn.cluster.kmeans_plusplus(X32, n_clusters, random_state=...)` selects on
the same machine, where `X32` is the float32 input, and it consumes the same
random numbers. A fixed `random_state` therefore gives the centers it gave when
`KMeans` called that helper directly. Like the helper, the selection follows the
installed scikit-learn release: 1.8 changed the helper's sampling prefix sums
from float64 to float32, which changes the candidates a seed draws.

The helper draws `2 + int(log(n_clusters))` candidates for every new center
and evaluates float64 distances from each of them to every row. Only rows
nearer to a candidate than to their current center matter: on MNIST 49% of the
(candidate, row) pairs for the second center and 0.16% for the 257th. Each
step now works as follows.

1. A Metal kernel compares every (candidate, row) pair in float32 and proves
   most pairs unchanged with a rigorous error bound. For inputs with at least
   256 features it first compares 64-dimensional images of the rows, whose
   distances bound the true ones from below, and reads full rows only for the
   pairs the images cannot settle. On data without a few dominant directions
   the images settle little, and the kernel that reads every row takes over.
2. For the remaining pairs the kernel returns a float32 bracket of the
   distance. Potentials summed over both bracket ends rule out candidates that
   cannot win, and the host evaluates the other candidates' open pairs in
   float64.
3. Sampling, candidate potentials and the choice among candidates use the
   helper's NumPy expressions on the resulting values.

The match with the helper is certified. The helper rounds float64 results to
float32, and its float64 dot products come from a BLAS routine whose last bits
depend on the shape of the call. A product over selected rows is therefore an
interval. If both ends round to the same float32, that value is the helper's;
if not, the helper's own chunk is recomputed with its shapes. A row that
coincides with a candidate has only round-off for a distance and keeps its
interval. A step is accepted when both interval ends give the same candidates
and the same potentials. When they do not, the helper itself runs from the
initial random state. This happens when more centers are requested than there
are distinct rows, where the helper samples its own round-off.

Small problems call the helper directly: it is as fast as a GPU round trip
below about `trials * n_samples * (n_features + 48) = 2**22`. So do MLX
releases that cannot compile kernels in Metal's safe math mode, which the
error bounds assume.

Besides the input, seeding holds two `trials x n_samples` float32 tables
(lower and upper bounds), float64 row norms, and for inputs with at least 256
features the `n_samples x 64` float32 images.

Seeding time on MNIST (60,000 x 784, pixels in [0, 1], `random_state=10`,
M4 Max; the same rows are selected in every case):

| Centers | scikit-learn helper | Metal screen | Speedup |
|---:|---:|---:|---:|
| 1,024 | 26.9 s | 0.90 s | 30x |
| 2,048 | 54.2 s | 1.62 s | 34x |
| 4,096 | 108.6 s | 2.82 s | 38x |
| 8,192 | 217.1 s | 5.41 s | 40x |

Whole fits with default settings on the same data. `init="random"` runs ten
restarts, k-means++ one:

| Centers | `KMeans(k)` before | `KMeans(k)` | `KMeans(k, init="random")` |
|---:|---:|---:|---:|
| 1,024 | 27.5 s | 1.39 s | 6.70 s |
| 2,048 | 54.9 s | 2.40 s | 7.97 s |
| 4,096 | 110.4 s | 4.42 s | 10.66 s |
| 8,192 | 218.7 s | 7.02 s | 13.59 s |

Scripts, per-stage profiles and raw results are in
[`benchmarks/experiments/kmeans-plusplus-2026-11-19`](../benchmarks/experiments/kmeans-plusplus-2026-11-19/README.md).

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

That report holds the pass budget equal, so the two Metal methods take the same
time and differ only in MSE. The
[Lloyd-25 addendum](k-means-lloyd25-reference-2026-11-17.md) instead runs KMeans
for 25 iterations against EMeans at its two- or three-epoch budget, which is the
comparison the e-means paper makes.

The scikit-learn benchmark uses unmodified Lloyd KMeans with `n_init=1`,
`tol=0` and `copy_x=False`. Scikit-learn centers and restores this input in place,
which can change FP32 bits, so the benchmark reloads exact cached data outside
the timer between warmup and timed fits. Its empty-cluster relocation can create
large full-matrix temporaries; disabling the initial input copy does not remove
those allocations. Native label dtypes and observed iteration counts are
recorded, while shared held-out and training-sample scoring provides the quality
comparison independently of each implementation's inertia reduction.
