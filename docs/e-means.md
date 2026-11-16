# E-means on Apple Silicon

`mlxlearn.cluster.EMeans` implements the scheduled exponential-moving-average
clustering algorithm from [Bessa et al., arXiv:2611.09501v1](https://arxiv.org/abs/2611.09501).
It runs nearest-centroid assignment, batch statistics, EMA updates, and donor
splitting on the Apple Silicon GPU through MLX and custom Metal kernels.
It requires a working Metal GPU and does not silently fall back to CPU.

```python
import numpy as np
from mlxlearn.cluster import EMeans

X = np.asarray(embeddings, dtype=np.float32)
model = EMeans(n_clusters=1024, max_iter=3, random_state=1234).fit(X)
centers = model.cluster_centers_
labels = model.labels_
test_mse = -model.score(test_embeddings) / len(test_embeddings)
```

`max_iter` is an epoch budget, not a mini-batch count. `batch_size=None` and
`alpha0=None` select the batch size and initial EMA step from the scaling laws;
`model.recipe_` records their resolved values. Training retains the last,
possibly shorter, batch of each pass. Fractional budgets follow the reference
extension: 2.5 epochs runs one full pass, one half pass, then one full closing
pass. A budget below one epoch uses only that shuffled prefix.

By default, fitting makes a final assignment pass so `labels_` describes the
returned centers and `inertia_` is the associated sum of squared distances.
Use `compute_labels=False` to omit that extra pass when measuring training:
`labels_` and `inertia_` are then `None`; `predict` and `score` remain available.
`fit_predict` computes final labels even if `compute_labels=False`.

## Finalization and spherical clustering

The published algorithm provides two finalizers. The default `last_epoch`
returns the means of the points assigned to each cluster during the last pass,
while the centers were moving. It does not run an additional Lloyd update.
`slow_ema` uses a separate constant-speed EMA of the same assigned data, with
the same donor/child split mapping. `ema` returns the main optimizer state.

```python
model = EMeans(
    n_clusters=1024,
    max_iter=3,
    finalization="slow_ema",
    slow_ema_half_life=100,  # logical mini-batches
    random_state=1234,
).fit(X)
```

A half-life `h` gives the second EMA a step of `1 - 2**(-1/h)`. When no
half-life is given, its step is `slow_ema_multiplier * 0.1 * alpha0`, capped
at one; the default multiplier is one. This expresses the tuning convention
in Appendix I and does not claim that a universal slow-EMA speed is optimal.
`ema_centers_` and `last_epoch_centers_` are always available. Specifying a
half-life also computes `slow_ema_centers_` when another finalizer is selected.

Set `spherical=True` to normalize input vectors and assignment centroid
directions, keeping the unprojected EMA state internally. Zero rows remain
zero. Assignment maximizes inner product, and the reported spherical objective
is `2 - 2 * dot(x, center)`; its value for a zero vector is two, following the
reference's objective convention. `transform` returns its square root in
spherical mode, and Euclidean distances in ordinary mode.

## Paper recipe versus current Faiss recipe

The [pinned Faiss reference](https://github.com/facebookresearch/faiss/tree/ebab8f784c1d177129602269d5ad966c525fbdcb/contrib/e_means)
differs from the attached paper in several numerical and boundary conventions.
The estimator makes this choice explicit rather than mixing their defaults.

| Setting | `recipe="paper"` (default) | `recipe="faiss"` |
|---|---|---|
| Laws | Rounded coefficients in Table 2 | Precise coefficients shipped in `base.py` |
| Automatic small batch | Above 2 million rows | At 16 or fewer dimensions |
| Automatic batch cap | Number of rows | Reference half-dataset caps |
| Initial cluster mass | `1e-10` | `2**-16` |
| Split perturbation | BF16 epsilon, `2**-7` | `1e-4` |
| Default revival | Threshold `0.1` | Disabled at two or fewer dimensions; otherwise `0.1` |
| Step-law lower clamp | None | `1e-4` |
| Empty-cluster mass when step size α=1 | Zero, per Equation 2 | Previous mass retained |

For either recipe, `batch_law="throughput"` selects the larger law;
`batch_law="quality"` divides its unrounded batch by eight. An explicit
`batch_size` bypasses the batch law, but the step law still uses the resulting
batch size unless `alpha0` is also specified.

Faiss mode matches the statistical recipe, not bitwise execution. Metal
reductions have a different order, donor-selection Gumbel keys are float32
rather than the CPU reference's float64. In float32/float16 mode multiplication
leaves zero-valued donor centers at zero; BF16 mode repairs collapsed represented
splits with an adjacent BF16 value, matching the reference's separation rule.
Inputs in spherical mode are normalized by this estimator;
the reference expects the caller to normalize them. The current reference's
random initialization and epoch permutation seed rules are preserved.

Donor random-number consumption also differs deliberately. Faiss draws donor
uniforms only when a revival check finds at least one split. The GPU path draws
them at every five-batch check, including checks with no splits, so it never
has to synchronize with the CPU to decide whether to advance the generator.
After an empty check, later revivals therefore use different uniforms under the
same seed. Both use independent draws for the same mass-weighted sampling
without replacement, subject to their floating-point precision, but their
seeded trajectories need not match. Shared initialization and epoch order do
not imply identical centroids or final MSE. The relative contributions of this
RNG difference, reduction order, centered assignment and algebraically
equivalent EMA arithmetic have not been isolated by an ablation.

## Metal implementation and precision

For float32, `assignment="auto"` uses MLX's tuned Metal matrix multiplication in bounded
row tiles, then a custom SIMD argmin. On M5-generation GPUs, MLX would round float32
operands to TF32 there, so those products use mlx-learn's binary32 simdgroup kernel
instead (`mlxlearn_device._binary32`). `assignment="fused"` selects the custom
SIMD-group matrix implementation, which keeps a partial winner per 64-center
block and merges those winners instead of storing the full distance matrix.
These are exhaustive searches across all centers; float32 arithmetic can still
change very close ties. Optional `assignment_batch_size` bounds the row tile
independently of the statistical mini-batch size.

For float32/float16, the default row tile targets a score-matrix scratch budget of one sixty-fourth
of Metal's recommended working-set size, capped at 1 GiB (with a 32 MiB floor).
On the 128 GiB M4 Max used for these measurements, that cap is 1 GiB. This is
the score-tile budget, not a bound on total model or training memory. An explicit
`assignment_batch_size` overrides it.

Batch coordinates and counts are accumulated by a fused Metal scatter-add.
A separate fused kernel updates centroid positions and masses. Revival runs
every five mini-batches, uses weighted sampling without replacement, and
computes its dynamic split count on the GPU. The host produces seeded random
permutations and donor random variates. No cluster statistics are copied back
to the CPU between batches.

Float32/float16 Euclidean assignment subtracts a fixed global mean from both operands before
the dot products to reduce cancellation on data with a large common offset.
EMA statistics and multiplicative splits stay in the original coordinates,
preserving their intended geometry. Squared residuals for selected labels are
recomputed directly rather than recovered from the dot-product identity.

`precision="float32"` is the default. Explicit `precision="float16"` casts
assignment operands after centering; matrix accumulation, norms, statistical
state and residual scoring remain float32. It always uses the fused kernel to
avoid a half-precision score matrix. It is an approximate assignment option,
not the paper's BF16, FP8, or INT8 implementation. Unsafe half casts are
rejected using an upfront bound that also accounts for possible donor splits;
this bound can conservatively reject borderline inputs. Use float32 for them.

`precision="bfloat16"` implements the reference H100 **cast** route. Resident
inputs and initial centroid state are float32. Each gathered mini-batch is
rounded to BF16 inside training; both assignment and coordinate sums use these
represented values. Centroid mirrors are BF16, with norms computed in float32
from the represented values. Dot products, batch sums, optimizer state and
returned centroids remain float32. There is no centering before BF16 rounding.
The primary EMA step and decay are rounded independently to float32. Collapsed
splits move child coordinate zero to the next representable BF16 value, with a
finite inward fallback. This includes zero and subnormal coordinates.

BF16 `assignment="auto"` uses a native BF16 Metal SIMD matrix kernel with
coalesced tile loads and an argmin computed directly from accumulator registers.
It stores a score and index per 64-center tile instead of a full score matrix.
For large 96- and 128-dimensional assignments, partial scores use contiguous
row writes and a matching coalesced merge, with a wider row tile. Other
dimensions retain the previous tile layout. BF16 training also gathers shuffled
rows directly into BF16 with vector loads, avoiding an intermediate float32 batch
while preserving the represented values and permutation exactly.
The default BF16 partial-score/index budget is capped at 256 MiB and also
respects the smaller device-dependent budget above. An explicit row limit may
reduce it further. This is workspace, not total process or MLX memory.
The explicit `assignment="gemm"` baseline widens BF16-rounded operands before
float32 GEMM, preserving float32 scores; a plain BF16-output GEMM would add
unwanted score rounding. Both routes use the same BF16 statistics and finalizer.
Metal and CUDA accumulation order and random streams still differ.
The native BF16 path was validated on M4 Max with MLX 0.32.4 and macOS 26.6.2;
other Apple GPU/software combinations were not benchmarked in this campaign.

BF16 `predict` and `score` use the represented operands. `transform` continues
to report distances from float32 input rows to float32 returned centers. The
benchmark separately measures quality using original float32 inputs and direct
float64 residuals, outside training. Online objective and count diagnostics are
float32 because Metal has no native float64 arithmetic.

All inputs must be finite. Float32 mode also checks a conservative magnitude
bound before assignment: it covers squared distances, per-pass objectives,
accumulated coordinates and possible split growth. Inputs that exceed it raise
a rescaling error rather than allowing overflowing GPU arithmetic. Extremely
small EMA steps that would underflow float32 are rejected. These checks run
before training and when receiving new inference inputs, not between batches.

## Implementation scope

This is the core e-means optimizer and its finalizers, implemented for one
Apple Silicon GPU, with a spherical extension. It is not a complete port of
every large-scale execution path in the paper and reference repository.

| Paper/reference component | Status here |
|---|---|
| Algorithm 1; scaling laws; cosine schedule; donor splitting | Implemented, with the explicit recipe differences above |
| Last-epoch means and slow EMA (Section 3.2, Appendix I) | Implemented |
| Exhaustive GPU assignment | Implemented in Metal/MLX; float32 default |
| BF16 cast route (Appendix J) | Implemented with native BF16 Metal assignment, float32 accumulation/state |
| E4M3 FP8, direct INT8 and calibrated/rescaled INT8 (Appendix J) | Not implemented; optional centered float16 is a different approximation |
| Dimension-specialized CuTe kernels and two-pass query grouping (Appendix J.1) | Not ported; the Metal tile kernel is a separate implementation |
| CPU-resident centroid tables transferred to GPU in pieces (Appendix J.1) | Not implemented; the complete centroid table must fit |
| Multi-GPU data/centroid sharding and collectives (Appendix M.1) | Not implemented |
| SSD staging, prefetching and block-biased sampling (Appendix M.1) | Not implemented |
| IVF approximate centroid assignment and index rebuild schedules (Appendix M.2) | Not implemented |
| IVF retrieval/index construction and Recall@10 experiments (Appendix L) | Not implemented or benchmarked |

Training currently loads the full input matrix onto the GPU. A memory-mapped
NumPy input does not turn this into a streaming implementation. The complete
10-billion-vector experiment, its distributed configurations and its GPU-hour
claims have not been reproduced here. `transform` returns an
`n_samples × n_clusters` result and therefore requires that much host memory
even though computation is tiled.
GPU row permutations require at most `2**31 - 1` input rows. Centroid and
statistical-batch element counts must each fit unsigned 32-bit indexing;
oversized configurations raise an error rather than wrapping their indices.
Statistical batches are limited to `2**24` rows so float32 atomic counts remain
exact. Larger datasets remain supported through multiple statistical batches.

## Reproducing measurements

The benchmark driver uses the pinned Faiss download and split definitions for
the public core datasets. Additional VIBE corpora use documented deterministic
holdouts because their Appendix D.3 split indices are not shipped. A smaller
`--rows` run is explicitly a subset experiment, not a full paper-scale result.
Matching a dataset name or a training-set size does not establish reproduction
of every paper figure: cluster counts, seeds, precision, finalizer, epoch budget,
test split and hardware are recorded separately. The reported Apple Silicon
measurements cover only the configurations present in the result files.

From the repository root, with the benchmark dependencies and pinned Faiss
checkout available:

```bash
.venv/bin/python -m benchmarks.e_means.run download --suite core1m
.venv/bin/python -m benchmarks.e_means.run run --suite core1m \
  --methods metal metal-lloyd --epochs 3 --lloyd-iters 25 \
  --seeds 10 11 12 --repeats 3 --recipe paper
.venv/bin/python -m benchmarks.e_means.run summary
```

Use `--faiss-source` and `--cache` to override the reference checkout and dataset
cache locations. `--methods faiss-emeans faiss-lloyd` selects CPU baselines;
`--recipe faiss` uses the reference recipe for e-means comparisons. CPU and GPU
wall-clock comparisons are hardware comparisons as well as implementation
comparisons. The same-device Metal Lloyd baseline helps distinguish the
algorithmic epoch reduction from CPU/GPU differences.

Each run records input/split and initialization hashes, software and hardware,
resolved recipe, synchronized training time, and held-out MSE. Downloads,
external data preparation, sampled initialization, warm-up and evaluation are
excluded from fit timing; input validation and NumPy-to-device conversion inside
`fit` are included. `optimization_time_` excludes the optional final labels pass;
`fit_time_` includes it. `iteration_stats_` contains moving-assignment objectives,
which should not be confused with loss evaluated against fixed final centers.

See the benchmark directory's dataset manifest and recorded results for the
actual coverage, configurations and measurements produced on this machine.

The [direct Faiss comparison](e-means-faiss-comparison-2026-11-14.md) uses a
separate fresh-process protocol with both implementations on the Faiss recipe.
It measures fit-window physical footprint and RSS externally, and records
MLX allocator usage separately. Its first-fit times are distinct from warmed
timings in the earlier [Metal Lloyd comparison](e-means-benchmarks-2026-11-14.md).

For resident-data timing, `prepared = model.prepare_device(X)` validates and
uploads inputs and initial centers once. `model.train_device(prepared,
capture_assignments=True)` restarts from that initialization, keeps outputs on
the GPU, and captures online final-pass int64 labels. It includes shuffling,
casting, state initialization, training, diagnostics and finalization. It does
not perform the extra final reassignment used by sklearn's `compute_labels`.
Rows not visited in a sub-one-epoch fit have label -1. This low-level method
returns device outputs without populating the estimator's host fitted attributes.
Prepared inputs require the same estimator configuration, including the same
explicit initialization object.

The [H100 protocol audit](../benchmarks/e_means/H100_PROTOCOL.md) and
`benchmarks/e_means/h100.py` specify the separate BF16 rerun: full-fit warm-up,
resident inputs, published epoch budgets where identifiable, last-epoch means,
and the reference's adaptive one-or-three timing calls. VIBE budget/seed/split
uncertainties are recorded explicitly; these are not exact H100 reproductions.
Historical float32 result files retain their original source hashes. Their
four-file implementation snapshot is preserved in
`benchmarks/e_means/reference/metal-v1-source.zip`.

The [hardware-gap investigation](e-means-gap-investigation-2026-11-14.md)
compares the subsequent optimizations against the frozen native-BF16 version in
`benchmarks/e_means/reference/metal-v2-source.zip`. Its `gap_bench.py` runner uses
fresh paired processes and the same resident-fit timing/evaluation routines.
Historical timings are kept separate because unchanged-code timing also varied
between sessions. Matrix-throughput diagnostics are empirical reference points,
not official Apple hardware limits or replacements for full clustering fits.

The [five-dataset 10M-vector comparison](e-means-10m-comparison-2026-11-14.md)
covers DEEP, SPACEV, SimSearchNet++, DINO and SIFT with 65,536 centroids.
It compares Metal FP32 with original Faiss CPU FP32 using matched epoch budgets,
finalization and warmed resident-fit timing, and reports native BF16 separately.
The report includes test MSE, fit-window physical footprint, preparation peaks,
source archives and all 15 raw records. Preparation memory matters: the largest
observed DINO preparation peak was 78.10 GiB, above its 43–45 GiB Metal fit peaks.
