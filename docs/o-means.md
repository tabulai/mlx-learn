# O-means on Apple Silicon

`mlxlearn.cluster.OMeans` is [e-means](e-means.md) with a two-level
nearest-centroid search. E-means compares every training row with every
centroid; o-means compares it with the centroids of a few nearby coarse cells
only, 10–21% of them on the embedding datasets measured here. It follows the
[o-means post](https://owlgebra.ai/blog/o-means/) and the `contrib/o_means`
package of the owlgebra-ai/faiss fork (MIT). The Metal kernels are a separate
implementation; the upstream CUDA kernels were not ported.

```python
import numpy as np
from mlxlearn.cluster import OMeans

X = np.asarray(embeddings, dtype=np.float32)
model = OMeans(n_clusters=16384, max_iter=2, random_state=1234).fit(X)
centers = model.cluster_centers_
labels = model.labels_          # full search against the returned centers
```

`OMeans` is for large `n_clusters`: thousands to 65,536 centroids on one Apple
GPU, with the data resident there. Below about 4,200 centroids a row's probes
cover more than half of the centroids and little is saved; `OMeans` then
trains with the full search, exactly as e-means does. It requires a Metal GPU and never falls
back to the CPU. It is an explicit estimator and is not part of
`patch_sklearn()`.

## What changes relative to e-means

The recipe, starting centroids, mini-batch EMA update, cosine step schedule,
split revival and epoch plan are those of `EMeans(recipe="faiss")`. Two things
differ.

**Two-level assignment.** The centroids are grouped into `n_cells` coarse
cells by k-means over the centroids themselves. A row is scored against the
cell centres, keeps its `n_probe` nearest cells and is compared with the
centroids in those cells only. With 16,384 centroids there are 128 cells of
128 centroids on average, so a row probing 16 cells evaluates about
128 + 16 × 128 = 2,176 distances instead of 16,384. The search is
approximate: a row whose nearest centroid sits in a cell it did not probe
receives the best centroid of the cells it did probe.

Centroids move after every batch, so the cells are kept current:

- During the first pass, every five batches each centroid is filed again under
  its nearest cell centre. Revival is most active then, and a new child
  centroid would otherwise stay filed under the dead centroid it replaced.
- After the first pass, every half pass the cell centres are recomputed with
  two Lloyd iterations, starting from the previous centres.
- Between two builds, each split records a link from the donor to its child.
  The assignment also tests the child and grandchild of a row's best match.

**Closing mean.** E-means ends with the mean of the rows assigned to each
centroid during the last pass. Below two epochs, o-means weights every batch
of the run by `exp(-r / n_samples)`, where `r` is the number of rows processed
after that batch. A split gives donor and child half of the donor's
accumulated mass each. From two epochs up o-means uses e-means' last-pass
mean. In the measurements below the recency-weighted mean lowered held-out
error at one epoch by 0.3–1.1% with a full search.

`recipe_` records the resolved cells, probes, rebuild cadence, closing mean and
the search that was used. `iteration_stats_` records, per pass, the share of
the full search's distances that were evaluated.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `n_clusters` | 8 | Number of centroids |
| `max_iter` | 3 | Epoch budget; fractional values are allowed. Below one epoch the batch and step sizes of a one-epoch fit are used |
| `batch_law` | `"auto"` | Throughput law up to two million rows, quality law above |
| `n_cells` | `n_clusters // 128`, rounded up to a multiple of 16, at least 16 | Coarse cells |
| `n_probe` | 16; 12 from 512 features; 24 above 16,384 centroids | Cells each row searches, at most 128 |
| `assignment` | `"auto"` | `"two_level"`, `"dense"`, or `"auto"` |
| `precision` | `"float32"` | `"float32"` or `"bfloat16"` assignment operands; state and sums stay float32 |
| `batch_size`, `alpha0`, `unused_threshold`, `init`, `random_state`, `assignment_batch_size`, `compute_labels`, `verbose` | as `EMeans` | |

`assignment="auto"` uses the two-level search when all of these hold, and the
full search otherwise. `recipe_["search_reason"]` states which applied.

- The probes are expected to cover at most half of the centroids.
- `n_clusters` is at most 65,536 and `n_probe` at most 128.
- The GPU is one on which whole fits were measured faster: M1 Ultra, M4 Max
  and M5 Ultra as of MLX 0.32.4. Other GPUs use the full search until they
  are measured; `assignment="two_level"` runs there on request.
- The cells stay even. After the first batch, after the first revival check
  and at the end of every pass, `auto` reads the share of centroids the
  searches actually compared. Above one half it returns to the full search
  for the rest of the fit and records the step in
  `recipe_["search_fallback_step"]`.

`labels_`, `inertia_`, `predict`, `transform` and `score` always use a full
search against the returned centers. Labels captured by
`train_device(capture_assignments=True)` come from the training search against
moving centroids.

## Differences from the reference implementation

- Batch sums are float32 atomics, as in `EMeans`, so repeated fits can differ
  in the last bits. The reference accumulates exact fixed-point sums.
- Assignment operands are float32 by default, with BF16 as an option. The
  reference compares in FP16.
- Per-centroid state stays in centroid order and only the assignment mirror is
  sorted by cell. Revival therefore fills dead slots in centroid order, where
  the reference fills them in cell order.
- `auto` can return to a full search; the reference always searches cells.
- Float32 assignment is centred on the data mean, as in `EMeans`.

## Metal implementation

One training batch is a single lazy MLX graph. Nothing is read back to the
host between batches.

| Step | Implementation |
|---|---|
| Sorted mirror | Centroids gathered in cell order, with their squared norms |
| Coarse scores | Binary32 GEMM of the batch against the cell centres |
| Probes | `select_topk` over the cell scores |
| Buckets | Counting sort of the (row, probe) pairs by cell: an integer-atomic histogram, a prefix sum, and an atomic-cursor placement. No sort and no host read |
| Scan | One threadgroup scores up to 64 rows that probe the same cell against that cell's contiguous centroid block, 64 centroids at a time, with SIMD-group matrix multiply-accumulate. The row minimum stays in registers across the block |
| Reduce | Minimum over a row's probes, then the split links, then the centroid id |
| Update | The e-means statistics, EMA and revival kernels, unchanged |

Batch size, cell count and all offsets reach the kernels as data, so a tail
batch or a re-filed table compiles nothing new. A scalar scan with one SIMD
group per (row, probe) pair is kept as the reference for tests and for MLX
releases that cannot build the tiled scan.

### Assignment speed

One batch of 32,768 rows against 16,384 centroids taken from a one-epoch
e-means fit, default cells and probes, median of synchronized evaluations.
"Useful" counts only the distances the search needs; the scan also computes
padding where a cell does not end on a tile boundary.

| GPU | Dataset | d | Precision | Full search | Two-level | Speed-up | Scan, useful TFLOP/s | Full search TFLOP/s |
|---|---|---:|---|---:|---:|---:|---:|---:|
| M4 Max | DEEP 1M | 96 | float32 | 9.93 ms | 2.60 ms | 3.82× | 7.20 | 10.38 |
| M4 Max | SPACEV 1M | 100 | float32 | 11.04 ms | 3.09 ms | 3.57× | 6.54 | 9.73 |
| M4 Max | ImageNet-CLIP | 512 | float32 | 45.65 ms | 7.65 ms | 5.97× | 9.04 | 12.04 |
| M4 Max | GooAQ | 768 | float32 | 69.73 ms | 13.33 ms | 5.23× | 9.16 | 11.83 |
| M4 Max | AG-News | 1,024 | float32 | 97.17 ms | 19.03 ms | 5.11× | 8.16 | 11.32 |
| M4 Max | ImageNet-CLIP | 512 | bfloat16 | 40.23 ms | 6.87 ms | 5.86× | 10.09 | 13.67 |
| M5 Ultra | SPACEV 1M | 100 | float32 | 7.51 ms | 1.62 ms | 4.63× | 11.86 | 14.30 |
| M5 Ultra | SPACEV 1M | 100 | bfloat16 | 4.62 ms | 1.59 ms | 2.90× | 12.07 | 23.26 |
| M5 Ultra | ImageNet-CLIP | 512 | float32 | 22.04 ms | 3.63 ms | 6.07× | 18.40 | 24.95 |
| M5 Ultra | ImageNet-CLIP | 512 | bfloat16 | 18.08 ms | 3.55 ms | 5.09× | 18.84 | 30.40 |

The M4 Max rows were measured before the default tile shape was unified; the
BF16 row used a 64 × 64 tile with depth 32. A sweep on M4 Max then found the
64 × 64 tile with depth 16 fastest or within noise for both precisions at 100
and 768 features. Narrower centroid tiles, and skipping half a tile on short
cell tails, were slower. On M5 Ultra four tile shapes were within about 5%.

Where a batch's time goes on M4 Max, net of one dispatch per stage:

| Stage | SPACEV 1M, d=100, 32,768 rows | GooAQ, d=768, 65,536 rows |
|---|---:|---:|
| Scan | 2.48 ms | 29.84 ms |
| Probe (coarse GEMM and selection) | 0.72 ms | 2.13 ms |
| Bucket | 0.20 ms | 0.35 ms |
| Statistics | 0.12 ms | 1.92 ms |
| Batch gather and centring | 0.10 ms | 2.01 ms |
| EMA update | 0.11 ms | 0.93 ms |
| Mirror, reduce, residuals, closing sums | 0.24 ms | 2.33 ms |
| Total | 3.97 ms | 39.52 ms |

The scan is 62–76% of a batch. The largest remaining items are the padding
inside the scan (about a fifth of its work) and, at low dimension, probing.

## Quality

Held-out MSE relative to `EMeans(recipe="faiss")` at the same epochs, seed and
starting centroids, with the reference scalar scan, on M4 Max. "Missed" is the
share of rows whose two-level label differs from the full-search label on the
final centroids.

| Dataset | d | k | Distances evaluated | Missed | Error at 1 / 2 / 3 epochs |
|---|---:|---:|---:|---:|---|
| SPACEV 1M | 100 | 16,384 | 13.8% | 0.6–0.8% | −0.58% / +0.11% / +0.35% |
| DEEP 1M | 96 | 16,384 | 13.2% | 0.4–0.5% | −0.21% / −0.26% / −0.20% |
| ImageNet-CLIP | 512 | 16,384 | 11.5% | 0.2% | −0.65% / −0.10% / +0.06% |
| GooAQ | 768 | 16,384 | 10.7% | 0.8–1.0% | −0.17% / −0.03% / −0.02% |
| AG-News | 1,024 | 16,384 | 10.4% | 0.3–0.4% | −0.63% / +0.02% / −0.15% |
| SimpleWiki | 3,072 | 8,192 | 21% | 0.1–0.2% | −1.11% / +0.13% / +0.01% |
| GloVe-200 (as stored, not normalized) | 200 | 16,384 | 98–100% | 0.9–1.1% | −0.38% / +0.07% / −0.01% |

With a full search (`assignment="dense"`), o-means matched e-means within
0.04% from two epochs up on all seven datasets, as it should: the two then
run the same algorithm.

Fit times, time to equal error and the comparison across seeds are in the
[benchmark report](o-means-benchmarks-2026-11-18.md).

## Limitations

- **Uneven cells remove the saving.** On unnormalized GloVe-200 one cell
  collects most of the centroids, every row probes it, and the search compares
  98–100% of the centroids. Forced two-level fits were about 20% slower than
  e-means there. `auto` detects this after the first batch and returns to the
  full search; the fit then takes about as long as e-means. The upstream
  package reports the same effect on synthetic high-dimensional mixtures.
- **The search is approximate**, with no worst-case bound on missed
  neighbours. The upstream package reports up to 1.7% higher error at equal
  epochs on dense synthetic Gaussian mixtures; a larger `n_probe` trades speed
  for accuracy.
- **Above 65,536 centroids** the two-level search is not available: its cell
  sort stays inside MLX's verified sort widths, and larger tables were not
  evaluated upstream either.
- **Hardware coverage** is M1 Ultra, M4 Max and M5 Ultra with MLX 0.32.4.
- No spherical mode, slow EMA or paper recipe; use `EMeans` for those. The
  whole input must fit on the GPU, as for `EMeans`.

## Reproducing measurements

```bash
.venv/bin/python -m benchmarks.e_means.two_level_bench --dataset spacev-1M \
    --precision float32 bfloat16
.venv/bin/python -m benchmarks.e_means.omeans_compare run --seeds 10 11 12
.venv/bin/python -m benchmarks.e_means.omeans_report
```

`two_level_bench` times one batch stage by stage and accepts `--data file.npy`
for machines without the dataset cache. `omeans_compare` runs each (dataset,
precision, seed) in a fresh process and records the hash of every measured
source file; run it from a checkout nobody else edits, and pass
`--faiss-source` when working in a git worktree.
