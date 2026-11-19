# O-means on Apple Silicon

`mlxlearn.cluster.OMeans` is [e-means](e-means.md) with a two-level
nearest-centroid search. E-means compares every training row with every
centroid; o-means compares it with the centroids of a few nearby coarse cells
only: 10–25% of them at 8,192 to 16,384 centroids on the embedding datasets
measured here, and 6% at 65,536. It follows the
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
GPU, with the data resident there. With few centroids a row's probes cover
more than half of them and little is saved; `OMeans` then trains with the
full search, exactly as e-means does. With the default cells and probes that
is below 4,224 centroids, or below 2,176 from 512 features up. It requires a
Metal GPU and never falls back to the CPU. It is an explicit estimator and is
not part of `patch_sklearn()`.

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
mean. With a full search, the recency-weighted mean lowered held-out error at
one epoch by 0.24–1.20% across 18 datasets.

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

- The probes are expected to cover at most half of the centroids. With the
  default cells and probes that is from 4,224 centroids, or from 2,176 from
  512 features up.
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
| M4 Max | DEEP 1M | 96 | float32 | 10.42 ms | 2.89 ms | 3.61× | 6.88 | 9.89 |
| M4 Max | DEEP 1M | 96 | bfloat16 | 8.33 ms | 2.52 ms | 3.30× | 7.50 | 12.38 |
| M4 Max | SPACEV 1M | 100 | float32 | 11.82 ms | 3.26 ms | 3.63× | 6.29 | 9.08 |
| M4 Max | SPACEV 1M | 100 | bfloat16 | 11.44 ms | 3.03 ms | 3.78× | 6.85 | 9.39 |
| M4 Max | ImageNet-CLIP | 512 | float32 | 46.67 ms | 7.59 ms | 6.15× | 9.16 | 11.78 |
| M4 Max | ImageNet-CLIP | 512 | bfloat16 | 40.75 ms | 6.92 ms | 5.89× | 10.04 | 13.49 |
| M4 Max | GooAQ | 768 | float32 | 69.51 ms | 12.55 ms | 5.54× | 8.90 | 11.86 |
| M4 Max | GooAQ | 768 | bfloat16 | 59.15 ms | 10.86 ms | 5.45× | 10.31 | 13.94 |
| M4 Max | AG-News | 1,024 | float32 | 90.13 ms | 17.37 ms | 5.19× | 8.90 | 12.20 |
| M4 Max | AG-News | 1,024 | bfloat16 | 78.12 ms | 14.40 ms | 5.42× | 10.78 | 14.08 |
| M1 Ultra | SPACEV 1M | 100 | float32 | 17.24 ms | 4.35 ms | 3.96× | 4.48 | 6.23 |
| M1 Ultra | SPACEV 1M | 100 | bfloat16 | 12.76 ms | 4.10 ms | 3.11× | 4.70 | 8.42 |
| M1 Ultra | ImageNet-CLIP | 512 | float32 | 42.34 ms | 9.05 ms | 4.68× | 7.33 | 12.98 |
| M1 Ultra | ImageNet-CLIP | 512 | bfloat16 | 36.80 ms | 8.42 ms | 4.37× | 8.04 | 14.94 |
| M5 Ultra | SPACEV 1M | 100 | float32 | 7.51 ms | 1.62 ms | 4.63× | 11.86 | 14.30 |
| M5 Ultra | SPACEV 1M | 100 | bfloat16 | 4.62 ms | 1.59 ms | 2.90× | 12.07 | 23.26 |
| M5 Ultra | ImageNet-CLIP | 512 | float32 | 22.04 ms | 3.63 ms | 6.07× | 18.40 | 24.95 |
| M5 Ultra | ImageNet-CLIP | 512 | bfloat16 | 18.08 ms | 3.55 ms | 5.09× | 18.84 | 30.40 |

All rows use the default tile: 64 rows by 64 centroids, depth 16, for both
precisions. On useful distances the scan reaches 61–78% of the full search's
throughput on M4 Max, 54–72% on M1 Ultra and 52–83% on M5 Ultra. A sweep on
M4 Max found that tile fastest or within noise at 100 and 768 features;
narrower centroid tiles, and skipping half a tile on short cell tails, were
slower. On M5 Ultra four tile shapes were within about 5%. The M4 Max is a
laptop: its rows were taken after a 20-second pause at nominal thermal
pressure, because sustained load slows its full search (see the
[benchmark report](o-means-benchmarks-2026-11-18.md)).

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

## Fit speed and quality

Three seeds on an M5 Ultra, 18 datasets, against `EMeans(recipe="faiss")`
with the same batch law, seed and starting centroids, both from resident
inputs. "Same epochs" divides e-means' fit time by o-means' at one and two
epochs. "To e-means' error" divides e-means' time at one, two and three
epochs by the time o-means needs to reach the same held-out error. Both are
geometric means, with the range over datasets and budgets in parentheses.

| Datasets | Centroids compared | Same epochs, float32 | Same epochs, BF16 | To e-means' error, float32 | To e-means' error, BF16 |
|---|---:|---:|---:|---:|---:|
| 15 embedding sets, 0.18–1.5M rows, 8,192–16,384 centroids | 10–25% | 4.2× (1.6–5.6) | 3.3× (1.5–5.2) | 4.2× (1.8–6.1) | 3.5× (1.7–5.2) |
| SPACEV and DEEP, 10M rows, 65,536 centroids | 6% | 9.6× (9.0–10.3) | 5.9× (5.2–6.6) | 8.8× (7.7–10.4) | 5.9× (5.0–6.6) |
| All 18, GloVe included | 6–100% | 4.44× | 3.42× | 4.44× | 3.55× |

Held-out error at the same epochs was −1.14% to +0.10% of e-means' at one
epoch, where the closing mean helps, and −0.17% to +0.36% at two. O-means
needs more epochs than e-means to match its two- and three-epoch error, which
is why the speed-up at equal error falls from 4.7× at one epoch to 4.1× at
three in float32. With a full search (`assignment="dense"`), o-means matched
e-means within 0.05% at two epochs and ran at its speed (0.96–1.03×), as it
should: the two then run the same algorithm.

On the final centroids of a one-epoch fit, 0.25–1.3% of rows receive a
different centroid from the two-level search than from a full search
(measured on DEEP 1M, SPACEV 1M, ImageNet-CLIP, GooAQ and AG-News).

On a MacBook Pro M4 Max the same campaign, timed with the configurations
taking turns because that laptop throttles, gave 4.13× (float32) and 3.96×
(BF16) to e-means' error over the 18 datasets, 4.20× and 3.90× at the same
epochs, and 7.3× and 7.0× to e-means' error at ten million rows and 65,536
centroids: [M4 Max report](o-means-benchmarks-m4-max-2026-11-19.md). On M1
Ultra, same epochs at 16,384 centroids: 2.8–4.1× on SPACEV 1M and
ImageNet-CLIP. Per-dataset tables, the comparison at equal time, the sweep
over the centroid count and the method are in the
[benchmark report](o-means-benchmarks-2026-11-18.md).

## Limitations

- **Uneven cells reduce or remove the saving.** On unnormalized GloVe-200
  most cells hold one to four centroids and a few hold the rest. With two of
  three seeds the search compared 42–45% of the centroids and fits were
  1.7–2.6× faster than e-means on M5 Ultra. With the third it compared
  99–100%, and a forced two-level fit ran at 0.82–1.27× of e-means' speed.
  `auto` detects that case after the first batch and returns to the full
  search; the fit then takes as long as e-means. The upstream package reports
  the same effect on synthetic high-dimensional mixtures.
- **The search is approximate**, with no worst-case bound on missed
  neighbours. The upstream package reports up to 1.7% higher error at equal
  epochs on dense synthetic Gaussian mixtures; a larger `n_probe` trades speed
  for accuracy.
- **Above 65,536 centroids** the two-level search is not available: its cell
  sort stays inside MLX's verified sort widths, and larger tables were not
  evaluated upstream either.
- **Hardware coverage** is M1 Ultra, M4 Max and M5 Ultra with MLX 0.32.4.
- **Laptops throttle.** Under sustained load a MacBook Pro M4 Max slowed the
  full search far more than the two-level search, so o-means' advantage there
  grows with run time. Compare the two only in matched thermal states; the
  benchmark tools record the thermal pressure level for that reason.
- No spherical mode, slow EMA or paper recipe; use `EMeans` for those. The
  whole input must fit on the GPU, as for `EMeans`.

## Reproducing measurements

```bash
.venv/bin/python -m benchmarks.e_means.two_level_bench --dataset spacev-1M \
    --precision float32 bfloat16
.venv/bin/python -m benchmarks.e_means.omeans_compare run --seeds 10 11 12
.venv/bin/python -m benchmarks.e_means.omeans_report \
    --append benchmarks/e_means/omeans_report_notes.md
.venv/bin/python -m benchmarks.e_means.omeans_centroid_sweep run \
    --datasets spacev-1M imagenet-clip-512-normalized --output sweep.jsonl
```

- `two_level_bench` times one batch stage by stage. It accepts
  `--data file.npy` for machines without the dataset cache, and `--rest` for
  a pause before timing on a laptop.
- `omeans_compare` runs each (dataset, precision, seed) in a fresh process
  and records the hash of every measured source file and the thermal pressure
  level of every timed call. Run it from a checkout nobody else edits, on a
  machine whose fit time stays flat under continuous load, and pass
  `--faiss-source` when working in a git worktree.
- `omeans_interleaved` runs the same campaign with the timed calls of all
  configurations taken in turn, for a laptop whose GPU throttles: it keeps
  the configurations in one machine state where `omeans_compare` would time
  them minutes apart.
- `omeans_report` refuses mixed, incomplete or throttled campaigns; records
  of `omeans_interleaved` may have seen thermal pressure, which the report
  then states.
- `omeans_centroid_sweep` times e-means, forced two-level and `auto` in turn
  over the centroid count; `omeans_device_fits` does the same from `.npy`
  files on a machine without the dataset cache.
