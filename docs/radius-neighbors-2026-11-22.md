# Radius neighbors: review of PR #3 and the exact, accelerated rewrite (2026-11-22)

PR tabulai/mlx-learn#3 added `NearestNeighbors.radius_neighbors`,
`radius_neighbors_graph`, `RadiusNeighborsClassifier` and
`RadiusNeighborsRegressor`. This note records how it compared with
scikit-learn, what the review found, and the rewrite that replaced its search
pipeline. The benchmark harness is `benchmarks/radius/`; the correctness
contract is in `src/mlxlearn/_kernels/radius_contract.py`.

## 1. PR #3 against scikit-learn

The PR did not outperform scikit-learn in general. On an M5 Ultra (MLX 0.32.4,
scikit-learn 1.9.1), over 229 configurations (core, estimator and real-data
grids):

- Layer 1 beat the best scikit-learn variant (default `algorithm="auto"`,
  `"brute"`, or `n_jobs=-1`) on 54 of 229, geometric mean 0.19x. Against
  scikit-learn's default it won 67 of 229, geometric mean 0.25x. The worst
  cells ran at 0.01x.
- It won sparse low-dimensional queries through its cKDTree route (1.3-3.5x)
  and sparse high-dimensional queries with large training sets (1.5-3x).
- It lost badly on dense outputs (0.02-0.2x), on d=8-15 data without the tree
  (its float64 host Gram route took 0.3-4 s where scikit-learn took 20-130 ms),
  on clustered or offset data, and on the estimators (0 of 44 estimator cells).

The stage profile showed why. `pack_radius` sorted every result pair with one
`np.lexsort` and then looped over rows in Python. Every Metal candidate was
refined in float64 on the host. The Metal route synchronized after every
256-row tile. The Gram screen ran on uncentred coordinates, so offset data
admitted nearly every pair.

## 2. Review findings

A six-dimension review, with an adversarial verifier per finding, confirmed 62
findings. None was a membership error at normal scales. The main ones:

- **Gram screen not translation-invariant.** Its slack scaled with squared
  norms from the origin. Offset data flooded the host with candidates (90x
  slower; out of memory at scale).
- **Output assembly.** A global `lexsort` over all pairs dominated dense
  queries on every route. The classifier and regressor looped over rows in
  Python.
- **Host Gram screening.** It was 3-26x slower than scikit-learn's brute force.
- **Patched routing.** The stock route was forced to `algorithm="brute"` where
  scikit-learn's default uses a KD-tree. The first call of every fit went to
  stock, so repeated calls were not bitwise reproducible.
- **Routing sample.** The 8 x 128 density sample could not see sparse
  densities and counted self-pairs.
- **Memory.** Dense results used 100-140 bytes per pair, and up to six copies
  of X stayed alive.
- **API parity.** Sparse queries, per-query radii and values beyond float32
  raised instead of falling back, the regressor dropped column targets, and the
  docs gave the wrong version.
- **Benchmarks.** The PR's grid in `benchmarks/run.py` used `radius=5.0` for
  every dimension (87% density at d=8, empty at d=64). It also broke
  `tests/unit/test_grid_refresh_benchmark.py`.

## 3. The rewrite

### Contract (every route)

- **Membership.** A training row `x` is a neighbor of `q` iff `S(q, x) <= R2`,
  where `S` is the exact real squared distance of the float32 coordinates and
  `R2 = fl64(radius**2)`. Engines classify their candidates against thresholds
  rounded outward. A pair inside the narrow float64 band is decided exactly
  with integers (`S * 2**298` against `floor(R2 * 2**298)`). Integer data
  whose float64 sums are exact skips the band. Membership is therefore
  identical on every route, batch size, tiling, thread count and machine.
- **Distances.** float64 `sqrt(min(s, R2))`, so a returned distance never
  exceeds the radius. `s` comes from one canonical function per installation:
  a strictly sequential sum per feature in the native extension, or blocked
  `einsum` without it.
- **Order.** Training indices ascend within each row; with
  `sort_results=True`, rows are ordered by `(distance, index)`.

### Engines

- **Native multithreaded KD-tree** (`_radius_tree.cpp`). Parallel build
  (26 ms at 1M rows against 267-489 ms for scikit-learn's), certified float32
  prune and bulk-include box tests, and rows sorted while still in cache.
- **Native brute force** for small calls, at microseconds per call.
- **Metal route.** A certified binary32 Gram screen on centred coordinates
  (`mlxlearn_device/_radius.py`), which reuses the k-NN index's Gram arrays.
  Tiles are batched and pipelined, overflowing rows are rescanned, and
  candidates go to the native finisher.
- **Native finisher** (`finish_pairs`). Canonical distances, exact
  classification and row sorting for candidates from any screen.
- **Fused votes and means** for the classifier and regressor; no ragged arrays
  are built.
- **NumPy fallbacks** for installs without the extension (cKDTree with sorted
  results, a centred float64 GEMM screen, NumPy refinement). They return the
  same neighbor sets.

### Routing (`radius_router.py`)

- `plan_radius` is a pure function of the call: the same shapes, data
  statistics, configuration and machine always give the same plan.
- Small calls go to native brute force. Up to d=8 the tree is used (up to
  d=16 without a GPU). Larger d uses the Metal screen.
- The density sample (at most 32 queries x 1024 training rows, with
  self-pairs excluded) estimates neighbors per query and the screen's candidate
  count. A non-selective screen, such as far-apart clusters, falls back to the
  tree (d <= 16) or the float64 GEMM screen.
- The patched layer hands a whole call to stock scikit-learn in two cases. The
  first is a capability gap: sparse queries, per-query, negative or NaN radii,
  values beyond float32, callable weights, or metric_params. These honor
  `fallback_policy`. The second is when the router predicts scikit-learn is
  faster: dense outputs on a small screen, heavy host brute force, or, without
  the native extension, dense outputs.
- The stock twin is configured with the fit-time parameters and the fitted
  feature names. It answers float32-rounded queries and returns scikit-learn's
  results unchanged.

## 4. Results

All measurements used an M5 Ultra (36 cores, GPU g17), MLX 0.32.4,
scikit-learn 1.9.1 and NumPy 2.5.4. The method was `benchmarks/radius/bench_radius.py`:
warm queries timed in rotated order after an idle drain (the gapped regime),
two cold rounds and four warm rounds per configuration, and correctness checked
against scikit-learn's float64 brute force in every cell. "Best scikit-learn"
is the fastest of `algorithm="auto"` (the default), `"brute"` and `n_jobs=-1`.

**Layer 1 (`mlxlearn.neighbors`) against scikit-learn.** The rewrite was
measured on 306 configurations: the core, estimator and real-data grids plus
an adversarial grid of offsets, hubs, label-sorted rows, self-queries, latency
calls and `n_jobs=-1` users. PR #3 was measured on the 229 configurations of
the first three grids.

| Region | Cells | Rewrite: wins vs best | Rewrite: geomean vs best (min) | Rewrite: vs default | PR #3: wins vs best | PR #3: geomean vs best |
|---|---|---|---|---|---|---|
| low-d (d <= 8) | 52 | 52/52 | 4.12x (1.25) | 7.74x | 13/48 | 0.24x |
| mid-d (9-15) | 4 | 4/4 | 1.45x (1.09) | 38.1x | 0/4 | 0.05x |
| high-d (d >= 16) | 115 | 90/115 | 1.77x (0.64) | 1.79x | 33/109 | 0.24x |
| real data (MNIST, CLIP-512, SPACEV) | 24 | 24/24 | 4.36x (1.24) | 4.39x | 8/24 | 0.35x |
| estimators, graphs, self-queries | 71 | 70/71 | 3.11x (0.90) | 4.07x | 0/44 | 0.07x |
| adversarial grid | 40 | 31/40 | 2.04x (0.62) | 3.13x | — | — |
| **all** | **306** | **271/306** | **2.54x (0.62)** | **3.34x** | **54/229** | **0.19x** |

- **`n_jobs=-1` users.** Against scikit-learn with `n_jobs=-1` the rewrite won
  all 98 cells where that setting applies (geometric mean 6.3x).
- **Same-session control.** On 135 core cells, an intermediate build of the
  rewrite ran 6.2x faster than PR #3 (geometric mean): 15x at low d, 21x at
  mid d and 3.6x at high d.
- **Remaining Layer 1 losses (0.62-0.98x).** All are on the Metal route, with
  dense or clustered outputs on small screens: at most 1000 queries against at
  most 1M rows, often with hundreds to thousands of neighbors per query. Host
  finishing and the GPU's cold start dominate there; see section 6.

**Patched layer (`patch_sklearn()`) against the scikit-learn configuration the
user chose.** In the final campaign, 287 of 306 cells were within 0.95x and 270
were at or above 1.0x (geometric mean 3.61x).

- **Planning trims.** Two later changes targeted the patched misses: a smaller
  density sample on small screens, and caching the stock twin's parameters.
  Profiled directly, the delegated path then ran within 0.35-1.1 ms of plain
  scikit-learn.
- **Re-measured misses.** Of the 18 missed cells re-measured, 11 moved within
  0.95x (geometric mean 0.97x, worst 0.88x).
- **What remains.**
  - Five are small calls delegated to stock that still pay about 1 ms of
    planning on 13-20 ms calls (0.88-0.94x).
  - A clustered d=16 call, also delegated to stock, stays at 0.93x.
  - One d=16 self-query on clustered data runs on Metal at 0.90x.

**Other hardware.** An 11-cell checkpoint plus three estimator cells ran on an
M1 Ultra (GPU g13, MLX 0.31.1, scikit-learn 1.8.0). The rewrite won all 14
cells, at 1.2-8.9x against the best scikit-learn variant.

Selected cells (M5 Ultra, warm, milliseconds; the rewrite from the final
campaign, PR #3 from its own campaign earlier the same day, scikit-learn's
fastest variant from the final campaign):

| Cell | Rewrite | PR #3 | scikit-learn (best) |
|---|---|---|---|
| uniform d=3, n=100k, 10k queries, ~100 neighbors each | 12.3 | 211.8 | 39.5 (`n_jobs=-1`) |
| uniform d=8, n=100k, 10k queries, ~10 neighbors each | 11.8 | 2582 | 63.9 (`n_jobs=-1`) |
| Gaussian d=12, n=100k, 10k queries, ~10 neighbors each | 44.7 | 2525 | 98.0 (brute) |
| Gaussian d=64, n=100k, 10k queries, ~1000 neighbors each (16M pairs) | 149.6 | 7435 | 230.1 (auto) |
| Gaussian d=128, n=1M, 10k queries, ~10 neighbors each | 275.7 | 1368 | 2463 (auto) |
| MNIST (d=784), n=50k, 10k queries, ~10 neighbors each | 110.0 | 4333 | 685.8 (auto) |
| distance-weighted regressor, d=3 blobs, 10k queries | 12.7 | 582.2 | 129.6 (`n_jobs=-1`) |

## 5. Verification

- **Exact oracle.** `tests/_radius_oracle.py` decides membership with integer
  arithmetic. It has 61 adversarial cases: lattices with exact ties,
  permuted-coordinate ties that defeat naive float64 sums, offsets up to 1e12,
  subnormal and near-FLT_MAX coordinates, R2 from 0 to inf, hubs, and
  label-sorted rows.
- **Exactness suite.** `tests/parity/test_radius_exactness.py` forces every
  route against the oracle, with and without sorting and for X=None.
- **Unit tests.** Native engine tests cover bitwise canonical distances,
  thread invariance, error propagation, GIL release and result lifetime. They
  ran clean under ASan, UBSan and TSan builds.
- **Second adversarial review.** It confirmed 40 findings in the rewrite, and
  all are fixed with regression tests in `tests/parity/test_radius_regressions.py`.
  Fuzzing found no membership, order or distance violation on any route.
- **Full suite.** 5617 passed, 9 skipped.

## 6. Limitations and follow-ups

- **Cold regime.** After a 0.3 s idle, host stages on the M4 Max run 3-9x
  slower (cores clock down and caches go cold). Small batches with dense
  outputs on the Metal route then lose to scikit-learn's batched float64 GEMM.
  The patched layer routes those calls to stock. Moving finishing onto the GPU
  (ordered emission plus double-single refinement; milestone M3 of the design
  plan) is the fix.
- **Router constants.** They were set from M5 Ultra, M4 Max and M1 Ultra
  measurements. A full forced-route calibration (milestone M4) would replace
  the hand rules with per-generation tables.
- **Object arrays.** Ragged results still cost about 1 us per row per array to
  build; scikit-learn pays the same. Graphs and predictions skip them.

## 7. M4 Max follow-up on November 23

Automatic fits with up to eight features keep the training rows on the CPU for
radius searches. The tree preserves index order when splitting leaves, and
dense results use a bitmap to emit indices in order. For wide brute-force
searches, a lower bound from the first 16 features can reject distant rows
before the remaining features are read.

For moderately dense Metal queries, the GPU emits one candidate bit per
training row. The CPU walks those bits in index order and applies the existing
exact membership test. Large batches below the smallest positive binary32
coordinate separation use a stable index of equal rows. Self queries remove
only the query's own index, including when other training rows are identical.

Automatic delegation to scikit-learn now requires evidence that the squared
distances and cutoff comparison are exact. One translated dataset produced
4,096 neighbors per query in the old automatic route, and a zero-centered
dataset produced 1,026. Both have exactly 1,024 neighbors at distance 0.5.
Explicit stock requests and capability fallbacks still use scikit-learn.

The follow-up benchmark compared the patched public estimator with commit
`77d20f8` and stock scikit-learn 1.9.1 on 22 workloads. Cold time includes
fit and the first query in each of five fresh processes, after imports and data
generation. Warm time is the median of 27 queries across three primed processes.
Every query returned the same neighbor set across all three implementations.
The float64 cases used the same coordinate values as their float32 counterparts.

| Workload | Previous cold (ms) | New cold (ms) | Stock cold (ms) | Previous warm (ms) | New warm (ms) | Stock warm (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128,000 × 4, 256 sparse queries | 27.48 | 9.59 | 28.66 | 0.29 | 0.27 | 2.60 |
| 32,768 × 4, 256 dense queries with distances | 27.31 | 10.25 | 35.11 | 3.45 | 2.32 | 27.53 |
| 32,000 × 32, 256 dense queries with sorted distances | 49.63 | 54.76 | 25.88 | 8.72 | 4.86 | 8.29 |
| 8,192 × 16, zero-radius self query | 62.79 | 5.72 | 39.77 | 4.42 | 2.00 | 25.48 |

With equal weight per workload, geometric-mean speedups over the previous
commit were 1.62× cold and 1.55× warm. Against stock they were 0.94× cold
and 4.06× warm. Stock won 14 of 22 cold cases, including every high-dimensional
Metal search. The new warm median was lower than stock in all 22 cases.
The very dense integer sort delegated to stock and finished
within 2% of it.
