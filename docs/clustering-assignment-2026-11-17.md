# Exact Lloyd pruning and dense FP32 center assignment

On an M4 Max, assigning 10,000,000 DEEP vectors to 65,536 fixed centers took
14.77 s, down from 22.88 s on main. All 10 million labels and squared-distance
bits matched. Peak active MLX memory above the resident inputs fell from
1,100.5 MiB to 152.6 MiB.

The full DEEP Lloyd fit with two updates and a fresh final assignment fell
from **115.56 to 55.67 s** in one pair. Total peak active MLX memory fell
from **5.824 to 4.041 GiB**. Both builds returned the same 10 million labels,
512 held-out predictions and inertia of **3,441,619.0**. Of 6,291,456 center
coordinates, 910 differed, by at most 2.98e−8. Held-out MSE differed by
less than 7e−13. This run and the fixed-center pass used one trial per build.

## What changed

Lloyd can reuse a previous label even when the centers have moved. For a
point's proposed center, let `r` be its current distance and `s` the distance
from that center to its nearest competitor. Every competing center is at
least `s − r` from the point. When `s > 2r`, the squared-distance advantage
is at least `s(s − 2r)`. The implementation rounds the separation down,
inflates the point radius, and requires enough margin to cover rounding in
both competing FP32 scores. A duplicate center gives zero separation. Points
near a boundary go through exhaustive assignment with the original tie rule.

Calculating all center separations costs a scan over the centers for each
center. Before paying that cost, Lloyd tests 128 points against their proposed
centers and requires 96 certificates. The full check considers large M4 Max
fits with at least 4,096 centers, eight points per center, 64–1,024 features
and automatic assignment batching. It uses current centers on each eligible
pass, including after empty-center relocation and after restoring the final
centers to the original coordinates. `iteration_stats_[i]["pruned_rows"]`
counts original training points that bypassed a dense scan. Mixed tiles pad
their fallback rows to at least 128 to preserve the dot-product reduction
order, and up to one of four batch sizes per power of two. The assignment
kernels are compiled per row count, so this bounds the compilations.

For exhaustive assignment, the shared kernel for Lloyd and e-means can keep
FP32 dot products in registers. Each tile of 64 centers writes only its
winning score and index. The merge breaks ties by global center index, and
the final squared distance is computed directly from the selected center.
The measured implementation applies to M4 Max, safe-math-capable MLX,
64–512 features, at least 4,096 centers and at least 128 rows per tile.

## Matched workload results

The public fits below include preparation, training, final labels, inertia
and host output conversion. Prefix timings are medians of three paired
runs, each after one same-shape warmup. Every pair uses the same inputs and
explicit initial centers.

| Workload | Rows × features | Centers | Lloyd fit, main → new | Speedup |
|---|---:|---:|---:|---:|
| DEEP | 131,072 × 96 | 8,192 | 181.82 → 134.15 ms | 1.36× |
| SIFT | 131,072 × 128 | 8,192 | 215.89 → 167.25 ms | 1.29× |
| Separated clusters | 131,072 × 96 | 8,192 | 90.98 → 47.34 ms | 1.92× |
| Tied integer data | 65,536 × 64 | 4,096 | 44.40 → 31.64 ms | 1.40× |
| DINO | 32,768 × 1,024 | 4,096 | 164.55 → 167.82 ms | 0.98× |

On the separated workload Lloyd converged after two updates. The second pass
reused all 131,072 labels. DEEP and SIFT reused zero labels on the measured
passes, so their improvements came from dense assignment. Five-epoch
e-means fits on the same DEEP and SIFT prefixes improved from 237.96 to
171.02 ms and 281.42 to 213.24 ms, respectively.

An ablation compared Lloyd with and without pruning while using the same
streaming kernel. Pruning reduced the separated fit from **68.83 to
46.08 ms**. Where it could exclude no centers, the sample check added a
small cost. DEEP took 133.18 → 135.07 ms and the DINO prefix took
165.64 → 168.96 ms. All three ablations used three paired runs.

One-warmup standalone timings on tied integers varied from 6 to 16 ms and
favored main by median (7.70 → 11.45 ms). With two warmups, the new kernel
won all seven paired repeats (6.24 → 3.62 ms median). The fixed-center
labels and residual bits matched in both runs.

At DINO's native 65,536 centers, the benchmark used 524,288 real 1,024-feature
vectors. Two Lloyd updates and final assignment took 24.25 → 21.97 s in a
single pair. Both used exhaustive searches and agreed on every label, final
inertia and held-out MSE. Neither this run nor the smaller repeated DINO
control establishes a speedup for the previously published 10M-row DINO fit.

## Quality and reproduction

All 33 paired fixed-center measurements agreed on every label and squared
distance bit. All 29 Lloyd pairs, including the pruning ablation, agreed on
every training and held-out label, inertia, iteration count and stopping
reason. Their largest difference in a center coordinate was 3.58e−7. All
1,184 sampled comparisons against an exhaustive float64 reference returned
the expected center index.

GPU statistics accumulate FP32 sums atomically. On the five-epoch DEEP
e-means prefix, two main fits and all three candidate fits produced the
same labels. The third main fit
differed on 74,798 of 131,072 labels, and its held-out MSE was 0.429414
versus 0.427640 on the other trajectory.

The machine had 128 GiB of memory and ran MLX 0.32.4. The driver launches
separate isolated processes from frozen Git sources. It times a synchronized
kernel call or the public `fit` method. The full-scale assignment comparison
used main `6d871cc` and the dense implementation `fae0d6d`. The subsequent
fit comparison used `0149c80`, which adds Lloyd pruning. A later capability
check routes older MLX releases to GEMM when safe-math kernels are unavailable.
The full repository and e-means benchmark test suites passed with 4,499
passes and eight skips. Seven skips require pandas and one requires a C
compiler with Python framework headers.

```sh
python -m benchmarks.e_means.assignment_comparison \
  --baseline 6d871cc --candidate 0149c80 --dataset deep1b-10M \
  --rows 131072 --centers 8192 --iterations 5 --repeats 3 --warmups 1 \
  --output /tmp/deep-clustering-prefix.json
```

Real data use the repository's e-means benchmark cache and Faiss dataset
loader. The synthetic control can be reproduced with `--dataset separated`
and `--features 96`. [Driver](../benchmarks/e_means/assignment_comparison.py)
and [all raw comparisons](../benchmarks/results/clustering-assignment-2026-11-17).
