# o-means against e-means on Apple silicon

Generated from 108 records over 18 datasets, seeds [10, 11, 12], on Apple M5 Ultra (MLX 0.32.4, Python 3.14.2). Both methods use the Faiss e-means recipe, the same batch law, seed and starting centroids, with the training matrix resident on the GPU. Times are medians of synchronized `train_device` calls after a full warm-up fit, then the median over seeds; errors are held-out MSE with exact float32 assignment and a float64 residual, averaged over seeds. Every timed call started and ended at nominal macOS thermal pressure.

## Summary

Geometric means over datasets and e-means budgets. "Same epochs" divides e-means' fit time by o-means' at the budgets both were measured at.

| Precision | Datasets | Time to e-means' error, interpolated | First measured budget | Slowest | Fastest | Same epochs | Distances at same epochs, median |
|---|---:|---:|---:|---|---|---:|---:|
| bfloat16 | 18 | 3.55× | 3.21× | 1.62× (vibe-glove, 3 epochs) | 6.63× (deep1b-10M, 2 epochs) | 3.42× | 0.114 |
| float32 | 18 | 4.44× | 3.96× | 1.82× (vibe-celeba, 3 epochs) | 10.35× (deep1b-10M, 1 epochs) | 4.44× | 0.114 |

## bfloat16

### Time to e-means' error

Speed-up is e-means' fit time divided by the time o-means needs to reach the same held-out error. `≥` marks a target the smallest o-means budget already met. The measured column uses the first o-means budget that reaches the target, without interpolation. Distances are those o-means evaluated to reach the target, as a fraction of e-means' n × k × T.

| Dataset | n | d | k | Speed-up at 1 | Speed-up at 2 | Speed-up at 3 | Measured at 1 | Measured at 2 | Measured at 3 | Distances at 1 / 2 / 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deep1b-10M | 10,000,000 | 96 | 65,536 | 6.17 | 6.63 | 5.85 | 5.21 | 6.45 | 5.37 | 0.05 / 0.04 / 0.06 |
| deep1b-1M | 1,000,000 | 96 | 16,384 | 2.40 | 2.33 | 2.59 | 2.31 | 2.05 | 2.40 | 0.12 / 0.14 / 0.13 |
| dino10b-1M-div4096 | 1,000,000 | 1024 | 16,384 | 5.21 | 4.89 | 5.05 | 4.68 | 4.24 | 4.93 | 0.10 / 0.11 / 0.11 |
| fb-ssnpp-1M | 1,000,000 | 256 | 16,384 | 3.39 | 4.51 | 2.78 | 3.04 | 3.72 | 2.54 | 0.14 / 0.11 / 0.19 |
| imagenet-clip-512-normalized | 1,000,000 | 512 | 16,384 | 4.86 | 4.69 | 4.63 | 4.24 | 4.52 | 4.54 | 0.10 / 0.11 / 0.12 |
| sift1b-1M-div512 | 1,000,000 | 128 | 16,384 | 2.88 | 2.81 | 2.90 | 2.70 | 2.80 | 2.72 | 0.12 / 0.14 / 0.14 |
| spacev-10M | 10,000,000 | 100 | 65,536 | 6.57 | 5.46 | 4.95 | 5.77 | 5.12 | 4.29 | 0.05 / 0.07 / 0.08 |
| spacev-1M | 1,000,000 | 100 | 16,384 | 2.91 | 2.54 | 2.61 | 2.66 | 2.31 | 2.26 | 0.12 / 0.15 / 0.15 |
| vibe-agnews | 749,088 | 1024 | 16,384 | 4.26 | 4.55 | 4.45 | 3.77 | 3.86 | 3.94 | 0.09 / 0.11 / 0.13 |
| vibe-arxiv | 1,324,643 | 768 | 16,384 | 5.17 | 4.84 | 4.33 | 4.50 | 4.11 | 4.08 | 0.09 / 0.11 / 0.13 |
| vibe-ccnews | 475,295 | 768 | 8,192 | 2.26 | 2.48 | 2.34 | 1.94 | 2.44 | 2.05 | 0.17 / 0.19 / 0.23 |
| vibe-celeba | 181,573 | 2048 | 8,192 | 1.73 | 2.36 | 3.24 | 1.47 | 2.24 | 3.20 | 0.20 / 0.16 / 0.12 |
| vibe-codesearchnet | 1,354,065 | 768 | 16,384 | 4.48 | 3.92 | 3.28 | 4.44 | 3.09 | 3.11 | 0.11 / 0.14 / 0.18 |
| vibe-glove | 1,172,514 | 200 | 16,384 | 1.79 | 2.10 | 1.62 | 1.66 | 2.03 | 1.35 | 0.57 / 0.48 / 0.68 |
| vibe-gooaq | 1,455,024 | 768 | 16,384 | 4.92 | 4.65 | 4.02 | 4.58 | 4.13 | 4.02 | 0.10 / 0.12 / 0.14 |
| vibe-landmark-dino | 740,740 | 768 | 16,384 | 4.37 | 4.10 | 4.50 | 3.78 | 3.82 | 3.92 | 0.09 / 0.12 / 0.12 |
| vibe-landmark-nomic | 740,729 | 768 | 16,384 | 4.10 | 4.65 | 3.44 | 3.63 | 4.55 | 3.04 | 0.10 / 0.11 / 0.17 |
| vibe-simplewiki | 240,372 | 3072 | 8,192 | 2.37 | 2.10 | 2.24 | 1.94 | 2.01 | 1.79 | 0.18 / 0.23 / 0.24 |
| **Geometric mean** | | | | 3.60 | 3.63 | 3.42 | 3.21 | 3.32 | 3.11 | 0.13 |

Geometric mean over all datasets and targets: 3.55× interpolated, 3.21× measured-only.

### Same epoch budget

Only the budgets both methods were measured at. o-means and e-means at the same number of epochs: how much faster o-means' fit is, its held-out error relative to e-means' (negative is better), the share of distances it evaluated, and the error of o-means with a full search, which isolates the closing mean.

| Dataset | Faster at 1 | Faster at 2 | Error at 1 | Error at 2 | Distances | Full-search error at 1 / 2 |
|---|---:|---:|---:|---:|---:|---:|
| deep1b-10M | 5.21× | 5.94× | -0.27% | -0.17% | 0.058 | -0.24% / +0.00% |
| deep1b-1M | 2.31× | 2.48× | -0.25% | +0.02% | 0.132 | -0.42% / +0.00% |
| dino10b-1M-div4096 | 4.68× | 5.01× | -0.39% | +0.01% | 0.107 | -0.45% / +0.00% |
| fb-ssnpp-1M | 3.04× | 3.09× | -0.29% | +0.06% | 0.159 | -0.40% / +0.00% |
| imagenet-clip-512-normalized | 4.24× | 4.52× | -0.54% | -0.06% | 0.114 | -0.62% / +0.00% |
| sift1b-1M-div512 | 2.70× | 2.80× | -0.26% | -0.01% | 0.136 | -0.31% / +0.00% |
| spacev-10M | 5.77× | 6.57× | -0.37% | +0.19% | 0.059 | -0.48% / +0.00% |
| spacev-1M | 2.66× | 2.78× | -0.47% | +0.13% | 0.138 | -0.54% / +0.00% |
| vibe-agnews | 3.77× | 4.84× | -0.65% | +0.10% | 0.106 | -0.87% / +0.00% |
| vibe-arxiv | 4.50× | 5.11× | -0.52% | +0.06% | 0.107 | -0.54% / +0.00% |
| vibe-ccnews | 1.94× | 2.44× | -0.77% | -0.07% | 0.198 | -0.80% / -0.00% |
| vibe-celeba | 1.47× | 1.97× | -0.70% | -0.02% | 0.238 | -0.86% / +0.00% |
| vibe-codesearchnet | 4.44× | 5.03× | -0.04% | +0.36% | 0.111 | -0.59% / +0.00% |
| vibe-glove | 1.66× | 1.76× | -0.37% | +0.00% | 0.623 | -0.39% / +0.01% |
| vibe-gooaq | 4.58× | 5.19× | -0.27% | +0.09% | 0.106 | -0.50% / +0.00% |
| vibe-landmark-dino | 3.78× | 4.81× | -0.59% | +0.12% | 0.106 | -0.68% / -0.00% |
| vibe-landmark-nomic | 3.63× | 4.55× | -0.65% | -0.06% | 0.115 | -0.71% / +0.00% |
| vibe-simplewiki | 1.94× | 2.11× | -1.13% | +0.04% | 0.218 | -1.20% / +0.00% |
| **Geometric mean** | 3.21× | 3.65× | | | | |

### Error at e-means' time

o-means' held-out error relative to e-means', when o-means is given the time e-means needed. Negative is better.

| Dataset | At e-means' 1-epoch time | At e-means' 2-epoch time | At e-means' 3-epoch time |
|---|---:|---:|---:|
| deep1b-10M | -1.18% | -0.53% | -0.19% |
| deep1b-1M | -1.27% | -0.46% | -0.39% |
| dino10b-1M-div4096 | -1.75% | -0.51% | -0.28% |
| fb-ssnpp-1M | -0.76% | -0.39% | -0.18% |
| imagenet-clip-512-normalized | -2.21% | -0.68% | -0.36% |
| sift1b-1M-div512 | -1.44% | -0.41% | -0.03% |
| spacev-10M | -2.40% | -0.76% | -0.34% |
| spacev-1M | -2.43% | -0.94% | -0.49% |
| vibe-agnews | -3.40% | -1.02% | -0.44% |
| vibe-arxiv | -2.42% | -0.72% | -0.36% |
| vibe-ccnews | -2.84% | -1.00% | -0.53% |
| vibe-celeba | -1.30% | -0.34% | -0.27% |
| vibe-codesearchnet | -2.56% | -0.86% | -0.31% |
| vibe-glove | -1.68% | -0.72% | -0.25% |
| vibe-gooaq | -2.51% | -0.90% | -0.35% |
| vibe-landmark-dino | -1.96% | -0.47% | -0.20% |
| vibe-landmark-nomic | -2.67% | -0.99% | -0.31% |
| vibe-simplewiki | -3.10% | -0.98% | -0.65% |

## float32

### Time to e-means' error

Speed-up is e-means' fit time divided by the time o-means needs to reach the same held-out error. `≥` marks a target the smallest o-means budget already met. The measured column uses the first o-means budget that reaches the target, without interpolation. Distances are those o-means evaluated to reach the target, as a fraction of e-means' n × k × T.

| Dataset | n | d | k | Speed-up at 1 | Speed-up at 2 | Speed-up at 3 | Measured at 1 | Measured at 2 | Measured at 3 | Distances at 1 / 2 / 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deep1b-10M | 10,000,000 | 96 | 65,536 | 10.35 | 7.80 | 7.70 | 9.04 | 6.22 | 6.65 | 0.05 / 0.07 / 0.08 |
| deep1b-1M | 1,000,000 | 96 | 16,384 | 5.09 | 3.72 | 4.76 | 4.72 | 3.08 | 4.42 | 0.12 / 0.17 / 0.13 |
| dino10b-1M-div4096 | 1,000,000 | 1024 | 16,384 | 5.68 | 5.42 | 5.09 | 5.10 | 5.41 | 4.94 | 0.09 / 0.11 / 0.11 |
| fb-ssnpp-1M | 1,000,000 | 256 | 16,384 | 4.95 | 6.07 | 3.79 | 4.39 | 5.08 | 3.44 | 0.14 / 0.11 / 0.19 |
| imagenet-clip-512-normalized | 1,000,000 | 512 | 16,384 | 5.75 | 5.34 | 5.47 | 5.07 | 4.52 | 5.24 | 0.10 / 0.11 / 0.12 |
| sift1b-1M-div512 | 1,000,000 | 128 | 16,384 | 5.34 | 4.93 | 5.09 | 4.87 | 4.82 | 4.55 | 0.12 / 0.13 / 0.13 |
| spacev-10M | 10,000,000 | 100 | 65,536 | 9.74 | 9.22 | 8.55 | 9.07 | 8.01 | 7.75 | 0.06 / 0.06 / 0.07 |
| spacev-1M | 1,000,000 | 100 | 16,384 | 5.05 | 4.37 | 4.14 | 4.69 | 3.90 | 3.70 | 0.12 / 0.15 / 0.16 |
| vibe-agnews | 749,088 | 1024 | 16,384 | 4.73 | 5.15 | 4.86 | 4.18 | 4.27 | 4.00 | 0.09 / 0.11 / 0.12 |
| vibe-arxiv | 1,324,643 | 768 | 16,384 | 5.87 | 5.20 | 4.09 | 5.09 | 4.40 | 3.35 | 0.09 / 0.11 / 0.15 |
| vibe-ccnews | 475,295 | 768 | 8,192 | 2.69 | 2.90 | 2.59 | 2.26 | 2.85 | 2.40 | 0.16 / 0.19 / 0.24 |
| vibe-celeba | 181,573 | 2048 | 8,192 | 1.85 | 2.45 | 1.82 | 1.55 | 2.39 | 1.51 | 0.20 / 0.18 / 0.32 |
| vibe-codesearchnet | 1,354,065 | 768 | 16,384 | 4.93 | 4.31 | 3.68 | 3.21 | 4.31 | 3.23 | 0.12 / 0.14 / 0.17 |
| vibe-glove | 1,172,514 | 200 | 16,384 | 2.63 | 2.91 | 2.30 | 2.44 | 2.81 | 2.23 | 0.58 / 0.48 / 0.64 |
| vibe-gooaq | 1,455,024 | 768 | 16,384 | 5.63 | 4.97 | 3.50 | 5.24 | 4.44 | 3.34 | 0.10 / 0.12 / 0.17 |
| vibe-landmark-dino | 740,740 | 768 | 16,384 | 5.01 | 5.44 | 5.11 | 4.38 | 5.40 | 4.21 | 0.09 / 0.11 / 0.11 |
| vibe-landmark-nomic | 740,729 | 768 | 16,384 | 4.75 | 5.11 | 4.42 | 4.20 | 4.13 | 4.07 | 0.10 / 0.12 / 0.14 |
| vibe-simplewiki | 240,372 | 3072 | 8,192 | 2.40 | 2.17 | 2.25 | 1.97 | 2.17 | 2.17 | 0.18 / 0.22 / 0.23 |
| **Geometric mean** | | | | 4.71 | 4.56 | 4.08 | 4.11 | 4.12 | 3.67 | 0.13 |

Geometric mean over all datasets and targets: 4.44× interpolated, 3.96× measured-only.

### Same epoch budget

Only the budgets both methods were measured at. o-means and e-means at the same number of epochs: how much faster o-means' fit is, its held-out error relative to e-means' (negative is better), the share of distances it evaluated, and the error of o-means with a full search, which isolates the closing mean.

| Dataset | Faster at 1 | Faster at 2 | Error at 1 | Error at 2 | Distances | Full-search error at 1 / 2 |
|---|---:|---:|---:|---:|---:|---:|
| deep1b-10M | 9.04× | 10.19× | -0.29% | +0.04% | 0.058 | -0.25% / -0.04% |
| deep1b-1M | 4.72× | 4.73× | -0.30% | +0.02% | 0.132 | -0.38% / -0.02% |
| dino10b-1M-div4096 | 5.10× | 5.41× | -0.40% | -0.00% | 0.107 | -0.45% / +0.00% |
| fb-ssnpp-1M | 4.39× | 4.21× | -0.32% | +0.05% | 0.159 | -0.42% / +0.00% |
| imagenet-clip-512-normalized | 5.07× | 5.35× | -0.56% | +0.00% | 0.114 | -0.60% / -0.04% |
| sift1b-1M-div512 | 4.87× | 4.82× | -0.34% | -0.03% | 0.136 | -0.34% / +0.00% |
| spacev-10M | 9.07× | 10.26× | -0.23% | +0.09% | 0.059 | -0.49% / +0.00% |
| spacev-1M | 4.69× | 4.75× | -0.40% | +0.07% | 0.138 | -0.57% / +0.00% |
| vibe-agnews | 4.18× | 5.34× | -0.67% | +0.06% | 0.106 | -0.82% / -0.02% |
| vibe-arxiv | 5.09× | 5.52× | -0.52% | +0.05% | 0.107 | -0.56% / -0.00% |
| vibe-ccnews | 2.26× | 2.85× | -0.75% | -0.06% | 0.198 | -0.85% / -0.05% |
| vibe-celeba | 1.55× | 2.14× | -0.73% | +0.03% | 0.238 | -0.81% / -0.00% |
| vibe-codesearchnet | 5.14× | 5.48× | +0.10% | +0.31% | 0.111 | -0.60% / +0.00% |
| vibe-glove | 2.44× | 2.43× | -0.34% | +0.02% | 0.623 | -0.35% / -0.02% |
| vibe-gooaq | 5.24× | 5.56× | -0.28% | +0.14% | 0.106 | -0.51% / +0.01% |
| vibe-landmark-dino | 4.38× | 5.40× | -0.54% | -0.01% | 0.106 | -0.65% / -0.01% |
| vibe-landmark-nomic | 4.20× | 5.17× | -0.61% | +0.02% | 0.115 | -0.72% / +0.00% |
| vibe-simplewiki | 1.97× | 2.17× | -1.14% | -0.00% | 0.218 | -1.20% / +0.02% |
| **Geometric mean** | 4.22× | 4.67× | | | | |

### Error at e-means' time

o-means' held-out error relative to e-means', when o-means is given the time e-means needed. Negative is better.

| Dataset | At e-means' 1-epoch time | At e-means' 2-epoch time | At e-means' 3-epoch time |
|---|---:|---:|---:|
| deep1b-10M | -1.30% | -0.38% | -0.18% |
| deep1b-1M | -1.68% | -0.32% | -0.15% |
| dino10b-1M-div4096 | -1.74% | -0.51% | -0.27% |
| fb-ssnpp-1M | -0.91% | -0.36% | -0.16% |
| imagenet-clip-512-normalized | -2.18% | -0.58% | -0.29% |
| sift1b-1M-div512 | -1.66% | -0.65% | -0.37% |
| spacev-10M | -2.35% | -0.82% | -0.42% |
| spacev-1M | -2.94% | -1.08% | -0.52% |
| vibe-agnews | -3.43% | -1.11% | -0.53% |
| vibe-arxiv | -2.40% | -0.72% | -0.35% |
| vibe-ccnews | -3.02% | -1.09% | -0.51% |
| vibe-celeba | -1.31% | -0.32% | -0.20% |
| vibe-codesearchnet | -2.50% | -0.95% | -0.45% |
| vibe-glove | -1.98% | -0.83% | -0.43% |
| vibe-gooaq | -2.50% | -0.81% | -0.25% |
| vibe-landmark-dino | -2.00% | -0.69% | -0.32% |
| vibe-landmark-nomic | -2.60% | -0.94% | -0.40% |
| vibe-simplewiki | -3.05% | -0.98% | -0.68% |

## Notes

### Which machine, and why

The campaign runs on a Mac Studio M5 Ultra because its fit time stays flat
under continuous load. Four minutes of back-to-back e-means fits (SPACEV 1M,
16,384 centroids, float32) took 0.2362–0.2371 s each there, at nominal
thermal pressure throughout.

A first three-seed campaign on a MacBook Pro M4 Max was discarded. Under the
same four-minute test (DEEP 1M) that laptop's e-means fit went from 0.315 s
to 0.75 s while macOS raised its thermal pressure level from nominal to
heavy. The benchmark kept 74–80% of the GPU's time during the slowdown, so
the cause was throttling and not the other applications on the machine, which
held 15–23%. O-means slows less: over 40 seconds of continuous load its fit grew
15% where e-means' grew 36%. A campaign that times the two methods one after
the other therefore times them in different states, and overstates o-means.
For DEEP 1M, float32, one epoch, that campaign reported 5.85×; the
state-matched measurement below gives 3.3×.

Two things changed as a result.

- `omeans_compare` waits for nominal thermal pressure before every timed
  call and records the level before and after it. This report is refused if
  any call ran under pressure.
- Measurements on the laptop are taken round robin: the models being compared
  take turns, one timed fit each, so that they share one machine state.
  Thermal pressure does not show the first stage of the laptop's slowdown
  (up to 30% before any pressure is reported), which is why its campaign was
  not simply repeated with the guard.

The laptop's campaign was then repeated with interleaved timing
(`omeans_interleaved`), all 18 datasets and three seeds:
[o-means against e-means on a MacBook Pro M4 Max](o-means-benchmarks-m4-max-2026-11-19.md).
It reaches e-means' error 4.13× faster in float32 and 3.96× in BF16, against
4.44× and 3.55× here, and 7.3× and 7.0× faster at ten million rows and 65,536
centroids, against 8.8× and 5.9×.

### What the speed-up consists of

Both methods are mlx-learn estimators on the same GPU with the same recipe,
and e-means' full search is the production Metal kernel. Nothing here comes
from one side having better kernels: the gain is the two-level search and,
at one epoch, the closing mean.

- **Fewer distances, each a little dearer.** At the same epochs o-means
  compares a row with a median 11.4% of the centroids (5.6–25% outside
  GloVe), 8.8 times fewer, and its fit is 4.4× faster in float32 and 3.4× in
  BF16. Per distance, end to end, it therefore costs about twice what the
  full search does. The scan reaches 52–83% of the full search's throughput
  on this GPU (kernel table in the [guide](o-means.md)), and the batch
  gather, probing, statistics, EMA and cell upkeep do not shrink with the
  search.
- **One epoch: the closing mean helps.** With a full search, o-means' error
  at one epoch is 0.24–1.20% below e-means', which is the recency-weighted
  closing mean alone. With the two-level search it is still −1.14% to +0.10%.
  O-means therefore reaches e-means' one-epoch error within one epoch in 35
  of the 36 dataset and precision pairs, and the speed-up at equal error
  (4.7× float32) exceeds the one at equal epochs (4.2×).
- **Two and three epochs: the approximate search costs epochs.** From two
  epochs both use the same closing mean, and with a full search o-means
  matches e-means within 0.05%. The two-level search then costs −0.17% to
  +0.36% in error. E-means itself gains only 0.17–0.68% (median 0.39%)
  between two and three epochs, so making up a small deficit takes many
  rows: o-means first matches e-means' two-epoch error at 2 to 2.5 epochs on
  most datasets, and its three-epoch error at 3.2 to 5. The speed-up at equal
  error falls to 4.1× (float32) at three epochs, although a two-epoch fit is
  4.7× cheaper.
- **More centroids, larger gain.** At 65,536 centroids over ten million rows
  a row is compared with 5.6–6.0% of the centroids and the fit is 9.0–10.3×
  faster at the same epochs in float32: 1.00 s against 9.05 s for one epoch
  of SPACEV 10M.
- **BF16 gains less than float32 here.** The M5 Ultra's BF16 full search is
  much faster than its float32 one at low dimension (23 against 14 TFLOP/s
  at 100 features), while the two-level scan runs at the same speed in both,
  so there is less to save.

### Uneven cells: GloVe

Unnormalized GloVe-200 is the one dataset whose cells come out uneven, and
how uneven depends on the seed. The cells are built by k-means over the
centroids, started from randomly chosen centroids. On GloVe most of the 128
cells end up with one to four centroids and a few hold the rest: after the
first build the 16 largest cells held 56–60% of the centroids with seeds 11
and 12, and 99% with seed 10. A row's 16 probes then covered 42–45% of the
centroids with seeds 11 and 12, against 13% expected for even cells, and
99–100% with seed 10. On every other dataset the share varied by at most
0.03 between seeds.

| Seed | Centroids compared | float32, same epochs | BF16, same epochs |
|---|---:|---:|---:|
| 10 | 0.99–1.00 | 1.25–1.27× | 0.82–0.87× |
| 11 | 0.44–0.45 | 2.42–2.45× | 1.66–1.76× |
| 12 | 0.43 | 2.51–2.58× | 1.76–1.84× |

The tables above take the median time over seeds, which for GloVe is one of
the two better seeds; the per-seed rows here are the complete picture. This
campaign forces the two-level search so that such fits are measured.
The default `assignment="auto"` reads the share after the first batch and
returns to the full search above one half, which here means seed 10 only. On
M4 Max, the seed-10 fit forced to two-level ran at 0.93–1.02× of e-means'
speed and the `auto` fit, after falling back at the first batch, at
0.97–1.04×. A cell build that keeps cells even on such data is an open
improvement.

### Other GPUs, same epochs

E-means and o-means at the same epoch budget, 16,384 centroids, timed round
robin (`omeans_centroid_sweep --k 16384` on M4 Max, `omeans_device_fits` on
M1 Ultra). "Faster" is e-means' fit time divided by o-means'; the error is
o-means' held-out error relative to e-means'. The M4 Max rows are a quick
check on four datasets with one seed; its full campaign is the separate
report linked above.

| GPU | Dataset | d | Precision | Faster at 1 / 2 / 3 epochs | Error at 1 / 2 / 3 epochs | Distances |
|---|---|---:|---|---|---|---:|
| M4 Max | DEEP 1M | 96 | float32 | 3.27× / 3.43× / n/a | −0.20% / −0.14% / n/a | 0.132 |
| M4 Max | DEEP 1M | 96 | bfloat16 | 2.67× / 2.93× / n/a | −0.33% / −0.00% / n/a | 0.132 |
| M4 Max | SPACEV 1M | 100 | float32 | 3.36× / 3.49× / n/a | −0.52% / +0.10% / n/a | 0.138 |
| M4 Max | SPACEV 1M | 100 | bfloat16 | 3.45× / 3.61× / n/a | −0.47% / −0.00% / n/a | 0.138 |
| M4 Max | ImageNet-CLIP | 512 | float32 | 5.15× / 5.81× / n/a | −0.65% / −0.09% / n/a | 0.115 |
| M4 Max | ImageNet-CLIP | 512 | bfloat16 | 4.97× / 5.22× / n/a | −0.47% / +0.05% / n/a | 0.115 |
| M4 Max | GooAQ | 768 | float32 | 5.48× / 5.81× / n/a | −0.16% / −0.03% / n/a | 0.107 |
| M4 Max | GooAQ | 768 | bfloat16 | 5.06× / 5.47× / n/a | −0.23% / +0.01% / n/a | 0.107 |
| M1 Ultra | SPACEV 1M | 100 | float32 | 3.99× / 3.76× / 4.01× | −0.53% / +0.10% / +0.20% | 0.138 |
| M1 Ultra | SPACEV 1M | 100 | bfloat16 | 2.86× / 2.80× / 3.01× | −0.47% / +0.03% / +0.25% | 0.138 |
| M1 Ultra | ImageNet-CLIP | 512 | float32 | 3.64× / 3.60× / 3.01× | −0.65% / −0.07% / +0.07% | 0.115 |
| M1 Ultra | ImageNet-CLIP | 512 | bfloat16 | 3.81× / 3.89× / 4.07× | −0.47% / +0.06% / +0.01% | 0.115 |

On both machines o-means with a full search (`assignment="dense"`) ran at
e-means' speed, 0.99–1.01× on M1 Ultra, so the loop itself adds nothing.
Four of the twenty M4 Max rows saw moderate thermal pressure during a call
that had started at nominal; their ratios remain state-matched.

### Where the two-level search starts to pay

`assignment="auto"` uses the two-level search when the probes are expected to
cover at most half of the centroids. With the default cells and probes that
is from 4,224 centroids below 512 features, and from 2,176 centroids from 512
features up. The sweep below is the measurement behind that rule: fit time of
`EMeans(recipe="faiss")` divided by that of `OMeans(assignment="two_level")`
on M4 Max, as the range over both precisions and one- and two-epoch fits,
timed round robin. † marks the cells where `auto` uses the full search.

| Centroids | SPACEV 1M, d=100 | SSNPP 1M, d=256 | ImageNet-CLIP, d=512 | CC-News, d=768 | CelebA, d=2,048 |
|---:|---:|---:|---:|---:|---:|
| 2,176 | 0.92–1.42× † | 0.84–1.12× † | 1.21–1.38× | 1.20–1.40× | 1.06–1.19× |
| 3,072 | 1.06–1.44× † | 0.96–1.23× † | 1.44–1.63× | 1.25–1.41× | 1.13–1.43× |
| 4,096 | 1.11–1.25× † | 1.07–1.20× † | 1.51–1.72× | 1.44–1.60× | 1.20–1.51× |
| 4,224 | 1.28–1.43× | 1.26–1.38× | 1.81–2.02× | 1.64–1.93× | 1.37–1.83× |
| 6,144 | 1.45–1.66× | 1.46–1.68× | 2.01–2.46× | 1.88–2.30× | 1.43–2.05× |
| 8,192 | 1.85–2.07× | 1.76–2.14× | 2.55–3.43× | 2.32–3.15× | 1.66–2.21× |
| 12,288 | 2.43–2.92× | 2.47–3.02× | 3.28–4.79× | 3.24–4.30× | 1.96–2.97× |

Where `auto` keeps the two-level search, its fit time is that of a forced
two-level fit (median ratio 0.998 over 116 fits), so the checks it makes
cost nothing measurable. Where it chooses the full search it runs at
e-means' speed (median 1.000 over 24 fits). Held-out error across the sweep
was −1.39% to +0.01% of e-means' at one epoch and −0.23% to +0.27% at two.
Ten of the 140 rows saw moderate thermal pressure during a call.

### Reproducing

```bash
.venv/bin/python -m benchmarks.e_means.omeans_compare run --seeds 10 11 12
.venv/bin/python -m benchmarks.e_means.omeans_report \
    --append benchmarks/e_means/omeans_report_notes.md
.venv/bin/python -m benchmarks.e_means.omeans_centroid_sweep run \
    --datasets spacev-1M fb-ssnpp-1M imagenet-clip-512-normalized vibe-ccnews vibe-celeba \
    --output centroid-sweep.jsonl
.venv/bin/python -m benchmarks.e_means.two_level_bench --dataset spacev-1M \
    --precision float32 bfloat16 --rest 20
```

Records: `benchmarks/e_means/results/omeans_comparison/` (the campaign) and
`benchmarks/e_means/results/omeans_devices/` (kernel-level timings, the
centroid sweep and the same-epoch fits per GPU).
