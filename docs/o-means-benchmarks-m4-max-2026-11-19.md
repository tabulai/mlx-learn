# o-means against e-means on a MacBook Pro M4 Max

Generated from 108 records over 18 datasets, seeds [10, 11, 12], on Apple M4 Max (MLX 0.32.4, Python 3.14.2). Both methods use the Faiss e-means recipe, the same batch law, seed and starting centroids, with the training matrix resident on the GPU. Times are medians of synchronized `train_device` calls after a full warm-up fit, then the median over seeds; errors are held-out MSE with exact float32 assignment and a float64 residual, averaged over seeds. The timed calls were interleaved: within each dataset, precision and seed, all configurations took turns in freshly drawn orders, so that they share the machine's thermal state. Each call waited for nominal macOS thermal pressure, for a bounded time; the worst level seen before or after a call was: nominal 97.9%, moderate 1.9%, heavy 0.1%.

## Summary

Geometric means over datasets and e-means budgets. "Same epochs" divides e-means' fit time by o-means' at the budgets both were measured at.

| Precision | Datasets | Time to e-means' error, interpolated | First measured budget | Slowest | Fastest | Same epochs | Distances at same epochs, median |
|---|---:|---:|---:|---|---|---:|---:|
| bfloat16 | 18 | 3.96× | 3.57× | 1.58× (vibe-glove, 3 epochs) | 8.01× (spacev-10M, 1 epochs) | 3.90× | 0.114 |
| float32 | 18 | 4.13× | 3.67× | 1.53× (vibe-glove, 3 epochs) | 7.97× (deep1b-10M, 3 epochs) | 4.20× | 0.114 |

## bfloat16

### Time to e-means' error

Speed-up is e-means' fit time divided by the time o-means needs to reach the same held-out error. `≥` marks a target the smallest o-means budget already met. The measured column uses the first o-means budget that reaches the target, without interpolation. Distances are those o-means evaluated to reach the target, as a fraction of e-means' n × k × T.

| Dataset | n | d | k | Speed-up at 1 | Speed-up at 2 | Speed-up at 3 | Measured at 1 | Measured at 2 | Measured at 3 | Distances at 1 / 2 / 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deep1b-10M | 10,000,000 | 96 | 65,536 | 7.08 | 7.45 | 6.28 | 5.81 | 7.14 | 5.94 | 0.05 / 0.04 / 0.06 |
| deep1b-1M | 1,000,000 | 96 | 16,384 | 3.05 | 2.78 | 2.96 | 2.86 | 2.44 | 2.75 | 0.12 / 0.14 / 0.13 |
| dino10b-1M-div4096 | 1,000,000 | 1024 | 16,384 | 5.90 | 5.33 | 5.32 | 5.31 | 4.75 | 5.24 | 0.10 / 0.11 / 0.11 |
| fb-ssnpp-1M | 1,000,000 | 256 | 16,384 | 3.90 | 5.01 | 3.18 | 3.46 | 4.13 | 2.84 | 0.14 / 0.11 / 0.19 |
| imagenet-clip-512-normalized | 1,000,000 | 512 | 16,384 | 5.65 | 5.25 | 5.20 | 4.90 | 5.05 | 5.09 | 0.10 / 0.11 / 0.12 |
| sift1b-1M-div512 | 1,000,000 | 128 | 16,384 | 3.20 | 3.12 | 3.29 | 3.01 | 3.11 | 3.08 | 0.12 / 0.14 / 0.14 |
| spacev-10M | 10,000,000 | 100 | 65,536 | 8.01 | 7.02 | 6.10 | 6.72 | 6.58 | 5.12 | 0.05 / 0.07 / 0.08 |
| spacev-1M | 1,000,000 | 100 | 16,384 | 3.80 | 3.37 | 3.40 | 3.42 | 3.03 | 2.90 | 0.12 / 0.15 / 0.15 |
| vibe-agnews | 749,088 | 1024 | 16,384 | 4.86 | 4.65 | 4.03 | 4.24 | 3.89 | 3.57 | 0.09 / 0.11 / 0.13 |
| vibe-arxiv | 1,324,643 | 768 | 16,384 | 5.63 | 4.99 | 4.08 | 4.80 | 4.21 | 3.80 | 0.09 / 0.11 / 0.13 |
| vibe-ccnews | 475,295 | 768 | 8,192 | 2.82 | 2.92 | 2.65 | 2.36 | 2.86 | 2.35 | 0.17 / 0.19 / 0.23 |
| vibe-celeba | 181,573 | 2048 | 8,192 | 1.93 | 2.71 | 3.75 | 1.62 | 2.58 | 3.70 | 0.20 / 0.16 / 0.12 |
| vibe-codesearchnet | 1,354,065 | 768 | 16,384 | 5.38 | 4.16 | 3.16 | 5.34 | 3.35 | 2.94 | 0.11 / 0.14 / 0.18 |
| vibe-glove | 1,172,514 | 200 | 16,384 | 2.16 | 2.50 | 1.58 | 2.01 | 2.38 | 1.58 | 0.57 / 0.48 / 0.66 |
| vibe-gooaq | 1,455,024 | 768 | 16,384 | 5.32 | 4.96 | 3.69 | 4.92 | 4.40 | 3.69 | 0.10 / 0.12 / 0.14 |
| vibe-landmark-dino | 740,740 | 768 | 16,384 | 5.29 | 4.01 | 4.15 | 4.47 | 3.69 | 3.60 | 0.09 / 0.12 / 0.12 |
| vibe-landmark-nomic | 740,729 | 768 | 16,384 | 4.75 | 5.19 | 3.77 | 4.17 | 5.09 | 3.19 | 0.10 / 0.11 / 0.16 |
| vibe-simplewiki | 240,372 | 3072 | 8,192 | 2.89 | 2.21 | 2.17 | 2.31 | 2.14 | 1.71 | 0.18 / 0.23 / 0.24 |
| **Geometric mean** | | | | 4.22 | 4.07 | 3.62 | 3.72 | 3.72 | 3.31 | 0.13 |

Geometric mean over all datasets and targets: 3.96× interpolated, 3.57× measured-only.

### Same epoch budget

Only the budgets both methods were measured at. o-means and e-means at the same number of epochs: how much faster o-means' fit is, its held-out error relative to e-means' (negative is better), the share of distances it evaluated, and the error of o-means with a full search, which isolates the closing mean.

| Dataset | Faster at 1 | Faster at 2 | Error at 1 | Error at 2 | Distances | Full-search error at 1 / 2 |
|---|---:|---:|---:|---:|---:|---:|
| deep1b-10M | 5.81× | 6.63× | -0.27% | -0.17% | 0.058 | -0.24% / +0.00% |
| deep1b-1M | 2.86× | 2.97× | -0.25% | +0.02% | 0.132 | -0.42% / +0.00% |
| dino10b-1M-div4096 | 5.31× | 5.46× | -0.38% | +0.01% | 0.107 | -0.45% / +0.00% |
| fb-ssnpp-1M | 3.46× | 3.42× | -0.29% | +0.06% | 0.159 | -0.40% / +0.00% |
| imagenet-clip-512-normalized | 4.90× | 5.05× | -0.54% | -0.06% | 0.114 | -0.62% / +0.00% |
| sift1b-1M-div512 | 3.01× | 3.11× | -0.26% | -0.01% | 0.136 | -0.31% / +0.00% |
| spacev-10M | 6.72× | 8.63× | -0.37% | +0.19% | 0.059 | -0.48% / +0.00% |
| spacev-1M | 3.42× | 3.73× | -0.47% | +0.13% | 0.138 | -0.54% / +0.00% |
| vibe-agnews | 4.24× | 4.96× | -0.65% | +0.10% | 0.106 | -0.87% / +0.00% |
| vibe-arxiv | 4.80× | 5.28× | -0.52% | +0.06% | 0.107 | -0.54% / +0.00% |
| vibe-ccnews | 2.36× | 2.86× | -0.77% | -0.07% | 0.198 | -0.80% / +0.00% |
| vibe-celeba | 1.62× | 2.25× | -0.72% | -0.04% | 0.238 | -0.86% / +0.00% |
| vibe-codesearchnet | 5.34× | 5.40× | -0.04% | +0.36% | 0.111 | -0.59% / +0.00% |
| vibe-glove | 2.01× | 2.01× | -0.37% | -0.01% | 0.623 | -0.39% / +0.00% |
| vibe-gooaq | 4.92× | 5.54× | -0.27% | +0.09% | 0.106 | -0.50% / +0.00% |
| vibe-landmark-dino | 4.47× | 4.84× | -0.59% | +0.12% | 0.106 | -0.68% / +0.00% |
| vibe-landmark-nomic | 4.17× | 5.09× | -0.65% | -0.07% | 0.115 | -0.71% / +0.00% |
| vibe-simplewiki | 2.31× | 2.22× | -1.13% | +0.04% | 0.218 | -1.20% / +0.00% |
| **Geometric mean** | 3.72× | 4.09× | | | | |

### Error at e-means' time

o-means' held-out error relative to e-means', when o-means is given the time e-means needed. Negative is better.

| Dataset | At e-means' 1-epoch time | At e-means' 2-epoch time | At e-means' 3-epoch time |
|---|---:|---:|---:|
| deep1b-10M | -1.24% | -0.58% | -0.24% |
| deep1b-1M | -1.39% | -0.51% | -0.39% |
| dino10b-1M-div4096 | -1.73% | -0.49% | -0.26% |
| fb-ssnpp-1M | -0.82% | -0.38% | -0.16% |
| imagenet-clip-512-normalized | -2.23% | -0.68% | -0.36% |
| sift1b-1M-div512 | -1.49% | -0.40% | -0.03% |
| spacev-10M | -2.46% | -0.78% | -0.36% |
| spacev-1M | -2.71% | -1.05% | -0.49% |
| vibe-agnews | -3.40% | -1.02% | -0.44% |
| vibe-arxiv | -2.43% | -0.72% | -0.36% |
| vibe-ccnews | -2.99% | -1.07% | -0.53% |
| vibe-celeba | -1.31% | -0.37% | -0.26% |
| vibe-codesearchnet | -2.57% | -0.86% | -0.31% |
| vibe-glove | -1.65% | -0.85% | -0.25% |
| vibe-gooaq | -2.52% | -0.90% | -0.35% |
| vibe-landmark-dino | -1.98% | -0.47% | -0.20% |
| vibe-landmark-nomic | -2.74% | -0.98% | -0.31% |
| vibe-simplewiki | -3.27% | -1.02% | -0.67% |

## float32

### Time to e-means' error

Speed-up is e-means' fit time divided by the time o-means needs to reach the same held-out error. `≥` marks a target the smallest o-means budget already met. The measured column uses the first o-means budget that reaches the target, without interpolation. Distances are those o-means evaluated to reach the target, as a fraction of e-means' n × k × T.

| Dataset | n | d | k | Speed-up at 1 | Speed-up at 2 | Speed-up at 3 | Measured at 1 | Measured at 2 | Measured at 3 | Distances at 1 / 2 / 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deep1b-10M | 10,000,000 | 96 | 65,536 | 7.87 | 7.64 | 7.97 | 6.73 | 6.59 | 7.45 | 0.05 / 0.06 / 0.06 |
| deep1b-1M | 1,000,000 | 96 | 16,384 | 3.52 | 2.85 | 4.00 | 3.27 | 2.37 | 3.65 | 0.12 / 0.17 / 0.13 |
| dino10b-1M-div4096 | 1,000,000 | 1024 | 16,384 | 6.00 | 5.10 | 4.70 | 5.29 | 5.08 | 4.54 | 0.09 / 0.11 / 0.11 |
| fb-ssnpp-1M | 1,000,000 | 256 | 16,384 | 4.59 | 5.41 | 3.17 | 4.08 | 4.46 | 3.01 | 0.14 / 0.11 / 0.19 |
| imagenet-clip-512-normalized | 1,000,000 | 512 | 16,384 | 6.08 | 6.16 | 4.26 | 5.01 | 5.93 | 3.34 | 0.10 / 0.11 / 0.12 |
| sift1b-1M-div512 | 1,000,000 | 128 | 16,384 | 4.11 | 3.81 | 4.25 | 3.62 | 3.74 | 3.83 | 0.12 / 0.13 / 0.13 |
| spacev-10M | 10,000,000 | 100 | 65,536 | 7.76 | 6.59 | 6.00 | 7.04 | 6.53 | 5.04 | 0.05 / 0.07 / 0.08 |
| spacev-1M | 1,000,000 | 100 | 16,384 | 4.29 | 3.80 | 3.30 | 3.96 | 3.10 | 2.88 | 0.12 / 0.14 / 0.16 |
| vibe-agnews | 749,088 | 1024 | 16,384 | 6.01 | 4.91 | 5.15 | 5.50 | 4.49 | 4.87 | 0.09 / 0.11 / 0.11 |
| vibe-arxiv | 1,324,643 | 768 | 16,384 | 6.04 | 4.79 | 5.15 | 5.24 | 4.20 | 4.93 | 0.09 / 0.12 / 0.11 |
| vibe-ccnews | 475,295 | 768 | 8,192 | 2.91 | 2.99 | 2.56 | 2.42 | 2.99 | 2.37 | 0.16 / 0.20 / 0.24 |
| vibe-celeba | 181,573 | 2048 | 8,192 | 2.23 | 3.03 | 1.78 | 1.88 | 2.90 | 1.47 | 0.20 / 0.18 / 0.32 |
| vibe-codesearchnet | 1,354,065 | 768 | 16,384 | 5.19 | 4.23 | 3.03 | 3.39 | 3.25 | 2.66 | 0.11 / 0.14 / 0.17 |
| vibe-glove | 1,172,514 | 200 | 16,384 | 2.09 | 2.33 | 1.53 | 1.92 | 2.23 | 1.50 | 0.57 / 0.48 / 0.65 |
| vibe-gooaq | 1,455,024 | 768 | 16,384 | 6.51 | 5.24 | 3.62 | 6.11 | 4.86 | 3.60 | 0.10 / 0.12 / 0.17 |
| vibe-landmark-dino | 740,740 | 768 | 16,384 | 5.31 | 4.96 | 4.17 | 4.46 | 4.09 | 4.08 | 0.09 / 0.11 / 0.11 |
| vibe-landmark-nomic | 740,729 | 768 | 16,384 | 5.11 | 4.64 | 3.79 | 4.51 | 3.72 | 3.23 | 0.10 / 0.12 / 0.13 |
| vibe-simplewiki | 240,372 | 3072 | 8,192 | 3.09 | 2.18 | 1.91 | 2.40 | 1.81 | 1.85 | 0.18 / 0.22 / 0.23 |
| **Geometric mean** | | | | 4.61 | 4.24 | 3.60 | 3.97 | 3.78 | 3.29 | 0.13 |

Geometric mean over all datasets and targets: 4.13× interpolated, 3.67× measured-only.

### Same epoch budget

Only the budgets both methods were measured at. o-means and e-means at the same number of epochs: how much faster o-means' fit is, its held-out error relative to e-means' (negative is better), the share of distances it evaluated, and the error of o-means with a full search, which isolates the closing mean.

| Dataset | Faster at 1 | Faster at 2 | Error at 1 | Error at 2 | Distances | Full-search error at 1 / 2 |
|---|---:|---:|---:|---:|---:|---:|
| deep1b-10M | 6.73× | 8.71× | -0.35% | +0.07% | 0.058 | -0.27% / +0.01% |
| deep1b-1M | 3.27× | 3.59× | -0.31% | +0.03% | 0.132 | -0.43% / +0.00% |
| dino10b-1M-div4096 | 5.29× | 5.08× | -0.40% | -0.00% | 0.107 | -0.45% / +0.00% |
| fb-ssnpp-1M | 4.08× | 3.62× | -0.32% | +0.05% | 0.159 | -0.42% / +0.00% |
| imagenet-clip-512-normalized | 5.01× | 5.93× | -0.58% | -0.07% | 0.114 | -0.59% / -0.00% |
| sift1b-1M-div512 | 3.62× | 3.74× | -0.33% | -0.03% | 0.136 | -0.34% / +0.00% |
| spacev-10M | 7.04× | 8.68× | -0.29% | +0.18% | 0.059 | -0.50% / +0.00% |
| spacev-1M | 3.96× | 3.99× | -0.43% | +0.02% | 0.138 | -0.58% / +0.00% |
| vibe-agnews | 5.50× | 4.98× | -0.67% | +0.06% | 0.106 | -0.84% / +0.01% |
| vibe-arxiv | 5.24× | 5.17× | -0.52% | +0.09% | 0.107 | -0.54% / -0.00% |
| vibe-ccnews | 2.42× | 2.99× | -0.81% | -0.00% | 0.199 | -0.78% / -0.00% |
| vibe-celeba | 1.88× | 2.60× | -0.70% | -0.03% | 0.238 | -0.82% / +0.00% |
| vibe-codesearchnet | 5.35× | 5.33× | +0.08% | +0.29% | 0.111 | -0.60% / -0.00% |
| vibe-glove | 1.92× | 1.88× | -0.35% | -0.03% | 0.623 | -0.37% / -0.03% |
| vibe-gooaq | 6.11× | 5.67× | -0.26% | +0.14% | 0.106 | -0.53% / +0.01% |
| vibe-landmark-dino | 4.46× | 5.32× | -0.59% | +0.05% | 0.106 | -0.68% / -0.01% |
| vibe-landmark-nomic | 4.51× | 4.73× | -0.64% | +0.03% | 0.115 | -0.75% / +0.00% |
| vibe-simplewiki | 2.40× | 2.19× | -1.11% | +0.01% | 0.218 | -1.16% / +0.00% |
| **Geometric mean** | 4.07× | 4.32× | | | | |

### Error at e-means' time

o-means' held-out error relative to e-means', when o-means is given the time e-means needed. Negative is better.

| Dataset | At e-means' 1-epoch time | At e-means' 2-epoch time | At e-means' 3-epoch time |
|---|---:|---:|---:|
| deep1b-10M | -1.24% | -0.50% | -0.26% |
| deep1b-1M | -1.65% | -0.47% | -0.37% |
| dino10b-1M-div4096 | -1.73% | -0.51% | -0.27% |
| fb-ssnpp-1M | -0.88% | -0.36% | -0.16% |
| imagenet-clip-512-normalized | -2.17% | -0.59% | -0.27% |
| sift1b-1M-div512 | -1.55% | -0.65% | -0.37% |
| spacev-10M | -2.32% | -0.70% | -0.29% |
| spacev-1M | -2.70% | -1.05% | -0.49% |
| vibe-agnews | -3.45% | -1.10% | -0.58% |
| vibe-arxiv | -2.40% | -0.71% | -0.39% |
| vibe-ccnews | -2.95% | -1.05% | -0.50% |
| vibe-celeba | -1.31% | -0.40% | -0.18% |
| vibe-codesearchnet | -2.43% | -0.88% | -0.37% |
| vibe-glove | -1.65% | -0.74% | -0.26% |
| vibe-gooaq | -2.57% | -0.86% | -0.32% |
| vibe-landmark-dino | -2.03% | -0.66% | -0.34% |
| vibe-landmark-nomic | -2.64% | -0.96% | -0.46% |
| vibe-simplewiki | -3.17% | -1.01% | -0.61% |

## Notes

### How this campaign differs from the reference

The reference campaign ran on a Mac Studio M5 Ultra, one configuration after
another: [o-means against e-means on Apple silicon](o-means-benchmarks-2026-11-18.md).
This one repeats it on a MacBook Pro M4 Max with `omeans_interleaved`: the
same models, budgets, datasets and seeds, with the timed calls of all 14
configurations of a (dataset, precision, seed) taken in turn.

The laptop needs that. Its GPU slows under sustained load, and the full
search slows more than the two-level search, so configurations timed minutes
apart are timed in different states. A first, sequential campaign on this
machine was discarded for that reason; the reference report has the
measurements.

Each timed call waited for nominal thermal pressure, for at most 120 seconds.
The laptop reaches moderate pressure after 55 to 95 seconds of continuous
load and needs 74 to 81 seconds idle to return to nominal, so the campaign
spent 152 minutes of its six hours waiting. Of the 3,452 timed calls, 97.9%
started and ended at nominal pressure, 1.9% saw moderate and 0.1% heavy
pressure, in each case rising during the call. 31 of those 71 calls were fits
on the two 10M-row datasets, which run for up to a minute.

### How well the states were matched

O-means with a full search does the same work as e-means, and from two epochs
the same computation, so its fit time relative to e-means' measures how well
the interleaving matched the machine state. Over the 216 pairs of one record
and one budget the ratio had a median of 0.99, and 80% of the pairs lay
between 0.93 and 1.07. After the median over seeds it was 0.88 to 1.14 per
dataset, with a geometric mean of 0.996 in float32 and 0.989 in BF16. The
same ratio ranged from 0.76 to 1.45 in the discarded sequential campaign.

A single ratio in the tables above therefore carries several percent of
timing noise. The geometric means are much tighter.

### Against the M5 Ultra

| | M4 Max, float32 | M5 Ultra, float32 | M4 Max, BF16 | M5 Ultra, BF16 |
|---|---:|---:|---:|---:|
| Time to e-means' error, 18 datasets | 4.13× | 4.44× | 3.96× | 3.55× |
| Same, measured budgets only | 3.67× | 3.96× | 3.57× | 3.21× |
| Same epochs, 18 datasets | 4.20× | 4.44× | 3.90× | 3.42× |
| Time to e-means' error, 10M rows and 65,536 centroids | 7.26× | 8.84× | 6.96× | 5.91× |
| Same epochs, 10M rows and 65,536 centroids | 7.74× | 9.62× | 6.88× | 5.85× |

Held-out errors and the share of centroids compared are those of the
reference campaign up to run-to-run noise, since they do not depend on the
GPU. BF16 gains about as much as float32 here: on this GPU the BF16 full
search is only modestly faster than the float32 one (9–14 against 9–12
TFLOP/s in the kernel table of the [guide](o-means.md)), whereas on the M5
Ultra it is up to 1.6 times faster.

### GloVe

As in the reference campaign, the cells of unnormalized GloVe-200 depend on
the seed. With seed 10 the search compared 99–100% of the centroids and the
fit ran at 0.92–1.04× of e-means' speed; with seeds 11 and 12 it compared
43–45% and was 1.75–2.12× faster. The tables take the median over seeds.

### Reproducing

```bash
.venv/bin/python -m benchmarks.e_means.omeans_interleaved run --seeds 10 11 12 \
    --settle 120 --out benchmarks/e_means/results/omeans_interleaved_m4_max
.venv/bin/python -m benchmarks.e_means.omeans_report \
    --results benchmarks/e_means/results/omeans_interleaved_m4_max \
    --output docs/o-means-benchmarks-m4-max-2026-11-19.md \
    --title "o-means against e-means on a MacBook Pro M4 Max" \
    --append benchmarks/e_means/omeans_report_notes_m4_max.md
```
