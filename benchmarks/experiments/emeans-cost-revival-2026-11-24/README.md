# Cost-based revival for e-means, 2026-11-24

Scripts and raw results behind the
[Cost-based revival](../../../docs/e-means.md#cost-based-revival) section of the e-means
guide: the assessment of two reference changes
([facebookresearch/faiss#5915](https://github.com/facebookresearch/faiss/pull/5915),
cost-based cluster revival, and
[#5965](https://github.com/facebookresearch/faiss/pull/5965), fused IVF fine search), the
prototype that preceded the implementation, the study that chose the closing rule, and the
measurements of `EMeans(revival="cost")`. Measured on a MacBook Pro M4 Max (40-core GPU,
128 GB) with MLX 0.32.4, Python 3.14 and faiss 1.16.0 (scoring only), and where stated on a
Mac Studio M5 Ultra with the same packages.

All quality numbers are held-out mean squared error on the test split of the e-means
benchmark datasets, scored by `benchmarks.e_means.run.mse` (faiss float32 assignment,
float64 residuals), against `EMeans(recipe="faiss")` with the same seed, initial centroids
and epoch budget. Seeds are 10, 11 and 12.

## Data

The cached datasets of the [e-means benchmarks](../../e_means/README.md)
(`~/.cache/mlxlearn/e_means`) and the pinned faiss checkout that defines their splits. In a
git worktree pass `--faiss-source /path/to/main-checkout/tmp/e_means/faiss`: that folder is
untracked.

## Scripts

| Script | What it does |
|---|---|
| `assess.py stages` | Times the stages of `OMeans`' two-level assignment, with the probe stage split into its coarse product and its selection. |
| `assess.py proto` | Fits one configuration per mode and scores it. Modes: `lib` (`EMeans`), `libcost` (`EMeans(revival="cost")`), and the prototypes written before the implementation: `mass` (the library rule in the prototype loop, to validate the loop), `cost` (the reference rule, closing from the rows after a centroid's last split) and `massaxis` (mass-selected donors split along the long axis). `--option` passes further `EMeans` options to the library modes. |
| `analyze.py` | Change in error at equal epochs, and the epoch budget at which the mass rule reaches the same error. |
| `closing_study.py` | Trains once per configuration, keeps the closing-pass labels, and evaluates every closing rule on that one trajectory. |
| `closing_summary.py` | Tabulates the closing study. |
| `second_score_bench.py` | Times `assign` with and without the margin, and the two statistics kernels next to `batch_stats` and `ema_update`. |
| `fit_time.py` | Times whole fits with both revival rules in turn from resident inputs. It waits for nominal thermal pressure before every timed call. |

This laptop's GPU throttles under sustained load. Timings were taken with the two arms in
alternating order and at nominal thermal pressure; quality runs are not timed.

## Results

### `EMeans(revival="cost")` against the mass rule

`assess.py proto --modes lib libcost` and `analyze.py --mode libcost`;
`results/library_grid.jsonl`. 16,384 centroids, float32, mean of three seeds. "Needs" is the
epoch budget at which the mass rule reaches the same error, interpolated between its
results at 1, 1.5, 2, 2.5, 3 and 4 epochs.

| Dataset | 1 epoch | Needs | 2 epochs | Needs | 3 epochs | Needs |
|---|---:|---:|---:|---:|---:|---:|
| DEEP 1M (d=96) | −0.71% | 1.43 | −0.22% | 3.54 | −0.34% | >4 |
| SPACEV 1M (d=100) | −1.21% | 1.41 | −0.64% | 3.29 | −0.55% | >4 |
| FB-SSNPP 1M (d=256) | −0.50% | 1.46 | −0.19% | 2.97 | −0.24% | >4 |
| ImageNet-CLIP (d=512) | −1.18% | 1.46 | −0.57% | >4 | −0.52% | >4 |
| GloVe-200 | −0.87% | 1.30 | −0.43% | 2.66 | −0.39% | >4 |
| Mean | −0.89% | | −0.41% | | −0.41% | |

All 45 seed-level comparisons favour the cost rule (the smallest is −0.08%).

BF16 operands (`--option '{"precision": "bfloat16"}'`, `results/library_bf16.jsonl`):

| Dataset | 1 epoch | 2 epochs |
|---|---:|---:|
| DEEP 1M | −0.77% | −0.36% |
| SPACEV 1M | −1.16% | −0.63% |
| ImageNet-CLIP | −1.25% | −0.64% |

SPACEV 10M, 65,536 centroids, seed 10 (`results/library_10m.jsonl`): mass rule 6388.48 at
two epochs and 6363.74 at three; cost rule 6347.14 at two epochs (−0.65%).

M5 Ultra, where float32 `auto` assignment takes the GEMM route instead of the M4 Max's
streaming tiles (`results/library_grid_m5_ultra.jsonl`, three seeds):

| Dataset | 1 epoch | 2 epochs | 3 epochs |
|---|---:|---:|---:|
| SPACEV 1M | −1.15% | −0.61% | −0.55% |
| ImageNet-CLIP | −1.18% | −0.55% | −0.54% |

The clustering unit tests, including the new ones, pass on that machine (390 passed; the
four M4 Max streaming tests skip). On a Mac Studio M1 Ultra with MLX 0.31.1 the new tests
pass for float32 and float16 (83 passed); the 24 that use native BF16 assignment fail as the
existing BF16 tests do there, because that route needs MLX 0.32.

### Fit time

`fit_time.py`, two epochs, 16,384 centroids, five alternating rounds, every timed call at
nominal thermal pressure; `results/fit_time_m4_max.jsonl`.

| Dataset | Precision | Mass rule | Cost rule | Ratio |
|---|---|---:|---:|---:|
| DEEP 1M | float32 | 0.714 s | 0.746 s | 1.044 |
| DEEP 1M | BF16 | 0.573 s | 0.589 s | 1.028 |
| SPACEV 1M | float32 | 0.807 s | 0.837 s | 1.037 |
| SPACEV 1M | BF16 | 0.760 s | 0.779 s | 1.025 |
| ImageNet-CLIP | float32 | 3.376 s | 3.513 s | 1.041 |
| ImageNet-CLIP | BF16 | 2.627 s | 2.727 s | 1.038 |

On a Mac Studio M5 Ultra, which does not throttle, with seven alternating rounds
(`results/fit_time_m5_ultra.jsonl`; that machine reads `.npy` copies of the datasets):

| Dataset | Precision | Mass rule | Cost rule | Ratio |
|---|---|---:|---:|---:|
| SPACEV 1M | float32 | 0.471 s | 0.478 s | 1.016 |
| SPACEV 1M | BF16 | 0.293 s | 0.306 s | 1.044 |
| ImageNet-CLIP | float32 | 1.398 s | 1.437 s | 1.028 |
| ImageNet-CLIP | BF16 | 1.146 s | 1.174 s | 1.024 |

`second_score_bench.py` (`results/second_score_bench_m4_max.log`): the margin changes the
time of `assign` by −3% to +6% at 16,384 and 65,536 centroids, inside this laptop's noise;
`removal_stats` takes about as long as `batch_stats` (0.26–0.87 ms for 32,768 rows at 96 to
512 features) and `removal_update` 0.22–0.52 ms.

### Closing rule

`closing_study.py` and `closing_summary.py`; `results/closing_study.jsonl`. Two seeds, the
prototype's training loop, so that every rule is scored on the same trajectory. Held-out
error of cost-based revival under each closing rule against the mass rule with its own
closing, mean over five datasets:

| Closing rule | 1 epoch | 1.5 epochs | 2 epochs |
|---|---:|---:|---:|
| EMA centroids, no closing mean | −0.95% | −0.44% | −0.41% |
| Mean of the closing rows, no correction | −0.27% | −0.31% | −0.37% |
| Rows after a centroid's last split (reference without captured labels) | −0.34% | −0.26% | −0.36% |
| Clear the moved centroid only | −0.54% | −0.37% | −0.39% |
| Twin step: a donor's earlier rows go to the nearer half | −0.95% | −0.45% | −0.42% |
| Hand-over: twin step and second-nearest for moved centroids (the reference's default, reimplemented) | −0.96% | −0.46% | −0.42% |
| **EMA position for centroids involved in a closing-pass move (implemented)** | −0.95% | −0.46% | −0.42% |
| Assign every closing row again to the final centroids (one more pass) | −1.51% | −0.60% | −0.49% |

Closing from the rows after the last split was worse than no correction on FB-SSNPP (+0.75%
against the mass rule at one epoch, where 23,000 moves happen in the single pass) and on
GloVe. The implemented rule matches the hand-over within 0.02% on every average and needs
neither second labels nor another pass over the rows.

### Prototype and ablation

`results/proto_grid.jsonl`, `results/proto_axis.jsonl`, `results/proto_10m.jsonl`: the
prototype of the reference rule written for the assessment, closing from the rows after the
last split. Its one-epoch results on FB-SSNPP (+0.74%) and GloVe (+0.01%) are what the
closing study explains. Splitting mass-selected donors along the long axis, which needs no
second-nearest distance, recovers a fraction of the gain: −0.18%, −0.10% and −0.08% at one,
two and three epochs on DEEP, SPACEV and FB-SSNPP, against −0.81%, −0.35% and −0.38% for the
cost rule on the same three datasets.

## Two-level assignment stages

`assess.py stages`, 32,768 rows against 16,384 centroids of a one-epoch fit on SPACEV 1M
(128 cells, 16 probes), and 65,536 rows against 65,536 centroids on SPACEV 10M (512 cells,
24 probes); `results/stages_spacev.json`, `results/stages_spacev10m.json`. Each stage is
timed with its own synchronization, so the stages sum to more than the whole.

| Stage | 16,384 centroids | 65,536 centroids |
|---|---:|---:|
| Dense assignment (`EMeans`) | 12.14 ms | 94.11 ms |
| Two-level assignment, whole | 3.66 ms | 11.52 ms |
| Probe: coarse product and selection | 0.83 ms | 4.34 ms |
| Probe: selection alone | 0.66 ms | 2.56 ms |
| Bucket | 0.47 ms | 0.41 ms |
| Scan | 2.74 ms | 7.55 ms |
| Reduce | 0.31 ms | 0.29 ms |

The fused fine search of #5965 is the design `OMeans` already has (pairs bucketed by cell,
one tile of rows per threadgroup against the cell's centroids, one result slot per pair, one
merge). Its probe selection addresses the stage that takes 23% and 38% of the two-level
assignment here; its radix select relies on a histogram primitive of the reference's kernel
language and targets 1,024 lists or more, so it is not a drop-in.
