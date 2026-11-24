# Cost-based revival for e-means, 2026-11-24

Scripts and raw results behind the
[Cost-based revival](../../../docs/e-means.md#cost-based-revival) section of the e-means
guide: the assessment of two reference changes
([facebookresearch/faiss#5915](https://github.com/facebookresearch/faiss/pull/5915),
cost-based cluster revival, and
[#5965](https://github.com/facebookresearch/faiss/pull/5965), fused IVF fine search), the
prototype that preceded the implementation, the two studies behind the closing rule, and the
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
| `surplus.py` | Mixtures of Gaussian modes with more centroids than separated modes: fits `EMeans` with both rules and, with `--rules`, scores every closing rule on the prototype loop. `--summarize` tabulates. |
| `surplus_moves.py` | On such a mixture, counts at every check the moved centroids whose nearest centroid moves too, and the modes left without a centroid. |
| `second_score_bench.py` | Times `assign` with and without the margin, and the two statistics kernels next to `batch_stats` and `ema_update`. |
| `fit_time.py` | Times whole fits with both revival rules in turn from resident inputs. It waits for nominal thermal pressure before every timed call. |
| `closing_profile.py` | Times the three parts of the closing of a cost-rule fit with a synchronizing copy of `EMeans._reroute_closing`. |

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
| DEEP 1M (d=96) | −0.88% | 1.55 | −0.20% | 3.59 | −0.36% | >4 |
| SPACEV 1M (d=100) | −1.39% | 1.48 | −0.65% | 3.34 | −0.55% | >4 |
| FB-SSNPP 1M (d=256) | −0.35% | 1.32 | −0.20% | 3.06 | −0.23% | >4 |
| ImageNet-CLIP (d=512) | −1.38% | 1.65 | −0.58% | >4 | −0.54% | >4 |
| GloVe-200 | −0.84% | 1.29 | −0.38% | 2.65 | −0.35% | >4 |
| Mean | −0.97% | | −0.40% | | −0.41% | |

All 45 seed-level comparisons favour the cost rule (the smallest is −0.08%). The mass
rule's fits at 1.5, 2.5 and 4 epochs are those of the first campaign. Its fits at 1, 2 and 3
epochs were run again with the cost rule's: 18 of the 45 reproduced exactly and the largest
difference was 0.10%, because the batch sums are float32 atomics whose order differs from
run to run, and a revival decision can then differ.

BF16 operands (`--option '{"precision": "bfloat16"}'`, `results/library_bf16.jsonl`):

| Dataset | 1 epoch | 2 epochs |
|---|---:|---:|
| DEEP 1M | −0.93% | −0.36% |
| SPACEV 1M | −1.37% | −0.64% |
| ImageNet-CLIP | −1.46% | −0.65% |

SPACEV 10M, 65,536 centroids, seed 10 (`results/library_10m.jsonl`): mass rule 6388.48 at
two epochs and 6363.74 at three; cost rule 6346.98 at two epochs (−0.65%).

M5 Ultra, where float32 `auto` assignment takes the GEMM route instead of the M4 Max's
streaming tiles (`results/library_grid_m5_ultra.jsonl`, three seeds; that machine reads
`.npy` copies of four of the datasets):

| Dataset | 1 epoch | 2 epochs | 3 epochs |
|---|---:|---:|---:|
| DEEP 1M | −0.87% | −0.22% | −0.31% |
| SPACEV 1M | −1.33% | −0.62% | −0.55% |
| FB-SSNPP 1M | −0.35% | −0.17% | −0.22% |
| ImageNet-CLIP | −1.38% | −0.56% | −0.55% |

There 35 of 36 seed-level comparisons favour the cost rule; the other is +0.01% (DEEP, two
epochs, seed 11).

The e-means and o-means unit tests, including the new ones, pass on that machine (431
passed; the four M4 Max streaming tests skip). On a Mac Studio M1 Ultra with MLX 0.31.1 the
new test file passes for float32 and float16 (115 passed, 2 skipped); the 28 tests that use
native BF16 assignment fail as the existing BF16 tests do there, because that route needs
MLX 0.32.

### Fit time

`fit_time.py`, 16,384 centroids, alternating rounds from resident inputs, median seconds of
the mass rule and of the cost rule and their ratio.

Mac Studio M5 Ultra, which does not throttle, seven rounds
(`results/fit_time_m5_ultra.jsonl`):

| Dataset | Precision | 1 epoch | 2 epochs |
|---|---|---:|---:|
| DEEP 1M | float32 | 0.214 → 0.227, 1.062 | 0.426 → 0.436, 1.024 |
| DEEP 1M | BF16 | 0.120 → 0.132, 1.103 | 0.238 → 0.252, 1.058 |
| SPACEV 1M | float32 | 0.237 → 0.254, 1.070 | 0.473 → 0.485, 1.026 |
| SPACEV 1M | BF16 | 0.147 → 0.163, 1.107 | 0.293 → 0.309, 1.056 |
| ImageNet-CLIP | float32 | 0.699 → 0.754, 1.080 | 1.402 → 1.457, 1.039 |
| ImageNet-CLIP | BF16 | 0.571 → 0.620, 1.085 | 1.148 → 1.187, 1.035 |

MacBook Pro M4 Max, five rounds, every timed call after waiting for nominal thermal pressure
(`results/fit_time_m4_max.jsonl`; ImageNet-CLIP with BF16 at one epoch saw moderate pressure
after the longest wait):

| Dataset | Precision | 1 epoch | 2 epochs |
|---|---|---:|---:|
| DEEP 1M | float32 | 0.313 → 0.341, 1.091 | 0.744 → 0.753, 1.012 |
| DEEP 1M | BF16 | 0.298 → 0.323, 1.086 | 0.591 → 0.625, 1.058 |
| SPACEV 1M | float32 | 0.439 → 0.479, 1.091 | 0.894 → 0.934, 1.045 |
| SPACEV 1M | BF16 | 0.412 → 0.457, 1.110 | 0.843 → 0.881, 1.045 |
| ImageNet-CLIP | float32 | 2.105 → 2.258, 1.072 | 3.992 → 4.103, 1.028 |
| ImageNet-CLIP | BF16 | 1.577 → 1.709, 1.084 | 3.025 → 3.150, 1.041 |

The first implementation, which closed without looking at any row again, took 1.016–1.044
times the mass rule's time at two epochs on both machines; one-epoch fits were not timed
then.

`closing_profile.py` on the M5 Ultra (`results/closing_profile_m5_ultra.log`). The closing
takes 7–37 ms of a one-epoch fit (4.1–5.8%) and 4–16 ms of a two-epoch fit (1.0–1.6%):

| Part of the closing | 1 epoch | 2 epochs |
|---|---:|---:|
| Labels, twins and the stale mask read back | 1.0–1.2 ms | 0.8–1.1 ms |
| Stale rows assigned again (2.4–3.6% of the rows at one epoch, 0.06–0.18% at two) | 3.4–22.7 ms | 0.7–1.4 ms |
| Twins routed and sums rebuilt (one pass over the closing rows) | 2.5–13.4 ms | 2.4–13.4 ms |

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
| **Twin step, and the rows of moved centroids assigned again (implemented)** | −1.02% | −0.46% | −0.42% |
| EMA position for centroids involved in a closing-pass move (first implementation) | −0.95% | −0.46% | −0.42% |
| Assign every closing row again to the final centroids (one more pass) | −1.51% | −0.60% | −0.49% |

Closing from the rows after the last split was worse than no correction on FB-SSNPP (+0.75%
against the mass rule at one epoch, where 23,000 moves happen in the single pass) and on
GloVe.

On these datasets five rules are within 0.07% of each other, and the first implementation
took the cheapest of them: the EMA position of every centroid a closing-pass move involved,
which needs neither the labels nor another look at any row. That choice did not hold up
where most centroids move (next section), and the implementation now assigns the rows of
moved centroids again and applies the twin step. FB-SSNPP is where the first implementation
was better: −0.53% against −0.36% at one epoch and −0.26% against −0.17% at 1.5.

### More centroids than separated clusters

`surplus.py`; `results/surplus.jsonl`. Rows are drawn around Gaussian modes with unit noise
whose centres are spread by three noise units per feature. With 1.37 centroids per mode the
surplus centroids pair up inside modes, a paired centroid is cheap to remove, and 32–98% of
the centroids are involved in a move of the closing pass. Eight seeds, `recipe="faiss"`,
20,000 held-out rows.

Closing rules on the prototype loop, against the mass rule, mean over five such mixtures
(the first five rows of the next table):

| Closing rule | 1 epoch | 2 epochs | 3 epochs |
|---|---:|---:|---:|
| EMA centroids, no closing mean | +6.9% | +1.7% | +11.8% |
| Mean of the closing rows, no correction | +46.6% | +10.8% | +8.9% |
| Rows after a centroid's last split | +0.7% | −1.2% | +8.0% |
| Twin step | +2.7% | −1.2% | +6.0% |
| Hand-over (the reference's default, reimplemented) | −2.9% | −5.0% | −1.2% |
| **Twin step, and the rows of moved centroids assigned again (implemented)** | −5.0% | −5.7% | −1.7% |
| EMA position for centroids involved in a closing-pass move (first implementation) | +6.9% | +1.7% | +11.3% |
| Assign every closing row again (one more pass) | −6.1% | −6.4% | −2.5% |

`EMeans(revival="cost")` against `EMeans`, held-out error of the closing means, with the
seeds that favour the cost rule and the worst seed where one does not:

| Modes / centroids, features, rows | 1 epoch | 2 epochs | 3 epochs |
|---|---:|---:|---:|
| 3,000 / 4,096, 32, 200,000 | −7.90% (8 of 8) | −10.63% (8) | −6.74% (8) |
| 1,500 / 2,048, 32, 100,000 | −10.63% (8) | −11.33% (8) | −9.08% (8) |
| 300 / 410, 16, 40,000 | +11.43% (3, worst +42.93%) | −4.83% (7, worst +3.79%) | +0.80% (5, worst +15.44%) |
| 200 / 273, 24, 30,000 | −9.78% (8) | −9.71% (8) | +10.88% (4, worst +51.17%) |
| 150 / 205, 16, 20,000 | +3.53% (5, worst +40.12%) | −0.82% (5, worst +11.31%) | +7.39% (3, worst +38.08%) |
| 3,000 / 4,096 with overlapping modes (spread 1) | −1.59% (8) | −0.64% (8) | −0.27% (8) |
| 8,192 / 4,096 (more modes than centroids) | −3.85% (8) | −2.61% (8) | −2.01% (8) |

The EMA centroids of the same fits, which the first implementation returned for nearly every
centroid here, were +6.3%, −6.0% and +2.4% on the first mixture and between −5.2% and +57.8%
on the three small ones.

The closing repairs the positions of centroids that moved late. It cannot give a centroid to
a cluster that has none, and the losses that remain come from that. `surplus_moves.py`
(`results/surplus_moves.log`, eight seeds per case) counts, at every check, the moved
centroids whose nearest centroid moves at the same check: 41–74% of them, the two centroids
of a pair leaving together. A check leaves more modes without a centroid than it found
(for 200 modes at three epochs, 12.7–18.2 before and 18.9–25.1 after, averaged over the
checks of a fit), and training recovers them between checks. Where the last check falls on the last step of
training nothing recovers: on that mixture at three epochs (45 steps) it moved up to 96
centroids and left up to 43 modes without a centroid, against 8–18 under the mass rule; at
one epoch (59 steps, last check at step 55) the cost rule ended with 0–4 such modes against
8–19. The reference checks at the same steps, so this belongs to the rule; a rule that keeps
one centroid of a pair, or stops moving centroids shortly before the end, would depart from
it and was not tried.

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
