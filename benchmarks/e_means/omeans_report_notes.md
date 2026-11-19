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

Fits shorter than two epochs need a smaller share on the M5 Ultra. The same
sweep there (SPACEV 1M, SSNPP 1M, CC-News and CelebA at 2,176 to 8,192
centroids, 96 rows, none under thermal pressure) and the MNIST-family runs
put fits under two epochs whose probes covered 38–39% of the centroids at
0.93–1.45× of e-means' speed. Every other group on either GPU was at least
1.06× faster. The slowest were MNIST, Fashion-MNIST and Kannada-MNIST at
4,096 centroids in float32 and CelebA at 2,176 and 3,072 centroids in BF16.
Below two epochs `auto` therefore requires an expected share of at most 0.36
on M5 Ultra, and on M1 Ultra, which was not measured in this regime. That
keeps the full search for such fits with 2,176 to 4,223 centroids at 512
features or more. The M4 Max keeps the limit of one half. Fits of two epochs
or more were at least 1.08× faster on both GPUs and are unchanged.

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
