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
