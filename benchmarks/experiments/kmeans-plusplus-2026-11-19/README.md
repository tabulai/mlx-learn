# K-means++ seeding on Metal, 2026-11-19

Scripts and raw results behind the
[K-means++ initialization](../../../docs/k-means.md#k-means-initialization) section of the
KMeans guide: where `sklearn.cluster.kmeans_plusplus` spends its time, and seeding and
whole-fit times before and after `KMeans` moved its seeding to a Metal screen. Measured
at `d776e92` on an M4 Max (12+4 CPU cores, 40-core GPU, 128 GB) with Python 3.14.2,
scikit-learn 1.9.1, MLX 0.32.4 and NumPy 2.5.4 (Accelerate), on MNIST: 60,000 x 784
float32, pixels scaled to [0, 1], `random_state=10`.

## Data

MNIST from the [MNIST-family benchmarks](../mnist-family-2026-11-02/README.md); set
`MNIST_FAMILY_ROOT` as described there.

## Scripts

| Script | What it does |
|---|---|
| `profile.py reference` | Runs the helper's steps with a timer around each stage and, with `--verify`, checks that the replica selects the helper's rows. It also records how many (candidate, row) pairs each step changes. |
| `profile.py screen` | Reports the stages of mlxlearn's seeding and counts of the host's work. |
| `timings.py seeding` | Times the helper and mlxlearn's seeding in interleaved rounds and compares the selected rows. |
| `timings.py fits` | Times `KMeans(k)`, `KMeans(k, init="random")`, and `KMeans(k)` with the seeding routed to the helper as before the change (`reference`). |
| `summarize.py` | Prints the seeding and whole-fit tables from `results/`. |

This laptop's GPU throttles under sustained load. `timings.py` therefore waits for nominal
macOS thermal pressure (level 0) before every timed call and records the level and the
wait.

## What the helper does per center

For every new center the helper draws `2 + int(log(k))` candidates from the current
squared distances, evaluates float64 squared distances from each candidate to all rows,
takes the minimum with each row's closest distance, sums the result per candidate, and
keeps the candidate with the smallest sum. For float32 input it upcasts the rows to
float64 in chunks and recomputes their squared norms on every call.

`profile.py reference --centers 1024 --verify`: 26.3 s for 1,024 centers with 8 candidates
each, 25.7 ms per center. The replica selects the helper's rows.

| Stage | Seconds | Share | Per center |
|---|---:|---:|---:|
| distances: row norms | 13.65 | 51.9% | 13.34 ms |
| distances: GEMM | 6.67 | 25.3% | 6.52 ms |
| distances: upcast rows to float64 | 5.01 | 19.0% | 4.89 ms |
| distances: add norms, round to float32 | 0.53 | 2.0% | 518 us |
| prefix sums | 0.14 | 0.5% | 141 us |
| potentials, best candidate | 0.05 | 0.2% | 53 us |
| minimum with closest | 0.04 | 0.2% | 44 us |
| search, gather candidates | 0.03 | 0.1% | 30 us |
| random draws | 0.01 | 0.0% | 12 us |

Distances take 99% of the time. Half of that recomputes row norms that do not change
between centers, a fifth converts the rows to float64 again, and a quarter is the matrix
product. The share of pairs a step changes falls quickly:

| Step | Pairs nearer than the row's closest center |
|---:|---:|
| 1 | 48.75% |
| 4 | 14.16% |
| 16 | 2.41% |
| 64 | 0.85% |
| 256 | 0.16% |

## What the Metal screen does per center

`profile.py screen --centers 1024 2048 4096 8192`, seconds per stage:

| Centers | Candidates | Total | `screen` | `evaluate` | `select` | `draw` | `collect` | `project` | `prepare` | Per center |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 8 | 0.94 | 0.40 | 0.28 | 0.08 | 0.09 | 0.02 | 0.04 | 0.03 | 0.92 ms |
| 2,048 | 9 | 1.51 | 0.60 | 0.42 | 0.21 | 0.18 | 0.04 | 0.03 | 0.02 | 0.74 ms |
| 4,096 | 10 | 2.93 | 1.31 | 0.65 | 0.43 | 0.40 | 0.07 | 0.03 | 0.02 | 0.71 ms |
| 8,192 | 11 | 5.40 | 2.48 | 0.94 | 0.96 | 0.83 | 0.13 | 0.03 | 0.02 | 0.66 ms |

Stages: `prepare` computes float64 row norms and the first center; `project` builds the
64-dimensional row images; `draw` forms prefix sums, draws candidates and certifies the
draw; `screen` is the Metal kernel; `collect` reads the rows with open pairs; `evaluate`
brackets the candidates' potentials and evaluates open pairs in float64; `select` sums
potentials and picks the winner.

Counts of the host's work in the same runs. Every run was certified from start to end.
Rows left with an interval are the chosen centers themselves, whose distance to their own
candidate is round-off in the helper.

| Centers | Pairs evaluated in float64 | Share of all pairs | Steps evaluated densely | Steps with a potential contest | Helper chunks recomputed | Steps on row images | Rows with an interval at the end |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 410,813 | 0.084% | 1 | 591 | 11 | 992 | 1,022 |
| 2,048 | 595,600 | 0.054% | 1 | 661 | 9 | 2,016 | 2,045 |
| 4,096 | 799,099 | 0.033% | 1 | 720 | 22 | 4,064 | 4,093 |
| 8,192 | 1,025,027 | 0.019% | 1 | 762 | 34 | 8,160 | 8,188 |

## Seeding time

`timings.py seeding --rounds 3 --reference-rounds 1`. The helper ran once per size,
mlxlearn three times (median, range). The selected rows are identical in every run.

| Centers | scikit-learn helper | Metal screen | Speedup |
|---:|---:|---:|---:|
| 1,024 | 26.9 s | 0.90 s (0.88-0.90) | 30x |
| 2,048 | 54.2 s | 1.62 s (1.59-1.65) | 34x |
| 4,096 | 108.6 s | 2.82 s (2.80-2.87) | 38x |
| 8,192 | 217.1 s | 5.41 s (5.36-5.43) | 40x |

## Whole fits

`timings.py fits --rounds 3 --reference-rounds 1`. The "before" column is the `reference`
variant, which routes the seeding to the helper as `KMeans` did until this change. It ran
once per size, the others three times (median, range). `init="random"` resolves
`n_init="auto"` to ten restarts, k-means++ to one.

| Centers | `KMeans(k)` before | `KMeans(k)` | of which Lloyd | `KMeans(k, init="random")` |
|---:|---:|---:|---:|---:|
| 1,024 | 27.5 s | 1.39 s (1.33-1.41) | 0.45 s | 6.70 s (6.69-7.67) |
| 2,048 | 54.9 s | 2.40 s (2.39-2.67) | 0.83 s | 7.97 s (7.79-9.13) |
| 4,096 | 110.4 s | 4.42 s (4.38-4.63) | 1.61 s | 10.66 s (10.60-11.01) |
| 8,192 | 218.7 s | 7.02 s (6.97-7.29) | 1.63 s | 13.59 s (13.27-14.16) |

Both seedings of `KMeans(k)` give the same fit. Its inertia against the best of the ten
random restarts:

| Centers | Lloyd iterations after k-means++ | Inertia, k-means++ | Inertia, best of ten random starts |
|---:|---:|---:|---:|
| 1,024 | 36 | 1,209,893 | 1,211,322 |
| 2,048 | 40 | 1,074,286 | 1,085,179 |
| 4,096 | 42 | 932,083 | 954,305 |
| 8,192 | 22 | 755,546 | 805,016 |

All 44 timed calls started at thermal pressure level 0; 1 of them waited for it (90 s).

## Results

`results/` holds the JSON the scripts wrote (`profile_reference.json`,
`profile_screen.json`, `seeding.json`, `fits.json`) and their console output (`*.txt`).
