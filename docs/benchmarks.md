# Benchmarks

Every number here was measured, on one machine, with nothing else benchmarking beside it:
an Apple M4 Max (16 cores, 12 performance and 4 efficiency; 128 GiB), macOS 26.6.2, MLX
0.32.2. Stock scikit-learn is 1.9.0 on Python 3.13.3 with NumPy 2.5.2 and SciPy 1.18.1;
every dispatch threshold was also checked against scikit-learn 1.7.2 on Python 3.10.19
with NumPy 2.2.6 and SciPy 1.15.3, the other end of the supported range. The runners log
the load average and the busiest processes before every grid; beside the benchmark itself
the busiest was the window server, at about half of one core.

Two tools produce the tables here, and the text before each says which:

```bash
# The calibration grids behind every dispatch threshold, once per scikit-learn version,
# then side by side:
python -m benchmarks.calibrate list
python -m benchmarks.calibrate run knn --output bench/calibration/knn-1.9.0.jsonl
python -m benchmarks.calibrate table knn bench/calibration/knn-*.jsonl
# ...or re-render every calibration table in this page from the committed results:
python -m benchmarks.calibrate update docs/benchmarks.md benchmarks/results/calibration/*.jsonl

# Patched dispatch against stock on both sides of every threshold (the gate), and the
# kernel cases published below:
python -m benchmarks.run --profile smoke --report bench/smoke.md
python -m benchmarks.run --profile published --report bench/published.md
```

A **calibration table** is introduced by its grid's name, and its source carries the
command that renders it (`python -m benchmarks.calibrate table <grid>`, in an HTML
comment). Every range that the README, the CHANGELOG, the private `Tuning` values and the
dispatch docstrings quote is read off one of them, by grid name. Its `dispatch` column is computed from the rules as they stand, so a table can be
checked against the thresholds it justifies. The raw results the tables were rendered from
are in [`benchmarks/results/calibration/`](../benchmarks/results/calibration/).
`smoke` measures what `patch_sklearn()` users get, on both sides of every threshold;
`published` forces the Layer 1 classes onto the MLX device and reports kernel timings.
`--profile full` is the regression grid; see [`development.md`](development.md#benchmarks).
The release-over-release timings in the CHANGELOG come from `benchmarks/ab.py`, which runs
two source trees in alternating fresh processes; its results against 0.1.0a2 are
[`benchmarks/results/ab-0.1.0a2.jsonl`](../benchmarks/results/ab-0.1.0a2.jsonl).

## Method

**Synchronized.** MLX is lazy, so a benchmark that stops the clock after issuing work
measures how long it took to *schedule* — a spectacular and meaningless number. Every
timing forces evaluation first.

**Interleaved.** mlxlearn and scikit-learn are timed in pairs rather than all of one then
all of the other, so a thermal ramp or a background process degrades both sides equally
instead of whichever ran second. Within a pair the two sides take turns going first —
A B, B A, A B — and every timed call gets the same preparation, whichever backend it runs
on: a garbage collection before it and a synchronization inside its window. A harness
that always timed mlxlearn first, straight after the collection, and made only that side
pay the synchronization read identical stock work in both slots as 0.77–0.85× and failed
the gate in 18–19 of 20 trials.

**An idle GPU is part of the cost.** The collection before every timed call idles the
GPU, and an MLX call made after that pays to bring it back up: it runs its first few
milliseconds of work at a lower clock. A 64-query neighbor search on 262 144 × 16 took
3.2 ms back to back and 5.5–8.3 ms after the GPU had idled for 10 ms — through a
collection, a sleep or busy host code; 32 queries, 2.0 against 5.0–5.1 ms; 256 queries,
8.6 against 9.1–9.4 ms (scikit-learn 1.9.0, median of 11). So the cost
is largest, relatively, for calls of a few milliseconds — which for the neighbor search
is a batch of tens to hundreds of queries against a mid-sized index, exactly where a
stock search costs about as much. A program that calls mlxlearn after doing other work
pays it, so the harness keeps it, and the thresholds below are set against it. A tight
loop of calls does not pay it, and runs faster than these tables say.

**Median with its spread, and best case too.** Both the median and the best-case ratio are
reported, because at small sizes a median alone can flatter either side; a crossover
requires a win on **both**.

**Significance.** A difference counts only when the two medians are three standard errors
apart, each standard error being that *of the median* — estimated from the interquartile
range as `1.2533 · (IQR / 1.349) / √n` — and combined in quadrature. Not the raw spread,
which for a heavy-tailed distribution declared a measured 10× "within noise"; and a real
interquartile range, not the full min–max range, so one outlier cannot hide a regression.
Fewer than five samples on either side is never significant. The `z = 3` rather than 1.96
is deliberate: the standard errors come from a handful of samples, and it keeps the
false-alarm rate per case near 1%. That rate is per case, and a nightly run gates well over
a hundred: simulated with 137 cases of identical work, 73–79% of nights would fail on one
measurement each. So the runner re-measures any case the gate would fail, over three
times the iterations, and records that as the result's `confirmation`; the gate fails the
case only if the loss reproduces there, which brings the simulated share of failing nights
to 0.4–0.7%. The report's `re-measured` column shows the confirmation.

**Patched dispatch is what is gated.** The performance claim is about the classes
`patch_sklearn()` installs, and those hand work to scikit-learn outright where it is
faster. The `smoke` and `full` profiles measure them by default, record which backend
each case actually ran on, and the gate judges those decisions against the configured
`Tuning` rather than a crossover inferred from the run it is judging. `--layer direct`
measures the Layer 1 classes, which never dispatch to scikit-learn and use their own CPU
path instead — useful for development, misleading as a headline. The calibration grids
force the patched classes onto their MLX path, which is what patched dispatch runs when a
threshold sends a problem there, so both sides of every threshold are on the record; the
published kernel tables force the Layer 1 classes. Both say what patched dispatch would
have done.

**Fixed hardware.** Absolute timings only compare across runs on the same machine. Shared
CI runners do correctness only; the nightly job pins itself to fixed self-hosted Apple
silicon.

## How patched dispatch decides

A threshold sends a problem to MLX only where MLX won, or tied within noise, against both
scikit-learn versions, in every configuration measured at that shape: every query batch
size from 1 to 4 096, on every law of data measured, for the neighbor estimators; every
kernel, `gamma`, degree, `coef0` and C from 0.1 to 100, on Gaussian data and on data of
lower intrinsic dimension, for `SVC`; binary and multinomial targets, with and without
signal, and `tol` and `max_iter`, for `LogisticRegression`. A fit cannot see how many
queries will follow it or how many iterations it will take, so where the answer turns on
those, the threshold takes the side that does not lose. Everything else goes to stock
scikit-learn. Some rules read more than the shape: a neighbor fit reads its `n_neighbors`
and probes its own rows once its index is built, and a query reads the `n_neighbors` it
asks for; `SVC` reads its kernel, C, scaled by the class and sample weights, and
`cache_size`; and `LogisticRegression` counts the classes and reads `tol` and `max_iter`.
Where a corner lost, the rules give up the whole region around it rather than grow a rule
per corner, so that each stays simple enough to state in a line; what that leaves
uncovered is [below](#known-limits-of-patched-dispatch).

| Estimator | Patched dispatch uses MLX when — all required | `Tuning` |
|---|---|---|
| neighbor estimators | at least 16 features, 4 096 rows and 6 000 000 elements (rows × features); with fewer than 256 features, at least 1 000 000 rows, and from 256 features at least 8 192; `n_neighbors` at most 50 — a query asking for more on an MLX-fitted model is answered by scikit-learn's brute force; and data the first search pass can certify — none of 512 training rows probed at fit uncertified, and, below 256 features, not integer-valued | `knn_min_features = 16`, `knn_min_train_samples = 4096`, `knn_min_work = 6_000_000`, `knn_wide_features = 256`, `knn_narrow_min_samples = 1_000_000`, `knn_wide_min_samples = 8192`, `knn_max_neighbors = 50`, `knn_probe_rows = 512`, `knn_probe_max_uncertified = 0.0` |
| `LogisticRegression` | at least 60 000 rows; multinomial: also at least 2 features; binary: also at least 200 000 rows and at most 64 features; `tol` at most scikit-learn's default, 1e-4, and `max_iter` at least 3 | `logreg_min_samples = 60_000`, `logreg_multinomial_min_features = 2`, `logreg_binary_min_samples = 200_000`, `logreg_binary_max_features = 64`, `logreg_min_max_iter = 3` |
| `SVC` | the rbf kernel, at least 256 features, 1 024 rows and 384 000 elements; with more than two classes, a mean one-vs-one pair `(2n/K) · d` of at least 384 000; C at most 1 — the mean box, C times the mean class and sample weight, to a relative 1e-9; `cache_size` at least scikit-learn's default, 200 MB | `svc_min_features = 256`, `svc_min_samples = 1024`, `svc_min_work = 384_000`, `svc_min_pair_work = 384_000`, `svc_max_c = 1.0` |

These feed `mlxlearn._common.config.Tuning`, which dispatch reads. That is the point: a
published crossover the dispatcher ignores is a claim, not a mechanism.

### Patched dispatch against stock: the gate

The `smoke` profile measures the classes `patch_sklearn()` installs on both sides of every
threshold. Run under both scikit-learn versions, 7 interleaved pairs each, it passed the
performance gate on both (`~` within noise, **bold** a significant loss; † a case the gate
would have failed, re-measured over 21 pairs as the runner does, and the cell is that
re-measure):

| case | patched dispatch | mlxlearn / stock, ms (1.9.0) | 1.9.0 | 1.7.2 |
|---|---|---|---|---|
| `knn/fit/1000x32/q256` | scikit-learn | 0.24 / 0.16 | **0.67×** | **0.72×** |
| `knn/kneighbors/1000x32/q256` | scikit-learn | 1.29 / 1.21 | 0.93×~ | 0.95×~ |
| `knn/fit/65536x256/q256` | MLX | 33.40 / 1.99 | **0.06×** | **0.05×** |
| `knn/kneighbors/65536x256/q256` | MLX | 6.13 / 21.09 | 3.44× | 3.90× |
| `knn/kneighbors/262144x64/q256` | scikit-learn | 25.01 / 20.69 | 0.83×~ | 1.00×~ |
| `knn/kneighbors/1048576x64/q256` | MLX | 18.01 / 73.01 | 4.05× | 4.42× |
| `knn/kneighbors/1048576x16/q256` | MLX | 13.17 / 44.66 | 3.39× | 3.61× |
| `knn/kneighbors/1048576x8/q256` | scikit-learn | 40.02 / 39.62 | 0.99×~ | 1.00×~ |
| `knn/kneighbors/1048576x8/q256/sklearn_algorithm=brute` | scikit-learn | 42.52 / 42.38 | 1.00×~ | 0.98×~ |
| `knn/kneighbors/50000x3/q256` | scikit-learn | 0.95 / 0.92 | 0.97×~ | 0.97×~ |
| `knn/kneighbors/24000x256/q256/data=dups` | scikit-learn | 10.63 / 11.60 | 1.09×~ | 0.95×~ |
| `knn/kneighbors/30000x256/q64/data=dup2` | scikit-learn | 2.99 / 3.31 | 1.10×~ | 0.93×~ |
| `knn/kneighbors/1048576x64/q256/data=binary` | scikit-learn | 71.65 / 71.86 | 1.00×~ | 1.02×~ |
| `knn/kneighbors/1048576x64/q1024/n_neighbors=100` | scikit-learn | 305.14 / 308.32 | 1.01×~ | 1.00×~ |
| `knn/kneighbors/65536x256/q256/query_neighbors=100` | scikit-learn | 28.18 / 28.05 | 1.00×~ | 1.04×~ |
| `knn/kneighbors/4096x2048/q32` | scikit-learn | 3.35 / 3.24 | 0.97×~ | 0.99×~ |
| `logreg/fit/5000x32` | scikit-learn | 1.52 / 1.44 | 0.95×~ | 0.92×~ |
| `logreg/fit/200000x128` | scikit-learn | 36.89 / 36.59 | 0.99×~ | 0.99× |
| `logreg/fit/100000x16` | scikit-learn | 13.90 / 13.71 | 0.99× | 0.99×~ |
| `logreg/fit/200000x16` | MLX | 10.68 / 27.95 | 2.62× | 2.41× |
| `logreg/fit/60000x64/n_classes=5` | MLX | 15.88 / 40.46 | 2.55× | 2.52× |
| `logreg/fit/60000x1/n_classes=3` | scikit-learn | 34.15 / 33.96 | 0.99×~ | 0.99× |
| `logreg/fit/200000x16/tol=0.001` | scikit-learn | 22.24 / 22.25 | 1.00×~ | 0.99×~ |
| `logreg/fit/200000x16/max_iter=2` | scikit-learn | 10.33 / 10.24 | 0.99×~ | 1.00×~ |
| `svc/fit/1000x32` | scikit-learn | 8.61 / 8.53 | 0.99× | 0.99× |
| `svc/fit/4000x64` | scikit-learn | 177.87 / 177.96 | 1.00×~ | 1.00×~ |
| `svc/fit/3000x128` | scikit-learn | 176.98 / 177.06 | 1.00×~ | 1.00×~ |
| `svc/fit/4000x256` | MLX | 158.26 / 591.63 | 3.74× | 3.47× |
| `svc/fit/2048x512/n_classes=4` | MLX | 29.95 / 63.18 | 2.11× | 1.96× |
| `svc/fit/2048x512/n_classes=16` | scikit-learn | 227.28 / 227.74 | 1.00×~ | 1.00×~ |
| `svc/fit/1500x256/C=10.0` | scikit-learn | 103.44 / 103.53 | 1.00×~ | 1.00×~ |
| `svc/fit/1500x256/kernel=poly` | scikit-learn | 90.58 / 90.74 | 1.00×~ | 1.00×~ |
| `svc/fit/1500x256/cache_size=1.0` | scikit-learn | 224.32 / 223.86 | 1.00×~ | 1.00×~ |
| `svc/predict/4000x256/q1` | MLX (host) | 0.24 / 0.38 | 1.60× | 1.41× |
| `svc/predict/4000x256/q1000` | MLX | 2.40 / 251.53 | 104.79× | 85.40× |

A handoff costs something of its own, a few hundredths of a millisecond for a neighbor
fit: a third to a half of a stock brute-force fit at 1 000 × 32 (0.24 against 0.16 ms on
1.9.0), which stores a reference and does nothing else. So patched `knn/fit` below the
floors reads 0.67×–0.72×, one more instance of the accepted `knn/fit` regression below.
The logistic and SVC handoffs were within noise of stock or within its 5%, but the
smallest sits at that edge: `logreg/fit/5000x32`, a fit of about 1.4 ms, read 0.95×~ on
1.9.0 and 0.92×~ on 1.7.2 in the table. The first 1.9.0 run of this table read it 0.91×,
and 0.94× re-measured, and failed the gate on that case alone — as runs of the previous
table did (0.90×, and 0.93× re-measured) — while the rerun shown, and earlier reruns,
read 0.94×~–0.97×~. Neither the fit nor its handoff changed in this release, and the
previous one's table read 0.99×~ and 0.97×~: the handoff's own few hundredths of a
millisecond are about 5% of a fit that short. `knn/kneighbors/262144x64/q256`, a handoff
that runs stock brute force against itself, read 0.83×~ and 1.00×~, noise. The narrow
corner the fused selection kernel opened, 1 048 576 × 16, runs on MLX (3.39×–3.61×);
below 16 features 1 048 576 × 8 goes to the KD-tree with `algorithm="auto"` and to stock
brute force with `algorithm="brute"`, both within noise of stock. The data handoffs —
rows repeated about 20 times at 24 000 × 256 and about twice at 30 000 × 256, and binary
features at 1 048 576 × 64, which the size rules alone would send to MLX — answered their
queries within noise of stock. So did the handoffs of the newer rules: 100 neighbors
asked of the fit at 1 048 576 × 64 and of a query on an index fitted with k = 10 at
65 536 × 256, 4 096 × 2 048 short of the wide-data row floor, a one-feature three-class
`LogisticRegression`, `tol = 1e-3` and `max_iter = 2` at 200 000 × 16, and `SVC` at 128
features, at C = 10, with the poly kernel at its defaults and with `cache_size = 1`.
`svc/predict/…/q1` is answered on the host, and the 1 000-row batch on the device.

## Neighbors

Every neighbor table here is `kneighbors` on a fitted model, k = 10, Gaussian data,
against stock scikit-learn — its brute force from 16 features, which is what
`algorithm="auto"` uses there, and its KD-tree below. Each cell is stock time over MLX
time, so above 1 MLX is faster. **Bold** is a loss by the gate's test (three standard
errors, and more than 5% slower); `~` is within noise. The fit column is the index build
against scikit-learn's fit, which for brute force stores a reference.

### Where MLX wins, and where it cannot

A fit cannot see how it will be queried, so the grids time twelve batch sizes, from 1 to
4 096 queries, against every index. Short of a million rows, mid-size batches lose below
256 features. Grid `knn`: 16 to 256 features, 12 000 to 1.6 million rows, both
versions; each cell is the worse version's ratio, a loss on either version shown as a
loss:

<!-- python -m benchmarks.calibrate table knn: scikit-learn 1.9.0 / 1.7.2, the worse of them -->
| rows × features | 1 query | 2 queries | 4 queries | 8 queries | 16 queries | 32 queries | 64 queries | 128 queries | 256 queries | 512 queries | 1024 queries | 4096 queries | fit | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 120 000 × 16 | **0.81** | 1.12 | 1.04~ | **0.90** | **0.69** | **0.55** | 1.26~ | 1.47 | 1.90 | 2.19 | 2.29 | 4.42 | **0.06** | scikit-learn |
| 180 000 × 16 | **0.88** | 1.29 | 1.14 | **0.95** | **0.70** | 1.19 | 1.35 | 1.73 | 2.19 | 2.30 | 2.76 | 4.88 | **0.06** | scikit-learn |
| 262 144 × 16 | 1.02~ | 1.57 | 1.43 | 1.18 | 1.45 | 1.36 | 1.62 | 1.95 | 2.34 | 2.84 | 3.55 | 5.79 | **0.06** | scikit-learn |
| 400 000 × 16 | 1.13 | 1.80 | 1.48 | 1.08 | 1.65 | 1.59 | 1.88 | 1.76 | 2.05 | 2.66 | 3.70 | 5.14 | **0.05** | scikit-learn |
| 600 000 × 16 | 1.29 | 2.10 | 1.66 | 2.20 | 1.95 | 1.80 | 1.78 | 2.04 | 2.20 | 3.12 | 4.14 | 5.45 | **0.05** | scikit-learn |
| 1 000 000 × 16 | 1.55 | 2.58 | 1.87 | 2.78 | 2.38 | 2.12 | 1.88 | 2.37 | 3.32 | 4.24 | 4.85 | 6.08 | **0.05** | MLX |
| 1 600 000 × 16 | 1.85 | 3.07 | 3.63 | 3.37 | 2.82 | 1.74 | 2.19 | 3.10 | 3.64 | 4.71 | 5.49 | 5.70 | **0.05** | MLX |
| 80 000 × 24 | **0.81** | 0.90~ | **0.87** | **0.78** | **0.68** | **0.59** | 1.20 | 1.38 | 1.57 | 1.68 | 2.04 | 3.51 | **0.07** | scikit-learn |
| 120 000 × 24 | **0.79** | 1.07~ | 0.96~ | **0.87** | **0.68** | **0.56** | 1.29 | 1.59 | 1.77 | 2.26 | 2.46 | 4.50 | **0.06** | scikit-learn |
| 180 000 × 24 | **0.93** | 1.21 | 1.06 | **0.90** | **0.67** | 1.27 | 1.49 | 1.77 | 1.98 | 2.41 | 3.19 | 5.21 | **0.06** | scikit-learn |
| 262 144 × 24 | 1.11~ | 1.44 | 1.30 | 1.11 | 1.35 | 1.44 | 1.57 | 1.97 | 2.16 | 2.95 | 3.84 | 6.18 | **0.06** | scikit-learn |
| 400 000 × 24 | 1.12 | 1.56 | 1.33 | 1.00~ | 1.52 | 1.53 | 1.97 | 1.58 | 1.94 | 2.91 | 3.96 | 5.71 | **0.06** | scikit-learn |
| 600 000 × 24 | 1.33 | 1.80 | 1.48 | 1.92 | 1.75 | 1.75 | 1.77 | 1.93 | 3.61 | 3.17 | 4.25 | 5.94 | **0.06** | scikit-learn |
| 1 000 000 × 24 | 1.68 | 2.10 | 1.63 | 2.25 | 1.96 | 2.05 | 1.73 | 2.20 | 3.28 | 4.54 | 5.58 | 6.67 | **0.06** | MLX |
| 1 600 000 × 24 | 1.90 | 2.36 | 2.72 | 2.60 | 2.26 | 1.57 | 2.03 | 3.14 | 3.51 | 4.66 | 5.59 | — | **0.06** | MLX |
| 80 000 × 32 | **0.78** | 0.96~ | **0.88** | 0.88~ | **0.71** | **0.61** | 1.28 | 1.47 | 1.65 | 1.76 | 2.20 | 3.77 | **0.07** | scikit-learn |
| 120 000 × 32 | **0.83** | 1.12 | 1.03~ | **0.90** | **0.71** | **0.59** | 1.36 | 1.69 | 1.90 | 2.41 | 2.57 | 5.32 | **0.06** | scikit-learn |
| 180 000 × 32 | 0.93~ | 1.27 | 1.11 | 0.96~ | **0.67** | 1.34 | 1.62 | 1.90 | 2.15 | 2.64 | 3.61 | 5.64 | **0.06** | scikit-learn |
| 262 144 × 32 | 1.10~ | 1.52 | 1.36 | 1.14 | 1.38 | 1.49 | 1.90 | 2.11 | 2.53 | 2.97 | 4.07 | 6.54 | **0.06** | scikit-learn |
| 400 000 × 32 | 1.28 | 1.63 | 1.39 | 1.08 | 1.60 | 1.72 | 2.08 | 1.81 | 2.19 | 3.11 | 4.04 | 6.16 | **0.06** | scikit-learn |
| 600 000 × 32 | 1.41 | 1.86 | 1.54 | 2.00 | 1.85 | 1.98 | 1.98 | 2.06 | 2.41 | 3.71 | 4.82 | 6.05 | **0.06** | scikit-learn |
| 1 000 000 × 32 | 1.82 | 2.13 | 1.69 | 2.29 | 2.09 | 2.24 | 1.86 | 2.45 | 3.30 | 4.44 | 5.58 | — | **0.06** | MLX |
| 1 600 000 × 32 | 2.06 | 2.40 | 2.65 | 2.57 | 2.33 | 1.69 | 2.44 | 3.37 | 4.57 | 4.99 | 5.91 | — | **0.06** | MLX |
| 50 000 × 48 | **0.83** | **0.92** | **0.81** | **0.78** | **0.71** | **0.70** | **0.62** | 1.35 | 1.67 | 2.07 | 2.07 | 3.13 | **0.07** | scikit-learn |
| 80 000 × 48 | **0.87** | 0.92~ | 0.94~ | **0.82** | **0.75** | **0.68** | 1.43 | 1.60 | 1.93 | 1.84 | 2.32 | 4.41 | **0.06** | scikit-learn |
| 120 000 × 48 | **0.93** | 1.10 | 1.04~ | **0.90** | **0.77** | **0.71** | 1.59 | 1.70 | 2.07 | 2.41 | 3.15 | 6.03 | **0.06** | scikit-learn |
| 180 000 × 48 | 1.07 | 1.22 | 1.11 | 0.96~ | **0.79** | 1.46 | 1.79 | 1.96 | 2.34 | 2.77 | 3.63 | 7.18 | **0.06** | scikit-learn |
| 262 144 × 48 | 1.33 | 1.49 | 1.32 | 1.12 | 1.49 | 1.60 | 1.94 | 2.25 | 2.46 | 3.08 | 4.94 | 7.59 | **0.06** | scikit-learn |
| 400 000 × 48 | 1.55 | 1.61 | 1.34 | 1.09 | 1.59 | 1.82 | 2.30 | 1.87 | 2.30 | 3.44 | 4.87 | 7.56 | **0.06** | scikit-learn |
| 600 000 × 48 | 1.82 | 1.75 | 1.43 | 1.89 | 1.74 | 1.97 | 2.08 | 2.08 | 3.01 | 3.77 | 4.84 | 7.18 | **0.06** | scikit-learn |
| 1 000 000 × 48 | 2.11 | 1.83 | 1.61 | 2.16 | 1.97 | 2.47 | 2.28 | 2.97 | 4.49 | 5.62 | 7.07 | — | **0.06** | MLX |
| 1 600 000 × 48 | 2.31 | 2.07 | 2.26 | 2.35 | 2.33 | 1.71 | 2.58 | 3.93 | 4.87 | 6.07 | 7.21 | — | **0.06** | MLX |
| 33 000 × 64 | **0.76** | **0.79** | **0.78** | **0.76** | **0.71** | **0.72** | **0.74** | 1.32 | 1.54 | 1.89 | 2.41 | 3.43 | **0.08** | scikit-learn |
| 50 000 × 64 | **0.83** | **0.90** | **0.87** | **0.79** | **0.74** | **0.70** | **0.66** | 1.49 | 1.70 | 2.15 | 2.24 | 2.66 | **0.07** | scikit-learn |
| 80 000 × 64 | **0.88** | 1.00~ | **0.90** | **0.87** | **0.77** | **0.70** | 1.40 | 1.74 | 1.95 | 1.90 | 2.56 | 5.16 | **0.06** | scikit-learn |
| 120 000 × 64 | 1.02~ | 1.14 | 1.01~ | **0.90** | **0.82** | **0.69** | 1.64 | 1.87 | 2.13 | 2.50 | 3.33 | 7.05 | **0.06** | scikit-learn |
| 180 000 × 64 | 1.21 | 1.26 | 1.16 | 0.98~ | **0.81** | 1.48 | 1.88 | 2.09 | 2.37 | 3.13 | 5.09 | 8.31 | **0.06** | scikit-learn |
| 262 144 × 64 | 1.47 | 1.52 | 1.27 | 1.19 | 1.51 | 1.79 | 2.13 | 2.32 | 2.79 | 3.58 | 5.70 | 10.47 | **0.06** | scikit-learn |
| 400 000 × 64 | 1.68 | 1.56 | 1.30 | 1.01~ | 1.54 | 1.92 | 2.33 | 1.83 | 3.09 | 4.37 | 6.13 | 8.73 | **0.06** | scikit-learn |
| 600 000 × 64 | 1.97 | 1.66 | 1.41 | 1.82 | 1.89 | 2.14 | 2.25 | 2.38 | 3.13 | 4.57 | 7.13 | — | **0.06** | scikit-learn |
| 1 000 000 × 64 | 2.28 | 1.65 | 1.46 | 1.98 | 1.89 | 2.48 | 2.19 | 3.30 | 4.14 | 6.52 | 8.50 | — | **0.06** | MLX |
| 1 600 000 × 64 | 2.62 | 1.95 | 2.07 | 2.03 | 2.17 | 3.45 | 2.78 | 3.99 | 5.17 | 6.56 | 7.96 | — | **0.06** | MLX |
| 20 000 × 96 | **0.75** | **0.82** | **0.76** | **0.73** | **0.75** | **0.80** | **0.86** | **0.72** | 1.53 | 1.86 | 2.61 | 2.90 | **0.08** | scikit-learn |
| 33 000 × 96 | **0.79** | **0.80** | **0.84** | **0.80** | **0.81** | **0.86** | **0.78** | 1.49 | 1.68 | 2.32 | 2.97 | 3.40 | **0.08** | scikit-learn |
| 50 000 × 96 | 0.91~ | 0.95~ | 0.94~ | **0.87** | **0.81** | **0.73** | **0.79** | 1.59 | 2.02 | 2.34 | 2.60 | 4.18 | **0.07** | scikit-learn |
| 80 000 × 96 | 1.05~ | 1.09~ | 1.04~ | 0.94~ | **0.90** | **0.84** | 1.61 | 1.91 | 2.48 | 2.59 | 3.91 | 6.61 | **0.07** | scikit-learn |
| 120 000 × 96 | 1.27 | 1.21 | 1.09 | 0.96~ | **0.91** | **0.78** | 1.82 | 2.17 | 2.75 | 3.62 | 5.66 | 9.42 | **0.06** | scikit-learn |
| 180 000 × 96 | 1.43 | 1.37 | 1.26 | 1.07~ | **0.91** | 1.51 | 2.07 | 2.79 | 3.30 | 4.82 | 5.73 | 10.83 | **0.07** | scikit-learn |
| 262 144 × 96 | 1.78 | 1.48 | 1.44 | 1.28 | 1.54 | 1.89 | 2.57 | 2.89 | 3.48 | 5.70 | 8.77 | 12.65 | **0.07** | scikit-learn |
| 400 000 × 96 | 2.06 | 1.52 | 1.42 | 1.11 | 1.64 | 1.95 | 2.79 | 2.41 | 4.71 | 6.47 | 8.80 | — | **0.07** | scikit-learn |
| 600 000 × 96 | 2.23 | 1.59 | 1.38 | 1.69 | 1.69 | 2.20 | 2.49 | 3.37 | 5.25 | 7.32 | 10.27 | — | **0.06** | scikit-learn |
| 1 000 000 × 96 | 2.50 | 1.73 | 1.54 | 1.87 | 1.84 | 2.39 | 2.80 | 4.51 | 6.45 | 9.19 | 12.13 | — | **0.06** | MLX |
| 20 000 × 128 | **0.73** | **0.85** | **0.81** | **0.80** | **0.83** | **0.82** | **0.89** | **0.77** | 1.52 | 2.30 | 3.12 | 3.49 | **0.08** | scikit-learn |
| 33 000 × 128 | **0.85** | **0.92** | **0.90** | **0.83** | **0.91** | **0.84** | **0.85** | 1.53 | 2.01 | 2.72 | 3.77 | 3.86 | **0.07** | scikit-learn |
| 50 000 × 128 | 1.05~ | 1.04~ | 0.96~ | **0.93** | 0.95~ | 0.90~ | **0.83** | 1.74 | 2.25 | 3.07 | 3.36 | 5.09 | **0.07** | scikit-learn |
| 80 000 × 128 | 1.19 | 1.19 | 1.10 | 1.01~ | 0.97~ | 0.95~ | 1.78 | 2.17 | 2.79 | 3.01 | 4.53 | 7.99 | **0.07** | scikit-learn |
| 120 000 × 128 | 1.36 | 1.31 | 1.18 | 1.11 | 0.99~ | **0.90** | 2.11 | 2.79 | 3.41 | 4.12 | 6.31 | 11.82 | **0.07** | scikit-learn |
| 180 000 × 128 | 1.66 | 1.48 | 1.35 | 1.17 | 1.02~ | 1.74 | 2.74 | 3.26 | 3.43 | 5.83 | 8.62 | 11.17 | **0.07** | scikit-learn |
| 262 144 × 128 | 1.94 | 1.59 | 1.48 | 1.23 | 1.58 | 2.14 | 2.90 | 3.48 | 4.58 | 7.12 | 10.19 | — | **0.07** | scikit-learn |
| 400 000 × 128 | 2.32 | 1.64 | 1.53 | 1.17 | 1.68 | 2.36 | 3.39 | 2.97 | 5.62 | 7.41 | 9.70 | — | **0.07** | scikit-learn |
| 600 000 × 128 | 2.62 | 1.66 | 1.56 | 1.58 | 1.78 | 2.46 | 3.16 | 4.31 | 5.81 | 8.65 | 11.41 | — | **0.07** | scikit-learn |
| 1 000 000 × 128 | 3.16 | 1.89 | 1.56 | 1.81 | 2.09 | 2.78 | 3.57 | 5.66 | 8.14 | 10.94 | — | — | **0.07** | MLX |
| 12 000 × 256 | **0.86** | **0.85** | **0.88** | **0.91** | **0.93** | 0.99~ | 1.01~ | 0.96~ | 1.23~ | 2.26 | 3.29 | 3.33 | **0.08** | scikit-learn |
| 20 000 × 256 | **0.90** | 1.01~ | 0.99~ | 0.94~ | 0.97~ | 1.03~ | 1.16~ | 0.99~ | 2.42 | 2.84 | 4.08 | 4.16 | **0.07** | scikit-learn |
| 33 000 × 256 | 1.11 | 1.15 | 1.16 | 1.08~ | 1.15 | 1.14 | 1.15~ | 2.20 | 2.76 | 3.66 | 4.64 | 7.38 | **0.07** | MLX |
| 50 000 × 256 | 1.35 | 1.22 | 1.24 | 1.17 | 1.15 | 1.07~ | 1.41 | 2.82 | 3.08 | 3.94 | 4.18 | 7.20 | **0.07** | MLX |
| 80 000 × 256 | 1.63 | 1.45 | 1.30 | 1.25 | 1.23 | 1.18 | 2.73 | 3.05 | 3.81 | 4.94 | 6.64 | 10.81 | **0.07** | MLX |
| 120 000 × 256 | 2.03 | 1.58 | 1.50 | 1.29 | 1.26 | 1.44 | 3.75 | 4.00 | 5.05 | 6.00 | 8.57 | — | **0.07** | MLX |
| 180 000 × 256 | 2.37 | 1.79 | 1.64 | 1.54 | 1.37 | 2.72 | 3.78 | 3.91 | 4.82 | 7.94 | 11.46 | — | **0.07** | MLX |
| 262 144 × 256 | 2.97 | 1.85 | 1.75 | 1.74 | 2.04 | 2.74 | 3.78 | 4.55 | 6.03 | 10.15 | 12.89 | — | **0.07** | MLX |
| 400 000 × 256 | 3.26 | 2.02 | 1.73 | 1.62 | 2.20 | 3.06 | 5.10 | 4.43 | 6.85 | 10.06 | 12.97 | — | **0.07** | MLX |

Three things decide it, and a fit can see only two of them.

- **Width.** Stock brute force pays for every feature of every training row on every
  query; MLX's search costs about the same per query at 16 features as at 256, and what
  it pays on top is fixed — a launch, and the wake-up of an idled GPU. At 256 features
  the stock search is expensive enough to cover that from 33 000 rows on, the first size
  here the work floor lets through (12 000 and 20 000 rows lost single rows and small
  batches, 0.85×–0.93×); at 16–128 features it is not, until the index is large.
- **Rows.** Below 256 features, some batch of 1–128 queries lost on one version or both
  at 27 of the 50 sizes measured below 1 000 000 rows, down to 0.55×, every one of them at
  180 000 rows or fewer. From 262 144 rows this table shows every batch winning or tying,
  but not every batch that matters: below 2²² (query row, training row) pairs a query
  runs MLX's own selection, not the fused kernel, and the largest such batch, which this
  table's powers of two step over, lost at 262 144 × 16 (`knn-boundary`, below). From
  1 000 000 rows every batch won or tied here and, at every k up to 50 and on every law
  the fit-time probe passes, in `knn-narrow` and the grids after it. Large batches never
  lost: 1 024 queries or more won at every size in this table (2.0×–15×), which is why a
  workload known to query in large batches should import `mlxlearn.neighbors` directly.
- **The query batch.** It is the one of the three a fit cannot see, and the rules are set
  so that it does not matter: every shape patched dispatch sends to MLX won or tied at
  every batch size, on both versions.

Every batch that lost below a million rows here was one MLX's own selection answers,
below 2²² pairs, where a call made after the GPU had idled cost MLX 1.1–4.2 ms and stock
brute force 0.9–3.9 ms. Grid `knn-boundary` times the largest such batch — the most
queries that stay under 2²² pairs — between this table's 262 144 and 1 000 000 rows:

<!-- python -m benchmarks.calibrate table knn-boundary: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | 6 queries | 8 queries | 10 queries | 15 queries | dispatch |
|---|---|---|---|---|---|
| 262 144 × 16 | — | 1.14 / 1.19 | — | **0.90** / **0.91** | scikit-learn |
| 400 000 × 16 | — | 1.13 / 1.13 | 0.99~ / 0.99~ | — | scikit-learn |
| 600 000 × 16 | 1.42 / 2.05 | 2.27 / 2.33 | — | — | scikit-learn |
| 262 144 × 48 | — | 1.18~ / 1.86 | — | 0.98~ / 1.02~ | scikit-learn |
| 400 000 × 48 | — | 1.13 / 1.89 | 1.02~ / 1.02~ | — | scikit-learn |
| 600 000 × 48 | 1.27 / 3.10 | 1.84 / 3.19 | — | — | scikit-learn |
| 262 144 × 128 | — | 1.33 / 2.13 | — | 1.15 / 1.21 | scikit-learn |
| 400 000 × 128 | — | 1.26 / 1.24 | 1.39 / 1.33 | — | scikit-learn |
| 600 000 × 128 | 1.35 / 1.38~ | 1.73 / 1.68 | — | — | scikit-learn |

Fifteen queries lost at 262 144 × 16 on both versions (0.90×–0.91×) and only tied at
48 features; ten at 400 000 rows only tied at 16 and 48 features (0.99×~–1.02×~); six,
at 600 000 rows, won or tied at every width (1.27×–3.1×). The million-row floor stays:
between 262 144 and 1 000 000 rows only Gaussian data at k = 10 was measured, and the
rules move only where every law and k measured won or tied. Answering mid-size batches
somewhere other than the fitted backend would re-decide the backend at inference, which
the sticky-backend contract rules out; a search that keeps the GPU from idling, or hides
its wake-up, is future work.

At a million rows the narrow widths were measured again, on Gaussian data and on two other
laws the fit-time probe (below) passes — 100 clusters of std 1, and uniform — with
everything else idle. Grid `knn-narrow`, k = 10:

<!-- python -m benchmarks.calibrate table knn-narrow: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, data | 1 query | 4 queries | 16 queries | 64 queries | 256 queries | 1024 queries | probe | dispatch |
|---|---|---|---|---|---|---|---|---|
| 1 000 000 × 16, gauss | 2.04 / 2.09 | 1.95 / 3.63 | 2.49 / 2.40 | 1.97 / 1.85 | 3.65 / 3.28 | 5.23 / 5.11 | 0/512 | MLX |
| 1 000 000 × 16, blobs | 2.23 / 2.17 | 1.93 / 1.92 | 2.53 / 2.44 | 1.98 / 1.87 | 4.66 / 3.35 | 5.78 / 5.26 | 0/512 | MLX |
| 1 000 000 × 16, uniform | 1.63 / 1.99 | 1.91 / 1.88 | 2.58 / 2.49 | 2.00 / 1.87 | 3.71 / 3.27 | 5.63 / 5.37 | 0/512 | MLX |
| 1 000 000 × 32, gauss | 1.89 / 2.49 | 1.73 / 1.75 | 2.25 / 2.17 | 1.95 / 1.87 | 3.75 / 3.28 | 6.09 / 5.27 | 0/512 | MLX |
| 1 000 000 × 32, blobs | 1.89 / 1.91 | 1.71 / 1.75 | 2.27 / 2.21 | 2.08 / 1.87 | 3.49 / 3.32 | 6.03 / 5.64 | 0/512 | MLX |
| 1 000 000 × 32, uniform | 2.04 / 2.50 | 1.74 / 1.71 | 2.20 / 2.19 | 1.99 / 1.94 | 3.82 / 3.51 | 6.08 / 5.79 | 0/512 | MLX |
| 1 600 000 × 32, gauss | 2.32 / 3.11 | 2.80 / 2.88 | 2.77 / 2.47 | 2.49 / 2.16 | 4.17 / 4.17 | 5.95 / 5.63 | 0/512 | MLX |
| 1 600 000 × 32, blobs | 2.28 / 3.10 | 3.02 / 2.81 | 2.55 / 2.44 | 2.61 / 2.51 | 4.06 / 4.30 | 6.04 / 5.65 | 0/512 | MLX |
| 1 600 000 × 32, uniform | 2.12 / 3.02 | 2.82 / 2.74 | 2.65 / 2.59 | 2.62 / 2.55 | 4.39 / 4.07 | 6.01 / 5.83 | 0/512 | MLX |
| 1 000 000 × 48, gauss | 2.27 / 2.83 | 1.58 / 3.33 | 2.30 / 2.15 | 2.16 / 2.10 | 3.76 / 3.77 | 6.92 / 6.39 | 0/512 | MLX |
| 1 000 000 × 48, blobs | 2.00 / 2.21 | 1.61 / 1.59 | 2.18 / 2.29 | 2.17 / 2.13 | 4.17 / 3.93 | 7.13 / 6.96 | 0/512 | MLX |
| 1 000 000 × 48, uniform | 2.17 / 2.91 | 1.55 / 1.66 | 2.17 / 2.10 | 2.07 / 2.26 | 4.06 / 3.74 | 7.51 / 7.18 | 0/512 | MLX |
| 1 000 000 × 64, gauss | 2.53 / 2.90 | 1.51 / 1.52 | 2.09 / 2.07 | 2.21 / 2.26 | 4.35 / 4.36 | 8.42 / 7.42 | 0/512 | MLX |
| 1 000 000 × 64, blobs | 2.62 / 3.05 | 1.50 / 1.45 | 2.24 / 2.30 | 2.57 / 2.10 | 5.43 / 4.39 | 8.25 / 7.34 | 0/512 | MLX |
| 1 000 000 × 64, uniform | 3.20 / 2.35 | 1.50 / 1.44 | 2.39 / 2.05 | 2.35 / 2.19 | 4.30 / 4.42 | 7.93 / 7.90 | 0/512 | MLX |
| 1 600 000 × 64, gauss | 2.77 / 2.62 | 2.32 / 2.21 | 2.35 / 2.25 | 4.84 / 5.13 | 5.77 / 4.81 | 7.24 / 6.89 | 0/512 | MLX |
| 1 600 000 × 64, blobs | 3.61 / 2.62 | 2.22 / 2.14 | 2.34 / 2.26 | 3.04 / 4.99 | 4.69 / 4.77 | 7.07 / 6.99 | 0/512 | MLX |
| 1 600 000 × 64, uniform | 2.70 / 2.65 | 2.13 / 2.17 | 2.24 / 2.25 | 3.36 / 3.12 | 4.25 / 4.56 | 7.55 / 7.10 | 0/512 | MLX |
| 1 000 000 × 96, gauss | 2.57 / 2.62 | 1.56 / 1.52 | 2.11 / 1.99 | 3.01 / 2.81 | 7.53 / 6.00 | 10.39 / 11.47 | 0/512 | MLX |
| 1 000 000 × 96, blobs | 2.75 / 3.05 | 1.53 / 1.50 | 2.03 / 1.93 | 4.13 / 3.00 | 6.88 / 6.03 | 10.07 / 11.76 | 0/512 | MLX |
| 1 000 000 × 96, uniform | 2.55 / 3.60 | 1.54 / 1.51 | 2.06 / 2.01 | 3.27 / 3.87 | 6.63 / 6.72 | 12.91 / 12.58 | 0/512 | MLX |
| 1 000 000 × 128, gauss | 3.78 / 3.30 | 1.54 / 1.73 | 2.32 / 1.94 | 2.91 / 3.05 | 6.93 / 5.97 | — | 0/512 | MLX |
| 1 000 000 × 128, blobs | 3.21 / 3.30 | 1.68 / 1.59 | 2.39 / 1.93 | 2.89 / 3.69 | 7.25 / 7.77 | — | 0/512 | MLX |
| 1 000 000 × 128, uniform | 3.16 / 3.24 | 1.70 / 1.63 | 2.07 / 2.07 | 3.28 / 3.64 | 7.34 / 8.16 | — | 0/512 | MLX |

Measured while MLX's own selection picked each tile's candidates, 16–48 features lost some
batch here on one version or both at every width but 1 600 000 × 32 — 16 queries at
0.73×–0.84×, where a call made after the GPU had idled cost MLX about 10–12 ms against
8–10 ms for stock brute force, and 1 024 at 0.62×–0.79× — and patched dispatch handed
every fit with fewer than 64 features to scikit-learn. The table above is the same grid
re-measured with the fused selection kernel ([`fp32_policy.md`](fp32_policy.md) §2.1),
which made those batches several times faster: at 1 000 000 × 16, 16 queries now take MLX
about 3 ms against 7–8 ms for stock, and 1 024 about 33 ms against 180 ms, where they took
about 260 ms before. Every batch won on both versions, 1.44× at worst (4 queries at
1 000 000 × 64 on 1.7.2) and 1.55× at 16–48 features. At the same shapes the other laws
of `knn-data` and `knn-structured` and k = 1–50 in `knn-k` won or tied too, wherever the
fit-time probe below passed the data (1.26×~ at worst; tables below). An A/B of the move
itself — patched queries at 1 000 000 and 1 600 000 × 16, 24, 32 and 48 on Gaussian and
std-1 cluster data, k = 5, 10 and 50, 1–4 096 queries, after an idle gap on both
versions, against stock brute force, which had answered them before — found every one of
its 336 configurations significantly faster, 1.33× at worst (clusters at
1 000 000 × 24, k = 50; see the CHANGELOG). So patched dispatch
now hands a fit to scikit-learn whatever its rows only below 16 features
(`knn_min_features = 16`): to its KD-tree where it would build one, to its brute force
otherwise ([below](#narrow-data-goes-to-scikit-learn-whatever-its-size)). From 16 features
the million-row floor decides. What the move costs is the index build — a patched fit at
1 000 000 × 16–48 and 1 600 000 × 32, the fit-time probe included, took 47–132 ms
against 1.9–6.0 ms for stock's (0.04×–0.06×, after an idle gap, medians of 9, both
versions) — and device memory, three times the float32 data for a float64 index.

**k.** The tables above are k = 10. Grid `knn-k` times k from 1 to 200 at a million rows
and more from 16 to 128 features and on short, wide training sets up to 4 096 features —
wider than any other grid — every batch size from 1 to 4 096; each cell is the worse
version's ratio. The rows at 16–48 features were measured with the fused selection
kernel, the rest before it:

<!-- python -m benchmarks.calibrate table knn-k: scikit-learn 1.9.0 / 1.7.2, the worse of them -->
| rows × features, k | 1 query | 2 queries | 4 queries | 8 queries | 16 queries | 32 queries | 64 queries | 128 queries | 256 queries | 512 queries | 1024 queries | 4096 queries | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 000 000 × 16, k = 1 | 1.54 | 2.58 | 1.99 | 3.08 | 2.52 | 2.18 | 1.94 | 2.58 | 3.43 | 4.04 | 4.98 | 5.75 | MLX |
| 1 000 000 × 16, k = 5 | 1.64 | 2.62 | 1.96 | 2.92 | 2.50 | 2.28 | 1.86 | 2.51 | 3.31 | 4.46 | 5.20 | 6.36 | MLX |
| 1 000 000 × 16, k = 10 | 1.65 | 2.57 | 1.88 | 2.86 | 2.40 | 2.20 | 1.91 | 2.48 | 3.12 | 4.05 | 5.23 | 6.22 | MLX |
| 1 000 000 × 16, k = 50 | 1.41 | 2.36 | 1.67 | 1.83 | 1.57 | 1.59 | 1.44 | 2.91 | 2.72 | 3.36 | 3.90 | 4.40 | MLX |
| 1 000 000 × 16, k = 100 | 1.36 | 2.11 | 1.45 | 0.95 | 0.76~ | 0.86~ | 0.94~ | 0.85~ | **0.71** | **0.56** | **0.54** | **0.72** | scikit-learn |
| 1 000 000 × 16, k = 200 | 1.22 | 2.06 | 1.41 | **0.89** | 1.15 | 0.88~ | 0.97~ | **0.76** | **0.78** | **0.82** | **0.81** | **0.85** | scikit-learn |
| 1 000 000 × 32, k = 1 | 1.77 | 2.23 | 1.73 | 2.51 | 2.22 | 2.32 | 1.90 | 2.40 | 3.44 | 4.76 | 5.70 | — | MLX |
| 1 000 000 × 32, k = 5 | 1.97 | 2.23 | 1.64 | 2.36 | 2.20 | 2.33 | 1.97 | 2.46 | 3.44 | 4.47 | 6.07 | — | MLX |
| 1 000 000 × 32, k = 10 | 1.84 | 2.23 | 1.73 | 2.31 | 2.19 | 2.33 | 1.87 | 2.35 | 3.36 | 4.61 | 5.77 | — | MLX |
| 1 000 000 × 32, k = 50 | 1.67 | 1.98 | 1.52 | 1.69 | 1.58 | 1.71 | 1.56 | 2.43 | 2.94 | 3.77 | 4.86 | — | MLX |
| 1 000 000 × 32, k = 100 | 1.65 | 1.84 | 1.40 | **0.92** | 1.33 | 0.99~ | 1.08~ | 0.80~ | **0.72** | **0.77** | **0.78** | — | scikit-learn |
| 1 000 000 × 32, k = 200 | 1.49 | 1.73 | 1.30 | **0.85** | 1.17~ | 1.03~ | 0.98~ | 0.88~ | **0.83** | **0.90** | **0.93** | — | scikit-learn |
| 1 600 000 × 32, k = 1 | 2.16 | 2.51 | 2.90 | 2.78 | 2.57 | 1.83 | 2.84 | 3.26 | 4.09 | 5.01 | 6.15 | — | MLX |
| 1 600 000 × 32, k = 5 | 2.05 | 2.47 | 2.86 | 2.66 | 2.47 | 1.85 | 2.45 | 3.71 | 4.46 | 5.06 | 5.99 | — | MLX |
| 1 600 000 × 32, k = 10 | 2.15 | 2.43 | 2.77 | 2.56 | 2.56 | 1.59 | 2.23 | 3.20 | 4.26 | 5.30 | 6.03 | — | MLX |
| 1 600 000 × 32, k = 50 | 1.91 | 2.19 | 2.06 | 1.93 | 1.26~ | 1.43~ | 2.02 | 2.75 | 3.57 | 4.10 | 4.98 | — | MLX |
| 1 600 000 × 32, k = 100 | 1.78 | 2.11 | 1.56 | 1.08~ | 1.58 | 1.29~ | 0.93~ | 0.89~ | **0.86** | **0.76** | **0.85** | — | scikit-learn |
| 1 600 000 × 32, k = 200 | 1.78 | 1.94 | 1.38 | 2.07~ | 1.20~ | 1.21~ | 0.88~ | **0.83** | **0.82** | **0.76** | **0.75** | — | scikit-learn |
| 1 000 000 × 48, k = 1 | 2.24 | 2.00 | 1.61 | 2.28 | 2.19 | 2.55 | 2.17 | 3.04 | 4.04 | 5.38 | 6.64 | — | MLX |
| 1 000 000 × 48, k = 5 | 2.15 | 1.95 | 1.62 | 2.17 | 2.14 | 2.50 | 1.97 | 2.94 | 3.96 | 5.27 | 6.89 | — | MLX |
| 1 000 000 × 48, k = 10 | 2.11 | 1.83 | 1.60 | 2.17 | 2.02 | 2.58 | 2.16 | 3.01 | 4.08 | 5.58 | 6.54 | — | MLX |
| 1 000 000 × 48, k = 50 | 1.96 | 1.76 | 1.40 | 1.58 | 1.56 | 1.93 | 1.70 | 2.76 | 3.47 | 4.25 | 4.97 | — | MLX |
| 1 000 000 × 48, k = 100 | 1.90 | 1.70 | 1.34 | **0.93** | 1.44~ | 1.22~ | 1.21~ | 0.83~ | **0.86** | 0.93~ | **0.80** | — | scikit-learn |
| 1 000 000 × 48, k = 200 | 1.74 | 1.57 | 1.26 | **0.89** | 1.39~ | 1.28~ | 1.05~ | 0.88~ | 0.96~ | **0.91** | 1.04~ | — | scikit-learn |
| 1 000 000 × 64, k = 1 | 2.40 | 1.84 | 1.48 | 1.21 | 2.15~ | 1.81 | 1.67~ | 1.37 | 1.48 | 1.74 | 1.39 | — | MLX |
| 1 000 000 × 64, k = 5 | 2.35 | 1.75 | 1.46 | 1.31~ | 1.00~ | 1.80 | 1.84 | 1.44 | 1.48 | 1.67 | 1.29 | — | MLX |
| 1 000 000 × 64, k = 10 | 2.31 | 1.81 | 1.47 | 1.18 | 0.98~ | 1.68 | 1.15~ | 1.41 | 1.49 | 1.69 | 1.16~ | — | MLX |
| 1 000 000 × 64, k = 50 | 2.11 | 1.64 | 1.39 | 1.09 | 1.06~ | 1.64 | 1.42~ | 1.24~ | 1.31 | 1.33 | 1.00~ | — | MLX |
| 1 000 000 × 64, k = 100 | 2.04 | 1.55 | 1.27 | 1.03~ | 1.30~ | 1.62~ | 1.10~ | 1.17~ | 1.33 | 0.97~ | **0.80** | — | scikit-learn |
| 1 000 000 × 64, k = 200 | 1.98 | 1.64 | 1.35 | 1.04~ | 1.61~ | 1.53~ | 1.18~ | 1.14~ | 1.23 | 0.96~ | **0.80** | — | scikit-learn |
| 1 600 000 × 64, k = 1 | 2.73 | 1.98 | 1.73 | 1.37~ | 1.86 | 1.55 | 1.55 | 1.60 | 1.69 | 1.84 | 1.40 | — | MLX |
| 1 600 000 × 64, k = 5 | 2.71 | 1.95 | 1.66 | 1.45~ | 2.44 | 1.94 | 1.52 | 1.62 | 1.87 | 1.74 | 1.42 | — | MLX |
| 1 600 000 × 64, k = 10 | 2.63 | 1.92 | 1.71 | 1.34 | 2.50 | 2.08 | 1.54 | 1.65 | 1.73 | 1.66 | 1.25 | — | MLX |
| 1 600 000 × 64, k = 50 | 2.55 | 1.93 | 1.75 | 1.26~ | 1.65 | 1.88 | 1.55 | 1.59 | 1.71 | 1.24 | 1.02~ | — | MLX |
| 1 600 000 × 64, k = 100 | 2.30 | 1.79 | 1.62 | 1.23~ | 1.85 | 1.36~ | 1.31~ | 1.45 | 1.31 | 0.87~ | **0.80** | — | scikit-learn |
| 1 600 000 × 64, k = 200 | 2.20 | 1.62 | 1.55~ | 2.49 | 1.58 | 1.16~ | 1.19~ | 1.30 | 0.89~ | **0.67** | **0.72** | — | scikit-learn |
| 1 000 000 × 96, k = 1 | 2.69 | 1.76 | 1.54 | 1.26~ | 1.19~ | 1.41 | 1.56~ | 1.96 | 2.20 | 2.68 | 2.36 | — | MLX |
| 1 000 000 × 96, k = 5 | 2.59 | 1.70 | 1.56 | 1.23 | 2.33 | 2.06 | 1.73 | 1.94 | 2.34 | 2.73 | 2.22 | — | MLX |
| 1 000 000 × 96, k = 10 | 2.60 | 1.75 | 1.54 | 1.20 | 2.29 | 1.37~ | 1.63~ | 1.97 | 2.32 | 2.60 | 2.26 | — | MLX |
| 1 000 000 × 96, k = 50 | 2.56 | 1.73 | 1.44 | 1.15~ | 1.89 | 1.32~ | 1.53~ | 1.80 | 2.15 | 2.27 | 1.79 | — | MLX |
| 1 000 000 × 96, k = 100 | 2.34 | 1.70 | 1.49 | 1.17~ | 1.55~ | 1.30~ | 1.55 | 1.72 | 1.88 | 1.76 | 1.50 | — | scikit-learn |
| 1 000 000 × 96, k = 200 | 2.17 | 1.75 | 1.53 | 1.19~ | 2.04 | 1.25~ | 1.58 | 1.62 | 1.86 | 1.68 | 1.46 | — | scikit-learn |
| 1 000 000 × 128, k = 1 | 3.32 | 1.79 | 1.75 | 1.37~ | 2.65 | 1.58 | 1.93 | 2.56 | 2.74 | 2.99 | — | — | MLX |
| 1 000 000 × 128, k = 5 | 3.14 | 1.90 | 1.61 | 1.43~ | 2.49 | 1.62 | 1.92 | 2.48 | 2.84 | 3.12 | — | — | MLX |
| 1 000 000 × 128, k = 10 | 3.38 | 1.93 | 1.66 | 1.30 | 2.41 | 1.68 | 1.94 | 2.50 | 2.83 | 3.11 | — | — | MLX |
| 1 000 000 × 128, k = 50 | 3.04 | 1.71 | 1.47 | 1.43~ | 2.21 | 1.66 | 1.96 | 2.21 | 2.55 | 2.73 | — | — | MLX |
| 1 000 000 × 128, k = 100 | 2.90 | 1.74 | 1.58 | 1.22~ | 1.26~ | 1.49~ | 1.79 | 1.99 | 2.21 | 2.03 | — | — | scikit-learn |
| 1 000 000 × 128, k = 200 | 2.78 | 1.73 | 1.52 | 1.46~ | 1.41~ | 1.48~ | 1.81 | 2.11 | 2.17 | 1.96 | — | — | scikit-learn |
| 24 000 × 256, k = 5 | 1.06~ | 1.09 | 1.10~ | 0.98~ | 1.06~ | 1.11~ | 1.19 | 1.14~ | 1.54 | 1.61 | 2.58 | 3.78 | MLX |
| 24 000 × 256, k = 10 | 1.01~ | 1.06~ | 1.06~ | 0.97~ | 1.00~ | 1.05~ | 1.13 | 1.16~ | 1.34~ | 1.56~ | 2.59 | 3.89 | MLX |
| 24 000 × 256, k = 50 | 1.03~ | 1.04~ | 1.00~ | 0.92~ | 0.93~ | 1.07~ | 1.13~ | 1.11~ | 1.34 | 1.41~ | 2.35 | 3.42 | MLX |
| 24 000 × 256, k = 100 | 0.99~ | 0.99~ | 1.03~ | 0.96~ | 0.98~ | 1.08~ | 1.06~ | 1.09~ | 1.20~ | 1.58 | 2.04 | 3.00 | scikit-learn |
| 24 000 × 256, k = 200 | 0.99~ | 0.99~ | 1.01~ | 0.99~ | **0.69** | 1.02~ | **0.79** | **0.90** | 0.98~ | 1.40 | 1.74 | 2.22 | scikit-learn |
| 100 000 × 256, k = 5 | 1.85 | 1.56 | 1.50 | 1.35 | 1.29 | 1.42 | 1.64 | 1.44~ | 2.65 | 3.20 | 4.54 | 6.22 | MLX |
| 100 000 × 256, k = 10 | 1.84 | 1.59 | 1.42 | 1.30 | 1.30 | 1.47 | 1.61 | 1.62 | 2.29 | 3.27 | 4.69 | 5.95 | MLX |
| 100 000 × 256, k = 50 | 1.79 | 1.43 | 1.35 | 1.27 | 1.18 | 1.41 | 1.40 | 1.66 | 2.48 | 3.18 | 4.08 | 5.25 | MLX |
| 100 000 × 256, k = 100 | 1.78 | 1.45 | 1.38 | 1.23 | 1.13 | 1.17~ | 1.36~ | 1.67~ | 2.18 | 2.81 | 2.88 | 4.13 | scikit-learn |
| 100 000 × 256, k = 200 | 1.82 | 1.47 | 1.33 | **0.84** | 1.08 | 1.20~ | 1.29 | 1.37~ | 1.96 | 2.43 | 2.95 | 3.53 | scikit-learn |
| 8 192 × 1024, k = 5 | 1.21 | 1.43 | 1.37 | 1.42 | 1.40 | 1.51 | 1.54 | 1.40 | 1.69 | 1.76 | 2.44 | 3.14 | MLX |
| 8 192 × 1024, k = 10 | 1.20 | 1.35 | 1.31 | 1.32 | 1.42 | 1.38 | 1.43 | 1.45 | 1.99 | 1.98 | 2.10 | 2.92 | MLX |
| 8 192 × 1024, k = 50 | 1.15~ | 1.29 | 1.30 | 1.34 | 1.24 | 1.41 | 1.28 | 1.25~ | 1.57 | 1.53 | 2.14 | 2.56 | MLX |
| 8 192 × 1024, k = 100 | 1.21 | 1.31 | 1.29 | 1.21 | 1.20 | 1.32 | 1.32 | 1.15 | 1.19 | 1.38 | 1.48 | 1.84 | scikit-learn |
| 8 192 × 1024, k = 200 | 1.15 | 1.27 | 1.24 | 1.15~ | 1.05~ | 1.12 | 0.92~ | **0.74** | 0.92~ | 0.90~ | 1.00~ | 1.31 | scikit-learn |
| 4 096 × 2048, k = 5 | 1.21 | 1.14~ | 1.10 | 1.20 | 1.26 | 1.41 | 1.30~ | 1.56 | 1.40 | 1.61 | 1.93 | 2.31 | scikit-learn |
| 4 096 × 2048, k = 10 | 1.13 | 1.14 | 1.09 | 1.23 | 1.20 | 1.35 | 0.96~ | 1.36 | 1.40 | 1.55 | 2.09 | 2.26 | scikit-learn |
| 4 096 × 2048, k = 50 | 1.19 | 1.09 | 1.08~ | 1.14 | 1.05~ | 1.04~ | 0.95~ | 0.94~ | 0.85~ | 1.01~ | 1.17 | 1.38 | scikit-learn |
| 4 096 × 2048, k = 100 | 1.09 | 1.02~ | 0.99~ | 0.97~ | 0.99~ | 0.89~ | **0.63** | **0.67** | **0.61** | **0.69** | **0.83** | 0.95~ | scikit-learn |
| 4 096 × 2048, k = 200 | 1.17 | 1.02~ | 0.99~ | **0.92** | **0.74** | **0.55** | **0.39** | **0.40** | **0.38** | **0.46** | **0.52** | **0.53** | scikit-learn |
| 6 000 × 2048, k = 5 | 1.38 | 1.23 | 1.22 | 1.26 | 1.20~ | 1.43 | 1.36 | 1.98 | 1.69 | 1.80 | 2.19 | 2.74 | scikit-learn |
| 6 000 × 2048, k = 10 | 1.36 | 1.22 | 1.27~ | 1.17 | 1.20~ | 1.60 | 1.29 | 1.71 | 1.61 | 1.64 | 2.14 | 2.72 | scikit-learn |
| 6 000 × 2048, k = 50 | 1.34 | 1.26 | 1.21 | 1.09~ | 1.17~ | 1.19~ | 1.17 | 1.09~ | 1.14~ | 1.19~ | 1.39 | 1.69 | scikit-learn |
| 6 000 × 2048, k = 100 | 1.32 | 1.17 | 1.14 | 1.02~ | 0.92~ | 0.99~ | **0.77** | **0.68** | 0.77~ | 0.86~ | 1.01~ | 1.17 | scikit-learn |
| 6 000 × 2048, k = 200 | 1.29 | 1.20~ | 1.06~ | 0.93~ | **0.77** | **0.77** | **0.53** | **0.64** | **0.51** | **0.61** | **0.69** | **0.67** | scikit-learn |
| 4 096 × 3072, k = 5 | 1.41 | 1.11 | 1.26 | 1.25 | 1.30 | 1.28 | 1.66 | 1.50 | 1.25~ | 1.46 | 2.03 | 2.31 | scikit-learn |
| 4 096 × 3072, k = 10 | 1.36 | 1.13 | 1.29 | 1.27 | 1.31 | 1.40 | 1.17~ | 1.40 | 1.48 | 1.57 | 2.06 | 2.38 | scikit-learn |
| 4 096 × 3072, k = 40 | 1.29 | 1.04~ | 1.25 | 1.17~ | 1.18~ | 1.07~ | 1.12~ | 0.90~ | 1.09~ | 0.99~ | 1.30 | 1.49 | scikit-learn |
| 4 096 × 3072, k = 50 | 1.39 | 1.10~ | 1.24 | 1.15~ | 1.08 | 1.09~ | 1.08~ | 0.95~ | 0.88~ | 0.90~ | 1.25 | 1.41 | scikit-learn |
| 4 096 × 3072, k = 100 | 1.36 | 1.08~ | 1.24 | 1.13~ | 0.92~ | **0.75** | **0.67** | **0.64** | **0.69** | **0.79** | 0.84~ | 0.89~ | scikit-learn |
| 4 096 × 3072, k = 200 | 1.39 | 1.20~ | 1.10~ | **0.85** | **0.69** | **0.48** | **0.33** | **0.48** | **0.40** | **0.47** | **0.52** | **0.50** | scikit-learn |
| 5 000 × 3072, k = 50 | 1.48 | 1.13~ | 1.07~ | 1.02~ | 1.11 | 0.99~ | 1.07~ | 0.85~ | 1.07~ | 1.22 | 1.18 | 1.65 | scikit-learn |
| 4 096 × 4096, k = 5 | 1.54 | 1.92 | 2.18 | 1.37 | 1.21~ | 1.34 | 1.19~ | 1.38 | 1.26 | 1.70 | 1.60 | 2.11 | scikit-learn |
| 4 096 × 4096, k = 10 | 1.54 | 2.31 | 2.29 | 2.11 | 2.02 | 1.81 | 1.23~ | 1.58 | 1.33 | 1.57 | 1.84 | 2.05 | scikit-learn |
| 4 096 × 4096, k = 40 | 1.55 | 2.28 | 2.11 | 1.79 | 1.87 | 1.38 | 1.06~ | 0.98~ | 0.90~ | 1.12~ | 1.31 | 1.55 | scikit-learn |
| 4 096 × 4096, k = 50 | 1.51 | 2.28 | 2.07 | 1.71 | 1.59 | 1.30 | 0.93~ | 0.90~ | 0.84~ | 1.10~ | 1.23 | 1.40 | scikit-learn |
| 4 096 × 4096, k = 100 | 1.45 | 1.80 | 1.89 | 1.51 | 1.01~ | **0.84** | **0.65** | **0.69** | **0.65** | **0.75** | **0.80** | **0.85** | scikit-learn |
| 4 096 × 4096, k = 200 | 1.47 | 1.84 | 1.61 | 1.10~ | **0.72** | **0.50** | **0.48** | **0.50** | **0.43** | **0.51** | **0.50** | **0.50** | scikit-learn |
| 5 000 × 4096, k = 50 | 1.59 | 1.30 | 1.18 | 1.18~ | 0.89~ | 0.94~ | 0.97~ | 0.99~ | 0.96~ | 1.15~ | 1.34~ | 1.56 | scikit-learn |
| 6 144 × 4096, k = 40 | 1.85 | 1.55 | 1.55 | 1.39~ | 1.41 | 1.34 | 1.17~ | 1.16~ | 1.03~ | 1.64 | 1.67 | 1.91 | scikit-learn |
| 7 500 × 4096, k = 50 | 1.84 | 1.70 | 1.82 | 1.36 | 1.24~ | 1.21 | 1.18~ | 1.35 | 1.23~ | 1.43 | 1.65 | — | scikit-learn |

MLX pays per query for every candidate it keeps — through each tile's selection, and in
refinement, a gather of k rows of every feature — where stock brute force pays almost
nothing more for a larger k. At k = 100 a 1 024-query batch lost at 1 000 000 × 64 on both
versions (0.80×), and at 4 096–6 000 rows and 2 048–4 096 features batches of 32–4 096
queries lost (0.61×–0.89×); k = 200 lost there at nearly every batch from 8 queries (down
to 0.33×), at 8 192 × 1 024 (0.74×–0.82×, 128 queries) and at a million rows and more at
64 features (0.67×–0.83×, 512–1 024 queries). At 16–48 features, with the fused kernel,
k = 1–50 won every batch (1.26×~ at worst) and k = 100 and 200 lost batches of 8–4 096
queries (0.54×–0.94×). k = 50 won or tied everywhere, but with 82–125
training rows per neighbor at 2 048–4 096 features — 4 096–5 000 rows at k = 40–50 — it only
tied at batches of 16–512 queries (0.84×–0.99×; an independent run at 4 096 × 3 072, k = 50,
lost 0.80×–0.84× at 128–256 queries), while from 150 rows per neighbor (6 144 and 7 500 ×
4 096) every batch won or tied at 1.03× or better. So patched dispatch hands a fit to
scikit-learn when its `n_neighbors` is above 50 (`knn_max_neighbors`), recorded as
`"below-crossover"`; the row floors — 8 192 rows from 256 features, below — leave more than
160 rows per neighbor at every k it allows, so the 150 needs no rule of its own.

The fit reads the constructor's `n_neighbors`, the one it can see. A query on an
MLX-fitted model that asks for more — `kneighbors(X, n_neighbors=...)`, or `n_neighbors`
raised with `set_params` — would cost MLX what a fit with that k does, so patched mode
answers it with scikit-learn's brute force instead, over a float64 copy of the training
rows — the float32 rows plus their residuals, the caller's input to about 2⁻⁴⁸,
translated by their mean as the MLX search is, with the queries — built on the first such
query and kept with the model, not pickled. The translation keeps data far from the
origin, such as Unix timestamps, at the MLX search's precision, where stock
scikit-learn's expanded distances lose it. Recorded as `"below-crossover"` on the call's
event. Grid `knn-query-k` times it at four shapes patched dispatch fits on MLX, in
float64 and float32: `kneighbors(n_neighbors=k)` on a model fitted with k = 10 against the
same call on a stock model — k = 50, which stays on MLX, and k = 51–200 three ways: as
dispatched, once the copy exists; as dispatched, the first such query, with the copy
dropped before every call (outside the timed window); and on MLX, which a forced fit keeps.
These ratios sit near 1, and at 9 pairs a point handoff losses of 0.82×–0.95× appeared and
did not reproduce in reruns of 21, so the grid takes 21:

<!-- python -m benchmarks.calibrate table knn-query-k: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, dtype, query k, path | 1 query | 16 queries | 64 queries | 256 queries | 1024 queries | dispatch |
|---|---|---|---|---|---|---|
| 1 000 000 × 64, float32, k = 50, on MLX (as dispatched) | 2.30 / 2.94 | 1.54~ / 0.86~ | 1.21~ / 1.34~ | 1.65 / 1.47 | 1.17 / 1.06~ | MLX |
| 1 000 000 × 64, float32, k = 51, as dispatched | 1.07 / 1.05~ | 0.96~ / **0.93** | 0.95~ / 0.99~ | 0.98~ / **0.95** | 1.00~ / **0.95** | scikit-learn |
| 1 000 000 × 64, float32, k = 51, as dispatched, first query | **0.17** / **0.13** | — | **0.45** / **0.37** | — | **0.88** / **0.82** | scikit-learn |
| 1 000 000 × 64, float32, k = 51, on MLX | 2.23 / 2.11 | 1.26 / 1.60 | 1.50~ / 1.43 | 1.50 / 1.56 | 1.03~ / 1.05~ | scikit-learn |
| 1 000 000 × 64, float32, k = 60, as dispatched | 1.04~ / 1.05~ | **0.94** / 0.94~ | 0.93~ / 0.95~ | 0.98~ / **0.94** | 0.97 / 0.96 | scikit-learn |
| 1 000 000 × 64, float32, k = 60, as dispatched, first query | **0.17** / **0.14** | — | **0.46** / **0.37** | — | **0.88** / **0.83** | scikit-learn |
| 1 000 000 × 64, float32, k = 60, on MLX | 2.16 / 2.10 | 1.67~ / 1.54 | 1.50 / 1.31~ | 1.52 / 1.52 | 1.00~ / 1.04~ | scikit-learn |
| 1 000 000 × 64, float32, k = 100, as dispatched | 1.05~ / 1.05 | **0.94** / **0.95** | 0.98~ / 0.98~ | 0.96~ / 0.96~ | 0.99~ / **0.94** | scikit-learn |
| 1 000 000 × 64, float32, k = 100, as dispatched, first query | **0.16** / **0.13** | — | **0.44** / **0.38** | — | **0.89** / **0.83** | scikit-learn |
| 1 000 000 × 64, float32, k = 100, on MLX | 2.03 / 1.99 | 1.51~ / 1.05~ | 1.26~ / 1.05~ | 1.29 / 1.26 | **0.84** / **0.86** | scikit-learn |
| 1 000 000 × 64, float32, k = 200, as dispatched | 1.04~ / 1.08 | **0.93** / 0.94~ | 0.99~ / 1.00~ | 1.01~ / 1.00~ | 0.99~ / 0.99~ | scikit-learn |
| 1 000 000 × 64, float32, k = 200, as dispatched, first query | **0.17** / **0.13** | — | **0.47** / **0.41** | — | **0.92** / **0.86** | scikit-learn |
| 1 000 000 × 64, float32, k = 200, on MLX | 1.91 / 1.97 | 1.43~ / 1.39~ | 1.28~ / 1.22~ | 1.26 / 1.20 | 0.95~ / **0.86** | scikit-learn |
| 1 000 000 × 64, float64, k = 50, on MLX (as dispatched) | 2.27 / 2.72 | 1.63 / 2.04 | 1.96 / 2.15 | 1.57 / 1.74 | 1.42 / 1.44 | MLX |
| 1 000 000 × 64, float64, k = 51, as dispatched | 0.97~ / 1.00~ | 0.99~ / 1.00~ | 1.00~ / 0.98~ | 1.01~ / 1.09~ | 1.02~ / 1.00~ | scikit-learn |
| 1 000 000 × 64, float64, k = 51, as dispatched, first query | **0.09** / **0.07** | — | **0.31** / **0.26** | — | **0.87** / **0.82** | scikit-learn |
| 1 000 000 × 64, float64, k = 51, on MLX | 2.13 / 2.16 | 1.72 / 1.85 | 1.94 / 1.97 | 1.78 / 1.56 | 1.32 / 1.13 | scikit-learn |
| 1 000 000 × 64, float64, k = 60, as dispatched | 1.00~ / 1.01~ | 1.01~ / 1.06~ | 1.00~ / 0.99~ | 1.00~ / 1.03~ | 1.00~ / 1.02~ | scikit-learn |
| 1 000 000 × 64, float64, k = 60, as dispatched, first query | **0.10** / **0.07** | — | **0.33** / **0.26** | — | **0.90** / **0.82** | scikit-learn |
| 1 000 000 × 64, float64, k = 60, on MLX | 2.10 / 2.07 | 1.66 / 1.90 | 1.91 / 1.77 | 1.57 / 1.51 | 1.07~ / 1.25 | scikit-learn |
| 1 000 000 × 64, float64, k = 100, as dispatched | 0.99~ / 1.00~ | 1.02~ / 0.96~ | 1.00~ / 0.99~ | 0.98~ / 0.97~ | 0.99~ / 0.98~ | scikit-learn |
| 1 000 000 × 64, float64, k = 100, as dispatched, first query | **0.10** / **0.07** | — | **0.33** / **0.26** | — | **0.86** / **0.83** | scikit-learn |
| 1 000 000 × 64, float64, k = 100, on MLX | 1.98 / 1.96 | 1.62 / 1.27~ | 1.13~ / 1.55 | 1.29 / 1.30 | **0.83** / **0.77** | scikit-learn |
| 1 000 000 × 64, float64, k = 200, as dispatched | 1.00~ / 1.00~ | 1.00~ / 0.99~ | 1.02~ / 0.98~ | 1.00~ / 0.98~ | 1.00~ / 1.00~ | scikit-learn |
| 1 000 000 × 64, float64, k = 200, as dispatched, first query | **0.10** / **0.07** | — | **0.32** / **0.25** | — | **0.87** / **0.84** | scikit-learn |
| 1 000 000 × 64, float64, k = 200, on MLX | 1.92 / 1.94 | 1.46 / 1.26~ | 1.66 / 1.44 | 1.39 / 1.27 | 1.05~ / **0.80** | scikit-learn |
| 65 536 × 256, float32, k = 50, on MLX (as dispatched) | 1.67 / 1.80 | 1.37 / 1.39 | 1.68 / 1.55 | 2.02 / 2.62 | 3.88 / 4.38 | MLX |
| 65 536 × 256, float32, k = 51, as dispatched | 1.04~ / 1.10~ | 1.10~ / 1.07~ | 1.05~ / 1.10~ | 1.00~ / 0.96~ | 1.02~ / 1.02~ | scikit-learn |
| 65 536 × 256, float32, k = 51, as dispatched, first query | **0.26** / **0.20** | — | **0.56** / **0.43** | — | **0.90** / **0.88** | scikit-learn |
| 65 536 × 256, float32, k = 51, on MLX | 1.64 / 1.71 | 1.59 / 1.31 | 1.71 / 1.57 | 1.86 / 2.06 | 3.73 / 4.46 | scikit-learn |
| 65 536 × 256, float32, k = 60, as dispatched | 1.06~ / 1.11~ | 1.14~ / 1.02~ | 1.08~ / 1.07~ | 1.07~ / 1.08~ | 1.03~ / 1.02~ | scikit-learn |
| 65 536 × 256, float32, k = 60, as dispatched, first query | **0.26** / **0.20** | — | **0.51** / **0.40** | — | 0.96~ / **0.85** | scikit-learn |
| 65 536 × 256, float32, k = 60, on MLX | 1.62 / 1.75 | 1.44 / 1.45 | 1.72 / 1.63 | 1.83~ / 2.12 | 3.57 / 4.45 | scikit-learn |
| 65 536 × 256, float32, k = 100, as dispatched | 1.07~ / 1.03~ | 1.05~ / 1.09~ | 1.03~ / 1.02~ | 1.09~ / 1.09~ | 1.00~ / 1.04~ | scikit-learn |
| 65 536 × 256, float32, k = 100, as dispatched, first query | **0.27** / **0.20** | — | **0.52** / **0.45** | — | 0.93~ / **0.90** | scikit-learn |
| 65 536 × 256, float32, k = 100, on MLX | 1.72 / 1.70 | 1.33 / 1.36 | 1.45 / 1.48 | 1.75 / 2.70 | 3.23 / 3.36 | scikit-learn |
| 65 536 × 256, float32, k = 200, as dispatched | 1.07~ / 1.10 | 1.06 / 1.01~ | 1.05~ / 1.05~ | 1.03~ / 1.07~ | 1.01~ / 1.01~ | scikit-learn |
| 65 536 × 256, float32, k = 200, as dispatched, first query | **0.25** / **0.21** | — | **0.60** / **0.49** | — | **0.89** / **0.92** | scikit-learn |
| 65 536 × 256, float32, k = 200, on MLX | 1.58 / 1.75 | 1.28 / 1.33 | 1.33 / 1.40 | 1.92 / 2.21 | 3.20 / 2.86 | scikit-learn |
| 65 536 × 256, float64, k = 50, on MLX (as dispatched) | 1.51 / 1.65 | 1.25 / 1.33 | 1.71 / 1.74 | 1.84 / 2.99 | 4.60 / 3.91 | MLX |
| 65 536 × 256, float64, k = 51, as dispatched | 0.99~ / 0.95~ | 1.02~ / 1.00~ | 1.01~ / 0.97~ | 1.05~ / 1.03~ | 0.99~ / 0.99~ | scikit-learn |
| 65 536 × 256, float64, k = 51, as dispatched, first query | **0.14** / **0.11** | — | **0.37** / **0.27** | — | **0.83** / **0.75** | scikit-learn |
| 65 536 × 256, float64, k = 51, on MLX | 1.48 / 1.56 | 1.23 / 1.29 | 1.79 / 1.74 | 2.21 / 2.69 | 3.58 / 3.55 | scikit-learn |
| 65 536 × 256, float64, k = 60, as dispatched | 1.00~ / 0.95~ | 0.97~ / 0.99~ | 1.05~ / 0.98~ | 1.02~ / 0.98~ | 1.01~ / 1.03~ | scikit-learn |
| 65 536 × 256, float64, k = 60, as dispatched, first query | **0.15** / **0.12** | — | **0.35** / **0.27** | — | **0.83** / **0.78** | scikit-learn |
| 65 536 × 256, float64, k = 60, on MLX | 1.54 / 1.57 | 1.34 / 1.30 | 1.75 / 1.64 | 1.89 / 2.89 | 3.83 / 3.73 | scikit-learn |
| 65 536 × 256, float64, k = 100, as dispatched | 0.99~ / 1.02~ | 0.99~ / 1.03~ | 1.03~ / 0.95~ | 0.98~ / 0.99~ | 0.97~ / 0.99~ | scikit-learn |
| 65 536 × 256, float64, k = 100, as dispatched, first query | **0.14** / **0.11** | — | **0.38** / **0.28** | — | **0.85** / **0.78** | scikit-learn |
| 65 536 × 256, float64, k = 100, on MLX | 1.47 / 1.55 | 1.18 / 1.27 | 1.47 / 1.44 | 1.56 / 2.70 | 3.41 / 3.12 | scikit-learn |
| 65 536 × 256, float64, k = 200, as dispatched | 0.95~ / 0.96~ | 1.04~ / 1.00~ | 0.99~ / **0.88** | 0.95~ / 1.04~ | 1.00~ / 0.99~ | scikit-learn |
| 65 536 × 256, float64, k = 200, as dispatched, first query | **0.14** / **0.12** | — | **0.41** / **0.29** | — | **0.86** / **0.84** | scikit-learn |
| 65 536 × 256, float64, k = 200, on MLX | 1.48 / 2.20 | 1.23 / 1.37 | 1.44 / 1.45 | 1.54 / 2.33 | 3.03 / 2.64 | scikit-learn |
| 8 192 × 1024, float32, k = 50, on MLX (as dispatched) | 2.06 / 2.24 | 1.71 / 1.75 | 1.93 / 2.00 | 2.00 / 2.27 | 3.02 / 3.54 | MLX |
| 8 192 × 1024, float32, k = 51, as dispatched | 1.41 / 1.44 | 1.39 / 1.30 | 1.14~ / 1.09~ | 1.10 / 1.05~ | 1.10 / 1.09 | scikit-learn |
| 8 192 × 1024, float32, k = 51, as dispatched, first query | **0.50** / **0.37** | — | **0.68** / **0.56** | — | 0.98~ / 0.95~ | scikit-learn |
| 8 192 × 1024, float32, k = 51, on MLX | 2.04 / 2.26 | 1.84 / 1.77 | 1.87 / 1.97 | 2.17 / 2.37 | 2.85 / 4.01 | scikit-learn |
| 8 192 × 1024, float32, k = 60, as dispatched | 1.46 / 1.43 | 1.36 / 1.34 | 1.19 / 1.11~ | 1.24 / 1.11~ | 1.06~ / 1.04~ | scikit-learn |
| 8 192 × 1024, float32, k = 60, as dispatched, first query | **0.46** / **0.36** | — | **0.67** / **0.55** | — | 0.98~ / 0.95~ | scikit-learn |
| 8 192 × 1024, float32, k = 60, on MLX | 2.07 / 2.18 | 1.77 / 1.75 | 1.74 / 1.73 | 2.03 / 2.10 | 2.65 / 3.02 | scikit-learn |
| 8 192 × 1024, float32, k = 100, as dispatched | 1.39 / 1.36 | 1.23 / 1.20 | 1.10~ / 1.19 | 1.08~ / 1.11 | 1.05~ / 1.06~ | scikit-learn |
| 8 192 × 1024, float32, k = 100, as dispatched, first query | **0.46** / **0.36** | — | **0.69** / **0.58** | — | 0.99~ / 0.95~ | scikit-learn |
| 8 192 × 1024, float32, k = 100, on MLX | 2.00 / 2.15 | 1.79 / 1.75 | 1.69 / 1.63 | 1.76 / 1.96 | 2.05 / 2.23 | scikit-learn |
| 8 192 × 1024, float32, k = 200, as dispatched | 1.37 / 1.43 | 1.19 / 1.25 | 1.18 / 1.10~ | 1.02~ / 1.08~ | 1.04~ / 1.09 | scikit-learn |
| 8 192 × 1024, float32, k = 200, as dispatched, first query | **0.46** / **0.37** | — | **0.72** / **0.62** | — | 1.01~ / **0.95** | scikit-learn |
| 8 192 × 1024, float32, k = 200, on MLX | 2.03 / 2.03 | 1.64 / 1.58 | 1.39 / 1.42 | 1.33 / 1.38 | 1.64 / 1.74 | scikit-learn |
| 8 192 × 1024, float64, k = 50, on MLX (as dispatched) | 1.35 / 1.24 | 1.37 / 1.34 | 1.39 / 1.54 | 1.52 / 1.42 | 2.26 / 2.07 | MLX |
| 8 192 × 1024, float64, k = 51, as dispatched | 0.95~ / 0.94~ | 0.96~ / 0.96~ | 1.00~ / 0.95~ | 0.98~ / 0.96~ | 0.96~ / 0.98~ | scikit-learn |
| 8 192 × 1024, float64, k = 51, as dispatched, first query | **0.19** / **0.16** | — | **0.41** / **0.36** | — | **0.81** / **0.78** | scikit-learn |
| 8 192 × 1024, float64, k = 51, on MLX | 1.43 / 1.32 | 1.35 / 1.24 | 1.39 / 1.50 | 1.57 / 1.72 | 2.40 / 2.38 | scikit-learn |
| 8 192 × 1024, float64, k = 60, as dispatched | 0.95~ / 0.97~ | 1.03~ / 0.93~ | 1.04~ / 1.01~ | 0.96~ / 1.01~ | 1.00~ / 1.01~ | scikit-learn |
| 8 192 × 1024, float64, k = 60, as dispatched, first query | **0.20** / **0.14** | — | **0.41** / **0.32** | — | **0.81** / **0.75** | scikit-learn |
| 8 192 × 1024, float64, k = 60, on MLX | 1.33 / 1.30 | 1.30 / 1.32 | 1.39 / 1.46 | 1.40 / 1.64 | 1.72 / 1.81 | scikit-learn |
| 8 192 × 1024, float64, k = 100, as dispatched | 0.89~ / 1.05~ | 1.00~ / 1.00~ | 1.02~ / 0.98~ | 0.97~ / 1.00~ | 1.00~ / 0.98~ | scikit-learn |
| 8 192 × 1024, float64, k = 100, as dispatched, first query | **0.20** / **0.15** | — | **0.47** / **0.33** | — | **0.86** / **0.81** | scikit-learn |
| 8 192 × 1024, float64, k = 100, on MLX | 1.42 / 1.34 | 1.31 / 1.21 | 1.29 / 1.34 | 1.19 / 1.36 | 1.73 / 1.86 | scikit-learn |
| 8 192 × 1024, float64, k = 200, as dispatched | 1.04~ / 0.99~ | 0.98~ / 0.94~ | 1.05~ / 1.00~ | 1.04~ / 0.99~ | 1.01~ / 0.97~ | scikit-learn |
| 8 192 × 1024, float64, k = 200, as dispatched, first query | **0.21** / **0.16** | — | **0.47** / **0.36** | — | **0.83** / **0.83** | scikit-learn |
| 8 192 × 1024, float64, k = 200, on MLX | 1.42 / 1.32 | 1.17 / 1.11~ | 1.08~ / 1.04~ | 0.97~ / 1.05~ | 1.33 / 1.33 | scikit-learn |
| 8 192 × 2048, float32, k = 50, on MLX (as dispatched) | 3.17 / 3.27 | 2.11 / 2.22 | 2.00 / 2.12 | 2.04 / 2.08 | 2.71 / 2.84 | MLX |
| 8 192 × 2048, float32, k = 51, as dispatched | 1.72 / 1.75 | 1.38 / 1.33 | 1.31 / 1.23 | 1.14 / 1.16 | 1.09 / 1.07 | scikit-learn |
| 8 192 × 2048, float32, k = 51, as dispatched, first query | **0.39** / **0.31** | — | **0.58** / **0.45** | — | **0.90** / **0.88** | scikit-learn |
| 8 192 × 2048, float32, k = 51, on MLX | 2.91 / 3.07 | 2.11 / 2.06 | 1.88 / 1.87 | 2.08 / 2.08 | 2.71 / 2.79 | scikit-learn |
| 8 192 × 2048, float32, k = 60, as dispatched | 1.58 / 1.76 | 1.46 / 1.33 | 1.27 / 1.33 | 1.09 / 1.11 | 1.05~ / 1.05 | scikit-learn |
| 8 192 × 2048, float32, k = 60, as dispatched, first query | **0.39** / **0.32** | — | **0.57** / **0.45** | — | **0.93** / **0.87** | scikit-learn |
| 8 192 × 2048, float32, k = 60, on MLX | 3.01 / 3.05 | 1.98 / 2.03 | 1.79 / 2.02 | 2.20 / 1.87 | 2.48 / 2.43 | scikit-learn |
| 8 192 × 2048, float32, k = 100, as dispatched | 1.72 / 1.71 | 1.44 / 1.38 | 1.39 / 1.31 | 1.14 / 1.10 | 1.05 / 1.06 | scikit-learn |
| 8 192 × 2048, float32, k = 100, as dispatched, first query | **0.39** / **0.32** | — | **0.56** / **0.50** | — | **0.94** / **0.91** | scikit-learn |
| 8 192 × 2048, float32, k = 100, on MLX | 2.85 / 2.86 | 1.85 / 2.10 | 1.51 / 1.60 | 1.50 / 1.56 | 1.68 / 1.87 | scikit-learn |
| 8 192 × 2048, float32, k = 200, as dispatched | 1.65 / 1.71 | 1.45 / 1.38 | 1.28 / 1.30 | 1.11 / 1.10 | 1.03~ / 1.03~ | scikit-learn |
| 8 192 × 2048, float32, k = 200, as dispatched, first query | **0.39** / **0.31** | — | **0.47** / **0.50** | — | **0.93** / **0.90** | scikit-learn |
| 8 192 × 2048, float32, k = 200, on MLX | 2.72 / 2.89 | 1.49 / 1.58 | 0.96~ / 1.18~ | 0.87~ / 1.08~ | 1.08 / 1.17 | scikit-learn |
| 8 192 × 2048, float64, k = 50, on MLX (as dispatched) | 1.69 / 1.69 | 1.43 / 1.49 | 1.47 / 1.44 | 1.65 / 1.58 | 1.97 / 1.97 | MLX |
| 8 192 × 2048, float64, k = 51, as dispatched | 1.00~ / 1.03~ | 0.92~ / 1.00~ | 0.93~ / 1.03~ | 1.01~ / 0.97~ | 0.98~ / 0.99~ | scikit-learn |
| 8 192 × 2048, float64, k = 51, as dispatched, first query | **0.15** / **0.11** | — | **0.30** / **0.21** | — | **0.78** / **0.74** | scikit-learn |
| 8 192 × 2048, float64, k = 51, on MLX | 1.61 / 1.61 | 1.44 / 1.51 | 1.31 / 1.42 | 1.81 / 1.65 | 1.95 / 1.88 | scikit-learn |
| 8 192 × 2048, float64, k = 60, as dispatched | 0.94~ / 0.97~ | 0.93~ / 1.00~ | 0.98~ / 0.95~ | 1.02~ / 0.95~ | 0.98~ / 0.97~ | scikit-learn |
| 8 192 × 2048, float64, k = 60, as dispatched, first query | **0.15** / **0.11** | — | **0.28** / **0.23** | — | **0.76** / **0.72** | scikit-learn |
| 8 192 × 2048, float64, k = 60, on MLX | 1.66 / 1.58 | 1.39 / 1.43 | 1.24 / 1.45 | 1.60 / 1.47 | 1.85 / 1.89 | scikit-learn |
| 8 192 × 2048, float64, k = 100, as dispatched | 0.98~ / 1.00~ | 1.05~ / 0.99~ | 0.96~ / 1.00~ | 0.94~ / 1.00~ | 0.98~ / 0.99~ | scikit-learn |
| 8 192 × 2048, float64, k = 100, as dispatched, first query | **0.15** / **0.11** | — | **0.28** / **0.23** | — | **0.79** / **0.73** | scikit-learn |
| 8 192 × 2048, float64, k = 100, on MLX | 1.68 / 1.61 | 1.26 / 1.33 | 1.07~ / 1.04~ | 1.12~ / 1.13 | 1.41 / 1.41 | scikit-learn |
| 8 192 × 2048, float64, k = 200, as dispatched | 1.00~ / 1.00~ | 1.01~ / 0.90~ | 0.95~ / 1.01~ | 1.04~ / 0.95~ | 0.99~ / 0.98~ | scikit-learn |
| 8 192 × 2048, float64, k = 200, as dispatched, first query | **0.14** / **0.11** | — | **0.31** / **0.23** | — | **0.77** / **0.77** | scikit-learn |
| 8 192 × 2048, float64, k = 200, on MLX | 1.61 / 1.66 | 1.13 / 1.05~ | **0.73** / 0.85~ | **0.75** / **0.80** | **0.85** / **0.86** | scikit-learn |

At k = 50 MLX won or tied every batch (1.17×–4.6× where it won). Handed over, float64
queries kept up with a stock model at every shape, k and batch, 0.88×–1.09×, with one
significant loss in 160: 0.88× on 1.7.2 at 65 536 × 256, k = 200, 64 queries, a point that
read 0.88×~, within noise, at 9 pairs, where k = 51 on 1.9.0 had read a 0.90× loss that 21
pairs put at 1.01×~. **Float32 data is the weak
spot.** Stock scikit-learn searches its own float32 rows, the handoff a float64 copy of
twice the bytes, and at 1 000 000 × 64 that lost on both versions at 16–1 024 queries
(0.93×–0.95×); at the other shapes float32 queries won or tied (up to 1.76×). On MLX the
same queries won at most shapes (1.08×–4.5×) but lost where the rule was set — 200
neighbors at 8 192 × 2 048 (0.73×–0.86× at 64–1 024 queries) and 100–200 at 1 000 000 × 64
with 1 024 queries (0.77×–0.86×) — so at a million float32 rows neither answer is free of
a loss, and the handoff's is the smaller. It stands as a known limit (below).

**The first such query pays for the copy**, `8 · n_samples · n_features` bytes, once per
model. The first query's median less the steady state's, over k = 51–200, 1–1 024 queries
and both versions: 2–15 ms at 8 192 × 1 024, 8–27 ms at 8 192 × 2 048, 4–29 ms at
65 536 × 256 and 25–85 ms at 1 000 000 × 64 (512 MB), float32 input at the low end of each
range. For a single row that made the first large-k query 0.07×–0.50× of stock; every later
query on the model costs the steady state above, and a reloaded pickle builds it again.

What the rule gives up is k = 100–200 at 96–128 features and a million rows (1.2×–3.0×),
k = 100 at 100 000 × 256 (1.1×–4.3×) and at 8 192 × 1 024 (1.15×–1.9×), and at query time
MLX's wins at k = 51–200 (`knn-query-k`, 1.08×–4.5×). The data rule (below) probes with the fitted k too, and the first pass's
noise band grows with the width and with k: on this Gaussian data it left 1 and 6 of 256
rows uncertified at 4 096 × 4 096 (k = 10 and 5), 1 at 4 096 × 3 072 (k = 5), and 3 and
7 at 24 000 and 100 000 × 256 (k = 200), so those fits go to scikit-learn by that rule.

Wide data needs rows too, but far fewer. Grid `knn-rows`: every batch size again, at 256
to 1 024 features, 2 048 to 131 072 rows; same notation:

<!-- python -m benchmarks.calibrate table knn-rows: scikit-learn 1.9.0 / 1.7.2, the worse of them -->
| rows × features | 1 query | 2 queries | 4 queries | 8 queries | 16 queries | 32 queries | 64 queries | 128 queries | 256 queries | 512 queries | 1024 queries | 4096 queries | fit | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 048 × 256 | **0.74** | **0.73** | **0.70** | **0.68** | **0.70** | **0.81** | **0.87** | 0.92~ | 0.89~ | 1.23~ | 1.26~ | **0.68** | **0.09** | scikit-learn |
| 4 096 × 256 | **0.76** | **0.76** | **0.70** | **0.80** | **0.82** | **0.83** | 0.98~ | 1.08~ | 1.18 | 1.52 | 1.72 | 1.59 | **0.10** | scikit-learn |
| 8 192 × 256 | **0.82** | 0.92~ | **0.86** | **0.85** | 0.95~ | 0.99~ | 1.11~ | 1.16~ | 1.37 | 1.70 | 1.93 | 2.17 | **0.08** | scikit-learn |
| 16 384 × 256 | 0.92~ | 1.02~ | 1.01~ | 0.98~ | 1.05~ | 1.15 | 1.20 | 1.37 | 1.51 | 1.60 | 2.49 | 3.70 | **0.07** | scikit-learn |
| 32 768 × 256 | 1.21 | 1.23 | 1.19 | 1.16 | 1.22 | 1.26 | 1.34 | 1.67 | 1.76 | 2.02 | 2.85 | 4.82 | **0.07** | MLX |
| 65 536 × 256 | 1.66 | 1.40 | 1.38 | 1.23 | 1.31 | 1.49 | 1.90 | 1.98 | 1.80 | 2.54 | 3.71 | 6.20 | **0.07** | MLX |
| 131 072 × 256 | 2.21 | 1.62 | 1.54 | 1.52 | 1.45~ | 1.84 | 1.97 | 2.26 | 2.91 | 3.79 | 5.24 | — | **0.06** | MLX |
| 2 048 × 384 | **0.71** | **0.77** | **0.70** | **0.71** | **0.78** | **0.85** | **0.82** | 0.90~ | 1.10~ | 1.35~ | 1.59 | **0.44** | **0.10** | scikit-learn |
| 4 096 × 384 | **0.79** | **0.82** | **0.78** | **0.77** | **0.83** | 0.94~ | 0.98~ | 1.15~ | 1.12~ | 1.27 | 1.44 | 1.50 | **0.09** | scikit-learn |
| 8 192 × 384 | 0.91~ | 0.92~ | 0.95~ | 0.96~ | 1.00~ | 1.04~ | 1.16~ | 1.39~ | 1.36 | 1.38 | 1.53 | 1.52 | **0.08** | scikit-learn |
| 16 384 × 384 | 1.02~ | 1.13 | 1.14 | 1.08~ | 1.16~ | 1.25 | 1.45 | 1.83 | 1.33 | 1.46 | 1.96 | 2.77 | **0.07** | MLX |
| 32 768 × 384 | 1.38 | 1.40 | 1.37 | 1.31 | 1.41 | 1.43 | 1.84 | 2.10 | 1.64 | 1.71 | 2.43 | 3.28 | **0.06** | MLX |
| 65 536 × 384 | 1.84 | 1.66 | 1.51 | 1.49 | 1.58 | 1.92 | 2.31 | 2.12 | 1.69 | 2.26 | 3.33 | 4.62 | **0.07** | MLX |
| 131 072 × 384 | 2.61 | 2.01 | 1.84 | 1.86 | 2.04 | 2.25 | 2.17 | 3.12 | 2.06 | 3.12 | 4.24 | — | **0.06** | MLX |
| 2 048 × 512 | **0.74** | **0.76** | **0.73** | **0.78** | **0.79** | **0.79** | **0.78** | **0.72** | 0.94~ | 1.30~ | 1.58 | **0.55** | **0.09** | scikit-learn |
| 4 096 × 512 | **0.79** | 0.86~ | **0.81** | **0.82** | 0.92~ | 1.00~ | 0.93~ | 1.06~ | 1.19 | 1.19~ | 1.40 | 1.46 | **0.08** | scikit-learn |
| 8 192 × 512 | 0.90~ | 1.11 | 1.02~ | 1.05~ | 1.08 | 1.19 | 1.27 | 1.52 | 1.44 | 1.40 | 1.81 | 1.81 | **0.07** | scikit-learn |
| 16 384 × 512 | 1.16 | 1.35 | 1.31 | 1.34 | 1.23~ | 1.30 | 1.81 | 2.27 | 1.51 | 1.89 | 2.13 | 3.03 | **0.07** | MLX |
| 32 768 × 512 | 1.60 | 1.58 | 1.55 | 1.43 | 1.50 | 1.60 | 2.23 | 2.27 | 1.74 | 2.81 | 3.31 | 4.55 | **0.07** | MLX |
| 65 536 × 512 | 2.24 | 1.97 | 1.77 | 1.79 | 2.05 | 2.32 | 2.89 | 2.69 | 2.06 | 3.06 | 3.76 | — | **0.06** | MLX |
| 131 072 × 512 | 2.93 | 2.36 | 2.46 | 2.10 | 2.29 | 2.50 | 3.11 | 3.71 | 2.97 | 3.98 | 5.59 | — | **0.06** | MLX |
| 2 048 × 768 | **0.82** | **0.73** | **0.73** | **0.79** | **0.81** | **0.79** | **0.81** | 1.03~ | 1.15~ | 1.38~ | 1.23~ | **0.60** | **0.09** | scikit-learn |
| 4 096 × 768 | 0.89~ | **0.95** | **0.88** | 0.93~ | 0.98~ | 1.15~ | 1.17 | 1.21~ | 1.26 | 1.22~ | 1.43 | 1.64 | **0.08** | scikit-learn |
| 8 192 × 768 | 1.10 | 1.17 | 1.04~ | 1.15 | 1.18 | 1.26 | 1.41 | 1.38 | 1.43 | 1.93 | 2.12 | 2.56 | **0.07** | MLX |
| 16 384 × 768 | 1.43 | 1.40 | 1.38 | 1.37 | 1.57 | 1.60 | 2.18 | 1.65 | 1.79 | 2.07 | 2.80 | 3.89 | **0.07** | MLX |
| 32 768 × 768 | 1.95 | 1.87 | 1.79 | 1.70 | 1.83 | 2.07 | 3.09 | 2.11 | 2.04 | 2.87 | 3.42 | 5.63 | **0.06** | MLX |
| 65 536 × 768 | 2.63 | 2.34 | 2.20 | 2.01 | 2.35 | 2.33 | 3.11 | 2.59 | 2.81 | 4.10 | 5.19 | — | **0.06** | MLX |
| 131 072 × 768 | 3.43 | 2.17 | 2.30 | 1.99 | 2.02 | 2.99 | 3.68 | 3.03 | 3.91 | 5.45 | 6.93 | — | **0.06** | MLX |
| 2 048 × 1024 | **0.82** | **0.79** | **0.76** | **0.82** | **0.82** | **0.76** | **0.75** | 0.95~ | 1.16~ | 1.15~ | 1.21~ | **0.60** | **0.08** | scikit-learn |
| 4 096 × 1024 | 0.94~ | 1.04~ | 1.01~ | 1.00~ | 1.15 | 1.11~ | 0.97~ | 1.20~ | 1.31 | 1.46 | 1.66 | 1.66 | **0.07** | scikit-learn |
| 8 192 × 1024 | 1.20 | 1.39 | 1.37 | 1.31 | 1.29 | 1.51 | 1.42 | 1.56 | 1.73 | 1.91 | 2.08 | 2.93 | **0.07** | MLX |
| 16 384 × 1024 | 1.69 | 1.70 | 1.79 | 1.57 | 1.35 | 2.16 | 2.73 | 2.13 | 2.08 | 2.43 | 3.40 | 4.37 | **0.06** | MLX |
| 32 768 × 1024 | 2.44 | 2.35 | 2.00 | 2.37 | 2.12 | 2.37 | 3.44 | 2.73 | 2.48 | 3.20 | 4.03 | — | **0.06** | MLX |
| 65 536 × 1024 | 3.36 | 2.49 | 2.66 | 2.55 | 2.89 | 3.24 | 4.33 | 2.80 | 3.52 | 4.63 | 6.37 | — | **0.06** | MLX |
| 131 072 × 1024 | 5.15 | 2.88 | 2.79 | 2.31 | 2.42 | 2.59 | 4.07 | 4.47 | 5.23 | 7.02 | — | — | **0.06** | MLX |

Below 4 096 rows a 4 096-query batch lost at every width (0.44×–0.79× at 2 048 rows): MLX
spends a few microseconds a query row however small the index, and stock brute force over
so small an index costs less. Between 4 096 rows and 6 million elements small batches
lost — 1 to 32 queries, 0.70×–0.95× at 4 096 and 8 192 rows — and at 4 194 304 elements,
where the previous work floor sat, single rows only tied (0.90×–0.99×). From 6 million
elements every batch won or tied on both versions.

**Short, wide training sets need more rows than the floors ask.** The grids above time
batch sizes that are powers of two. Between those, and right after a stock query — whose
OpenMP threads keep spinning for a while after it returns, as they do between the
harness's pairs — some MLX calls ran up to twice as slow as their neighbours: batches of about
256 KiB of float32 queries, 32 rows at 2 048 features and 16 at 4 096, and not every time
(with the spinning stopped, `KMP_BLOCKTIME=0`, they did not; nor did a sleep, busy host
code or a NumPy product in its place). Where the training set is short, a stock search
costs about as much as the slow call. Grid `knn-batches`: every few batch sizes from 8
to 64 queries, at 4 096–12 000 rows and 2 048–4 096 features, k = 10 and at the most
neighbors the rules allow, one process per shape; each cell is the worse version's ratio:

<!-- python -m benchmarks.calibrate table knn-batches: scikit-learn 1.9.0 / 1.7.2, the worse of them -->
| rows × features, k | 8 queries | 12 queries | 14 queries | 16 queries | 18 queries | 20 queries | 22 queries | 24 queries | 26 queries | 28 queries | 30 queries | 32 queries | 34 queries | 36 queries | 40 queries | 48 queries | 64 queries | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 096 × 2048, k = 10 | 1.26 | 1.38 | 1.29 | 1.23 | 1.31 | 1.26 | 1.26 | 1.29 | 1.31~ | 1.31 | 1.44 | 1.07~ | 1.25~ | 1.53 | 1.44 | 1.37 | 1.32 | scikit-learn |
| 5 000 × 2048, k = 10 | 1.23 | 1.23 | 1.26 | 1.07~ | 1.33 | 1.47 | 1.17~ | 1.06~ | 1.31 | 1.41 | 1.40 | 1.07~ | 1.36~ | 1.45 | 1.37 | 1.24 | 1.26~ | scikit-learn |
| 6 000 × 2048, k = 10 | 1.13~ | 1.28 | 1.24~ | 1.27 | 1.40~ | 1.31 | 1.44 | 1.41 | 1.51 | 1.43 | 1.72 | 1.19~ | 1.38 | 1.53 | 1.73 | 1.35 | 1.46 | scikit-learn |
| 8 192 × 2048, k = 10 | 1.56 | 1.64 | 1.83 | 1.58 | 1.85 | 1.79 | 1.76 | 1.89 | 1.93 | 2.11 | 1.62 | 1.70 | 1.96 | 2.13 | 2.04 | 1.49 | 1.53 | MLX |
| 8 192 × 2048, k = 50 | 1.38 | 1.62 | 1.55 | 1.47 | 1.66 | 1.73 | 1.70 | 1.59 | 1.84 | 1.83 | 1.76 | 1.28 | 1.61 | 1.87 | 1.76 | 1.34 | 1.33 | MLX |
| 12 000 × 2048, k = 10 | 1.81 | 2.08 | 2.20 | 1.94 | 2.29 | 2.10 | 2.19 | 1.99 | 2.13 | 2.32 | 2.44 | 1.76 | 2.27 | 2.66 | 2.44 | 1.83 | 2.14 | MLX |
| 4 500 × 2304, k = 10 | 1.03~ | 1.11~ | 1.09~ | 1.11~ | 1.25 | 1.26~ | 1.23 | 1.02~ | 1.32 | 1.10~ | **0.87** | 0.87~ | 1.51 | 1.43 | 1.33 | 1.10~ | 1.31 | scikit-learn |
| 4 096 × 2560, k = 10 | 1.25 | 1.33 | 1.31 | 1.26 | 1.53 | 1.42 | 1.41 | 1.33~ | **0.77** | 1.01~ | **0.87** | 1.43 | 1.57 | 1.60 | 1.13 | 1.18~ | 1.21~ | scikit-learn |
| 4 096 × 3072, k = 10 | 1.25 | 1.37 | 1.33 | 1.24 | 1.44 | 1.46 | **0.79** | 0.90~ | 0.88~ | 1.56 | 1.50 | 1.33~ | 1.42 | 1.37 | 1.23 | 1.35 | 1.19~ | scikit-learn |
| 6 000 × 3072, k = 10 | 1.43 | 1.43 | 1.47 | 1.34 | 1.68 | 1.54 | 1.21 | 1.52 | 1.58 | 1.87 | 1.75 | 1.31 | 1.28 | 1.38 | 1.36 | 1.37 | 1.50 | scikit-learn |
| 6 000 × 3072, k = 40 | 1.23~ | 1.33 | 1.47 | 1.39 | 1.47 | 1.58 | 1.02~ | 1.22~ | 1.44 | 1.55 | 1.54 | 1.13~ | 1.12~ | 1.15~ | 1.12~ | 1.26 | 1.29 | scikit-learn |
| 8 192 × 3072, k = 10 | 1.71 | 1.96 | 2.04 | 1.84 | 2.11 | 1.94 | 1.59 | 1.85 | 2.10 | 2.05 | 2.07 | 1.60 | 1.64 | 1.69 | 1.59 | 1.95 | 1.74 | scikit-learn |
| 5 000 × 4096, k = 10 | 1.29 | 1.45 | 1.40 | **0.80** | 1.40 | 1.64 | 1.27 | 1.28 | 1.25 | 1.29 | 1.24 | 1.12~ | 1.19 | 1.51 | 1.40 | 1.36 | 1.35 | scikit-learn |
| 6 000 × 4096, k = 10 | 1.43 | 1.68 | 1.93 | 1.22 | 2.01 | 1.76 | 1.66 | 1.40 | 1.44 | 1.38 | 1.32~ | 1.23 | 1.45 | 1.57 | 1.39 | 1.14~ | 1.47 | scikit-learn |
| 6 000 × 4096, k = 40 | 1.35 | 1.54 | 1.70 | **0.90** | 1.42 | 1.68 | 1.59 | 1.01~ | 1.26 | 1.18 | 1.18 | 1.07~ | 1.27 | 1.26 | 1.34 | 1.17 | 1.35 | scikit-learn |
| 7 500 × 4096, k = 10 | 1.88 | 2.08 | 1.90 | 1.18~ | 1.89 | 2.00 | 1.83 | 1.47 | 1.52 | 1.36 | 1.39 | 1.21 | 1.66 | 1.28 | 1.46 | 1.48 | 1.76 | scikit-learn |
| 7 500 × 4096, k = 50 | 1.61 | 1.72 | 1.74 | 1.13~ | 1.68 | 1.74 | 1.56 | 1.16~ | 1.17~ | 1.19 | 1.09~ | 1.04~ | 1.40 | 1.38 | 1.46 | 1.13~ | 1.31 | scikit-learn |
| 8 192 × 4096, k = 10 | 1.71 | 2.15 | 2.00 | 1.39~ | 1.96 | 2.11 | 2.33 | 1.65 | 1.75 | 1.57 | 1.54 | 1.74 | 2.09 | 1.92 | 1.63 | 1.90 | 1.89 | MLX |
| 8 192 × 4096, k = 50 | 1.74 | 1.97 | 1.99 | 1.25 | 1.91 | 1.90 | 1.75 | 1.36 | 1.32 | 1.53 | 1.24 | 1.20~ | 1.53 | 1.51 | 1.24 | 1.32 | 1.36 | MLX |

At 4 096–7 500 rows batches of 16–32 queries lost on one version or both (0.77×–0.90×,
at 4 096 × 2 560 and × 3 072, 4 500 × 2 304, 5 000 × 4 096, and 6 000 × 4 096 with
k = 40) or only
tied (1.04×~ at 7 500 × 4 096, k = 50); the `knn-k` cell at 4 096 × 2 048, 32 queries,
measured 1.35× in one long run and 0.83×–0.86× in fresh processes. From 8 192 rows every
batch won or tied, 1.20×~ at worst. So from 256 features patched dispatch also needs
8 192 rows (`knn_wide_min_samples`), recorded as `"below-crossover"`; the size floors alone
send such fits to MLX from about 1 500 features. It gives up what won at 4 096–7 500 rows
and 2 048–4 096 features, most batches there, 1.02×–3.0× (this table, `knn-k` and
`knn-structured`), and every fit of 6 000–8 191 rows at 1 000–2 047 features, which no
grid measured. A directly imported class, which has no scikit-learn to hand the fit to,
searches on the GPU from 4 096 rows as before.

Where patched dispatch uses MLX, then — tall data from 16 features, wide data from 256
features and 8 192 rows, at most 50 neighbors, on data the fit-time probe passes — the
Gaussian grids `knn`, `knn-rows` and `knn-narrow` answered single rows 1.0×–5.7× faster
than stock, 16 queries 1.15×–3.2×, 256 queries 1.3×–8.3× and 4 096 queries 2.6×–11.7×,
on both versions, and every batch size between those won or tied too (the low ends but
16 queries' are `knn-rows` shapes, measured before the fused selection kernel). With the
other laws and k of the grids below (`knn-data`, `knn-structured`, `knn-probe`, `knn-k`,
`knn-batches`), every routed batch still won or tied, 0.92×~ at worst (8 queries at
24 000 × 256, k = 50).

Queries win; the index build does not. Stock brute force stores a reference, while
mlxlearn uploads the training rows and builds the translated, scaled copy the search runs
on, with its row norms and, for float64 input, a float32 residual: 0.05×–0.09× of the
stock fit where patched dispatch uses MLX (11–254 ms against 1–16 ms, the fit columns
above, which force the MLX path and so leave out the fit-time probe below). With the
probe, a patched fit measured 0.04×–0.07× of stock (17–232 ms against 1.1–12 ms, at
8 192 × 1 024 to 1 600 000 × 64 and at 1 000 000 × 16–48 on both versions, after an idle
gap, medians of 9), the
probe itself — one first pass for 512 queries at k = 5, timed alone — 4.0–14.1 ms
back to back and 5.1–17.1 ms right after the GPU has idled (a collection and a 10 ms sleep
before each call; medians of 9, both versions) at 8 192–65 536 rows and 256–4 096
features, and 16–28 ms at a million rows and more. It is repaid after about 210–1 700
queries in batches of 256 to 4 096 — the fit's extra time over what each query of the
batch saves, at the `knn` and `knn-rows` shapes patched dispatch sends to MLX — and with
the probe after at most about 310–4 500 (17 ms of probe added to every fit below a
million rows, 28 ms from it), which is why
`knn/fit` is listed in `benchmarks/gate.py::ACCEPTED_REGRESSIONS` rather than left to slip
through the "no crossover measured, so not gated" branch. A model fitted to answer a
handful of queries is faster on stock scikit-learn.

In memory, a float64 index holds 3.0× the float32 data on the device — rows, residual and
the translated copy — and 3.5–4.2× at the peak of the build, measured at 50 000 × 32 and
128 000 × 768.

### Data the size rules cannot see

The tables above are Gaussian data, on which the search's first float32 pass certifies
every query row ([`fp32_policy.md`](fp32_policy.md) §2.1). A row it cannot certify is
searched again, with a much wider candidate set and then exactly, and that costs more than
the certified search saves. Grid `knn-data` times the same batches on other data at shapes
the rules above send to MLX, k = 10, both versions. `probe` is what a patched fit finds
when it searches 512 of its own rows with the first pass alone: how many it could not
certify, or `integer` for data on the exact integer path, which has nothing to certify:

<!-- python -m benchmarks.calibrate table knn-data: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, data | 1 query | 16 queries | 64 queries | 256 queries | 1024 queries | probe | dispatch |
|---|---|---|---|---|---|---|---|
| 1 000 000 × 16, gauss | 1.53 / 2.06 | 2.35 / 2.17 | 1.97 / 1.91 | 3.60 / 3.43 | 5.69 / 5.53 | 0/512 | MLX |
| 1 000 000 × 16, blobs | 1.56 / 1.86 | 2.36 / 2.25 | 1.99 / 1.91 | 3.67 / 3.48 | 5.91 / 5.55 | 0/512 | MLX |
| 1 000 000 × 16, uniform | 1.55 / 1.98 | 2.30 / 2.27 | 1.94 / 1.88 | 3.65 / 3.43 | 5.63 / 5.61 | 0/512 | MLX |
| 1 000 000 × 16, unscaled | **0.74** / **0.77** | 0.67~ / **0.57** | **0.34** / **0.32** | **0.30** / **0.27** | **0.23** / **0.22** | 478/512 | scikit-learn |
| 1 000 000 × 16, dups | **0.80** / 1.00~ | 1.02~ / 0.93~ | 0.69~ / 0.70~ | **0.69** / **0.66** | **0.47** / **0.45** | 498/512 | scikit-learn |
| 1 000 000 × 16, tight-blobs | 1.48 / 1.87 | 1.51 / 1.47 | 1.19~ / 1.13~ | 1.81 / 1.82 | 2.14 / 2.03 | 33/512 | scikit-learn |
| 1 000 000 × 16, binary | 1.62 / 1.64 | **0.78** / 1.86~ | 0.91~ / 3.30~ | 1.00~ / 0.95~ | 0.88~ / 0.90~ | integer | scikit-learn |
| 1 000 000 × 16, counts | 1.62 / 1.65 | **0.78** / 1.51 | 1.10~ / 1.06~ | 1.06~ / 0.96~ | 0.90~ / **0.85** | integer | scikit-learn |
| 1 000 000 × 32, gauss | 1.90 / 2.53 | 2.21 / 2.25 | 1.87 / 1.84 | 3.76 / 3.71 | 6.13 / 6.32 | 0/512 | MLX |
| 1 000 000 × 32, blobs | 1.85 / 2.34 | 2.14 / 2.00 | 2.01 / 1.97 | 4.90 / 3.45 | 6.41 / 5.92 | 0/512 | MLX |
| 1 000 000 × 32, uniform | 1.83 / 1.84 | 2.09 / 2.08 | 1.89 / 1.87 | 3.60 / 3.31 | 6.46 / 6.11 | 0/512 | MLX |
| 1 000 000 × 32, unscaled | **0.72** / **0.83** | **0.48** / **0.47** | **0.32** / **0.30** | **0.18** / **0.16** | **0.18** / **0.17** | 491/512 | scikit-learn |
| 1 000 000 × 32, dups | **0.95** / 1.19 | 0.98~ / 1.13~ | 0.83~ / 0.95~ | **0.82** / **0.79** | **0.63** / **0.62** | 500/512 | scikit-learn |
| 1 000 000 × 32, tight-blobs | 1.83 / 2.43 | **0.83** / **0.83** | 1.30~ / 1.22~ | 1.47 / 1.41 | 1.71 / 1.82 | 128/512 | scikit-learn |
| 1 000 000 × 32, binary | 1.87 / 1.85 | **0.81** / 1.81 | 1.33~ / 1.23~ | 1.18~ / 1.07~ | 1.06~ / 1.07~ | integer | scikit-learn |
| 1 000 000 × 32, counts | 1.97 / 1.96 | 0.84~ / 1.28 | 0.96~ / 0.91~ | **0.79** / **0.81** | **0.55** / **0.53** | integer | scikit-learn |
| 24 000 × 256, gauss | 1.07~ / 1.02~ | 1.08~ / 1.03~ | 1.18 / 1.21 | 2.63 / 2.95 | 4.10 / 4.31 | 0/512 | MLX |
| 24 000 × 256, blobs | 1.05~ / 1.05~ | 1.08~ / 1.07~ | 1.15 / 1.13~ | 2.44 / 2.35 | 3.95 / 3.93 | 0/512 | MLX |
| 24 000 × 256, uniform | 1.05~ / 1.01~ | 1.08~ / 1.04~ | 1.10 / 1.09 | 2.94 / 2.82 | 4.15 / 3.79 | 0/512 | MLX |
| 24 000 × 256, unscaled | **0.51** / **0.47** | **0.50** / **0.44** | **0.41** / **0.36** | **0.48** / **0.55** | 0.76~ / 0.78~ | 485/512 | scikit-learn |
| 24 000 × 256, dups | **0.59** / **0.51** | **0.56** / **0.48** | **0.50** / **0.43** | 0.67~ / **0.67** | 1.00~ / 0.95~ | 495/512 | scikit-learn |
| 24 000 × 256, tight-blobs | **0.59** / **0.50** | **0.50** / **0.47** | **0.45** / **0.40** | **0.51** / **0.59** | 0.86~ / 0.81~ | 512/512 | scikit-learn |
| 24 000 × 256, binary | 1.13 / 1.17 | 1.19 / 1.15 | 1.28 / 1.16 | 1.55 / 1.64 | 2.42 / 2.60 | integer | MLX |
| 24 000 × 256, counts | 1.18 / 1.15 | 1.15 / 1.18 | 1.23 / 1.30 | 1.57 / 1.68 | 2.82 / 2.45 | integer | MLX |
| 65 536 × 256, gauss | 1.61 / 1.55 | 1.38 / 1.44 | 2.79 / 3.10 | 3.65 / 3.60 | 6.09 / 6.49 | 0/512 | MLX |
| 65 536 × 256, blobs | 1.65 / 1.58 | 1.30 / 1.41 | 2.75 / 2.89 | 3.03 / 2.81 | 6.23 / 6.78 | 0/512 | MLX |
| 65 536 × 256, uniform | 1.53 / 1.55 | 1.27 / 1.30 | 3.14 / 2.79 | 3.64 / 3.82 | 7.04 / 7.31 | 0/512 | MLX |
| 65 536 × 256, unscaled | **0.59** / **0.55** | **0.26** / **0.35** | **0.28** / **0.36** | **0.29** / **0.22** | **0.31** / **0.29** | 480/512 | scikit-learn |
| 65 536 × 256, dups | **0.93** / **0.84** | **0.69** / **0.65** | **0.79** / 0.92~ | 1.14~ / 1.17~ | 1.88 / 1.76 | 495/512 | scikit-learn |
| 65 536 × 256, tight-blobs | **0.90** / **0.79** | **0.64** / **0.56** | 0.92~ / **0.76** | 1.00~ / 0.99~ | 1.61 / 1.54 | 512/512 | scikit-learn |
| 65 536 × 256, binary | 1.67 / 1.67 | 1.44 / 1.35 | 1.90 / 2.21 | 2.11 / 2.23 | 4.33 / 4.34 | integer | MLX |
| 65 536 × 256, counts | 1.71 / 1.69 | 1.47 / 1.48 | 1.94 / 1.94 | 2.25 / 2.13 | 4.53 / 4.29 | integer | MLX |
| 8 192 × 1024, gauss | 1.26 / 1.16 | 1.25 / 1.32 | 1.55 / 1.52 | 1.68 / 2.15 | 2.95 / 2.63 | 0/512 | MLX |
| 8 192 × 1024, blobs | 1.17 / 1.21 | 1.22~ / 1.29 | 0.97~ / 0.90~ | 1.50 / 1.36 | 2.22 / 1.76 | 14/512 | scikit-learn |
| 8 192 × 1024, uniform | 1.21 / 1.26 | 1.35 / 1.42 | 1.31~ / 1.96 | 1.97 / 1.63 | 2.82 / 2.32 | 0/512 | MLX |
| 8 192 × 1024, unscaled | **0.71** / **0.59** | **0.57** / **0.50** | **0.57** / **0.66** | **0.40** / **0.47** | **0.61** / **0.67** | 488/512 | scikit-learn |
| 8 192 × 1024, dups | **0.71** / **0.64** | **0.71** / **0.59** | **0.61** / **0.67** | **0.54** / **0.67** | **0.71** / **0.66** | 493/512 | scikit-learn |
| 8 192 × 1024, tight-blobs | **0.73** / **0.63** | **0.65** / **0.56** | **0.55** / **0.65** | **0.45** / **0.51** | **0.57** / **0.56** | 512/512 | scikit-learn |
| 8 192 × 1024, binary | 1.70 / 1.32 | 1.17 / 1.21 | 1.46 / 1.56 | 1.70 / 1.72 | 1.98 / 2.27 | integer | MLX |
| 8 192 × 1024, counts | 1.25 / 1.35 | 1.12 / 1.22~ | 1.38~ / 1.39 | 2.08 / 2.02 | 1.96 / 1.92 | integer | MLX |

`gauss`, `blobs` (100 clusters of std 1 around centres of std 3) and `uniform` are data the
first pass certifies. `unscaled` is one column uniform on 0–10 000 beside standardized
ones; `dups` repeats every row about 20 times; `tight-blobs` is 100 clusters of std 0.1;
`binary` is 0/1 features at 20%, `counts` Poisson(2) counts.

- **Far from certified.** Every shape of the three data the first pass cannot certify
  lost on some batch, on both versions, down to 0.22× at the wide shapes (256 queries at
  65 536 × 256, unscaled) and 0.16× at a million rows, and their probes found 478–512 of
  512 rows uncertified — but only 33 and 128 for tight clusters at a million rows and 16
  or 32 features, where each cluster holds 10 000 rows, still enough for the probe to
  hand them off. Gaussian and uniform data left none uncertified at every shape;
  clusters of std 1 none at 256 features or at a million rows, and 14 of 512 at
  8 192 × 1 024, where 64 queries only tied. At a million rows and 16 or 32 features
  those three won every batch here (1.53× at worst), and patched dispatch sends them to
  MLX (the width rule, `knn-narrow` above).
- **Integer data** is searched on the exact integer path, which kept up with stock brute
  force from 256 features (1.12× at worst) but, at a million rows and 16 or 32 features,
  lost batches of 16 queries and, of counts, of 256 and 1 024 (0.53×–0.85×). Below
  `knn_wide_features` it goes to scikit-learn too, with the same reason — conservatively
  at 33–255 features, which this grid did not measure on MLX.

**A few rows in a hundred are enough to lose.** A row the first pass cannot certify costs
its batch a second search, and the whole batch waits for it, so the cost is per *batch*:
with two to five rows in a hundred uncertified, most batches of 16–64 queries hold one.
Grid `knn-structured` times every batch size on data with that much structure — rows
repeated about twice (`dup2`), one column on 100 times the scale of the rest
(`scaled100`), 100 clusters of std 0.5 and 0.4 (`blobs0.5`, `blobs0.4`) around the same
centres of std 3 — beside Gaussian data, at shapes on and near the size floors, k = 10;
each cell is the worse version's ratio:

<!-- python -m benchmarks.calibrate table knn-structured: scikit-learn 1.9.0 / 1.7.2, the worse of them -->
| rows × features, data | 1 query | 2 queries | 4 queries | 8 queries | 16 queries | 32 queries | 64 queries | 128 queries | 256 queries | 512 queries | 1024 queries | 4096 queries | probe | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 000 000 × 16, gauss | 1.66 | 2.61 | 1.91 | 2.79 | 2.40 | 2.24 | 1.91 | 2.34 | 3.18 | 4.20 | 5.04 | 6.04 | 0/512 | MLX |
| 1 000 000 × 16, dup2 | 1.62 | 2.60 | 1.89 | 2.88 | 1.52 | 1.12 | 1.92 | 1.60 | 1.82 | 2.32 | 2.45 | 3.11 | 19/512 | scikit-learn |
| 1 000 000 × 16, scaled100 | 1.54 | 1.46 | 1.22 | 2.85 | 1.44 | 1.00~ | 0.98~ | 1.79 | 1.92 | 2.05 | 2.29 | 2.52 | 22/512 | scikit-learn |
| 1 000 000 × 16, blobs0.5 | 1.55 | 2.45 | 1.86 | 2.81 | 2.35 | 2.27 | 1.86 | 2.45 | 3.54 | 4.14 | 5.35 | 6.02 | 0/512 | MLX |
| 1 000 000 × 16, blobs0.4 | 1.54 | 2.46 | 1.87 | 2.72 | 2.38 | 2.25 | 1.90 | 2.39 | 3.43 | 3.80 | 5.42 | 6.16 | 0/512 | MLX |
| 1 000 000 × 32, gauss | 1.80 | 2.24 | 1.69 | 2.37 | 2.09 | 2.34 | 1.88 | 2.79 | 3.29 | 4.56 | 5.65 | — | 0/512 | MLX |
| 1 000 000 × 32, dup2 | 1.78 | 1.32 | 1.70 | 2.36 | 2.17 | 1.21~ | 1.59 | 1.70 | 2.49 | 2.92 | 3.03 | — | 27/512 | scikit-learn |
| 1 000 000 × 32, scaled100 | 1.79 | 2.22 | 1.68 | 2.37 | 1.38 | 1.24 | 1.64~ | 2.01 | 2.12 | 2.28 | 2.63 | — | 34/512 | scikit-learn |
| 1 000 000 × 32, blobs0.5 | 1.88 | 2.15 | 1.66 | 2.35 | 2.14 | 2.30 | 1.94 | 2.78 | 3.69 | 4.85 | 5.92 | — | 0/512 | MLX |
| 1 000 000 × 32, blobs0.4 | 1.86 | 2.19 | 1.70 | 2.36 | 2.17 | 2.32 | 1.93 | 2.40 | 3.65 | 4.85 | 5.88 | — | 0/512 | MLX |
| 1 000 000 × 48, gauss | 2.17 | 1.87 | 1.60 | 2.12 | 2.19 | 2.61 | 2.01 | 2.83 | 3.76 | 5.65 | 6.76 | — | 0/512 | MLX |
| 1 000 000 × 48, dup2 | 2.15 | 1.91 | 1.05~ | 2.07 | 2.10 | 2.54 | 2.18 | 2.36 | 2.42 | 2.98 | 3.58 | — | 16/512 | scikit-learn |
| 1 000 000 × 48, scaled100 | 2.16 | 1.93 | 1.06~ | 2.11 | 2.04 | 1.27~ | 1.50 | 1.55 | 1.98 | 2.81 | 3.18 | — | 28/512 | scikit-learn |
| 1 000 000 × 48, blobs0.5 | 2.13 | 1.85 | 1.54 | 2.18 | 2.16 | 2.60 | 2.01 | 2.56 | 4.10 | 5.40 | 6.83 | — | 0/512 | MLX |
| 1 000 000 × 48, blobs0.4 | 2.14 | 1.97 | 1.59 | 2.21 | 2.07 | 2.53 | 2.13 | 2.80 | 4.10 | 5.55 | 6.80 | — | 0/512 | MLX |
| 1 000 000 × 64, gauss | 2.26 | 1.72 | 1.42 | 2.08 | 2.23 | 2.67 | 2.25 | 3.46 | 4.44 | 6.33 | 7.90 | — | 0/512 | MLX |
| 1 000 000 × 64, dup2 | 2.36 | 1.69 | 1.44 | 2.01 | 2.24 | 1.85 | 1.96 | 2.35 | 2.68 | 3.56 | 4.45 | — | 22/512 | scikit-learn |
| 1 000 000 × 64, scaled100 | 2.26 | 1.73 | 1.44 | 1.38 | 1.13 | 1.97 | 1.91 | 2.38 | 2.47 | 3.14 | 3.38 | — | 30/512 | scikit-learn |
| 1 000 000 × 64, blobs0.5 | 2.32 | 1.73 | 1.48 | 1.99 | 2.21 | 2.65 | 2.14 | 3.21 | 4.47 | 6.37 | 8.36 | — | 0/512 | MLX |
| 1 000 000 × 64, blobs0.4 | 2.35 | 1.75 | 1.65 | 2.12 | 2.15 | 2.81 | 2.15 | 3.39 | 4.84 | 5.51 | 8.34 | — | 0/512 | MLX |
| 24 000 × 256, gauss | 1.01~ | 1.07~ | 0.98~ | 0.97~ | 1.04~ | 1.14 | 1.09 | 1.21~ | 2.36 | 3.20 | 3.69 | 4.38 | 0/512 | MLX |
| 24 000 × 256, dup2 | 1.03~ | 1.02~ | 1.07~ | 0.99~ | **0.57** | **0.58** | 1.09~ | **0.80** | 1.62 | 2.35 | 2.65 | 3.34 | 23/512 | scikit-learn |
| 24 000 × 256, scaled100 | 1.08~ | 1.02~ | **0.50** | 1.00~ | 1.04~ | **0.69** | **0.67** | **0.72** | 1.47 | 2.08 | 2.58 | 2.93 | 27/512 | scikit-learn |
| 24 000 × 256, blobs0.5 | 1.06~ | 1.02~ | 1.02~ | 0.97~ | **0.53** | 1.10~ | 1.13 | 1.28 | 1.78 | 2.33 | 3.27 | 3.54 | 7/512 | scikit-learn |
| 24 000 × 256, blobs0.4 | 1.00~ | 1.01~ | 1.04~ | 0.99~ | **0.58** | **0.64** | **0.72** | **0.83** | 1.86 | 2.20 | 2.88 | 3.20 | 28/512 | scikit-learn |
| 30 000 × 256, gauss | 1.06~ | 1.13~ | 1.12 | 1.04~ | 1.07 | 1.07~ | 1.21 | 1.32 | 2.67 | 3.36 | 4.31 | 5.51 | 0/512 | MLX |
| 30 000 × 256, dup2 | 1.12 | 1.10 | 1.13 | 1.05~ | **0.56** | **0.68** | **0.77** | 0.89~ | 1.70 | 2.41 | 3.19 | 5.22 | 23/512 | scikit-learn |
| 30 000 × 256, scaled100 | 1.11 | 1.15 | 1.12 | **0.59** | **0.61** | **0.72** | **0.68** | 0.97~ | 1.74 | 2.06 | 2.71 | 4.48 | 37/512 | scikit-learn |
| 30 000 × 256, blobs0.5 | 1.08~ | 1.11~ | 1.13 | 1.00~ | **0.52** | 1.14 | **0.76** | 1.01~ | 1.95 | 2.69 | 3.61 | 6.00 | 3/512 | scikit-learn |
| 30 000 × 256, blobs0.4 | 1.14 | 1.09~ | 1.23 | 1.07~ | **0.55** | **0.69** | **0.68** | **0.84** | 1.54 | 2.27 | 2.95 | 5.37 | 19/512 | scikit-learn |
| 65 536 × 256, gauss | 1.70 | 1.41 | 1.35 | 1.31 | 1.33 | 1.49 | 2.73 | 2.84 | 3.08 | 4.02 | 6.25 | 10.02 | 0/512 | MLX |
| 65 536 × 256, dup2 | 1.62 | 1.35 | 1.37 | 1.27 | **0.69** | **0.87** | 1.68 | 2.09 | 2.01 | 2.94 | 5.47 | 6.71 | 32/512 | scikit-learn |
| 65 536 × 256, scaled100 | 1.55 | 1.38 | 1.37 | **0.77** | 1.39 | **0.83** | 1.33 | 1.82 | 2.05 | 3.19 | 4.26 | 5.48 | 46/512 | scikit-learn |
| 65 536 × 256, blobs0.5 | 1.59 | 1.42 | 1.38 | 1.30 | 1.32 | 0.91~ | 1.38 | 2.02 | 2.49 | 3.48 | 4.90 | 7.14 | 9/512 | scikit-learn |
| 65 536 × 256, blobs0.4 | 1.62 | 1.38 | 1.37 | 1.30 | **0.70** | 0.95~ | 1.49 | 1.79 | 2.33 | 2.90 | 4.54 | 5.76 | 40/512 | scikit-learn |
| 100 000 × 256, gauss | 1.85 | 1.61 | 1.40 | 1.34 | 1.18 | 1.43 | 2.98 | 3.51 | 4.27 | 4.86 | 8.37 | 12.61 | 0/512 | MLX |
| 100 000 × 256, dup2 | 1.90 | 1.57 | 1.51 | **0.91** | **0.88** | 1.04~ | 1.73 | 2.10 | 2.32 | 3.96 | 5.49 | 7.58 | 36/512 | scikit-learn |
| 100 000 × 256, scaled100 | 1.89 | 1.56 | 1.43 | **0.73** | **0.67** | **0.81** | 1.40 | 1.87 | 2.13~ | 4.21 | 4.71 | 6.21 | 53/512 | scikit-learn |
| 100 000 × 256, blobs0.5 | 1.82 | 1.49 | 1.47 | 1.30 | 1.27 | 1.37 | 1.75 | 2.27 | 3.01 | 4.11 | 5.46 | 7.58 | 10/512 | scikit-learn |
| 100 000 × 256, blobs0.4 | 1.86 | 1.03~ | 1.52 | 1.35 | **0.72** | 0.99~ | 1.54 | 1.62 | 2.56 | 3.62 | 5.40 | 6.44 | 44/512 | scikit-learn |
| 12 000 × 512, gauss | 1.03~ | 1.07~ | 1.12 | 1.19 | 1.20~ | 1.27 | 1.41 | 1.50 | 1.29 | 1.93 | 2.41 | 2.35 | 0/512 | MLX |
| 12 000 × 512, dup2 | 1.01~ | 1.14~ | 1.15 | 1.10~ | **0.67** | **0.65** | **0.77** | 1.01~ | 0.89~ | 1.44 | 1.73 | 1.91 | 22/512 | scikit-learn |
| 12 000 × 512, scaled100 | 0.97~ | **0.62** | **0.66** | 1.21 | **0.64** | **0.65** | **0.82** | 0.92~ | **0.71** | 1.28~ | 1.44 | 1.39 | 56/512 | scikit-learn |
| 12 000 × 512, blobs0.5 | 1.02~ | 1.19 | 1.22 | **0.57** | **0.57** | **0.61** | **0.74** | 1.03~ | 0.89~ | 1.17 | 1.42 | 1.32 | 50/512 | scikit-learn |
| 12 000 × 512, blobs0.4 | 1.03~ | **0.65** | **0.62** | **0.53** | **0.59** | **0.59** | **0.65** | 0.80~ | **0.57** | **0.78** | 0.88~ | 1.06~ | 146/512 | scikit-learn |
| 50 000 × 512, gauss | 2.01 | 1.73 | 1.70 | 1.62 | 1.74 | 1.78 | 2.16 | 3.57 | 2.42 | 2.98 | 4.45 | 6.27 | 0/512 | MLX |
| 50 000 × 512, dup2 | 1.88 | 1.74 | 1.61 | 1.51 | 0.95~ | 1.45 | 1.36 | 2.34 | 1.91 | 2.52 | 3.39 | 4.55 | 27/512 | scikit-learn |
| 50 000 × 512, scaled100 | 1.88 | 1.59 | 1.58 | **0.76** | **0.88** | 1.03~ | 1.29~ | 2.03 | 1.70 | 2.07 | 2.47 | 3.25 | 104/512 | scikit-learn |
| 50 000 × 512, blobs0.5 | 1.87 | 1.79 | 1.15~ | 1.07~ | **0.83** | 0.98~ | 1.29~ | 1.68 | 1.48 | 2.02 | 2.43 | 2.87 | 120/512 | scikit-learn |
| 50 000 × 512, blobs0.4 | 0.98~ | 1.65 | **0.78** | **0.81** | **0.74** | **0.79** | 0.98~ | 1.14~ | 1.36 | 1.41 | 1.61 | 2.18 | 252/512 | scikit-learn |
| 16 384 × 768, gauss | 1.45 | 1.48 | 1.37 | 1.34 | 1.37 | 1.62 | 2.08 | 1.55 | 2.56 | 2.47 | 3.07 | 4.57 | 0/512 | MLX |
| 16 384 × 768, dup2 | 1.38 | 1.41 | 1.34 | 1.38 | **0.88** | **0.78** | 1.22~ | 1.18~ | 1.51 | 1.87 | 2.21 | 3.29 | 22/512 | scikit-learn |
| 16 384 × 768, scaled100 | 1.48 | 1.41 | **0.84** | **0.72** | **0.78** | **0.79** | 1.07~ | **0.82** | 1.20~ | 1.38 | 1.99 | 2.24 | 107/512 | scikit-learn |
| 16 384 × 768, blobs0.5 | **0.70** | 1.49 | **0.73** | **0.71** | **0.69** | **0.81** | 0.90~ | 0.90~ | 0.81~ | 0.94~ | 1.06~ | 1.40 | 245/512 | scikit-learn |
| 16 384 × 768, blobs0.4 | **0.71** | **0.79** | **0.68** | **0.65** | **0.65** | **0.70** | **0.78** | **0.63** | **0.70** | **0.73** | 0.89~ | 1.10~ | 401/512 | scikit-learn |
| 8 192 × 1024, gauss | 1.22 | 1.33 | 1.31 | 1.32 | 1.27 | 1.33 | 1.50 | 1.44 | 1.95 | 2.15 | 2.26 | 2.84 | 0/512 | MLX |
| 8 192 × 1024, dup2 | 1.22 | 1.37 | 1.31 | **0.76** | 1.29 | **0.73** | 0.84~ | 0.96~ | 1.06~ | 1.46 | 1.85 | 2.27 | 53/512 | scikit-learn |
| 8 192 × 1024, scaled100 | 1.19~ | 1.29 | **0.66** | **0.71** | **0.77** | **0.66** | 0.86~ | 0.86~ | 0.88~ | 1.05~ | 1.21 | 1.54 | 106/512 | scikit-learn |
| 8 192 × 1024, blobs0.5 | 1.19~ | 1.33 | **0.68** | **0.62** | **0.56** | **0.63** | **0.68** | **0.57** | **0.63** | **0.63** | **0.71** | 0.89~ | 327/512 | scikit-learn |
| 8 192 × 1024, blobs0.4 | **0.62** | **0.64** | **0.65** | **0.64** | **0.57** | **0.58** | **0.65** | **0.53** | **0.53** | **0.50** | **0.62** | **0.71** | 461/512 | scikit-learn |
| 4 096 × 2048, gauss | 1.13 | 1.10 | 1.09~ | 1.19 | 1.21 | **0.80** | 1.38 | 1.34 | 1.36 | 1.47 | 1.68 | 2.24 | 0/512 | scikit-learn |
| 4 096 × 2048, dup2 | 1.17 | 1.10~ | 1.02~ | 1.28 | 1.25 | 0.96~ | 0.90~ | 1.24 | 1.10~ | 1.34 | 1.72 | 1.96 | 2/512 | scikit-learn |
| 4 096 × 2048, scaled100 | 1.15 | 1.11~ | 1.12~ | 1.20 | **0.62** | **0.54** | **0.75** | 0.98~ | 0.90~ | 1.22 | 1.36 | 1.62 | 34/512 | scikit-learn |
| 4 096 × 2048, blobs0.5 | 0.63~ | **0.82** | **0.70** | **0.65** | **0.56** | **0.55** | **0.52** | **0.47** | **0.45** | **0.56** | **0.70** | **0.76** | 275/512 | scikit-learn |
| 4 096 × 2048, blobs0.4 | **0.66** | **0.60** | **0.57** | **0.57** | **0.50** | **0.39** | **0.38** | **0.36** | **0.38** | **0.44** | **0.48** | **0.52** | 473/512 | scikit-learn |

These four laws lost some batch at 32 of the 52 shapes measured, down to 0.36× — at 256
features batches of 4–128 queries, 0.50×–0.91×, which the probe's first rule (more than
8 of 64 rows uncertified) mostly let through. The probe found 3–473 of 512 rows
uncertified at every one of the 32, and 3 were enough to lose: clusters of std 0.5 at
30 000 × 256, 0.52× at 16 queries. Where it found none — clusters at a million rows and
16–64 features — every batch won or tied, as it did on Gaussian data at every shape the
size rules send to MLX (at 4 096 × 2 048, below the row floor, a batch of 32 lost,
0.80×). So patched mode
probes 512 of the rows of every fit the size rules would send to MLX and hands it to
scikit-learn if it cannot certify any one of them (`knn_probe_rows = 512`,
`knn_probe_max_uncertified = 0`), recorded as `"data-probe"` with the count on the event.
It probes with the fitted k, since the first pass's noise band grows with k and the width
(the `knn-k` table above). Sixty-four rows, the probe's first size, see no uncertified
row about half the time when one row in a hundred is; 256 miss one time in thirteen, and
512 one time in 170 (grid `knn-probe`, below).

What the rule gives up is those laws where a second search was affordable: 11 of the 52,
which won or tied every batch at shapes the size rules route — rows repeated twice at
50 000 × 512 and at a million rows and 16–64 features, the column on a larger scale at a
million rows and 16–64 features, and clusters of std 0.5 at 65 536 and 100 000 × 256 —
up to 7.9× at 4 096 queries; and the
wins of the other 32 at large batches (up to 8.0×).

**How many rows the probe needs.** On clusters of std 0.5 at 24 000 and 30 000 × 256,
about one row in a hundred cannot be certified, few enough that a probe of 256 rows sees
none of them in some draws of the data. Grid `knn-probe` times 64 draws of that law at
each shape, and 8 draws each of Gaussian, low-rank (three latent directions plus noise of
std 0.05) and std-1 cluster data there and at 8 192 × 1 024, batches of 16–256 queries,
k = 10; each draw records what probes of 256 and of 512 rows found. A row of the table
counts the draws each probe passed and gives the worst batch among them, per version:

<!-- python -m benchmarks.calibrate table knn-probe: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, data | draws | passed by 256 rows | worst batch of those | passed by 512 rows | worst batch of those | dispatch |
|---|---|---|---|---|---|---|
| 24 000 × 256, gauss | 8 | 8 | 1.01~ / 1.03~ | 8 | 1.01~ / 1.03~ | MLX for 8 of 8 |
| 24 000 × 256, blobs | 8 | 8 | 1.05~ / 1.03~ | 8 | 1.05~ / 1.03~ | MLX for 8 of 8 |
| 24 000 × 256, blobs0.5 | 64 | 4 | **0.63** / **0.56** | 0 | — | MLX for 0 of 64 |
| 24 000 × 256, lowrank | 8 | 8 | 1.02~ / 1.01~ | 8 | 1.02~ / 1.01~ | MLX for 8 of 8 |
| 30 000 × 256, gauss | 8 | 8 | 1.04~ / 1.09~ | 8 | 1.04~ / 1.09~ | MLX for 8 of 8 |
| 30 000 × 256, blobs | 8 | 8 | 1.03~ / 1.06~ | 6 | 1.03~ / 1.06~ | MLX for 6 of 8 |
| 30 000 × 256, blobs0.5 | 64 | 5 | **0.75** / **0.61** | 0 | — | MLX for 0 of 64 |
| 30 000 × 256, lowrank | 8 | 8 | 1.05~ / 1.05~ | 8 | 1.05~ / 1.05~ | MLX for 8 of 8 |
| 8 192 × 1024, gauss | 8 | 7 | 1.16 / 1.28 | 6 | 1.21~ / 1.30 | MLX for 6 of 8 |
| 8 192 × 1024, blobs | 8 | 0 | — | 0 | — | MLX for 0 of 8 |
| 8 192 × 1024, lowrank | 8 | 8 | 1.11~ / 1.25~ | 7 | 1.11~ / 1.25~ | MLX for 7 of 8 |

A probe of 256 rows passed 9 of the 128 clustered draws, and 8 of those 9 lost a batch of
16–64 queries on one version or both, down to 0.56×; with 512 rows it passed none — it
found 1–11 of 512 uncertified in every one. It gave up 5 of the 72 other draws, which
the smaller probe passed and which won or tied every batch (1.06×–3.7×): two of Gaussian
data and one low-rank at 8 192 × 1 024, and two of std-1 clusters at 30 000 × 256. Every
draw it passed won or tied every batch, 1.01×~ at worst.

The probe costs the fit one first pass for 512 queries — 5.1–17.1 ms after an idle gap at
256–4 096 features and 16–28 ms at a million rows and more (above) — on top of the index
build (the `knn/fit` accepted regression), and a fit it hands over has paid for both
before scikit-learn's own. A directly imported estimator does not probe: it searches on
MLX, certified, whatever the data.

### Narrow data goes to scikit-learn, whatever its size

With at most 15 features, scikit-learn's default `algorithm="auto"` builds a KD-tree.
Grid `knn-tree`, against the tree on Gaussian data — the fit column is the tree's build time over MLX's
index build, and the single-row column times a different row in every pair, since what
one row costs a tree depends on where it lands:

<!-- python -m benchmarks.calibrate table knn-tree: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | 1 query | 16 queries | 256 queries | 4096 queries | fit | dispatch |
|---|---|---|---|---|---|---|
| 4 096 × 2 | **0.27** / **0.19** | **0.27** / **0.22** | **0.32** / **0.28** | **0.31** / 0.52~ | **0.48** / **0.48** | scikit-learn |
| 65 536 × 2 | **0.26** / **0.21** | **0.22** / **0.18** | 0.14~ / 0.14~ | **0.14** / **0.14** | 3.31 / 3.43 | scikit-learn |
| 1 048 576 × 3 | **0.18** / **0.13** | **0.05** / **0.07** | **0.04** / **0.04** | **0.03** / **0.03** | 14.66 / 13.62 | scikit-learn |
| 65 536 × 4 | **0.27** / **0.21** | **0.28** / **0.23** | 0.42~ / 0.31~ | **0.49** / **0.50** | 3.92 / 3.81 | scikit-learn |
| 1 048 576 × 5 | **0.18** / **0.18** | **0.17** / **0.15** | **0.17** / **0.17** | **0.15** / **0.13** | 17.82 / 14.55 | scikit-learn |
| 65 536 × 6 | **0.32** / **0.24** | **0.53** / **0.52** | 1.00~ / 0.73~ | 1.98 / 1.97 | 6.16 / 5.31 | scikit-learn |
| 262 144 × 6 | **0.32** / **0.22** | **0.36** / **0.32** | 0.97~ / **0.83** | **0.74** / **0.64** | 11.05 / 9.26 | scikit-learn |
| 4 096 × 8 | **0.33** / **0.24** | **0.45** / **0.40** | 2.15 / 2.11 | 4.40 / 4.49 | **0.65** / **0.61** | scikit-learn |
| 65 536 × 8 | **0.37** / **0.29** | **0.85** / **0.79** | 1.48 / 1.45~ | 4.42 / 4.24 | 5.85 / 5.18 | scikit-learn |
| 262 144 × 8 | **0.43** / **0.29** | 0.98~ / **0.87** | 2.34 / 2.23 | 2.41 / 2.15 | 11.87 / 9.22 | scikit-learn |
| 1 048 576 × 8 | **0.27** / **0.23** | **0.41** / **0.74** | 1.32 / 0.98~ | 1.29 / 1.17 | 19.15 / 15.18 | scikit-learn |
| 4 096 × 12 | **0.33** / **0.24** | **0.73** / **0.63** | 4.74 / 4.40 | 10.39 / 10.54 | **0.68** / **0.64** | scikit-learn |
| 65 536 × 12 | **0.67** / **0.61** | 2.93 / 2.65 | 7.51 / 7.19 | 24.65 / 22.78 | 5.16 / 4.60 | scikit-learn |
| 262 144 × 12 | 0.88~ / 0.89~ | 5.08 / 4.59 | 12.73 / 11.94 | 20.97 / 21.89 | 9.43 / 7.91 | scikit-learn |
| 1 048 576 × 12 | 1.12~ / 1.13~ | 3.22 / 4.38 | 11.77 / 12.44 | 14.89 / 14.10 | 17.06 / 14.57 | scikit-learn |
| 16 384 × 14 | **0.46** / **0.44** | 1.99 / 2.10 | 12.52 / 11.62 | 28.70 / 25.48 | 2.30 / 1.87 | scikit-learn |
| 262 144 × 14 | 1.47~ / 1.51~ | 9.47 / 7.79 | 26.29 / 21.06 | 47.59 / 38.76 | 10.58 / 7.81 | scikit-learn |
| 65 536 × 15 | 0.97~ / 1.02~ | 5.18 / 5.44 | 15.34 / 14.91 | 47.23 / 43.45 | 5.35 / 4.69 | scikit-learn |

Grid `knn-tree-lowrank`: the same, on data of low intrinsic dimension — three latent
dimensions embedded in 8 or 12 features, plus 5% noise:

<!-- python -m benchmarks.calibrate table knn-tree-lowrank: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | 1 query | 16 queries | 256 queries | 4096 queries | fit | dispatch |
|---|---|---|---|---|---|---|
| 65 536 × 8 | **0.28** / **0.23** | **0.26** / **0.24** | **0.36** / **0.42** | **0.49** / **0.47** | 5.29 / 4.70 | scikit-learn |
| 262 144 × 8 | **0.19** / **0.19** | **0.13** / **0.12** | **0.21** / **0.22** | **0.15** / **0.15** | 10.17 / 9.96 | scikit-learn |
| 65 536 × 12 | **0.25** / **0.23** | **0.28** / **0.26** | **0.50** / **0.48** | **0.67** / **0.68** | 4.93 / 4.53 | scikit-learn |
| 262 144 × 12 | **0.19** / **0.18** | **0.16** / **0.15** | **0.32** / **0.32** | **0.25** / **0.25** | 8.96 / 7.34 | scikit-learn |

A single row is where the tree wins: faster than an MLX call at every size measured
up to 8 features (0.13×–0.43× for MLX, both versions), and at 12 features up to
65 536 rows (0.24×–0.67×); at 2–5 features it won every batch. Single rows tie at
262 144 × 12 and from there up, and cost a tree very differently from row to row — the
`~` there is that spread, not a small sample. Gaussian data is where a KD-tree does
worst. On data of low intrinsic dimension, the other extreme, the tree won every batch.

Where scikit-learn would not build a tree — `algorithm="brute"`, a metric the tree does
not take, `n_neighbors` of half the rows or more — its brute force answers small batches
on narrow data faster too (grid `knn-narrow-brute`):

<!-- python -m benchmarks.calibrate table knn-narrow-brute: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | 1 query | 16 queries | 256 queries | 4096 queries | dispatch |
|---|---|---|---|---|---|
| 8 192 × 2 | **0.68** / **0.70** | **0.65** / **0.73** | **0.71** / **0.72** | 1.66 / 1.62 | scikit-learn |
| 131 072 × 2 | **0.70** / **0.66** | **0.62** / **0.62** | 1.71 / 1.74 | 4.15 / 3.88 | scikit-learn |
| 524 288 × 2 | **0.78** / 0.89~ | 1.15 / 1.17 | 2.11 / 1.70 | 3.55 / 3.03 | scikit-learn |
| 2 097 152 × 2 | **0.91** / 1.51 | 0.78~ / 1.27~ | 1.09~ / 1.04~ | 1.22 / 1.17 | scikit-learn |
| 8 192 × 8 | **0.71** / **0.70** | **0.71** / **0.72** | **0.76** / 0.87~ | 1.74 / 1.68 | scikit-learn |
| 131 072 × 8 | **0.86** / 0.90~ | **0.66** / **0.65** | 1.78 / 1.73 | 4.60 / 4.29 | scikit-learn |
| 524 288 × 8 | 1.50 / 1.38 | 1.30 / 1.29 | 2.23 / 2.07 | 5.98 / 5.49 | scikit-learn |
| 2 097 152 × 8 | 1.93 / 3.37 | 2.07 / 2.06 | 4.34 / 3.90 | 5.99 / 5.67 | scikit-learn |

With the fused selection kernel MLX won every batch at 8 features from 524 288 rows (1.29×
at worst), but at 2 features a single row lost at every size measured, on 1.9.0 up to
2 097 152 rows (0.78×–0.91× from 524 288), and below 524 288 rows batches of 1–256
queries lost at both widths (0.62×–0.86×). Patched dispatch therefore hands every fit
with at most 15 features to scikit-learn: to its KD-tree where it would build one (reason
`"tree-wins"`), and to its brute force otherwise (`knn_min_features = 16`; from 16
features, `knn-narrow` above).

What that gives up is real. On Gaussian data MLX answered 256-query batches
1.3×–26× faster than the tree from 8 features up (one tie, 1 048 576 × 8 on 1.7.2),
16-query batches 2.6×–9.5× faster from 12 features and 65 536 rows, and built its
index 3×–19× faster from 65 536 rows. A workload known to query in batches can
import `mlxlearn.neighbors` directly, which always searches on MLX.

### The published neighbor cases

The `published` profile's neighbor cases, reproducible with the command at the top: the
Layer 1 classes forced onto the MLX device against stock brute force, k = 10, 25
interleaved pairs, scikit-learn 1.9.0. `dispatch` is where patched dispatch sends the
fit; cases it hands to scikit-learn are on the record as evidence, not claims.

| samples × features | queries | mlxlearn (ms) | scikit-learn (ms) | median | best | dispatch |
|---|---|---|---|---|---|---|
| 1 000 × 32 | 256 | 1.32 | 1.35 | 1.03×~ | 1.02× | scikit-learn |
| 4 096 × 64 | 256 | 1.78 | 1.91 | 1.08×~ | 1.09× | scikit-learn |
| 32 768 × 16 | 256 | 4.81 | 3.34 | **0.70×** | 1.28× | scikit-learn |
| 50 000 × 32 | 256 | 6.19 | 4.60 | 0.74×~ | 1.08× | scikit-learn |
| 8 192 × 768 | 256 | 4.08 | 6.67 | 1.64× | 1.36× | MLX |
| 131 072 × 128 | 256 | 8.08 | 20.63 | 2.55× | 2.41× | scikit-learn |
| 128 000 × 768 | 256 | 12.01 | 63.74 | 5.31× | 6.70× | MLX |
| 1 000 000 × 16 | 256 | 42.69 | 45.13 | 1.06× | 1.02× | MLX |
| 1 000 000 × 64 | 256 | 52.30 | 73.38 | 1.40× | 1.49× | MLX |
| 1 000 × 32 | 1 | 1.02 | 0.92 | **0.90×** | 1.22× | scikit-learn |
| 1 000 × 32 | 64 | 1.10 | 1.10 | 1.00×~ | 1.02× | scikit-learn |
| 1 000 × 32 | 4096 | 6.39 | 2.42 | **0.38×** | 0.34× | scikit-learn |
| 8 192 × 768 | 1 | 0.92 | 1.45 | 1.58× | 1.45× | MLX |
| 8 192 × 768 | 64 | 1.93 | 3.17 | 1.64× | 1.45× | MLX |
| 8 192 × 768 | 4096 | 37.07 | 94.98 | 2.56× | 2.98× | MLX |
| 131 072 × 128 | 1 | 1.42 | 2.50 | 1.76× | 1.84× | scikit-learn |
| 131 072 × 128 | 64 | 5.70 | 5.95 | 1.05×~ | 1.00× | scikit-learn |
| 131 072 × 128 | 4096 | 85.23 | 342.36 | 4.02× | 4.14× | scikit-learn |
| 1 000 000 × 16 | 1 | 1.90 | 3.08 | 1.62× | 1.76× | MLX |
| 1 000 000 × 16 | 64 | 11.36 | 14.82 | 1.30× | 1.24× | MLX |
| 1 000 000 × 16 | 4096 | 702.62 | 829.28 | 1.18× | 1.14× | MLX |
| 1 000 000 × 64 | 1 | 2.26 | 5.17 | 2.29× | 2.37× | MLX |
| 1 000 000 × 64 | 64 | 14.63 | 25.43 | 1.74× | 1.83× | MLX |
| 1 000 000 × 64 | 4096 | 732.02 | 1346.33 | 1.84× | 1.61× | MLX |

| samples × features | mlxlearn fit (ms) | scikit-learn fit (ms) | ratio | dispatch |
|---|---|---|---|---|
| 1 000 × 32 | 1.04 | 0.20 | **0.20×** | scikit-learn |
| 4 096 × 64 | 1.65 | 0.25 | **0.15×** | scikit-learn |
| 32 768 × 16 | 2.30 | 0.28 | **0.12×** | scikit-learn |
| 50 000 × 32 | 4.14 | 0.42 | **0.10×** | scikit-learn |
| 8 192 × 768 | 10.44 | 0.99 | **0.10×** | MLX |
| 131 072 × 128 | 26.45 | 2.25 | **0.09×** | scikit-learn |
| 128 000 × 768 | 136.91 | 12.23 | **0.09×** | MLX |
| 1 000 000 × 16 | 33.07 | 2.14 | **0.06×** | MLX |
| 1 000 000 × 64 | 104.11 | 7.69 | **0.07×** | MLX |

131 072 × 128, which the previous floors sent to MLX, won single rows and large batches
here, but 64 queries only tied (1.05×~), and at 120 000 × 128 in the `knn` table 32
queries lost (0.90×–0.93×): patched dispatch now hands it to scikit-learn. These cases
were measured before the first pass selected its candidates with the fused kernel. Then,
1 000 000 × 16 won every batch timed here (1.06×–1.62×) but lost batches of 16 and
1 024 queries in the `knn-narrow` table, and patched dispatch handed it to scikit-learn;
with the kernel it won every batch of `knn-narrow` (1.63× at worst), and 1 000 000 rows
from 16 features is the narrowest corner it sends to MLX.

## SVC — width decides

Grid `svc`: binary fits on the benchmark data (`y = x₀ + noise > 0`), against LIBSVM.
Each cell is stock time over MLX time, scikit-learn 1.9.0 / 1.7.2 (— where a configuration
was not measured), median of 5 interleaved pairs; bold and `~` as above. C = 1 and 10
everywhere, 0.1 and 100 around the floors:

<!-- python -m benchmarks.calibrate table svc: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | rbf, C = 0.1 | rbf, C = 1 | rbf, C = 10 | rbf, C = 100 | poly, C = 0.1 | poly, C = 1 | poly, C = 10 | poly, C = 100 | dispatch |
|---|---|---|---|---|---|---|---|---|---|
| 4 000 × 16 | — | 1.12 / 1.00~ | **0.39** / **0.35** | — | — | **0.63** / **0.58** | **0.16** / **0.15** | — | scikit-learn |
| 4 000 × 32 | — | 1.23 / 1.11 | **0.53** / **0.49** | — | — | **0.86** / **0.79** | **0.55** / **0.50** | — | scikit-learn |
| 8 000 × 32 | — | 1.23 / 1.16 | **0.52** / **0.50** | — | — | **0.83** / **0.78** | **0.39** / **0.37** | — | scikit-learn |
| 4 000 × 64 | 2.64 / 2.39 | 1.57 / 1.45 | **0.80** / **0.73** | **0.79** / **0.74** | 2.22 / 1.98 | 1.17 / 1.06 | **0.83** / **0.77** | **0.85** / **0.77** | scikit-learn |
| 1 000 × 128 | 2.18 / 1.96 | 1.09 / 1.00~ | **0.74** / **0.67** | **0.75** / **0.68** | 1.70 / 1.54 | **0.90** / **0.81** | **0.68** / **0.61** | **0.68** / **0.61** | scikit-learn |
| 1 500 × 128 | — | 1.38 / 1.29 | **0.90** / **0.83** | — | — | 1.06 / 1.00~ | **0.87** / **0.79** | — | scikit-learn |
| 2 000 × 128 | 3.09 / 2.78 | 1.65 / 1.51 | 1.04 / 0.95 | 1.05~ / **0.95** | 2.89 / 2.62 | 1.29 / 1.17 | 1.01~ / **0.92** | 1.01 / **0.90** | scikit-learn |
| 2 500 × 128 | 3.72 / 3.35 | 1.81 / 1.66 | 1.13 / 1.02~ | 1.13 / 1.04 | 2.85 / 2.51 | 1.41 / 1.27 | 1.11 / 1.03 | 1.08 / 1.02 | scikit-learn |
| 3 000 × 128 | 3.96 / 3.55 | 2.00 / 1.90 | 1.25 / 1.13 | 1.23 / 1.11 | 3.06 / 2.78 | 1.53 / 1.36 | 1.12 / 1.04~ | 1.14 / 1.04 | scikit-learn |
| 4 000 × 128 | 4.06 / 3.66 | 2.27 / 2.09 | 1.36 / 1.26 | 1.38 / 1.26 | 3.98 / 3.59 | 1.69 / 1.55 | 1.28 / 1.19 | 1.26 / 1.19 | scikit-learn |
| 8 000 × 128 | — | 2.55 / 2.42 | 1.54 / 1.43 | — | — | 1.83 / 1.70 | 1.54 / 1.46 | — | scikit-learn |
| 1 000 × 256 | 3.64 / 3.24 | 1.88 / 1.74 | 1.20 / 1.09 | 1.20 / 1.08 | 2.44 / 2.18 | 1.58 / 1.40 | 1.15 / 1.04 | 1.16 / 1.04 | scikit-learn |
| 1 500 × 256 | 4.98 / 4.55 | 2.41 / 2.21 | 1.50 / 1.41 | 1.53 / 1.42 | 3.78 / 3.42 | 1.85 / 1.67 | 1.53 / 1.36 | 1.53 / 1.37 | MLX at rbf, C = 0.1, rbf, C = 1 |
| 2 000 × 256 | 5.38 / 4.88 | 2.73 / 2.52 | 1.77 / 1.63 | 1.77 / 1.59 | 3.99 / 3.56 | 2.25 / 2.01 | 1.73 / 1.57 | 1.73 / 1.56 | MLX at rbf, C = 0.1, rbf, C = 1 |
| 4 000 × 256 | 7.09 / 6.48 | 3.68 / 3.43 | 2.44 / 2.23 | 2.43 / 2.27 | 6.32 / 5.80 | 2.82 / 2.58 | 2.42 / 2.24 | 2.42 / 2.21 | MLX at rbf, C = 0.1, rbf, C = 1 |
| 8 000 × 256 | — | 4.38 / 4.05 | 3.16 / 2.83 | — | — | 3.28 / 3.00 | 3.83 / 3.52 | — | MLX at rbf, C = 1 |
| 1 000 × 512 | 5.42 / 4.98 | 3.08 / 2.79 | 1.91 / 1.74 | 1.91 / 1.74 | 4.39 / 3.91 | 2.66 / 2.38 | 2.12 / 1.91 | 2.13 / 1.91 | scikit-learn |
| 2 000 × 512 | 8.38 / 7.49 | 4.22 / 3.84 | 2.73 / 2.49 | 2.70 / 2.50 | 7.02 / 6.15 | 3.75 / 3.40 | 3.00 / 2.68 | 3.00 / 2.65 | MLX at rbf, C = 0.1, rbf, C = 1 |

SMO is sequential, and its iteration counts match LIBSVM's to within a few percent, so the
comparison is per iteration. At 4 000 rows and C = 10, mlxlearn spent 27–41 µs an
iteration from 16 to 256 features — nearly flat, because up to the kernel cache's
capacity (about 7 240 rows per one-vs-one pair at `cache_size=200`) a pair's whole Gram is
one MLX matmul and the loop then runs over it in NumPy — while LIBSVM spent 11 µs at 16
features and 103 µs at 256, because every kernel row it misses costs `O(n·d)`; once its
cache holds the rows an iteration revisits, a few µs. On this target, at the default
`gamma` and `degree`, they cross near 100 features: below 128 features rbf and poly lost
at C = 10 and 100 everywhere (0.15×–0.85×), and poly at C = 1 up to 32 features; from 128
features they won at every C — once the fit had rows. At 128 features 1 000–2 000 rows
still lost at C = 10 and above on one version or both (0.61×–0.95×; 2 000 × 128 on 1.7.2
only), and from 3 000 rows every kernel and C won or tied on both.

**The labels move the crossing.** How many rows become support vectors, and so how many
iterations revisit rows LIBSVM has cached, is set by the labels, which a fit cannot see
before it runs. Grid `svc-hard`: labels with little or no signal — `x₀ + N(0, 3²) > 0`
(noisy), random and XOR, which make nearly every row a support vector — binary and
one-vs-one, and rbf kernels far narrower than the data, from 10 to 1 000 times `"scale"`,
either side of the width floor, C = 1 to 100:

<!-- python -m benchmarks.calibrate table svc-hard: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, labels, kernel | C = 1 | C = 10 | C = 100 | dispatch |
|---|---|---|---|---|
| 3 000 × 128, rbf, gamma = 10 × scale | 1.72 / 1.58 | 1.52 / 1.38 | — | scikit-learn |
| 3 000 × 128, rbf, gamma = 30 × scale | 1.93 / 1.76 | 1.45 / 1.33 | — | scikit-learn |
| 3 000 × 128, rbf, gamma = 100 × scale | 1.20 / 1.13 | 0.96 / **0.90** | — | scikit-learn |
| 3 000 × 128, rbf, gamma = 1000 × scale | 1.31 / 1.22 | 1.04 / 0.96 | — | scikit-learn |
| 3 000 × 128, XOR, rbf, gamma = 1 × scale | 2.66 / 2.38 | **0.94** / **0.85** | **0.94** / **0.85** | scikit-learn |
| 3 000 × 128, noisy, rbf, gamma = 1 × scale | 2.40 / 2.16 | **0.91** / **0.83** | **0.91** / **0.83** | scikit-learn |
| 3 000 × 128, noisy, poly, degree 3, coef0 = 0 | 1.84 / 1.66 | 1.22 / 1.11 | 1.22 / 1.11 | scikit-learn |
| 3 000 × 128, random, rbf, gamma = 1 × scale | 2.71 / 2.43 | 0.96 / **0.87** | 0.96 / **0.87** | scikit-learn |
| 4 500 × 128, K = 3, random, rbf, gamma = 1 × scale | 2.69 / 2.41 | **0.89** / **0.81** | **0.89** / **0.81** | scikit-learn |
| 6 000 × 128, XOR, rbf, gamma = 1 × scale | 3.53 / 3.24 | 1.05 / 0.96 | 1.04 / 0.96 | scikit-learn |
| 6 000 × 128, noisy, rbf, gamma = 1 × scale | 3.20 / 2.93 | 1.02 / **0.94** | 1.02 / **0.94** | scikit-learn |
| 6 000 × 128, random, rbf, gamma = 1 × scale | 3.65 / 3.37 | 0.99 / **0.91** | 0.99 / **0.91** | scikit-learn |
| 6 000 × 128, K = 4, random, rbf, gamma = 1 × scale | 2.62 / 2.36 | **0.91** / **0.82** | **0.91** / **0.82** | scikit-learn |
| 8 000 × 128, XOR, rbf, gamma = 1 × scale | 3.08 / 2.89 | 1.76 / 1.65 | 1.76 / 1.65 | scikit-learn |
| 8 000 × 128, noisy, rbf, gamma = 1 × scale | 2.94 / 2.72 | 1.72 / 1.61 | 1.72 / 1.61 | scikit-learn |
| 8 000 × 128, random, rbf, gamma = 1 × scale | 3.16 / 2.95 | 1.83 / 1.72 | 1.83 / 1.71 | scikit-learn |
| 1 500 × 256, rbf, gamma = 10 × scale | 1.56 / 1.44 | 1.40 / 1.27 | — | MLX at C = 1 |
| 1 500 × 256, rbf, gamma = 30 × scale | 1.56 / 1.44 | 1.30 / 1.17 | — | MLX at C = 1 |
| 1 500 × 256, rbf, gamma = 100 × scale | 1.16 / 1.07 | 1.00~ / **0.91** | — | MLX at C = 1 |
| 1 500 × 256, rbf, gamma = 1000 × scale | 1.13 / 1.05 | 0.97 / **0.89** | — | MLX at C = 1 |
| 1 500 × 256, XOR, rbf, gamma = 1 × scale | 3.14 / 2.87 | 1.34 / 1.23 | 1.34 / 1.22 | MLX at C = 1 |
| 1 500 × 256, noisy, rbf, gamma = 1 × scale | 3.04 / 2.76 | 1.34 / 1.22 | 1.33 / 1.21 | MLX at C = 1 |
| 1 500 × 256, noisy, poly, degree 3, coef0 = 0 | 1.94 / 1.76 | 1.54 / 1.40 | 1.54 / 1.40 | scikit-learn |
| 1 500 × 256, random, rbf, gamma = 1 × scale | 3.18 / 2.89 | 1.35 / 1.25 | 1.35 / 1.23 | MLX at C = 1 |
| 2 250 × 256, K = 3, random, rbf, gamma = 1 × scale | 3.25 / 2.94 | 1.39 / 1.25 | 1.39 / 1.25 | MLX at C = 1 |
| 2 500 × 256, XOR, rbf, gamma = 1 × scale | 3.84 / 3.55 | 1.68 / 1.54 | 1.67 / 1.53 | MLX at C = 1 |
| 2 500 × 256, noisy, rbf, gamma = 1 × scale | 3.62 / 3.32 | 1.66 / 1.52 | 1.67 / 1.52 | MLX at C = 1 |
| 2 500 × 256, random, rbf, gamma = 1 × scale | 3.76 / 3.44 | 1.74 / 1.58 | 1.75 / 1.56 | MLX at C = 1 |
| 3 000 × 256, K = 4, random, rbf, gamma = 1 × scale | 3.29 / 2.94 | 1.37 / 1.23 | 1.37 / 1.23 | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 10 × scale | 2.78 / 2.60 | 2.32 / 2.15 | — | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 30 × scale | 3.16 / 2.90 | 2.25 / 2.07 | — | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 100 × scale | 1.98 / 1.83 | 1.51 / 1.41 | — | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 1000 × scale | 1.95 / 1.83 | 1.48 / 1.39 | — | MLX at C = 1 |
| 4 000 × 256, XOR, rbf, gamma = 1 × scale | 5.38 / 4.91 | 1.95 / 1.79 | 1.95 / 1.79 | MLX at C = 1 |
| 4 000 × 256, noisy, rbf, gamma = 1 × scale | 5.22 / 4.75 | 1.92 / 1.76 | 1.92 / 1.76 | MLX at C = 1 |
| 4 000 × 256, noisy, poly, degree 3, coef0 = 0 | 3.15 / 2.89 | 2.43 / 2.20 | 2.42 / 2.20 | scikit-learn |
| 4 000 × 256, random, rbf, gamma = 1 × scale | 5.33 / 4.86 | 2.01 / 1.86 | 1.99 / 1.86 | MLX at C = 1 |
| 7 500 × 256, K = 10, random, rbf, gamma = 1 × scale | 3.27 / 2.97 | 1.36 / 1.25 | 1.36 / 1.24 | MLX at C = 1 |
| 1 024 × 512, rbf, gamma = 10 × scale | 2.18 / 1.99 | 1.74 / 1.58 | — | MLX at C = 1 |
| 1 024 × 512, rbf, gamma = 30 × scale | 2.18 / 1.96 | 1.56 / 1.42 | — | MLX at C = 1 |
| 1 024 × 512, rbf, gamma = 100 × scale | 1.53 / 1.40 | 1.21 / 1.09 | — | MLX at C = 1 |
| 1 024 × 512, rbf, gamma = 1000 × scale | 1.50 / 1.37 | 1.18 / 1.07 | — | MLX at C = 1 |
| 1 024 × 512, XOR, rbf, gamma = 1 × scale | 3.92 / 3.54 | 1.82 / 1.65 | 1.82 / 1.67 | MLX at C = 1 |
| 1 024 × 512, noisy, rbf, gamma = 1 × scale | 3.79 / 3.42 | 1.87 / 1.68 | 1.86 / 1.68 | MLX at C = 1 |
| 1 024 × 512, noisy, poly, degree 3, coef0 = 0 | 2.60 / 2.36 | 2.11 / 1.87 | 2.11 / 1.90 | scikit-learn |
| 1 024 × 512, random, rbf, gamma = 1 × scale | 4.06 / 3.69 | 1.90 / 1.72 | 1.91 / 1.70 | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 10 × scale | 1.88 / 1.72 | 1.34 / 1.21 | — | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 30 × scale | 1.75 / 1.59 | 1.22 / 1.10 | — | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 100 × scale | 1.33 / 1.22 | 1.00~ / **0.91** | — | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 1000 × scale | 1.31 / 1.19 | 0.98 / **0.88** | — | MLX at C = 1 |
| 1 536 × 512, K = 3, random, rbf, gamma = 1 × scale | 4.20 / 3.81 | 1.94 / 1.77 | 1.93 / 1.76 | MLX at C = 1 |
| 2 000 × 512, rbf, gamma = 10 × scale | 3.08 / 2.79 | 2.51 / 2.27 | — | MLX at C = 1 |
| 2 000 × 512, rbf, gamma = 30 × scale | 3.15 / 2.86 | 2.27 / 2.05 | — | MLX at C = 1 |
| 2 000 × 512, rbf, gamma = 100 × scale | 2.07 / 1.92 | 1.61 / 1.49 | — | MLX at C = 1 |
| 2 000 × 512, rbf, gamma = 1000 × scale | 2.04 / 1.89 | 1.58 / 1.46 | — | MLX at C = 1 |
| 2 000 × 512, XOR, rbf, gamma = 1 × scale | 5.90 / 5.36 | 2.58 / 2.33 | 2.58 / 2.34 | MLX at C = 1 |
| 2 000 × 512, noisy, rbf, gamma = 1 × scale | 5.50 / 4.99 | 2.57 / 2.34 | 2.57 / 2.33 | MLX at C = 1 |
| 2 000 × 512, random, rbf, gamma = 1 × scale | 5.88 / 5.34 | 2.67 / 2.40 | 2.65 / 2.40 | MLX at C = 1 |
| 2 048 × 512, K = 4, random, rbf, gamma = 1 × scale | 4.16 / 3.78 | 1.97 / 1.79 | 1.97 / 1.78 | MLX at C = 1 |

- **At 128 features the benchmark's target was the easy case.** On labels without
  signal 3 000–6 000 rows lost at C = 10 and 100 on 1.7.2 (0.83×–0.94×) and 3 000 on
  1.9.0 too (0.91×–0.94×), and one-vs-one pairs of 3 000 × 128 on random labels lost on
  both (0.81×–0.91×) — where the `svc` and `svc-multi-overlap` tables had them winning.
  From 256 features every one of those laws won at every C, binary or one-vs-one, from
  1 500 × 256 and 1 024 × 512 (1.21×–5.9×).
- **A kernel far narrower than the data** makes every row a support vector too: at C = 10
  `gamma` of 100 and 1 000 times `"scale"` — which `gamma=1` is, on standardized data of
  100–1 000 features — lost on 1.7.2 at 1 500 × 256 and 1 100 × 512 (0.88×–0.91×) and
  3 000 × 128 (0.90×), and tied on 1.9.0; at 30 times it won at every shape (1.10×–2.3×),
  and at C = 1 every `gamma` won (1.05×–3.2×).

The size rules take the width, rows and work at which both kernels win at every C and on
every label law measured: 256 features, 1 024 rows and 384 000 elements — 1 500 × 256,
which keeps 1 000 × 256 on LIBSVM (a 1.04× win at worst on the benchmark's target) — and
`svc_min_samples` lets 1 024 × 512 through. Above all three floors every fit measured at
the default kernel parameters won on both versions: 1.36×–8.4× on the benchmark's target
(1.36×–4.4× at C of 1 and 10, up to 8 000 rows) and 1.21×–5.9× on the labels above — on
this Gaussian data. Above C = 1, data of lower intrinsic dimension loses at these widths,
and patched dispatch no longer takes those wins (`svc-structured`, below). What
the 256-feature floor gives up is everything at 128–255 features that won: the
benchmark's target from 3 000 × 128 (1.11×–4.1×), labels without signal at 8 000 × 128
(1.61×–3.2×) and at C = 1, one-vs-one pairs of 3 000 × 128 on overlapping classes
(1.09×–2.0×), and the kernels of `svc-params` there.

**The kernel parameters move the crossing.** A kernel close to the linear one takes many
iterations per row, over rows LIBSVM has already cached — the linear kernel's own case —
and at those the iteration count, not the width, decides. Grid `svc-params`: the same
binary fits at shapes the floors send to MLX, rbf from 0.03 to 10 times scikit-learn's
`"scale"` value of `gamma` (`1 / (d · Var X)`; the default is 1), poly at degree 2 to 4 and
`coef0` 0 and 1, C = 1 to 100:

<!-- python -m benchmarks.calibrate table svc-params: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, kernel | C = 1 | C = 10 | C = 100 | dispatch |
|---|---|---|---|---|
| 3 000 × 128, rbf, gamma = 0.03 × scale | 3.92 / 3.53 | 1.95 / 1.75 | **0.61** / **0.54** | scikit-learn |
| 3 000 × 128, rbf, gamma = 0.1 × scale | 3.27 / 2.93 | 1.14 / 1.02 | **0.33** / **0.30** | scikit-learn |
| 3 000 × 128, rbf, gamma = 0.3 × scale | 2.63 / 2.37 | **0.67** / **0.61** | **0.61** / **0.56** | scikit-learn |
| 3 000 × 128, rbf, gamma = 0.5 × scale | 2.37 / 2.13 | **0.85** / **0.77** | **0.84** / **0.76** | scikit-learn |
| 3 000 × 128, rbf, gamma = 0.8 × scale | 2.13 / 1.93 | 1.10 / 1.01 | 1.10 / 1.01 | scikit-learn |
| 3 000 × 128, rbf, gamma = 3 × scale | 1.70 / 1.53 | 1.65 / 1.48 | 1.65 / 1.47 | scikit-learn |
| 3 000 × 128, rbf, gamma = 10 × scale | 1.73 / 1.56 | 1.52 / 1.37 | 1.52 / 1.37 | scikit-learn |
| 3 000 × 128, poly, degree 2, coef0 = 0 | 1.84 / 1.66 | **0.63** / **0.57** | **0.63** / **0.57** | scikit-learn |
| 3 000 × 128, poly, degree 2, coef0 = 1 | **0.76** / **0.70** | **0.58** / **0.53** | **0.58** / **0.53** | scikit-learn |
| 3 000 × 128, poly, degree 3, coef0 = 1 | **0.89** / **0.81** | **0.89** / **0.81** | **0.89** / **0.81** | scikit-learn |
| 3 000 × 128, poly, degree 4, coef0 = 0 | 1.83 / 1.66 | 1.16 / 1.04 | 1.16 / 1.04 | scikit-learn |
| 8 000 × 128, rbf, gamma = 0.03 × scale | 3.31 / 3.10 | 2.47 / 2.32 | 1.07 / 0.99 | scikit-learn |
| 8 000 × 128, rbf, gamma = 0.1 × scale | 2.96 / 2.80 | 1.78 / 1.67 | **0.50** / **0.46** | scikit-learn |
| 8 000 × 128, rbf, gamma = 0.3 × scale | 2.78 / 2.64 | 1.04 / 0.97 | **0.86** / **0.81** | scikit-learn |
| 8 000 × 128, rbf, gamma = 0.5 × scale | 2.86 / 2.71 | 1.06 / 0.99 | 0.98 / **0.92** | scikit-learn |
| 8 000 × 128, rbf, gamma = 0.8 × scale | 2.67 / 2.53 | 1.33 / 1.26 | 1.33 / 1.26 | scikit-learn |
| 8 000 × 128, rbf, gamma = 3 × scale | 2.42 / 2.29 | 2.79 / 2.63 | 2.79 / 2.64 | scikit-learn |
| 8 000 × 128, rbf, gamma = 10 × scale | 2.35 / 2.22 | 2.12 / 2.00 | 2.12 / 2.00 | scikit-learn |
| 8 000 × 128, poly, degree 2, coef0 = 0 | 2.07 / 1.94 | **0.54** / **0.50** | **0.46** / **0.43** | scikit-learn |
| 8 000 × 128, poly, degree 2, coef0 = 1 | 1.13 / 1.04 | **0.51** / **0.48** | **0.51** / **0.47** | scikit-learn |
| 8 000 × 128, poly, degree 3, coef0 = 1 | 1.01 / 0.96 | 1.00~ / **0.94** | 1.00~ / **0.94** | scikit-learn |
| 8 000 × 128, poly, degree 4, coef0 = 0 | 1.65 / 1.56 | 1.07 / 1.03 | 1.05 / 1.01 | scikit-learn |
| 1 500 × 256, rbf, gamma = 0.03 × scale | 4.91 / 4.50 | 2.07 / 1.88 | **0.41** / **0.37** | MLX at C = 1 |
| 1 500 × 256, rbf, gamma = 0.1 × scale | 4.26 / 3.85 | **0.95** / **0.84** | **0.42** / **0.38** | MLX at C = 1 |
| 1 500 × 256, rbf, gamma = 0.3 × scale | 2.97 / 2.70 | **0.88** / **0.81** | **0.81** / **0.74** | MLX at C = 1 |
| 1 500 × 256, rbf, gamma = 0.5 × scale | 2.65 / 2.40 | 1.12 / 0.99~ | 1.11 / 1.00~ | MLX at C = 1 |
| 1 500 × 256, rbf, gamma = 0.8 × scale | 2.45 / 2.21 | 1.43 / 1.27 | 1.42 / 1.28 | MLX at C = 1 |
| 1 500 × 256, rbf, gamma = 3 × scale | 2.14 / 1.92 | 1.95 / 1.75 | 1.94 / 1.73 | MLX at C = 1 |
| 1 500 × 256, rbf, gamma = 10 × scale | 1.55 / 1.40 | 1.41 / 1.24 | 1.38 / 1.24 | MLX at C = 1 |
| 1 500 × 256, poly, degree 2, coef0 = 0 | 2.18 / 1.95 | 1.49 / 1.32 | 1.50 / 1.33 | scikit-learn |
| 1 500 × 256, poly, degree 2, coef0 = 1 | **0.94** / **0.85** | **0.83** / **0.74** | **0.83** / **0.74** | scikit-learn |
| 1 500 × 256, poly, degree 3, coef0 = 1 | 1.18 / 1.06 | 1.18 / 1.06 | 1.18 / 1.06 | scikit-learn |
| 1 500 × 256, poly, degree 4, coef0 = 0 | 2.00 / 1.77 | 1.45 / 1.28 | 1.44 / 1.29 | scikit-learn |
| 4 000 × 256, rbf, gamma = 0.03 × scale | 7.26 / 6.54 | 3.79 / 3.44 | 1.07 / 0.97 | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 0.1 × scale | 5.92 / 5.37 | 2.06 / 1.88 | **0.68** / **0.62** | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 0.3 × scale | 4.83 / 4.39 | 1.25 / 1.15 | 1.23 / 1.13 | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 0.5 × scale | 4.25 / 3.90 | 1.70 / 1.57 | 1.70 / 1.57 | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 0.8 × scale | 3.89 / 3.57 | 2.24 / 2.06 | 2.23 / 2.05 | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 3 × scale | 3.30 / 3.02 | 3.00 / 2.75 | 3.01 / 2.76 | MLX at C = 1 |
| 4 000 × 256, rbf, gamma = 10 × scale | 2.81 / 2.58 | 2.34 / 2.13 | 2.33 / 2.13 | MLX at C = 1 |
| 4 000 × 256, poly, degree 2, coef0 = 0 | 3.73 / 3.39 | 1.77 / 1.62 | 1.77 / 1.62 | scikit-learn |
| 4 000 × 256, poly, degree 2, coef0 = 1 | 1.42 / 1.30 | 1.24 / 1.14 | 1.24 / 1.14 | scikit-learn |
| 4 000 × 256, poly, degree 3, coef0 = 1 | 1.82 / 1.67 | 1.82 / 1.67 | 1.82 / 1.66 | scikit-learn |
| 4 000 × 256, poly, degree 4, coef0 = 0 | 3.26 / 2.98 | 2.34 / 2.13 | 2.34 / 2.11 | scikit-learn |
| 1 100 × 512, rbf, gamma = 0.03 × scale | 5.58 / 5.04 | 2.91 / 2.70 | **0.45** / **0.41** | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 0.1 × scale | 5.39 / 5.00 | 1.08 / 1.00~ | **0.58** / **0.52** | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 0.3 × scale | 4.51 / 4.11 | 1.08 / 0.98 | 1.07 / 0.97 | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 0.5 × scale | 3.75 / 3.40 | 1.41 / 1.27 | 1.40 / 1.26 | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 0.8 × scale | 3.29 / 3.02 | 1.81 / 1.63 | 1.79 / 1.64 | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 3 × scale | 1.98 / 1.83 | 1.58 / 1.44 | 1.59 / 1.44 | MLX at C = 1 |
| 1 100 × 512, rbf, gamma = 10 × scale | 1.85 / 1.72 | 1.31 / 1.20 | 1.31 / 1.21 | MLX at C = 1 |
| 1 100 × 512, poly, degree 2, coef0 = 0 | 2.28 / 2.07 | 1.94 / 1.74 | 1.93 / 1.74 | scikit-learn |
| 1 100 × 512, poly, degree 2, coef0 = 1 | 1.23 / 1.12 | 1.17 / 1.07 | 1.16 / 1.07 | scikit-learn |
| 1 100 × 512, poly, degree 3, coef0 = 1 | 1.62 / 1.47 | 1.61 / 1.47 | 1.62 / 1.47 | scikit-learn |
| 1 100 × 512, poly, degree 4, coef0 = 0 | 2.40 / 2.17 | 1.98 / 1.78 | 1.98 / 1.78 | scikit-learn |
| 4 000 × 512, rbf, gamma = 0.03 × scale | 11.15 / 10.36 | 5.90 / 5.43 | 1.32 / 1.22 | MLX at C = 1 |
| 4 000 × 512, rbf, gamma = 0.1 × scale | 9.75 / 9.10 | 2.85 / 2.64 | 1.25 / 1.15 | MLX at C = 1 |
| 4 000 × 512, rbf, gamma = 0.3 × scale | 7.28 / 6.85 | 2.28 / 2.11 | 2.04 / 1.90 | MLX at C = 1 |
| 4 000 × 512, rbf, gamma = 0.5 × scale | 6.57 / 6.19 | 2.74 / 2.56 | 2.78 / 2.58 | MLX at C = 1 |
| 4 000 × 512, rbf, gamma = 0.8 × scale | 6.12 / 5.72 | 3.55 / 3.31 | 3.57 / 3.31 | MLX at C = 1 |
| 4 000 × 512, rbf, gamma = 3 × scale | 4.45 / 4.17 | 4.61 / 4.27 | 4.57 / 4.27 | MLX at C = 1 |
| 4 000 × 512, rbf, gamma = 10 × scale | 4.19 / 3.90 | 3.16 / 2.92 | 3.14 / 2.93 | MLX at C = 1 |
| 4 000 × 512, poly, degree 2, coef0 = 0 | 6.04 / 5.62 | 4.21 / 3.90 | 4.24 / 3.88 | scikit-learn |
| 4 000 × 512, poly, degree 2, coef0 = 1 | 2.62 / 2.44 | 2.20 / 2.07 | 2.21 / 2.06 | scikit-learn |
| 4 000 × 512, poly, degree 3, coef0 = 1 | 3.17 / 2.96 | 3.17 / 2.94 | 3.18 / 2.95 | scikit-learn |
| 4 000 × 512, poly, degree 4, coef0 = 0 | 5.35 / 4.97 | 4.44 / 4.08 | 4.42 / 4.09 | scikit-learn |

- **rbf, `gamma` below the default: loses at large C.** From 0.03 to 0.5 times `"scale"`
  it lost at C = 10 or 100 at five of the six shapes, down to 0.30× (0.1 × scale,
  3 000 × 128, C = 100: 28 409 SMO iterations against LIBSVM's 28 343, three times as
  long). At 0.8 it won or tied everywhere, and from 1 to 10 it won. At C = 1 every
  `gamma` below the default won at every shape, 2.1×–11×: too few iterations to lose.
- **poly, degree 2 or `coef0` ≠ 0: loses.** Degree 2 lost at C = 10 and 100 at
  128 features (0.43×–0.63×), and `coef0 = 1` at degree 2 or 3 lost at 128 features
  (down to 0.47×) and, at degree 2, at 1 500 × 256 (0.74×–0.94×). Degree 3 and 4 at
  `coef0 = 0` won or tied everywhere on this Gaussian data, but not on data unlike it
  (`svc-poly`, below).

So these kernels lose above C = 1, where their many iterations multiply; at C = 1 every
`gamma` below the default won at every shape (2.1×–11×) and, in the `svc-hard` table,
every `gamma` above it (1.05×–3.2×). The poly kernel of degree 2 with `coef0 = 1` lost at
C = 1 too, at 1 500 × 256 (0.85×–0.94×), and degree 3 lost at C ≤ 1 on data these grids
do not hold (below), so patched dispatch hands every poly fit to LIBSVM, whatever its size
and parameters. For rbf the C rule below covers these corners; the previous rule's bounds
on `gamma`, 0.8 to 30 × `"scale"` above C = 1, are gone.

**The data move it too.** Every grid above draws `X` from an isotropic Gaussian, whose
intrinsic dimension is its width, and from 256 features that made C = 10 and 100 as safe
as C = 1. Data whose rows lie near a few directions is another matter: iterations multiply
with C at any width, and LIBSVM, which pays `O(n·d)` for a kernel row once and a few µs
per visit after, wins them back. Grid `svc-structured`: binary fits at 1 024–4 000 rows
and 256–512 features on five or ten latent directions plus noise (`lowrank`,
`lowrank10`), singular values 1/i (`decay`), features averaged over 16 neighbours
(`smooth`), Student-t(2) features (`t2`) and column scales 10^U(0, 2) (`colscale`), and
scikit-learn's handwritten digits upsampled to 16², 20² and 28² pixels, at C = 0.1, 1 and
10; and at C = 1, `gamma` from 0.03 to 100 × `"scale"`, the poly kernel, `tol = 1e-5`,
balanced class weights on 10% positives (`lowrank-imb`), and 8 000 rows:

<!-- python -m benchmarks.calibrate table svc-structured: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, data, kernel | C = 0.1 | C = 0.2 | C = 1 | C = 10 | dispatch |
|---|---|---|---|---|---|
| 1 500 × 256, colscale, rbf, gamma = 1 × scale | 5.14 / 4.60 | — | 2.50 / 2.23 | 1.18 / 1.08 | MLX at C = 0.1, C = 1 |
| 1 500 × 256, decay, rbf, gamma = 0.03 × scale | — | — | 4.46 / 4.09 | — | MLX |
| 1 500 × 256, decay, rbf, gamma = 0.3 × scale | — | — | 3.26 / 2.96 | — | MLX |
| 1 500 × 256, decay, rbf, gamma = 1 × scale | 4.17 / 3.79 | — | 2.15 / 1.93 | **0.73** / **0.66** | MLX at C = 0.1, C = 1 |
| 1 500 × 256, decay, rbf, gamma = 1 × scale, tol = 1e-05 | — | — | 1.53 / 1.37 | — | MLX |
| 1 500 × 256, decay, rbf, gamma = 10 × scale | — | — | 2.07 / 1.89 | — | MLX |
| 1 500 × 256, decay, rbf, gamma = 100 × scale | — | — | 1.26 / 1.15 | — | MLX |
| 1 500 × 256, decay, poly, degree 3, coef0 = 0 | — | — | 2.17 / 1.91 | **0.34** / **0.30** | scikit-learn |
| 1 500 × 256, lowrank, rbf, gamma = 0.03 × scale | — | — | 4.31 / 3.87 | — | MLX |
| 1 500 × 256, lowrank, rbf, gamma = 0.3 × scale | — | — | 3.32 / 2.96 | — | MLX |
| 1 500 × 256, lowrank, rbf, gamma = 1 × scale | 4.50 / 4.09 | — | 2.40 / 2.14 | **0.76** / **0.69** | MLX at C = 0.1, C = 1 |
| 1 500 × 256, lowrank, rbf, gamma = 1 × scale, tol = 1e-05 | — | — | 1.87 / 1.72 | — | MLX |
| 1 500 × 256, lowrank, rbf, gamma = 10 × scale | — | — | 1.97 / 1.79 | — | MLX |
| 1 500 × 256, lowrank, rbf, gamma = 100 × scale | — | — | 1.18 / 1.06 | — | MLX |
| 1 500 × 256, lowrank, poly, degree 3, coef0 = 0 | — | — | 1.41 / 1.24 | **0.34** / **0.29** | scikit-learn |
| 1 500 × 256, lowrank-imb, rbf, gamma = 1 × scale | — | — | 1.45 / 1.30 | — | MLX |
| 1 500 × 256, lowrank-imb, rbf, gamma = 1 × scale, class_weight = balanced | 2.61 / 2.35 | 2.42 / 2.20 | 1.90 / 1.71 | — | MLX |
| 1 500 × 256, lowrank10, rbf, gamma = 1 × scale | 4.60 / 4.13 | — | 2.36 / 2.15 | **0.60** / **0.54** | MLX at C = 0.1, C = 1 |
| 1 500 × 256, smooth, rbf, gamma = 1 × scale | 4.86 / 4.42 | — | 2.46 / 2.24 | **0.92** / **0.82** | MLX at C = 0.1, C = 1 |
| 1 500 × 256, t2, rbf, gamma = 1 × scale | 4.92 / 4.35 | — | 3.05 / 2.73 | **0.87** / **0.78** | MLX at C = 0.1, C = 1 |
| 1 797 × 256, digits16, rbf, gamma = 1 × scale | 4.78 / 4.34 | — | 1.95 / 1.74 | **0.65** / **0.57** | MLX at C = 0.1, C = 1 |
| 4 000 × 256, colscale, rbf, gamma = 1 × scale | 7.11 / 6.43 | — | 5.71 / 5.19 | 1.31 / 1.20 | MLX at C = 0.1, C = 1 |
| 4 000 × 256, decay, rbf, gamma = 1 × scale | 6.15 / 5.54 | — | 3.61 / 3.28 | 1.30 / 1.18 | MLX at C = 0.1, C = 1 |
| 4 000 × 256, lowrank, rbf, gamma = 1 × scale | 6.72 / 6.05 | — | 4.71 / 4.23 | 1.60 / 1.46 | MLX at C = 0.1, C = 1 |
| 4 000 × 256, lowrank10, rbf, gamma = 1 × scale | 6.85 / 6.18 | — | 4.54 / 4.09 | 1.21 / 1.10 | MLX at C = 0.1, C = 1 |
| 4 000 × 256, smooth, rbf, gamma = 1 × scale | 6.90 / 6.24 | — | 4.08 / 3.74 | 1.24 / 1.14 | MLX at C = 0.1, C = 1 |
| 4 000 × 256, t2, rbf, gamma = 1 × scale | 7.33 / 6.57 | — | 5.35 / 4.79 | 1.38 / 1.25 | MLX at C = 0.1, C = 1 |
| 8 000 × 256, decay, rbf, gamma = 1 × scale | — | — | 3.63 / 3.43 | — | MLX |
| 8 000 × 256, lowrank, rbf, gamma = 1 × scale | — | — | 4.40 / 4.20 | — | MLX |
| 1 797 × 400, digits20, rbf, gamma = 1 × scale | 6.14 / 5.58 | — | 2.24 / 2.07 | **0.81** / **0.73** | MLX at C = 0.1, C = 1 |
| 1 024 × 512, colscale, rbf, gamma = 1 × scale | 5.91 / 5.34 | — | 3.92 / 3.45 | 1.65 / 1.48 | MLX at C = 0.1, C = 1 |
| 1 024 × 512, decay, rbf, gamma = 0.03 × scale | — | — | 5.05 / 4.51 | — | MLX |
| 1 024 × 512, decay, rbf, gamma = 0.3 × scale | — | — | 4.06 / 3.62 | — | MLX |
| 1 024 × 512, decay, rbf, gamma = 1 × scale | 4.88 / 4.45 | — | 2.58 / 2.29 | **0.92** / **0.82** | MLX at C = 0.1, C = 1 |
| 1 024 × 512, decay, rbf, gamma = 1 × scale, tol = 1e-05 | — | — | 1.84 / 1.65 | — | MLX |
| 1 024 × 512, decay, rbf, gamma = 10 × scale | — | — | 2.44 / 2.19 | — | MLX |
| 1 024 × 512, decay, rbf, gamma = 100 × scale | — | — | 1.42 / 1.30 | — | MLX |
| 1 024 × 512, decay, poly, degree 3, coef0 = 0 | — | — | 3.23 / 2.84 | **0.48** / **0.43** | scikit-learn |
| 1 024 × 512, lowrank, rbf, gamma = 0.03 × scale | — | — | 5.11 / 4.62 | — | MLX |
| 1 024 × 512, lowrank, rbf, gamma = 0.3 × scale | — | — | 3.99 / 3.60 | — | MLX |
| 1 024 × 512, lowrank, rbf, gamma = 1 × scale | 5.16 / 4.71 | — | 2.84 / 2.54 | **0.89** / **0.81** | MLX at C = 0.1, C = 1 |
| 1 024 × 512, lowrank, rbf, gamma = 1 × scale, tol = 1e-05 | — | — | 2.00 / 1.82 | — | MLX |
| 1 024 × 512, lowrank, rbf, gamma = 10 × scale | — | — | 2.48 / 2.23 | — | MLX |
| 1 024 × 512, lowrank, rbf, gamma = 100 × scale | — | — | 1.63 / 1.46 | — | MLX |
| 1 024 × 512, lowrank, poly, degree 3, coef0 = 0 | — | — | 1.34 / 1.20 | **0.29** / **0.26** | scikit-learn |
| 1 024 × 512, lowrank-imb, rbf, gamma = 1 × scale | — | — | 1.72 / 1.58 | — | MLX |
| 1 024 × 512, lowrank-imb, rbf, gamma = 1 × scale, class_weight = balanced | 3.12 / 2.79 | 2.95 / 2.65 | 2.31 / 2.04 | — | MLX |
| 1 024 × 512, lowrank10, rbf, gamma = 1 × scale | 5.48 / 4.87 | — | 2.73 / 2.48 | **0.74** / **0.66** | MLX at C = 0.1, C = 1 |
| 1 024 × 512, smooth, rbf, gamma = 1 × scale | 5.71 / 5.12 | — | 2.93 / 2.65 | 1.54 / 1.39 | MLX at C = 0.1, C = 1 |
| 1 024 × 512, t2, rbf, gamma = 1 × scale | 4.97 / 4.43 | — | 4.28 / 3.75 | 1.07 / 0.95 | MLX at C = 0.1, C = 1 |
| 2 000 × 512, colscale, rbf, gamma = 1 × scale | 8.41 / 7.42 | — | 5.52 / 4.87 | 2.31 / 2.08 | MLX at C = 0.1, C = 1 |
| 2 000 × 512, decay, rbf, gamma = 1 × scale | 7.39 / 6.68 | — | 4.12 / 3.74 | 1.49 / 1.34 | MLX at C = 0.1, C = 1 |
| 2 000 × 512, lowrank, rbf, gamma = 1 × scale | 8.24 / 7.43 | — | 5.05 / 4.49 | 1.57 / 1.39 | MLX at C = 0.1, C = 1 |
| 2 000 × 512, lowrank10, rbf, gamma = 1 × scale | 8.20 / 7.30 | — | 4.53 / 4.02 | 1.16 / 1.04 | MLX at C = 0.1, C = 1 |
| 2 000 × 512, smooth, rbf, gamma = 1 × scale | 9.13 / 8.15 | — | 4.82 / 4.39 | 1.98 / 1.79 | MLX at C = 0.1, C = 1 |
| 2 000 × 512, t2, rbf, gamma = 1 × scale | 7.54 / 6.65 | — | 6.03 / 5.33 | 1.64 / 1.48 | MLX at C = 0.1, C = 1 |
| 1 797 × 784, digits28, rbf, gamma = 1 × scale | 9.20 / 8.46 | — | 3.49 / 3.11 | 1.26 / 1.13 | MLX at C = 0.1, C = 1 |
| 1 797 × 784, K = 3, digits28, rbf, gamma = 1 × scale | — | — | 2.74 / 2.48 | 1.05 / **0.94** | MLX at C = 1 |

At C = 10 every law but the column scales lost at 1 500 × 256, and most at 1 024 × 512,
on both versions — rbf down to 0.54×, poly to 0.26×, the digits at 256 and 400 pixels
0.57×–0.81× and at 784 pixels with three classes 0.94× on 1.7.2 — and won from 2 000 rows
(1.04×–2.3×); at C = 100, in runs outside the grid, they went down to 0.28×. At C ≤ 1
every fit here won on both versions, 1.06×–9.2×: at every `gamma` (1.06× at 100 ×
`"scale"`), with `tol = 1e-5`, with balanced class weights and at 8 000 rows. A fit cannot
see its data's intrinsic dimension before it runs, so patched dispatch hands every rbf
fit with C above 1 to LIBSVM (`svc_max_c = 1`), recorded as `"below-crossover"`, and every
poly fit whatever its C (below). It reads the mean box, C times the mean class and sample
weight over the samples kept: C itself without weights, with balanced class weights and
with sample weights normalized to mean 1, which average to one; `C · w` for weights that
are all `w`, which is exactly the problem with that C. Averages to one hold only up to
rounding — balanced weights at 2% positives average to 1.0000000000000002 — so the box is
compared with a relative tolerance of 1e-9.

What the rule gives up, with the poly rule below, is every rbf and poly fit above C = 1
that won: the benchmark's target at C = 10 and 100 from 256 features (`svc`, 1.36×–3.8×),
labels with little or no signal (`svc-hard`, 1.10×–2.7×), overlapping one-vs-one classes
(`svc-multi-overlap`, 1.39×–2.0×), `gamma` of 0.8–10 × `"scale"` (`svc-params`,
1.20×–4.6×), and the structured laws at 2 000–4 000 rows here (1.04×–2.3×). A directly
imported class keeps them on MLX.

**The poly kernel stays on LIBSVM.** Degree 3 and 4 at `coef0 = 0` won every Gaussian
fit above the floors at C ≤ 1 (`svc`, `svc-hard`, `svc-params`) and the low-rank and
decaying fits of `svc-structured` at C = 1, but not data unlike them. Grid `svc-poly`: poly fits at
C = 0.2–1 on scikit-learn's digits upsampled to 16², 20² and 28² pixels (digit ≥ 5, its
parity, and digit mod 4 one-vs-one), on five latent directions plus noise with random
labels, on Cauchy features, and at `gamma` 1.25 and 1.5 × `"scale"` on five latent
directions; rbf beside them at C = 1 on the same data. At 256 features the Cauchy rows are
too far from the origin for the poly kernel's float32 inner product, which refuses the fit
(`UnsupportedInputError`, a capability fallback in patched mode), so they are timed at
512 features only:

<!-- python -m benchmarks.calibrate table svc-poly: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, data, kernel | C = 0.2 | C = 0.5 | C = 1 | dispatch |
|---|---|---|---|---|
| 1 500 × 256, cauchy, rbf, gamma = 1 × scale | — | — | 4.42 / 3.94 | MLX |
| 1 500 × 256, lowrank, poly, degree 3, coef0 = 0, gamma = 1.25 × scale | — | — | **0.75** / **0.68** | scikit-learn |
| 1 500 × 256, lowrank, poly, degree 3, coef0 = 0, gamma = 1.5 × scale | — | — | **0.52** / **0.46** | scikit-learn |
| 1 500 × 256, lowrank-rand, rbf, gamma = 1 × scale | — | — | 3.43 / 3.05 | MLX |
| 1 500 × 256, lowrank-rand, poly, degree 3, coef0 = 0 | — | 1.19 / 1.07 | **0.73** / **0.65** | scikit-learn |
| 1 797 × 256, digits16, rbf, gamma = 1 × scale | — | — | 1.93 / 1.78 | MLX |
| 1 797 × 256, digits16, poly, degree 3, coef0 = 0 | 1.22 / 1.10 | **0.73** / **0.65** | **0.52** / **0.46** | scikit-learn |
| 1 797 × 256, digits16-parity, rbf, gamma = 1 × scale | — | — | 1.86 / 1.70 | MLX |
| 1 797 × 256, digits16-parity, poly, degree 3, coef0 = 0 | 1.26 / 1.13 | **0.67** / **0.61** | **0.52** / **0.47** | scikit-learn |
| 3 000 × 256, cauchy, rbf, gamma = 1 × scale | — | — | 6.25 / 5.60 | MLX |
| 3 000 × 256, lowrank-rand, rbf, gamma = 1 × scale | — | — | 4.91 / 4.40 | MLX |
| 3 000 × 256, lowrank-rand, poly, degree 3, coef0 = 0 | — | 2.53 / 2.29 | 1.73 / 1.56 | scikit-learn |
| 1 797 × 400, digits20, rbf, gamma = 1 × scale | — | — | 2.24 / 2.07 | MLX |
| 1 797 × 400, digits20, poly, degree 3, coef0 = 0 | 1.57 / 1.40 | 0.98 / **0.87** | **0.70** / **0.62** | scikit-learn |
| 1 024 × 512, cauchy, rbf, gamma = 1 × scale | — | — | 4.63 / 4.07 | MLX |
| 1 024 × 512, cauchy, poly, degree 3, coef0 = 0 | — | 2.97 / 2.64 | 3.32 / 2.90 | scikit-learn |
| 1 024 × 512, lowrank-rand, rbf, gamma = 1 × scale | — | — | 4.29 / 3.83 | MLX |
| 1 024 × 512, lowrank-rand, poly, degree 3, coef0 = 0 | — | 2.19 / 1.97 | 1.12 / 0.99~ | scikit-learn |
| 2 000 × 512, lowrank, poly, degree 3, coef0 = 0, gamma = 1.25 × scale | — | — | 1.68 / 1.50 | scikit-learn |
| 2 000 × 512, lowrank, poly, degree 3, coef0 = 0, gamma = 1.5 × scale | — | — | 0.99 / **0.89** | scikit-learn |
| 1 797 × 784, K = 4, digits28, rbf, gamma = 1 × scale | — | — | 2.27 / 2.10 | MLX |
| 1 797 × 784, K = 4, digits28, poly, degree 3, coef0 = 0 | 1.56 / 1.38 | **0.84** / **0.75** | **0.66** / **0.59** | scikit-learn |

Poly lost at C = 1 on every digits law and on random labels at 1 500 × 256 (0.46×–0.73×),
at C = 0.5 on the digits (0.61×–0.87×), and at `gamma` above `"scale"` (0.46×–0.89×); it
won or tied at C = 0.2, on random labels at 3 000 × 256 and 1 024 × 512, and on the Cauchy rows it
could serve. rbf won on every law here, 1.70×–6.3×. How
many iterations the poly kernel takes turns on the data and on `gamma`, which a fit cannot
see, and a rule reading them would be one more corner. So patched dispatch sends only rbf
to MLX: poly joins linear and sigmoid on LIBSVM at every size, recorded as
`"below-crossover"`. That gives up every poly fit the previous rule — degree 3 or more,
`coef0 = 0`, C ≤ 1 — sent to MLX and won: 1.67×–7.0× on the benchmark's target (`svc`),
1.76×–3.2× on labels with little or no signal (`svc-hard`), 1.77×–5.4× at degree 4
(`svc-params`), 1.20×–3.2× on low-rank and decaying data (`svc-structured`) and 1.07×–3.3×
here. A directly imported class keeps poly on its size rules, and on the GPU above them.

**A small `cache_size` stays on LIBSVM.** Every table above ran at scikit-learn's default
cache, 200 MB, which holds the whole Gram of a pair up to about 7 240 rows; a smaller one
makes mlxlearn evaluate the rows it cannot hold on demand, a device call at a time. Grid
`svc-cache`: rbf at C = 1, both sides given the same `cache_size`, 1 MB to 1 000 MB:

<!-- python -m benchmarks.calibrate table svc-cache: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, data | cache_size = 1 | cache_size = 3 | cache_size = 10 | cache_size = 40 | cache_size = 100 | cache_size = 200 | cache_size = 1000 | dispatch |
|---|---|---|---|---|---|---|---|---|
| 4 000 × 256, Gaussian | 2.22 / 2.11 | — | 3.66 / 3.48 | — | — | 3.70 / 3.41 | 3.72 / 3.42 | MLX at cache_size = 200, cache_size = 1000 |
| 4 000 × 256, lowrank | 0.95 / **0.92** | 1.09 / 1.03 | 1.47 / 1.36 | 2.59 / 2.41 | 4.64 / 4.24 | 4.71 / 4.21 | 4.69 / 4.24 | MLX at cache_size = 200, cache_size = 1000 |
| 8 000 × 256, lowrank | 1.27 / 1.23 | — | — | — | — | 4.51 / 4.27 | 6.58 / 6.10 | MLX at cache_size = 200, cache_size = 1000 |
| 3 000 × 512, lowrank | 1.25 / 1.23 | — | 2.13 / 2.03 | — | — | 6.33 / 5.76 | — | MLX at cache_size = 200 |

At 1 MB five latent directions at 4 000 × 256 lost on 1.7.2 (0.92×; 0.95× on 1.9.0); from
3 MB up every fit won, and at the default and at 1 000 MB, 3.4×–6.6×. So patched dispatch
hands a fit whose `cache_size` is below scikit-learn's default to LIBSVM, recorded as
`"below-crossover"` — a rule that reads the default, not a knob — giving up 3–100 MB where
they won (1.03×–4.6×). A larger cache only lets mlxlearn build a pair's whole Gram at
once.

Narrow data loses at C = 1 too. At 2–8 features rbf lost on 1.7.2 at every size measured,
2 000 to 16 000 rows (0.50×–0.91×), and on 1.9.0 everywhere but a tie at 4 000 rows —
most past the kernel cache, where rows are fetched on demand (grid `svc-narrow`):

<!-- python -m benchmarks.calibrate table svc-narrow: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | rbf, C = 1 | dispatch |
|---|---|---|
| 2 000 × 2 | **0.64** / **0.56** | scikit-learn |
| 4 000 × 2 | 1.00~ / **0.88** | scikit-learn |
| 8 000 × 2 | **0.79** / **0.72** | scikit-learn |
| 16 000 × 2 | **0.54** / **0.50** | scikit-learn |
| 2 000 × 4 | **0.67** / **0.58** | scikit-learn |
| 4 000 × 4 | 1.02 / **0.89** | scikit-learn |
| 8 000 × 4 | **0.88** / **0.78** | scikit-learn |
| 16 000 × 4 | **0.68** / **0.61** | scikit-learn |
| 2 000 × 8 | **0.65** / **0.56** | scikit-learn |
| 4 000 × 8 | 1.02 / **0.89** | scikit-learn |
| 8 000 × 8 | 0.99~ / **0.91** | scikit-learn |
| 16 000 × 8 | **0.84** / **0.77** | scikit-learn |

Wide data past the cache won by more — rbf, C = 1, scikit-learn 1.9.0, three pairs each
(a stock fit at 32 000 × 256 takes nearly two minutes), too few for the significance test
to judge, so the ratio alone is the evidence (grid `svc-large`):

<!-- python -m benchmarks.calibrate table svc-large: scikit-learn 1.9.0 -->
| rows × features | rbf, C = 1 | dispatch |
|---|---|---|
| 16 000 × 256 | 10.52~ | MLX |
| 32 000 × 256 | 20.21~ | MLX |

At 32 000 × 256 the MLX fit took 6.9 s against 139 s, over the same number of SMO
iterations to within 0.1% (22 646 against LIBSVM's 22 667).

**The sigmoid kernel** never takes the MLX path. It won at C = 0.1 everywhere, and lost
at C = 100 from 64 features and at C = 10 from 128 (bar one win, at 8 000 × 256) — down to
0.06× — and at C = 1 at 2 000 × 128 and 1 000 × 256; the rules cannot see C (grid
`svc-sigmoid`):

<!-- python -m benchmarks.calibrate table svc-sigmoid: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | sigmoid, C = 0.1 | sigmoid, C = 1 | sigmoid, C = 10 | sigmoid, C = 100 | dispatch |
|---|---|---|---|---|---|
| 4 000 × 16 | 2.65 / 2.39 | 1.96 / 1.80 | 0.98~ / **0.89** | 1.08 / 0.97 | scikit-learn |
| 4 000 × 64 | 3.68 / 3.29 | 2.62 / 2.34 | 1.08 / 0.98 | **0.58** / **0.52** | scikit-learn |
| 2 000 × 128 | 3.56 / 3.20 | **0.93** / **0.84** | **0.53** / **0.48** | **0.12** / **0.11** | scikit-learn |
| 4 000 × 128 | 4.44 / 3.96 | 1.41 / 1.27 | **0.60** / **0.55** | **0.43** / **0.39** | scikit-learn |
| 1 000 × 256 | 3.63 / 3.25 | **0.44** / **0.40** | **0.17** / **0.15** | **0.07** / **0.06** | scikit-learn |
| 4 000 × 256 | 6.82 / 6.14 | 1.36 / 1.24 | **0.49** / **0.45** | **0.37** / **0.33** | scikit-learn |
| 8 000 × 256 | 5.51 / 5.12 | 2.91 / 2.71 | 1.29 / 1.19 | **0.61** / **0.57** | scikit-learn |
| 2 000 × 512 | 7.91 / 7.12 | 1.68 / 1.54 | **0.28** / **0.25** | **0.17** / **0.15** | scikit-learn |

**The linear kernel** never takes the MLX path either. Its iterations are the cheapest of
any kernel's, so per-iteration overhead is nearly the whole cost, and no width makes up
for it (grid `svc-linear`):

<!-- python -m benchmarks.calibrate table svc-linear: scikit-learn 1.9.0 -->
| rows × features | linear, C = 1 | dispatch |
|---|---|---|
| 2 000 × 20 | **0.10** | scikit-learn |
| 2 000 × 256 | **0.07** | scikit-learn |

**Many classes.** One-vs-one solves `K(K−1)/2` subproblems of about `2n/K` rows, each with
its own Gram and its own loop, so the mean pair's work, `(2n/K)·d`, must clear the binary
work floor itself, 384 000 elements. Balanced, well-separated blobs (the benchmark's
multiclass data), rbf, C = 1, same notation (grid `svc-multi`):

<!-- python -m benchmarks.calibrate table svc-multi: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features | K = 4 | K = 8 | K = 16 | K = 32 | K = 64 | dispatch |
|---|---|---|---|---|---|---|
| 4 000 × 32 | **0.36** / **0.32** | **0.25** / **0.22** | **0.18** / **0.16** | **0.10** / **0.09** | — | scikit-learn |
| 8 000 × 32 | **0.39** / **0.38** | **0.32** / **0.31** | **0.24** / **0.23** | **0.17** / **0.16** | — | scikit-learn |
| 8 000 × 256 | 1.85 / 1.71 | 1.65 / 1.55 | 1.21 / 1.17 | **0.77** / **0.70** | — | MLX at K = 4, K = 8 |
| 2 048 × 512 | 2.07 / 1.92 | 1.39 / 1.28 | **0.81** / **0.73** | **0.47** / **0.40** | **0.23** / **0.22** | MLX at K = 4 |

Pairs of 256 × 512 and 500 × 256 lost, 0.73×–0.81× and 0.70×–0.77×. At 32 features,
where each pair takes a hundred-odd iterations, every K lost, as the width floor already
says. On overlapping classes — the binary target's score cut into K quantile bands — at
C = 1 to 100, grid `svc-multi-overlap`:

<!-- python -m benchmarks.calibrate table svc-multi-overlap: scikit-learn 1.9.0 / 1.7.2 -->
| rows × features, classes | C = 1 | C = 10 | C = 100 | dispatch |
|---|---|---|---|---|
| 3 000 × 128, K = 3 (pairs of 2 000) | 1.65 / 1.50 | 1.00~ / **0.92** | 1.00~ / **0.92** | scikit-learn |
| 4 000 × 128, K = 4 (pairs of 2 000) | 1.61 / 1.49 | 1.00~ / **0.92** | 1.00 / **0.91** | scikit-learn |
| 4 500 × 128, K = 3 (pairs of 3 000) | 1.94 / 1.79 | 1.18 / 1.09 | 1.18 / 1.09 | scikit-learn |
| 6 000 × 128, K = 4 (pairs of 3 000) | 1.95 / 1.79 | 1.13 / 1.04 | 1.14 / 1.04 | scikit-learn |
| 7 500 × 128, K = 5 (pairs of 3 000) | 1.96 / 1.80 | 1.12 / 1.03 | 1.12 / 1.03 | scikit-learn |
| 15 000 × 128, K = 10 (pairs of 3 000) | 2.00 / 1.82 | 1.10 / 1.01 | 1.10 / 1.01 | scikit-learn |
| 2 000 × 256, K = 4 (pairs of 1 000) | 2.00 / 1.83 | 1.21 / 1.09 | 1.22 / 1.09 | scikit-learn |
| 3 000 × 256, K = 4 (pairs of 1 500) | 2.54 / 2.35 | 1.52 / 1.39 | 1.52 / 1.39 | MLX at C = 1 |
| 1 536 × 512, K = 3 (pairs of 1 024) | 3.48 / 3.19 | 2.01 / 1.82 | 2.00 / 1.82 | MLX at C = 1 |
| 2 048 × 512, K = 4 (pairs of 1 024) | 3.52 / 3.20 | 2.04 / 1.85 | 2.03 / 1.86 | MLX at C = 1 |

Pairs of 2 000 × 128 lost at C = 10 and 100 on 1.7.2 (0.91×–0.92×) and tied on 1.9.0;
every pair from 3 000 × 128 up won or tied on both (1.01×–3.5×) — on these overlapping
classes. On random labels pairs of 3 000 × 128 lost (`svc-hard`, 0.81×–0.91×), so every
pair below 256 features now goes to LIBSVM with the binary fits of that width, while pairs
of 1 500 × 256 and 1 024 × 512 won on both kinds of labels (1.23×–4.2×). The pair floor, at
the binary work floor, gives up pairs of 512 × 512 (2 048 rows, K = 8; 1.28×–1.39× above)
and of 1 000 × 256 (1.09×–2.0×), which won.

**None of this is a correctness problem.** The primal objective agrees with LIBSVM to
5.2e-07 relative and the KKT conditions hold on an independent float64 check.

### Inference: a few rows on the host

A model fitted on MLX used to answer every `predict` on the fitted device, and for a
batch of a few rows what the device charges is the call, not the work: 0.64–0.98 ms after
the GPU has idled, against 0.22–0.82 ms for LIBSVM's whole single-row predict (0.19–0.35
ms on the host, below). With the host threshold at 0 — every batch on the GPU — a single
row lost to LIBSVM at five of the six fits below on one version or both (0.27×–0.78×),
and two to four rows at four of them; only at 6 000 × 512, where LIBSVM's own predict is
slowest, did the device tie or win. So a batch of at most 2²⁵ rows ×
support vectors × features is evaluated on the host, in NumPy float32, with the device
tiles' own arithmetic (`svc_host_inference_work`); larger batches stay on the device.
Grid `svc-predict`: `predict` on an rbf model (C = 1) fitted on MLX against one fitted by
LIBSVM, 15 interleaved pairs; `dispatch` says which batch sizes the host answered:

<!-- python -m benchmarks.calibrate table svc-predict: scikit-learn 1.9.0 / 1.7.2 -->
| fit (support vectors) | 1 row | 2 rows | 4 rows | 8 rows | 16 rows | 32 rows | 64 rows | 128 rows | 256 rows | 1024 rows | 4096 rows | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 000 × 128 (2 169 SVs) | 1.13~ / 1.13 | 1.66 / 1.60 | 2.61 / 2.66 | 3.95 / 4.20 | 6.39 / 7.06 | 10.22 / 10.36 | 14.24 / 14.55 | 13.38 / 13.85 | 23.49 / 23.02 | 67.08 / 60.06 | 124.26 / 93.15 | scikit-learn |
| 1 500 × 256 (1 357 SVs) | 1.06~ / 1.21 | 1.42 / 1.51 | 2.37 / 2.56 | 3.80 / 4.04 | 6.15 / 6.76 | 10.60 / 10.80 | 14.80 / 15.72 | 12.81 / 12.15 | 21.25 / 21.86 | 59.01 / 55.10 | 102.15 / 85.24 | host to 64 rows, MLX above |
| 4 000 × 256 (3 115 SVs) | 1.62 / 1.66 | 2.06 / 2.21 | 3.69 / 3.90 | 5.99 / 6.59 | 10.34 / 11.17 | 16.46 / 16.51 | 13.87 / 14.56 | 24.66 / 25.32 | 43.45 / 40.45 | 101.83 / 85.11 | 151.61 / 98.77 | host to 32 rows, MLX above |
| 8 000 × 256, K = 4 (7 648 SVs) | 2.23 / 2.40 | 2.83 / 2.94 | 5.08 / 5.03 | 9.04 / 9.29 | 14.97 / 14.59 | 15.64 / 16.41 | 29.00 / 28.76 | 50.85 / 47.96 | 77.53 / 68.47 | 128.65 / 102.05 | 131.61 / 105.48 | host to 16 rows, MLX above |
| 1 100 × 512 (1 089 SVs) | 1.19 / 1.27 | 1.54 / 1.75 | 2.72 / 2.80 | 4.33 / 4.83 | 7.98 / 8.54 | 13.97 / 14.18 | 8.34 / 8.18 | 14.84 / 14.80 | 25.53 / 25.38 | 65.09 / 61.91 | 104.54 / 89.22 | host to 32 rows, MLX above |
| 6 000 × 512 (4 961 SVs) | 2.53 / 2.68 | 2.88 / 3.01 | 5.50 / 5.63 | 9.78 / 10.71 | 8.41 / 9.30 | 16.88 / 17.18 | 32.26 / 31.94 | 55.41 / 55.21 | 88.84 / 83.58 | 150.16 / 131.45 | 163.70 / 135.68 | host to 8 rows, MLX above |

Every batch size won or tied on both versions: single rows 1.06×–2.7×, and from 8 rows
up 3.8×–164×. The host stayed faster than a GPU that has idled up to about 5·10⁷ of that
work at 128 features, and further out at wider data, so the threshold sits below the
crossover at every shape; the dip where a shape crosses it (64 rows at 1 100 × 512,
14× → 8×) is the device's wake-up coming back, not a loss.

### The published SVC cases

Forced onto MLX, 25 interleaved pairs (5 for the linear kernel), scikit-learn 1.9.0:

| case | mlxlearn (ms) | scikit-learn (ms) | median | best | dispatch |
|---|---|---|---|---|---|
| 4 000 × 32 | 92.63 | 124.57 | 1.34× | 1.31× | scikit-learn |
| 4 000 × 64 | 115.29 | 181.83 | 1.58× | 1.59× | scikit-learn |
| 4 000 × 256 | 165.93 | 616.98 | 3.72× | 3.79× | MLX |
| 1 000 × 512 | 24.92 | 76.15 | 3.06× | 3.06× | scikit-learn |
| 4 000 × 64, C = 10 | 333.22 | 278.91 | **0.84×** | 0.84× | scikit-learn |
| 4 000 × 64, C = 10, kernel = poly | 315.80 | 276.13 | **0.87×** | 0.86× | scikit-learn |
| 2 000 × 128, C = 10 | 104.99 | 111.20 | 1.06× | 1.04× | scikit-learn |
| 2 000 × 128, C = 10, kernel = poly | 96.65 | 98.65 | 1.02× | 1.03× | scikit-learn |
| 4 000 × 128, C = 10 | 299.11 | 421.73 | 1.41× | 1.41× | scikit-learn |
| 4 000 × 128, C = 10, kernel = poly | 299.77 | 395.03 | 1.32× | 1.32× | scikit-learn |
| 4 000 × 256, C = 10 | 294.43 | 735.17 | 2.50× | 2.50× | scikit-learn |
| 4 000 × 256, C = 10, kernel = poly | 285.66 | 655.52 | 2.29× | 2.32× | scikit-learn |
| 4 000 × 256, C = 10, kernel = sigmoid | 647.17 | 426.00 | **0.66×** | 0.66× | scikit-learn |
| 2 000 × 20, kernel = linear (5 pairs) | 552.00 | 37.80 | **0.07×** | 0.07× | scikit-learn |
| 2 000 × 256, kernel = linear (5 pairs) | 21512.67 | 1059.74 | **0.05×** | 0.05× | scikit-learn |
| 2 048 × 512, K = 4 | 29.11 | 59.15 | 2.03× | 2.06× | MLX |
| 2 048 × 512, K = 16 | 283.88 | 226.48 | **0.80×** | 0.80× | scikit-learn |
| 2 500 × 20, K = 4 | 14.88 | 3.35 | **0.23×** | 0.22× | scikit-learn |

The forced wins below the floors that patched dispatch leaves on LIBSVM are the price of
rules that cannot see C, the labels or the iteration count: the same shapes at C = 10
lost. The poly cases go to LIBSVM at any size now (`svc-poly`, above). 2 000 × 128 at C = 10 won by 1.02×–1.06× here and lost on scikit-learn 1.7.2 (the
`svc` table); 4 000 × 128 won here, on the benchmark's target, while labels without
signal lost at 3 000–6 000 × 128 (the `svc-hard` table), which is why the width floor
now sits at 256 features. 4 000 × 256 at C = 10 won here too, on Gaussian data, and goes
to LIBSVM anyway: above C = 1 data of lower intrinsic dimension lost at those widths (the
`svc-structured` table). Every case here that patched dispatch sends to MLX won,
2.03×–3.72×.

## Logistic regression — tall, or many classes

`fit` with `max_iter=200`. Binary targets are the benchmark data; multinomial ones are
overlapping Gaussian blobs, one per class, whose separation does not grow with the width.
Each cell is stock time over MLX time, scikit-learn 1.9.0 / 1.7.2, median of 9 interleaved
pairs; bold and `~` as above. Grids `logreg-binary` and `logreg-multi`:

<!-- python -m benchmarks.calibrate table logreg-binary: scikit-learn 1.9.0 / 1.7.2 -->
| rows | 1 feature | 4 features | 16 features | 32 features | 64 features | 128 features | 256 features | 1024 features | dispatch |
|---|---|---|---|---|---|---|---|---|---|
| 5 000 | **0.26** / **0.24** | **0.23** / **0.28** | **0.32** / **0.28** | **0.29** / **0.29** | — | — | — | — | scikit-learn |
| 20 000 | **0.48** / **0.55** | **0.45** / **0.52** | **0.42** / 0.54~ | **0.43** / **0.49** | **0.59** / **0.50** | 0.74~ / **0.55** | **0.83** / **0.65** | **0.87** / **0.83** | scikit-learn |
| 50 000 | 1.30 / 1.03~ | 1.27 / 1.10~ | 1.02~ / 0.93~ | 1.01~ / 0.81~ | 1.11 / 0.97~ | 1.15 / 1.08 | 1.25 / 1.09 | 1.11 / 1.03 | scikit-learn |
| 100 000 | 2.22 / 2.10 | 1.82 / 1.71 | 1.64 / 1.52 | 1.65 / 1.51 | 1.48 / 1.35 | 1.45 / 1.35 | 1.35 / 1.30 | 1.45 / 1.39 | scikit-learn |
| 200 000 | 3.19 / 2.98 | 2.39 / 2.20 | 2.32 / 2.12 | 2.00 / 1.88 | 1.79 / 1.63 | 1.65 / 1.55 | 1.63 / 1.58 | 1.58 / 1.53 | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |
| 1 000 000 | 6.17 / 5.54 | 4.39 / 3.77 | 4.39 / 3.96 | 3.82 / 3.55 | 2.85 / 2.63 | 2.30 / 2.14 | 1.95 / 1.87 | — | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |

<!-- python -m benchmarks.calibrate table logreg-multi: scikit-learn 1.9.0 / 1.7.2 -->
| rows | 3 classes × 1 | 3 classes × 4 | 3 classes × 16 | 3 classes × 64 | 3 classes × 512 | 5 classes × 64 | 10 classes × 16 | 10 classes × 128 | dispatch |
|---|---|---|---|---|---|---|---|---|---|
| 5 000 | 0.64~ / 0.78~ | **0.56** / **0.55** | **0.41** / **0.45** | **0.37** / **0.39** | 1.23 / 1.14 | **0.46** / **0.48** | 0.81~ / 0.91~ | **0.66** / **0.62** | scikit-learn |
| 20 000 | 0.78~ / 1.15~ | 0.99~ / 0.98~ | 1.22 / 1.18 | **0.86** / 0.79~ | 1.80 / 1.73 | 1.11 / 1.06 | 2.34 / 2.29 | 1.44 / 1.39 | scikit-learn |
| 60 000 | 1.88 / 1.87 | 3.38 / 3.49 | 1.52 / 1.50 | 2.13 / 2.17 | 2.79 / 2.93 | 2.76 / 2.53 | 3.34 / 3.04 | 2.33 / 2.32 | MLX at 3 classes × 4, 3 classes × 16, 3 classes × 64, 3 classes × 512, 5 classes × 64, 10 classes × 16, 10 classes × 128 |
| 200 000 | 2.67 / 2.67 | 3.43 / 3.21 | 4.76 / 4.60 | 3.77 / 3.78 | 3.96 / 3.81 | 4.46 / 4.28 | 8.58 / 8.42 | 5.23 / 5.24 | MLX at 3 classes × 4, 3 classes × 16, 3 classes × 64, 3 classes × 512, 5 classes × 64, 10 classes × 16, 10 classes × 128 |

What an MLX fit pays once is a pass over `X` — the column means, the centered float32
copy and the float32 range check, each split by rows across up to eight threads — its
upload, and a device round trip per objective evaluation; what it saves is per
iteration, the loss and its gradient on the GPU. The iteration counts follow
scikit-learn's: no fit patched dispatch sends to MLX took more than one iteration more
than scikit-learn's in any table here, and multinomial fits on overlapping blobs often
took fewer (22 against 36 at 60 000 × 4). The first step is L-BFGS-B's own, a unit move
along the first gradient, and the line search takes an interpolated step anywhere past
1% of its bracket rather than bisecting until the step clears 10%.

- **Binary: faster from 100 000 rows at every width up to 1 024,** 1.30×–6.2× on both
  versions; at 50 000 rows it depends on the width, and below it loses.
- **Multinomial: faster from 60 000 rows at every width,** 1.50×–8.6× at 3, 5 and 10
  classes and 1–512 features on both versions — on these classes; on labels without
  signal, not at one feature (below). At 20 000 rows it depends on the classes and the
  width (0.78×–2.3×); at 5 000 it loses or ties at every width but 512.

A fit that converges in one or two iterations is mostly the fixed cost, and both of those
targets hide one. Classes that are well separated — `benchmarks/run.py`'s multiclass
data, whose centres move apart with the width — converge in one scikit-learn iteration
from 64 features on; grid `logreg-separable`:

<!-- python -m benchmarks.calibrate table logreg-separable: scikit-learn 1.9.0 / 1.7.2 -->
| rows | 3 classes × 16 | 3 classes × 64 | 3 classes × 128 | 3 classes × 256 | 3 classes × 512 | 5 classes × 64 | 5 classes × 256 | 10 classes × 128 | 10 classes × 512 | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|
| 60 000 | 1.54 / 1.49 | 1.20 / 1.14 | 1.36 / 1.30 | 1.02~ / 1.00~ | 1.11 / 1.09 | 2.35 / 2.31 | 1.08 / 1.06 | 2.29 / 2.25 | 1.22 / 1.21 | MLX |
| 200 000 | 4.72 / 4.74 | 2.64 / 2.67 | 1.25 / 1.22 | 1.20 / 1.18 | 1.35 / 1.29 | 4.15 / 4.00 | 1.29 / 1.28 | 4.52 / 4.37 | 1.51 / 1.50 | MLX |

Labels that carry no signal converge in one or two, and noisy ones in about four; grid
`logreg-labels` fits both, random labels and `x₀ + N(0, 3²) > 0`:

<!-- python -m benchmarks.calibrate table logreg-labels: scikit-learn 1.9.0 / 1.7.2 -->
| labels, rows | 1 feature | 4 features | 16 features | 32 features | 64 features | 256 features | 3 classes × 1 | 3 classes × 2 | 3 classes × 4 | 3 classes × 16 | 3 classes × 64 | 3 classes × 256 | 5 classes × 1 | 5 classes × 2 | 6 classes × 1 | 7 classes × 1 | 8 classes × 1 | 9 classes × 1 | 10 classes × 1 | 10 classes × 64 | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noisy, 100 000 | 1.71 / 1.16 | 1.40 / 1.07~ | 1.25 / **0.88** | 1.36 / 1.03~ | 1.24 / 1.08~ | 1.01~ / 0.97 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | scikit-learn |
| noisy, 200 000 | 2.71 / 2.37 | 1.89 / 1.58 | 1.93 / 1.53 | 1.74 / 1.55 | 1.43 / 1.25 | 1.20 / 1.13 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |
| noisy, 500 000 | 3.91 / 3.43 | 2.36 / 2.07 | 2.46 / 2.17 | 2.16 / 1.94 | 1.69 / 1.55 | 1.37 / 1.29 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |
| noisy, 1 000 000 | 4.52 / 3.97 | 2.95 / 2.61 | 2.77 / 2.46 | 2.42 / 2.16 | 1.94 / 1.73 | 1.46 / 1.37 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |
| random, 60 000 | — | — | — | — | — | — | **0.74** / 0.70~ | 1.27 / 1.11 | 1.12 / 1.06~ | 1.22 / 1.18 | 1.71 / 1.66 | 1.42 / 1.38 | **0.92** / 0.87~ | 1.31 / 1.24 | 0.98~ / 1.03 | 1.09 / 1.17 | 1.06 / 1.10 | 1.09 / 1.14~ | 1.15 / 1.74 | 1.76 / 1.70 | MLX at 3 classes × 2, 3 classes × 4, 3 classes × 16, 3 classes × 64, 3 classes × 256, 5 classes × 2, 10 classes × 64 |
| random, 100 000 | 1.66 / 1.01~ | **0.81** / **0.61** | 1.17 / **0.77** | 1.16 / **0.82** | 1.01~ / **0.76** | **0.90** / **0.84** | 1.53~ / 1.51~ | 1.34 / 1.31 | 1.48 / 1.51 | 1.48 / 1.47 | 2.11 / 2.09 | 1.63 / 1.56 | 2.02 / 2.02 | 1.63 / 1.67 | — | — | — | — | 1.93 / 1.84 | 2.28 / 2.17 | MLX at 3 classes × 2, 3 classes × 4, 3 classes × 16, 3 classes × 64, 3 classes × 256, 5 classes × 2, 10 classes × 64 |
| random, 120 000 | — | — | — | — | — | — | 1.21~ / 1.36~ | — | — | — | — | — | — | — | — | — | — | — | — | — | scikit-learn |
| random, 150 000 | — | — | — | — | — | — | 1.07~ / 1.22~ | — | — | — | — | — | — | — | — | — | — | — | — | — | scikit-learn |
| random, 200 000 | 1.67 / 1.15 | 1.18 / 0.97~ | 1.36 / 1.12 | 1.45 / 1.27 | 1.18 / 1.03 | 1.03 / 0.96 | 1.34~ / 1.19~ | 1.32 / 1.36 | 1.58 / 1.55 | 2.40 / 2.45 | 2.10 / 1.98 | 2.02 / 1.98 | 1.37~ / 1.91 | 1.81 / 1.81 | — | — | — | — | 1.84 / 2.18 | 2.25 / 2.26 | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features, 3 classes × 2, 3 classes × 4, 3 classes × 16, 3 classes × 64, 3 classes × 256, 5 classes × 2, 10 classes × 64 |
| random, 500 000 | 1.84 / 1.47 | 2.00 / 1.70 | 1.78 / 1.53 | 1.58 / 1.35 | 1.26 / 1.09 | 1.17 / 1.09 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |
| random, 1 000 000 | 2.27 / 1.76 | 1.94 / 1.57 | 1.89 / 1.58 | 1.59 / 1.35 | 1.41 / 1.21 | 1.06 / 0.97 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | MLX at 1 feature, 4 features, 16 features, 32 features, 64 features |

Separable classes tied at worst (1.00×~ at 60 000 × 256, three classes), so the
multinomial row floor stands. Random labels at a single feature did not: a fit
scikit-learn finishes in one iteration of 7–12 ms, where MLX's fixed cost is about 10 ms at
60 000 × 1 whatever the class count — its matrix product with a one-column output is slow
— against 7 at 60 000 × 2. At 60 000 × 1 three to nine classes lost or tied
(0.70×–1.17×; six read 0.93×–0.94× on 1.9.0 in two runs outside the grid), and
120 000–200 000 × 1 with three stayed within noise (1.07×–1.36×), while from two features
every multinomial fit won or tied (1.06×~ at worst, 60 000 × 4 with three classes). So a
multinomial fit needs two features (`logreg_multinomial_min_features`); that replaces a
work floor of 360 000 rows × features × classes, which 60 000 × 1 with six classes met.
It gives up the one-feature fits that won: five and ten random classes at
100 000–200 000 rows (1.37×–2.2×), ten at 60 000 (1.15×–1.74×), and 200 000 × 1 with three
overlapping ones (2.67× in the `logreg-multi` table). Labels with little or no signal lost at 100 000 rows — on
1.7.2 at 4–256 features (0.61×–0.88×), on 1.9.0 at 4 and 256 (0.81×, 0.90×) — and random
ones at 256 features stood at 0.96×–1.17× from 200 000 rows, too close to call. So a binary fit needs 200 000 rows and at most 64 features, where every label set
measured won or tied (0.97× at worst); the price is the binary wins between 100 000 and
200 000 rows, and beyond 64 features (1.30×–2.3× in the `logreg-binary` table).

**Solver parameters stop a fit early too.** A looser `tol` or a smaller `max_iter` stops a
fit with real signal within an iteration or two, as labels without any stop. Grid
`logreg-params`: the benchmark's binary target, labels with moderate multinomial-logit
signal (`signal`: the most likely class under `XW/√d` plus Gumbel noise) and random
labels, binary at 200 000 × 1–64, three classes at 60 000 × 2–64 and overlapping blobs at
60 000 × 16, at `tol` from 1e-5 to 0.01 and `max_iter` from 1 to 5. These ratios sit close
to 1, and at 9 pairs a point they did not reproduce — an earlier run read `tol = 0.01` on
1.7.2 at 200 000 × 16 as a 0.82× loss and a rerun as 1.03×, and counted 11 and then 6 losses at
`max_iter = 1` — so this grid takes 21 pairs a point, and a second full run on both
versions found the same losses at the same points, each within 0.03 of this one, and no other:

<!-- python -m benchmarks.calibrate table logreg-params: scikit-learn 1.9.0 / 1.7.2 -->
| labels, rows × features | defaults | tol = 1e-05 | tol = 0.0005 | tol = 0.001 | tol = 0.003 | tol = 0.01 | max_iter = 1 | max_iter = 2 | max_iter = 3 | max_iter = 5 | dispatch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| binary, K = 2, 200 000 × 1 | 3.17 / 2.97 | 3.52 / 3.42 | 3.46 / 3.36 | 3.41 / 3.27 | 3.22 / 3.01 | 3.02 / 2.75 | 1.93 / 1.53 | 2.42 / 1.91 | 2.56 / 2.35 | 3.06 / 2.77 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |
| binary, K = 2, 200 000 × 4 | 2.45 / 2.27 | 2.59 / 2.44 | 2.36 / 2.19 | 2.37 / 2.20 | 2.27 / 2.10 | 2.05 / 1.83 | 1.15 / 0.96~ | 1.46 / 1.25 | 1.68 / 1.49 | 2.03 / 1.82 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |
| binary, K = 2, 200 000 × 16 | 2.35 / 2.20 | 2.48 / 2.35 | 2.30 / 2.15 | 2.24 / 2.14 | 2.23 / 2.05 | 1.97 / 1.82 | 1.14 / 1.02~ | 1.45 / 1.25 | 1.67 / 1.47 | 2.00 / 1.77 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |
| binary, K = 2, 200 000 × 64 | 1.77 / 1.66 | 1.85 / 1.75 | 1.77 / 1.65 | 1.70 / 1.56 | 1.69 / 1.53 | 1.57 / 1.41 | **0.91** / **0.77** | 1.12 / 0.99~ | 1.27 / 1.17 | 1.59 / 1.42 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |
| blobs, K = 3, 60 000 × 16 | 1.54 / 1.49 | 1.49 / 1.51 | 1.40 / 1.38 | 1.35 / 1.31 | 1.49 / 1.42 | 1.40 / 1.38 | 0.95~ / **0.93** | 1.05 / 1.05~ | 1.16 / 1.12 | 1.29 / 1.24 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |
| random, K = 2, 200 000 × 16 | 1.39 / 1.19 | 1.54 / 1.34 | 1.39 / 1.16 | 1.38 / 1.17 | **0.80** / **0.58** | **0.80** / **0.59** | 1.37 / 1.16 | 1.34 / 1.17 | 1.36 / 1.17 | 1.38 / 1.19 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |
| random, K = 2, 200 000 × 64 | 1.23 / 1.07 | 1.22 / 1.08 | 1.05 / **0.93** | 1.05 / **0.93** | **0.65** / **0.50** | **0.64** / **0.51** | 1.07 / **0.92** | 1.23 / 1.06 | 1.23 / 1.05 | 1.19 / 1.05 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |
| signal, K = 2, 200 000 × 1 | 4.10 / 3.82 | 3.67 / 3.41 | 3.18 / 2.99 | 3.19 / 2.93 | 3.04 / 2.77 | 2.79 / 2.39 | 1.87 / 1.49 | 2.30 / 2.00 | 2.64 / 2.32 | 3.07 / 2.81 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |
| signal, K = 2, 200 000 × 4 | 1.97 / 1.73 | 2.11 / 1.94 | 1.73 / 1.54 | 1.74 / 1.54 | 1.74 / 1.52 | 1.46 / 1.27 | 1.16 / 0.97~ | 1.49 / 1.28 | 1.72 / 1.52 | 1.92 / 1.74 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |
| signal, K = 2, 200 000 × 16 | 1.97 / 1.75 | 2.12 / 1.88 | 1.71 / 1.52 | 1.44 / 1.26 | 1.46 / 1.27 | 1.18 / 0.98~ | 1.19 / 0.97~ | 1.46 / 1.28 | 1.72 / 1.49 | 1.95 / 1.70 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |
| signal, K = 2, 200 000 × 64 | 1.65 / 1.55 | 1.77 / 1.66 | 1.52 / 1.38 | 1.33 / 1.21 | 1.36 / 1.22 | 1.14 / 1.00~ | **0.89** / **0.76** | 1.13 / 1.02~ | 1.31 / 1.20 | 1.68 / 1.54 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |
| signal, K = 3, 60 000 × 2 | 1.45 / 1.45 | 1.56 / 1.58 | 1.37 / 1.37 | 1.36 / 1.36 | 1.38 / 1.38 | 1.28 / 1.27 | 1.12 / 1.14 | 1.27 / 1.30 | 1.39 / 1.37 | 1.48 / 1.44 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |
| signal, K = 3, 60 000 × 8 | 1.48 / 1.45 | 1.59 / 1.54 | 1.52 / 1.44 | 1.41 / 1.40 | 1.43 / 1.39 | 1.20 / 1.20 | 1.02~ / 1.06 | 1.19 / 1.20 | 1.33 / 1.30 | 1.47 / 1.47 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |
| signal, K = 3, 60 000 × 64 | 2.14 / 2.14 | 2.44 / 2.33 | 2.05 / 2.01 | 2.06 / 2.00 | 2.22 / 2.18 | 1.58 / 1.58 | 1.33 / 1.30 | 1.59 / 1.58 | 1.78 / 1.79 | 2.04 / 2.00 | MLX at defaults, tol = 1e-05, max_iter = 3, max_iter = 5 |

Random labels are where a looser `tol` loses: `tol` of 3e-3 and 0.01 at 200 000 × 16 and
64 on both versions (0.50×–0.82×), and 5e-4 and 1e-3 at 200 000 × 64 on 1.7.2
(0.90×–0.93× over the two runs). At the default and at 1e-5 every fit won (1.07× at
worst). `max_iter = 1` lost at 200 000 × 64 on both versions and at 60 000 × 16 with
three classes on 1.7.2 (0.76×–0.94×); `max_iter = 2` tied at worst (0.98×~), and from 3
up every fit won (1.05× at worst). So patched dispatch hands a fit whose `tol` is above
scikit-learn's default, 1e-4 — the `tol` every other table here was measured at — or
whose `max_iter` is below 3 to scikit-learn, whatever its size (`logreg_min_max_iter`;
the `tol` rule reads the default and has no knob), recorded as `"below-crossover"`; a
directly imported class keeps it on MLX. That gives up every looser `tol` that won
(1.05×–3.5×) and `max_iter = 2` (1.05×–2.4×).

The 0.1.0a2 rule — 20 000 rows and 2·10⁷ elements, both required, no width limit — sent
20 000 × 1 024 (0.83×–0.87× even now) to MLX and kept 1 000 000 × 16 (4.0×–4.4×) from it.
Before the host passes were threaded and the first step matched L-BFGS-B's, binary fits
lost at 64 features and more up to 200 000 rows, and separable multinomial fits lost
0.31×–0.46× at 256 and 512 features, where MLX took 11–15 iterations to get where
scikit-learn's one did.

### The published logistic cases

Forced onto MLX, 25 interleaved pairs, scikit-learn 1.9.0; the multinomial targets are
the benchmark's well-separated blobs:

| case | mlxlearn (ms) | scikit-learn (ms) | median | best | dispatch |
|---|---|---|---|---|---|
| 20 000 × 16 | 7.07 | 3.08 | **0.44×** | 0.66× | scikit-learn |
| 200 000 × 16 | 11.29 | 25.46 | 2.26× | 2.34× | MLX |
| 1 000 000 × 16 | 31.81 | 142.24 | 4.47× | 5.21× | MLX |
| 200 000 × 128 | 22.59 | 37.01 | 1.64× | 1.70× | scikit-learn |
| 20 000 × 1024 | 21.68 | 19.24 | **0.89×** | 0.90× | scikit-learn |
| 100 000 × 1024 | 62.14 | 90.76 | 1.46× | 1.49× | scikit-learn |
| 20 000 × 64, K = 5 | 10.59 | 12.09 | 1.14× | 1.17× | scikit-learn |
| 60 000 × 64, K = 5 | 18.67 | 52.77 | 2.83× | 5.27× | MLX |
| 200 000 × 64, K = 5 | 31.37 | 108.77 | 3.47× | 4.67× | MLX |

## Known limits of patched dispatch

Every rule on this page holds on the data its grids measured, and that is all a
measurement can promise. The grids cover:

- **Neighbor estimators.** Gaussian data at 2–4 096 features and 2 048–2 097 152 rows,
  every batch of 1–4 096 queries, at k = 10 (`knn`, `knn-rows`, `knn-narrow`,
  `knn-boundary`, `knn-narrow-brute`, `knn-tree`, `knn-batches`) and k = 1–200, asked of
  the fit (`knn-k`) or of a query, in float64 and float32 (`knn-query-k`); three latent
  dimensions in 8–12 features (`knn-tree-lowrank`); uniform data, 100 clusters of std
  0.1–1, rows repeated about 2 and 20 times, one column on 100 or 10⁴ times the scale of
  the rest, binary and count features (`knn-data`, `knn-structured`, `knn-narrow`).
- **`SVC`.** Gaussian data at 2–512 features and 1 000–32 000 rows, labels with a clear
  signal, a weak one, none and XOR, 2–64 classes, overlapping and separated, C = 0.1–100,
  rbf `gamma` 0.03–1 000 × `"scale"` (`svc`, `svc-hard`, `svc-params`, `svc-multi`,
  `svc-multi-overlap`, `svc-narrow`, `svc-large`, `svc-predict`); five or ten latent
  directions plus noise, singular values 1/i, smoothed, Student-t(2), Cauchy and
  column-scaled features, and scikit-learn's digits upsampled to 256–784 pixels, with
  `tol = 1e-5` and balanced class weights among them (`svc-structured`, `svc-poly`);
  `cache_size` of 1–1 000 MB (`svc-cache`); and the poly, sigmoid and linear kernels,
  which is why they stay on LIBSVM (`svc-params`, `svc-poly`, `svc-sigmoid`,
  `svc-linear`).
- **`LogisticRegression`.** Binary and 3–10-class labels with a clear signal, a moderate
  one and none, overlapping and separated classes, 1–1 024 features and
  5 000–1 000 000 rows (`logreg-binary`, `logreg-multi`, `logreg-separable`,
  `logreg-labels`), and `tol` of 1e-5–0.01 and `max_iter` of 1–200 (`logreg-params`).

One measured case loses by a few percent inside them: a query asking for more than 50
neighbors of a model fitted on MLX from float32 data at a million rows, which the
float64 copy answers at 0.93×–0.95× of stock (`knn-query-k`, above) — and MLX, at 1 024
queries, slower still.

**Why data outside them can still lose.** MLX pays more than scikit-learn per call and per
solver iteration: a float32 copy of the data and its upload, a device round trip for each
objective evaluation or batch of kernel rows, and, after the GPU has idled, a slower first
few milliseconds. A rule can weigh that against what a fit shows before it runs — its
shape, its parameters, and for a neighbor fit a probe of 512 of its own rows. It cannot
see what decides the rest: how many rows become support vectors and how often SMO revisits
them, which the labels and the data's intrinsic dimension set, and how many iterations
L-BFGS takes to converge. Where those went against MLX on a grid, the rules hand the whole
region to scikit-learn — C above 1, any kernel but rbf, a `cache_size` below the default,
a `tol` looser than the default — rather than add a rule per corner. Data unlike every grid
can still give a fit the rules send to MLX more iterations or more support vectors than
any grid did, and then it runs slower than stock scikit-learn. A neighbor query's cost is
set by the shape and k, which the rules read, and by the data only through the first
pass's certificate, which the probe reads; the SVC and logistic fits are where the
unknowns are.

**Keeping a workload on scikit-learn.** Patch only the estimators you want accelerated
(`patch_sklearn(["NearestNeighbors"])`), call `unpatch_sklearn()`, or set
`MLXLEARN_FORCE_BACKEND=sklearn` in the environment before mlxlearn is imported — or call
`mlxlearn.reset_config()` after setting it — which sends every patched fit to
scikit-learn whatever its size, recorded as `"forced"` (a directly imported class, which
has no scikit-learn to hand a fit to, then raises).

**Reporting a case.** Open an issue at <https://github.com/tabulai/mlx-learn/issues> with
the shape, the estimator's parameters, the fit's `fit_event_` (which rule sent it where),
the timings with and without `patch_sklearn()`, the scikit-learn, MLX and mlxlearn
versions and the machine, and, if you can, a few lines that generate data like yours. A
case that reproduces becomes a grid point, and the rule moves.

## Reading these numbers honestly

- Below a threshold, mlxlearn being slower is the design, not a defect: patched dispatch
  never runs that path. The calibration and kernel tables show those cases so that the
  thresholds can be checked, not as claims.
- "No significant regression" is the gate; "faster" is the goal. A case inside the measured
  noise passes.
- Nothing is averaged across operations. An average over fit and predict describes a
  workload nobody runs.
- `knn/fit` is the one knowingly-accepted regression, and it is listed by name; a patched
  fit the data probe hands to scikit-learn pays for the index and the probe first.
- The rules were measured on synthetic data and scikit-learn's digits, and data unlike
  them can land elsewhere: see [Known limits of patched dispatch](#known-limits-of-patched-dispatch)
  for what the grids cover and why a fit outside them can still lose.
