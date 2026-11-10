# Benchmarks

The [fused narrow-kernel rerun of the three original tall datasets](fused-narrow-kernel-2026-10-27.md)
measures 11.8×, 5.8× and 2.7× sklearn speedups for 500,000 × 32,
500,000 × 128 and 60,000 × 1,024, with every measured fit independently
checked for accuracy. The [earlier recheck](tall-logistic-recheck-2026-10-27.md)
(5.24×, 3.05× and 2.64×) records the timings before that kernel change.

For the latest development logistic GPU implementation, completed benchmark
panels and validation status, see the
[cooperative L-BFGS and guarded Newton report](rank-safe-newton-performance-2026-10-27.md).
It includes independent tight references and general rank-deficiency tests,
with remaining speed limitations and failed historical comparisons retained. The
[preceding ill-conditioned solver investigation](ill-conditioned-hvp-investigation-2026-10-27.md)
records the earlier unresolved gap and rejected correction candidates.
The previous [wide accuracy and device-sketch report](wide-accuracy-device-repair-2026-10-27.md)
records the preceding development measurements.

The development version uses native Metal C++ kernels for binary GPU logistic
regression. The [earlier overhead comparison](overhead-logistic-benchmarks.md)
measures complete fits against a frozen copy of the previous wide implementation
and stock scikit-learn, with all raw samples, stage profiling, timing variability,
and convergence/quality diagnostics. Its reproduction commands include the exact
reference archive. The [native command-loop comparison](native-loop-optimization-2026-10-26.md)
records the subsequent lifetime fix, adaptive scheduling, accuracy checks and mixed timings.
The [earlier prototype comparison](metal-command-benchmarks.md) is historical.
The [previous wide comparison](wide-logistic-benchmarks.md)
records the preceding matrix-kernel changes.
The [preceding native comparison](native-logistic-benchmarks.md) records the
earlier optimizer changes and tall-data results.
The [earlier fusion comparison](gpu-logistic-benchmarks.md) remains available as
the intermediate optimization baseline.
The measurements below describe the **0.1.0a2 baseline**, including the earlier
logistic solver. Current dispatch decisions are documented with the wide results.

The linked timing panels are historical and predate the current shifted-input
convergence changes. The [convergence investigation](convergence-accuracy-2026-10-27.md)
records an earlier single-step wide progress rule, including a 9–11% slowdown
on tested 20,000 × 4,096 fits, a separate `C=10` counterexample, and the
revised two-step average. The revised rule passed the full suite and targeted
panel, with 12.9–14.1% slower 4,096-feature fits than the frozen baseline; the
full panel also passed, with 12.1–14.5% slower 4,096-feature fits. The stopping
condition and selective CPU float64 certificate are
heuristics rather than universal accuracy guarantees; four severely shifted
multinomial stress cases still report nonconvergence. The original-coordinate
certificate may add host scoring passes when it is triggered.

The later [device and Hessian-sketch report](gpu-hessian-device-logistic-2026-10-27.md)
describes the new GPU sketch and NumPy-free runtime path; its final validation
and same-input measurements supersede the historical panels for those paths.

The current dispatch benchmark can be run with:

```bash
python -m benchmarks.run --profile smoke --report bench/report.md
```

Reproducing the historical measurements below requires the 0.1.0a2 source and
its recorded environment; running that command on development code measures the
updated implementation.

## Method

**Synchronized.** MLX is lazy, so a benchmark that stops the clock after issuing work
measures how long it took to *schedule* — a spectacular and meaningless number. Every
timing forces evaluation first.

**Interleaved.** mlxlearn and scikit-learn are timed A, B, A, B rather than all of one then
all of the other, so a thermal ramp or a background process degrades both sides equally
instead of whichever ran second.

**Median with its spread, and best case too.** At small sizes scikit-learn's timings are
heavy-tailed: thread-pool scheduling and garbage collection produce occasional outliers
several times the median. Measured on a 500 × 8 neighbor query, scikit-learn's median was
26.3 ms and its own minimum 1.55 ms — the same case reads as 17× on medians and 1.2× on
minima. Both are true; publishing only the first would not be. A crossover requires a win
on **both**.

**Significance.** A difference counts only when it exceeds the uncertainty *of the median*,
roughly `IQR / √n` — not the raw IQR, which for a heavy-tailed distribution declared a
measured 10× "within noise".

**Patched dispatch is what is measured.** The performance claim is about the classes
`patch_sklearn()` installs, and those hand sub-crossover problems to scikit-learn outright.
`--layer direct` measures the Layer 1 classes, which never dispatch to scikit-learn and use
their own CPU path instead — useful for development, misleading as a headline.

**Fixed hardware.** Absolute timings only compare across runs on the same machine. Shared
CI runners do correctness only; the nightly job pins itself to fixed self-hosted Apple
silicon.

Environment for the numbers below: Apple M4 Max, 128 GiB, macOS 25.2.0, Python 3.13.3,
MLX 0.31.2, scikit-learn 1.7.2, NumPy 2.5.1.

## Measured crossovers, and the thresholds they set

| Algorithm / operation | Crossover | Best speedup | Threshold in `Tuning` |
|---|---|---|---|
| `knn/kneighbors` | ~250–1 000 samples | 17.4× | `knn_min_train_samples = 256`, `knn_min_work = 2048` |
| `knn/fit` | never | — | accepted regression, see below |
| `svc/fit` | ~2 000 samples **and** ~10⁶ elements; never for `kernel="linear"` | 2.93× | `svc_min_samples = 2048`, `svc_min_work = 1_000_000` |
| `logreg/fit` | ~20 000 samples **and** ~2·10⁷ elements | 1.34× | `logreg_min_samples = 20_000`, `logreg_min_work = 20_000_000` |
| `pca/fit` | by solver: `covariance_eigh` ~8·10⁶ elements **or** 512 features; `randomized` ~8·10⁶ elements; `full` and `arpack` never | 14.3× | `pca_covariance_min_work = 8_000_000`, `pca_covariance_min_features = 512`, `pca_randomized_min_work = 8_000_000` |

These feed `mlxlearn._common.config.Tuning`, which dispatch reads. That is the point: a
published crossover the dispatcher ignores is a claim, not a mechanism.

## Neighbors — the strong result

Query speedups against stock scikit-learn's brute-force path, forced onto the MLX device to
isolate the kernel from the dispatch decision, k = 10, 256 queries, 25 iterations:

| samples × features | mlxlearn (ms) | scikit-learn (ms) | median | best |
|---|---|---|---|---|
| 250 × 8 | 1.48 | 4.26 | 2.87× | 0.92× |
| 500 × 64 | 1.78 | 29.86 | 16.79× | 3.30× |
| 1 000 × 768 | 2.43 | 25.54 | 10.50× | 2.86× |
| 4 000 × 64 | 2.46 | 26.95 | 10.96× | 4.80× |
| 8 000 × 768 | 5.13 | 44.39 | 8.66× | 4.91× |
| 32 000 × 768 | 15.32 | 124.73 | 8.14× | 5.55× |
| 128 000 × 768 | 30.74 | 440.69 | 14.34× | — |

The MLX path wins at every size tested from 250 rows upward, on both statistics except the
smallest case. The initial threshold of 4 096 was a placeholder that gave most of this away;
it is now 256.

### KNN `fit` is slower, and that is structural

0.17×–0.36×. mlxlearn's `fit` uploads the training matrix, its transpose, and its row norms
to the device; scikit-learn's brute-force `fit` stores a reference and does no work at all.

It is listed in `benchmarks/gate.py::ACCEPTED_REGRESSIONS` rather than left to slip through
the "no crossover measured, so not gated" branch — which is the most comfortable place for
a real regression to hide. The gate fails on any *unlisted* operation that regresses
significantly with no crossover.

The trade is worth taking whenever the model is queried at all. At 50 000 × 32: fit costs
5.05 ms against 0.88 ms, and the first query alone saves 30.8 ms.

## SVC — width decides, not sample count

The first grid was damning. Over 64 configurations — n ∈ {1 000, 2 500}, d ∈ {10, 20}, 2 and
4 classes, four kernels, C ∈ {1, 10} — **mlxlearn was slower in 58 of 64, worst 0.013×**. A
representative slice at 2 500 × 20, four classes:

| kernel | C | mlxlearn (s) | scikit-learn (s) | ratio |
|---|---|---|---|---|
| linear | 1 | 6.87 | 0.24 | 0.04× |
| linear | 10 | 42.82 | 0.68 | **0.02×** |
| rbf | 1 | 0.110 | 0.044 | 0.40× |
| rbf | 10 | 0.319 | 0.053 | 0.17× |
| poly | 10 | 0.394 | 0.060 | 0.15× |
| sigmoid | 1 | 0.054 | 0.054 | 1.00× |

But that grid was too narrow to see the real variable. Holding the sample count fixed and
widening the data:

| case | mlxlearn (s) | scikit-learn (s) | ratio |
|---|---|---|---|
| rbf, 4 000 × 32 | 0.16 | 0.07 | 0.47× |
| rbf, 4 000 × **256** | 0.20 | 0.57 | **2.93×** |
| linear, 4 000 × 32 | 45.56 | 1.59 | 0.03× |
| linear, 4 000 × 256 | 178.81 | 11.45 | 0.06× |

Same `n`, 6× the speedup swing. SMO is sequential — each iteration picks a working pair
from the current gradient and cannot start before the previous one finished — so MLX can
accelerate the kernel *rows* an iteration needs but not the iterations themselves, and
LIBSVM's inner loop over a cache-resident row is hard to beat per iteration. What decides
the outcome is therefore whether a kernel row is expensive enough to be worth dispatching,
and that is a question about the number of features.

The linear kernel loses at every width, because its rows are the cheapest of any kernel and
per-iteration dispatch is nearly all of the cost. It also degrades with C, which increases
the iteration count.

A sample-count threshold cannot express any of this — which is exactly why the original
`svc_min_samples = 2048`, with no work floor, routed every one of those measured losses onto
the MLX path.

**None of this is a correctness problem.** The primal objective agrees with LIBSVM to
5.2e-07 relative and the KKT conditions hold on an independent float64 check. mlxlearn
solves the right problem; it is slow getting there.

Consequences, applied rather than merely noted:

- The crossover is on **work**: `svc_min_samples = 2048` *and* `svc_min_work = 1_000_000`
  elements, both required. Checked against the measurements — 2 500 × 20 (5·10⁴),
  4 000 × 32 (1.3·10⁵) and 1 000 × 20 (2·10⁴) all fall below and go to scikit-learn;
  4 000 × 256 (1.0·10⁶) clears it and keeps the 2.93×.
- **`kernel="linear"` never takes the MLX path**, at any size.
- Under `patch_sklearn()` this is invisible: those problems go to scikit-learn. A directly
  imported `mlxlearn.svm.SVC` is Layer 1 and by design never dispatches to scikit-learn, so
  it will use its own CPU path and be slower. That is the documented Layer 1 contract, and
  it is a real reason to prefer the patched entry point for SVC in 0.1.0a2.

The useful region is narrower than the plan assumed: wide data on a non-linear kernel. The
fix that would widen it is not a threshold — it is making an iteration cheap enough to be
worth dispatching, which means keeping the gradient and the working-set selection resident
on the device across iterations instead of crossing the boundary each time. That is 0.2.x
work, and it is the same underlying problem as the logistic solver's below.

## Logistic regression — an honest disappointment

**mlxlearn's L-BFGS is 10–50× slower than scikit-learn's below 1 024 features.** Measured on
the MLX device, `fit` with `max_iter=200`:

| samples × features | mlxlearn (ms) | scikit-learn (ms) | speedup |
|---|---|---|---|
| 1 000 × 16 | 83.5 | 2.8 | 0.03× |
| 5 000 × 128 | 120.7 | 13.0 | 0.11× |
| 20 000 × 16 | 950.8 | 16.4 | 0.02× |
| 60 000 × 128 | 148.0 | 109.6 | 0.74× |
| 20 000 × 1 024 | 245.8 | 309.0 | 1.26× |
| 60 000 × 1 024 | 497.1 | 665.8 | 1.34× |

The solver is not doing extra work — its iteration counts match scikit-learn's almost
exactly (17 against 17, 20 against 19). It pays **per-iteration overhead**: at 20 000 × 16 an
L-BFGS iteration costs about 5.9 ms while the matmul inside it costs 163 µs. The line search
evaluates the objective several times per step, each evaluation forces a host
synchronization, and the fixed cost of crossing that boundary dominates until the arithmetic
per iteration becomes large.

So the crossover is set high, and both conditions must hold: MLX is used only for problems
that are tall *and* wide. Everything else goes to scikit-learn, which is genuinely faster.
Under `patch_sklearn()` the measured result is 0.79×–1.01×, both within noise, because
dispatch is doing its job.

**This is an optimization target, not a permanent property.** What would move it: evaluating
the Wolfe conditions on device so an accepted step costs one synchronization instead of
several, and batching the line-search candidate evaluations. Neither is in 0.1.0a2.

## PCA — the solver decides

PCA has four solvers and they cross over in different places, so one shape floor cannot be
right for all of them. The first floor (256 samples or 65 536 elements) sent every solver to
the GPU far too early; measured against scikit-learn it ran at 0.07×–0.9× below the
crossovers recorded here.

Measured on 2026-11-10 on an Apple M4 Max, 128 GiB, with Python 3.13.3, MLX 0.32.2,
scikit-learn 1.9.0, NumPy 2.5.2 and SciPy 1.18.1; `python -m benchmarks.pca_crossover` runs
the same panels. Other CPU-bound
jobs were running on the machine, so absolute times carry 10–20% noise; every figure is a
ratio of medians from runs that rotate the order of the routes. Each cell is the **worst
case** over what a caller can vary without changing the shape: dtype, data (centered, mean
at 100 standard deviations, or low-rank with a noise floor 10⁻³ as large), `n_components`,
`iterated_power`, and `fit` against `fit_transform`.

### `covariance_eigh`: 8·10⁶ elements, or 512 features

GPU route against scikit-learn, 416 configurations. Rows are features, columns are elements
(`n_samples × n_features`).

| float32 | 0.25M | 0.5M | 1M | 2M | 4M | 8M | 16M | 32M |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.43 | 0.58 | 0.94 | 1.13 | 1.36 | 1.59 | 1.75 | 2.01 |
| 16 | 0.28 | 0.43 | 0.62 | 0.73 | 0.99 | 1.21 | 1.44 | 1.72 |
| 32 | 0.26 | 0.43 | 0.55 | 0.72 | 0.89 | 1.12 | 1.40 | 1.65 |
| 64 | 0.37 | 0.47 | 0.55 | 0.64 | 1.01 | 1.12 | 1.37 | 1.58 |
| 128 | 0.54 | 0.60 | 0.68 | 0.75 | 0.95 | 1.12 | 1.32 | 1.55 |
| 256 | · | · | 0.95 | 1.01 | 1.08 | 1.09 | 1.16 | 1.30 |
| 512 | · | · | · | · | 1.37 | 1.33 | 1.31 | 1.24 |
| 1000 | · | · | · | · | · | · | 1.23 | 1.19 |

| float64 | 0.25M | 0.5M | 1M | 2M | 4M | 8M | 16M | 32M |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.33 | 0.48 | 0.66 | 0.84 | 0.94 | 1.21 | 1.42 | 1.60 |
| 16 | 0.28 | 0.45 | 0.62 | 0.71 | 0.95 | 1.25 | 1.50 | 1.68 |
| 32 | 0.31 | 0.44 | 0.64 | 0.72 | 0.93 | 1.20 | 1.65 | 2.01 |
| 64 | 0.41 | 0.57 | 0.66 | 0.86 | 1.07 | 1.43 | 1.80 | 2.05 |
| 128 | 0.55 | 0.69 | 0.81 | 0.92 | 1.19 | 1.56 | 1.87 | 2.12 |
| 256 | · | · | 1.04 | 1.15 | 1.33 | 1.70 | 1.89 | 2.17 |
| 512 | · | · | · | · | 1.43 | 1.53 | 1.70 | · |

The solver reads `X` twice and then works on a `d × d` matrix. On narrow data the host BLAS
is finished before a GPU round trip, about 0.6 ms, has paid for itself. From 512 features up
the eigensolve dominates both sides and the GPU Gram matrix wins at every row count for
which `svd_solver="auto"` picks this solver, so a fit is handed away only when it is small
on **both** counts.

### `randomized`: 8·10⁶ elements

The GPU iteration makes two round trips per power iteration, a fixed cost of several
milliseconds. Over 912 small configurations (300–16 000 samples, 32–4 096 features, 4 and 32
components) the worst cases in float32 were:

| samples \ features | 32 | 64 | 128 | 256 | 512 | 1 024 | 2 048 | 4 096 |
|---|---|---|---|---|---|---|---|---|
| 300 | 0.83 | 0.07 | 0.19 | 0.10 | 0.12 | 0.18 | 0.30 | 0.63 |
| 1 000 | 0.12 | 0.26 | 0.29 | 0.16 | 0.21 | 0.28 | 0.47 | 1.03 |
| 4 000 | 0.90 | 0.28 | 0.67 | 0.40 | 0.54 | 0.89 | 1.12 | 1.40 |
| 16 000 | 0.72 | 0.71 | 1.32 | 1.25 | 1.26 | 1.62 | 1.99 | · |

A second panel of 784 configurations at 8·10⁶ and 16·10⁶ elements (32–32 768 features; 4, 32
and 128 components; `iterated_power` of `"auto"`, 0 and 2) had the GPU ahead in 98.6% of
cases at both sizes, with a median of 2.1× and 2.5×. The 1.4% at or just under parity
(0.91–1.00) were all 128-component float32 fits, and both causes were then fixed rather than
routed around: the Gram route ran its host iterations in float64 through a Householder QR,
and low-rank data dropped the orthonormalization onto a host QR. Re-measured afterwards,
every previously marginal case is between 1.09× and 29×. The two panels above predate those
fixes, so they understate the GPU route; the floor was not lowered on the strength of that.

### `full` and `arpack`: always on the host

The GPU `full` route solves through a float32 Gram matrix and refines the part of the
spectrum below about 1.5 decades with a host SVD of the deflated tail. On well-conditioned
data that refinement never runs and the route is 3–5× faster. On data with a decaying
spectrum, which is most real data whenever every component is requested, the tail is most
of the matrix and its SVD costs as much as scikit-learn's whole fit:

| shape | float32, flat spectrum | float32, low-rank | float64, flat spectrum | float64, low-rank |
|---|---|---|---|---|
| 500 × 500 | 1.20 | 0.60 | 1.58 | 0.90 |
| 2 000 × 300 | 3.44 | 0.85 | 4.99 | 1.33 |
| 300 × 2 000 | 3.11 | 0.67 | 4.24 | 1.02 |
| 1 000 × 1 000 | 1.43 | 0.65 | 1.82 | 0.93 |
| 3 000 × 600 | 3.51 | 0.88 | 5.30 | 1.47 |
| 1 500 × 1 500 | 1.47 | 0.66 | 2.22 | 1.06 |

A route that loses on ordinary data in float32 at every measured size cannot be selected by
shape, so `full` stays on LAPACK. `arpack` iterates in SciPy on the host whichever backend
is chosen; its GPU route added a centered copy and measured within 4% of scikit-learn.
Both remain reachable with `force_backend="mlx"`. Making the tail refinement cheap, for
example by deflating recursively on the GPU, is what would change this.

### What the routing delivers

Speedup over scikit-learn for `PCA(n_components=32)` on isotropic Gaussian data, medians of
nine interleaved runs. "Before" is the estimator as first merged; "direct" is
`mlxlearn.decomposition.PCA` and "patched" is the class installed by `patch_sklearn()`.

| operation, shape, dtype | before | direct | patched | route |
|---|---|---|---|---|
| fit, 1 000 × 64, float64 | 0.48 | 0.99 | 0.93 | host / scikit-learn |
| fit, 10 000 × 64, float32 | 0.66 | 1.55 | 0.94 | host / scikit-learn |
| fit, 100 000 × 64, float32 | 1.32 | 2.31 | 0.97 | host / scikit-learn |
| fit, 1 000 × 1 024, float32 | 0.64 | 1.02 | 0.99 | host / scikit-learn |
| fit, 100 000 × 256, float64 | 1.25 | 2.34 | 2.32 | GPU |
| fit, 500 000 × 256, float64 | 1.45 | 4.52 | 5.78 | GPU |
| fit, 100 000 × 1 024, float64 | 5.14 | 8.86 | 11.6 | GPU |
| fit, 100 000 × 1 024, float32 | 5.74 | 5.46 | 7.80 | GPU |
| fit, 200 000 × 1 024, float64 | 6.14 | 12.2 | 14.3 | GPU |
| fit, 50 000 × 4 096, float64 | 4.80 | 9.30 | 11.1 | GPU |
| transform, 100 000 × 1 024, float64 | 0.65 | 1.67 | 1.70 | GPU |
| transform, 100 000 × 256, float64 | 0.72 | 1.44 | 1.47 | GPU |

On the GPU rows "direct" and "patched" run the same code, so the spread between those two
columns is the run-to-run noise of this machine. Across all 54 cases of the grid the direct
estimator went from 0.48×–8.4× to 0.99×–12.2×, and the patched one from 0.40×–9.8× to
0.92×–14.3×.

Consequences, applied rather than merely noted:

- Below the floors a patched estimator hands the whole fit to scikit-learn before
  validating, scanning or casting anything, and lands at 0.9×–1.0×: the remainder is the
  dispatch itself, about 0.04 ms.
- A directly imported `PCA` never dispatches to scikit-learn, so below the floors it runs
  its own host path. That path was brought to scikit-learn's speed or better (0.9×–2.5×,
  the low end on fits under 0.2 ms) instead of being left as a slow fallback: `covariance_eigh` without a centered copy, scikit-learn's
  LU normalizer for `randomized`, and one pass to validate input for `transform`.
- The float64 gains above the floors come from input handling, not arithmetic: one parallel
  narrowing cast replaced two scans of the float64 array and a serial cast, which had been
  more than half of a large fit.

## Reading these numbers honestly

- Below a crossover, mlxlearn being slower is the design, not a defect: patched dispatch
  never runs that path.
- "No significant regression" is the gate; "faster" is the goal. A case inside the measured
  noise passes.
- Nothing is averaged across operations. An average over fit and predict describes a
  workload nobody runs.
- `knn/fit` is the one knowingly-accepted regression, and it is listed by name.
