# LinearRegression: review of PR #4 and the certified host rewrite (2026-11-22)

PR #4 added `mlxlearn.linear_model.LinearRegression`: normal equations on a
float64 or binary32 Gram matrix, an eigendecomposition pseudo-inverse, and
`np.linalg.lstsq` whenever its certificate failed. This note records how it
compared with scikit-learn, what the review found, and the rewrite that
replaces its solver. Measurements are on the M5 Ultra (scikit-learn 1.9.1)
and the M1 Ultra (scikit-learn 1.8.0) unless marked otherwise.

## 1. PR #4 against scikit-learn

On 81 configurations (Gaussian sizes; offsets, condition numbers 1e6/1e10/1e14,
exact rank loss, one-hot columns collinear with the intercept, mixed scales,
integer data; weights, no intercept, ten targets, float32, wide and tiny fits),
PR #4 was correct everywhere and fast on well-conditioned data, but slower than
scikit-learn on every ill-conditioned or rank-deficient cell:

- Gaussian, offset and integer designs: 1.1-6.0x (100k x 1024: 279 vs 1676 ms).
- cond 1e6/1e10/1e14, rank deficiency, one-hot, mixed scales: 0.52-0.87x. Its
  certificate rejected them after a Gram product and an eigendecomposition, and
  it then ran `np.linalg.lstsq` on the full design.
- 10000 x 4096: 0.74x (eigendecomposition with vectors at d = 4096).
- The GPU route was never faster than the CPU route.
- Geometric mean 1.41x, minimum 0.52x.

## 2. Review findings

A multi-lens review confirmed 54 findings. The ones that shaped the rewrite:

- **Wrong cutoff on scikit-learn 1.7 and 1.8 (critical).** Stock passes `tol`
  to `lstsq` only from 1.9; 1.7 and 1.8 use `max(n_samples, n_features) * eps`
  of the working dtype. PR #4 always used `tol`, so `rank_`, `coef_` and the
  objective diverged on two of the three supported versions (its own wide test
  failed there).
- **A certificate that accepted `coef_ = 0`** when the residual dwarfed the
  signal (binary32 route).
- **Losses on common data** (above), from a conservative acceptance test plus a
  full fallback that redid all the work.
- **`singular_` only float32-accurate** on the binary32 routes.
- **Memory 4-6x scikit-learn's** for float32 input.
- API issues: float32-range rejection of valid float64 data, error types and
  messages for negative weights, MLX arrays crashing `fit`, `predict` refusing
  hand-set coefficients under `patch_sklearn()`, internal `RuntimeWarning`s
  leaking, a process-global MLX device switch, an unused residual pass, stale
  docs.

### Resolution of the review findings

| Findings | Resolution |
|---|---|
| Cutoff on 1.7/1.8 (C00, C02, C04, C09) | Version-aware cutoff; tests run on 1.7.2, 1.8.0 and 1.9.1 |
| `coef_ = 0` accepted (C01) | No binary32 route; refinement accepted only at the backward-stable floor; test with a residual 1e9x the signal |
| Slower than stock on common data (C03, C05, C06, C10, C11, C13, C15, C16) | Certified Gram/Ritz/tiled-QR paths; benchmarks now include hard data |
| Memory (C07, C12, C28, C31) | One float64 working design (stock's size for float64 input); no MLX copies |
| Speed left on the table (C08) | Eigenvalues only (no vectors) on the main path, Cholesky solve, a pivoted-Cholesky screen before the eigensolver |
| GPU never faster (C14) | GPU route removed; `device="gpu"` solves on the host with a diagnostic |
| `singular_` float32-accurate (C17, C19, C24, C32) | float64 eigenvalues, small values measured on the data; rtol 1e-8 against stock |
| float32-range rejection (C20, C44, C49, C51) | Removed; values beyond float32 range are fitted |
| Unused RSS pass, dead code (C21, C27, C36, C43) | Removed |
| Negative/zero `sample_weight` errors (C22, C53) | `_check_sample_weight` rebuilt from public API with each version's dtype rule and messages; zero-sum weights raise what each version raises |
| MLX arrays crash `fit` (C23) | Validated as stock validates them (to float32 NumPy); `check_float32_range` also accepts foreign arrays, which fixes the same crash in other patched estimators |
| `predict` with hand-set coefficients (C25) | Skips the sticky-backend bookkeeping when no backend was recorded |
| No bound on `n_features` (C26) | `linreg_max_features` and `linreg_min_aspect` in the patched crossover; eigensolver padding |
| Lists bypass the crossover (C29, C52) | Converted once, then routed by shape |
| Underflow, warnings (C33, C34) | Scaled norms, magnitude guards, `np.errstate`; tests at 1e-200..1e200 under warnings-as-errors |
| Process-global MLX device (C35) | No MLX in the solver |
| Test gaps (C18, C37-C40, C48) | Rank, `singular_`, `tol`, every path and the patched routing are tested |
| Docs (C30, C41, C42, C45-C47, C50) | Version listed as unreleased; `copy_X` docstring; float32 policy exception |

## 3. The rewrite

### Contract

- **The installed scikit-learn's problem, exactly.** Centre and weight as stock
  does; the cutoff is `tol` on 1.9 and `max(n_samples, n_features) * eps` of the
  working dtype (float32 for float32 input) on 1.7 and 1.8. `rank_` counts the
  singular values above it, `singular_` lists all of them and `coef_` is the
  truncated minimum-norm solution.
- **`rank_` always matches stock.** A rank decision is accepted only when it
  clears both a rigorous bound on mlxlearn's own rounding and scikit-learn's own
  rounding band around the cutoff (`gelsd`'s backward error on the centred
  design, which mlxlearn forms exactly as stock does). Anything closer runs
  stock's arithmetic.
- **Bit-identical fallback.** `_stock_lstsq` repeats stock's computation call
  for call (`numpy.average` on X in its own layout, in-place centring,
  `sqrt(sw)` scaling, LAPACK `gelsd`), minus stock's extra copies. It is used for
  tiny problems, near-square designs, `tol` outside (0, 1) (where `gelsd`
  silently substitutes machine epsilon) and decisions inside stock's rounding.
- **float64 on the certified paths**, including float32 input (widened exactly
  after centring in float32; `y` is cast to the working dtype first, as stock
  casts it). The bit-identical path runs stock's arithmetic in the working
  dtype. Values beyond float32
  range are fitted, not rejected. Errors match stock's on every version: `X`
  and `y` are validated as stock validates them, and the sample weights go
  through a public-API copy of `_check_sample_weight` with each version's
  dtype rule and messages (the project allows no private scikit-learn imports
  outside `patching/`).
- **No GPU.** Metal has no float64; every float32 or mixed-precision variant
  either loses accuracy (`singular_` off by 1e-5) or costs as much as the CPU
  float64 Gram it would replace. `device="gpu"` solves on the host and records a
  `host-solve` diagnostic, as PCA does for its host solvers.
- **No unsafe LAPACK.** Accelerate's blocked `dgeqrf` (what `numpy.linalg.qr` and
  `scipy.linalg.qr` call) overruns a heap buffer on some squarish shapes
  (600 x 600, 602 x 256-301; found with guard malloc). The solver never calls
  `numpy`/`scipy` QR, runs blocked `dgeqrf` only on leaves 4-1280 columns wide
  and at least 2048 rows taller than wide (the range guard-malloc sweeps found
  clean), and merges leaves with `dtpqrt`.
- **Defensive `gelsd`.** On the M5 Ultra (macOS 27.2) Accelerate's `dgelsd`
  silently returned garbage for one benchmark design (a rank of
  4.6e18, unsorted singular values, a residual worse than zero), so stock
  scikit-learn returned a wrong model. Every `gelsd` result is checked (rank in
  range, singular values finite and sorted; above 2^18 entries also the
  residuals of up to eight targets, on stock's centred matrix, allowing
  `gelsd`'s own backward error) and recomputed with `gelss` when it fails.

### Paths (`src/mlxlearn/_solvers/ols.py`)

1. **Prepare.** One threaded pass over fixed row chunks writes the centred,
   weighted, augmented design `A = [Xc | yc]` and accumulates `X^T X` and
   `X^T Y` chunk by chunk (one product of `A` with itself for few targets, two
   products for many). Chunking bounds the Gram rounding by
   `gamma(rows + chunks)` instead of `gamma(n)`, which is what makes the
   certificate tight. Designs narrower than 8 columns are stored zero-padded to
   8: Accelerate's GEMM and GEMV are several times slower below that width.
2. **Certified Gram path** (well-conditioned data). Eigenvalues of the feature
   Gram (padded off Accelerate's slow sizes at multiples of 512) with an error
   budget `E` give every singular value; Cholesky plus one or two refinement
   steps against `A` gives the coefficients. A refinement that stalls is
   accepted only at the accuracy a backward-stable solver attains,
   `u (kappa + kappa^2 ||r|| / (sigma_max ||b||))`.
3. **Ritz path** (exact rank loss, small singular values). Directions whose
   eigenvalues are too small to trust are measured on the data: `W = A V0` is
   formed directly and reduced by a tiled QR, so nothing is squared. Cauchy
   interlacing bounds their singular values from above, the Li-Li quadratic
   bound limits how far coupling to the rest of the spectrum can lower them, and
   refinement uses the Ritz values as a preconditioner. One-hot designs with the
   intercept, duplicate columns, structured rank loss and designs mixing column
   scales land here. It measures at most 30% of the directions, or 60% on long
   designs (`n >= 16 d`), where the measurement is cheap next to stock's QR; a
   pivoted-Cholesky screen counts the small directions first, so graded spectra
   (70% or more below the split) skip the eigensolver entirely. Between 1024
   and 2048 features it measures at most 32 directions; above 2048, none.
4. **Tiled QR path** (graded spectra, condition numbers to 1e14). `A` is reduced
   by Householder QR in row tiles on parallel threads (scipy's `dgeqrf`
   releases the GIL; `dtpqrt` and `dgeqrt` do not), then `gelsd` runs on the
   small triangular factor, so `rank_`, `singular_` and `coef_` carry `gelsd`'s
   semantics. Its answer is kept unless a singular value lies inside stock's
   rounding band of the cutoff.
5. **Wide designs** use the dual Gram `Xc Xc^T` with the same certification.
   A centred design has the constant direction `q` (almost) in its left null
   space; a Householder reflection deflates it. The design is then the deflated
   matrix plus a rank-one part `z = Xc^T q`, bounded rigorously by blocked,
   pairwise sums (tight enough for the 1.7/1.8 cutoff on most designs; small or
   heavily offset ones then fall back): keep decisions stand, every drop must clear `sqrt(s^2 + ||z||^2)`, and the deflated
   direction's singular value is reported as `||P z||` (the design's own, to
   first order) rather than as the bound. Anything else falls back to the
   bit-identical path.
6. **The bit-identical path** (`_stock_lstsq`) repeats stock's arithmetic for
   tiny and near-square designs, `tol` outside (0, 1), float16 and every
   decision only stock's rounding can make. It skips stock's extra copies and
   reuses offsets already computed. For tall designs with 16 or more features
   it writes the Fortran-ordered copy `gelsd` needs itself, in row blocks on
   parallel threads (centred in a C-ordered block, then copied across, which
   avoids cache aliasing when the column length is a power of two), and lets
   `gelsd` overwrite it: with scipy 1.18 this path is 1.1-1.2x faster than
   stock on large fits. Wide designs and the targets keep stock's own call
   pattern (scipy rounds an in-place wide copy differently in the last bit).

### Routing

Layer 1 (`mlxlearn.linear_model.LinearRegression`) always uses this solver
and selects its internal path from the data. Automatic fits under
`patch_sklearn()` use it for tall designs with at least 4,096 samples,
16 to 1,024 features, `n * d >= 2^18`, at least 16 samples per feature and no
more targets than features. The same work floor applies to wide float64
designs with 256 to 1,024 samples, at least three features per sample, at most
4,096 features and no more targets than samples. In those ranges, M4 Max
measurements recovered substantial speedups on common dense designs.
An ill-conditioned design that needs the exact fallback can still pay for a
failed attempt at rank certification.

The M5 Ultra and M1 Ultra results below used the original crossover of at
least 8,192 samples, `n * d >= 2^19` and at least 32 samples per feature
for tall designs. That crossover sent all wide fits to scikit-learn.

## 4. Results

Timings are medians of four rounds after a warm-up. Each fit is preceded by a
pause and a garbage collection, with the variants in rotated order, so BLAS
threads start cold, as in interleaved use. "Rewrite" is the direct class
(`mlxlearn.linear_model.LinearRegression`); "patched" is the class
`patch_sklearn()` installs. The grids are `benchmarks/linreg/bench_linreg.py`'s
`core` (Gaussian, 1,000 to 1,000,000 samples, 4 to 4,096 features), `hard`
(offsets, cond 1e6/1e10/1e14, exact rank loss, one-hot with the intercept, mixed
scales and integer data at five shapes) and `options` (weights, no intercept,
ten targets, float32, wide and tiny fits): 81 configurations.

### M5 Ultra (scikit-learn 1.9.1, scipy 1.18.1)

| | faster than stock | geomean | min |
|---|---:|---:|---:|
| PR #4 | 50 of 81 | 1.41x | 0.52x |
| Rewrite | 74 of 81 | 2.20x | 0.87x |
| Patched, the 52 fits it hands to mlxlearn | 51 of 52 | 2.78x | 0.87x |
| Patched, the 29 fits it hands to stock | — | 0.99x | 0.92x |

The rewrite beats PR #4 in 67 of the 81 cells. PR #4 lost on every
ill-conditioned or rank-deficient cell (0.52-0.87x); on those data kinds the
rewrite's geometric means are 1.4-2.9x, with the exceptions listed below.
Every `rank_` matches stock's, and every objective matches to 1e-6.

| group | cells | rewrite geomean | rewrite min | patched geomean | patched min | PR #4 geomean |
|---|---:|---:|---:|---:|---:|---:|
| kind: cond10 | 5 | 1.88x | 0.93x | 1.92x | 1.00x | 0.73x |
| kind: cond14 | 5 | 1.96x | 1.08x | 1.93x | 1.00x | 0.73x |
| kind: cond6 | 5 | 1.39x | 0.87x | 1.40x | 0.87x | 0.73x |
| kind: gauss | 17 | 2.20x | 1.10x | 1.80x | 0.97x | 1.89x |
| kind: integer | 5 | 3.35x | 2.50x | 2.71x | 1.00x | 2.67x |
| kind: mixed | 5 | 2.02x | 1.50x | 1.84x | 1.00x | 0.73x |
| kind: offset | 5 | 3.42x | 2.25x | 2.78x | 0.99x | 2.79x |
| kind: onehot | 4 | 2.94x | 2.06x | 2.94x | 2.08x | 0.71x |
| kind: rankdef | 5 | 2.08x | 1.46x | 1.94x | 1.00x | 0.72x |
| option: dtype=float32 | 3 | 2.93x | 2.26x | 2.90x | 2.28x | 2.34x |
| option: intercept=False | 7 | 2.51x | 1.02x | 1.79x | 0.95x | 2.18x |
| option: k=10 | 3 | 3.01x | 1.83x | 2.96x | 1.83x | 2.36x |
| option: sw=True | 3 | 3.24x | 2.38x | 3.24x | 2.38x | 2.51x |
| tiny | 5 | 1.00x | 0.98x | 0.97x | 0.92x | 1.15x |
| wide | 4 | 1.73x | 0.99x | 1.01x | 0.99x | 1.64x |

Below parity:

- **cond6, 1,000,000 x 32 (0.87x, also patched).** Condition number 1e6
  against 1.9's `tol=1e-6` puts the smallest singular value on the cutoff, so
  only stock's arithmetic can decide the rank; the fit pays for the Gram and
  the measurement that found the edge, then runs it. (It was 0.78x before the
  measurement learned to skip the tiled QR in that case.)
- **cond6 and cond10, 20,000 x 1,024 (0.93x direct; patched hands them to
  stock).** Graded spectra at 20 samples per feature: the screen refuses after
  the Gram, and the tiled QR needs more rows per feature.
- **Small fits patched mode hands to stock (0.92-0.99x).** A fixed dispatch
  cost (about 5 us hot, up to ~0.1 ms when caches are cold, as here) on fits
  of 1-5 ms. An earlier version also validated the parameters twice on this
  route (0.83x); stock's own validation now runs alone.

### M1 Ultra (scikit-learn 1.8.0, scipy 1.17.0)

| | faster than stock | geomean | min |
|---|---:|---:|---:|
| Rewrite | 74 of 81 | 2.55x | 0.76x |
| Patched, the 52 fits it hands to mlxlearn | 51 of 52 | 3.26x | 0.88x |
| Patched, the 29 fits it hands to stock | — | 1.01x | 0.68x |

| group | cells | rewrite geomean | rewrite min | patched geomean | patched min |
|---|---:|---:|---:|---:|---:|
| kind: cond10 | 5 | 2.32x | 1.02x | 2.30x | 1.00x |
| kind: cond14 | 5 | 1.80x | 0.89x | 1.76x | 0.88x |
| kind: cond6 | 5 | 2.16x | 0.92x | 2.16x | 0.99x |
| kind: gauss | 17 | 2.46x | 1.00x | 1.86x | 0.87x |
| kind: integer | 5 | 3.88x | 2.32x | 3.02x | 1.01x |
| kind: mixed | 5 | 2.23x | 0.93x | 2.30x | 0.99x |
| kind: offset | 5 | 4.19x | 2.44x | 3.11x | 1.00x |
| kind: onehot | 4 | 3.50x | 2.09x | 3.43x | 2.08x |
| kind: rankdef | 5 | 2.36x | 0.94x | 2.42x | 1.00x |
| option: dtype=float32 | 3 | 2.83x | 1.80x | 2.74x | 1.77x |
| option: intercept=False | 7 | 3.13x | 1.00x | 1.89x | 0.99x |
| option: k=10 | 3 | 3.61x | 2.18x | 3.55x | 2.08x |
| option: sw=True | 3 | 3.61x | 2.32x | 3.60x | 2.13x |
| tiny | 5 | 1.10x | 0.76x | 1.09x | 0.68x |
| wide | 4 | 2.01x | 1.01x | 1.02x | 1.01x |

Every `rank_` matches stock's, including the 1.8 cutoff `max(n, d) * eps`.
Below parity:

- **cond14, 100,000 x 256 (0.89x, patched 0.88x).** 1.8's cutoff (about
  2.2e-11 sigma_max here) falls inside this graded spectrum, and one singular
  value lies within stock's rounding band of it (the band is about 1.3% of the
  cutoff at 100,000 samples, against 2e-7 of 1.9's `tol=1e-6`). Only stock's
  arithmetic can decide that rank, so the fit pays for the Gram and the tiled
  QR that found it. On 1.7 and 1.8 this can happen on any graded design whose
  spectrum crosses the cutoff; on 1.9 only on knife-edge designs.
- **Graded and rank-deficient designs at 20,000 x 1,024 (0.92-0.94x direct;
  patched hands them to stock).** As on the M5, but with scipy 1.17 the
  fallback has no copy advantage to offset the attempt.
- **Fits patched mode hands straight to stock** read 0.68-1.12x (geomean
  1.01x), in both directions and on fits of 1-17 ms: run-to-run noise on the
  shared machine, since that route adds only the ~0.1 ms dispatch.


### Patched-dispatch crossover

The crossover times the slowest of four data kinds at each shape (cold, as
above). Shapes in bold are the ones the calibrated floors hand to mlxlearn.

**M5 Ultra (scikit-learn 1.9.1, scipy 1.18.1)**: slowest of Gaussian, one-hot, mixed scales and cond 1e10 per shape, speedup over stock (bold: routed to mlxlearn in patched mode)

| n \ d | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.00 | 1.03 | 0.97 | 0.98 | 1.00 |  |  |  |  |  |
| 128 | 1.03 | 0.97 | 0.95 | 0.86 | 1.04 | 0.83 |  |  |  |  |
| 256 | 0.95 | 1.00 | 0.97 | 1.02 | 0.96 | 0.83 | 0.78 |  |  |  |
| 512 | 0.93 | 0.95 | 0.95 | 1.02 | 0.91 | 0.95 | 0.98 | 0.95 |  |  |
| 1024 | 1.01 | 1.00 | 0.97 | 1.10 | 0.99 | 1.01 | 1.04 | 1.07 | 1.05 |  |
| 2048 | 0.99 | 1.01 | 0.94 | 0.97 | 1.06 | 1.05 | 1.15 | 1.10 | 1.09 | 0.99 |
| 4096 | 0.97 | 0.99 | 0.96 | 1.00 | 1.08 | 1.06 | 1.16 | 1.08 | 1.08 | 0.96 |
| 8192 | 1.01 | 0.99 | 1.00 | 1.30 | 1.35 | **1.54** | **1.64** | **1.52** | 1.03 | 1.05 |
| 16384 | 0.99 | 0.98 | 1.00 | 1.63 | **1.79** | **2.34** | **2.94** | **2.23** | **1.38** | 1.09 |
| 32768 | 0.99 | 0.82 | 0.92 | **1.90** | **2.25** | **2.70** | **3.07** | **2.83** | **2.79** | **2.15** |

**M1 Ultra (scikit-learn 1.8.0, scipy 1.17.0; shared machine)**: slowest of Gaussian, one-hot, mixed scales and cond 1e10 per shape, speedup over stock (bold: routed to mlxlearn in patched mode)

| n \ d | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.77 | 0.67 | 0.86 | 0.84 | 0.81 |  |  |  |  |  |
| 128 | 0.68 | 0.93 | 0.77 | 0.84 | 0.78 | 0.61 |  |  |  |  |
| 256 | 0.82 | 0.80 | 0.77 | 0.79 | 0.93 | 0.66 | 0.58 |  |  |  |
| 512 | 0.82 | 0.96 | 0.89 | 0.85 | 0.71 | 0.83 | 0.67 | 0.72 |  |  |
| 1024 | 0.67 | 0.79 | 0.85 | 0.78 | 0.82 | 0.92 | 0.95 | 0.79 | 0.81 |  |
| 2048 | 0.91 | 0.81 | 0.80 | 0.71 | 0.94 | 1.03 | 0.89 | 0.96 | 0.76 | 0.78 |
| 4096 | 0.90 | 0.89 | 0.85 | 0.78 | 0.87 | 0.95 | 0.97 | 0.96 | 0.97 | 0.70 |
| 8192 | 0.86 | 0.85 | 0.75 | 0.90 | 0.97 | **1.57** | **2.65** | **1.56** | 0.92 | 0.76 |
| 16384 | 0.80 | 0.93 | 0.86 | 1.14 | **1.28** | **1.79** | **2.72** | **2.08** | **1.00** | 0.94 |
| 32768 | 0.85 | 0.77 | 0.85 | **1.32** | **1.36** | **2.37** | **3.38** | **3.05** | **2.37** | **1.98** |

The M1 Ultra is a shared machine (other users' theorem provers kept 2-4 cores
busy), and its small cells are dominated by that load: the same fits, timed
hot, measured 1.00x on the exact tier. Its mid-size losses are partly real:
with scipy 1.17 the stock-arithmetic fallback has no copy advantage, so a
refused attempt costs 10-20% there.


## 5. Verification

- **Three scikit-learn versions.** `tests/parity/test_linear.py` and
  `tests/unit/test_ols.py` pass on 1.9.1 (286 tests), 1.8.0 and 1.7.2 (270
  passed, 16 that need 1.9's `tol` cutoff skipped); PR #4's own wide test
  failed on 1.7 and 1.8. They compare `rank_`, `singular_`
  (rtol 1e-8) and the minimum-norm coefficients with stock on Gaussian,
  offset, cond 1e6/1e10/1e14, rank-deficient, one-hot, mixed-scale, unscaled
  and timestamp designs, tall, wide and near-square, with weights, several
  targets, float32 and float16, `tol` from 0 to 5, extreme magnitudes (1e-200
  to 1e200 targets, 1e-160 to 1e150 designs) and the float32 cutoff of 1.7/1.8.
- **Paths are exercised and asserted.** Tests that target a solver path
  disable the small exact tier and assert the path the fit took
  (`ols-<method>` diagnostics): Gram, Ritz (tall, wide, past 1024 features),
  tiled QR and its `gelss` retry, dual (with and without extra null
  directions), near-square, many targets, the screens, the target-magnitude
  guard and the wide fail-fast. A dedicated test places a measured singular
  value at the cutoff, 3e-14 sigma_max above it (inside scikit-learn's
  rounding band but outside mlxlearn's own bound) and clear of it, on both
  cutoffs. Line coverage of `ols.py` over the suite leaves about 50 of 1,600
  lines unexecuted: LAPACK error codes, platform probes, input checks for
  direct calls and refusals that fall back to stock's arithmetic.
- **Bit identity.** The exact path is compared bit for bit with stock on 60
  random tall and wide shapes (several targets, weights, C and Fortran layouts,
  thread budgets), float32, float16 and single-feature inputs. Doing this found
  that scipy 1.18's batched `lstsq` rounds a wide matrix differently when given
  an in-place Fortran copy; wide designs therefore keep stock's call pattern.
- **Errors.** Input and weight validation is compared with stock on every
  version, directly and in patched mode: strings, complex, NaN/inf, 3-D, empty,
  ragged or None-holding lists, object and string targets holding None/'nan'/
  'inf', sparse `X` or `y`, `y` beyond float32 range for float32 `X`, the order
  of errors when several apply, and eleven forms of `sample_weight` (seven
  invalid, four valid).
- **Safety.** A test forbids `numpy.linalg.qr`/`scipy.linalg.qr` and pins blocked
  `dgeqrf` to the swept leaf shapes (and makes sure both a blocked and a short
  leaf run); a forked child must fit within two minutes; fits on hard designs
  run under warnings-as-errors, threaded and not.
- **The rest of the suite.** All repository tests pass on 1.9.1 (4834 passed;
  8 skipped for pandas and a C compiler), including scikit-learn's estimator
  checks for both layers; `tools/compliance.py` and `ruff check` pass.
- **LAPACK on macOS 27.2.** A guard-malloc sweep of every LAPACK call the
  solver makes (136 runs) was clean on the M5 Ultra, where even the shapes
  that overran on macOS 26.6 no longer do; guard malloc there does catch a
  deliberate overrun.
- **Adversarial review of the rewrite.** Five lenses (correctness, parity,
  robustness, performance, tests and docs), each finding re-run by a skeptic,
  confirmed 32 findings: 1 critical (inputs whose centring overflows reached
  `gelsd` unchecked and could hang), 20 major, 11 minor. All are resolved; the
  main ones are summarized below.

| Finding | Resolution |
|---|---|
| Rank could differ from stock on offset data: the band ignored stock's column-mean error | The solver now centres exactly as stock does (same `numpy.average` offsets, same in-place arithmetic), so it decides on stock's own matrix |
| Overflowing centring reached `gelsd` with `check_finite=False` (hang, garbage rank) | `check_finite` as stock; the exact path runs under `errstate` and raises stock's error |
| `y` beyond float32 for float32 `X` gave NaN coefficients; MLX float32 validated as float64; string `X` fitted | Validation follows stock step for step |
| Private `_check_sample_weight` import (CI gate) | Public-API replica with each version's messages |
| Many targets 3-6.5x slower (k x k block, identity multiply in refinement) | Separate `X^T X`; refinement `O(n d k)`; `X^T Y` deferred to the first refinement pass |
| Thread-pool race, warnings from workers, failures at interpreter shutdown | One never-replaced pool, `errstate` inside workers, serial fallback |
| `X^T y` underflow returned `coef_ = 0` labelled certified | Target-magnitude guard sends such fits to the exact path |
| Wide fits with an intercept discarded the dual Gram (1.3-1.55x slower) | Exact bound on the constant direction checked after the eigenvalues; tighter pairwise bound |
| Tests that never reached the certified paths; guard test that could not fail; untested padding, float32 cutoff, margins | Rewritten as listed above |
| LAPACK-safety docs overclaimed | Reworded to what the sweeps covered; leaves capped at the swept width |

### Second review round

After the calibration work a second adversarial review (the same five lenses,
each finding re-run by a skeptic who had to reproduce it) examined everything
changed since the first round. The bit-identity reviewer alone compared about
23,000 exact-path fits with stock (every layout, dtype, weight and target form,
thread budget and fallback route) and found two breaks. 37 findings survived
verification (31 confirmed, 6 partially; 6 rated major after verification, none
refuted). 34 are fixed and pinned by tests; three are kept by design and
documented below.

| Finding | Resolution |
|---|---|
| Wide fits with an intercept: drops of measured directions were certified on the deflated matrix, but the design can sit at sqrt(s^2 + ||z||^2) (adversarial designs got a different `rank_`) | Every drop clears that bound; the deflated direction's singular value is measured (`||P z||`) instead of reported as its bound, or the fit falls back |
| The `gelsd` plausibility check rejected correct answers: float16 products, offset cancellation, cutoffs that keep noise-level directions (float32 with tol 3e-8) | Residuals on stock's centred matrix in float64, with `gelsd`'s backward error in the slack; residual pass only above 2^18 entries |
| The `gelss` retry took cutoffs outside (0, 1) literally (rank 0 at tol >= 1, and for float16 with 1024+ samples on 1.7/1.8) | Given `gelsd`'s effective cutoff |
| Wide zero-weight rows reported as exact zeros where stock's rounding band reaches the cutoff | That band must clear the cutoff, or stock decides |
| float16 without an intercept: `coef_` rounded to float16 (stock keeps float32) | Cast only with an intercept, as `_set_intercept` does |
| Validation: NaN weights, object/string `y` holding None/'nan', list `X` with None, zero weights before `y` checks on 1.7/1.8, sparse subclasses | Stock's own checks in stock's order |
| The measurement-skip heuristic also skipped small but kept directions (2-2.5x lost on 1.7/1.8) | Skips only expensive measurements (more than 32 directions) |
| Deferring `X^T Y` cost certified many-target fits 12-19% | Reverted; the target-magnitude guard is gathered in the preparation pass |
| A refused submission at interpreter shutdown could run work twice (weighted rows scaled twice) | Each unit runs exactly once; weighted fills write final values |
| Coverage gaps: many targets, near-square route, QR retry, screens, band margins, truncation, wide fail-fast, misnamed tests; overclaiming docs | 40 new or rewritten tests; docs corrected |

Kept by design: fitted attributes stay float64 (the project's float32 policy;
in patched mode a fit routed to stock returns stock's dtypes); parameter
validation runs even under `skip_parameter_validation=True`, as in every
mlxlearn estimator (a project-wide follow-up); and the plausibility residual
reads at most eight targets (a failed `gelsd` corrupts the shared SVD, which
the rank and spectrum checks see; checking every target would cost up to 20%
of a many-target fit).

## 6. Limitations and follow-ups

- **Ranks only stock's rounding can decide.** When a singular value lies
  within scikit-learn's rounding band of the cutoff, only stock's arithmetic
  reproduces its rank, so the fit runs it after the attempt that found the
  edge: 0.87x on the knife-edge cond 1e6 design at 1,000,000 x 32 against
  1.9's `tol=1e-6`. On 1.7 and 1.8 the band is a few percent of the tiny
  cutoff `max(n, d) * eps` (1.3% at 100,000 samples), so graded designs whose
  spectrum crosses it hit this far more often (cond 1e14 at 100,000 x 256:
  0.88x on the M1 Ultra, also in patched mode).
- **Patched dispatch** adds about 5 us (hot) to 0.1 ms (cold caches) to the
  fits it hands to stock; on 1-2 ms fits that is up to 8%.
- **Graded spectra on short designs.** With 70% or more of the directions far
  below `sigma_max` (cond 1e10) and too few rows for the tiled QR, the fit runs
  stock's arithmetic after a screen; the threaded copy makes that path about
  stock's speed or faster, but it does not win much.
- **Wide designs** win on Gaussian and mixed data (2-4x) but not on one-hot or
  graded data, so patched mode leaves every wide fit to scikit-learn.
- **Large d.** Above about 2000 features the cubic eigensolver dominates.
  Accelerate's two-stage `dsyevd_2stage` is about 4x faster at d = 4096 but is
  not wrapped by SciPy; calling it through ctypes would need a self-tested
  binding.
- **Minimum-norm accuracy on the Ritz path.** The component of `coef_` along
  dropped directions is of order `kappa^2 u` there (`gelsd`: `kappa u`), far
  below the coefficients' sensitivity to rounding in the data.
- **Memory.** The working design is float64, like stock's centred copy for
  float64 input; for float32 input it is twice stock's float32 copy.
- **No GPU.** An exact-digit (Ozaki) Gram on Metal was prototyped by the
  design panel at 1.1-1.2x on the Gram phase only; not worth its complexity.
- **Platform bugs found on the way** (not reported upstream yet): Accelerate's
  blocked QR heap overrun (macOS 26.x; see above), Accelerate's `dgelsd`
  returning garbage silently on macOS 27.2 (stock scikit-learn is wrong
  there), and SciPy 1.18.1's `cholesky(lower=True)` returning a wrong factor
  for n > 32 (mlxlearn uses `cho_factor`).

## Appendix: M5 Ultra campaign (scikit-learn 1.9.1)

Speedups over stock; "path ok" means `rank_` equal to stock's and the objective within 1e-6.

| config | stock ms | PR 4 | rewrite | patched | path ok |
|---|---:|---:|---:|---:|---|
| cond10 10000x64 | 15.8 | 0.87x | 1.31x | 1.31x | yes |
| cond10 20000x1024 | 360.2 | 0.76x | 0.93x | 1.00x | yes |
| cond10 100000x64 | 88.9 | 0.76x | 2.59x | 2.60x | yes |
| cond10 100000x256 | 325.0 | 0.76x | 3.31x | 3.35x | yes |
| cond10 1000000x32 | 197.7 | 0.53x | 2.26x | 2.27x | yes |
| cond14 10000x64 | 16.0 | 0.86x | 1.34x | 1.36x | yes |
| cond14 20000x1024 | 376.6 | 0.76x | 1.08x | 1.00x | yes |
| cond14 100000x64 | 89.4 | 0.78x | 2.52x | 2.61x | yes |
| cond14 100000x256 | 323.1 | 0.76x | 3.24x | 3.29x | yes |
| cond14 1000000x32 | 196.4 | 0.52x | 2.45x | 2.26x | yes |
| cond6 10000x64 | 16.1 | 0.87x | 1.23x | 1.23x | yes |
| cond6 20000x1024 | 391.5 | 0.76x | 0.93x | 1.00x | yes |
| cond6 100000x64 | 90.3 | 0.78x | 2.04x | 1.97x | yes |
| cond6 100000x256 | 326.1 | 0.76x | 2.56x | 2.56x | yes |
| cond6 1000000x32 | 195.6 | 0.53x | 0.87x | 0.87x | yes |
| gauss 100x4 | 1.3 | 1.11x | 1.02x | 0.96x | yes |
| gauss 256x2048 | 41.5 | 2.23x | 2.54x | 1.02x | yes |
| gauss 256x2048 intercept=False | 37.6 | 2.41x | 2.52x | 0.95x | yes |
| gauss 500x16 | 1.4 | 1.15x | 0.98x | 0.92x | yes |
| gauss 1000x4 | 1.5 | 1.13x | 0.98x | 0.95x | yes |
| gauss 1000x16 | 1.8 | 1.23x | 1.03x | 1.01x | yes |
| gauss 1000x64 | 3.8 | 1.59x | 1.56x | 0.99x | yes |
| gauss 1000x256 | 21.1 | 1.47x | 1.73x | 1.00x | yes |
| gauss 1000x1000 | 88.4 | 1.01x | 1.10x | 1.00x | yes |
| gauss 1000x1000 intercept=False | 88.0 | 0.70x | 1.02x | 0.99x | yes |
| gauss 1000x1024 | 91.4 | 1.01x | 0.99x | 1.03x | yes |
| gauss 1000x4096 | 152.9 | 1.50x | 1.60x | 0.99x | yes |
| gauss 1000x4096 intercept=False | 152.7 | 1.56x | 1.70x | 1.00x | yes |
| gauss 2000x64 | 5.3 | 1.94x | 1.80x | 0.99x | yes |
| gauss 2000x8192 | 788.0 | 2.13x | 2.23x | 1.00x | yes |
| gauss 2000x8192 intercept=False | 783.0 | 2.14x | 2.32x | 1.00x | yes |
| gauss 10000x4 | 2.3 | 1.16x | 0.99x | 1.00x | yes |
| gauss 10000x16 | 4.0 | 1.74x | 1.48x | 0.98x | yes |
| gauss 10000x64 | 16.0 | 2.67x | 2.71x | 2.75x | yes |
| gauss 10000x64 dtype=float32 | 12.4 | 2.17x | 2.26x | 2.28x | yes |
| gauss 10000x64 intercept=False | 14.7 | 3.37x | 3.08x | 3.09x | yes |
| gauss 10000x64 k=10 | 18.1 | 2.06x | 1.83x | 1.83x | yes |
| gauss 10000x64 sw=True | 17.6 | 2.40x | 2.38x | 2.38x | yes |
| gauss 10000x256 | 60.8 | 1.81x | 2.28x | 2.28x | yes |
| gauss 10000x1024 | 236.3 | 1.84x | 2.05x | 1.00x | yes |
| gauss 10000x4096 | 3411.4 | 0.74x | 1.14x | 1.00x | yes |
| gauss 100000x4 | 11.0 | 1.10x | 1.37x | 1.03x | yes |
| gauss 100000x16 | 25.2 | 1.93x | 1.96x | 1.95x | yes |
| gauss 100000x64 | 90.3 | 2.52x | 3.64x | 3.62x | yes |
| gauss 100000x256 | 331.6 | 3.76x | 5.56x | 5.48x | yes |
| gauss 100000x256 dtype=float32 | 234.3 | 2.85x | 4.18x | 4.11x | yes |
| gauss 100000x256 intercept=False | 319.7 | 3.99x | 5.51x | 5.52x | yes |
| gauss 100000x256 k=10 | 336.5 | 3.49x | 5.58x | 5.43x | yes |
| gauss 100000x256 sw=True | 337.0 | 3.33x | 4.98x | 4.98x | yes |
| gauss 100000x1024 | 1701.1 | 6.00x | 7.63x | 7.67x | yes |
| gauss 1000000x4 | 57.9 | 1.15x | 1.20x | 0.97x | yes |
| gauss 1000000x16 | 118.5 | 2.00x | 2.01x | 1.99x | yes |
| gauss 1000000x32 dtype=float32 | 165.5 | 2.06x | 2.67x | 2.60x | yes |
| gauss 1000000x32 intercept=False | 189.6 | 3.03x | 3.70x | 3.61x | yes |
| gauss 1000000x32 k=10 | 240.1 | 1.84x | 2.68x | 2.62x | yes |
| gauss 1000000x32 sw=True | 216.1 | 1.99x | 2.88x | 2.87x | yes |
| gauss 1000000x64 | 483.6 | 3.41x | 5.84x | 5.90x | yes |
| integer 10000x64 | 16.0 | 2.71x | 2.50x | 2.48x | yes |
| integer 20000x1024 | 369.8 | 2.62x | 2.92x | 1.00x | yes |
| integer 100000x64 | 89.6 | 2.48x | 3.56x | 3.59x | yes |
| integer 100000x256 | 325.9 | 3.43x | 5.38x | 5.48x | yes |
| integer 1000000x32 | 202.4 | 2.24x | 3.04x | 2.99x | yes |
| mixed 10000x64 | 16.2 | 0.87x | 1.54x | 1.47x | yes |
| mixed 20000x1024 | 367.2 | 0.76x | 1.50x | 1.00x | yes |
| mixed 100000x64 | 90.2 | 0.78x | 2.11x | 2.08x | yes |
| mixed 100000x256 | 322.5 | 0.76x | 3.55x | 3.53x | yes |
| mixed 1000000x32 | 198.9 | 0.53x | 1.96x | 1.96x | yes |
| offset 10000x64 | 9.1 | 2.71x | 2.25x | 2.25x | yes |
| offset 20000x1024 | 399.6 | 2.92x | 3.24x | 0.99x | yes |
| offset 100000x64 | 88.5 | 2.50x | 3.45x | 3.83x | yes |
| offset 100000x256 | 324.7 | 3.79x | 5.62x | 5.54x | yes |
| offset 1000000x32 | 199.0 | 2.26x | 3.31x | 3.50x | yes |
| onehot 10000x64 | 16.4 | 0.83x | 2.06x | 2.08x | yes |
| onehot 100000x64 | 91.0 | 0.77x | 2.84x | 2.83x | yes |
| onehot 100000x256 | 323.9 | 0.76x | 5.00x | 5.05x | yes |
| onehot 1000000x32 | 200.2 | 0.53x | 2.56x | 2.51x | yes |
| rankdef 10000x64 | 15.1 | 0.84x | 1.52x | 1.50x | yes |
| rankdef 20000x1024 | 356.9 | 0.76x | 1.46x | 1.00x | yes |
| rankdef 100000x64 | 91.9 | 0.79x | 2.30x | 2.36x | yes |
| rankdef 100000x256 | 323.2 | 0.74x | 3.73x | 3.71x | yes |
| rankdef 1000000x32 | 196.0 | 0.53x | 2.07x | 2.10x | yes |

M5 Ultra: 81 cells; rewrite faster than stock in 74; geomean 2.20x (min 0.87x); patched geomean 1.92x (min 0.87x); PR 4 geomean 1.41x (min 0.52x)

## Appendix: M1 Ultra campaign (scikit-learn 1.8.0)

| config | stock ms | rewrite | patched | path ok |
|---|---:|---:|---:|---|
| cond10 10000x64 | 49.0 | 1.64x | 1.69x | yes |
| cond10 20000x1024 | 878.7 | 1.02x | 1.00x | yes |
| cond10 100000x64 | 166.5 | 3.38x | 3.29x | yes |
| cond10 100000x256 | 712.6 | 3.53x | 3.44x | yes |
| cond10 1000000x32 | 399.8 | 3.33x | 3.41x | yes |
| cond14 10000x64 | 48.1 | 1.69x | 1.64x | yes |
| cond14 20000x1024 | 899.2 | 1.05x | 1.01x | yes |
| cond14 100000x64 | 166.0 | 3.35x | 3.39x | yes |
| cond14 100000x256 | 716.6 | 0.89x | 0.88x | yes |
| cond14 1000000x32 | 402.9 | 3.53x | 3.39x | yes |
| cond6 10000x64 | 48.6 | 1.72x | 1.70x | yes |
| cond6 20000x1024 | 884.4 | 0.92x | 0.99x | yes |
| cond6 100000x64 | 165.2 | 2.92x | 2.86x | yes |
| cond6 100000x256 | 714.0 | 3.42x | 3.40x | yes |
| cond6 1000000x32 | 409.0 | 3.00x | 2.86x | yes |
| gauss 100x4 | 3.7 | 1.29x | 1.35x | yes |
| gauss 256x2048 | 72.2 | 2.30x | 1.05x | yes |
| gauss 256x2048 intercept=False | 68.8 | 2.83x | 0.99x | yes |
| gauss 500x16 | 3.3 | 0.76x | 0.98x | yes |
| gauss 1000x4 | 5.0 | 1.50x | 1.55x | yes |
| gauss 1000x16 | 3.8 | 1.12x | 0.68x | yes |
| gauss 1000x64 | 10.1 | 1.47x | 0.92x | yes |
| gauss 1000x256 | 37.7 | 1.71x | 1.04x | yes |
| gauss 1000x1000 | 182.5 | 1.00x | 1.00x | yes |
| gauss 1000x1000 intercept=False | 183.5 | 1.00x | 1.01x | yes |
| gauss 1000x1024 | 188.4 | 1.01x | 1.01x | yes |
| gauss 1000x4096 | 400.3 | 2.50x | 1.01x | yes |
| gauss 1000x4096 intercept=False | 402.0 | 2.86x | 1.03x | yes |
| gauss 2000x64 | 17.2 | 2.98x | 0.87x | yes |
| gauss 2000x8192 | 2888.6 | 2.83x | 1.01x | yes |
| gauss 2000x8192 intercept=False | 2859.8 | 3.12x | 1.00x | yes |
| gauss 10000x4 | 4.7 | 0.98x | 1.12x | yes |
| gauss 10000x16 | 10.6 | 1.44x | 0.92x | yes |
| gauss 10000x64 | 48.4 | 2.56x | 2.52x | yes |
| gauss 10000x64 dtype=float32 | 32.7 | 1.80x | 1.77x | yes |
| gauss 10000x64 intercept=False | 48.2 | 3.31x | 2.66x | yes |
| gauss 10000x64 k=10 | 51.1 | 2.18x | 2.08x | yes |
| gauss 10000x64 sw=True | 50.4 | 2.32x | 2.13x | yes |
| gauss 10000x256 | 125.7 | 2.95x | 2.96x | yes |
| gauss 10000x1024 | 534.8 | 2.65x | 1.00x | yes |
| gauss 10000x4096 | 20049.2 | 1.84x | 1.00x | yes |
| gauss 100000x4 | 17.6 | 1.00x | 0.91x | yes |
| gauss 100000x16 | 48.0 | 1.75x | 1.75x | yes |
| gauss 100000x64 | 163.0 | 5.05x | 4.39x | yes |
| gauss 100000x256 | 715.1 | 4.83x | 4.71x | yes |
| gauss 100000x256 dtype=float32 | 473.7 | 3.46x | 3.29x | yes |
| gauss 100000x256 intercept=False | 698.4 | 5.68x | 5.46x | yes |
| gauss 100000x256 k=10 | 726.1 | 5.30x | 5.39x | yes |
| gauss 100000x256 sw=True | 737.1 | 4.97x | 5.10x | yes |
| gauss 100000x1024 | 3275.4 | 6.10x | 6.60x | yes |
| gauss 1000000x4 | 77.5 | 1.52x | 1.03x | yes |
| gauss 1000000x16 | 202.2 | 3.17x | 3.13x | yes |
| gauss 1000000x32 dtype=float32 | 305.3 | 3.64x | 3.53x | yes |
| gauss 1000000x32 intercept=False | 365.8 | 6.15x | 5.73x | yes |
| gauss 1000000x32 k=10 | 507.1 | 4.07x | 3.99x | yes |
| gauss 1000000x32 sw=True | 436.4 | 4.08x | 4.28x | yes |
| gauss 1000000x64 | 1082.1 | 9.08x | 9.06x | yes |
| integer 10000x64 | 46.7 | 2.32x | 2.94x | yes |
| integer 20000x1024 | 895.6 | 4.18x | 1.01x | yes |
| integer 100000x64 | 164.3 | 4.36x | 4.11x | yes |
| integer 100000x256 | 715.4 | 5.05x | 4.87x | yes |
| integer 1000000x32 | 401.2 | 4.14x | 4.26x | yes |
| mixed 10000x64 | 50.0 | 1.87x | 1.90x | yes |
| mixed 20000x1024 | 889.9 | 0.93x | 0.99x | yes |
| mixed 100000x64 | 165.3 | 2.86x | 3.02x | yes |
| mixed 100000x256 | 716.6 | 3.49x | 3.53x | yes |
| mixed 1000000x32 | 408.8 | 3.17x | 3.18x | yes |
| offset 10000x64 | 46.2 | 2.44x | 2.42x | yes |
| offset 20000x1024 | 902.1 | 4.20x | 1.00x | yes |
| offset 100000x64 | 164.1 | 4.32x | 4.21x | yes |
| offset 100000x256 | 714.5 | 5.68x | 5.54x | yes |
| offset 1000000x32 | 408.9 | 5.14x | 5.13x | yes |
| onehot 10000x64 | 46.5 | 2.09x | 2.08x | yes |
| onehot 100000x64 | 166.0 | 4.01x | 3.71x | yes |
| onehot 100000x256 | 717.7 | 4.62x | 4.53x | yes |
| onehot 1000000x32 | 402.3 | 3.89x | 3.95x | yes |
| rankdef 10000x64 | 48.7 | 1.90x | 1.88x | yes |
| rankdef 20000x1024 | 888.5 | 0.94x | 1.00x | yes |
| rankdef 100000x64 | 169.9 | 3.25x | 3.44x | yes |
| rankdef 100000x256 | 713.6 | 3.72x | 3.78x | yes |
| rankdef 1000000x32 | 403.3 | 3.40x | 3.39x | yes |

M1 Ultra: 81 cells; rewrite faster than stock in 74; geomean 2.55x (min 0.76x); patched geomean 2.14x (min 0.68x)
