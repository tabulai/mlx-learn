# The float32 policy

MLX computes in float32 on the GPU. scikit-learn computes in float64. That difference is
not a detail to be papered over with a loose tolerance — it decides which parity claims
mlxlearn can honestly make, and which `check_estimator` checks can pass.

This document is the decision, written down before the estimators were implemented, so
that "we xfail that one" is a policy and not a reaction to a red test.

## 1. The dtype contract

| Boundary | dtype |
|---|---|
| Input `X`, `y` | anything scikit-learn accepts; float64/int/bool are converted |
| Computation on the MLX device, GPU or CPU | **float32**, always |
| Host-side arithmetic around it | **float64**: the L-BFGS bookkeeping, the finish of `LogisticRegression`'s objective on the GPU — the device's partial sums added, normalized and penalized (§2.2) — the SMO dual variables and gradients, the contraction of SVC decision values, the rbf entries SVC recomputes directly (§2.4), neighbor votes, and all of `LogisticRegression`'s inference, which is scikit-learn's own NumPy code on `coef_`. The one exception is SVC's one-vs-one kernel on the GPU, whose dual variables and gradients are double-float on the device — and verified, and if need be finished, in float64 on the host (§2.5) |
| Returned distances, probabilities, decision values | **float64** |
| Fitted attributes (`coef_`, `intercept_`, `dual_coef_`, `support_vectors_`) | **float64** |
| Returned indices, labels | `int64` / the input label dtype, matching scikit-learn |

Returning float64 from a float32 computation is not a claim of float64 accuracy. It is
there because scikit-learn's ecosystem — metrics, meta-estimators, `Pipeline` — is written
for float64, and handing it float32 causes dtype churn far from mlxlearn. The ancestor
leaked float32 out of `kneighbors` and every downstream tolerance had to absorb it. The
accuracy actually available is described below, not by the output dtype.

**Values float32 cannot hold are rejected, not converted.** A value that is finite in
float64 but overflows float32 (`|x| > ~3.4e38`) raises `UnsupportedInputError` with
reason `"float32-overflow"`; it is never silently turned into infinity. The check reads
the input the way scikit-learn is about to: float64 arrays, object arrays, and pandas
frames that mix float64 with bool or nullable columns (which NumPy would turn into object
arrays) are all range-checked. float32, integer and boolean data cannot reach the limit
and are not scanned.

What happens next depends on when it is found:

- **At `fit`** it is a capability mismatch. A directly imported estimator raises;
  under `patch_sklearn()` the fit falls back to scikit-learn's float64 path, subject to
  `fallback_policy`. A patched fit small enough to go to scikit-learn on size alone is
  decided from the input's shape, where it has one, before any scan — so it never
  reaches the check, and `fallback_policy="raise"` does not trip on it.
- **At inference** it is raised, in both layers. The backend is sticky: a model fitted on
  MLX has no scikit-learn state to hand the call to. The neighbor estimators and `SVC`
  raise for a query float32 cannot represent, and also for a query so far from the
  training data that the scale chosen at `fit` cannot hold it — for `SVC`, one whose inner
  product with a support vector float32 cannot hold (§2.4). `LogisticRegression`
  predicts with scikit-learn's float64 code and has no such limit.

**Values whose squares leave float32.** A value can fit in float32 while its square does
not: `|x| ≈ 1e19` squares to `1e38`, and a sum of a few of those is past float32's range;
`|x| ≈ 1e-20` squares into the subnormal range, which the GPU flushes to zero. A distance
or a Gram-form kernel on such data comes out infinite or zero, silently. mlxlearn scales
the data by an exact power of two before anything is squared — the neighbor index by the
training data's span around its mean, `SVC` by its kernel frame (§2.4), with the scale
folded back into `gamma`. Scaling by a power of two is exact, so data already in range is
unchanged bit for bit, and data around `1e19` and `1e-20` is served (both are tested).
What still does not fit raises `"float32-overflow"`: neighbor training data spanning more
than `FLOAT32_MAX / 8` from its mean, or spanning so far beyond the gaps between its rows
that no scale holds both (§2.1), and a linear-kernel `SVC` row whose squared norm
exceeds float32 — the linear kernel has no `gamma` to absorb a scale. mlxlearn also
defines a squared-range guard with a second reason slug, `"float32-underflow"`, for a
path that cannot rescale; every 0.1.x estimator rescales, so none of them raises it.

**Sparse pandas frames are sparse input.** A `DataFrame` whose *every* column has a
`SparseDtype` counts as sparse, by scikit-learn's own rule (`check_array` converts such
a frame to a sparse matrix): Layer 1 raises `SparseInputError`, and patched mode falls
back. A frame with only some sparse columns is dense input, as it is to scikit-learn.

**Devices.** `device` chooses where the work that dispatch sends to MLX runs: the GPU, or
MLX's CPU backend. Both run the same float32 code, so `device="cpu"` is not a route to
float64 — the float64 answer is stock scikit-learn's (§3.4). The two devices can round a
long sum in a different order, so results can differ in their last bits; for the neighbor
estimators that is visible only where a near-tie is decided (§3.1). `LogisticRegression`
differs by a little more: the GPU finishes its objective on the host in float64, where
MLX's CPU finishes it on the device in float32 (§2.2), so a fit whose stop rounding
decides can take a different number of iterations on each. On the pinned problems of
`tests/unit/test_lbfgs.py` both take the same, with objectives up to 4.9e-9 apart. A
model is served on the device it was fitted on, with one exception that re-decides
nothing: a GPU-fitted model loaded in a process with no usable GPU runs the same code on
MLX's CPU under `device="auto"` or `"cpu"`, recorded in diagnostics as
`"gpu-unavailable"`; under `device="gpu"` it raises `DeviceError`. Inference that is host
code on the fitted state — `LogisticRegression`'s float64 predictions, `radius_neighbors`
on an MLX-fitted `NearestNeighbors` — uses no device, and is unaffected.

## 2. Where float32 actually costs accuracy

### 2.1 Squared distances via the expanded Gram identity

‖q − t‖² = ‖q‖² + ‖t‖² − 2⟨q, t⟩ is exact in real arithmetic and hostile in float32. When
q and t are close, a small true value is obtained by subtracting two large nearly equal
ones, so the absolute error is on the order of `eps · ‖q‖²` ≈ `1.2e-7 · ‖q‖²`. For data
centered far from the origin this swamps the true separation entirely — differences
collapse to exactly equal, or to negative values that need clamping.

The identity is one matmul, so mlxlearn still uses it — but only to *select* candidates,
and then it proves the selection right:

1. **Translation and scaling.** Distances are translation-invariant, so the training mean
   is subtracted from the training and the query rows before the identity is applied. That
   moves ‖·‖² from "distance from the origin squared" down to "variance scale", where the
   identity is far better behaved. float64 input is translated by its float64 mean, with
   the rows and the mean each carried as a float32 value plus a float32 residual, so the
   translation is good to about 2⁻⁴⁸ of the coordinate and a large constant offset — Unix
   timestamps, projected coordinates — costs nothing: timestamps a few seconds apart
   certify on the first pass. (Rounding the mean to float32 first left up to 64 of offset
   near 1.7e9 in every row, and sent every such query to the second pass at 4–5× the
   time.) The translated rows are then scaled by a power of two, which is exact, so that
   data around 1e19, whose squares overflow float32, and data around 1e-20, whose squares
   underflow it (the GPU flushes those to zero), both come back into range. Only as far as
   that: ordinary data is not scaled at all, and a large span is scaled down just until its
   squares, with room for queries 2²⁰ times farther out, stop overflowing. Scaling down
   shrinks the gaps between neighbors too. Normalizing every span to 1, as mlxlearn once
   did, let a single unmasked fill value of 1e20 among unit-scale rows push their squared
   gaps below float32's normal range, and the GPU answered every query with zeros.
2. **Over-selection.** The identity selects `k + margin` candidates per query, the margin
   growing with `k` and with the number of features.
3. **Refinement.** The candidates' distances are recomputed directly as `Σ(t − q)²` from the
   *untranslated* coordinates. Translation rounds every coordinate at the precision of its
   distance from the mean, which for multi-modal or heavy-tailed data is far coarser than
   the input itself; subtracting the raw values rounds once, relative to the difference.
   float64 input keeps a float32 residual beside its float32 value, so the difference is
   taken to about 2⁻⁴⁸ of the coordinate. The refined values are what get ranked and
   reported.
4. **Certification.** For every query, a rigorous bound on the identity's float32 error —
   `γ_{d+2}·(‖q‖² + ‖t‖²)`, combined with the reverse triangle inequality so that one
   far-out training row cannot spoil it for everyone — must prove that no training row left
   out of the candidates can come within the k-th refined distance. The test is strict: a
   row that might merely *tie* the k-th is uncertain too, because §3.1 decides ties by
   index, and only a search that saw the tied rows knows their indices.
5. **Fallback.** A query the certificate rejects is searched again with at least 256 more
   candidates and, if that is rejected too, exactly: direct differences against every
   training row, on the device, in budgeted chunks, ranked by a composite
   `(distance, index)` key. A query goes to the exact search at once when a count on a
   sample of the training rows — one in each run of about 16 consecutive rows, at a place
   hashed from the run's position, so that rows repeating with a period do not line up
   with it — estimates that its Gram noise band holds at least twice as many rows as that
   wider search carries, too many for the wider search to certify it. The count is an
   estimate, not a proof, and a wrong call costs only time: whichever search answers a
   query computes a pair's distance the same way and ranks by the same key. Small-integer
   data — binary, one-hot, counts — makes the identity exact outright, and takes that
   exact search directly.

**How the candidates are selected.** Step 2 keeps each query row's first `k + margin`
training rows under `(Gram value, index)`. MLX's own operations do it in three passes over
each tile: the tile expression, which writes a second tile, then `argpartition`, which
MLX's GPU runs as a sort. On the GPU, from 2²² (query row, training row) pairs, a Metal
kernel of mlxlearn's own does it in one (`_kernels/_topk.py`): per-lane register lists
with threshold rejection and a SIMD-wide merge, the k-selection of Johnson, Douze and
Jégou ("Billion-scale similarity search with GPUs", 2019, arXiv:1702.08734). It computes
every tile value with the identity's float32 operations in MLX's order, fused
multiply-adds disabled, and keeps the same rows and the same largest kept value as MLX's
selection, bit for bit, so steps 3–5 see the same input: the certificate, the wider pass
and the exact search are unchanged, and so is every answer and every count in
diagnostics.

Exactness does not rest on that agreement. The certificate needs only that every training
row left out has a Gram value at least the largest kept one, which the kernel's ordered
lists guarantee by construction — and a row whose 32-long register lists might have
dropped a candidate (only above 32 candidates) is flagged and selected again by MLX.
What the bitwise agreement rests on is two behaviors of MLX 0.32.2 that MLX does not
promise: its GPU `argpartition` keeps the lowest index among equal values, as a stable
sort does, and its matmul gives a row the same bits whatever the number of rows (two or
more) and columns in the product — as long as the product is wider than it is deep. From
128 features MLX multiplies a product of at most `n_features` columns with a kernel whose
bits depend on its row count, and a one-row product of a few columns differs too; wider
than `n_features`, a row's bits were those of any wider product at every row count from 2
to 1 024 and every width measured, 64–4 096 features. So the kernel takes a first pass
only when every train tile of both paths, the ragged last one included, is wider than
`n_features`. A one-row product is a matrix-vector kernel that rounds differently, so the
kernel's blocks of query rows are unions of whole blocks of MLX's path, a one-row block
stays a block of its own, a row the kernel flags is selected again beside a second row,
on MLX's path's tiles, and the rows a block cannot certify go to the wider pass in the
groups MLX's blocks put them in. A block also makes choices for its rows together —
whether they carry a float64 residual, which the certificate then allows for, and, on
integer training data, whether they take the exact integer path — so a union of blocks
that would not all choose alike is searched as runs of blocks that do. On another MLX
release either behavior could change, and with it which of the rows tied at the last
carried Gram value is carried and which query rows take a wider pass: the time a query
takes and its diagnostics counts, not its answer, which the certificate and the
`(distance, index)` order fix. So the kernel runs only on the MLX releases on which that
identity was checked — 0.32.2, each patch release listed on its own, since which GEMM
kernel MLX picks for a shape can change in any — and only for work resolved to the GPU.
MLX's CPU device, MLX 0.31, and `MLXLEARN_METAL_KERNELS=0` select with MLX's own
operations, as before. A run-time self-test could not take the release list's place: both
behaviors depend on the shapes, and a test small enough to run on a first query exercises
only its own. What a first use does check, once per process, is the kernel itself — it
runs on a fixed input whose answer the host computes, and a kernel that does not build or
does not match is not used — because Apple GPUs other than the M4 Max it was measured on
are untested.

A fixed over-selection margin was not enough, and nothing checked it. Where the Gram noise
exceeds the gaps between neighbors — tight clusters, near-duplicates, one unscaled column
among unit-scale ones — it returned wrong neighbor sets without a word: 290 of 300 queries on
1000 clusters with a spread of 1e-3, 12% of queries on data with a single column spanning
0–1e4. The certificate turns each of those into a second pass rather than a wrong answer.

**What it costs.** On ordinary data — Gaussian, uniform, standardized features, normalized
embeddings, with or without a far-out training row — every query certifies on the first pass,
and the certificate is host arithmetic on one number per query. Where the old margin failed,
correctness costs the second pass. Measured on 50 000 training rows and 4096 queries (2048
for the last case), `k=10`, against 0.1.0a2's (wrong) query time on an M4 Max: 1.7× for
1000 clusters of spread 1e-3 at 32 features, 3.0× for `make_blobs(cluster_std=0.01)` at
16, 1.6× for the 0–1e4 column at 16, and 2.4× for two clusters at 0 and 1000 at 4
features. That last case needs the exact search for every query — direct differences
against every training row — so its cost grows with the number of features: 3.0× at 16,
5.3× at 64. So does a tie at the k-th distance, which the test above counts as uncertain:
a training row repeated more than about `k + 4` times sends every query whose k-th
neighbor is one of its copies to the wider pass, 1.3× the old time for 20 copies of each
row at 16 features. The old answer
there was right on the GPU only because MLX's GPU partition happens to keep the lowest
indices among equal values; on the CPU it was not. Binary and small-integer data got
*faster*, because it no longer goes through the identity at all. A far-out row is not
ordinary once it moves the mean itself — a fill value of 1e10 among a few thousand
unit-scale rows does: every translated norm then carries the shift, the noise band
outgrows the gaps, and every query takes the exact search. The count that decides whether
a query can skip the wider pass costs a device round trip of its own, which queries right
beside many copies of a training row pay for nothing when the wider pass then certifies
them: at 1 000 000 × 16, after an idle gap, about 1.25× the time for one query row and
1.03–1.05× for 256. When a call needed the wider pass or the exact search, its
diagnostics event says for how many rows, and for how many the wider pass was skipped.

Against stock scikit-learn rather than 0.1.0a2, that second pass is what decides: on such
data MLX lost at every shape the size rules send it, down to 0.22× (grid `knn-data` in
[`benchmarks.md`](benchmarks.md)), and since a batch waits for its slowest row, a few rows
in a hundred that need it — rows repeated twice, clusters of std 0.5, one column on 100
times the scale of the rest — were enough to lose mid-size batches (grid
`knn-structured`). So under `patch_sklearn()` a fit first searches 512 of its own rows
with the first pass alone and, if it cannot certify any one of them, hands the fit to
scikit-learn's float64 brute force (diagnostics reason `"data-probe"`), whose answer is
exact without a certificate. A directly imported estimator keeps the search, and its
correctness guarantee, on MLX.

What scaling cannot bring into range is refused (§1): refinement subtracts two translated
coordinates in float32, so training data spanning more than `FLOAT32_MAX / 8` (about
4.3e37) from its mean is a capability mismatch at `fit` — an `UnsupportedInputError`
directly imported, a fallback to scikit-learn under `patch_sklearn()` — and a query too far
from the training data for the scale chosen at `fit` is an error at inference. So is
training data whose span dwarfs the gaps between its rows — a fill value of 1e30, or the
NetCDF default of 9.97e36, left among unit-scale rows: no power of two keeps the span's
squares finite and the gaps' squares out of the underflow range (a scaled squared distance
below 2⁻⁹⁶) at once. When the span has to be scaled down, a sample of 64 training rows is
searched at `fit`, and if their own nearest neighbors come within 2¹⁰ of that limit, the
fit is refused the same way. A query that still lands nearer a training row than the limit
is an error at inference, never a distance of zero; an exact duplicate of a training row
is the one exception. Both devices refuse the same data, and within the limit both answer
it to float32 rounding.

**What remains.** Exact means exact with respect to the refined float32 distances. Two
training rows whose true distances from a query differ by less than float32 can resolve
(about 1e-7 relative) are ranked by their refined values rather than their true ones — the
storage limit of §3.4, not a selection error. `device="cpu"` runs this same float32
algorithm on MLX's CPU, so it is not a float64 escape hatch; the float64 answer is stock
scikit-learn's.

None of this applies to `radius_neighbors` and `radius_neighbors_graph`, which are
scikit-learn's own code run on the fitted state (§3.4).

### 2.2 Accumulation length

A `float32` sum over `n` terms carries relative error growing like `n · eps`. For
n = 10⁶ that is ~0.1 at the worst case and much smaller in practice, but it is why
mlxlearn does not claim `rtol=1e-7` parity on large reductions.

`LogisticRegression` keeps its sums over rows short where that costs nothing. On both
devices the loss is summed on the device into at most 1 024 partial sums, which the host
adds in float64. On the GPU the binary gradient is summed the same way — one float32
partial per block of rows, 16 to 256 blocks set by the row count alone, not by the
machine — and the host adds the partials, divides by the weight sum and adds the penalty
in float64. At 10⁶ rows that cut the gradient's error, against a float64 evaluation,
from up to 1.7e-7 to 1.1e-8 of its scale. The multinomial gradient on the GPU is one
float32 matrix product, blocked like the binary one only at one feature, and finished
on the host the same way. MLX's CPU device finishes both gradients on the device in
float32.

### 2.3 Iterative solvers

A float32-accumulated objective cannot resolve gradients much below `1e-6 · max|g₀|` — a
millionth of the largest gradient component at the starting point. `LogisticRegression`'s
L-BFGS evaluates the objective and its gradient from float32 terms on the device — on
the GPU finished in float64 on the host (§2.2); the terms stay float32, and so does that
floor — and makes every decision about steps and stopping in float64 on the host, so a
rounding wobble in one inner product cannot pass for convergence. Near the floor it does
two things:

- **Approximate Wolfe acceptance** (Hager & Zhang, 2005). When a step's decrease is smaller
  than the objective's own resolution (`1e-7` relative), the value cannot certify it, but
  the slope still can: a step that meets the strong-Wolfe curvature condition is accepted
  as a descent step.
- **A floor stop.** When the objective can no longer resolve the progress the gradient
  predicts, the run stops with the stop reason `"float32-floor"` instead of wandering at
  noise level until `max_iter`.

So a `tol` below what float32 can resolve — `tol=0`, say — is honored as "run to the
float32 floor" rather than rejected, and `n_iter_` reports the steps actually accepted. A
floor stop with `max|g| ≤ 1e-6 · max|g₀|` has done everything float32 allows: it emits no
`ConvergenceWarning`, and is recorded on the fit's own diagnostics event (`fit_event_`),
whose `extra` gains `stop_reason="float32-floor"`, `n_iter`, `n_fev`, `grad_max_abs`, `tol`
and `at_resolution_floor`, and whose `detail` explains the stop. A floor stop above that
level is a stall. It warns, as a `max_iter` or line-search-failure stop does, and those are
recorded the same way.

#### Preconditioned L-BFGS

L-BFGS starts every iteration from an initial inverse Hessian `γₖH₀`, and the two-loop
recursion needs nothing of `H₀` but symmetry and positive definiteness (Nocedal &
Wright, *Numerical Optimization*, §7.2; `γₖ = sₖᵀyₖ / yₖᵀH₀yₖ` from the newest pair).
`LogisticRegression` has two:

- **The diagonal** of the column scales, rounded to powers of four with a dead band, so
  that features in units of 1e8 move at all. Every fit had it, and every fit the gate
  below turns away still does.
- **The curvature at the zero start**, where the fit always begins and every predicted
  probability is 1/K. There the Hessian of the loss is its Fisher information, for the
  logistic loss also its Gauss-Newton matrix: `c·XᵀWX/Σw + l2·I` on the coefficients and
  `c` on the intercept, with `c = p(1 − p) = 1/4` for two classes and `(1/K)(1 − 1/K)`
  shared by every class's block for K — the matrix a Newton method for logistic
  regression works with (Lin, Weng & Keerthi, 2008), taken once and held fixed.
  `H₀` is its inverse, and an iteration without curvature pairs takes the Newton step
  `−H₀g` for it at a unit trial step. For two classes it bounds the curvature
  everywhere (`p(1 − p) ≤ 1/4`; Böhning, 1992), so that first step is conservative.

The second pays where features are correlated, which ten curvature pairs learn slowly,
and costs where classes are nearly separable: the fit ends confident, with a curvature
that is a small and differently shaped fraction of the zero-start one, and there it took
more evaluations than the diagonal and stopped at a higher objective. So a **gate**
decides, in two stages, on the host. The first reads at most 1 024 rows of the first 256
features, spread evenly over all the rows — over their cumulative weight when there are
sample weights, so that each drawn row stands for the same weight — and asks for
curvature the diagonal leaves to the pairs: a mean squared within-class correlation
above 0.01, a Cholesky-pivot condition estimate of the within-class correlations (64
columns, its sampling floor divided out) above 1.5, or a spread above 20 in the
curvature the coordinates show after the diagonal. Only then is the model built, from
`min(n, max(8 192, 4·d))` rows spread the same way, and asked how separable the classes
are in its own metric: the largest eigenvalue λ of `W⁻¹B`, between- over
within-class scatter along the most separating direction, read from the eigenvalues
`λ/(1 + λ)` of the class means' scatter against the model's curvature. At `λ ≥ 5` —
two classes 2√5 within-class deviations apart, overlapping by about 1% — the fit keeps
the diagonal. The penalty enters through the model, so a strongly regularized fit,
which cannot become confident, reads as less separable. Designs of one feature or more
than 1 024 keep the diagonal too.

**What a sample misses.** Rows taken at a stride from the start of the data never saw
its tail — up to half of it just below twice the sample's size — and on data ordered by
source or by label the gate read only the head: at 16 383 rows a label-sorted, nearly
separable problem read λ = 0.04 for 34 and was admitted, and AR(0.99) rows followed by
as many independent ones took 48 evaluations where in random order they took 9. Spread
over every row, both read the same either way. Weighted rows spread by row were missed
when few: ten rows of 60 000 carrying 90% of the weight left the model without most of
the curvature (164 evaluations to the diagonal's 30, and a stop 1.9e-4 above the
optimum); drawn by weight, they are 90% of the sample, and those classes — ten points
in sixteen dimensions — read as nearly separable. And no sample of 8 192 rows sees what
is rare in 60 000: a one-hot level of a handful of rows has a curvature of about `c·p`,
which the sample reads as `c·p²`, the centering's, or not at all, and the model's step
along it was orders of magnitude too long. With 100-200 Zipf levels beside AR(0.9)
numerics, 20 000-60 000 rows and `C` from 100 to `inf`, the dense path took 140-1 667
evaluations, and four fits of five ended on a line-search failure or a stall, up to 90%
above the optimum, where the diagonal path converged in 161-1 105. So one more pass over
every row computes each feature's weighted second moment, and a feature the sample reads
at less than half of it gets the whole design's curvature instead: the same five took
14-21 evaluations and converged at the optimum.

**Numerics.** The model is accumulated in float64 up to 256 features and in float32
above, from the float32 design the device holds; rows whose norm exceeds three times the
median count as if they had it (on heavy-tailed features a handful of rows would be the
whole model, and those rows carry almost no curvature near the optimum), in the sample
and in the pass over every row alike; a ridge of 1e-10 of each feature's own curvature
(1e-6, then 1e-4, for a float32 sample), and never less than of the curvatures'
geometric mean, keeps `C = inf` and rank-deficient designs factorizable — relative to
their mean, as it first was, one column in units of 1e8 swamped every other feature, the
separation read 0.005 for 1.05, and the fit stopped on `max_iter` 1.4% to 4.6% above
the optimum; the geometric mean keeps a constant column, which centering in float64
leaves a residue of 1e-13 collinear with the intercept, from a step of its own; a
feature whose Cholesky pivot is below 1e-6 of its variance (1e-3 in float32) is an exact
dependency — one-hot blocks beside an intercept — and gets the mean curvature, since
`1/ridge` there would amplify the gradient's float32 rounding (‖coef‖ = 1 260 against
16.7 on one-hot data at `C = inf` without it); and `H₀` is the explicit inverse of the
factorization. All of it runs on the host while the device computes the first
evaluation, which hides the gate's first stage on the fits patched dispatch routes but
not the model: its 2.2–7.7 ms at 60 000–200 000 rows outlast the evaluation, and a fit
the separation limit turns away pays them for nothing (below). The model's products and
factorizations run with the host's BLAS on one thread (on a stack whose NumPy and SciPy
each ship an OpenBLAS — NumPy 1.23 and SciPy 1.10 — a plan of 4 ms otherwise took a
median 0.28 s): a process-wide limit, so it is held only around the model, only where a
pool runs more than one thread, and one plan at a time. Another thread's own
threadpoolctl limit that begins inside it and ends after it still restores its single
thread when it ends — the gate, which every fit runs, holds none. The device graph and
every stopping rule are the diagonal path's, and a fit the gate turns away is the
diagonal path bit for bit.

**When the model is wrong anyway.** A run on it that would stop above the float32
floor without converging — a failed line search, or the stall that otherwise ends a run
on the floor — leaves it instead, with the curvature pairs gathered under it, and goes
on from the same point as the diagonal path starts. It falls back once, and only above
the floor: a run that converges, reaches the floor or uses up `max_iter` on the model is
unchanged. `fit_event_.extra` records `preconditioner` (`"dense"`, or `"diagonal"` for
the path the fit ended on), `preconditioner_reason` (`"correlated"`, `"fallback"`, or
why not: `"uncorrelated"`, `"separated"`, `"width"`, `"disabled"`, `"non-finite"`,
`"not-positive-definite"`, `"no-step"`) and `n_fev`.
`MLXLEARN_LBFGS_DENSE_PRECONDITIONER=0` (`Tuning.lbfgs_dense_preconditioner`) turns it
off, for tests and bisection.

**What it changes, measured.** The gate was calibrated on objective evaluations, which
do not depend on the machine: 582 fits over 23 data laws — isotropic, AR(0.5/0.9/0.99),
low-rank, one-hot (with rare levels too), column scales, Student-t₂ and Cauchy
features, overlapping, nearly separable and separable blobs, random and noisy labels,
raw and standardized digits (2-10 classes, 600-200 000 rows) and breast cancer — at
2-20 classes, 569-200 000 rows, 16-256 features, `C` from 1e-4 to `inf`, `tol` 1e-4
and 1e-6, sample and class weights and `fit_intercept=False`, each against a float64
Newton solution and scikit-learn 1.9. The dense path ran on 348 fits and took 13 694
evaluations where the diagonal took 59 313 (median 4.9× fewer; 4.75× on the 223 fits at
sizes patched dispatch routes); 12 of them fell back. It took more on 4: +12 at
`tol=1e-6` on column-scaled data, where that `tol` is below the float32 floor and both
runs ended on the floor, the dense one 1.1e-8 lower relative, and +1 to +4 on three fits
of 9-24 evaluations. Without the separation limit, ten-class digits at 20 000-200 000
rows took 65-116 evaluations where the diagonal took 35-42. The corrections to what a
sample misses, the ridge and the fallback cost the grid 367 evaluations against the model
as first integrated (13 327), which admitted the same 348 fits: they are for data the
grid did not cover. On 168 fits of such data — 100-200 Zipf one-hot levels at
20 000-200 000 rows, rows ordered by source, by label or with a differently scaled tail
at 12 000-65 535 rows, one column in units of 1e6-1e8, ten rows carrying 90% of the
weight, a constant column, each at `C` from 1 to `inf` — the dense path ran on 124 and
took 2 937 evaluations where the diagonal took 22 471, more on none, never stopped at a
higher objective, and converged wherever the diagonal did. The 44 it turned away are
separable classes sorted by label and the heavy weights, which make the weighted classes
nearly separable.

**In time.** Timed against the code before it on the GPU of an M4 Max with MLX 0.32.2,
under Python 3.13 with scikit-learn 1.9.0 and Python 3.10 with 1.7.2, with patched fits
at sizes patched dispatch routes (two classes at 200 000 × 16, 30 and 64; three, five
and ten at 60 000 × 16, 64 and 256; `max_iter=200`), each after a garbage collection and
a 10 ms idle gap, 18 calls each over every order of the two and stock scikit-learn,
significant at three standard errors. On 71 fits the gate admits — AR(0.9/0.99), five
and twenty latent factors, column scales, one-hot blocks, Zipf one-hot levels and breast
cancer — a fit took 1.29–4.87× less time (median 2.43×) under 1.9.0 and 1.19–4.68×
(median 2.32×) under 1.7.2, every one significant, with 1 019 evaluations where the
diagonal took 7 536, and ran 4.1–39.8× as fast as stock lbfgs (3.7–38.0× under 1.7.2)
where the diagonal ran 2.7–13.4×. A fit the first stage turns away pays 0.1–0.6 ms of
host work there, which the first evaluation hides: 33 fits of isotropic features,
separable blobs and random labels on isotropic features ran at 0.97–1.08× the code
before's speed, none significantly different. A fit the separation limit turns away has
built the model first, 2.2–7.7 ms of host work at these shapes — most of it the floor's
pass over every row and, at 256 features, the sample's product — which outlasts the
first evaluation: nearly separable blobs with AR(0.9) noise at 60 000–200 000 × 16–256
and raw ten-class digits at 60 000 and 200 000 × 64 ran at 0.79–0.92× the code before's
speed under 1.9.0 and 0.69–0.91× under 1.7.2, significantly slower in 9 of 10 fits on
each, with the same evaluations and coefficients — still 1.4–2.5× (blobs) and 8.7–16.9×
(digits) as fast as stock. A directly imported fit too short to hide the gate pays it:
two classes of isotropic features at 2 000–20 000 rows and 16–64 features, fits of
1.5–4.5 ms, ran at 0.88–0.96× the code before's speed, and the model wins where the gate
admits a fit that size (AR(0.9) features at 5 000 × 16, 1.56×). The CHANGELOG has the
figures in full.

The objective and the stopping test are unchanged, but on an ill-conditioned problem at
the default `tol` a different path stops at a different point. Of the 335 fits the dense
path ran on non-separable data, 302 stopped closer to the float64 optimum than the
diagonal path did (median relative gap 7.0e-8 against 2.1e-6). Of the 308 on which both
it and scikit-learn converged, 11 missed scikit-learn's objective by more than §3's
`rtol=1e-5`, against 25 of the same fits on the diagonal path: all at the default `tol`,
on problems whose three solvers stopped 1.5e-5 to 1.4e-3 above the optimum. At
`tol=1e-6` the 35 dense fits on which scikit-learn converged too met §3's objective,
probability and label tolerances, every one (`max|Δp|` 9.8e-4, labels 0.99997); on 7
more scikit-learn did not converge in 5 000 iterations. No fit the diagonal path
converged on stopped short on the dense path, and 35 went the other way. Of the 12 that
fell back — eight on Cauchy features, three on raw breast cancer, one on raw digits at
`C = inf` and `tol=1e-6` — two then converged; the diagonal path converges on none of
the other ten either, whose runs on both paths stall where float32 no longer resolves
their progress.

### 2.4 Kernel evaluation (`SVC`)

LIBSVM computes each kernel entry in double and caches it rounded to C `float`. mlxlearn
computes the entries in float32 on the device, inner products included, and keeps the
dual variables, gradients and working-set arithmetic in float64, as LIBSVM does — on
NumPy 1.x as well as 2.x. Decision values contract the float32 kernel entries against
float64 coefficients in float64, so the float32 error that reaches the output is the
kernel's. Most of it is the rounding of inner products — of `⟨a, b⟩`, or of the rbf squared
distance `‖a‖² + ‖b‖² − 2⟨a, b⟩` (§2.1) — measured at up to about `7u·(‖a‖² + ‖b‖²)` on
either device, from 2 to 4 096 features. That error grows with the rows' magnitude, not
with their difference, while the model learns from their differences. The frame, the rbf
refinement and the resolution check below keep the one below the other; the self-entries
keep the solver consistent with what remains.

**The compute frame.** Before anything is rounded, the data is mapped exactly, in float64,
into a *compute frame*. For the rbf kernel it is translated by the training mean first,
so a large offset never reaches float32. For every kernel but the linear one it is then
scaled by the power of two that puts the largest mapped entry in `[0.5, 1)`, and the scale
is folded back into `gamma` (`γ′ = γ / s²`), so the kernel is unchanged. The scale is
exact: data scaled by any power of two fits bit-identically, and poly, sigmoid and linear
results for data already in range are bit-identical to unscaled evaluation. The rbf
translation is not a no-op against 0.1.0a2, which cast first and subtracted a float32
mean: the mean is now subtracted before rounding, which changes the float32 inputs
slightly and can move where inside `tol` the solver stops — measured up to 6e-4 in the
decision values on `N(3, 1)` data. Data around `1e±19` is served,
and so is a `gamma` outside float32's own range (`gamma="scale"` on data around `1e-20`
resolves past float32's largest value, and folds back to order one).

**Rows close together, far from the mean (rbf).** Centering keeps the identity's error
small only near the training mean. Two rows close to each other but far out — tight
clusters far apart with `gamma` matched to the clusters, or an explicit `gamma` on
unscaled multi-modal data — got kernel entries wrong by up to 0.58, and fits returned
decision values 3–6 away from scikit-learn's, silently, under `patch_sklearn()` too. So
every entry whose identity error, in the exponent, can exceed `1e-4`, and whose value,
computed or true, can exceed `1e-8`, is recomputed on the host from the float32 rows, in
float64 — exact, for the values the device holds (§3.4), to float32's own rounding.
Distance is at least the difference of the norms, so only rows in a band of similar norms
can qualify: a lone far-out row costs its own band, not a pass over the Gram, and on
ordinary data the test never opens — on standardized data at `gamma="scale"` the bound
stays near `1e-5` up to 4 096 features. Measured: two clusters 2 000 apart, decision values
within 1.1e-3 of scikit-learn's (6.1 before); 40 clusters at 2 400 × 512 on the GPU, within
3.1e-4 (3.5 before), in 0.14 s against 0.05 s unrefined and 0.39 s for scikit-learn. Where
every entry inside a cluster needs it — two clusters of 2 000 or 5 000 rows each — the fit
took 2.6–3× as long as unrefined, 1.4–1.7× scikit-learn's time. The same test runs on
queries at inference.

**The raw inner product (linear, poly).** These kernels are not translation-invariant, so
the frame cannot centre them: their float32 error scales with the offset, while the
curvature the solver learns from, `K_ii + K_jj − 2K_ij`, scales with the spread. On
`make_classification · 0.01 + 20` a cubic kernel near 6.5e7 came out wrong by 40 against
curvatures of 3–100. mlxlearn compares that error at the largest row norm with the mean
squared distance between rows — and, for poly, `coef0`'s own rounding with `gamma` times
it — and past `1e-3` refuses the fit with `UnsupportedInputError`, reason
`"float32-cancellation"`, which patched mode hands to scikit-learn. The ratio overstates the
measured error per median curvature about threefold. On the linear kernel, where
scikit-learn still converges on such data, the decision values moved by 1.4e-3 of their
range at a ratio of 1.7e-4, by 1.1e-2 at 1.7e-3, and by 6e-2, with 98.8% of labels
agreeing, at 1.5e-2. Data whose every row float32 holds as the same nonzero row — often
rows that were not, rounded at a large offset — has no spread at all, leaves only rounding
to learn from, and is refused the same way.
`sigmoid` is not checked: on the same offsets it agreed with scikit-learn to 3e-4 or
better. The test is global, so structure that lives at a much finer scale than the data's
overall spread — tight clusters far from the origin — is not seen by it; standardize such
data before a poly kernel, or use scikit-learn.

**Self-entries.** The solver takes its curvature from the closed-form diagonal, in
float64, and its gradient updates from the rows. Every row carries that closed form,
rounded to float32, in its self-entry, so the two describe one quadratic. Left to the
float32 evaluation they disagreed — by 31 on that poly kernel, where each step then solved
one quadratic and applied another, and the fit circled its whole 10⁷-iteration budget:
376 s against scikit-learn's 0.001 s. On ordinary data the self-entries move by a rounding,
which can move where inside `tol` a fit stops: measured up to 2.3e-3 in the decision values
of in-range poly fits and 1.5e-2 in a linear one (whose dual is flat, §3.2), inside the §3
tolerance on either side.

What the frame cannot represent is a capability limit at `fit` — a `gamma` or `coef0` not
finite in float32 once folded (`UnsupportedParameterError`), a linear-kernel row whose
squared norm exceeds float32 (`UnsupportedInputError`) — and an error at inference, for a
query whose inner product with a support vector float32 cannot hold. A query far from the
training data is otherwise answered, as scikit-learn answers it: its rbf entries are
exactly 0 and its sigmoid entries ±1, so a CMIP `1e20` or netCDF `9.97e36` fill value in one
query row no longer fails the batch. A non-finite decision value is an error, never a
label.

A batch of a few query rows — at most 2²⁵ rows × support vectors × features — is evaluated
on the host in NumPy float32 rather than on the device, with the same operations in the
same order: the inner products, the rbf squared distance from the same float32 norms, the
clamp, poly's repeated squaring, and the same float64 recomputation of the rbf entries the
identity may get wrong. Only the BLAS that computes the inner products differs, so a row's
decision value moves between a small batch and a large one by float32 rounding — as it
already could between blockings — and never outside the §3 tolerance.

**Shrinking.** `shrinking=True`, scikit-learn's default, runs LIBSVM's shrinking
heuristic: samples pinned at a bound are set aside between gradient reconstructions. The
optimum and the stopping test are unchanged — the test is evaluated over every sample
before the solver may stop — but, as in LIBSVM, the path to the optimum can change on
every kernel, and with it where inside `tol` the solver stops. It moves by the same order
as LIBSVM's own shrinking on the same data, more as `C` grows, and most on the flat duals
of the polynomial and linear kernels. Measured through `SVC`, LIBSVM's movement on the
same data in brackets: at 2 000 × 10 the decision values moved by up to 4.5e-3 (3.8e-3)
for poly at `C=1` and 2.3e-2 (3.2e-2) at `C=10`, while rbf fits were often bit-identical
and moved by at most 6.2e-4 (7.5e-4) at `C=10` and `C=100`; at 2 000 × 20, linear fits
moved by up to 6.3e-3 (1.6e-2) at `C=1` and 1.0e-2 (2.0e-2) at `C=10`. The number of
support vectors changed by one in some of those fits, and a training point whose decision
value was that close to zero could change label, as LIBSVM's did. The gradient it rebuilds
is summed in a fixed order, so shrinking adds no dependence of its own on `cache_size` or
on the NumPy/BLAS build. The tolerances in §3 hold either way. The GPU's one-vs-one kernel,
and the host's finish of its pairs, do not shrink at all (§2.5); a pair the host loop
solves again from scratch there honors `shrinking`, as on the host path.

**`cache_size`.** When the row cache can hold a whole one-vs-one subproblem, its Gram is
built in one blocked matmul; otherwise rows are evaluated on demand, and the two can round
differently. On the GPU they produced bit-identical rows at narrow widths (all four
kernels at 2 000 × 10), a property of MLX's matmul rather than a promise; at a thousand
features and more MLX picks another matmul kernel for some shapes — the Gram's, or a batch
of 33 rows, the prefetch's own — and at 2 048 × 2 048 and 3 000 × 4 096 `cache_size=5`
moved the decision values by up to 9.2e-4. On MLX's CPU device they accumulate in
different orders at any width: at 2 000 × 10 the decision values moved by up to 2.2e-3 and
the iteration count changed. Every fit is KKT-optimal and inside the §3 tolerance.
scikit-learn is bit-identical across `cache_size`; pin it for bit-reproducible fits, on
either device. On the GPU's one-vs-one kernel path `cache_size` sets how many pairs' Grams
are held at once and nothing else: every `cache_size` that holds the largest pair's Gram
gives the same model, bit for bit (§2.5); one that does not sends the fit to the host loop.

### 2.5 SVC's one-vs-one pairs on the GPU: double-float state, verified on the host

A multiclass rbf `SVC` fit on the GPU solves its one-vs-one pairs concurrently, in a Metal
kernel of mlxlearn's own (`_kernels/_ovo.py`): one threadgroup per pair, the whole working
set at every step, and every pair at once. Metal has no float64, so the state the host
loop keeps in float64 — the dual variables and the gradient — is **double-float** on the
device: a float32 value plus a float32 residual, about 48 bits, added and multiplied
with error-free transformations (Dekker, 1971; Knuth, TAOCP vol. 2, §4.2.2) and `fma`
for a product's exact error. The **working-set key** — the second-order gain that ranks
the candidates for `j` — is **float32**. The kernel entries are the float32 Gram the host
loop would have built for the pair, from the same MLX operations, block for block —
§2.4's frame, rbf refinement and self-entries included — and the tests check it bit for
bit.

That changes the path, not the contract:

- **A different path to the same optimum.** Where two candidates' keys agree to
  float32's precision, or two gradients to double-float's, the kernel can pick a
  different working pair from the host loop's float64 choice. On the fixed problem the
  kernel is checked on at first use, the two moved together bit for bit for 17 steps and
  within 2e-14 for six more, then parted where two gradients stood 2e-16 apart; both
  stopped at the KKT test.
- **Every pair is verified on the host, in float64.** `smo.finish_c_svc` recomputes the
  gradient from the kernel's dual variables — in float64, from the pair's float32 Gram,
  in fixed chunks of rows — and applies the host loop's own stop: the full-set KKT
  violation below `tol`. A pair that fails it is **finished by the host loop**,
  warm-started where the kernel stopped and counting on from its iterations; a pair whose
  state went non-finite is solved again from scratch by the host loop. The bias is read
  off the float64 gradient by LIBSVM's rule. So what a fit returns is a solution the host
  loop would have accepted as its own; the kernel never decides convergence alone. The
  fit's diagnostics event counts the pairs each way (`ovo_kernel_pairs`,
  `ovo_host_finished`, `ovo_resolved_non_finite`, `ovo_resolved_no_free`, `ovo_waves`,
  `ovo_launches`).
- **A pair with no free dual variable is solved again on the host.** Its bias is not
  determined by the problem: with every dual variable at a bound, any value in an
  interval is optimal, and LIBSVM reads it off whichever sample its own rounding left a
  hair inside a bound. On a pair of a 9-class fit at C = 0.19 with balanced class weights,
  LIBSVM and the host loop took the bias, 0.46, from a sample 3.5e-15 below its box; the
  double-float state's rounding left a different sample 4e-14 below its box, and gave
  −0.38 — two KKT points, decision values 0.85 apart. The verification cannot tell them
  apart, so the rule is structural: a pair none of whose dual variables lies more than
  2⁻³² of its box inside it is solved again from scratch by the host loop, which rounds
  as LIBSVM does. In 200 random fits it took 90 of 5 794 pairs, in 6 fits — five at boxes
  of 0.1–0.4, two of them with balanced class weights, which it took every pair of, and
  one pair of a two-feature fit at C = 8.9 — and every bias that differed from the host
  loop's by more than 1e-2 was among them. A fit it takes every pair of costs the host
  loop's time plus the kernel's: random labels with balanced class weights at C = 0.1–0.2
  ran at 0.89–0.96× the host loop's speed (1.04–1.12× its time), still 2.3–2.7× stock
  scikit-learn's.
- **Deterministic.** No result is a sum across threads: every reduction in the kernel
  is a maximum, ties broken by the larger index, which no order of evaluation changes,
  and there are no atomics. A pair's result is a function of its data alone — not of
  the threadgroup width, the other pairs in its launch, or how its steps are split across
  launches — so repeated fits, fits from many threads at once, and fits at any
  `cache_size` that holds the largest pair's Gram are bit for bit the same. The host's
  float64 gradient depends on the BLAS build, as §2.4's does.
- **No shrinking in the kernel or the host's finish.** Both take every sample at every
  step, whatever `shrinking` says: shrinking is a speed heuristic, and the optimum and the
  full-set stopping test do not depend on it. A pair solved again from scratch — no free
  dual variable, or non-finite — is the host loop's, `shrinking` included, so that it is
  the host path's pair bit for bit; there `shrinking` can move the bias in its last
  bits, as on the host path.
- **`n_iter_`** counts, per pair, the kernel's steps plus the host finish's, and
  `max_iter` bounds their sum as it bounds the host loop; a pair solved again from scratch
  reports the host loop's own count. A pair that exhausts its budget on the device is
  reported as the host loop reports one — `fit_status_ = 1` and a `ConvergenceWarning` —
  without testing the last iterate, as the host loop does not test it.
- **A `tol` the state can resolve.** The kernel's own stop reads its double-float
  gradient, good to about 2⁻⁴⁸ of its scale, and every gradient a pair's SMO reaches is
  bounded by `1 + C_t · m`. Below what that resolves the kernel cannot see its violation
  fall under `tol`: at `tol=1e-14`, C = 10 and three classes of 200 rows, one pair stepped
  to its 10⁷-step budget in 58 s, and reported it exhausted, where the host loop converged
  in 8 083 steps. A fit whose `tol` is below 2⁻⁴⁰ · (1 + C_t · m), for the largest box
  and the largest pair, runs on the host loop.
- **Memory.** The pairs run in **waves** whose packed Grams fit together in the kernel
  cache `cache_size` already sizes (capped by `Tuning.smo_kernel_cache_max_bytes` and
  `smo_kernel_cache_fraction` of installed memory, as for the host loop's row cache), and
  each pair's Gram must fit it alone — exactly the pairs the host loop would have
  materialized — or the fit runs on the host loop. Everything else the build puts on the
  device stays within `Tuning.distance_block_bytes`, the budget `kernel_matrix`'s own
  blocks keep to: the training rows, uploaded class by class and kept there only while
  they take at most half of it, and every transient, the rows' squares included. A pair
  too large for that alone takes what `kernel_matrix` takes to build it on the host
  loop's path. So a fit's device memory is at most `cache_size` plus that budget (456 MiB
  at the defaults) unless one pair needs more. Measured in a fresh process per fit:
  264–353 MiB on the fits patched dispatch routes, at the default `cache_size=200`,
  against the host loop's 13–79 MiB, and 445–601 MiB at `cache_size=1000`; 447 MiB at
  20 000 × 4 096 with 50 classes, whose rows alone take 312 MiB; and at 16 000 × 32 with
  16 classes — 1.83 GiB of Grams in all — 326 MiB in ten waves, against 1.92 GiB holding
  every pair at once and 19 MiB on the host loop. Waves cost time once MLX's buffer cache
  is warm: the default took 1.15–1.33× the time of one wave after an idle gap
  (8 000 × 256 with 4 classes in two waves, 1.33×; 8 000 × 512 with 16 in three, 1.15×;
  16 000 × 32 in ten, 1.22×) and 1.08–1.18× back to back, while a process's first fit was
  faster in waves (107 against 124 ms at 16 000 × 32), the one large buffer costing more
  to allocate. The buffers go when the fit returns, to MLX's buffer cache, which the
  cache-release rule (`Tuning.mlx_cache_release_fraction`) empties after a fit that grew
  it past its share of memory, as after any other; every wave of a fit uses one buffer
  size, and a fit leaves that cache at 521–706 MiB on the routed fits, against the host
  loop's 228–430 MiB, and at 1.37 against 1.11 GiB at 16 000 × 32.

The kernel runs for three classes or more, the rbf kernel, a positive `max_iter`, work
resolved to the GPU, MLX 0.32 or later and `Tuning.metal_kernels`, pairs of at most 4 096
rows whose Grams fit the cache, boxes the double-float state holds (`C_t · m` at most
2⁴⁰, every `C_t` at least 2⁻¹⁰⁰), and a `tol` it resolves (at least 2⁻⁴⁰ · (1 + C_t · m));
anything else runs the host loop, pair after pair, bit for bit as before. Unlike §2.1's
kernel it needs no list of validated MLX releases: nothing it returns is trusted to match
MLX, so a release that changed its behavior could cost time — pairs finished on the host
— but never an answer.

Measured against scikit-learn 1.9.0 and 1.7.2 (the same on both), over 21 configurations
— 3 to 26 classes, C from 0.1 to 100, four `gamma` resolutions, random labels, low-rank
and imbalanced data, balanced and explicit class weights, sample weights with zeros,
`tol=1e-5`: decision values within 1.25e-3, labels and support-vector sets identical,
each pair's dual objective within 6.3e-7 relative, KKT margins within 6.7e-4. Over 200
random fits (`tests/parity/test_svc_ovo.py`, slow): within 1.9e-2 of the host loop's
decision values and, on every tenth, 1.8e-2 of scikit-learn's, KKT margins within 8.9e-4,
and no pair needed the host's finish. Over 300 more, drawn independently — 3 to 20
classes, 300 to 6 000 rows, 8 to 512 features, C from 0.1 to 100, five kinds of data,
class and sample weights — the same on both scikit-learn versions: no pair of 21 914
needed the host's finish or went non-finite, 94 in 14 fits had no free dual variable,
decision values were within 5.7e-3 of scikit-learn's, and at least 99.81% of labels the
same.

## 3. Tolerances the test suite uses

These are the numbers, and the reasoning for each. Tests do not invent their own.

**Precondition: both fits must have converged.** When either solver stops on `max_iter`
instead of its tolerance, the comparison is between two arbitrary intermediate iterates and
no tolerance can be stated for it. A randomized 48-configuration sweep at the default
`max_iter=100`, with feature scales varying over two orders of magnitude, produced apparent
violations up to `relJ = 1.8e-3` and `coef |Δ| = 0.145` — and on inspection *both* solvers
had hit the iteration limit and *both* had emitted `ConvergenceWarning`. Raising the budget
on the worst case: mlxlearn converged in **113** iterations, scikit-learn in **251**, and the
agreement fell inside the table below (`relJ = 2.3e-6`, `coef |Δ| = 1.5e-4`).

At the truncated budget mlxlearn's objective was the **lower** of the two. So the fair claim
is not "mlxlearn is within tolerance of scikit-learn at any budget" — it is "for converged
fits mlxlearn agrees to the tolerances below, and at equal truncated budgets its objective is
no worse." Both are tested.

| Quantity | Comparison against scikit-learn | Rationale |
|---|---|---|
| Neighbor **sets** (per row) | `assert_array_equal` on the sorted row — exact, except which of several rows tied at the k-th distance is returned | Which points are nearest is not a matter of opinion; which of *equally* near points makes the cut is — see §3.1 |
| Neighbor **index order** | exact where distances are strictly separated; set equality where they tie | See §3.1 — scikit-learn does not define an order for exact ties |
| Neighbor **distances** | `atol=1e-6, rtol=1e-6` | Refined float32, relative to the data scale |
| Classifier **predictions** | ≥ 99.5% label agreement | Labels are discrete; disagreements concentrate at the decision boundary. Measured: 0.9975 worst case over 40 randomized logistic configurations, 1.0 typical |
| Neighbor-vote **probabilities** | `atol=1e-5` | A sum of exact weights over an exact neighbor set |
| Linear-model **probabilities** | `atol=2e-3` on the fixed grid (n ≤ 5000, d ≤ 20) at default `tol`; on any converged problem at `tol=1e-6` | These inherit the coefficient error rather than being computed independently, so they cannot be tighter than the coefficients allow — and at default `tol` that error is where lbfgs stops, which a wide design sums over more features. Measured over the grid: 3e-4 typical, 1.34e-3 worst (multiclass with a 10× sample weight) at default `tol`; 6e-5–1.9e-4 at `tol=1e-6`. On 25 000 × 1 000, which 0.1.0a2's crossover sent to MLX: up to 4.9e-3 at default `tol`, where stock lbfgs and stock newton-cg differ from each other by as much, and 7.2e-5 at `tol=1e-6`. The objective assertion is what judges such a problem at default `tol` |
| Linear **coefficients** | `atol=2e-3, rtol=5e-3` | Dominated by *where the solver stops*, not by arithmetic — and on a rank-deficient design only the penalty pins them down at all. Measured: 5e-4 typical, 1.15e-3 worst over the fixed grid |
| **Objective value** at the solution | `rtol=1e-5` | The objective is far better conditioned than its argument — this is the assertion that actually catches a wrong solver. Measured: worst relative difference 2.9e-7 across every non-separable logistic case |
| SVM **decision values** (PSD kernels) | `atol=2e-2, rtol=5e-3` | Dominated by the SMO stopping point, not by arithmetic: both solvers stop when the maximal KKT violation drops below the shared `tol` (default `1e-3`), and the dual variables still have freedom there. Measured worst 1.56e-2 on a ±4.5 value range (linear, C=10); rbf cases sit at 5e-7–7e-4, and 1.25e-3 on the GPU one-vs-one kernel's grid (§2.5). Multiclass one-vs-rest values count votes, so they are compared on the rows where no one-vs-one value changes sign |
| SVM **decision values** (multiclass `kernel="sigmoid"`) | not asserted — see §3.2.1 | The sigmoid kernel is indefinite, so the dual is non-convex and two correct solvers reach different local optima. Measured 0.63 on the worst one-vs-one pair, and it does not shrink with `tol`. Binary sigmoid cases do meet the PSD tolerance above and are asserted at it |
| SVM **support-vector sets** | ≤ 5% symmetric difference | Measured: counts agree essentially exactly (1311 vs 1311, 827 vs 827; worst disagreement one point out of 548). Which points sit exactly on the margin is genuinely ill-conditioned, so exact set equality is not something either solver promises |
| SVM **KKT conditions** | margins within `2·tol` at mlxlearn's own solution | The assertion that does not appeal to scikit-learn at all, and the one that catches a solver agreeing with LIBSVM for the wrong reason. The stopping rule bounds every margin by `tol` in exact arithmetic; the second `tol` is the float32 kernel's. Measured worst 4.9e-4 (CPU device), 5.2e-4 (GPU), 7.9e-4 over a 72-configuration sweep; 6.7e-4 on the GPU one-vs-one kernel's grid and 8.9e-4 over its 200-fit soak (§2.5) |
| SVM **dual objective** (rbf, each one-vs-one pair) | `rtol=1e-5`, on the curated multiclass grid | Better conditioned than the decision values, as the objective is for the linear models. Measured worst 6.3e-7 on the GPU one-vs-one kernel's grid of 21 configurations (§2.5). Not asserted on random draws: on two-feature problems at a large box the host loop itself is up to 3.7e-5 from LIBSVM's, and 30% apart in a support-vector set, so the soak holds the kernel to the host loop's model instead |

**Coefficients are the weak assertion; objective value is the strong one.** For an
ill-conditioned problem two solvers can reach very different coefficients that are equally
optimal. A test that only compares `coef_` therefore fails for a correct implementation and
passes for a subtly wrong one. Every parity test that compares coefficients also compares
the objective.

**Objective comparisons are one-sided where mlxlearn can legitimately win.** On data with a
large constant offset mlxlearn reaches a *lower* objective than scikit-learn — 0.6104452
against 0.6104529 at an offset of 1e5, converging in 113 iterations against 175 — because
it centers the design and scikit-learn does not. A symmetric tolerance would fail mlxlearn
for being better, so those assertions are `j_mlxlearn ≤ j_sklearn · (1 + rtol)`.

### 3.1 Neighbor ties: mlxlearn is stricter than scikit-learn

mlxlearn returns the first `k` training rows under `(distance, training index)` — a total
order over the *whole* training set, not merely an ordering of whichever rows it returns.
Rows tied at the k-th distance are decided by index too, so `kneighbors(k)` is always a
prefix of `kneighbors(k + m)`, and no answer depends on block sizes, or on whether MLX
or mlxlearn's own kernel selected the candidates (§2.1). That includes
`kneighbors()` with no query, which is the same order with each row itself left out: a
row with more than `k` exact duplicates before it gets the first `k` of them. "Distance" is the
refined float32 value of §2.1. Where that value is exact — integer, binary and one-hot
features, halves and quarters — the answer is also identical on the GPU and the CPU;
elsewhere the two devices may round a long sum differently, and so break a near-tie
differently. (Until the certificate existed, none of this held on tie-heavy data: a tie
group larger than the over-selection margin was cut arbitrarily, and the cut moved with
the device, the block size and `k`.)

scikit-learn's brute-force path partitions and then sorts the candidates with a
non-stable sort, so it has **no defined order among exactly equal distances** — neither
within a returned row nor at the k-th boundary. Measured on 50 000 points at coordinate
scale 10⁴ with 200 queries and k=7: neighbor sets agreed on every row, distances agreed
exactly (`max |Δ| = 0`), and 26 of 1400 cells differed — every one of them a swap between two
neighbors at an identical distance.

So the parity assertion is: **distances are exact; sets are exact except for which of
several rows tied at the k-th distance is returned; index order is exact wherever distances
are strictly separated.** Asserting elementwise index equality would demand that mlxlearn
reproduce an ordering scikit-learn does not itself guarantee across releases.

Ties *within* the returned set are invisible downstream: predictions and probabilities
aggregate over the set. Ties *at the boundary* are not. scikit-learn keeps an arbitrary
subset of the tied rows and mlxlearn the lowest-indexed ones, so on tie-heavy data a vote
can legitimately differ between the two; mlxlearn's is the reproducible one. The parity
tests assert predictions elementwise on continuous data, where boundary ties do not occur.
A vote tied between classes goes to the class that sorts first in `classes_`, as it does
in scikit-learn.

### 3.2 SVC: the dual is flat, so compare the primal

For the rbf and sigmoid kernels, tightening `tol` from `1e-3` to `1e-5` shrinks the
decision-value gap against scikit-learn a hundredfold (8.97e-3 → 1.30e-5 and
6.33e-4 → 7.51e-6). That is the signature of a stopping-point difference: the underlying
solutions agree, the solvers just stopped at different moments.

For the **linear** kernel the gap does not shrink at all — 1.56e-2 at `tol=1e-3`, 1.71e-2 at
`tol=1e-5`. With n ≫ d the Gram matrix is rank-deficient, so the dual has a flat optimum:
many α vectors map to essentially the same primal solution, and the KKT rule both solvers
stop on measures *dual* violation.

The question "did mlxlearn solve the problem?" is therefore answered by the primal
objective `½‖w‖² + C·Σ hinge`, which is what both are minimizing. Measured: agreement to
**5.2e-07 relative**, worst 6.2e-06 at C = 10. Comparing decision values alone would report
a problem the objective shows does not exist — the same trap as comparing `coef_` for a
linear model.

### 3.2.1 The sigmoid kernel is indefinite, so its dual is not convex

`tanh(γ⟨x, y⟩ + r)` is not positive semi-definite for most parameter choices. The SVM dual
is then a **non-convex** quadratic program with multiple local optima, and two correct
solvers can land on genuinely different solutions — not different points on one flat
optimum, as with the rank-deficient linear case, but different optima.

Measured on the test's 5-class sigmoid problem (2000 rows, 15 features): the worst
one-vs-one pair's decision values differ by 0.63, and the gap does **not** shrink with
`tol`. The disagreement is isolated to that pair, whose dual objective differs by
`rtol = 2.3e-04` — and there mlxlearn's objective is the **higher** of the two
(−581.536821 against −581.671780). Neither solver is wrong: both solutions satisfy the KKT
conditions to within `tol`. On a 1500-row variant of the same data the order reverses, and
mlxlearn's is the lower. Which local optimum a solver reaches depends on the data.

So for multiclass `kernel="sigmoid"` the decision-value tolerance in the table above does
not apply and is not asserted. What is asserted is that mlxlearn's solution is a KKT point
of every pair (margins within `2·tol`), that each pair's dual objective is within `rtol=1e-3`
of LIBSVM's in either direction, and that predictions agree. The objective bound pins the
behavior on that dataset; for a non-convex problem it is not a guarantee either solver
could make in general. LIBSVM makes the same caveat about indefinite kernels; mlxlearn
states it rather than tuning a tolerance until the test goes green.

### 3.3 Where the objective assertion does not apply

On **perfectly separable** data at large `C` the minimizer is at infinity: the objective
approaches zero and both solvers stop on an absolute gradient tolerance somewhere along the
way. A relative comparison of two nearly-zero numbers is then meaningless. Measured at
n = 2000, d = 10, `C = 1e6`: scikit-learn reaches 2.45e-05, mlxlearn 2.73e-05 — a relative
difference of 0.114 from an absolute gap of 2.8e-06, with **100% label agreement** and
`predict_proba` differing by 1.8e-05. At `C = 1` on the same data mlxlearn's objective is
*lower* than scikit-learn's.

So: objective parity is asserted at `rtol=1e-5` on non-separable data, and on separable
data the assertions are the absolute gap, the predictions, and the probabilities. Both
solvers must terminate and produce finite coefficients — that part is never relaxed.

### 3.4 A limit that is about storage, not arithmetic

Some adversarial cases cannot be rescued by any kernel. Take 120 000 points spaced `1e-8`
apart and shift them to coordinate `1e4`: float32 has a spacing of about `1e-3` there, so
handed over as float32 the *input* has collapsed to three distinct values before a single
distance is computed. mlxlearn returns the correct answer for the data it was actually
given — verified against a float64 reference computed on the float32-rounded data — but it
cannot return the answer for values the input never held.

Handed over as float64, a constant offset like that one is no longer thrown away first.
Wherever the model is invariant to it, it is removed in float64, before anything is
rounded to float32:

- The **neighbor estimators** translate by the float64 training mean and keep a float32
  residual beside each float32 value, the mean's included, so the translation and the
  refined differences are taken to about 2⁻⁴⁸ of the coordinate (§2.1). 5 000 Unix
  timestamps within one hour, which collapse to 29 distinct values when cast first,
  return their exact neighbors. That is `kneighbors` and everything built on it; the
  radius methods are the exception below.
- **`LogisticRegression`** centers float64 input in float64 when it fits an intercept, and
  rounds only the centered values; the intercept is un-centered with exactly the mean that
  was subtracted. Without an intercept there is nothing to absorb the shift, so the design
  is used as given.
- **`SVC`** translates rbf input by the float64 training mean and scales every non-linear
  kernel's input by a power of two (§2.4).

What stays bounded by float32 is the *spread* of the data around that offset — 24 bits of
it — and any input that arrives as float32. The exception is `radius_neighbors` and
`radius_neighbors_graph` on an MLX-fitted `NearestNeighbors`, which are scikit-learn's own
code run on the fitted state: they compute against the index's float32 training copy —
the input rounded as given, neither translated nor refined by its residual — with the
query cast to float32 to match. That is what keeps them at stock speed, and it bounds
their precision by the data's raw magnitude, not its spread: Unix timestamps round to
128-second steps there and UTM northings to half-metre ones, so points within about that
step of the radius can land on the wrong side of it. Subtract such an offset before
fitting, or use stock scikit-learn. A sparse query that scikit-learn's brute force answers
on an MLX-fitted model — to the radius methods in both layers, and to `kneighbors` and the
predictions built on it under `patch_sklearn()` — is cast to float32 and computed against
the same copy. A float64 query float32 cannot hold is therefore an `UnsupportedInputError`,
as it is for `kneighbors`. A dense query that patched mode hands to scikit-learn's brute
force because it asks for more neighbors than the MLX search is kept for (more than 50)
does not use that copy: it runs on the float32 rows plus their float64 residuals, the
input to about 2⁻⁴⁸, translated by the training mean together with the query, which is
what keeps it at `kneighbors`' precision — Unix timestamps keep their seconds there, where
stock scikit-learn's own brute force, expanding squared distances around the origin,
returns them at distance 0 — at the cost of a float64 copy of the training rows, built on
the first such query.

The rule this yields: mlxlearn's accuracy claims are relative to the float32 representation
of the input after its exact float64 translation and scaling — for float32 input, the input
itself. When the data's own dynamic range exceeds that, the answer is scikit-learn itself —
stock, unpatched scikit-learn (import it before `patch_sklearn()`, or call
`unpatch_sklearn()`). `device="cpu"` is not a way to get it: it runs the same float32
algorithm on MLX's CPU backend, with the same rounding. Under `patch_sklearn()`, problems
below the measured crossover are handed to scikit-learn automatically, on either device.

## 4. `check_estimator` policy

mlxlearn runs scikit-learn's full estimator check suite against **both** layers — the
strict Layer 1 classes and the patched Layer 2 classes — because the two advertise
different tags and a tag that lies is worse than a missing feature.

**The mlxlearn-specific xfail list is empty.** Every suite passes: five estimators in each
of the two layers, each run twice — at the default configuration, and again with the MLX
GPU path forced, because the check data is smaller than every crossover and would
otherwise never meet MLX-fitted state behind the wide Layer 2 tags.
`check_param_validation`, which `check_estimator` does not run, is run separately for every
estimator in both layers, at the default configuration and with mlxlearn's own `fit` path
forced.

Two kinds of exemption exist and are kept strictly apart in
`tests/test_sklearn_compliance.py`:

**Inherited failures.** `check_sample_weight_equivalence_on_dense_data` and
`..._on_sparse_data` are declared by scikit-learn as expected failures for *its own* `SVC`
and `LogisticRegression` — "sample_weight is not equivalent to removing/repeating
samples." An mlxlearn estimator that subclasses one of them is not expected to pass a check
its base class does not. A test re-derives this list from scikit-learn's own source so the
exemption cannot outlive the behavior that justified it. mlxlearn's equivalent property is
asserted at float32 tolerances in `tests/parity/`.

**Rule-based xfails.** Checks scikit-learn passes and mlxlearn does not. Each must name a
rule below, and the list is asserted so a new one cannot be added silently to turn a red
test green.

**Rule A — float64 exactness.** A check that asserts float64-level agreement between two
computations of the same quantity may be xfailed. It is *not* xfailed if a
float32-appropriate tolerance would pass; in that case mlxlearn's implementation is fixed
instead.

**Rule B — dtype preservation.** A check asserting that float32 input yields float32 fitted
attributes is xfailed: mlxlearn deliberately returns float64 attributes (§1).

**Rule C — deliberate determinism (reserved).** No 0.1.x estimator draws random numbers,
so the `deterministic` and `random_state` options have no observable effect today and no
check is xfailed under this rule. It is reserved for the estimators that will: a check
asserting that `random_state=None` produces varying results may be xfailed under
`deterministic=True` (`_common/rng.py` resolves `None` to a fixed seed there), provided it
passes under `deterministic=False`.

**Never xfailed, under any circumstances:**

- `check_estimators_pickle` and every clone / `get_params` / `set_params` check
- `check_fit2d_predict1d`, `check_estimators_unfitted`, `n_features_in_` checks
- `check_classifiers_train`, `check_regressors_train` — an estimator that cannot learn is
  broken, not imprecise
- Any check about raising on invalid input

The live list lives in `tests/test_sklearn_compliance.py::EXPECTED_XFAILS`, with each entry
carrying its rule letter and a one-line justification. It is currently empty, and the goal
is to keep it that way.

## 5. What mlxlearn will not do

- **Silently downcast and claim parity.** Every deviation in this document is tested, not
  assumed.
- **Turn a value into infinity and carry on.** Values, squares and queries float32 cannot
  hold are rescaled exactly or refused (§1). The one infinity carried is exact where it
  lands: an `SVC` query so far out that its rbf squared distance overflows gets the kernel
  value `exp(−inf) = 0`, which is its value in float64 too, and only where the nearest
  support vector is past float32's `exp` (§2.4). A non-finite result is an error, never an
  answer.
- **Loosen a tolerance to make a test pass.** Tolerances come from this document. A failure
  at these tolerances, on the problems and at the `tol` §3 states them for, is a bug in
  mlxlearn.
- **Offer a float64 GPU path.** MLX does not have one, and mlxlearn's CPU path is the same
  float32 computation on another device. Requesting float64 accuracy means requesting
  scikit-learn: `patch_sklearn()` users get it for problems below the crossover, and
  otherwise by using stock scikit-learn for that estimator.
