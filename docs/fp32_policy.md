# The float32 policy

MLX computes in float32 on the GPU. scikit-learn computes in float64. That difference is
not a detail to be papered over with a loose tolerance — it decides which parity claims
mlxlearn can honestly make, and which `check_estimator` checks can pass.

This document is the decision, written down before the estimators were implemented, so
that "we xfail that one" is a policy and not a reaction to a red test.

## 1. The dtype contract

| Boundary | dtype |
|---|---|
| Input features `X` | supported float64/int/bool inputs are converted to **float32** |
| KNN regression targets `y` | owned **float64** values, including values outside float32 range |
| Internal computation | primarily **float32**; KNN host search and regression target reductions use **float64** |
| Returned distances, probabilities, decision values | **float64**; GPU values are generally upcast, automatic CPU KNN distances are computed in float64 |
| Fitted attributes (`coef_`, `dual_coef_`, `intercept_`, `centroids`) | **float64**, upcast |
| Returned indices, labels | `int64` / the input label dtype, matching scikit-learn |

Returning float64 from a float32 computation is not a claim of float64 accuracy. It is
there because scikit-learn's ecosystem — metrics, meta-estimators, `Pipeline` — is written
for float64, and handing it float32 causes dtype churn far from mlxlearn. The ancestor
leaked float32 out of `kneighbors` and every downstream tolerance had to absorb it. The
accuracy actually available is described below, not by the output dtype.

Feature coordinates that are finite in float64 but overflow float32 (`|x| > ~3.4e38`) are **rejected**
with `UnsupportedInputError`, not silently converted to infinity.

## 2. Where float32 actually costs accuracy

### 2.1 Squared distances via the expanded Gram identity

‖q − t‖² = ‖q‖² + ‖t‖² − 2⟨q, t⟩ is exact in real arithmetic and hostile in float32. When
q and t are close, a small true value is obtained by subtracting two large nearly equal
ones, so the absolute error is on the order of `eps · ‖q‖²` ≈ `1.2e-7 · ‖q‖²`. For data
centered far from the origin this swamps the true separation entirely — differences
collapse to exactly equal, or to negative values that need clamping.

GPU searches with at most 32 features and 16 neighbors use direct squared
differences. Large batches with at most eight neighbors can first scan an
additional half-storage copy of the training data. Screening arithmetic remains
float32, and the original float32 training data remains owned by the index.
Measured quantization errors produce conservative distance lower bounds; packed
keys and omitted-candidate bounds permit shorter local lists. Retained candidates
use original-coordinate compensated accumulation. A strict bound comparison
certifies the selected neighbors; uncertain queries run the full original scan.
The [compact-search proof](knn-compact-certificate.md) covers rounding, ties,
underflow, overflow and the omitted-candidate bounds.

Other small searches scan every reference with compensated arithmetic. GPU
selection compares conservative rounding intervals first, then uses guarded
double-single arithmetic and exact integer comparison on original float32
coordinate bits for unresolved pairs. Native fit preserves these bits with an
owned integer-view copy, including subnormal coordinates.
Only truly equal original-coordinate distances use training-index order.
Reported squared distances remain float32; the public interface takes their
square roots in float64. Returned values that round equal therefore need not
appear in index order when their underlying distances differ.

For other GPU shapes, centered Gram values now serve only as a certified search
screen. Every reference has a conservative lower bound, including a bound for
points outside each retained list. Original-coordinate refinement must strictly
separate the kth winner from every omitted candidate; uncertain rows execute a
complete direct scan. Self identities are excluded before selection, including
single-row tile tails. This removes the former unverified `k+4` truncation that
could discard a copied training row before refinement.

Automatic NumPy KNN uses a separate host search on owned original float32
coordinates. When built, the optional C++ extension handles narrow CPU workloads
with at most 32 features, including single-query searches. Its native scan can
use float32 arithmetic when there are at most 64 features and every coordinate
is zero or has magnitude in `[2**-32, 2**40]`. These bounds keep nonzero arithmetic
intermediates normal and prevent overflow. Conservative distance intervals decide
separated comparisons; a bound on the first eight features can reject a reference
only when it is already strictly beyond the kth candidate's upper bound.
Selected distances are recomputed in float64. Other inputs use a direct float64
scan, with exact conversion of subnormal coordinates. Both scans use Neon on
arm64 where applicable.

Overlapping intervals trigger exact 640-bit integer keys derived from the
original float32 bits. Those keys cover squared distances across the entire
finite float32 range and supported feature count. When a selected candidate has
an exact key, its squared distance is rounded to float64 from that key before
the square root; this gives truly equal distances identical output values even
when separate SIMD reductions would round differently. Only truly equal
distances use training-index order, including ties at the kth boundary. For
suitable small batches with unequal feature ranges, an owned cached permutation
puts the largest-range coordinates first for the partial bound. The same column
permutation is applied to queries, without scaling or changing coordinate values
or the metric. The extension is compiled ahead of time, never during fit or query.
The neighbor estimator API imports this CPU dependency during module setup, so
native-library loading belongs to import cost rather than the first search.
Cold-start comparisons must include that setup cost when measuring total startup.
Automatic fits eligible for host search can leave GPU availability unknown until
a query actually needs it; those automatic host fits and queries do not initialize MLX.

The portable CPU implementation retains the same ordering contract when the
extension is unavailable or a workload uses its other certified paths. Direct
float64 distance tiles avoid float32 Gram cancellation. For suitable batches,
float64 Gram values propose candidates, with conservative lower bounds covering
every omitted reference. Candidate distances are recomputed from original
coordinates, and uncertain rows run a complete direct scan. Overlapping intervals
use exact integer comparisons. A bounded integer-data path computes the complete
Gram expression exactly in float32 when every intermediate is provably
representable; this remains the route for suitable 64-feature integer batches.
The optional native selector scans those certified Gram values and maintains
distance/index order directly, avoiding a full candidate sort. Its input checks
require finite integer coordinates and bounded intermediates; uncertified inputs
use the other search paths. The portable selector remains available when the
extension is missing.
Public host distances use float64 squared sums and square roots; host voting and
regression reduction also use float64. This improves arithmetic on float32-rounded
features without recovering precision discarded by input conversion.

The explicit MLX CPU path, including CPU execution through the `mlxlearn_device`
API, uses bounded direct float64 distance tiles with stable distance/index
sorting, ranking at float64 precision. Returned `mlxlearn_device` distances keep
their float32 dtype. Extremely large
finite coordinates can overflow a reported float32 squared distance even when
GPU ordering distinguishes the underlying distances. Device classifier voting
continues to consume squared distances to avoid an extra rounded square root.

See the [ordering and certificate argument](knn-precision-ordering.md) and
[precision-fix benchmark report](knn-precision-fix-2026-10-30.md).

### 2.2 Accumulation length

A `float32` sum over `n` terms carries relative error growing like `n · eps`. For
n = 10⁶ that is ~0.1 at the worst case and much smaller in practice, but it is why
mlxlearn does not claim `rtol=1e-7` parity on large reductions.

### 2.3 Iterative solvers

Gradient norms below ~`1e-6 · ‖initial gradient‖` are not resolvable in float32.
`tol` values smaller than that are honored as "run to the float32 floor" rather than
rejected, and `n_iter_` reports what actually happened. An estimator that stops early
because the objective stopped moving records it in diagnostics.

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
| Neighbor **sets** (per row) | `assert_array_equal` on the sorted row — exact | Which points are nearest is not a matter of opinion |
| Neighbor **index order** | exact where distances are strictly separated; set equality where they tie | See §3.1 — scikit-learn does not define an order for exact ties |
| Neighbor **distances** | `atol=1e-6, rtol=1e-6` | Refined float32, relative to the data scale |
| Classifier **predictions** | ≥ 99.5% label agreement | Labels are discrete; disagreements concentrate at the decision boundary. Measured: 0.9975 worst case over 40 randomized logistic configurations, 1.0 typical |
| Neighbor-vote **probabilities** | `atol=1e-5` | A sum of exact weights over an exact neighbor set |
| Linear-model **probabilities** | `atol=2e-3` | These inherit the coefficient error rather than being computed independently, so they cannot be tighter than the coefficients allow. Measured: 3e-4 typical, 1.34e-3 worst (multiclass with a 10× sample weight) at default `tol`; 6e-5–1.9e-4 at `tol=1e-6` |
| Linear **coefficients** | `atol=2e-3, rtol=5e-3` | Dominated by *where the solver stops*, not by arithmetic — and on a rank-deficient design only the penalty pins them down at all. Measured: 5e-4 typical, 1.15e-3 worst over the fixed grid |
| **Objective value** at the solution | `rtol=1e-5` | The objective is far better conditioned than its argument — this is the assertion that actually catches a wrong solver. Measured: worst relative difference 2.9e-7 across every non-separable logistic case |
| SVM **decision values** (PSD kernels) | `atol=2e-2, rtol=5e-3` | Dominated by the SMO stopping point, not by arithmetic: both solvers stop when the maximal KKT violation drops below the shared `tol` (default `1e-3`), and the dual variables still have freedom there. Measured worst 1.56e-2 on a ±4.5 value range (linear, C=10); rbf cases sit at 5e-7–7e-4 |
| SVM **decision values** (`kernel="sigmoid"`) | not asserted — see §3.2.1 | The sigmoid kernel is indefinite, so the dual is non-convex and two correct solvers reach different local optima. Measured 3.95e-1, and it does not shrink with `tol` |
| SVM **support-vector sets** | ≤ 5% symmetric difference | Measured: counts agree essentially exactly (1311 vs 1311, 827 vs 827; worst disagreement one point out of 548). Which points sit exactly on the margin is genuinely ill-conditioned, so exact set equality is not something either solver promises |
| SVM **KKT conditions** | must hold at mlxlearn's own solution | The assertion that does not appeal to scikit-learn at all, and the one that catches a solver agreeing with LIBSVM for the wrong reason |

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

GPU and automatic NumPy CPU searches order neighbors by original-coordinate
squared distance and then training index, resolving ambiguous comparisons
exactly. The explicit MLX CPU path uses direct float64 distance and stable index
order. Rounded output distances alone do not identify true ties. The independent
precision tests compare underlying distances, including membership at the kth
boundary.

scikit-learn's brute-force path partitions and then sorts the candidates with a
non-stable sort, so it has **no defined order among exactly equal distances**. Measured on
50 000 points at coordinate scale 10⁴ with 200 queries and k=7: neighbor sets agreed on
every row, distances agreed exactly (`max |Δ| = 0`), and 26 of 1400 cells differed — every
one of them a swap between two neighbors at an identical distance.

For that measured case, sets and distances agree and differences are confined
to ordering within ties. More generally, ties crossing the kth boundary can
also change which points sklearn selects. mlxlearn's precision checks require
canonical training-index order against an independent exact reference; sklearn
comparisons separately report valid alternate tie membership and numerical
failures rather than treating every index mismatch as a search error.

Reordering the same selected set preserves its mathematical vote. Choosing
different tied points at the kth boundary can change predictions and
probabilities when their targets differ; agreement with sklearn is not
guaranteed in that case.

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
**5.2e-07 relative**, worst 5.9e-06 at C = 10. Comparing decision values alone would report
a problem the objective shows does not exist — the same trap as comparing `coef_` for a
linear model.

### 3.2.1 The sigmoid kernel is indefinite, so its dual is not convex

`tanh(γ⟨x, y⟩ + r)` is not positive semi-definite for most parameter choices. The SVM dual
is then a **non-convex** quadratic program with multiple local optima, and two correct
solvers can land on genuinely different solutions — not different points on one flat
optimum, as with the rank-deficient linear case, but different optima.

Measured on a 5-class sigmoid problem, 1500 training rows: `max |Δ decision| = 3.95e-01`,
and it does **not** shrink with `tol` (3.95e-01 at `tol=1e-3`, 3.945e-01 at `tol=1e-7`).
The disagreement is isolated to one one-vs-one pair whose dual objective differs by
`rtol = 1.05e-04` — and mlxlearn's objective there is the **lower** of the two
(−413.690008 against −413.646611).

So for `kernel="sigmoid"` the decision-value tolerance in the table above does not apply and
is not asserted. What is asserted is that mlxlearn's dual objective is no worse, and that
predictions agree. LIBSVM makes the same caveat about indefinite kernels; mlxlearn states it
rather than tuning a tolerance until the test goes green.

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
the *input* collapses to three distinct values before a single distance is computed.
mlxlearn returns the correct answer for the data it was actually given — verified against a
float64 reference computed on the float32-rounded data — but it cannot return the answer
for the float64 data it never saw.

The rule this yields: mlxlearn's accuracy claims are always relative to the float32
representation of the input. When the input's own dynamic range exceeds float32, the answer
is scikit-learn, and `patch_sklearn()` users get it via `device="cpu"`.

## 4. `check_estimator` policy

mlxlearn runs scikit-learn's full estimator check suite against **both** layers — the
strict Layer 1 classes and the patched Layer 2 classes — because the two advertise
different tags and a tag that lies is worse than a missing feature.

**As of 0.1.0a2 the mlxlearn-specific xfail list is empty.** All ten suites pass.

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

**Rule C — deliberate determinism.** A check asserting that `random_state=None` produces
varying results is xfailed when `deterministic=True` is configured (§`rng.py`). It passes
under `deterministic=False`.

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
- **Loosen a tolerance to make a test pass.** Tolerances come from this document. A failure
  at these tolerances is a bug in mlxlearn.
- **Offer a float64 GPU path.** MLX does not have one. To preserve original float64
  feature coordinates, use stock scikit-learn. Automatic KNN host search refines
  float32-rounded features; `device="cpu"` also retains the float32 input contract.
