# SVC working sets and persistent prediction — October 31, 2026

The next substantial training improvement comes from changing how the optimizer
uses the data. A native hierarchical solver now makes many pair updates inside
a small working set, then applies their combined effect to the full gradient.
Against the [previous native implementation](svc-native-2026-10-31.md), complete
fits are **1.10–24.95× faster** across eight cases and two dataset seeds on an
Apple M4 Max. These are additional gains, measured directly against that native
baseline; they are not multiplied estimates from the earlier Python baseline.

## Source study

The requested reference repository delegates SVC training to an external native
backend. The useful transferable ideas are hierarchical working-set
decomposition, reuse of selected kernel rows, batched gradient updates, and a
persistent trained inference representation. The wrapper alone does not explain
the speedup; its native backend and the published algorithm do. The pinned
repositories, inspected files, and paper are listed in
[source study and provenance](provenance.md).

The implementation here is independent and uses the project's existing MLX and
native extension infrastructure. No reference-library source was incorporated
and no external solver dependency was added. It retains float32 kernels with
float64 optimization, rather than moving double-precision optimizer state onto
an unsuitable device or changing the numerical contract.

## Implementation

- [`_smo_block_native.cpp`](../src/mlxlearn/_solvers/_smo_block_native.cpp)
  selects a subset of violating variables, retains useful members from the
  preceding subset, and performs local SMO steps in C++. A tiled native pass
  combines their changes into the global gradient. Inner optimization contains
  no Python or NumPy calculations. Convergence requires reconstructing the full
  gradient and passing the original KKT tolerance. `max_iter` counts actual pair
  attempts, including stalled attempts, rather than outer batches.
- [`svc_fused.py`](../src/mlxlearn/_kernels/svc_fused.py) evaluates a batch of
  missing kernel rows in one Metal launch. Dot products or direct squared
  distances and the nonlinear transform remain fused. The CPU path batches
  rows through MLX matrix multiplication. Retained rows stay in native memory;
  only missing rows return to Python for evaluation.
- [`_smo_native.cpp`](../src/mlxlearn/_solvers/_smo_native.cpp) implements true
  active-set shrinking for eligible pair-solver fits. It restores the full
  gradient and active set before certifying convergence or returning an
  exhausted iterate. `shrinking=False` retains the unshrunk pair path. The
  portable solver remains available without native extensions.
- [`svc_predict.py`](../src/mlxlearn/_kernels/svc_predict.py) prepares owned
  support vectors, their centered device representation, and RBF norms during
  fit. Later predictions reuse them. A native, exact float32 comparison detects
  in-place support-vector or center mutations without allocating a converted
  matrix. Kernel parameters and backend are also checked; weights and bias are
  read on every call. Refit clears preparation, and pickle stores host snapshots
  while recreating device arrays lazily.

Working-set decomposition is selected for linear, RBF, and nonnegative-offset
polynomial subproblems with at least 256 samples. Width is capped at 64. The
bounded-cache path reserves room for both its owned rows and a temporary batch;
it never needs a full Gram matrix. At least four available working-set rows are
required, or 32 for linear kernels. Smaller or unsuitable problems retain the
pair solver. Sigmoid and negative-offset polynomial kernels retain the
conservative unshrunk pair path; these kernels need not be positive semidefinite.
Pair shrinking is enabled only on eligible full-Gram fits. CPU/GPU automatic
routing thresholds are unchanged.

The objective remains exact C-SVC, with no feature approximation or subsampling.
Working-set order changes the path to a solution, so identical iteration counts
and alphas are not expected. For example, seed-41 linear uses 90,146 pair updates
instead of 65,463, but spends much less time scanning all samples. Analytic
diagonals, box constraints, equality constraints, curvature handling, final rho,
and full-data KKT residuals retain their precision requirements.

NumPy still supplies validation, labels, buffer allocation and public arrays.
Optional portable fallbacks still perform NumPy calculations. Prepared inference
also adds storage: approximately eight bytes per support-vector coordinate for
the host float32 snapshot and resident float32 design, plus norms and center, on
top of the existing public model arrays. It caches no query results.

## Measurements

The baseline is a complete source snapshot taken after the first native SVC
optimization and before this work. The harness verifies its archive, extracted
sources, compiled extensions, and imported module locations. Each case and
implementation runs in a separate subprocess. Five fresh fits and their first
and repeated 257-row predictions supply the medians. An initial fit and both
predictions are recorded separately, outside those medians. Imports and fixture
generation are excluded; these are not process-startup or cold-cache timings.

All MLX work is synchronized. Inputs are float32. CPU/GPU selection is forced
identically for both mlxlearn versions, and CPU thread limits are one. The
third comparator is stock scikit-learn. This panel measures implementations,
not new automatic-routing crossovers.

| Case | Training shape | Kernel device | Previous native fit, seed 41 (ms) | Current fit, seed 41 (ms) | Speedup, seed 41 | Speedup, seed 42 |
|---|---:|---|---:|---:|---:|---:|
| RBF, narrow | 2,000 × 20 | CPU | 12.97 | 9.26 | 1.40× | 1.51× |
| RBF, wide | 4,000 × 256 | GPU | 57.67 | 35.71 | 1.62× | 1.68× |
| Linear | 1,200 × 20 | CPU | 254.95 | 24.31 | 10.49× | 12.58× |
| Linear, medium | 2,500 × 32 | CPU | 2,591.18 | 117.54 | 22.05× | 24.95× |
| Polynomial | 2,000 × 32 | CPU | 18.77 | 10.61 | 1.77× | 1.80× |
| Sigmoid | 2,000 × 32 | CPU | 9.11 | 8.21 | 1.11× | 1.10× |
| RBF, four classes | 1,600 × 32 | GPU | 16.72 | 14.03 | 1.19× | 1.22× |
| RBF, 0.05 MiB cache | 1,600 × 32 | GPU | 258.97 | 134.88 | 1.92× | 1.89× |

Every measured fit converged, and all held-out predictions matched both the
previous native version and sklearn in both seeds. Continuous one-vs-one margins
passed the existing absolute 0.02 and relative 0.005 tolerance against both
comparators. Raw records retain every timing sample, first calls, support counts,
actual iterations, KKT diagnostics, source hashes and extension hashes.

Prediction is still mixed. Final-source first-prediction ratios range from
**0.96× to 1.53×**, and repeat-prediction ratios from **0.92× to 1.54×**. Narrow
RBF regresses on both prediction measurements for seed 42. Initial runs also
showed a seed-41 multiclass repeat-prediction regression, at **0.78×**, despite
improved first prediction; it did not recur in the final run. Persistent
preparation does not imply a win for every shape, especially at sub-millisecond
durations. Both sets of raw measurements are retained below.

The additional work closes a substantial gap on the measured linear cases:
seed-41 linear now takes 24.31 ms versus sklearn's 39.32 ms; medium linear takes
117.54 ms versus 276.30 ms. Wide RBF takes 35.71 ms versus 403.48 ms. The tiny-cache
case still takes 134.88 ms versus 37.09 ms, despite its improvement. These results
do not establish universal speedups, hardware-wide crossovers, or improved
indefinite-kernel optimization. Sigmoid's small fit changes are not evidence of
working-set acceleration, because that path is deliberately excluded.

Environment: Apple M4 Max, macOS 26.6.2, Python 3.14.2, MLX 0.32.4,
NumPy 2.5.4, scikit-learn 1.9.1.

- [Seed 41](../benchmarks/results/svc-working-set-seed41-2026-10-31.json)
- [Seed 42](../benchmarks/results/svc-working-set-seed42-2026-10-31.json)
- [Medium linear, seed 41](../benchmarks/results/svc-working-set-linear-medium-seed41-2026-10-31.json)
- [Medium linear, seed 42](../benchmarks/results/svc-working-set-linear-medium-seed42-2026-10-31.json)
- [Baseline manifest](../benchmarks/reference/svc-pre-working-set-2026-10-31.manifest.json)

The initial runs measured the same numerical implementation before two final
edge-case fixes: preserving a scalar center's shape in the missing-native-helper
fallback, and rejecting fractional indices in the single-row helper. Neither
changes the numerical path used by these benchmark cases. Initial fit speedups
ranged from 1.08× to 26.41×. Their records remain available for variability checks:
[seed 41](../benchmarks/results/svc-working-set-initial-seed41-2026-10-31.json),
[seed 42](../benchmarks/results/svc-working-set-initial-seed42-2026-10-31.json),
[medium linear seed 41](../benchmarks/results/svc-working-set-initial-linear-medium-seed41-2026-10-31.json),
[medium linear seed 42](../benchmarks/results/svc-working-set-initial-linear-medium-seed42-2026-10-31.json).
The [source audit](../benchmarks/results/svc-working-set-audit-2026-10-31.json)
verifies that the final-run source and extension hashes match the finished tree.

## Validation and reproduction

**490 targeted tests passed.** Tests cover direct native solver contracts, independent objective/KKT checks,
zero and exhausted budgets, unequal and zero-width bounds, shrinking and full
gradient restoration, fused batch rows, bounded cache dispatch, mutation/pickle/
refit behavior, missing-extension fallbacks, weighted multiclass parity, patching,
and both SVC sklearn-compliance layers. The benchmark also checks convergence,
predictions and continuous margins for every timed fit.

Ruff and the repository's mechanical compliance gates pass. Source and wheel
builds succeed; the source archive contains all three SVC native implementations.
An extracted wheel imports all three extensions from its own package and passes
CPU/GPU linear/RBF fits, sklearn prediction agreement, and pickle round trips.
CI requires all three extensions to import after installation. Pandas-specific
and optional array-API compliance subchecks were skipped by sklearn in this
environment.

The OVR parity helper now accounts explicitly for integer votes changing when
an already-close continuous pair margin crosses zero. It first checks every OVO
margin with the unchanged numerical tolerance, then independently verifies each
model's OVR conversion, and compares the remaining continuous confidence term.
Regression fixtures reject large margin errors and unexplained vote offsets.
Prediction and support-set agreement requirements remain unchanged. This avoids
mistaking a legitimate discontinuity near zero for an order-one numerical error;
see the [precision policy](fp32_policy.md).

```bash
python setup.py build_ext --inplace
VECLIB_MAXIMUM_THREADS=1 python -m benchmarks.svc_working_set \
  --repeats 5 --seed 41 --baseline-dir /tmp/mlxlearn-svc-previous \
  --output benchmarks/results/svc-working-set-seed41-2026-10-31.json
VECLIB_MAXIMUM_THREADS=1 python -m benchmarks.svc_working_set \
  --repeats 5 --seed 42 --baseline-dir /tmp/mlxlearn-svc-previous \
  --output benchmarks/results/svc-working-set-seed42-2026-10-31.json
```

Run benchmarks alone. Add `--cases linear_medium` and a separate output path to
reproduce the larger linear case. The baseline is extracted and built outside
the repository; cached sources and binaries are reverified before reuse.

See the [test log](../benchmarks/results/svc-working-set-tests-2026-10-31.log)
and [package validation](../benchmarks/results/svc-working-set-package-2026-10-31.json).
