# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`mlxlearn.linear_model.LinearRegression`** — dense ordinary least squares on
  MLX, including multi-output targets and `sample_weight`. The device computes
  the Gram products of the (centered, weighted) design; `positive=True` raises
  on Layer 1 and falls back under `patch_sklearn()`.

### Changed

- The native binary logistic step now runs one fused line-search pass for
  every feature count. The narrow line-search kernel and its unconditional
  backward pass are gone; designs of at most 128 features use two new
  kernels that batch four rows per SIMD group (eight lanes per row at up to
  32 features), and one shared geometry table serves the MLX step, the
  native command loop and the device bridge. Iteration counts, objective
  gaps and prediction agreement in the independent accuracy audit are
  unchanged; see the [report](docs/fused-narrow-kernel-2026-10-27.md).
- Every GPU logistic fit now prepares its input through the fused
  centering pass, including fits without an intercept, fewer than 512
  features or fewer than four million elements. The separate upload,
  centering and initialization kernels of the older route are deleted. The
  owned copy and the initial statistics are bitwise what those kernels
  produced; the per-column mean now comes from the statistics kernel of
  the former wide route, which agrees with the deleted centering kernel to
  one float32 ulp, so intercept fits of fewer than 512 features or four
  million elements can move at float32 rounding level with the same
  iteration counts. Intercept fits of 512 to 4,096 features below four
  million elements (reachable only with `force_backend="mlx"`) now take the
  native command loop, the scale-skew threshold and the optional Hessian
  sketch like larger ones. The borrowed input stays owned until the fit has
  drained the device, and a fit without an intercept on float64 input now
  holds one owned float32 copy of the design during `fit` where it used to
  solve on a view of its private conversion. With both changes the
  500,000 × 32 and 500,000 × 128 fits measure 6.30 ms and 13.20 ms (11.8×
  and 5.8× stock sklearn, down from 13.86 ms and 23.96 ms) on the measured
  M4 Max.
- The generic L-BFGS solvers (multinomial, CPU and generic weighted fits)
  no longer stop at the first failed line search along a quasi-Newton
  direction: they clear the curvature history and retry once along the
  gradient, and only a failure there ends the run with
  `line-search-failure`. A stale history on a badly scaled problem could
  otherwise end a fit at the first direction along which the float32
  objective shows no decrease; runs that never fail a search are unchanged,
  and a fit that has reached its float32 objective floor still stops with
  the same reason.
- Stabilized small Metal logistic loss differences against fast-math
  reassociation so line searches retain meaningful decreases across GPU and
  macOS versions. Weighted native fits use a small internal stopping reserve
  to account for rounding in the running loss estimate; convergence still
  requires the existing independent gradient and objective checks. Native
  command-loop initialization now supports
  isolated Python 3.12+ subinterpreters while preserving the Python 3.10
  stable ABI and per-interpreter buffer ownership. The [compatibility report](docs/ci-compatibility-2026-10-27.md)
  records the tests and supplemental accuracy panel.
- Binary and multinomial intercept refinement now preserve small represented
  loss changes with explicit stable arithmetic. This lets shifted weighted
  softmax fits certify their original-coordinate gradients across Metal
  implementations without changing the requested stopping tolerance.

- Rechecked the three original tall-data fixtures after the solver changes:
  complete GPU fits reached 5.24×, 3.05× and 2.64× stock sklearn on the
  measured M4 Max, with independent accuracy checks on every measured fit.
  The [rerun report](docs/tall-logistic-recheck-2026-10-27.md) records the
  method and raw results; the full suite passes 1,118 tests.

- Wide GPU L-BFGS directions now use a cooperative 256-thread recurrence,
  retaining the scalar metric and 32-thread narrow path. Selected strongly
  scale-skewed wide binary fits use a bounded L-BFGS warm start followed by
  native Metal Newton-CG, with a numerical dual-gap diagnostic and compensated
  original-input gradient check. Weakly regularized inputs retain L-BFGS.
  The [development report](docs/rank-safe-newton-performance-2026-10-27.md)
  records the final timing panels, independent accuracy references, general
  rank-deficiency checks and validation status. Ordinary 20,000 × 4,096 fits
  are about twice as fast as the frozen implementation and default sklearn
  on the measured Apple M4 Max; gains are not uniform across all workloads.
- Added a private NumPy-free GPU Hessian-vector building block and additional
  4,096-feature benchmark stress cases. It was initially kept separate from
  model fitting. The [ill-conditioned solver investigation](docs/ill-conditioned-hvp-investigation-2026-10-27.md)
  documents the remaining objective gap and candidate solver limitations.
- Wide binary GPU fits now use a stricter relative loss-progress check, and
  seeded resident-device fits use a small reserve in their native stopping
  tolerance while retaining the independent compensated certificate. The
  [wide accuracy report](docs/wide-accuracy-device-repair-2026-10-27.md)
  records the runtime cost and a remaining ill-conditioned objective gap.
- Weighted binary GPU fits now require an additional two-step relative loss
  progress check before convergence is reported. The GPU generic fallback uses
  a stricter weighted gradient threshold. Both are empirical safeguards, not
  objective-error bounds; iteration limits and nonconvergence warnings remain.
  A failed managed native line search is no longer promoted by a small direct
  gradient alone. See the [weighted accuracy investigation](docs/weighted-wide-investigation-2026-10-27.md).
- Long native binary GPU fits now submit up to 16 optimizer iterations per
  command buffer after the first 32 accepted steps, capped by the remaining
  budget. Earlier checks, optimization math, and convergence criteria remain
  unchanged. The [investigation](docs/sketch-weighted-optimization-2026-10-27.md)
  records both the weighted improvement and unresolved sketch and accuracy limits.
- Added an experimental NumPy-free `mlxlearn_device.fit_logistic` runtime path
  for binary float32 MLX arrays and GPU Hessian-sketch L-BFGS history seeding.
  The device check uses compensated GPU arithmetic without a formal error bound;
  unsupported data and unsuccessful fits are reported explicitly. The existing
  distribution continues to depend on NumPy and scikit-learn.
- The sklearn-compatible binary GPU solver supports an opt-in sketch via private
  tuning, and tightens the internal gradient threshold for strongly scale-skewed
  prepared inputs. This fixes the reproduced objective gaps but is not a general
  objective-parity guarantee. See the [device and Hessian-sketch report](docs/gpu-hessian-device-logistic-2026-10-27.md).
- Native binary optimization now accounts for attempted iterations and resumes
  from the checked parameters with only the remaining budget; it never silently
  restarts an exhausted fit. Strongly skewed, wide, unseeded fits use the GPU
  managed loop directly to avoid spending their budget on a problematic native
  trajectory. The device score uses coalesced compensated gradient reductions
  with a bounded-workspace fallback.

- Logistic convergence checks now account for the original feature coordinates
  after centering, without changing the centered optimization objective or search
  gradients. When a heuristic indicates rounding risk, a selective CPU float64
  certificate evaluates the returned model against the caller's input. Stalled
  centered fits may refine intercepts within the existing iteration budget;
  unresolved cases report nonconvergence. The certificate trigger is not a
  global rounding-error bound, and some heavily shifted multinomial stress cases
  still do not converge at the requested tolerance.
- A two-step average of accepted loss progress is used as an
  additional stopping condition for native binary fits with at least 2,048
  features. This remains a heuristic, not a general guarantee of objective
  parity. See
  the [convergence investigation](docs/convergence-accuracy-2026-10-27.md).
- Added an optional Apple Silicon native Metal command loop for prepared wide
  binary logistic fits. It retains MLX arrays, DLPack capsules and pending
  command buffers until completion is known, including exceptional and
  interpreter-teardown paths. Only stable MLX 0.32.x uses this bridge; other
  versions and unavailable extensions retain the established solver.
- Native command submission now checks after eight steps, then after one more,
  then in bounded groups of four. This reduces redundant dispatches when a
  weighted fit converges on its ninth step; final direct certification remains.
- Apple Silicon development wheels now include a CPython stable-ABI extension;
  source builds without the optional extension retain the existing solver.
- Wide logistic GPU preparation now combines input validation and centering with
  native binary initialization, removing separate matrix passes. Large contiguous
  float64 inputs use bit-equivalent NumPy conversion in temporary worker blocks.
  Borrowed input storage remains alive through GPU evaluation and cleanup;
  that preparation change retained the then-current convergence checks.
- The wide benchmark accepts a complete frozen-package reference archive, isolates
  lazily imported Metal kernels, and preserves failed quality rows. Stage profiling
  accounts for preparation moved outside the optimizer timer.
- Wide binary GPU logistic steps now fuse line-search loss differences with the
  first candidate's gradient in one matrix pass. Acceptance selects that gradient
  only when the corresponding step is accepted; backtracking retains a separate
  gradient evaluation, and direct final convergence verification remains required.
- GPU logistic input validation combines finiteness checking with an owned copy
  of float32 inputs. Float64 inputs reuse their checked private conversion without
  a second CPU scan or upload copy. Caller-owned inputs remain isolated from fit.
- Added `python -m benchmarks.logistic_wide --profile full --dtype both --threads 1`
  with a source-hashed frozen baseline, and `python -m benchmarks.logistic_profile`
  for reproducible stage profiling.
- Binary GPU logistic regression now uses handwritten Metal C++ kernels for
  L-BFGS history and directions, fused directional projection and batched Armijo
  search, residual updates, and tiled gradient reductions. A curvature-scaled
  first step reduces matrix passes; a direct final objective and gradient check
  certifies convergence. CPU and multinomial fits retain the fused solver.
- Float64 range validation avoids matrix-sized masked and absolute-value copies.
  Logistic centering now runs on the selected device and corrects the intercept
  using the exact float32 shift that was applied.
- Added `python -m benchmarks.logistic_native --profile full --dtype both` for
  fresh-fit comparisons with the prior fused solver and stock scikit-learn.
- Logistic regression now keeps optimizer vectors and curvature history on the
  selected MLX device and uses kernel fusion for the L-BFGS recursion, updates,
  loss, gradient, and directional derivative. It synchronizes small statistics
  batches at control decisions, preserving adaptive strong-Wolfe line search,
  host float64 loss accumulation, centering and gradient-based convergence.
- Signed binary logits and explicitly shifted multinomial logits reduce loss
  cancellation in float32. Exact unit weights use the unweighted compiled graph.
- Added a reproducible full-fit comparison against the previous solver and stock
  scikit-learn, including convergence and independent float64 quality checks:
  `python -m benchmarks.logistic_gpu --profile full`.

### Measured

- The revised rule passed 700 tests and a 30-repeat targeted benchmark; 20k ×
  4,096 fits were 12.9–14.1% slower than the frozen baseline, while remaining
  1.52×–2.44× faster than sklearn. The 14-case full panel also passed its
  audits and measured 12.1–14.5% slower 4,096-feature fits than the frozen baseline.
  The following performance comparisons predate the convergence changes. An earlier
  single-step rule cost about 9–11% at 20,000 × 4,096 in two panels and did
  not pass a separate `C=10` objective comparison. Details and limits are in
  the [convergence investigation](docs/convergence-accuracy-2026-10-27.md).
- In the preceding 30-repeat targeted comparison, 20,000 × 1,024 float32 fits
  measured 6.555 ms versus 6.925 ms for the prior solver and 12.829 ms for
  sklearn; weighted fits measured 6.921 ms versus 7.806 ms and 13.944 ms.
  All 120 targeted fits passed independent quality checks. The broader panel
  also shows regressions and timing variability, and the first native fit has
  additional shader-compilation cost. That historical version retained a
  default-tolerance objective gap at 4,096 features and an original-coordinate
  gradient limitation on heavily shifted inputs. See the
  [native loop report](docs/native-loop-optimization-2026-10-26.md).
- An earlier experimental direct Metal command loop measured 7.203 ms versus 7.567 ms
  for the existing solver at 20,000 × 1,024 float32 in a 30-repeat comparison,
  with regressions on weighted inputs. That prototype was not included because
  gains were mixed and exceptional buffer-lifetime paths needed further work. See the
  [experiment report](docs/metal-command-benchmarks.md) for all timings,
  accuracy checks, and a source-only reproduction archive.
- The final 30-repeat overhead panel reaches 1.68× sklearn at 20,000 × 1,024
  float32 (7.647 ms versus 12.884 ms), a 1.10× gain over the frozen previous
  implementation in the same run. At 60,000 × 1,024, speedups reach 2.61× for
  float32 and 3.15× for float64. All 583 tests pass. The existing default-tolerance
  objective gap at 4,096 features remains; a shared `tol=1e-5` passes the strict
  comparison. See the [overhead results](docs/overhead-logistic-benchmarks.md) for
  every case, timing spread, failed default rows, and stricter comparisons.
- Wide float32 binary logistic fits now reach 2.11× over sklearn at 60,000 × 1,024
  (16.702 ms versus 35.182 ms), up from the preceding 1.38× result. The smaller
  20,000 × 1,024 case reaches 1.41×. At 4,096 features, the expanded panel exposes
  an existing objective-parity limitation at default tolerance; all three solvers
  pass the stricter comparison at `tol=1e-5`, with a 2.44× float64 GPU speedup.
  See the [wide results](docs/wide-logistic-benchmarks.md) for every shape, variance,
  profiling, and the retained failed default-tolerance rows. All 514 tests pass.
- Native binary GPU logistic fits reach 4.86× over stock scikit-learn and 5.56×
  over the previous fused implementation on the measured M4 Max. The largest
  gains are on tall data; small inputs still favor scikit-learn. Complete fits,
  all measured shapes and quality checks are in the
  [native results](docs/native-logistic-benchmarks.md).
- The preceding fused GPU logistic fits were 1.12–3.79× faster than the original solver across
  six workloads and float32/float64 input panels on M4 Max. Weighted multiclass
  beat stock scikit-learn by 1.31–1.68×; binary cases in that earlier run favored
  scikit-learn. See the [historical fusion comparison](docs/gpu-logistic-benchmarks.md).

## [0.1.0a2] — 2026-09-02

### Changed

- Renamed the source repository from `tabulai/mlxlearn` to `tabulai/mlx-learn` and
  updated package metadata, development instructions, and release automation to use the
  canonical URL.
- Updated the release metadata and runtime version for the repository-name migration.
- Corrected the CI support matrix so Python 3.10 is tested with its compatible
  scikit-learn 1.7.x release line; scikit-learn 1.8+ requires Python 3.11 or newer.

## [0.1.0a1] — 2026-09-02

First alpha. Published for external testers before the second wave of estimators; the API
is not yet stable.

### Added

- **Layer 1 estimators** (`import mlxlearn.…`), MLX-accelerated with NumPy in and NumPy out:
  - `mlxlearn.neighbors.NearestNeighbors`, `KNeighborsClassifier`, `KNeighborsRegressor` —
    exact brute-force neighbors with blocked distance evaluation, `mx.argpartition` top-k
    selection, deterministic `(distance, index)` tie-breaking, and candidate-distance
    refinement that avoids fp32 cancellation in the expanded Gram formula.
  - `mlxlearn.linear_model.LogisticRegression` — L2 / unpenalized, binary and multinomial,
    L-BFGS on the exact scikit-learn objective. The design is centered before solving
    (exact when there is an intercept to absorb the shift), which is what makes the fit
    survive features carrying a large constant offset — and makes it converge in fewer
    iterations than scikit-learn on such data.
  - `mlxlearn.svm.SVC` — exact SMO with a memory-bounded kernel cache and one-vs-one
    multiclass. No approximations ship under a scikit-learn estimator name.
- **Layer 2 patching** (`mlxlearn.patch_sklearn` / `unpatch_sklearn`) — idempotent,
  fully reversible, module-attribute based, with a registry of Layer 1 estimators.
- **Sticky backend invariant** — the backend is selected during `fit`, recorded on
  `_execution_backend_`, and reused by every inference call on that fitted model.
- **Fallback taxonomy** — capability mismatches fall back under
  `fallback_policy="warn" | "raise" | "silent"`; unexpected MLX runtime failures raise by
  default and are never silently rerun on scikit-learn. Anything scikit-learn *can* serve
  is in the capability family so the fallback actually works — including float64 values
  that overflow float32 and negative sample weights, both of which scikit-learn accepts.
- **Diagnostics** — `get_backend_diagnostics()`, `get_last_backend_event()`,
  `clear_backend_diagnostics()`.
- **Configuration** — `set_config` / `get_config` / `config_context` over `device`,
  `fallback_policy`, `output_type`, `deterministic`, `random_state`, `diagnostics`;
  private tuning via `MLXLEARN_*` environment variables.
- **Benchmarks** — `python -m benchmarks.run`, `mx.eval`-synchronized, warmup plus median
  of ≥5, cold-start / fit / predict reported separately, per-algorithm size grids, and
  measured crossovers wired into the dispatch thresholds.
- **Compliance gates in CI** — Intel/sklearnex branding grep, private `sklearn._*` import
  check (quarantined to `patching/`), header/provenance audit, dependency license scan.

### Measured

Full numbers and method in [`docs/benchmarks.md`](docs/benchmarks.md); tolerances and
parity claims in [`docs/fp32_policy.md`](docs/fp32_policy.md).

- **Neighbor queries: 2.9×–17× against stock scikit-learn**, from ~250 samples upward. The
  crossover was measured at ~250–1 000 samples, not the 4 096 originally assumed.
- **Neighbor `fit`: 0.17×–0.36×**, and knowingly so — mlxlearn uploads to the device where
  scikit-learn stores a reference. Listed by name in `benchmarks/gate.py` rather than
  hidden behind "no crossover measured".
- **`LogisticRegression` is 10–50× slower than scikit-learn below 1 024 features.** Its
  iteration count matches scikit-learn's, but each iteration pays a host synchronization
  per line-search evaluation: 5.9 ms per iteration where the matmul inside it costs 163 µs.
  The crossover is set high accordingly (20 000 samples **and** 2·10⁷ elements), so patched
  dispatch measures 0.79×–1.01×, both within noise. This is an optimization target, not a
  property of the design.
- **`SVC` is decided by width, not sample count.** 2.93× at 4 000 × 256 (rbf), 0.47× at the
  same sample count and 32 features, and slower in 58 of 64 configurations on the initial
  narrow grid — worst 0.013×. SMO is sequential, so MLX can accelerate the kernel rows an
  iteration needs but not the iterations; whether that pays depends on how expensive a row
  is. The crossover is therefore on work (`svc_min_samples` **and** `svc_min_work`), and
  `kernel="linear"` never takes the MLX path at any size. Its primal objective agrees with
  LIBSVM to 5.2e-07 relative and the KKT conditions hold on an independent check — this is
  a dispatch story, not a correctness one.
- **All ten scikit-learn estimator check suites pass with zero mlxlearn-specific xfails.**
  Two checks are exempt only because scikit-learn declares them as expected failures for
  its own `SVC` and `LogisticRegression`; a test re-derives that list from scikit-learn so
  the exemption cannot outlive its justification.

### Compatibility

- Tested against scikit-learn **1.7.2 and 1.9.0**; 289 tests pass on both. Three parameters
  scikit-learn is deprecating mid-range — `LogisticRegression.multi_class` (removed in 1.9),
  `LogisticRegression.penalty` (now a sentinel, with `l1_ratio` carrying the meaning) and
  `SVC.probability` (now a sentinel) — are handled by resolving what the installed version
  actually means rather than reading the raw attribute. See
  [`docs/support_matrix.md`](docs/support_matrix.md).

### Notes

- Ships as a pure-Python wheel. The compiled-extension decision is recorded with
  measurements in [`phase0/cython_decision.md`](phase0/cython_decision.md): the compiled
  SMO core was worth 1.26× at 2 000 samples, 1.16× at 6 000, and 1.00× at 12 000 — the
  advantage vanishes exactly where performance matters. Revisit when `TSNE` lands in 0.2.
- Requires macOS on Apple silicon. Behavior on other platforms is untested and unsupported.
- Not reviewed by a lawyer.

### Deliberately not carried over from the ancestor

Each of these is now covered by a regression test:

- `exclude_self` compared each row against its **block-local** index, so `kneighbors(None)`
  returned a row as its own nearest neighbor for any training set larger than one query
  block (1 wrong row in 1 500 with default blocking).
- The GPU top-k returned `uint32` indices, so padding a short trailing tile with `-1` raised
  `Converting -1 to uint32 would result in overflow` — presented to users as an unexplained
  fallback, because the dispatcher caught every exception.
- `metric_params` was accepted and silently ignored, returning unweighted Euclidean results
  for a weighted-Minkowski request.
- Module-level caches keyed on `id()` or a weakref to a caller's array, so a mutated or
  freed-and-reallocated array produced predictions computed on the wrong data.
- A `fit` continuation cache that warm-started from a previous unrelated fit of the same
  shape, making `fit` a non-pure function of its arguments.
- `except Exception:` around the accelerated path, rerunning silently on scikit-learn.

[Unreleased]: https://github.com/tabulai/mlx-learn/compare/v0.1.0a2...HEAD
[0.1.0a2]: https://github.com/tabulai/mlx-learn/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/tabulai/mlx-learn/releases/tag/v0.1.0a1
