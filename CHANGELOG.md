# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- SVC now solves eligible subproblems with native hierarchical working sets,
  batches GPU kernel-row requests within the configured cache budget, and
  certifies convergence using a reconstructed full gradient. The native pair
  solver implements active-set shrinking. Inference reuses owned support-vector
  preparation and RBF norms, with exact mutation checks and safe pickle/refit
  behavior. Public tolerances and iteration budgets are preserved. See the
  [additional speedups, source study and limitations](docs/svc-working-set-2026-10-31.md).
- SVC now uses a native float64 SMO loop with fused gradient/working-set scans
  and a bounded native row cache. Metal kernels fuse nonlinear transforms and
  on-demand row evaluation; native CPU routines replace NumPy statistics,
  prediction reductions and multiclass voting. The exact C-SVC objective,
  float32 kernel/float64 optimizer precision split, public API, and conservative
  automatic routing remain intact. Final KKT diagnostics now describe the
  returned iterate even when the iteration budget is exhausted. See the
  [benchmarks and validation](docs/svc-native-2026-10-31.md).
- Native KNN selection for `k=5` now validates dot products and selects exact
  integer neighbors in one pass, committing private results only after every
  row passes. Precision, canonical ties, malformed-input checks and error
  precedence are preserved. All 432 Digits GPU-control comparisons now win,
  with a minimum 1.047× speedup; the former weighted-regression miss is
  1.051× paired / 1.079× ratio-of-medians faster than historical GPU. The full
  six-panel rerun retains all startup/first-query wins against the three sklearn
  configurations, all 81 explicit Gaussian-distribution broad wins and all 108
  clustered/offset broad wins. All 1,474 tests
  pass on sklearn 1.7, 1.8 and 1.9, plus 65 harness checks and 14,472 broad
  numerical audits. Repeated Breast Cancer and smaller first-query losses
  against preceding automatic remain; an exact-native A/B/A probe does not
  reproduce the large Breast losses. See the [complete results and limitations](docs/knn-digits-fix-2026-10-30.md).
- Direct KNN estimators now use sklearn's public estimator interfaces without
  eagerly importing its concrete neighbor classes. Direct instances no longer
  inherit those stock classes; patched variants retain stock inheritance and
  graph-helper compatibility. Constructors, validation, metadata, scoring, graphs, radius
  operations and older saved models remain covered. Native integer selection
  and concentrated-feature scans are faster without changing precision or GPU
  kernels. In 450 fresh processes, all 18 fixture medians beat all three sklearn
  configurations on total startup-to-result, with imports included and 15–18%
  lower latency. All 1,408 relevant tests pass on sklearn 1.7, 1.8 and 1.9;
  65 harness checks and 14,472 current broad audits also pass. Some isolated
  fresh-process first-query subphases and legacy GPU comparisons still regress;
  initialized fresh-model first-query and combined fit/query medians improve on
  all 18 fixtures versus both prior automatic and the three sklearn settings; see the
  [complete startup results and compatibility limits](docs/knn-startup-fix-2026-10-30.md).
- The earlier KNN latency revision fused certified integer Gram selection in
  the native CPU extension and computed feature ranges with a bit-preserving
  native helper. Eligible host fits deferred GPU probing, and CPU classifier
  predictions avoided GPU voting imports. Native loading moved into API setup;
  its cost remained included in cold startup measurements. All 27 warmed Digits
  cells improved 1.82–2.45× over the preceding automatic build, and all 162
  targeted medians beat the three sklearn
  configurations in that run. Its 1,191 relevant tests and 14,472 broad accuracy
  audits passed. Total process startup lost to sklearn auto on 11 of 18
  fixtures, and some controlled cells regressed against earlier implementations;
  see the [latency results and limitations](docs/knn-latency-fix-2026-10-30.md).
- KNN now captures an automatic CPU/GPU policy at fit and routes each query
  batch by size and certified data properties. The optional compiled CPU scan
  resolves ambiguous distances and ties exactly; portable CPU search retains
  the same ordering contract. Explicit device choices remain fixed, and the
  preceding GPU kernels are unchanged. Its original benchmark run recorded all
  162 targeted small/single-query medians beating the three sklearn configurations,
  with 1,025 relevant tests passing; see the
  [original results and limitations](docs/knn-auto-dispatch-2026-10-30.md).
- Small KNN GPU workloads now combine direct screening, selection, precision
  certification and exact fallback in one launch. Conservative fallback pruning
  preserves boundary ties, and compiled calls retain independent training/query
  inputs. Canonical finite float32 queries skip repeated generic conversion while
  retaining sklearn feature checks. The established large-batch routes remain
  unchanged; see the [small-workload results](docs/knn-small-workload-fix-2026-10-30.md).
- KNN compiles reusable Gram preparation and evaluates it with the first query,
  removing a separate GPU submission and synchronization. Shape-only sampling
  plans reuse indices without caching training values or results. All six
  controlled 128-feature Gaussian fresh-fit cases beat both preceding builds;
  clustered/offset gains and precision checks remain intact. The full sklearn
  rerun passes all 14,472 current audits and 714 relevant tests pass; see the
  [fresh-fit fix and complete results](docs/knn-gaussian-fresh-fix-2026-10-30.md).
- KNN routes tight clusters to the existing precise direct scan and uses guarded
  exact centering for integer screening on large-offset data. Final distances,
  exact tie ordering and complete fallback retain the original coordinates.
  Cached distribution and scale decisions share one small owned-training sample;
  automatic memory plans remain distinct from explicitly requested tiles.
  All 108 targeted clustered/offset medians beat the three sklearn baselines;
  see the [full rerun and precision validation](docs/knn-distribution-fix-2026-10-30.md).
- KNN searches with 33–256 features and k<=8 now fuse shortlist refinement,
  precise ordering and omission certification, and retain k+2 candidates.
  The distance bounds, exact comparisons and exhaustive fallback are unchanged.
  See the [128-feature fix and complete v3 benchmarks](docs/knn-128-feature-fix-2026-10-30.md).
- KNN now resolves close GPU neighbors using guarded higher-precision comparisons
  and exact original-coordinate integer ordering when necessary. General GPU
  screening certifies omitted candidates; CPU search selects from direct float64
  distance tiles. This fixes rounded cutoff ties and copied-row loss from Gram
  cancellation. Candidate refinement shares precision work across GPU lanes,
  large lists verify an initial sort, and native fit preserves subnormal bits.
  See the [precision fixes and full v3 rerun](docs/knn-precision-fix-2026-10-30.md).
- Large native float32 KNN inputs now use a vectorized finiteness check while
  retaining sklearn validation errors and independently owned fit storage.
  This removes the residual unequal-scale fresh-fit cost without changing
  the Gaussian search path. See the [fresh-fit comparison against v3](docs/knn-fresh-fit-fix-2026-10-29.md).
- Guarded integer KNN dispatch with a cached estimate of training feature
  variation. Unequal-scale data retains the compact filter, avoiding costly
  integer-certificate fallbacks while preserving the Gaussian fast path.
  Distance arithmetic and both numerical certificates are unchanged. See the
  [regression tests and v3 comparisons](docs/knn-scale-regression-fix-2026-10-29.md).
- Large-batch KNN now uses compact training storage, conservative quantization
  bounds, packed candidate keys and short local lists, followed by original
  float32 refinement and exact fallback. GPU fits defer the independent host
  mirror until radius queries, pickling or private-state access needs it. On
  50,000 × 32 data, all 54 search/inference operation–dataset medians across two
  runs exceed default-thread sklearn by 2.10–3.02×, including fresh fit plus
  query. See the [complete benchmark tables](docs/knn-v3-2026-10-28.md) and
  [numerical argument](docs/knn-compact-certificate.md).
- Further optimized paired KNN scans with packed four-feature reads and named
  scalar candidate state for k=5, preserving compensated arithmetic and input-view
  support. Wide-float fits reuse their private checked conversion buffer instead
  of copying it again. A controlled idle/warmup experiment now distinguishes
  intermittent latency from fully warmed measurements; see the
  [implementation and complete comparison](docs/knn-v2-2026-10-28.md), including
  retained speed-target misses and regression results.
- KNN uses a shared NumPy-free MLX search core, with an experimental
  `mlxlearn_device.fit_neighbors` API. Small searches fuse compensated Euclidean
  distances and top-k selection; wider searches retain centered matrix products
  with bounded candidate selection and fused refinement. Classifier voting stays
  on-device, uniform weights skip distance output, and float64 input preparation
  reuses logistic regression's checked conversion. Tie selection, original-coordinate
  refinement, self-exclusion and target ownership are covered by new regression
  tests. See the [implementation and benchmark report](docs/knn-pipeline-2026-10-28.md).
  A subsequent pass increases direct-scan parallelism, shares training loads
  between query pairs, moves overflow recovery out of the feature loop and
  removes a redundant fit copy and synchronization. See the
  [three-dataset follow-up](docs/knn-acceleration-2026-10-28.md) for measured
  improvements and the remaining gap to consistent 2× default-thread sklearn.
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
  gradient when evaluations remain in the iteration's `max_line_search`
  budget. A failed retry or exhausted budget ends the run with
  `line-search-failure`. Both searches share that limit. A stale history on a
  badly scaled problem could
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

### Fixed

- The L-BFGS curvature floor is relative to the pair (`sᵀy > min_curvature·yᵀy`,
  in the native solver's metric) in the Metal direction kernel, the native
  command loop and both Python solvers. The former `max(yᵀy, 1)` form was an
  absolute 1e-12 floor whenever one feature's scale made `yᵀH₀y` small, which
  rejected every valid pair, froze the history and left a 1e5-scaled design at
  the null model (accuracy 0.51 against scikit-learn's 0.86). The MLX loop now
  also treats accepted steps that leave every cached logit unchanged as a
  stall: it resets the history once, then reports a failed search so the
  intercept polish and float64 certificate run instead of spending `max_iter`.
- Zero, subnormal, and finite tolerances above float32's maximum no longer
  raise on the Newton route; such fits stay on L-BFGS. Native tolerance
  conversion clamps large finite values before casting, without overflow
  warnings or relaxing the requested threshold.
- Fits with 512 to 2047 features and an initial curvature-diagonal spread
  greater than the reciprocal of float32 epsilon now use the existing
  Newton-CG refinement when its regularization and device requirements are
  met. This avoids large objective gaps from L-BFGS history roundoff on
  extremely scaled columns, while retaining the original objective,
  accepted-step budget and independent convergence checks. Newton routing
  also checks CG's 64 MiB workspace limit before spending a warm-start budget;
  unsupported shapes keep the full L-BFGS budget.
- Native CG no longer discards positive low-curvature directions merely
  because an earlier direction had much higher curvature when every feature's
  regularization is at least the Newton routing floor. That cutoff stalled
  even a diagonal positive-definite system with a 1e10 curvature spread.
  Weakly regularized systems retain the original cutoff to limit null-space
  drift; finite/positive curvature, residual, budget and outer convergence
  checks remain in force.
- The `MLXLEARN_LBFGS_SKETCH_SIZE` opt-in (and `tuning_context(lbfgs_sketch_size=...)`)
  is treated as a capability of the prepared native loop: fits it cannot seed
  (fewer than 512 or more than 4096 features, no intercept,
  multinomial targets, a non-default history, no extension) run unsketched at
  the requested tolerance instead of raising. A direct `solve_logistic`
  request for a sketch without that capability is still an error.
- The native-to-MLX resume guard no longer discards a continuation the direct
  check certifies, and compares uncertified continuations against the native
  objective within a float32 rounding band, keeping the iterate with the
  smaller direct gradient. Tight-tolerance wide fits previously returned a
  non-converged native iterate with the discarded steps still counted.

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
