# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a1] — unreleased

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

[Unreleased]: https://github.com/tabulai/mlxlearn/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/tabulai/mlxlearn/releases/tag/v0.1.0a1
