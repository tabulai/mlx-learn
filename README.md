# mlxlearn

**Classical machine learning on Apple silicon, accelerated with [MLX](https://github.com/ml-explore/mlx), with a scikit-learn compatible API.**

> mlxlearn is not officially associated with scikit-learn or PROBABL, nor with Apple.

`mlxlearn` has two layers.

**Layer 1 — the library.** Explicit imports, typed solver configuration and state, strict
errors, eager `fit`, no mutation of scikit-learn, no data-keyed caches:

```python
from mlxlearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

**Layer 2 — the patching shell.** A thin adapter that points scikit-learn's own names at
Layer 1 estimators, so existing code — including code an LLM generated against the
scikit-learn API — runs accelerated without being edited:

```python
from mlxlearn import patch_sklearn
patch_sklearn()

from sklearn.svm import SVC          # now resolves to the accelerated class
```

The numerical library is fully usable and fully tested with the patching mechanism absent.

## Status

**`0.1.0a2` — alpha.** The API may change. See [CHANGELOG.md](CHANGELOG.md).

| Estimator | Status |
|---|---|
| `mlxlearn.neighbors.NearestNeighbors` | 0.1.0 |
| `mlxlearn.neighbors.KNeighborsClassifier` | 0.1.0 |
| `mlxlearn.neighbors.KNeighborsRegressor` | 0.1.0 |
| `mlxlearn.linear_model.LogisticRegression` | 0.1.0 |
| `mlxlearn.linear_model.LinearRegression` | 0.1.0 |
| `mlxlearn.svm.SVC` (exact) | 0.1.0 |
| `Ridge`, `PCA`, `KMeans`, `DBSCAN`, `TSNE` | planned, 0.2.x |
| `SVR`, `NuSVR`, `NuSVC` | deferred — will not ship until they implement the true objective |
| tree ensembles | out of scope |

## Requirements

- macOS on Apple silicon (arm64)
- Python 3.10 – 3.13
- scikit-learn ≥ 1.7, MLX ≥ 0.29

Python 3.10 is supported with scikit-learn 1.7.x; scikit-learn 1.8+ requires Python ≥ 3.11.
CI tests only valid pairs — see [`docs/support_matrix.md`](docs/support_matrix.md).

## Install

```bash
pip install mlx-learn
```

The released `0.1.0a2` wheel is pure Python. Development builds on Apple silicon
can include an optional stable-ABI Metal extension; the existing MLX solver remains
available if that extension is absent.

## Import-order semantics

Patching replaces attributes **on scikit-learn modules**. Attribute access
(`sklearn.svm.SVC`) therefore resolves to the mlxlearn class whether `import sklearn`
happened before or after `patch_sklearn()`.

Symbols captured *before* patching are local bindings and cannot be rebound:

```python
from sklearn.svm import SVC     # binds the stock class into your namespace
patch_sklearn()                 # cannot reach back and change SVC
SVC()                           # still the stock class
```

That is Python name binding, not a bug in mlxlearn. **The supported pattern is patch-first.**
`patch_sklearn()` is idempotent, and `unpatch_sklearn()` fully restores scikit-learn.

## When mlxlearn falls back

Layer 1 raises a precise error when it cannot honor a request. Layer 2 falls back to stock
scikit-learn instead — but only for *capability* mismatches (sparse input, an unsupported
parameter, an unsupported dtype), never to hide a bug:

| Situation | Direct import | Patched |
|---|---|---|
| Capability mismatch | precise `MLXLearnError` | falls back, warns once per class, records diagnostics |
| Problem below the measured crossover | internal CPU path | dispatches to scikit-learn |
| Invalid user input | sklearn-equivalent validation error | scikit-learn's own exception, unmasked |
| Unexpected MLX runtime failure | raises | **raises** — never silently rerun on scikit-learn |

Set the policy with `fallback_policy="warn" | "raise" | "silent"`. Diagnostics are recorded
in every mode:

```python
import mlxlearn

with mlxlearn.config_context(fallback_policy="raise"):
    ...                                     # any capability fallback becomes an error

mlxlearn.get_last_backend_event()           # what the last fit/predict actually did
mlxlearn.get_backend_diagnostics()          # everything recorded this session
```

**Sticky backend.** The backend is chosen during `fit` and recorded as
`estimator._execution_backend_` (`"mlx"`, `"cpu"`, or `"sklearn"`). Every subsequent
`predict` / `predict_proba` / `transform` / `kneighbors` uses the backend the model was
fitted with. A model fitted on MLX can never wander into scikit-learn inference with
incompatible state. A new `fit` clears state and may pick a different backend.

## Configuration

The public surface is deliberately small:

```python
mlxlearn.set_config(
    device="auto",              # "auto" | "gpu" | "cpu"
    fallback_policy="warn",     # "warn" | "raise" | "silent"
    output_type="numpy",        # 0.1.0: NumPy in, NumPy out
    deterministic=True,
    random_state=0,
    diagnostics=True,
)
```

Crossover thresholds, block sizes, and solver tuning are private, typed, and overridable
only through `MLXLEARN_*` environment variables for CI and debugging. They are not part of
the public API.

## Performance

The latest [rerun of the original three tall datasets](docs/fused-narrow-kernel-2026-10-27.md)
measures complete float32 GPU fits at 6.30 ms (500,000 × 32), 13.20 ms
(500,000 × 128), and 13.01 ms (60,000 × 1,024): 11.8×, 5.8× and 2.7×
stock sklearn on the measured M4 Max. Every timed fit passed independent
accuracy checks. The two narrow cases gained from one fused line-search pass
and one input pipeline for every feature count; the [earlier recheck](docs/tall-logistic-recheck-2026-10-27.md)
records the previous timings and the exact settings, variability and limits
of the historical comparison.

The development version uses native Metal C++ kernels for binary GPU logistic
regression, including optimizer history, batched line search, and matrix reductions.
Wide fits fuse candidate losses and the likely accepted gradient into one matrix
pass. Wide-input preparation also combines validation, centering, and initial
gradient statistics, while large contiguous float64 inputs use a checked parallel
conversion. See the [earlier overhead comparison](docs/overhead-logistic-benchmarks.md)
for complete fresh-fit timings against the frozen previous implementation and stock
scikit-learn, including timing variability and numerical quality. In that earlier
comparison the 20,000 × 1,024 float32 case reached 1.68× sklearn; larger cases
benefited more.
Reproduce it with:

```bash
python -m benchmarks.logistic_wide \
  --reference-archive benchmarks/reference/wide-logistic-2026-10-26.zip \
  --profile full --dtype both --repeats 30 --warmup 3 --threads 1
```

The [previous wide comparison](docs/wide-logistic-benchmarks.md) records the matrix
kernel changes and the pre-existing 4,096-feature objective-parity limitation.
The development version also includes an optional native Metal command loop for
prepared wide binary fits on MLX 0.32.x. It retains input buffers through GPU
completion and falls back to the established solver when unavailable. See the
[native loop results](docs/native-loop-optimization-2026-10-26.md) for warm and
first-fit timings, numerical checks, supported inputs, and limitations.
The [preceding native comparison](docs/native-logistic-benchmarks.md) records the
earlier tall-data improvements and small-data limitations.

Development work also corrects logistic convergence checks for centered inputs:
the stopping gradient is evaluated in the original feature coordinates. For
inputs at risk of float32 rounding error, a selective CPU float64 pass checks the
returned model against the original input; it is a certificate, not a second
solver. Its trigger is heuristic and does not cover every possible rounding error.
Some severely shifted multinomial stress cases still report nonconvergence at
the requested tolerance. A two-step wide-model loss-progress heuristic addresses
an observed 4,096-feature objective gap; it also is not
a guarantee of objective parity. The speed results linked above are historical,
before these convergence changes. The revised solver's targeted benchmarks show
12.9–14.1% slower 4,096-feature fits than the frozen baseline. See the
[convergence investigation](docs/convergence-accuracy-2026-10-27.md)
for evidence, limitations and the broader benchmark.

The experimental `mlxlearn_device.fit_logistic` interface accepts and returns MLX
arrays without importing NumPy or scikit-learn. It supports binary float32 inputs
on the tested Metal bridge and can initialize L-BFGS from GPU Hessian-sketch
curvature pairs. The sklearn-compatible path exposes the sketch as an opt-in
private tuning setting. The device convergence check is compensated GPU
arithmetic, not a formal error bound; the runtime API, supported inputs and
accuracy limitations are described in the [device and Hessian-sketch report](docs/gpu-hessian-device-logistic-2026-10-27.md).
The [preceding sketch and weighted investigation](docs/sketch-weighted-optimization-2026-10-27.md)
measures a scheduling improvement for long fits and records the unbalanced-data
objective gap found under the earlier gradient stopping rule.
The [follow-up weighted accuracy report](docs/weighted-wide-investigation-2026-10-27.md)
describes the additional progress safeguard, its cost and its limits, and the
wide GPU candidates evaluated against the starting implementation.
The [wide accuracy and device-sketch report](docs/wide-accuracy-device-repair-2026-10-27.md)
documents the subsequent ordinary 4,096-feature improvement, the device retry
fix, and the remaining strongly correlated accuracy gap.
The [ill-conditioned solver investigation](docs/ill-conditioned-hvp-investigation-2026-10-27.md)
examines that unresolved gap and a private GPU Hessian-vector building block.
That investigation enabled no new solver. The subsequent
[cooperative L-BFGS and guarded Newton report](docs/rank-safe-newton-performance-2026-10-27.md)
describes the implemented parallel direction recurrence and bounded Newton
refinement for strongly scale-skewed wide inputs. The final standard panel
measures 30.24 ms versus sklearn's 65.52 ms on 20,000 × 4,096 ordinary inputs,
and independent references confirm the reproduced ill-conditioned and weighted
accuracy fixes. General rank-deficiency checks pass; the report records cases
without a speed win, reference limitations and full-suite validation status.

"Never slower than scikit-learn" is the design goal. The shipped *gate* is measurable: on
each benchmarked workload class, patched dispatch shows no statistically significant
regression against stock scikit-learn. Crossover points are measured per algorithm and per
operation, published in [`docs/benchmarks.md`](docs/benchmarks.md), and wired into the
dispatch thresholds — small problems are handed to scikit-learn on purpose.

What that means in practice for 0.1.0a2, on an M4 Max:

| | |
|---|---|
| **Neighbor queries** | **2.9×–17×** from ~250 samples up. This is the reason to use mlxlearn. |
| Neighbor `fit` | 0.17×–0.36×. mlxlearn uploads to the device; scikit-learn stores a reference. The first query repays it several times over. |
| `SVC` | **2.93× on wide data** (4 000 × 256, rbf), but slower than LIBSVM on narrow data — 0.47× at 4 000 × 32, worse below. `kernel="linear"` never uses MLX. Width decides, and patched dispatch routes accordingly. |
| `LogisticRegression` | **Slower than scikit-learn below 1 024 features**, by a lot. The crossover is set high so patched dispatch hands those to scikit-learn; see [`docs/benchmarks.md`](docs/benchmarks.md) for why and what would change it. |

Reproduce them:

```bash
python -m benchmarks.run --profile smoke
```

## Precision

MLX computes in float32 on the GPU. mlxlearn's parity tests are written to float32
tolerances, and the cases where float32 makes strict scikit-learn equivalence impossible
are enumerated — not waved away — in [`docs/fp32_policy.md`](docs/fp32_policy.md).

## Development

```bash
pip install -e ".[dev]"
pytest
```

Contributor rules, the estimator gate checklist, and the compliance checks are in
[`docs/development.md`](docs/development.md).

## Provenance

mlxlearn succeeds a private research fork of an Intel-maintained scikit-learn accelerator.
That history is not carried here: this repository was bootstrapped from an audited
allowlist, and no source file was copied. The audit, the recorded behavioral baseline of
the ancestor, and the authorship attestation are in [`phase0/`](phase0/). Full lineage,
disclaimers, and third-party notices are in [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
