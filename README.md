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

**`0.1.0a1` — alpha.** The API may change. See [CHANGELOG.md](CHANGELOG.md).

| Estimator | Status |
|---|---|
| `mlxlearn.neighbors.NearestNeighbors` | 0.1.0 |
| `mlxlearn.neighbors.KNeighborsClassifier` | 0.1.0 |
| `mlxlearn.neighbors.KNeighborsRegressor` | 0.1.0 |
| `mlxlearn.linear_model.LogisticRegression` | 0.1.0 |
| `mlxlearn.svm.SVC` (exact) | 0.1.0 |
| `LinearRegression`, `Ridge`, `PCA`, `KMeans`, `DBSCAN`, `TSNE` | planned, 0.2.x |
| `SVR`, `NuSVR`, `NuSVC` | deferred — will not ship until they implement the true objective |
| tree ensembles | out of scope |

## Requirements

- macOS on Apple silicon (arm64)
- Python 3.10 – 3.13
- scikit-learn ≥ 1.7, MLX ≥ 0.29

Python 3.10 is supported against scikit-learn ≤ 1.8; scikit-learn 1.9 requires Python ≥ 3.11.
CI tests only valid pairs — see [`docs/support_matrix.md`](docs/support_matrix.md).

## Install

```bash
pip install mlxlearn
```

`0.1.0a1` ships as a pure-Python wheel — no compiler, no build step, no ABI matrix.

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

"Never slower than scikit-learn" is the design goal. The shipped *gate* is measurable: on
each benchmarked workload class, patched dispatch shows no statistically significant
regression against stock scikit-learn. Crossover points are measured per algorithm and per
operation, published in [`docs/benchmarks.md`](docs/benchmarks.md), and wired into the
dispatch thresholds — small problems are handed to scikit-learn on purpose.

What that means in practice for 0.1.0a1, on an M4 Max:

| | |
|---|---|
| **Neighbor queries** | **2.9×–17×** from ~250 samples up. This is the reason to use mlxlearn. |
| Neighbor `fit` | 0.17×–0.36×. mlxlearn uploads to the device; scikit-learn stores a reference. The first query repays it several times over. |
| `SVC` | 1.20× at 4 000 × 32, parity below. |
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
