# Historical baseline summary

- Source revision: `720053bdf9b7a377ee45bd5bd573bc09a7df1743`
- Hardware: `Apple M4 Max`
- Python: `3.13.3`
- scikit-learn: `1.7.2`
- MLX: `0.31.2`

## Commands

- **FAIL** `historical-pytest` — exit 1, 62.688s; see `test.log`.
- **PASS** `historical-fidelity-benchmark` — exit 0, 191.521s; see `benchmark.log`.

## Correctness tests

- `1 failed, 187 passed, 1 warning in 61.04s (0:01:01)`
- Historical failure: `tests/unit/test_runtime_diagnostics.py::test_pca_small_projection_prefers_sklearn_after_mps_fit`

## Fidelity benchmark

- Cells: 30
- Correct: 30
- Incorrect: 0
- Runtime errors: 0

### Observed speedup ranges

- `fidelity/linear_regression`: 3/3 correct; 2.972×–16.793×
- `fidelity/elastic_net`: 3/3 correct; 0.739×–1.909×
- `fidelity/lasso`: 3/3 correct; 0.680×–1.920×
- `fidelity/pca`: 3/3 correct; 0.584×–1.187×
- `fidelity/kmeans`: 3/3 correct; 0.791×–1.464×
- `fidelity/logistic_regression`: 3/3 correct; 1.062×–5.819×
- `fidelity/knn`: 3/3 correct; 0.404×–2.777×
- `fidelity/random_forest`: 3/3 correct; 1.025×–1.198×
- `fidelity/svm`: 2/2 correct; 0.085×–1.382×
- `fidelity/dbscan`: 2/2 correct; 3.175×–3.890×
- `fidelity/tsne`: 2/2 correct; 1.721×–3.338×

These are one-measure historical fidelity-smoke timings, not final per-operation performance gates or crossover thresholds.

A failing historical command is evidence to document, not a reason to rewrite the baseline. Re-run with `--require-green` only when using this as a strict regression gate.
