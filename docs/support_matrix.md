# Support matrix

## Platform

mlxlearn requires **macOS on Apple silicon (arm64)**. MLX is the compute backend and there
is no meaningful mlxlearn without it. The package will import on other platforms and its
CPU path will run, but nothing there is tested or supported, and no bug filed against it
will be treated as a bug.

## Python and scikit-learn

The matrix is **paired**, not a cross product. scikit-learn 1.9 requires Python ≥ 3.11, so
a naive cross product would schedule combinations that cannot resolve.

| Python | scikit-learn | Tested in CI | Notes |
|---|---|---|---|
| 3.10 | 1.7.2 | ✅ | oldest supported pair |
| 3.10 | 1.8.x | ✅ | |
| 3.10 | 1.9.x | ❌ | scikit-learn 1.9 requires Python ≥ 3.11 |
| 3.11 | 1.8.x | ✅ | |
| 3.11 | 1.9.x | supported, not in the CI matrix | covered by 3.12 + 1.9 |
| 3.12 | 1.9.x | ✅ | |
| 3.13 | 1.8.x | ✅ | |
| 3.13 | 1.9.x | ✅ | development pair |

`pyproject.toml` declares `scikit-learn>=1.7,<1.10`. The upper bound is deliberate: mlxlearn
subclasses scikit-learn estimators, so a major change to their constructor signatures or
fitted-attribute contracts is a compatibility event that needs a test run, not an
optimistic `>=`.

## MLX

`mlx>=0.29,<1.0`. The floor is asserted in CI at runtime as well as in the metadata, so a
resolver that quietly picks an older wheel fails loudly rather than producing subtly
different numerics.

Development is on MLX 0.31.2.

## NumPy

`numpy>=1.23.5`. Both NumPy 1.x and 2.x work.

## Estimators

| Estimator | Since | Accelerated path | Falls back for |
|---|---|---|---|
| `neighbors.NearestNeighbors` | 0.1.0 | exact brute-force Euclidean | non-Euclidean metrics, `kd_tree`/`ball_tree`, `metric_params`, sparse |
| `neighbors.KNeighborsClassifier` | 0.1.0 | as above | as above, plus callable `weights`, multi-output `y` |
| `neighbors.KNeighborsRegressor` | 0.1.0 | as above | as above |
| `linear_model.LogisticRegression` | 0.1.0 | L-BFGS, L2 / unpenalized | `l1`/`elasticnet`, non-`lbfgs` solvers, `dual=True`, sparse, `warm_start` |
| `svm.SVC` | 0.1.0 | exact SMO, linear/poly/rbf/sigmoid | `kernel="precomputed"`, callable kernels, `probability=True`, sparse |

`radius_neighbors` and `radius_neighbors_graph` are inherited from scikit-learn and run
unaccelerated on the CPU in float32. Correct, but not fast — stated here rather than left
for a profiler to reveal.

### Not shipped, and why

| | |
|---|---|
| `SVR`, `NuSVR`, `NuSVC` | Deferred. They will ship when they implement their true objectives. The ancestor patched `SVR` with a ridge surrogate, so `sklearn.svm.SVR` solved a different optimization problem than its name promises. |
| `LinearRegression`, `Ridge`, `PCA`, `Lasso`, `ElasticNet`, `KMeans`, `DBSCAN`, `TSNE` | Planned for 0.2.x, after `0.1.0a1` feedback. |
| Tree ensembles (`RandomForest*`, `ExtraTrees*`) | Out of scope. |
| `EmpiricalCovariance`, `DummyRegressor`, `LocalOutlierFactor` | Out of scope — the ancestor registered these as pass-throughs that added no acceleration. |

## Array boundary

0.1.0 is **NumPy in, NumPy out**, everywhere. `output_type="mlx"` and `output_type="input"`
are reserved for the 0.2.x experimental MLX boundary and are rejected today rather than
accepted and ignored.
