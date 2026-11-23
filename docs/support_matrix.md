# Support matrix

## Platform

mlxlearn requires **macOS on Apple silicon (arm64)**. MLX is the compute backend and there
is no meaningful mlxlearn without it. The package will import on other platforms and its
CPU path will run, but nothing there is tested or supported, and no bug filed against it
will be treated as a bug.

## Python and scikit-learn

The matrix is **paired**, not a cross product. scikit-learn 1.8+ requires Python ≥ 3.11, so
a naive cross product would schedule combinations that cannot resolve.

| Python | scikit-learn | Tested in CI | Notes |
|---|---|---|---|
| 3.10 | 1.7.2 | ✅ | oldest supported pair |
| 3.10 | 1.8.x | ❌ | scikit-learn 1.8 requires Python ≥ 3.11 |
| 3.10 | 1.9.x | ❌ | scikit-learn 1.9 requires Python ≥ 3.11 |
| 3.11 | 1.8.x | ✅ | |
| 3.11 | 1.9.x | supported, not in the CI matrix | covered by 3.12 + 1.9 |
| 3.12 | 1.9.x | ✅ | |
| 3.13 | 1.8.x | ✅ | |
| 3.13 | 1.9.x | ✅ | development pair |

`pyproject.toml` declares `scikit-learn>=1.7,<1.10`. The upper bound is deliberate: mlxlearn
uses scikit-learn's estimator contracts, so a major change to constructor signatures or
fitted-attribute contracts is a compatibility event that needs a test run, not an
optimistic `>=`.

### Parameters scikit-learn is deprecating mid-range

Deprecations land as **string sentinel defaults**, which is a trap for any code that reads
the attribute directly — a truthy string arrives where a bool was expected, or an unknown
value where one of a fixed set was. Three of these fall inside the declared range:

| Parameter | 1.7 | 1.9 | How mlxlearn handles it |
|---|---|---|---|
| `LogisticRegression.multi_class` | present, deprecated | **removed** | read via `hasattr`; checked only where it exists |
| `LogisticRegression.penalty` | real, default `"l2"` | sentinel `"deprecated"`; the penalty comes from `l1_ratio`, and `C=inf` means unpenalized | `_effective_penalty()` resolves the same way the installed version does, and capability is decided on the resolved value |
| `SVC.probability` | real, default `False` | sentinel `"deprecated"` | the sentinel is treated as `False`; an explicit `True` is still refused |

All three broke on 1.9 and were found by installing the built wheel into a clean
environment, which resolved 1.9 while development ran on 1.7. `tests/unit/test_sklearn_version_compat.py`
now covers them against whichever version is installed rather than pinning the problem away.

## MLX

`mlx>=0.29,<1.0`. The floor is asserted in CI at runtime as well as in the metadata, so a
resolver that quietly picks an older wheel fails loudly rather than producing subtly
different numerics.

The latest release benchmarks use MLX 0.32.4.

## NumPy

`numpy>=1.23.5`. Both NumPy 1.x and 2.x work.

## Estimators

| Estimator | Since | Accelerated path | Falls back for |
|---|---|---|---|
| `neighbors.NearestNeighbors` | 0.1.0 | exact brute-force Euclidean | non-Euclidean metrics, `kd_tree`/`ball_tree`, `metric_params`, sparse |
| `neighbors.KNeighborsClassifier` | 0.1.0 | as above | as above, plus callable `weights`, multi-output `y` |
| `neighbors.KNeighborsRegressor` | 0.1.0 | as above | as above |
| `neighbors.RadiusNeighborsClassifier` | 0.1.7 | exact Euclidean radius search, class votes, uniform or distance weights | unsupported neighbor metrics and inputs, callable weights, multi-output targets |
| `neighbors.RadiusNeighborsRegressor` | 0.1.7 | exact Euclidean radius search, uniform or distance-weighted means | same as the radius classifier |
| `linear_model.LogisticRegression` | 0.1.0 | L-BFGS, L2 / unpenalized | `l1`/`elasticnet`, non-`lbfgs` solvers, `dual=True`, sparse, `warm_start` |
| `linear_model.LinearRegression` | 0.1.7 | dense host OLS with rank certification, multi-output and `sample_weight` | `positive=True`, sparse inputs, and automatic patched fits outside the [measured size and dtype ranges](linear-regression-2026-11-22.md#routing) |
| `svm.SVC` | 0.1.0 | exact SMO, linear/poly/rbf/sigmoid | `kernel="precomputed"`, callable kernels, `probability=True`, sparse |
| `decomposition.PCA` | 0.1.2 | dense covariance and randomized PCA, CPU/GPU projection | sparse input, values outside float32 range |

`radius_neighbors` and `radius_neighbors_graph` are exact for the float32 training
and query coordinates: a training row is returned iff its exact squared distance is at
most `fl64(radius**2)`. CPU and Metal searches apply the same membership rule.
Rows hold ascending training indices (or `(distance, index)` order with
`sort_results=True`) and are ragged object arrays of 1-d rows. Under `patch_sklearn()`, a
call goes to stock scikit-learn, configured as the user configured the estimator, when
it uses something the owned engines do not implement (sparse queries, per-query or
negative radii, values beyond float32, callable weights). Automatic routing can also
select stock when it predicts a faster search and certifies exact membership. Layer 1
densifies sparse queries and rejects values outside float32 range and non-scalar,
negative or NaN radii.

Direct KNN classes use public sklearn estimator mixins without inheriting stock
`sklearn.neighbors` classes. They preserve the estimator API, but an `isinstance`
check against a stock neighbor class captured before patching is false. This
startup-focused compatibility change avoids loading sklearn's neighbor stack
for owned Euclidean search; permitted fallback still loads it when needed.

This also affects APIs that identify a fitted neighbor object by concrete class:
use `model.kneighbors_graph(...)` or `model.radius_neighbors_graph(...)` on a
direct estimator. Passing that estimator as the data argument to sklearn's
standalone graph helper functions is not supported. Passing ordinary data to
those sklearn functions is unchanged.

The patched KNN variants retain stock inheritance, since patching already loads
the stock classes. Captured-stock `isinstance` checks and sklearn's fitted-model
graph helper inputs continue to work for those variants.

### Not shipped, and why

| | |
|---|---|
| `SVR`, `NuSVR`, `NuSVC` | Deferred. They will ship when they implement their true objectives. The ancestor patched `SVR` with a ridge surrogate, so `sklearn.svm.SVR` solved a different optimization problem than its name promises. |
| `Ridge`, `Lasso`, `ElasticNet`, `KMeans`, `DBSCAN`, `TSNE` | Planned for 0.2.x, after `0.1.0a1` feedback. |
| Tree ensembles (`RandomForest*`, `ExtraTrees*`) | Out of scope. |
| `EmpiricalCovariance`, `DummyRegressor`, `LocalOutlierFactor` | Out of scope — the ancestor registered these as pass-throughs that added no acceleration. |

## Array boundary

0.1.0 is **NumPy in, NumPy out**, everywhere. `output_type="mlx"` and `output_type="input"`
are reserved for the 0.2.x experimental MLX boundary and are rejected today rather than
accepted and ignored.
