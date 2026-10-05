# Support matrix

## Platform

mlxlearn requires **macOS on Apple silicon (arm64)**. MLX is the compute backend and there
is no meaningful mlxlearn without it. The package will import on other platforms and its
CPU path will run, but nothing there is tested or supported, and no bug filed against it
will be treated as a bug.

CI runs on GitHub's Apple silicon runners with a working Metal device, and requires one:
the workflows set `MLXLEARN_REQUIRE_GPU=1`, under which the test suite refuses to start
without a GPU instead of skipping the tests that need it (`tests/_gpu.py`). On a machine
without one, a plain `pytest` skips those tests, each with a reason.

### Processes and `fork`

MLX is not fork-safe. Once a process has loaded MLX or scikit-learn's estimators — the
first `fit` or `predict`, or `patch_sklearn()` — MLX in a child created by `fork()`
without `exec` crashes on its first operation, on the CPU device too. mlxlearn detects
such a child and refuses instead: a directly imported estimator raises `DeviceError`, and
a patched one hands the fit to scikit-learn (diagnostics reason `"forked-process"`). It
cannot detect every unsafe case — numerical NumPy work in the parent can have the same
effect — so the rule stands. Supported: the `spawn` start method (the macOS default),
`forkserver`, and scikit-learn's `n_jobs` (joblib's `loky` workers are spawned). Not
supported: an explicit `fork` context, `os.fork()`, Celery's prefork pool, and gunicorn
`--preload` of an app that has already used mlxlearn.

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
| 3.14 | 1.9.x | ✅ | unpinned: the newest scikit-learn the declared range admits (1.9.1 today) |

The other cells pin their scikit-learn release. The 3.14 cell does not, so a pull request
that raises the declared cap — a Dependabot one included — runs the suite on the release it
admits.

Two more jobs test the edges of the declared ranges rather than points inside them:

- **Minimum versions** (every PR). Python 3.10 with every floor `pyproject.toml` declares —
  NumPy 1.23.5, scikit-learn 1.7.0, MLX 0.31.2, and the test extra's, pandas 2.2.3 among
  them — installed exactly from `.github/constraints/minimum-versions.txt`, then checked
  to be what was installed. A floor nobody installs is a guess; this leg is what makes
  each one a tested claim.
- **Upstream pre-releases** (weekly, `.github/workflows/upstream-prereleases.yml`). The
  next scikit-learn and NumPy from the scientific-python nightly wheels, and any MLX
  pre-release, installed past the declared caps on purpose. It gates nothing; it is the
  early warning that says whether lifting a cap will be a version bump or a porting job.

`pyproject.toml` declares `scikit-learn>=1.7,<1.10`. The upper bound is deliberate: mlxlearn
subclasses scikit-learn estimators, so a major change to their constructor signatures or
fitted-attribute contracts is a compatibility event that needs a test run, not an
optimistic `>=`.

### Parameters scikit-learn is deprecating mid-range

Deprecations land as **string sentinel defaults**, which is a trap for any code that reads
the attribute directly — a truthy string arrives where a bool was expected, or an unknown
value where one of a fixed set was. Three of these fall inside the declared range, and
they do not all land in the same release:

| Parameter | 1.7 | 1.8 | 1.9 | How mlxlearn handles it |
|---|---|---|---|---|
| `LogisticRegression.multi_class` | present, default sentinel `"deprecated"` | **removed** | removed | read via `hasattr`; checked only where it exists |
| `LogisticRegression.penalty` | real, default `"l2"`; `l1_ratio` defaults to `None` | sentinel `"deprecated"`; the penalty comes from `l1_ratio` (default `0.0`), and `C=inf` means unpenalized | as 1.8 | `_effective_penalty()` resolves the same way the installed version does, and capability is decided on the resolved value |
| `SVC.probability` | real, default `False` | real, default `False` | sentinel `"deprecated"` | the sentinel is treated as `False`; an explicit `True` is still refused |

All three broke on 1.9 — two of them had already changed in 1.8 — and were found by
installing the built wheel into a clean environment, which resolved 1.9 while development
ran on 1.7. `tests/unit/test_sklearn_version_compat.py` now covers them against whichever
version is installed rather than pinning the problem away. On the MLX path mlxlearn emits
the same fit-time `FutureWarning`s scikit-learn's own `fit` would for these parameters, so
the advance notice of a removal does not depend on which path ran.

## MLX

`mlx>=0.31.2,<1.0`. 0.31.2 is the first release on which fitting from several threads at
once does not abort the process: `test_concurrent_fits_match_sequential_fits` in
`tests/unit/test_invariants.py` crashed on every earlier release tried (0.29.0, 0.29.4,
0.30.6, 0.31.0 and 0.31.1) and passes on 0.31.2. The floor is
installed and checked in CI (the minimum-versions job) as well as declared in the metadata,
so a resolver that quietly picks an older wheel fails loudly.

Development is on MLX 0.32.

## NumPy

`numpy>=1.23.5`. Both NumPy 1.x and 2.x work, and the minimum-versions job holds that to
the floor. It is not automatic: NumPy 1.x keeps `float64 scalar * float32 array` in
float32 where NumPy 2 promotes it, which once cost the SVC solver's float64 gradient its
precision on 1.x only. Mixed-precision arithmetic in the CPU paths states its dtype.

## Estimators

"Falls back for" is what a directly imported (Layer 1) class refuses with a precise
`CapabilityError`, and what a patched (Layer 2) class hands to scikit-learn — recorded in
diagnostics, warned about once, and subject to `fallback_policy`. Every estimator also
does this for sparse input (a SciPy sparse matrix, or a pandas frame whose every column is
sparse) and for float64 values beyond float32's range
([`fp32_policy.md`](fp32_policy.md) §1).

| Estimator | Since | Accelerated path | Falls back for |
|---|---|---|---|
| `neighbors.NearestNeighbors` | 0.1.0 | exact brute-force search, `metric` in `euclidean`, `l2`, `sqeuclidean`, or `minkowski` at `p=2`; `metric_params={"p": ...}` resolved as scikit-learn resolves it | any other metric, including `minkowski` at `p != 2`; `algorithm="kd_tree"`/`"ball_tree"`; `metric_params` other than Minkowski's `p` (a weight vector `w`, another metric's parameters); a prebuilt `KDTree`, `BallTree` or fitted estimator passed as `X`; training data spanning more than `FLOAT32_MAX / 8` from its mean, or so far beyond the gaps between its rows that float32 cannot resolve both (a fill value of `1e30` among unit-scale rows); more than 2³² training rows |
| `neighbors.KNeighborsClassifier` | 0.1.0 | as above; multi-output `y` included, voted per output; callable `weights` | as above |
| `neighbors.KNeighborsRegressor` | 0.1.0 | as above; multi-output `y` included, averaged per output; callable `weights` | as above |
| `linear_model.LogisticRegression` | 0.1.0 | L-BFGS, L2 / unpenalized, binary and multinomial, with a dense zero-start-curvature preconditioner on correlated features ([`fp32_policy.md`](fp32_policy.md) §2.3); scikit-learn 1.9 callbacks (`set_callbacks`) are driven on this path | `l1`/`elasticnet` however spelled; non-`lbfgs` solvers; `dual=True`; `warm_start=True`; `multi_class` other than `"deprecated"`/`"auto"` (1.7 only); `intercept_scaling != 1` with `fit_intercept=True`; multi-label or multi-output `y`; negative `sample_weight`; under `patch_sklearn()` on scikit-learn 1.9, `array_api_dispatch=True` |
| `svm.SVC` | 0.1.0 | exact SMO, linear/poly/rbf/sigmoid, one-vs-one; `shrinking` honored by the host solver; on the GPU, a multiclass rbf fit solves every one-vs-one pair at once in mlxlearn's Metal kernel, each pair verified on the host in float64 and finished there if it fails ([`fp32_policy.md`](fp32_policy.md) §2.5) | `kernel="precomputed"`, callable kernels, `probability=True`; `gamma=0.0` with a non-linear kernel; a `gamma` or `coef0` float32 cannot represent once the data is mapped into its compute frame; a linear-kernel row whose squared norm overflows float32; a linear or poly kernel on data whose variation float32's inner product cannot resolve — so far from the origin, for its spread, that its rounding swamps the differences between rows, or rows that all round, in float32, to the same nonzero row (reason `"float32-cancellation"`); a class whose every sample has zero effective weight (reason `"empty-class"`) |

A few inputs that look like capability questions are served directly. `SVC` treats a
negative `sample_weight` as a zero weight, which is exactly what LIBSVM does, and a
sample with no effective weight is dropped as LIBSVM drops it — unless that drops a whole
class, which is the `"empty-class"` fallback above: scikit-learn fits the classes that
remain. `SVC(kernel="linear", gamma=0.0)` fits, since the linear kernel never reads
`gamma`.

**Queries are not re-dispatched.** Inference runs on the backend `fit` chose (the sticky
backend in the README). A query float32 cannot represent is therefore an error at
inference in both layers, not a fallback — for the neighbor estimators and `SVC`, which
evaluate queries in float32; `LogisticRegression` predicts with scikit-learn's float64
code. The one exception: under `patch_sklearn()`, a sparse query to `kneighbors`,
`kneighbors_graph`, `predict` or `predict_proba` of a neighbor estimator fitted on MLX is
answered by scikit-learn's brute-force search on the fitted state, with the query cast to
float32 to match the training copy, whose rounding its distances carry; it is recorded as
a fallback and honors `fallback_policy`. Directly imported, it raises `SparseInputError`.
Where the fitted backend's arithmetic runs is a separate matter: an `SVC` fitted on MLX
evaluates a batch of up to 2²⁵ rows × support vectors × features on the host, in NumPy
float32 with the device's own arithmetic, because an idle GPU costs a small batch more
than its work (`compute="numpy-float32"` on the call's event).

`radius_neighbors` and `radius_neighbors_graph` are not accelerated. On an MLX-fitted
`NearestNeighbors` they run scikit-learn's own code at stock speed — a float32 query
against the float32 training copy — so their distances carry the float32 rounding of the
training data at its raw magnitude — unlike `kneighbors`, they do not translate float64
input by its mean, so a large constant offset costs them precision
([`fp32_policy.md`](fp32_policy.md) §3.4) — and a float64 query float32 cannot hold raises
`UnsupportedInputError`, as `kneighbors` does. They are host code, so they need no GPU on
a GPU-fitted model, and they take a sparse query in both layers, as scikit-learn does,
recorded as an ordinary call rather than a fallback. Stated here rather than left for a
profiler to reveal.

### Handed to scikit-learn by shape (patched only)

Separately from capability, `patch_sklearn()` hands problems to scikit-learn on purpose
where scikit-learn is faster: small ones, and — depending on the estimator — narrow or
wide ones. These are decided from the input's shape before any scan or copy of `X` —
`SVC`'s many-class rule from the number of distinct raw labels in `y` too, and
`LogisticRegression`'s binary rule from whether the raw labels hold two values, whatever
their type (strings and a pandas `Series` of class names included) — recorded with reason
`"below-crossover"` (or `"tree-wins"`), and never warned about. Some rules read more than
the shape, once the shape rules have passed a fit to MLX: `SVC` reads the kernel, C and
`cache_size`, and after validation the class and sample weights that scale C; a neighbor
fit reads its `n_neighbors`, and probes its own rows once its index is built (reason
`"data-probe"`), and a neighbor query reads the `n_neighbors` it asks for; and
`LogisticRegression` counts the classes and reads `tol` and `max_iter`. The thresholds
are private `Tuning` values, or scikit-learn's own defaults where a rule reads one, each
calibrated on a table in [`benchmarks.md`](benchmarks.md) that
`python -m benchmarks.calibrate` regenerates; the `Tuning` values are overridable with
`MLXLEARN_*` environment variables. The rules hold on the data those tables measured;
[`benchmarks.md`](benchmarks.md#known-limits-of-patched-dispatch) says what that covers,
and what it cannot.

| Estimator | Goes to scikit-learn when |
|---|---|
| neighbor estimators | fewer than `knn_min_train_samples` rows **or** fewer than `knn_min_work` rows × features |
| neighbor estimators | fewer than `knn_min_features` features, at any size: to scikit-learn's KD-tree where `algorithm="auto"` would build one (at most 15 features, `n_neighbors < n_samples // 2`, a KD-tree metric; reason `"tree-wins"`), to its brute force otherwise |
| neighbor estimators | fewer than `knn_wide_features` features **and** fewer than `knn_narrow_min_samples` rows: mid-size query batches lose there once the GPU has idled |
| neighbor estimators | `knn_wide_features` features or more **and** fewer than `knn_wide_min_samples` rows: an MLX call made right after a stock scikit-learn one runs some mid-size batches up to twice as slow, and stock brute force answers them about as fast on so few rows |
| neighbor estimators | `n_neighbors` above `knn_max_neighbors`: MLX pays per query for every neighbor it keeps. The constructor's `n_neighbors` decides the fit; a query on an MLX-fitted model that asks for more — `kneighbors(n_neighbors=...)`, or `n_neighbors` raised with `set_params` — is answered by scikit-learn's brute force over a float64 copy of the training rows, translated by their mean as the MLX search is, built on the first such query and kept with the model (not pickled); that first query pays for the copy, `8 · n_samples · n_features` bytes (25–85 ms at a million rows and 64 features) |
| neighbor estimators | data the first float32 search pass cannot certify — more than `knn_probe_max_uncertified` (none, by default) of `knn_probe_rows` (256) training rows probed at fit — or integer-valued data with fewer than `knn_wide_features` features (reason `"data-probe"`): duplicated rows, clusters or a column on a far larger scale than the rest send a query row to a slower second search, and the batch waits for it, so a few such rows in a hundred are enough |
| `LogisticRegression` | fewer than `logreg_min_samples` rows; or a multinomial fit with fewer than `logreg_multinomial_min_features` features; or a binary fit with fewer than `logreg_binary_min_samples` rows **or** more than `logreg_binary_max_features` features: a fit that converges in an iteration or two, as labels without signal do, is mostly MLX's fixed cost |
| `LogisticRegression` | `tol` above scikit-learn's default (1e-4), or `max_iter` below `logreg_min_max_iter`, at any size: they stop a fit within a few iterations too (read from the parameters, before any scan of `X`) |
| `SVC` | `kernel="linear"` or `kernel="sigmoid"`, at any size; fewer than `svc_min_features` features, `svc_min_samples` rows **or** `svc_min_work` rows × features; or, with more than two classes, a mean one-vs-one pair `(2n/K)·n_features` below `svc_min_pair_work` |
| `SVC` | any kernel but rbf — `kernel="poly"` too, at any size and any degree; an rbf fit with a mean box above `svc_max_c`: C times the mean class and sample weight over the samples kept — C itself without weights, with balanced class weights, or with sample weights normalized to mean 1 (compared with a relative tolerance of 1e-9, so that rounding cannot move a fit), read from the parameters before any scan of `X`; other weights after validation. How many SMO iterations a larger box costs depends on the data's intrinsic dimension, which a fit cannot see; and a `cache_size` below scikit-learn's default (200 MB), where rows are evaluated on demand |

`device` does not change any of this: `device="cpu"` selects MLX's CPU device for work
that is sent to MLX, not scikit-learn. A directly imported class never dispatches to
scikit-learn; below its crossover it runs on MLX's CPU device instead, and a directly
imported neighbor estimator always searches by brute force — on the GPU at any width once
it clears `knn_min_train_samples` and `knn_min_work`, since the width and data rules above are
patched-only — so on low-dimensional data stock scikit-learn's KD-tree is the faster
choice unless the queries come in large batches.
`MLXLEARN_FORCE_BACKEND` outranks every size rule; it is for tests and bisection.

### Deliberate differences from scikit-learn

- **`SVC(max_iter=-1)`** resolves to upstream LIBSVM's budget, `max(10⁷, 100·n)` per
  one-vs-one subproblem of `n` rows, rather than scikit-learn's unlimited. mlxlearn's SMO
  iterates in Python, and uncapped a fit scikit-learn finishes in a minute could run for
  hours. A fit that stops on the budget sets `fit_status_=1` and names the budget in its
  `ConvergenceWarning`.
- **`SVC.support_` with zero or negative weights** indexes the training data the caller
  passed, so `support_vectors_ == X[support_]` always holds. scikit-learn indexes the
  filtered problem instead.
- **`SVC(cache_size=...)`** can change a fit, because a materialized Gram and on-demand
  rows can round differently: on MLX's CPU device at any width (up to 2.2e-3 in the
  decision values at 2 000 × 10, and a different iteration count), on the GPU at a
  thousand features and more (up to 9.2e-4; bit-identical at 2 000 × 10). See
  [`fp32_policy.md`](fp32_policy.md) §2.4.
- **`SVC(shrinking=True)`**, the default, can move where inside `tol` the solver stops,
  on every kernel, as LIBSVM's shrinking does and by the same order — most at large `C` on
  the polynomial and linear kernels (up to 2.3e-2 in the decision values measured, poly at
  `C=10`, where LIBSVM's moved 3.2e-2). The support-vector count can change by one, and a
  training point that close to the boundary can change label. See
  [`fp32_policy.md`](fp32_policy.md) §2.4. On the GPU's one-vs-one kernel — a fit with
  three or more classes, the rbf kernel, and pairs of at most 4 096 rows whose Grams fit
  `cache_size` — the kernel and the host's finish of its pairs do not shrink, whatever
  `shrinking` says: shrinking is a speed heuristic, and the optimum and the full-set
  stopping test do not depend on it. A pair the host solver solves again from scratch
  there honors it, as on the host path. There `n_iter_` counts the kernel's steps per pair
  plus any the host took to finish it, and `cache_size` bounds how many pairs' Grams the
  GPU holds at once ([`fp32_policy.md`](fp32_policy.md) §2.5).
- **Neighbor ties** are decided by `(distance, training index)` over the whole training
  set, where scikit-learn's order among exactly equal distances is undefined
  ([`fp32_policy.md`](fp32_policy.md) §3.1).
- **Fitted attributes are float64** on the MLX path, even for float32 input
  ([`fp32_policy.md`](fp32_policy.md) §1 and §4, Rule B).
- **`n_jobs`, `leaf_size` and `verbose`** are accepted for signature compatibility and do
  nothing on the MLX path.

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
