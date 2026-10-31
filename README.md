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

Direct KNN classes use sklearn's public estimator interface without inheriting
its concrete neighbor classes, avoiding those imports during ordinary startup.
`isinstance` against a stock neighbor class is therefore false for direct KNN
objects. Use their own graph methods; sklearn's standalone graph helpers do not
accept these objects as fitted-estimator inputs. Patched KNN classes retain
stock inheritance. See the [compatibility details](docs/support_matrix.md).

## Status

**`0.1.0a2` — alpha.** The API may change. See [CHANGELOG.md](CHANGELOG.md).

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

Python 3.10 is supported with scikit-learn 1.7.x; scikit-learn 1.8+ requires Python ≥ 3.11.
CI tests only valid pairs — see [`docs/support_matrix.md`](docs/support_matrix.md).

## Install

```bash
pip install mlx-learn
```

The released `0.1.0a2` wheel is pure Python. Development builds on Apple silicon
can include optional Metal and C++ CPU extensions. The CPU extension handles
small KNN workloads and is compiled at build time; fitting or querying never
invokes the C++ compiler. A compatible wheel includes the extensions built into it.
Without a compiler or CPU extension, the certified portable CPU implementation
remains available with the same ordering contract; native-build benchmark timings
do not describe that fallback's performance.

The CPU module uses the full CPython C API, so wheels containing it target a
specific Python minor version. The Metal component itself retains its stable
`abi3` interface. Installing a compatible prebuilt wheel needs no compiler.

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
| KNN with `device="auto"` | owned CPU/GPU search | owned CPU/GPU search |
| Other estimators below the measured crossover | internal CPU path | dispatches to scikit-learn |
| Invalid user input | sklearn-equivalent validation error | scikit-learn's own exception, unmasked |
| Unexpected backend runtime failure | raises | **raises** — never silently rerun on scikit-learn |

Set the policy with `fallback_policy="warn" | "raise" | "silent"`. Diagnostics are recorded
in every mode:

```python
import mlxlearn

with mlxlearn.config_context(fallback_policy="raise"):
    ...                                     # any capability fallback becomes an error

mlxlearn.get_last_backend_event()           # what the last fit/predict actually did
mlxlearn.get_backend_diagnostics()          # everything recorded this session
```

**Sticky fitted policy.** `execution_backend_` records the fit backend (`"mlx"`,
`"cpu"`, or `"sklearn"`). Inference reuses that backend, except for NumPy KNN
estimators fitted with `device="auto"`: their `execution_policy_` is `"auto"`,
allowing each query batch to use the owned CPU or GPU search. Small datasets and
single queries can avoid GPU launch overhead, while larger batches retain the GPU
path. `last_query_backend_` and backend diagnostics report the actual search.
Explicit KNN device choices use a `"fixed"` policy. Changing global configuration
after fit does not alter either fitted policy, and automatic CPU search does not
invoke a scikit-learn solver. A new `fit` clears state and may choose differently.

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

SVC uses native C++ hierarchical working sets and pairwise SMO with active-set
shrinking, preserving float64 optimizer state and float32 kernel rows. GPU kernel
transforms and requested row batches use fused Metal kernels. Prepared support
vectors and norms are reused for prediction with exact mutation checks. Native
CPU routines handle statistics, decision reductions and multiclass voting;
NumPy remains at the validation and public-array boundary, with portable
fallbacks if the optional extensions are unavailable. Automatic backend
thresholds remain conservative. See the
[working-set implementation and benchmarks](docs/svc-working-set-2026-10-31.md)
and the [earlier native-loop acceleration](docs/svc-native-2026-10-31.md).

KNN shares a NumPy-free GPU search core with `mlxlearn_device.fit_neighbors`.
Small neighbor counts use fused Metal distance/selection kernels, and GPU
classifier search and voting stay on the device until the final output. Automatic
NumPy KNN also has an owned CPU search and host voting for latency-sensitive
queries; the native device API retains explicit device execution. The sklearn API
retains its NumPy outputs and float64 regression target arithmetic. See the
[KNN implementation and benchmark report](docs/knn-pipeline-2026-10-28.md).
The [compact-search pass](docs/knn-v3-2026-10-28.md) reaches **2.10–3.02×
default-thread sklearn** across all 54 search/inference operation–dataset medians
in two runs on 50,000 training rows, 32 features, 256 queries and k=5. Fresh fit
plus query is included. Compact storage and bounded shortlists reduce search
work while exact refinement and fallback preserve the preceding float32 results.
The report includes every implemented benchmark operation, the previous version,
single-thread and default-thread sklearn, per-call variability and fit-only costs.
Earlier [acceleration](docs/knn-acceleration-2026-10-28.md) and
[packed-load](docs/knn-v2-2026-10-28.md) reports retain their original target misses
and distinguish intermittent calls from fully warmed operation.
The subsequent integer filter now uses a cached training-data guard to retain
compact search when feature variation is concentrated. The
[regression-fix report](docs/knn-scale-regression-fix-2026-10-29.md) compares both
Gaussian and unequal-scale data against v3, including fresh-fit costs and
identical-code controls. The guard does not rescale the requested metric.
A subsequent [input-validation optimization](docs/knn-fresh-fit-fix-2026-10-29.md)
removes the residual unequal-scale fresh fit-plus-query slowdown while retaining
the Gaussian gains; the report includes frozen-v3 reruns and validation checks.
The [precision and ordering fix](docs/knn-precision-fix-2026-10-30.md) resolves
rounded-distance ties and Gram candidate loss using certified screening and
adaptive original-coordinate comparisons. It includes the full v3 rerun and
retains benchmark failures in the comparison baselines.
The [128-feature optimization](docs/knn-128-feature-fix-2026-10-30.md) combines
shortlist refinement, ordering and certification while preserving the precision
fixes; it includes controlled comparisons with v3 and the prior precision build.
The [pre-fix broader sklearn comparison](docs/knn-broad-benchmark-2026-10-30.md) covers
72 dataset instances and nine search/inference operations. That GPU build
reaches 2× default-thread brute sklearn on **402/648 medians**, but the gain is
workload-dependent: large Gaussian and unequal-scale batches perform well,
while small datasets, clustered data and some offset cases miss the target.
The report includes single-thread and automatic-algorithm sklearn baselines,
every retained timing, and separate precision/tie audits.
The [clustered/offset fix](docs/knn-distribution-fix-2026-10-30.md) uses cached
distribution routing and guarded exact centering for screening. All 108 targeted
dataset-operation medians now beat all three sklearn configurations; the full
panel reaches 2× on **482/648** medians against default-thread brute and
**524/648** against auto. Gaussian gains and precision checks remain; the report
quantifies small fresh-fit costs and workloads that still favor sklearn.
The [Gaussian fresh-fit fix](docs/knn-gaussian-fresh-fix-2026-10-30.md) removes
the remaining 128-feature setup regression by evaluating compiled Gram
preparation with the first query. Fresh fit-plus-query is **1.20–1.28×** faster
than the pre-regression build across six controlled seeds; clustered/offset
gains and all precision checks hold. The full 72-instance sklearn rerun reaches
2× on **492/648** default-brute, **246/648** single-thread-brute and **528/648**
auto medians. All 108 clustered/offset medians beat all three configurations.
The preceding [small-workload GPU optimization](docs/knn-small-workload-fix-2026-10-30.md)
combines selection and precision certification in one launch and trims query
validation overhead. All 162 targeted controlled comparisons improve by paired
median (1.08–1.48× versus the preceding build). The full panel reaches 2× on
**526/648** default-brute, **252/648** single-thread-brute and **551/648** auto
medians, with all 14,472 accuracy audits passing. In that fixed-GPU run,
only **5/162** targeted medians beat single-thread sklearn.
The preceding [automatic CPU/GPU dispatch](docs/knn-auto-dispatch-2026-10-30.md)
introduced an owned, exact-order CPU search for suitable small and single-query
workloads. That run recorded **162/162** targeted wins against each sklearn
configuration, a narrowest win of **1.010×**, and **1,025** passing tests.

The preceding [latency fix](docs/knn-latency-fix-2026-10-30.md) added native
integer selection, faster feature-range preparation and deferred GPU probing.
That run improved all 27 warmed Digits cells **1.82–2.45×** over its predecessor,
but total process startup beat sklearn auto on only **7/18** fixtures.

The preceding [startup revision](docs/knn-startup-fix-2026-10-30.md) avoids loading
unused stock neighbor implementations during direct owned search and retains
the native CPU improvements. All **162** targeted and **108** clustered/offset
medians beat all three sklearn configurations. All **14,472** current broad
accuracy audits and **1,473** distinct checks pass: 1,408 relevant tests,
also rerun on sklearn 1.7, 1.8 and 1.9, plus 65 harness checks.

In the [450-process cold panel](docs/knn-startup-cold-2026-10-30.md), all 18
fixtures have **15–18% lower process-to-result medians** than the three sklearn
configurations, with imports and native loading included. All 18 also beat the
preceding automatic build by **1.181–1.213×**. First-query and fit-plus-query
medians beat sklearn on all 18, but improve over the preceding automatic build
on only **14/18** and **17/18**, respectively. The balanced implementation control
wins all 162 target cells against both GPU implementations; its legacy-order
repeat retains one Digits slowdown against the older GPU build: **2.1% by
median paired ratio and 3.7% by ratio of medians**. These
measured gains do not imply a universal 2× speedup or zero regressions.

The separate [2,700-fresh-model panel](docs/knn-startup-first-query-2026-10-30.md)
tests first use after normal process initialization. All 18 first-query and
fit-plus-first-query medians beat all three sklearn configurations and preceding
automatic. Against preceding automatic, the speedups are **1.019–1.234×** for
first query and **1.152–1.574×** for combined fit and first query.

The latest [Digits revision](docs/knn-digits-fix-2026-10-30.md) combines integer
dot validation and exact top-five selection in one native pass. It removes the
remaining Digits GPU-control miss: **432/432** comparisons across both GPU
baselines, replicas, statistics and control orderings are faster, with a
minimum **1.047×** speedup. The previously failing weighted-regression cell is
now **1.051× paired / 1.079× ratio-of-medians** versus historical GPU.
All **162** small/single-query, **108** clustered/offset and **81** explicit
Gaussian-distribution broad cells still beat all three sklearn settings.
All 18 startup and first-query fixture medians also beat those settings.

The revision passes **1,474** relevant tests on each of sklearn 1.7, 1.8 and
1.9, plus **65** harness checks and **14,472** current broad numerical audits.
It does not establish zero regression versus the preceding automatic build:
three Breast Cancer slowdowns recur in separate-worker controls, and some
small first-query medians remain slower. An additional exact-native A/B/A probe
did not reproduce the large Breast losses; all observations and that test's
limits are retained in the report. The 1.10× Digits margin target passes
46/48 checks, while all 432 above-parity checks pass.

```python
import mlx.core as mx
from mlxlearn_device import fit_neighbors

index = fit_neighbors(mx.array([[0., 0.], [1., 1.], [2., 2.]]))
distances, indices = index.query(mx.array([[0.2, 0.1]]), k=2)
mx.eval(distances, indices)  # results remain MLX arrays
```

Performance depends on training size, feature count, query batch, neighbor count
and data distribution. Automatic KNN selects its owned CPU or GPU search under
the policy captured at fit; explicit device choices remain fixed. The linked
reports include complete timings, variability and workloads where sklearn remains
faster. Other patched solvers can hand small problems to scikit-learn.

What that means for the measured development build on an M4 Max:

| | |
|---|---|
| **Neighbor queries/inference** | Wins **625/648**, **648/648** and **648/648** medians against default brute, single-thread brute and auto sklearn on the [full panel](docs/knn-digits-benchmarks-2026-10-30.md). The ≥2× counts are **587**, **300** and **592**, respectively. |
| Neighbor `fit` and startup | Owns training data and defers search preparation where appropriate. Fit-only, first fit-plus-query and total process startup are reported separately; query gains do not guarantee faster fit or startup. |
| `SVC` | Native working sets improve measured linear and RBF fits; tiny-cache RBF still trails sklearn and prediction gains vary. See the [current results and limitations](docs/svc-working-set-2026-10-31.md). Patched routing remains conservative, and linear is excluded from automatic GPU selection. |
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
python3 setup.py build_ext --inplace
pytest
```

Rebuild the extensions with `python3 setup.py build_ext --inplace` after changing
their C++ or Objective-C++ sources. The optional native CPU search uses bounded
float32 screening, float64 output distances, Neon on arm64, and exact 640-bit
dyadic keys for ambiguous ordering. The portable certified path remains available
when that extension was not built; 64-feature integer batches retain their exact
Gram path.

Reproducing native benchmark results from a source archive requires building its
extension or restoring the archived binary for the exact matching Python ABI,
operating system, and architecture. The source-only portable fallback remains
supported and tested for precision, but is not the measured native configuration.

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
