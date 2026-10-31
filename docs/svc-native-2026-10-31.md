# Native SMO and fused SVC kernels — October 31, 2026

SVC now applies the same main ideas as the recent KNN and logistic-regression
work: move repeated numerical work out of Python, combine passes over the same
data, keep memory bounded, and preserve the precision contract. Complete fits
are **1.35–4.56× faster than the preceding implementation** across seven cases
and two dataset seeds on an Apple M4 Max. The optimizer alone is **4.07–4.99×
faster** on identical kernel matrices. Prediction performance is mixed, as
reported separately below.

## Implementation

- [`_smo_native.cpp`](../src/mlxlearn/_solvers/_smo_native.cpp) implements the
  exact C-SVC dual optimizer. Its gradient update also selects the next first
  working variable and measures the KKT bounds in one pass. Another pass
  selects the second variable by predicted objective improvement. There are no
  Python or NumPy calculations inside the iteration loop. A complete cached
  Gram enters native code once; a bounded native LRU returns to Python only
  when a missing row needs evaluation. Long fits remain interruptible.
- [`svc_fused.py`](../src/mlxlearn/_kernels/svc_fused.py) fuses the RBF
  norm/clamp/exponential, polynomial repeated squaring, or sigmoid transform
  after MLX's matrix multiplication. On a row-cache miss, one Metal launch
  computes both the row's dot products or direct squared distances and its
  nonlinear transform. Narrow rows share a SIMD group. RBF row distances use
  direct differences to avoid cancellation between squared norms.
- [`_svc_cpu_native.cpp`](../src/mlxlearn/_kernels/_svc_cpu_native.cpp) computes
  variance, column means, analytic kernel diagonals, decision reductions and
  multiclass votes. Small decision reductions use compensated native/NEON
  arithmetic. Larger ones call Accelerate GEMV/GEMM from native code, with
  **at most 4 MiB** of conversion scratch, rather than widening an unbounded
  kernel block. The scratch is not initialized before the conversion overwrites
  it. Bias subtraction is included in the reduction.
- RBF centering executes in MLX. Symmetric Gram construction reuses the
  resident design and its norms instead of uploading/reducing each left block
  again. The GPU row path no longer computes unused norms.

Kernel rows remain float32; alphas, gradients, analytic diagonals and optimizer
arithmetic remain float64. Last-index working-set ties, clipping, unequal box
bounds, the nonpositive-curvature floor, binary signs, one-vs-one coefficient
layout, and the distinct OVO/OVR zero-decision rules are preserved. The objective
has no approximation or subsampling. Final KKT diagnostics now describe the
returned iterate, including zero and exhausted iteration budgets, in both the
native and portable solvers.

NumPy is still used for scikit-learn validation, labels, weight/model assembly,
buffer allocation/conversion and public arrays. The optional native extensions
use the buffer protocol, not the NumPy C API. Installations without them retain
the portable numerical fallbacks. Broken binary imports are not silently hidden.
The optimizer runs on the CPU even when kernel evaluation uses the GPU; this
preserves its double-precision state.

## Measured fits

The baseline is the source-frozen implementation from commit `7e2bcf9`, including
its estimator, kernels and Python/NumPy solver. The harness verifies the
[archive manifest](../benchmarks/reference/svc-pre-native-2026-10-31.manifest.json)
and loads independent modules without replacing production globals.

Each run measures five complete fresh fits per implementation, alternating
order, and a 257-row prediction per fitted model. First calls are stored
separately, outside the medians; these are **not process-startup measurements**.
All device work is synchronized. Inputs are float32. The CPU/GPU selection in
the table is forced identically for the baseline and candidate. Accelerate's
thread limit and threadpoolctl's managed-pool limit are both set to one.

| Case | Train shape | Kernel device | Prior fit, seed 41 (ms) | Native fit, seed 41 (ms) | Fit speedup, seed 41 | Fit speedup, seed 42 |
|---|---:|---|---:|---:|---:|---:|
| RBF, narrow | 2,000 × 20 | CPU | 25.82 | 12.99 | 1.99× | 2.18× |
| RBF, wide | 4,000 × 256 | GPU | 116.94 | 58.19 | 2.01× | 2.04× |
| Linear | 1,200 × 20 | CPU | 1,127.04 | 254.65 | 4.43× | 4.56× |
| Polynomial | 2,000 × 32 | CPU | 50.24 | 18.85 | 2.67× | 2.66× |
| Sigmoid | 2,000 × 32 | CPU | 15.12 | 9.33 | 1.62× | 1.62× |
| RBF, four classes | 1,600 × 32 | GPU | 65.93 | 16.94 | 3.89× | 3.92× |
| RBF, 0.05 MiB cache | 1,600 × 32 | GPU | 354.96 | 263.80 | 1.35× | 1.36× |

All fits converged. Native predictions agreed with sklearn on **every held-out
query in both runs**. Decision-value checks against the preceding implementation
passed. Separate RBF and linear solver probes used the same float32 Gram bytes;
their alphas were bit-identical to the original solver, their iteration counts
matched, and independent equality, box, objective and KKT checks passed.

Environment: Apple M4 Max, macOS 26.6.2, Python 3.14.2, MLX 0.32.4,
NumPy 2.5.4, scikit-learn 1.9.1. The raw results include source and extension
hashes, every timing sample, first-call results, convergence diagnostics and
quality comparisons:

- [Seed 41](../benchmarks/results/svc-native-2026-10-31.json)
- [Seed 42](../benchmarks/results/svc-native-repeat-2026-10-31.json)

## Limits and prediction results

This is a fit-speed improvement, not a universal inference speedup. Prediction
speed ratios against the prior implementation range from **0.75× to 1.14×**.
The multiclass seed-41 case regressed from about 0.68 ms to 0.91 ms; the seed-42
case improved. Several other sub-millisecond predictions also regressed.
The earlier scalar-only native reduction was substantially slower; bounded
Accelerate reduction and removal of redundant scratch initialization avoided
that larger regression. Remaining differences are retained in the raw results.

Nor does faster fitting imply a win over sklearn in every case: seed-41 linear
still takes 254.65 ms versus sklearn's 39.27 ms, and the tiny-cache RBF case
takes 263.80 ms versus 37.50 ms. Wide RBF takes 58.19 ms versus 403.23 ms.
Automatic routing thresholds and the linear GPU exclusion therefore remain
unchanged. This small, single-machine panel does not establish new crossovers
across datasets or hardware.

Changing the cache budget can change float32 accumulation order. The direct
GPU row evaluator and the blocked Gram evaluator need not produce bit-identical
models. Similarly, query-block shapes can change MLX matmul/transcendental
rounding; blocking tests use numerical precision bounds, not a blanket bitwise
invariance claim. Sigmoid remains an indefinite, nonconvex kernel.

## Reproduction and validation

```bash
python setup.py build_ext --inplace
VECLIB_MAXIMUM_THREADS=1 python -m benchmarks.svc_acceleration \
  --repeats 5 --seed 41 --output benchmarks/results/svc-native-2026-10-31.json
VECLIB_MAXIMUM_THREADS=1 python -m benchmarks.svc_acceleration \
  --repeats 5 --seed 42 --output benchmarks/results/svc-native-repeat-2026-10-31.json
```

Run performance comparisons alone. The benchmark fails if its convergence,
prediction-agreement or solver-quality gates fail; it does not assert a speed
threshold. `--smoke` limits the estimator panel to narrow/wide RBF while still
running both same-Gram optimizer probes.

**303 tests passed.** Validation covers the existing SVC parity suite, native solver/kernel/arithmetic
tests, weighted multiclass and cache-mode integration, missing-extension
fallbacks, malformed native buffers, benchmark isolation/quality gates,
patching, invariants, and both SVC sklearn-compliance layers. See the
[test log](../benchmarks/results/svc-tests-2026-10-31.log).
Pandas-specific and optional array-API compliance checks were skipped by sklearn
in this environment. Ruff and the repository's mechanical compliance gates pass.

Source and wheel builds succeeded. Both native sources are present in the sdist;
the extracted wheel imports both extensions and passes CPU and GPU fit/predict
smoke checks using its own packaged code. See the
[package check](../benchmarks/results/svc-package-2026-10-31.json). CI now requires
both extensions to import after installation, so optional build failures cannot
silently turn the CI run into a portable-only test.
