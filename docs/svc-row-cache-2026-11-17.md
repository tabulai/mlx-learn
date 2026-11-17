# SVC kernel-row caching

The native SVC solver now keeps previously evaluated kernel rows after they leave the working set. A training example can re-enter the working set several iterations later, or be needed when the solver reconstructs the full gradient to check convergence. Both operations can read the saved float32 row.

The change applies when a one-vs-one subproblem uses the native working-set solver and its Gram matrix exceeds the cache budget. For a budget of B rows and working-set width W, the estimator reserves W rows for the callback and retains up to B − W rows. Rows are allocated on demand. The solver protects all hits in the current batch before recycling the least recently requested rows. The budget covers the stored kernel values and callback result. Inputs, optimizer arrays, device scratch, allocator overhead, and cache metadata also contribute to process memory.

## End-to-end RBF fits

We compared main (`ccc84f6`) with the cache implementation (`42bb585`) on an Apple M4 Max with 128 GiB of memory, Python 3.14.2, MLX 0.32.4, and scikit-learn 1.9.1. Each version used a separately built native extension and its own Python process. Each pair used identical inputs, backend, and cache size in MiB. Timings include constructing a fresh public SVC estimator, fitting it and synchronizing MLX. Input generation, scoring, and row-fetch instrumentation took place outside the timed fits.

The synthetic binary workloads used float32 data, 257 holdout rows, and seven timed fits per version after a warmup. Fashion-MNIST used all 60,000 training images, float64 pixels scaled to [0, 1], ten classes, all 10,000 test images, and three timed fits after a warmup. Execution order alternated between versions. Times below are medians.

| Workload | Cache (MiB) | Main fit | New fit | Speedup | Rows fetched, main → new |
|---|---:|---:|---:|---:|---:|
| 1,600 × 32, CPU, seed 41 | 0.05 | 68.57 ms | 69.41 ms | 0.99× | 4,397 → 4,397 |
| 4,000 × 32, CPU, seed 41 | 4 | 48.78 ms | 43.59 ms | 1.12× | 5,649 → 4,529 |
| 4,000 × 32, CPU, seed 41 | 24 | 49.23 ms | 26.52 ms | 1.86× | 5,649 → 1,374 |
| 4,000 × 32, CPU, seed 42 | 24 | 53.21 ms | 29.07 ms | 1.83× | 6,133 → 1,450 |
| 4,000 × 32, CPU, seed 41 | 64 | 42.47 ms | 42.54 ms | 1.00× | 0 → 0 |
| 4,000 × 256, GPU, seed 41 | 24 | 116.13 ms | 75.87 ms | 1.53× | 8,852 → 3,283 |
| Fashion-MNIST, GPU | 200 | 10.294 s | 4.590 s | 2.24× | unmeasured |
| Fashion-MNIST, GPU | 2,000 | 5.139 s | 5.121 s | 1.00× | unmeasured |

At 24 MiB the two 4,000 × 32 fits evaluated 76% fewer kernel rows. At 0.05 MiB the solver had room for just four active rows plus four temporary rows, and both versions fetched exactly the same rows. At 64 MiB on the 4,000-row problem and at 2,000 MiB on Fashion-MNIST, the full Gram matrix fit for each binary subproblem. Those rows are controls for the path that bypasses the LRU. The small differences on the controls are within the variation of individual fits on this machine.

## Convergence and decision values

Both versions converged in every fit. Every binary subproblem’s full-gradient KKT residual was below the requested 0.001 tolerance, including all 45 pairs per Fashion-MNIST fit. An independent float64 reconstruction using direct squared distances and the fitted dual coefficients also satisfied that tolerance over all training rows for each synthetic case.

All held-out predictions and support-vector indices matched main. Six of the eight workloads had identical decision values, including all 450,000 one-vs-one margins per Fashion-MNIST fit. In the two 24 MiB CPU cases, the number of SMO iterations changed and the largest absolute decision difference was 0.000746. Both cases passed the project's [margin tolerances](fp32_policy.md) and the independent convergence check.

CPU cache misses are evaluated as a batch. In a separate probe on the two CPU fixtures, the same float32 kernel row evaluated alone versus among 64 rows differed by up to 3.28 × 10⁻⁷. A change in the miss batch size can therefore produce slightly different kernel values and a different sequence of SMO steps. The immutable-row regression test holds kernel values fixed and verifies identical coefficients, intercept, iteration count, and KKT residual across cache capacities.

The full repository suite passed with 4,281 tests and eight environment-dependent skips. Seven skips require pandas and one requires Python framework headers. Ruff and the repository’s mechanical compliance checks passed.

## Reproduce

From the candidate checkout, use the project environment that contains the build and runtime dependencies. The image runs also require `fashion_mnist.npz` in the local [MNIST-family dataset cache](../benchmarks/experiments/mnist-family-2026-11-02/README.md).

```sh
bench_dir=$(mktemp -d)
python_bin="$PWD/.venv/bin/python"
git worktree add --detach "$bench_dir/main" ccc84f6
(cd "$bench_dir/main" && "$python_bin" setup.py build_ext --inplace --force)
"$python_bin" setup.py build_ext --inplace --force
"$python_bin" -m benchmarks.svc_row_cache \
  --baseline "$bench_dir/main" --output "$bench_dir/results.json"
```

The [benchmark driver](../benchmarks/svc_row_cache.py) and [raw results](../benchmarks/results/svc-row-cache-2026-11-17.json) include the fixtures, source and binary identities, individual timings, row-fetch counts, and numerical checks. To reproduce the synthetic cases alone, add `--cases tiny cpu4 cpu24a cpu24b cpu64 gpu24`.
