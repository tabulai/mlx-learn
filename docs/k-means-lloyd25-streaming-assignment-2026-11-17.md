# E-means versus 25-iteration Lloyd on the 10M datasets

**INCOMPLETE: 8/10 accepted measurements.** Each configuration uses 10,000,000 rows, 65,536 centers, FP32 input and seed 10. These are single-seed, single-fit measurements, with no error bars.

Without a complete pair, and absent from every table and range below: DINO10B prefix.

The [matched-budget comparison](k-means-10m-comparison-2026-11-15.md) stops Lloyd at the e-means pass budget, so its runtimes are equal by construction. Here Metal KMeans runs the paper's 25-iteration reference budget and Metal e-means keeps its own two- or three-epoch budget. Both start from identical explicit centers and share the assignment and statistics kernels.

Across the 4 paired datasets, 25-iteration Metal Lloyd took 6.58× to 9.49× the time of e-means with final labels in both timers, and 8.42× to 13.67× on training alone. E-means test MSE relative to Lloyd ranged from -0.280% to +0.196%; it was equal or lower on 2/4. Lloyd did not reach the e-means training inertia within its budget on 1/4.

| Dataset | Lloyd budget / actual | E-means epochs | Lloyd (s) | E-means (s) | Lloyd / e-means | E-means test MSE vs Lloyd |
|---|---:|---:|---:|---:|---:|---:|
| DEEP | 25 / 25 | 2 | 545.595 | 63.329 | 8.62× | +0.196% |
| SPACEV | 25 / 25 | 2 | 672.730 | 70.870 | 9.49× | +0.096% |
| SimSearchNet++ | 25 / 25 | 2 | 1421.438 | 159.599 | 8.91× | -0.189% |
| SIFT | 25 / 25 | 3 | 718.550 | 109.223 | 6.58× | -0.280% |

Times include training, fresh final labels and inertia. A ratio above one means e-means is faster. A negative MSE change means e-means has lower held-out error.

## Training time and inertia crossing

| Dataset | Lloyd training (s) | Lloyd s / iteration, median [min, max] | E-means training (s) | Training ratio | Lloyd updates to reach e-means inertia | Matched training ratio |
|---|---:|---:|---:|---:|---:|---:|
| DEEP | 524.772 | 20.96 [20.66, 21.29] | 42.271 | 12.41× | 17 | 8.44× |
| SPACEV | 647.229 | 25.93 [24.51, 28.31] | 47.350 | 13.67× | 16 | 8.74× |
| SimSearchNet++ | 1365.168 | 54.50 [53.96, 56.39] | 106.733 | 12.79× | 16 | 8.17× |
| SIFT | 691.247 | 27.61 [27.32, 28.17] | 82.132 | 8.42× | not reached in 25 | > 8.42× |

Training time excludes the final label pass. Every Lloyd iteration does the same work, so the spread of its per-iteration times shows thermal drift or other load during the fit. The crossing is the smallest number of Lloyd updates whose full-training FP32 inertia is at or below the e-means final inertia. Inertia after u updates is the pre-update inertia of iteration u+1, or the final inertia once Lloyd stops. This is a training-inertia diagnostic, not the paper's per-seed test-MSE crossing search. Where Lloyd never reaches it, the full training ratio is a lower bound.

## Quality

| Dataset | Lloyd test MSE | E-means test MSE | Lloyd train-sample MSE | E-means train-sample MSE | E-means training inertia vs Lloyd | Fit footprint GiB L / E |
|---|---:|---:|---:|---:|---:|---:|
| DEEP | 0.34178449 | 0.34245585 | 0.32968122 | 0.33055500 | +0.198% | 4.840 / 4.541 |
| SPACEV | 6375.06305209 | 6381.19034911 | 6265.39971183 | 6282.25833348 | +0.240% | 4.876 / 4.657 |
| SimSearchNet++ | 109532.81946912 | 109326.02122924 | 107076.43205412 | 107188.29906854 | +0.088% | 11.171 / 11.176 |
| SIFT | 0.19347117 | 0.19292923 | 0.17461021 | 0.17447211 | -0.267% | 5.922 / 5.868 |

Test and train-sample MSE use shared FP32 Faiss assignment and direct FP64 residuals on the test set and a fixed 10,000-row training sample. The inertia column compares each method's own FP32 final inertia over all training rows; a negative change means e-means fits the training set more tightly.

## Change from the earlier campaign

The [earlier campaign](../benchmarks/e_means/results/kmeans_lloyd25/validation.json) used the same protocol, harness, data, initial centers and machine. 8 measured source files differ between the two: `src/mlxlearn/_kernels/distance.py`, `src/mlxlearn/_kernels/e_means.py`, `src/mlxlearn/_kernels/e_means_bf16.py`, `src/mlxlearn/_kernels/lloyd_bounds.py`, `src/mlxlearn/_solvers/block_smo.py`, `src/mlxlearn/cluster/_k_means.py`, `src/mlxlearn_device/_neighbors.py`, `src/mlxlearn_device/_neighbors_general.py`.

| Dataset | Lloyd-25 (s), earlier → now | Speedup | Lloyd s / iteration, earlier → now | Speedup | E-means (s), earlier → now | Speedup | Lloyd / e-means, earlier → now |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEEP | 846.7 → 545.6 | 1.55× | 32.64 → 20.96 | 1.56× | 101.2 → 63.3 | 1.60× | 8.36× → 8.62× |
| SPACEV | 898.0 → 672.7 | 1.33× | 34.50 → 25.93 | 1.33× | 102.5 → 70.9 | 1.45× | 8.76× → 9.49× |
| SimSearchNet++ | 1553.7 → 1421.4 | 1.09× | 58.98 → 54.50 | 1.08× | 167.5 → 159.6 | 1.05× | 9.27× → 8.91× |
| SIFT | 926.6 → 718.6 | 1.29× | 35.64 → 27.61 | 1.29× | 142.7 → 109.2 | 1.31× | 6.50× → 6.58× |

Each campaign is one timed fit per method, taken in a different session on the same machine, so these speedups combine the source change with any difference in machine state.

The [same-session control](../benchmarks/e_means/results/kmeans_lloyd25_gemm_replicate) measures the earlier sources again in the session of this campaign: the full e-means fit, and a Lloyd fit stopped after 3 of its identical iterations. Its speedups hold the machine state fixed.

| Dataset | E-means (s), earlier record | Earlier sources, this session | This campaign | Speedup | Lloyd s / iteration, earlier record | Earlier sources, this session | This campaign | Speedup | Control reproduces the earlier result bit for bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DEEP | 101.2 | 72.3 | 63.3 | 1.14× | 32.64 | 31.07 | 20.96 | 1.48× | Lloyd |
| SPACEV | 102.5 | 108.4 | 70.9 | 1.53× | 34.50 | 34.24 | 25.93 | 1.32× | e-means |
| SimSearchNet++ | 167.5 | 169.2 | 159.6 | 1.06× | 58.98 | 54.04 | 54.50 | 0.99× | e-means and Lloyd |
| SIFT | 142.7 | 141.5 | 109.2 | 1.30× | 35.64 | 35.03 | 27.61 | 1.27× | e-means and Lloyd |

The last column compares the control's e-means inertia and MSE, and the Lloyd inertia of each iteration it ran, with the earlier records.

| Dataset | Lloyd test MSE | Largest Lloyd training-inertia difference over the fit | Lloyd rows reused | E-means test MSE | Fit footprint GiB, Lloyd | Fit footprint GiB, e-means |
|---|---|---:|---:|---|---:|---:|
| DEEP | 0.34178449 → 0.34178449 (+2.5e-10%) | none in 26 states | 0 | 0.34240326 → 0.34245585 (+0.015%) | 7.031 → 4.840 | 5.873 → 4.541 |
| SPACEV | 6375.07177305 → 6375.06305209 (-0.00014%) | 1.5e-06 relative | 0 | 6381.19034911, identical | 7.170 → 4.876 | 6.913 → 4.657 |
| SimSearchNet++ | 109532.81946912, identical | none in 26 states | 0 | 109326.02122924, identical | 13.525 → 11.171 | 13.507 → 11.176 |
| SIFT | 0.19345382 → 0.19347117 (+0.009%) | 4.6e-06 relative | 0 | 0.19292923, identical | 8.076 → 5.922 | 8.200 → 5.868 |

A Lloyd state is the full-training FP32 inertia before each update and after the last. Rows reused counts training rows whose previous label was certified without a dense search, summed over the iterations.

## Protocol and limits

Metal KMeans: max_iter=25 with tol=0 and n_init=1; strict convergence can stop earlier and actual iterations are recorded. Metal e-means: the frozen protocol's integer epoch budgets 2,2,2,2,3 with the Faiss quality batch law.

Datasets, identities, k, seed 10, explicit shared initialization, FP32 precision, full untimed warmup, timed final labels and inertia, memory sampling, thread settings and shared quality scoring are those of the frozen protocol and its unmodified harness (kmeans_compare.py).

- One seed and one timed fit per method; no confidence intervals.
- The e-means budget is fixed in advance, not searched for the first budget that beats Lloyd at 25.
- Lloyd here shares the e-means assignment and statistics kernels in FP32. It is a same-device control, not Flash-KMeans, Faiss GPU or an H100 measurement.
- A 25-iteration fit runs for minutes to hours after an equally long warmup; thermal state can differ from the short e-means fit.

## Evidence

[Addendum protocol](../benchmarks/e_means/kmeans_lloyd25_protocol.json); [frozen protocol](../benchmarks/e_means/kmeans_protocol.json); [validation and raw hashes](../benchmarks/e_means/results/kmeans_lloyd25_streaming/validation.json); [paired CSV](../benchmarks/e_means/results/kmeans_lloyd25_streaming/summary.csv).

Recorded machine and software:

```json
{
  "hardware": {
    "cpu": "Apple M4 Max",
    "machine": "arm64",
    "platform": "macOS-26.6.2-arm64-arm-64bit-Mach-O",
    "ram_bytes": 137438953472
  },
  "thread_environment": {
    "OMP_NUM_THREADS": "8",
    "VECLIB_MAXIMUM_THREADS": null
  },
  "versions": {
    "faiss-cpu": "1.16.0",
    "h5py": "3.16.0",
    "mlx": "0.32.4",
    "numpy": "2.5.4",
    "python": "3.14.2",
    "scikit-learn": "1.9.1",
    "scipy": "1.18.1"
  }
}
```

| Dataset | Lloyd record | E-means record |
|---|---|---|
| DEEP | [JSON](../benchmarks/e_means/results/kmeans_lloyd25_streaming/deep1b-10M-metal-kmeans-s10-36c0c4d5f185.json) | [JSON](../benchmarks/e_means/results/kmeans_lloyd25_streaming/deep1b-10M-metal-emeans-s10-e3d418c0ac2f.json) |
| SPACEV | [JSON](../benchmarks/e_means/results/kmeans_lloyd25_streaming/spacev-10M-metal-kmeans-s10-2e1e58bdd903.json) | [JSON](../benchmarks/e_means/results/kmeans_lloyd25_streaming/spacev-10M-metal-emeans-s10-baa125657af0.json) |
| SimSearchNet++ | [JSON](../benchmarks/e_means/results/kmeans_lloyd25_streaming/fb-ssnpp-10M-metal-kmeans-s10-0a34857d9800.json) | [JSON](../benchmarks/e_means/results/kmeans_lloyd25_streaming/fb-ssnpp-10M-metal-emeans-s10-fc49df27fd8a.json) |
| SIFT | [JSON](../benchmarks/e_means/results/kmeans_lloyd25_streaming/sift1b-10M-div512-metal-kmeans-s10-bd640b14c66d.json) | [JSON](../benchmarks/e_means/results/kmeans_lloyd25_streaming/sift1b-10M-div512-metal-emeans-s10-add427d41a9c.json) |

**Incomplete campaign:** missing or rejected records are listed in the validation manifest. Partial tables do not establish full five-dataset coverage.
