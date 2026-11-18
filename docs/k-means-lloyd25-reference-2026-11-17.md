# E-means versus 25-iteration Lloyd on the 10M datasets

**INCOMPLETE: 8/10 accepted measurements.** Each configuration uses 10,000,000 rows, 65,536 centers, FP32 input and seed 10. These are single-seed, single-fit measurements, with no error bars.

Without a complete pair, and absent from every table and range below: DINO10B prefix.

The [matched-budget comparison](k-means-10m-comparison-2026-11-15.md) stops Lloyd at the e-means pass budget, so its runtimes are equal by construction. Here Metal KMeans runs the paper's 25-iteration reference budget and Metal e-means keeps its own two- or three-epoch budget. Both start from identical explicit centers and share the assignment and statistics kernels.

Across the 4 paired datasets, 25-iteration Metal Lloyd took 6.50× to 9.27× the time of e-means with final labels in both timers, and 8.30× to 13.34× on training alone. E-means test MSE relative to Lloyd ranged from -0.271% to +0.181%; it was equal or lower on 2/4. Lloyd did not reach the e-means training inertia within its budget on 1/4.

| Dataset | Lloyd budget / actual | E-means epochs | Lloyd (s) | E-means (s) | Lloyd / e-means | E-means test MSE vs Lloyd |
|---|---:|---:|---:|---:|---:|---:|
| DEEP | 25 / 25 | 2 | 846.733 | 101.232 | 8.36× | +0.181% |
| SPACEV | 25 / 25 | 2 | 897.972 | 102.488 | 8.76× | +0.096% |
| SimSearchNet++ | 25 / 25 | 2 | 1553.671 | 167.524 | 9.27× | -0.189% |
| SIFT | 25 / 25 | 3 | 926.554 | 142.653 | 6.50× | -0.271% |

Times include training, fresh final labels and inertia. A ratio above one means e-means is faster. A negative MSE change means e-means has lower held-out error.

## Training time and inertia crossing

| Dataset | Lloyd training (s) | Lloyd s / iteration, median [min, max] | E-means training (s) | Training ratio | Lloyd updates to reach e-means inertia | Matched training ratio |
|---|---:|---:|---:|---:|---:|---:|
| DEEP | 814.137 | 32.64 [31.22, 34.56] | 67.494 | 12.06× | 17 | 8.26× |
| SPACEV | 864.722 | 34.50 [33.30, 38.44] | 68.395 | 12.64× | 16 | 8.23× |
| SimSearchNet++ | 1495.536 | 58.98 [57.14, 65.20] | 112.069 | 13.34× | 16 | 8.64× |
| SIFT | 890.992 | 35.64 [35.40, 35.91] | 107.292 | 8.30× | not reached in 25 | > 8.30× |

Training time excludes the final label pass. Every Lloyd iteration does the same work, so the spread of its per-iteration times shows thermal drift or other load during the fit. The crossing is the smallest number of Lloyd updates whose full-training FP32 inertia is at or below the e-means final inertia. Inertia after u updates is the pre-update inertia of iteration u+1, or the final inertia once Lloyd stops. This is a training-inertia diagnostic, not the paper's per-seed test-MSE crossing search. Where Lloyd never reaches it, the full training ratio is a lower bound.

## Quality

| Dataset | Lloyd test MSE | E-means test MSE | Lloyd train-sample MSE | E-means train-sample MSE | E-means training inertia vs Lloyd | Fit footprint GiB L / E |
|---|---:|---:|---:|---:|---:|---:|
| DEEP | 0.34178449 | 0.34240326 | 0.32968122 | 0.32995314 | +0.196% | 7.031 / 5.873 |
| SPACEV | 6375.07177305 | 6381.19034911 | 6265.33333051 | 6282.25833348 | +0.240% | 7.170 / 6.913 |
| SimSearchNet++ | 109532.81946912 | 109326.02122924 | 107076.43205408 | 107188.29906854 | +0.088% | 13.525 / 13.507 |
| SIFT | 0.19345382 | 0.19292923 | 0.17459797 | 0.17447211 | -0.268% | 8.076 / 8.200 |

Test and train-sample MSE use shared FP32 Faiss assignment and direct FP64 residuals on the test set and a fixed 10,000-row training sample. The inertia column compares each method's own FP32 final inertia over all training rows; a negative change means e-means fits the training set more tightly.

## Protocol and limits

Metal KMeans: max_iter=25 with tol=0 and n_init=1; strict convergence can stop earlier and actual iterations are recorded. Metal e-means: the frozen protocol's integer epoch budgets 2,2,2,2,3 with the Faiss quality batch law.

Datasets, identities, k, seed 10, explicit shared initialization, FP32 precision, full untimed warmup, timed final labels and inertia, memory sampling, thread settings and shared quality scoring are those of the frozen protocol and its unmodified harness (kmeans_compare.py).

- One seed and one timed fit per method; no confidence intervals.
- The e-means budget is fixed in advance, not searched for the first budget that beats Lloyd at 25.
- Lloyd here shares the e-means assignment and statistics kernels in FP32. It is a same-device control, not Flash-KMeans, Faiss GPU or an H100 measurement.
- A 25-iteration fit runs for minutes to hours after an equally long warmup; thermal state can differ from the short e-means fit.

## Evidence

[Addendum protocol](../benchmarks/e_means/kmeans_lloyd25_protocol.json); [frozen protocol](../benchmarks/e_means/kmeans_protocol.json); [validation and raw hashes](../benchmarks/e_means/results/kmeans_lloyd25/validation.json); [paired CSV](../benchmarks/e_means/results/kmeans_lloyd25/summary.csv).

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
| DEEP | [JSON](../benchmarks/e_means/results/kmeans_lloyd25/deep1b-10M-metal-kmeans-s10-e85200ffbca1.json) | [JSON](../benchmarks/e_means/results/kmeans_lloyd25/deep1b-10M-metal-emeans-s10-230ae451918a.json) |
| SPACEV | [JSON](../benchmarks/e_means/results/kmeans_lloyd25/spacev-10M-metal-kmeans-s10-aab6dd256bff.json) | [JSON](../benchmarks/e_means/results/kmeans_lloyd25/spacev-10M-metal-emeans-s10-225226ff7495.json) |
| SimSearchNet++ | [JSON](../benchmarks/e_means/results/kmeans_lloyd25/fb-ssnpp-10M-metal-kmeans-s10-365c5d70ba69.json) | [JSON](../benchmarks/e_means/results/kmeans_lloyd25/fb-ssnpp-10M-metal-emeans-s10-95e72cc0ab04.json) |
| SIFT | [JSON](../benchmarks/e_means/results/kmeans_lloyd25/sift1b-10M-div512-metal-kmeans-s10-b543fb9853b8.json) | [JSON](../benchmarks/e_means/results/kmeans_lloyd25/sift1b-10M-div512-metal-emeans-s10-0568023d12e1.json) |

**Incomplete campaign:** missing or rejected records are listed in the validation manifest. Partial tables do not establish full five-dataset coverage.
