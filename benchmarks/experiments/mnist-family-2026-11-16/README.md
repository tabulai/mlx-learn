# MNIST-family benchmarks on M1 Ultra, M4 Max and M5 Ultra, 2026-11-16

This reruns every benchmark from [`mnist-family-2026-11-02`](../mnist-family-2026-11-02/) on three
Apple Silicon generations:
- LogisticRegression, KNeighborsClassifier and SVC (`run.sh`);
- the 24-configuration PCA panel (`pca.py`);
- the exact-neighbor and SVC agreement audits.

The settings, scripts and datasets are the same as in that directory.

## Setup

All three machines ran commit `aa729e5`, which is v0.1.3 plus:
- the binary32 GEMM routing for M5-generation GPUs;
- the wide `svd_solver="full"` PCA fixes.

The stack was identical everywhere: Python 3.14.2, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1 and MLX 0.32.4. The compiled extensions were rebuilt from that commit, and their code sections are identical to the binaries used. Each machine ran `bench_all.sh` once, all three at the same time, with no other benchmark on any of them.

| Label | Chip | CPU cores (P+E) | GPU cores | Memory | macOS | GPU architecture | Wall time |
|---|---|---|---|---|---|---|---|
| M1 | M1 Ultra | 16+4 | 64 | 128 GB | 26.5.1 | `applegpu_g13d` | 46.6 min |
| M4 | M4 Max | 12+4 | 40 | 128 GB | 26.6.2 | `applegpu_g16s` | 31.0 min |
| M5 | M5 Ultra | 12+24 | 80 | 512 GB | 27.2 | `applegpu_g17d` | 22.5 min |

Background load, sampled every 30 s:
- **M1:** two unrelated single-threaded CPU-bound processes ran at 100% each throughout. A media-analysis daemon used about two cores for about 11 minutes, and the window server about half a core.
- **M4:** browser and window-server activity took 0.3-0.5 cores.
- **M5:** idle.

## Results

### Speedup over scikit-learn (MNIST / Fashion-MNIST / Kannada-MNIST)

mlxlearn gets float32 input. scikit-learn gets float64 input, its faster setting for kNN. For PCA, both get the same dtype.

| Benchmark | M1 Ultra | M4 Max | M5 Ultra |
|---|---|---|---|
| `LogisticRegression` fit, `max_iter=100` | 7.5× / 7.3× / 14.0× | 4.9× / 4.7× / 7.9× | 5.5× / 5.3× / 9.4× |
| `LogisticRegression` fit, `max_iter=5000` | 19.0× / 18.7× / 13.9× | 12.5× / 7.1× / 8.1× | 14.4× / 13.3× / 9.5× |
| `KNeighborsClassifier` predict (best of 7) | 16.1× / 14.8× / 16.1× | 11.6× / 12.2× / 11.8× | 14.5× / 13.2× / 14.4× |
| `SVC` fit, 200 MB cache | 19.6× / 25.5× / 20.9× | 13.0× / 17.4× / 14.9× | 15.4× / 19.6× / 15.9× |
| `SVC` fit, 2,000 MB cache | 27.9× / 41.2× / 28.7× | 21.2× / 29.8× / 22.5× | 18.1× / 26.0× / 18.4× |
| `SVC` predict | 217× / 178× / 223× | 298× / 286× / 326× | 315× / 277× / 320× |
| `PCA` fit, default solver, float32 | 1.7–1.9× | 1.8–2.0× | 1.7–1.8× |
| `PCA` fit, default solver, float64 | 2.9–3.1× | 2.1–2.5× | 2.2–2.7× |
| `PCA` fit, `randomized`, float32 | 8.1–12.2× | 4.0–9.6× | 5.6–9.5× |
| `PCA` fit, `randomized`, float64 | 10.8–19.2× | 7.3–12.4× | 6.3–13.5× |
| `PCA` 10,000-row test transform, float32 | 0.97–0.99× | 1.22–1.46× | 1.10–1.24× |
| `PCA` 10,000-row test transform, float64 | 1.01–1.05× | 1.11–1.24× | 1.16–1.23× |

The PCA rows give the range over the three datasets and 32 or 128 components. The transform rows cover the default solver on MNIST and Fashion-MNIST.

### Times (MNIST / Fashion-MNIST / Kannada-MNIST)

| | M1 Ultra | M4 Max | M5 Ultra |
|---|---|---|---|
| mlxlearn `LogisticRegression` fit, `max_iter=5000` | 0.33 / 0.90 / 0.18 s | 0.23 / 1.13 / 0.14 s | 0.20 / 0.57 / 0.11 s |
| mlxlearn `KNeighborsClassifier` predict (best) | 101 / 109 / 103 ms | 105 / 109 / 106 ms | 52 / 57 / 53 ms |
| mlxlearn `SVC` fit, 200 MB | 9.6 / 10.3 / 8.6 s | 8.8 / 9.4 / 7.7 s | 5.0 / 5.3 / 4.4 s |
| mlxlearn `SVC` fit, 2,000 MB | 6.8 / 6.5 / 6.3 s | 5.4 / 5.2 / 5.1 s | 4.2 / 4.0 / 3.9 s |
| mlxlearn `SVC` predict | 0.17 / 0.34 / 0.15 s | 0.09 / 0.15 / 0.07 s | 0.07 / 0.13 / 0.06 s |
| scikit-learn `KNeighborsClassifier` predict (float64, best) | 1.62 / 1.62 / 1.65 s | 1.22 / 1.33 / 1.25 s | 0.75 / 0.75 / 0.76 s |
| scikit-learn `SVC` fit, 200 MB | 189 / 263 / 180 s | 114 / 163 / 115 s | 77 / 103 / 70 s |

`python compare.py` prints these tables and the per-variant details. The details include iterations, objectives, the median and best of every timing, all 24 PCA configurations with principal angles, and the M5 comparison with the pre-binary32 PCA code.

### Correctness on every machine

- **`KNeighborsClassifier`:** all four variants (mlxlearn and scikit-learn, float32 and float64 input) return exact, correctly ordered neighbors for 100% of rows on all three datasets.
  - The audit compares against exact integer squared distances on the raw pixels.
  - Predictions are identical to scikit-learn's.
  - On Fashion-MNIST, 1-2 of 10,000 mlxlearn neighbor sets differ from scikit-learn's only by which of two equally distant rows is kept.
- **`SVC`:**
  - Predictions agree with scikit-learn on 99.98–100% of rows, and test accuracy matches to 0.01 pp.
  - Support-vector counts are within 7 of scikit-learn's.
  - 0.05–0.25% of rows have a pairwise vote with the opposite sign.
- **`LogisticRegression`:**
  - Every `max_iter=5000` fit converges.
  - Final objectives are within 0.35% of scikit-learn's, mlxlearn's usually lower, and test accuracy is within 0.15 pp.
  - L-BFGS iteration counts differ slightly between GPU generations (Fashion-MNIST: 281 / 313 / 309; Kannada-MNIST: 59 / 63 / 59), because float32 reduction order differs.
- **`PCA`:**
  - Held-out reconstruction RMSE agrees with scikit-learn's to 3.2×10⁻⁶ relative.
  - With float32 input, the default solver and 128 components, the largest principal angle against a float64 reference is 0.25° on the M1, 0.24° on the M4 and 0.017° on the M5. scikit-learn's own float32 result is 0.23–0.25°.

## Observations

- **The M5 Ultra is fastest in absolute terms.** Against the M4 Max it is 2× for kNN predict, 1.3–1.8× for SVC fit and 1.2–2× for logistic regression. scikit-learn also runs fastest on its 36 CPU cores, so the M5's speedups over scikit-learn sit between the other two machines'.
- **The M1 Ultra's speedups overstate its GPU advantage.** scikit-learn's CPU work there competed with the background load listed above, and mlxlearn's GPU-bound work barely did. mlxlearn's kNN times on the M1 match the M4's. Its SVC fits take 1.1–1.3× as long as the M4's, and its float32 default-solver PCA fits about 2×. The M1's PCA transforms are even with scikit-learn.
- **The binary32 GEMM fix costs some PCA fit time on the M5.** `pca.py` also times the PCA code at `1859e09`, which used MLX's default TF32 matrix products on that GPU. Against it, the current default-solver fits take 1.26–1.30× as long on MNIST. In exchange, the largest principal angle drops from 0.2–0.55° to 0.0005–0.006°. PCA on the M5 still fits 1.7–2.7× faster than scikit-learn. On the M4 the two versions have the same best fit times and give identical principal angles.
- **Some PCA medians are bimodal.** On the M4, the float32 `randomized` 32-component fits alternate between about 19 ms and about 33 ms. The slow mode comes from running right after scikit-learn, whose OpenMP workers keep spinning. Best times match the 2026-11-10 run: 19.3 vs 18.7 ms on MNIST. Kannada-MNIST's 5,000-row test transform takes about 1 ms and lands at 0.66–1.35× on every machine.
- **The M4 rerun agrees with the 2026-11-02 M4 results** for logistic regression, kNN and PCA within noise.
  - mlxlearn's SVC fits at the 200 MB cache are faster than recorded then: 8.8 / 9.4 / 7.7 s against 12.5 / 13.6 / 10.2 s.
  - Since `c928419`, the only change to code on that path is the binary32 routing, which does not apply on the M4. scikit-learn also ran 2–7% faster.
  - The earlier run recorded more background load.
- **One M4 file is a rerun.** The first M4 `max_iter=5000` MNIST fits overlapped file transfers to the other two machines and a malware scan of freshly copied binaries: mlxlearn took 0.29–0.45 s per fit. The benchmark was rerun alone, giving 0.232–0.248 s against scikit-learn's 2.92 s. `results/m4/mnist_logreg_iter5000.json.gz` is that rerun; the first run is kept in `results/m4/probes/`. `results/m4/run.log` shows the first run.

## Files

`results/m1`, `results/m4` and `results/m5` each hold:
- the gzip-compressed JSON files that `run.sh` and `pca.py` wrote, with the same names as in `mnist-family-2026-11-02/results`;
- `pca.json.gz`;
- the console logs (`run.log`, `pca.log`);
- the audit outputs (`knn_exact.log`, `svc_compare.log`);
- `env.txt`.

`gzip -d` restores the original bytes. The auditing scripts in `../mnist-family-2026-11-02` read these files directly.

## Reproduce

On each machine, from a checkout with the gitignored extensions built:

```bash
./bench_all.sh "$(git rev-parse --show-toplevel)" /path/to/python /tmp/mnist-rerun /path/to/datasets ~/.cache/mlxlearn/mnist-family
python compare.py M1=/path/to/m1 M4=/path/to/m4 M5=/path/to/m5
```

The dataset layout is described in the [2026-11-02 README](../mnist-family-2026-11-02/README.md#data). The audits can also be rerun on the committed files:

```bash
cd ../mnist-family-2026-11-02
python knn_exact.py ../mnist-family-2026-11-16/results/m5/mnist_knn_k5.json.gz
python svc_compare.py ../mnist-family-2026-11-16/results/m5/mnist_svc_*.json.gz
```
