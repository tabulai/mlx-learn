# MNIST-family benchmarks, 2026-11-02

Scripts and raw results behind the README's
[MNIST, Fashion-MNIST and Kannada-MNIST](../../../README.md#mnist-fashion-mnist-and-kannada-mnist)
section: `LogisticRegression`, `KNeighborsClassifier` and `SVC` from mlxlearn against
scikit-learn on three 60,000 × 784 datasets. Measured at `c928419` on an M4 Max (12+4 CPU
cores, 40-core GPU, 128 GB) with Python 3.14.2, scikit-learn 1.9.1, MLX 0.32.4 and NumPy
2.5.4.

## Data

The Kaggle CSV versions of the datasets, laid out as `<root>/<dataset>/input/*.csv.gz`.
MNIST and Fashion-MNIST train on `train.csv.gz` and validate on `test.csv.gz`.
Kannada-MNIST's `test.csv.gz` is unlabeled, so it trains on `train.csv.gz` and validates
on `Dig-MNIST.csv.gz`. Set `MNIST_FAMILY_ROOT` to `<root>`. The first load of each dataset
parses the CSVs and caches the pixels as uint8 under `MNIST_FAMILY_CACHE` (default
`~/.cache/mlxlearn/mnist-family`, about 55 MB per dataset). Every script scales pixels to
[0, 1].

PCA uses Kannada's unlabeled 5,000-row `test.csv.gz` as well as the 10,240-row
Dig-MNIST holdout. Its inputs contain pixel columns only.

## Scripts

| Script | What it does |
|---|---|
| `logreg.py` | Fits mlxlearn (float32 and float64 input) and scikit-learn (float64), after one untimed mlxlearn warm-up fit. Records fit, `predict` and `predict_proba` times, iterations, convergence, accuracy and per-row predictions. Also evaluates scikit-learn's multinomial objective and maximum gradient in float64 for both models. |
| `knn.py` | Same pattern for `KNeighborsClassifier`, with scikit-learn on both input dtypes. The first `predict` on a fresh fit is timed separately. Records `kneighbors` indices and distances for the audit. |
| `svc.py` | One library per process (`--lib mlx` or `--lib sklearn`), with `cache_size` from `--cache-size` and everything else at defaults (rbf, `C=1`, `gamma="scale"`). Records fit, `predict` and `decision_function` times, support-vector counts and per-row decision values. |
| `knn_exact.py` | Checks every KNN variant's neighbors against exact integer squared distances on the raw pixels. |
| `svc_compare.py` | Compares SVC runs with the scikit-learn run among the files given: prediction agreement, pairwise votes and summed one-vs-one confidences. It recovers these from scikit-learn's ovr decision value, `votes + s / (3 (|s| + 1))`. |
| `pca.py` | Compares scikit-learn, the committed pre-transform-fix baseline and the working tree for fit and train/test transforms. It records per-round timings, solver and backend choices, reconstruction accuracy, projection accuracy and source-file checksums. |
| `summarize.py` | Prints the Markdown tables the README section was written from. |
| `omeans.py` | Times `OMeans` against `EMeans` in interleaved rounds at one centroid count, with a full-search control and the public `fit`, then fits mlxlearn's `KMeans` and scikit-learn's `MiniBatchKMeans` and `KMeans` once each. Records per-round timings, held-out error, the share of centroids compared, thermal pressure and source-file checksums. |
| `omeans_summarize.py` | Prints the o-means tables below from the result files. |
| `run.sh` | Reruns every timed benchmark with the settings behind `results/`. |
| `knn_probe_order.py`, `knn_probe_gap.py` | Probes that traced KNN timing drift to run order and other processes' GPU work. Their output was not saved. |

## Results

`results/` holds every JSON file the scripts wrote, gzip-compressed. `gzip -d` restores the
original bytes. Names are `<dataset>_<estimator>_<variant>.json.gz`:

- `*_logreg_iter100` and `*_logreg_iter5000`: `max_iter=100` (the default) and 5000. Each
  has 5 fits and 30 calls each to `predict` and `predict_proba` per variant.
- `*_knn_k5`: `k=5`, 5 fits and 7 calls each to `predict` and `predict_proba` per variant.
- `*_svc_mlx` and `*_svc_mlx_c2000`: 3 fits and 3 predictions at `cache_size` 200 MB (the
  default) and 2,000 MB.
- `*_svc_sklearn` and `*_svc_sklearn_c2000`: one fit and one prediction at each cache
  size. Each fit takes about 2 to 3 minutes.
- `pca-2026-11-10.json.gz`: 24 PCA configurations across the three datasets, two dtypes,
  two component counts and two solvers. Each has two warmups and nine interleaved rounds.
- `*_omeans_k<centroids>` and `omeans-m5-ultra/`: the o-means runs of 2026-11-19 on the
  M4 Max and the M5 Ultra, one file per dataset and centroid count.

The MNIST files were written before the scripts took `--dataset`, so their metadata has no
`dataset` key. `knn_exact.py` treats a missing key as MNIST.

`results/probes/` holds exploratory MNIST runs that the README does not quote:

- `mnist_logreg_iter5000_single_fit`: the first converged run, with one fit per variant.
- `mnist_knn_k5_first_run`: the first KNN run, with 3 predictions per variant. mlxlearn
  with float64 input looked 1.8× slower than float32 here. `knn_probe_order.py` showed the
  slowdown followed run order rather than dtype, so `mnist_knn_k5` repeats the run with 7
  predictions.
- `mnist_svc_10k_mlx` and `mnist_svc_10k_sklearn`: the first 10,000 training rows, run to
  size the full fits.

## Reproduce

Run from this directory with the repository's virtual environment:

```bash
export MNIST_FAMILY_ROOT=/path/to/datasets PYTHON="$(git rev-parse --show-toplevel)/.venv/bin/python"
./run.sh /tmp/mnist-family-rerun
```

The scikit-learn SVC runs take most of the 30 or so minutes. Audit and summarize the
committed results, or point the same commands at a rerun directory:

```bash
$PYTHON summarize.py
$PYTHON knn_exact.py results/fashion_mnist_knn_k5.json.gz
$PYTHON svc_compare.py results/fashion_mnist_svc_*.json.gz
```

## PCA

The PCA rerun used the M4 Max with Python 3.14.2, NumPy 2.5.4, SciPy 1.18.1,
scikit-learn 1.9.1 and MLX 0.32.4. Pixels were divided by 255 in the input dtype.
Both libraries saw the same input, and the test transforms used models fitted on the
60,000-row training split. The default solver selected `covariance_eigh` for these shapes.

For 128 components, the median fit and test-transform times in milliseconds were:

| Dataset | Dtype | Fit, sklearn / mlxlearn | Test transform, sklearn / mlxlearn |
|---|---|---:|---:|
| MNIST | float32 | 49.77 / 25.79 | 2.02 / 1.43 |
| MNIST | float64 | 87.28 / 37.48 | 3.98 / 3.55 |
| Fashion-MNIST | float32 | 52.03 / 26.42 | 2.09 / 1.43 |
| Fashion-MNIST | float64 | 90.31 / 42.31 | 4.05 / 3.58 |
| Kannada-MNIST | float32 | 50.19 / 26.09 | 1.08 / 1.15 |
| Kannada-MNIST | float64 | 88.54 / 37.38 | 2.15 / 1.87 |

The separate Dig-MNIST transform took 2.03 / 1.18 ms in float32 and 4.10 / 3.62 ms
in float64. The full panel includes 32 components and `svd_solver="randomized"`.
Held-out reconstruction RMSE agreed within 0.0013% between libraries. The raw result
contains all train, test and Dig-MNIST timings and numerical checks.

```bash
MNIST_FAMILY_ROOT=/path/to/datasets "$PYTHON" pca.py --out /tmp/pca-mnist.json
```

Three unrelated single-threaded processes and other applications' GPU work (about 19% GPU
utilization at idle) ran throughout the original measurements. KNN and short
`LogisticRegression` timings are the most sensitive to that load.

## O-means

`omeans.py`, run on 2026-11-19 with Python 3.14.2, NumPy 2.5.4, scikit-learn 1.9.1 and
MLX 0.32.4, on the M4 Max and on a Mac Studio M5 Ultra. Pixels were divided by 255 in
float32. Models train on the 60,000-row training split at 1,024, 2,048, 4,096 and 8,192
centroids with seeds 10, 11 and 12. The error is the mean squared distance from each
held-out row to its nearest centroid, computed in float64.

The 60,000 rows are few for `OMeans`, which is built for thousands of centroids:
8,192 centroids leave 7 rows per centroid. The results below show where it helps at this
size and where it does not.

### Training

`EMeans(recipe="faiss")` against `OMeans` with its default `assignment="auto"`, both
timed as `train_device` from resident inputs. All configurations of a seed take turns
in seven rounds after two warm-ups, each round in a freshly drawn order. Times are the
median over seeds of each seed's median, on the M4 Max. "Compared" is the share of
centroids a row was compared with, "Faster" is e-means' time divided by o-means' at 1,
2 and 3 epochs, and the control is e-means against `OMeans(assignment="dense")`, which
does the same work.

| Dataset | Centroids | `auto` uses | Compared | E-means, 1 / 2 / 3 epochs | O-means, 1 / 2 / 3 epochs | Faster | Error against e-means | Full-search control |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| MNIST | 1,024 | dense | 100% | 21 / 33 / 42 ms | 23 / 33 / 43 ms | 0.93× / 0.99× / 0.99× | -0.34% / +0.00% / +0.00% | 0.96× |
| MNIST | 2,048 | dense | 100% | 30 / 58 / 79 ms | 31 / 58 / 80 ms | 0.96× / 1.00× / 0.98× | -0.58% / +0.01% / +0.00% | 0.98× |
| MNIST | 4,096 | two-level | 40% | 55 / 93 / 139 ms | 50 / 67 / 100 ms | 1.09× / 1.38× / 1.39× | -0.86% / +0.08% / -0.09% | 0.97× |
| MNIST | 8,192 | two-level | 21% | 103 / 178 / 273 ms | 75 / 85 / 128 ms | 1.38× / 2.09× / 2.13× | -0.95% / +0.04% / -0.06% | 0.93× |
| Fashion-MNIST | 1,024 | dense | 100% | 21 / 33 / 42 ms | 23 / 33 / 43 ms | 0.93× / 0.99× / 0.98× | -0.61% / +0.00% / +0.00% | 0.96× |
| Fashion-MNIST | 2,048 | dense | 100% | 31 / 59 / 78 ms | 32 / 60 / 79 ms | 0.96× / 0.99× / 0.98× | -0.98% / +0.02% / -0.00% | 0.97× |
| Fashion-MNIST | 4,096 | two-level | 40% | 56 / 97 / 142 ms | 51 / 68 / 101 ms | 1.10× / 1.43× / 1.40× | -1.36% / -0.01% / +0.05% | 0.97× |
| Fashion-MNIST | 8,192 | two-level | 21% | 115 / 200 / 299 ms | 74 / 88 / 131 ms | 1.55× / 2.26× / 2.28× | -1.67% / +0.00% / +0.00% | 0.97× |
| Kannada-MNIST | 1,024 | dense | 100% | 21 / 33 / 42 ms | 23 / 33 / 44 ms | 0.92× / 0.99× / 0.97× | -0.04% / +0.00% / +0.00% | 0.96× |
| Kannada-MNIST | 2,048 | dense | 100% | 29 / 57 / 75 ms | 31 / 58 / 78 ms | 0.94× / 0.97× / 0.96× | -0.15% / +0.00% / +0.00% | 0.96× |
| Kannada-MNIST | 4,096 | two-level, full search in 1 of 3 seeds | 58% | 56 / 98 / 143 ms | 50 / 70 / 102 ms | 1.12× / 1.40× / 1.41× | -0.20% / -0.17% / -0.09% | 0.96× |
| Kannada-MNIST | 8,192 | two-level | 21% | 104 / 178 / 275 ms | 74 / 87 / 131 ms | 1.40× / 2.04× / 2.11× | -0.62% / -0.07% / -0.09% | 0.94× |

- **Up to 2,048 centroids** a row's probes would cover about 80% of the centroids, so
  `auto` trains with the full search and matches e-means. Forcing the two-level search
  there ran at 0.62–0.85× of e-means' speed.
- **At 4,096 centroids** the two-level search compares 40% of the centroids and is 1.4×
  faster from two epochs, but only 1.1× at one epoch, where the first cell build is not
  yet amortized over 15 batches.
- **At 8,192 centroids** it compares 21% and is 2.0–2.3× faster from two epochs.
- **Kannada-MNIST at 4,096 centroids:** with one of the three seeds the cells came out
  uneven, the searches compared more than half of the centroids, and `auto` returned to
  the full search after the fifth batch.
- **The control is 0.93–0.98×:** at this size `OMeans`' own loop costs 2–7% over
  e-means before any search saving.
- **M5 Ultra:** 1.15–1.21× at 4,096 centroids and 1.74–1.85× at 8,192 from two epochs.
  At one epoch it is 1.26–1.29× at 8,192, but 0.89–0.93× at 4,096: there `auto` picks
  the two-level search and a one-epoch fit ends up 7–11% slower than e-means.

### Fits against scikit-learn

The public `fit` of `OMeans` and `EMeans` on a NumPy matrix, three epochs with the final
label pass, timed in the same rounds. The others are single fits with default settings
and seed 10: mlxlearn's Metal `KMeans` with random and with its default k-means++
initialization, and scikit-learn's `MiniBatchKMeans` and `KMeans`. Percentages are
held-out error relative to scikit-learn's `KMeans`. M4 Max.

| Dataset | Centroids | `OMeans` | `EMeans` | mlxlearn `KMeans`, random init | mlxlearn `KMeans` | scikit-learn `MiniBatchKMeans` | scikit-learn `KMeans` |
|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 1,024 | 63 ms (+0.44%) | 62 ms (+0.44%) | 6.7 s (+0.50%) | 27.2 s (+0.00%) | 2.0 s (+2.62%) | 29.4 s (53 iterations) |
| MNIST | 2,048 | 114 ms (+0.29%) | 111 ms (+0.29%) | 9.0 s (+0.39%) | 54.1 s (-0.03%) | 4.1 s (+2.81%) | 57.2 s (37 iterations) |
| MNIST | 4,096 | 150 ms (+0.25%) | 190 ms (+0.35%) | 9.6 s (+0.54%) | 106.0 s (-0.45%) | 23.9 s (+3.53%) | 109.4 s (24 iterations) |
| MNIST | 8,192 | 228 ms (-0.06%) | 381 ms (-0.02%) | 12.1 s (-0.01%) | 209.5 s (-0.02%) | 90.0 s (+3.07%) | 218.0 s (23 iterations) |
| Fashion-MNIST | 1,024 | 63 ms (+0.05%) | 62 ms (+0.05%) | 6.7 s (+0.52%) | 26.8 s (-0.00%) | 1.8 s (+3.85%) | 28.0 s (48 iterations) |
| Fashion-MNIST | 2,048 | 113 ms (-0.00%) | 109 ms (-0.00%) | 7.5 s (+0.83%) | 52.6 s (-0.12%) | 3.8 s (+3.55%) | 55.5 s (43 iterations) |
| Fashion-MNIST | 4,096 | 156 ms (-0.68%) | 194 ms (-0.73%) | 9.6 s (+0.65%) | 105.7 s (-0.18%) | 23.4 s (+3.88%) | 107.2 s (21 iterations) |
| Fashion-MNIST | 8,192 | 242 ms (-0.82%) | 420 ms (-0.83%) | 13.5 s (+0.20%) | 209.2 s (+0.28%) | 89.4 s (+3.16%) | 217.7 s (23 iterations) |
| Kannada-MNIST | 1,024 | 63 ms (-0.39%) | 62 ms (-0.39%) | 5.6 s (-0.30%) | 26.7 s (-0.62%) | 2.0 s (+2.14%) | 28.2 s (51 iterations) |
| Kannada-MNIST | 2,048 | 107 ms (-0.03%) | 107 ms (-0.03%) | 6.8 s (+0.51%) | 52.4 s (-0.00%) | 4.1 s (+2.73%) | 54.7 s (28 iterations) |
| Kannada-MNIST | 4,096 | 163 ms (-0.07%) | 197 ms (+0.02%) | 8.9 s (+0.26%) | 110.4 s (-0.11%) | 24.4 s (+2.74%) | 113.3 s (20 iterations) |
| Kannada-MNIST | 8,192 | 230 ms (-0.93%) | 392 ms (-0.84%) | 12.9 s (-0.42%) | 220.6 s (+0.12%) | 93.1 s (+1.35%) | 223.4 s (12 iterations) |

- `OMeans.fit` is 445–971× faster than scikit-learn's `KMeans` with held-out error
  between −0.93% and +0.44% of it, and 29–405× faster than `MiniBatchKMeans`, whose
  error is 1.4–3.9% higher.
- Against `EMeans.fit`, `OMeans.fit` is 1.21–1.27× faster at 4,096 centroids and
  1.67–1.74× at 8,192, and the same below. On the M5 Ultra the two ratios are
  1.12–1.14× and 1.44–1.49×, and against scikit-learn's `KMeans` it is 675–1,854×
  faster.
- mlxlearn's `KMeans` spends most of its default fit in k-means++ initialization at these
  centroid counts: 27–221 s against 6–14 s with `init="random"`. This column predates
  the Metal k-means++ seeding: with it, the default fit on MNIST takes 1.4–7.0 s at 1,024
  to 8,192 centers on the same M4 Max, with the same centers per seed
  ([measurements](../kmeans-plusplus-2026-11-19/README.md)).

Five of the 684 timed M4 Max configurations saw moderate thermal pressure around a call;
the interleaving keeps e-means and o-means in the same state. The M5 Ultra saw none.

Results are `results/<dataset>_omeans_k<centroids>.json.gz` for the M4 Max and
`results/omeans-m5-ultra/` for the M5 Ultra.

```bash
MNIST_FAMILY_ROOT=/path/to/datasets "$PYTHON" omeans.py --dataset MNIST --k 4096 \
    --out /tmp/mnist_omeans_k4096.json.gz
"$PYTHON" omeans_summarize.py /tmp
```
