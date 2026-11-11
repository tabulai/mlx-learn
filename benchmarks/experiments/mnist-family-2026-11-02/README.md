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
