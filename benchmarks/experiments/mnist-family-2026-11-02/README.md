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

## Scripts

| Script | What it does |
|---|---|
| `logreg.py` | Fits mlxlearn (float32 and float64 input) and scikit-learn (float64), after one untimed mlxlearn warm-up fit. Records fit, `predict` and `predict_proba` times, iterations, convergence, accuracy and per-row predictions. Also evaluates scikit-learn's multinomial objective and maximum gradient in float64 for both models. |
| `knn.py` | Same pattern for `KNeighborsClassifier`, with scikit-learn on both input dtypes. The first `predict` on a fresh fit is timed separately. Records `kneighbors` indices and distances for the audit. |
| `svc.py` | One library per process (`--lib mlx` or `--lib sklearn`), with `cache_size` from `--cache-size` and everything else at defaults (rbf, `C=1`, `gamma="scale"`). Records fit, `predict` and `decision_function` times, support-vector counts and per-row decision values. |
| `knn_exact.py` | Checks every KNN variant's neighbors against exact integer squared distances on the raw pixels. |
| `svc_compare.py` | Compares SVC runs with the scikit-learn run among the files given: prediction agreement, pairwise votes and summed one-vs-one confidences. It recovers these from scikit-learn's ovr decision value, `votes + s / (3 (|s| + 1))`. |
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

Three unrelated single-threaded processes and other applications' GPU work (about 19% GPU
utilization at idle) ran throughout the original measurements. KNN and short
`LogisticRegression` timings are the most sensitive to that load.
