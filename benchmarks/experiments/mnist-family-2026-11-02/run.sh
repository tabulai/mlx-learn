#!/usr/bin/env bash
# Rerun every timed benchmark with the settings behind results/. Takes the output
# directory as its only argument so a rerun never overwrites the committed results.
# Set PYTHON to the interpreter with mlxlearn and scikit-learn installed.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 OUTPUT_DIR" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
OUT=$(cd "$1" && pwd)
PY=${PYTHON:-python}
cd "$HERE"
DATASETS=(MNIST Fashion_MNIST Kannada_MNIST)
for ds in "${DATASETS[@]}"; do
  t=$(echo "$ds" | tr '[:upper:]' '[:lower:]')
  for it in 100 5000; do
    echo "### $ds logreg max_iter=$it"
    "$PY" logreg.py --dataset "$ds" --max-iter "$it" --fit-repeats 5 --pred-repeats 30 --out "$OUT/${t}_logreg_iter$it.json.gz"
  done
  echo "### $ds knn k=5"
  "$PY" knn.py --dataset "$ds" --k 5 --fit-repeats 5 --pred-repeats 7 --out "$OUT/${t}_knn_k5.json.gz"
done
for ds in "${DATASETS[@]}"; do
  t=$(echo "$ds" | tr '[:upper:]' '[:lower:]')
  echo "### $ds svc"
  "$PY" svc.py --dataset "$ds" --lib mlx --fit-repeats 3 --pred-repeats 3 --out "$OUT/${t}_svc_mlx.json.gz"
  "$PY" svc.py --dataset "$ds" --lib mlx --cache-size 2000 --fit-repeats 3 --pred-repeats 3 --out "$OUT/${t}_svc_mlx_c2000.json.gz"
  "$PY" svc.py --dataset "$ds" --lib sklearn --out "$OUT/${t}_svc_sklearn.json.gz"
done
for ds in "${DATASETS[@]}"; do
  t=$(echo "$ds" | tr '[:upper:]' '[:lower:]')
  echo "### $ds svc sklearn cache 2000"
  "$PY" svc.py --dataset "$ds" --lib sklearn --cache-size 2000 --out "$OUT/${t}_svc_sklearn_c2000.json.gz"
done
"$PY" summarize.py "$OUT"
