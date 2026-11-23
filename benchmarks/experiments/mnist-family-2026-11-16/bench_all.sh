#!/bin/bash
# Run every MNIST-family benchmark on one machine: run.sh and pca.py from
# ../mnist-family-2026-11-02, then the exact-neighbor and SVC agreement audits.
#
# usage: bench_all.sh REPO PYTHON OUT DATA_ROOT CACHE_DIR
#   REPO       repository checkout to benchmark (its src/ goes on PYTHONPATH)
#   PYTHON     interpreter with numpy, scipy, scikit-learn and mlx installed
#   OUT        output directory, created if needed
#   DATA_ROOT  MNIST_FAMILY_ROOT: holds MNIST/, Fashion_MNIST/ and Kannada_MNIST/
#   CACHE_DIR  MNIST_FAMILY_CACHE: parsed-pixel cache
#
# monitor.log samples load, GPU utilization and the busiest processes every 30 s.
# It names other software running on the machine, so the committed results
# summarize it in README.md instead of including it.
set -uo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 REPO PYTHON OUT DATA_ROOT CACHE_DIR" >&2; exit 2; }
REPO=$1 PY=$2 OUT=$3 ROOT=$4 CACHE=$5
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export PYTHONPATH="$REPO/src" PYTHON="$PY" MNIST_FAMILY_ROOT="$ROOT" MNIST_FAMILY_CACHE="$CACHE"
cd "$REPO/benchmarks/experiments/mnist-family-2026-11-02"

(
  while true; do
    gpu=$(ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2)
    la=$(sysctl -n vm.loadavg | tr -d '{}')
    top=$(ps -Ao pid=,pcpu=,comm= -r | head -4 | awk '{n=$3; sub(".*/","",n); printf "%s(%s):%s ", n, $1, $2}')
    echo "$(date +%H:%M:%S) load=[$la] gpu=${gpu}% top: $top"
    sleep 30
  done
) > "$OUT/monitor.log" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT

{
  echo "chip=$(sysctl -n machdep.cpu.brand_string) macos=$(sw_vers -productVersion)"
  echo "commit=$(git -C "$REPO" rev-parse HEAD) status:"; git -C "$REPO" status --short
  "$PY" -c 'import sys,numpy,scipy,sklearn,mlx.core as mx; print(sys.version.split()[0], numpy.__version__, scipy.__version__, sklearn.__version__, mx.__version__, getattr(mx, "device_info", mx.metal.device_info)().get("architecture"))'
  date
} > "$OUT/env.txt" 2>&1

start=$(date +%s)
echo "=== run.sh $(date)"
./run.sh "$OUT" > "$OUT/run.log" 2>&1; echo "run.sh exit=$?" >> "$OUT/run.log"
echo "=== pca.py $(date)"
"$PY" pca.py --out "$OUT/pca.json" > "$OUT/pca.log" 2>&1; echo "pca.py exit=$?" >> "$OUT/pca.log"
gzip -9 -n -f "$OUT/pca.json"
echo "=== audits $(date)"
for t in mnist fashion_mnist kannada_mnist; do
  echo "## $t"; "$PY" knn_exact.py "$OUT/${t}_knn_k5.json.gz"
done > "$OUT/knn_exact.log" 2>&1
for t in mnist fashion_mnist kannada_mnist; do
  echo "## $t"; "$PY" svc_compare.py "$OUT/${t}"_svc_*.json.gz
done > "$OUT/svc_compare.log" 2>&1
echo "=== done after $(( $(date +%s) - start )) s $(date)"
