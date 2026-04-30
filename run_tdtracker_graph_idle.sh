#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
ITERS="${ITERS:-5000}"
WARMUP="${WARMUP:-1000}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/pupil_headless_bench_inductor}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SAFE_DEVICE="${DEVICE//:/_}"
OUT="benchmark_results/tdtracker_graph_idle_${SAFE_DEVICE}_${STAMP}.json"

mkdir -p benchmark_results

echo "=== GPU state before ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,clocks.sm,pstate,power.draw --format=csv,noheader,nounits || true

echo
echo "=== Running TDTracker Graph benchmark ==="
echo "python: ${PYTHON_BIN}"
echo "device: ${DEVICE}"
echo "iters: ${ITERS}"
echo "warmup: ${WARMUP}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-<unset>}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS:-<unset>}"
echo "output: ${OUT}"
echo

TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}" \
"${PYTHON_BIN}" benchmarks/tdtracker_graph_only_benchmark.py \
  --device "${DEVICE}" \
  --iters "${ITERS}" \
  --warmup "${WARMUP}" \
  --matmul-tf32 \
  --compile \
  --output-json "${OUT}"

echo
echo "=== GPU state after ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,clocks.sm,pstate,power.draw --format=csv,noheader,nounits || true

echo
echo "result: ${OUT}"
