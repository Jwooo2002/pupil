#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
ITERS="${ITERS:-1000}"
WARMUP="${WARMUP:-500}"
RESULTS_DIR="${RESULTS_DIR:-benchmark_results}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/pupil_headless_bench_inductor}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SAFE_DEVICE="${DEVICE//:/_}"
TD_OUT="${RESULTS_DIR}/i9_single_gpu_tdtracker_${SAFE_DEVICE}_${STAMP}.json"
ALL_OUT="${RESULTS_DIR}/i9_single_gpu_multimodal_${SAFE_DEVICE}_${STAMP}.json"

mkdir -p "${RESULTS_DIR}"

echo "=== GPU state before ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,clocks.sm,pstate,power.draw --format=csv,noheader,nounits || true

echo
echo "=== Runtime ==="
echo "python: ${PYTHON_BIN}"
echo "device: ${DEVICE}"
echo "iters: ${ITERS}"
echo "warmup: ${WARMUP}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-<unset>}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS:-<unset>}"
echo "torchinductor cache: ${TORCHINDUCTOR_CACHE_DIR}"
echo

echo "=== TDTracker only: eager / compile / CUDA Graph ==="
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}" \
"${PYTHON_BIN}" benchmarks/tdtracker_graph_only_benchmark.py \
  --device "${DEVICE}" \
  --iters "${ITERS}" \
  --warmup "${WARMUP}" \
  --cudnn-benchmark \
  --matmul-tf32 \
  --compile \
  --output-json "${TD_OUT}"

echo
echo "=== Same GPU multimodal: RITnet / TDTracker / combined ==="
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}" \
"${PYTHON_BIN}" benchmarks/headless_multimodal_benchmark.py \
  --device "${DEVICE}" \
  --iters "${ITERS}" \
  --warmup "${WARMUP}" \
  --cudnn-benchmark \
  --matmul-tf32 \
  --compile \
  --cuda-graph \
  --output-json "${ALL_OUT}"

echo
echo "=== GPU state after ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu,clocks.sm,pstate,power.draw --format=csv,noheader,nounits || true

echo
echo "tdtracker_result: ${TD_OUT}"
echo "multimodal_result: ${ALL_OUT}"
