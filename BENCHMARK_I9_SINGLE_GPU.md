# i9 Single-GPU Headless Benchmark

이 문서는 GUI, 카메라, Pupil runtime을 띄우지 않고 RTX 4090 한 장에서 RITnet/TDTracker 속도만 재는 방법이다.

## 준비

```bash
git clone https://github.com/Jwooo2002/pupil.git
cd pupil
git checkout bench/i9-single-gpu-headless
conda activate <pupil-or-tdtracker-env>
```

필수 조건:

- `torch`
- CUDA 사용 가능 PyTorch build
- repo에 포함된 `pupil_src/shared_modules/pupil_detector_plugins/best_model.pkl`
- repo에 포함된 `pupil_src/shared_modules/pupil_detector_plugins/best_checkpoint.pth`

## 권장 실행

```bash
./run_i9_single_gpu_bench.sh
```

기본값:

- Python: 현재 shell의 `python`
- Device: `cuda:0`
- Iterations: `1000`
- Warmup: `500`
- Output: `benchmark_results/*.json`
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`는 기본적으로 건드리지 않음

명시 실행:

```bash
PYTHON_BIN=python DEVICE=cuda:0 ITERS=2000 WARMUP=1000 ./run_i9_single_gpu_bench.sh
```

## TDTracker만 깨끗하게 보기

```bash
./run_tdtracker_graph_idle.sh
```

또는:

```bash
PYTHON_BIN=python DEVICE=cuda:0 ITERS=5000 WARMUP=1000 ./run_tdtracker_graph_idle.sh
```

## 직접 실행

TDTracker only:

```bash
python benchmarks/tdtracker_graph_only_benchmark.py \
  --device cuda:0 \
  --iters 1000 \
  --warmup 500 \
  --cudnn-benchmark \
  --matmul-tf32 \
  --compile \
  --output-json benchmark_results/i9_tdtracker_cuda0.json
```

RITnet + TDTracker same GPU:

```bash
python benchmarks/headless_multimodal_benchmark.py \
  --device cuda:0 \
  --iters 1000 \
  --warmup 500 \
  --cudnn-benchmark \
  --matmul-tf32 \
  --compile \
  --cuda-graph \
  --output-json benchmark_results/i9_multimodal_cuda0.json
```

## 봐야 하는 값

`tdtracker_graph_only_benchmark.py`:

- `eager_tdtracker_forward_decode`
- `compile_tdtracker_forward_decode`
- `graph_tdtracker_forward_decode`

`headless_multimodal_benchmark.py`:

- `ritnet_forward_gpu_argmax`
- `tdtracker_forward_decode`
- `combined_ritnet_tdtracker_same_gpu`
- `compile_combined_same_gpu`
- `graph_combined_same_gpu`

## 해석 기준

TDTracker only CUDA Graph가 1000Hz 이상이면 DVS 보조 경로 자체는 충분히 빠른 것이다.

RITnet+TDTracker same GPU가 1000Hz보다 낮아도 이상한 게 아니다. RITnet은 anchor/camera FPS 경로이고, 1000Hz 대상은 TDTracker/DVS publish 경로다.
