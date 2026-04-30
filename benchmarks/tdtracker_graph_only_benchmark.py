"""TDTracker-only CUDA Graph benchmark.

Use this when the GPUs are idle and you want a clean number for the DVS path.
It avoids loading RITnet and measures only TDTracker forward + SimDR decode.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from headless_multimodal_benchmark import (
    DEFAULT_TD_CHECKPOINT,
    DEFAULT_TD_METRICS,
    DEFAULT_TD_MODEL,
    TdForwardDecode,
    capture_cuda_graph,
    env_report,
    make_tdtracker,
    time_loop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--height", type=int, default=60)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--model", default=str(DEFAULT_TD_MODEL))
    parser.add_argument("--metrics", default=str(DEFAULT_TD_METRICS))
    parser.add_argument("--checkpoint", default=str(DEFAULT_TD_CHECKPOINT))
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--matmul-tf32", action="store_true")
    parser.add_argument("--no-cudnn-tf32", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.matmul_tf32)
    torch.backends.cudnn.allow_tf32 = not bool(args.no_cudnn_tf32)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    print("=== environment ===")
    environment = env_report(device)
    for key, value in environment.items():
        print(f"{key}: {value}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '')}")
    print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', '')}")

    model_path = Path(args.model).expanduser().resolve()
    metrics_path = Path(args.metrics).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    tdtracker, decode = make_tdtracker(model_path, metrics_path, checkpoint_path, device)
    runner = TdForwardDecode(tdtracker, decode).to(device).eval()

    td_input = torch.randint(
        0,
        16,
        (1, args.seq_len, 2, args.height, args.width),
        dtype=torch.int16,
        device=device,
    ).float().contiguous()

    print("\n=== input ===")
    print(f"td_input: {tuple(td_input.shape)} {td_input.dtype} {td_input.device}")

    results = {
        "environment": environment,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "input": tuple(td_input.shape),
        "timings": {},
    }

    print("\n=== eager ===")
    results["timings"]["eager_tdtracker_forward_decode"] = time_loop(
        "eager_tdtracker_forward_decode",
        device,
        args.warmup,
        args.iters,
        lambda: runner(td_input),
    )

    if args.compile:
        print("\n=== torch.compile ===")
        compiled_runner = torch.compile(runner, mode="reduce-overhead")
        results["timings"]["compile_tdtracker_forward_decode"] = time_loop(
            "compile_tdtracker_forward_decode",
            device,
            args.warmup,
            args.iters,
            lambda: compiled_runner(td_input),
        )

    print("\n=== CUDA Graph ===")
    results["timings"]["graph_tdtracker_forward_decode"] = capture_cuda_graph(
        "graph_tdtracker_forward_decode",
        device,
        args.warmup,
        args.iters,
        lambda: runner(td_input),
    )

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\njson_written: {output_path}")


if __name__ == "__main__":
    main()
