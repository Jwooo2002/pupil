"""Headless split-GPU benchmark.

Loads RITnet on one CUDA device and TDTracker on another. This is closer to the
intended runtime where RITnet and DVS/TDTracker may use separate GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from headless_multimodal_benchmark import (
    DEFAULT_RITNET_CHECKPOINT,
    DEFAULT_RITNET_MODEL,
    DEFAULT_TD_CHECKPOINT,
    DEFAULT_TD_METRICS,
    DEFAULT_TD_MODEL,
    RitnetForwardArgmax,
    TdForwardDecode,
    env_report,
    make_ritnet,
    make_tdtracker,
    percentile,
    print_summary,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ritnet-device", default="cuda:0")
    parser.add_argument("--td-device", default="cuda:1")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--ritnet-height", type=int, default=400)
    parser.add_argument("--ritnet-width", type=int, default=400)
    parser.add_argument("--td-seq-len", type=int, default=8)
    parser.add_argument("--td-height", type=int, default=60)
    parser.add_argument("--td-width", type=int, default=80)
    parser.add_argument("--ritnet-model", default=str(DEFAULT_RITNET_MODEL))
    parser.add_argument("--ritnet-checkpoint", default=str(DEFAULT_RITNET_CHECKPOINT))
    parser.add_argument("--td-model", default=str(DEFAULT_TD_MODEL))
    parser.add_argument("--td-metrics", default=str(DEFAULT_TD_METRICS))
    parser.add_argument("--td-checkpoint", default=str(DEFAULT_TD_CHECKPOINT))
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--matmul-tf32", action="store_true")
    parser.add_argument("--no-cudnn-tf32", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def sync_devices(devices: list[torch.device]) -> None:
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def summarize(values: list[float]) -> dict[str, float]:
    mean_ms = statistics.fmean(values)
    return {
        "mean_ms": mean_ms,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "hz": 1000.0 / mean_ms,
    }


def time_loop(
    label: str,
    devices: list[torch.device],
    warmup: int,
    iters: int,
    fn: Callable[[], Any],
) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        sync_devices(devices)

        values: list[float] = []
        for _ in range(iters):
            sync_devices(devices)
            start = time.perf_counter()
            fn()
            sync_devices(devices)
            values.append((time.perf_counter() - start) * 1000.0)

    result = summarize(values)
    print_summary(label, result)
    return result


def capture_graph(device: torch.device, warmup: int, fn: Callable[[], Any]) -> tuple[torch.cuda.CUDAGraph, Any]:
    if device.type != "cuda":
        raise RuntimeError("CUDA Graph requires CUDA")

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.device(device), torch.cuda.stream(stream), torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.device(device), torch.cuda.graph(graph), torch.no_grad():
        static_output = fn()
    torch.cuda.synchronize(device)
    return graph, static_output


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.matmul_tf32)
    torch.backends.cudnn.allow_tf32 = not bool(args.no_cudnn_tf32)

    ritnet_device = torch.device(args.ritnet_device)
    td_device = torch.device(args.td_device)
    devices = [ritnet_device, td_device]

    paths = {
        "ritnet_model": Path(args.ritnet_model).expanduser().resolve(),
        "ritnet_checkpoint": Path(args.ritnet_checkpoint).expanduser().resolve(),
        "td_model": Path(args.td_model).expanduser().resolve(),
        "td_metrics": Path(args.td_metrics).expanduser().resolve(),
        "td_checkpoint": Path(args.td_checkpoint).expanduser().resolve(),
    }

    print("=== environment ===")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '')}")
    print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', '')}")
    print(f"torch_num_threads: {torch.get_num_threads()}")
    print(f"torch_num_interop_threads: {torch.get_num_interop_threads()}")
    print("\n--- ritnet device ---")
    env_ritnet = env_report(ritnet_device)
    for key, value in env_ritnet.items():
        print(f"{key}: {value}")
    print("\n--- tdtracker device ---")
    env_td = env_report(td_device)
    for key, value in env_td.items():
        print(f"{key}: {value}")

    print("\n=== files ===")
    for key, path in paths.items():
        print(f"{key}: {path}")
        print(f"{key}_sha256: {sha256(path)}")

    ritnet = make_ritnet(paths["ritnet_model"], paths["ritnet_checkpoint"], ritnet_device)
    tdtracker, td_decode = make_tdtracker(
        paths["td_model"], paths["td_metrics"], paths["td_checkpoint"], td_device
    )
    ritnet_argmax = RitnetForwardArgmax(ritnet).to(ritnet_device).eval()
    td_forward_decode = TdForwardDecode(tdtracker, td_decode).to(td_device).eval()

    ritnet_input = torch.randn(
        1,
        1,
        args.ritnet_height,
        args.ritnet_width,
        dtype=torch.float32,
        device=ritnet_device,
    ).contiguous()
    td_input = torch.randint(
        0,
        16,
        (1, args.td_seq_len, 2, args.td_height, args.td_width),
        dtype=torch.int16,
        device=td_device,
    ).float().contiguous()

    print("\n=== inputs ===")
    print(f"ritnet_input: {tuple(ritnet_input.shape)} {ritnet_input.dtype} {ritnet_input.device}")
    print(f"td_input: {tuple(td_input.shape)} {td_input.dtype} {td_input.device}")

    results: dict[str, Any] = {
        "environment": {
            "ritnet": env_ritnet,
            "tdtracker": env_td,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "files": {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()},
        "timings": {},
    }

    print("\n=== eager timings ===")
    results["timings"]["ritnet_gpu0_forward_argmax"] = time_loop(
        "ritnet_gpu0_forward_argmax",
        [ritnet_device],
        args.warmup,
        args.iters,
        lambda: ritnet_argmax(ritnet_input),
    )
    results["timings"]["tdtracker_gpu1_forward_decode"] = time_loop(
        "tdtracker_gpu1_forward_decode",
        [td_device],
        args.warmup,
        args.iters,
        lambda: td_forward_decode(td_input),
    )
    results["timings"]["split_pair_eager_enqueue_both"] = time_loop(
        "split_pair_eager_enqueue_both",
        devices,
        args.warmup,
        args.iters,
        lambda: (ritnet_argmax(ritnet_input), td_forward_decode(td_input)),
    )

    if args.compile:
        print("\n=== torch.compile timings ===")
        compiled_ritnet = torch.compile(ritnet_argmax, mode="reduce-overhead")
        compiled_td = torch.compile(td_forward_decode, mode="reduce-overhead")
        results["timings"]["compile_ritnet_gpu0_forward_argmax"] = time_loop(
            "compile_ritnet_gpu0_forward_argmax",
            [ritnet_device],
            args.warmup,
            args.iters,
            lambda: compiled_ritnet(ritnet_input),
        )
        results["timings"]["compile_tdtracker_gpu1_forward_decode"] = time_loop(
            "compile_tdtracker_gpu1_forward_decode",
            [td_device],
            args.warmup,
            args.iters,
            lambda: compiled_td(td_input),
        )
        results["timings"]["split_pair_compile_enqueue_both"] = time_loop(
            "split_pair_compile_enqueue_both",
            devices,
            args.warmup,
            args.iters,
            lambda: (compiled_ritnet(ritnet_input), compiled_td(td_input)),
        )

    if args.cuda_graph:
        print("\n=== CUDA Graph timings ===")
        ritnet_graph = None
        td_graph = None
        ritnet_static = None
        td_static = None
        try:
            ritnet_graph, ritnet_static = capture_graph(
                ritnet_device, args.warmup, lambda: ritnet_argmax(ritnet_input)
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"graph_ritnet_gpu0_forward_argmax: unavailable ({message})")
            results["timings"]["graph_ritnet_gpu0_forward_argmax"] = {"error": message}

        try:
            td_graph, td_static = capture_graph(
                td_device, args.warmup, lambda: td_forward_decode(td_input)
            )
            results["timings"]["graph_tdtracker_gpu1_forward_decode"] = time_loop(
                "graph_tdtracker_gpu1_forward_decode",
                [td_device],
                args.warmup,
                args.iters,
                lambda: td_graph.replay(),
            )
            _ = td_static
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"graph_tdtracker_gpu1_forward_decode: unavailable ({message})")
            results["timings"]["graph_tdtracker_gpu1_forward_decode"] = {"error": message}

        if ritnet_graph is not None:
            try:
                results["timings"]["graph_ritnet_gpu0_forward_argmax"] = time_loop(
                    "graph_ritnet_gpu0_forward_argmax",
                    [ritnet_device],
                    args.warmup,
                    args.iters,
                    lambda: ritnet_graph.replay(),
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"graph_ritnet_gpu0_forward_argmax: unavailable ({message})")
                results["timings"]["graph_ritnet_gpu0_forward_argmax"] = {"error": message}
        if td_graph is not None:
            try:
                results["timings"]["graph_tdtracker_gpu1_forward_decode"] = time_loop(
                    "graph_tdtracker_gpu1_forward_decode",
                    [td_device],
                    args.warmup,
                    args.iters,
                    lambda: td_graph.replay(),
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"graph_tdtracker_gpu1_forward_decode: unavailable ({message})")
                results["timings"]["graph_tdtracker_gpu1_forward_decode"] = {"error": message}
        if ritnet_graph is not None and td_graph is not None:
            try:
                results["timings"]["split_pair_graph_replay_both"] = time_loop(
                    "split_pair_graph_replay_both",
                    devices,
                    args.warmup,
                    args.iters,
                    lambda: (ritnet_graph.replay(), td_graph.replay()),
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"split_pair_graph_replay_both: unavailable ({message})")
                results["timings"]["split_pair_graph_replay_both"] = {"error": message}
        _ = (ritnet_static, td_static)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\njson_written: {output_path}")


if __name__ == "__main__":
    main()
