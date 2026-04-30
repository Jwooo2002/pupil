"""Headless RITnet/TDTracker benchmark for the hybrid Pupil fork.

This script intentionally avoids starting Pupil Capture, GLFW, camera capture,
or the plugin runtime. It loads the two neural models directly and measures:

- RITnet segmentation forward
- RITnet plugin-style CPU argmax decode
- TDTracker forward + SimDR decode
- RITnet + TDTracker on the same CUDA device
- eager, torch.compile, and CUDA Graph replay where possible
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / "pupil_src" / "shared_modules" / "pupil_detector_plugins"
DEFAULT_RITNET_MODEL = PLUGIN_DIR / "densenet.py"
DEFAULT_RITNET_CHECKPOINT = PLUGIN_DIR / "best_model.pkl"
DEFAULT_TD_MODEL = PLUGIN_DIR / "dvs_models" / "TDTracker.py"
DEFAULT_TD_METRICS = PLUGIN_DIR / "dvs_metrics.py"
DEFAULT_TD_CHECKPOINT = PLUGIN_DIR / "best_checkpoint.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=200)
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
    parser.add_argument(
        "--only",
        choices=("all", "ritnet", "tdtracker"),
        default="all",
        help="Load and benchmark only one model, or all models together.",
    )
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def unwrap_state_dict(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    for key in ("state_dict", "model_state_dict", "net", "model"):
        value = obj.get(key)
        if isinstance(value, dict):
            return value
    return obj


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def summarize(values: list[float]) -> dict[str, float]:
    mean_ms = statistics.fmean(values)
    return {
        "mean_ms": mean_ms,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "hz": 1000.0 / mean_ms,
    }


def print_summary(label: str, summary: dict[str, float]) -> None:
    print(
        f"{label}: mean={summary['mean_ms']:.3f} ms | "
        f"p50={summary['p50_ms']:.3f} ms | "
        f"p95={summary['p95_ms']:.3f} ms | "
        f"hz={summary['hz']:.1f}"
    )


def time_loop(
    label: str,
    device: torch.device,
    warmup: int,
    iters: int,
    fn: Callable[[], Any],
) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        sync(device)

        values: list[float] = []
        for _ in range(iters):
            sync(device)
            start = time.perf_counter()
            fn()
            sync(device)
            values.append((time.perf_counter() - start) * 1000.0)

    result = summarize(values)
    print_summary(label, result)
    return result


class RitnetForwardArgmax(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.argmax(logits, dim=1)


class TdForwardDecode(nn.Module):
    def __init__(self, model: nn.Module, decode: Callable[[torch.Tensor, torch.Tensor], Any]):
        super().__init__()
        self.model = model
        self.decode = decode

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pw, ph = self.model(x)
        return self.decode(pw, ph)


class CombinedForward(nn.Module):
    def __init__(
        self,
        ritnet: nn.Module,
        tdtracker: nn.Module,
        td_decode: Callable[[torch.Tensor, torch.Tensor], Any],
    ):
        super().__init__()
        self.ritnet = ritnet
        self.tdtracker = tdtracker
        self.td_decode = td_decode

    def forward(
        self, ritnet_input: torch.Tensor, td_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ritnet_seg = torch.argmax(self.ritnet(ritnet_input), dim=1)
        pw, ph = self.tdtracker(td_input)
        td_pos, td_prob = self.td_decode(pw, ph)
        return ritnet_seg, td_pos, td_prob


def make_ritnet(model_path: Path, checkpoint_path: Path, device: torch.device) -> nn.Module:
    module = load_module("headless_ritnet_model", model_path)
    model = module.DenseNet2D(dropout=True, prob=0.2).to(device).eval()
    state = unwrap_state_dict(torch.load(checkpoint_path, map_location=device))
    incompatible = model.load_state_dict(state, strict=False)
    print(f"ritnet_missing: {list(incompatible.missing_keys)}")
    print(f"ritnet_unexpected: {list(incompatible.unexpected_keys)}")
    return model


def make_tdtracker(
    model_path: Path,
    metrics_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, Callable[[torch.Tensor, torch.Tensor], Any]]:
    td_module = load_module("headless_tdtracker_model", model_path)
    metrics = load_module("headless_tdtracker_metrics", metrics_path)
    model_args = types.SimpleNamespace(
        sensor_width=346,
        sensor_height=260,
        spatial_factor=0.125,
        pixel_tolerances=[1, 3, 5, 10, 15],
    )
    model = td_module.Model(model_args).to(device).eval()
    state = unwrap_state_dict(torch.load(checkpoint_path, map_location=device))
    incompatible = model.load_state_dict(state, strict=False)
    print(f"td_missing: {list(incompatible.missing_keys)}")
    print(f"td_unexpected: {list(incompatible.unexpected_keys)}")
    return model, metrics.decode_batch_sa_simdr


def cpu_ritnet_decode_like_plugin(logits: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = logits.size()
    _, indices = logits.cpu().max(1)
    return indices.view(batch_size, height, width)


def capture_cuda_graph(
    label: str,
    device: torch.device,
    warmup: int,
    iters: int,
    fn: Callable[[], Any],
) -> dict[str, float]:
    if device.type != "cuda":
        raise RuntimeError("CUDA Graph requires CUDA")

    stream = torch.cuda.Stream(device=device)
    # CUDA Graph capture can conflict with inference-mode tensors when captured
    # modules use in-place updates internally. no_grad keeps autograd off without
    # creating inference tensors.
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream(device).wait_stream(stream)
    sync(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        static_output = fn()
    sync(device)

    values: list[float] = []
    for _ in range(iters):
        sync(device)
        start = time.perf_counter()
        graph.replay()
        sync(device)
        values.append((time.perf_counter() - start) * 1000.0)

    result = summarize(values)
    print_summary(label, result)
    # Keep graph outputs alive.
    _ = static_output
    return result


def env_report(device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_mb": props.total_memory // (1024 * 1024),
            }
        )
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free,utilization.gpu,clocks.sm,pstate,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        report["nvidia_smi"] = smi.stdout.strip()
    except Exception as exc:
        report["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return report


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.matmul_tf32)
    torch.backends.cudnn.allow_tf32 = not bool(args.no_cudnn_tf32)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    paths = {
        "ritnet_model": Path(args.ritnet_model).expanduser().resolve(),
        "ritnet_checkpoint": Path(args.ritnet_checkpoint).expanduser().resolve(),
        "td_model": Path(args.td_model).expanduser().resolve(),
        "td_metrics": Path(args.td_metrics).expanduser().resolve(),
        "td_checkpoint": Path(args.td_checkpoint).expanduser().resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    print("=== environment ===")
    environment = env_report(device)
    for key, value in environment.items():
        print(f"{key}: {value}")

    print("\n=== files ===")
    for key, path in paths.items():
        print(f"{key}: {path}")
        print(f"{key}_sha256: {sha256(path)}")

    ritnet = None
    tdtracker = None
    td_decode = None
    if args.only in ("all", "ritnet"):
        ritnet = make_ritnet(paths["ritnet_model"], paths["ritnet_checkpoint"], device)
    if args.only in ("all", "tdtracker"):
        tdtracker, td_decode = make_tdtracker(
            paths["td_model"], paths["td_metrics"], paths["td_checkpoint"], device
        )

    ritnet_input = None
    td_input = None
    if ritnet is not None:
        ritnet_input = torch.randn(
            1, 1, args.ritnet_height, args.ritnet_width, dtype=torch.float32, device=device
        ).contiguous()
    if tdtracker is not None:
        td_input = torch.randint(
            0,
            16,
            (1, args.td_seq_len, 2, args.td_height, args.td_width),
            dtype=torch.int16,
            device=device,
        ).float().contiguous()

    print("\n=== inputs ===")
    if ritnet_input is not None:
        print(f"ritnet_input: {tuple(ritnet_input.shape)} {ritnet_input.dtype} {ritnet_input.device}")
    if td_input is not None:
        print(f"td_input: {tuple(td_input.shape)} {td_input.dtype} {td_input.device}")

    ritnet_argmax = RitnetForwardArgmax(ritnet).to(device).eval() if ritnet is not None else None
    td_forward_decode = (
        TdForwardDecode(tdtracker, td_decode).to(device).eval()
        if tdtracker is not None and td_decode is not None
        else None
    )
    combined = (
        CombinedForward(ritnet, tdtracker, td_decode).to(device).eval()
        if ritnet is not None and tdtracker is not None and td_decode is not None
        else None
    )

    results: dict[str, Any] = {
        "environment": environment,
        "files": {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()},
        "inputs": {
            "ritnet": tuple(ritnet_input.shape) if ritnet_input is not None else None,
            "tdtracker": tuple(td_input.shape) if td_input is not None else None,
        },
        "timings": {},
    }

    print("\n=== eager timings ===")
    if ritnet is not None and ritnet_input is not None and ritnet_argmax is not None:
        with torch.no_grad():
            ritnet_logits = ritnet(ritnet_input)
            sync(device)
        results["timings"]["ritnet_forward"] = time_loop(
            "ritnet_forward",
            device,
            args.warmup,
            args.iters,
            lambda: ritnet(ritnet_input),
        )
        results["timings"]["ritnet_forward_gpu_argmax"] = time_loop(
            "ritnet_forward_gpu_argmax",
            device,
            args.warmup,
            args.iters,
            lambda: ritnet_argmax(ritnet_input),
        )
        results["timings"]["ritnet_plugin_cpu_argmax_copy_only"] = time_loop(
            "ritnet_plugin_cpu_argmax_copy_only",
            device,
            args.warmup,
            args.iters,
            lambda: cpu_ritnet_decode_like_plugin(ritnet_logits),
        )
        results["timings"]["ritnet_forward_plugin_cpu_argmax"] = time_loop(
            "ritnet_forward_plugin_cpu_argmax",
            device,
            args.warmup,
            args.iters,
            lambda: cpu_ritnet_decode_like_plugin(ritnet(ritnet_input)),
        )
    if td_forward_decode is not None and td_input is not None:
        results["timings"]["tdtracker_forward_decode"] = time_loop(
            "tdtracker_forward_decode",
            device,
            args.warmup,
            args.iters,
            lambda: td_forward_decode(td_input),
        )
    if combined is not None and ritnet_input is not None and td_input is not None:
        results["timings"]["combined_ritnet_tdtracker_same_gpu"] = time_loop(
            "combined_ritnet_tdtracker_same_gpu",
            device,
            args.warmup,
            args.iters,
            lambda: combined(ritnet_input, td_input),
        )

    if args.compile:
        print("\n=== torch.compile timings ===")
        compile_results: dict[str, Any] = {}
        compile_jobs = []
        if ritnet_argmax is not None and ritnet_input is not None:
            compile_jobs.append(("compile_ritnet_forward_gpu_argmax", ritnet_argmax, lambda m: m(ritnet_input)))
        if td_forward_decode is not None and td_input is not None:
            compile_jobs.append(("compile_tdtracker_forward_decode", td_forward_decode, lambda m: m(td_input)))
        if combined is not None and ritnet_input is not None and td_input is not None:
            compile_jobs.append(("compile_combined_same_gpu", combined, lambda m: m(ritnet_input, td_input)))
        for label, module, call in compile_jobs:
            try:
                compiled = torch.compile(module, mode="reduce-overhead")
                compile_results[label] = time_loop(
                    label,
                    device,
                    args.warmup,
                    args.iters,
                    lambda compiled=compiled, call=call: call(compiled),
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"{label}: unavailable ({message})")
                compile_results[label] = {"error": message}
        results["timings"]["compile"] = compile_results

    if args.cuda_graph:
        print("\n=== CUDA Graph timings ===")
        graph_results: dict[str, Any] = {}
        graph_jobs = []
        if ritnet_argmax is not None and ritnet_input is not None:
            graph_jobs.append(("graph_ritnet_forward_gpu_argmax", lambda: ritnet_argmax(ritnet_input)))
        if td_forward_decode is not None and td_input is not None:
            graph_jobs.append(("graph_tdtracker_forward_decode", lambda: td_forward_decode(td_input)))
        if combined is not None and ritnet_input is not None and td_input is not None:
            graph_jobs.append(("graph_combined_same_gpu", lambda: combined(ritnet_input, td_input)))
        for label, call in graph_jobs:
            try:
                graph_results[label] = capture_cuda_graph(
                    label,
                    device,
                    args.warmup,
                    args.iters,
                    call,
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"{label}: unavailable ({message})")
                graph_results[label] = {"error": message}
        results["timings"]["cuda_graph"] = graph_results

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\njson_written: {output_path}")


if __name__ == "__main__":
    main()
