#!/usr/bin/env python3
"""Benchmark MemRift split exponent output paths."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

import torch
import zstandard as zstd

from flagscale.compress.float_split_stride_pin import float_split_stride_pin as fs_sp


DEFAULT_SHAPES = [
    (32, 2048, 2048),
    (2048, 28672),
    (2048, 1, 14336),
    (2048, 1, 4096),
    (32, 2048, 128),
]

DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part) for part in value.split(",") if part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid shape {value!r}; expected comma-separated integers"
        ) from exc

    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(
            f"invalid shape {value!r}; dimensions must be positive"
        )
    return shape


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
    }


def summarize_bytes(values: list[int]) -> dict[str, float | int]:
    return {
        "min_bytes": min(values),
        "max_bytes": max(values),
        "mean_bytes": statistics.fmean(values),
        "median_bytes": statistics.median(values),
    }


def summarize_field(iterations: list[dict[str, Any]], field: str) -> dict[str, float]:
    return summarize([float(item[field]) for item in iterations])


def shape_label(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def get_stream(kind: str) -> torch.cuda.Stream:
    if kind == "current":
        return torch.cuda.current_stream()
    return torch.cuda.Stream()


def measure_once(
    tensor: torch.Tensor,
    mode: str,
    stream: torch.cuda.Stream,
    compressor: zstd.ZstdCompressor,
) -> dict[str, Any]:
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record(stream)
    submit_start = time.perf_counter()
    if mode == "mapped":
        cpu_exp, sm_bits = fs_sp.split(tensor, stream.cuda_stream)
        staging = None
    else:
        cpu_exp, sm_bits, staging = fs_sp.split_copy(tensor, stream.cuda_stream)
    submit_ms = (time.perf_counter() - submit_start) * 1000.0
    end_evt.record(stream)

    ready_evt = end_evt
    wait_start = time.perf_counter()
    ready_evt.synchronize()
    event_wait_ms = (time.perf_counter() - wait_start) * 1000.0
    cuda_event_ms = start_evt.elapsed_time(end_evt)

    numpy_start = time.perf_counter()
    exp_numpy = cpu_exp.numpy()
    numpy_ms = (time.perf_counter() - numpy_start) * 1000.0

    zstd_start = time.perf_counter()
    compressed = compressor.compress(memoryview(exp_numpy))
    zstd_ms = (time.perf_counter() - zstd_start) * 1000.0

    compressed_len = len(compressed)
    del compressed, exp_numpy, cpu_exp, sm_bits, staging

    return {
        "submit_ms": submit_ms,
        "cuda_event_ms": cuda_event_ms,
        "event_wait_ms": event_wait_ms,
        "numpy_ms": numpy_ms,
        "zstd_ms": zstd_ms,
        "compressed_bytes": compressed_len,
    }


def warmup(
    tensor: torch.Tensor,
    mode: str,
    stream: torch.cuda.Stream,
    count: int,
) -> None:
    for _ in range(count):
        if mode == "mapped":
            cpu_exp, sm_bits = fs_sp.split(tensor, stream.cuda_stream)
            staging = None
        else:
            cpu_exp, sm_bits, staging = fs_sp.split_copy(tensor, stream.cuda_stream)
        ready_evt = stream.record_event()
        ready_evt.synchronize()
        del cpu_exp, sm_bits, staging


def benchmark_mode(
    shape: tuple[int, ...],
    dtype_name: str,
    mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stream = get_stream(args.stream)
    dtype = DTYPES[dtype_name]
    tensor = torch.randn(shape, device="cuda", dtype=torch.float32).to(dtype)
    if args.stream == "new":
        stream.wait_stream(torch.cuda.current_stream())
    compressor = zstd.ZstdCompressor(level=args.zstd_level)

    warmup(tensor, mode, stream, args.warmup)
    iterations = [
        measure_once(tensor, mode, stream, compressor) for _ in range(args.iters)
    ]

    del tensor
    if args.stream == "new":
        torch.cuda.current_stream().wait_stream(stream)

    return {
        "mode": mode,
        "shape": list(shape),
        "dtype": dtype_name,
        "numel": int(torch.Size(shape).numel()),
        "warmup": args.warmup,
        "iters": args.iters,
        "zstd_level": args.zstd_level,
        "stream": args.stream,
        "iterations": iterations,
        "summaries": {
            "submit": summarize_field(iterations, "submit_ms"),
            "cuda_event": summarize_field(iterations, "cuda_event_ms"),
            "event_wait": summarize_field(iterations, "event_wait_ms"),
            "numpy": summarize_field(iterations, "numpy_ms"),
            "zstd": summarize_field(iterations, "zstd_ms"),
            "compressed_bytes": summarize_bytes(
                [int(item["compressed_bytes"]) for item in iterations]
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MemRift mapped and split-copy exponent paths."
    )
    parser.add_argument("--mode", choices=("mapped", "copy", "both"), default="both")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument("--shape", action="append", type=parse_shape)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--zstd-level", type=int, default=18)
    parser.add_argument("--stream", choices=("current", "new"), default="current")
    parser.add_argument(
        "--out", default="outputs/memrift_split_path_benchmark.json"
    )
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")

    args.shapes = args.shape or DEFAULT_SHAPES
    return args


def check_environment(mode: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not fs_sp.is_available():
        raise RuntimeError("float_split_stride_pin extension is not available")
    if mode in ("copy", "both") and not hasattr(fs_sp, "split_copy"):
        raise RuntimeError("float_split_stride_pin split_copy is not available")


def selected_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["mapped", "copy"]
    return [mode]


def print_table(results: list[dict[str, Any]]) -> None:
    print(
        "mode shape dtype numel submit_mean_ms cuda_event_mean_ms "
        "event_wait_mean_ms zstd_mean_ms"
    )
    for result in results:
        summaries = result["summaries"]
        print(
            f"{result['mode']} "
            f"{shape_label(tuple(result['shape']))} "
            f"{result['dtype']} "
            f"{result['numel']} "
            f"{summaries['submit']['mean_ms']:.3f} "
            f"{summaries['cuda_event']['mean_ms']:.3f} "
            f"{summaries['event_wait']['mean_ms']:.3f} "
            f"{summaries['zstd']['mean_ms']:.3f}"
        )


def main() -> None:
    args = parse_args()
    check_environment(args.mode)

    results = []
    for shape in args.shapes:
        for mode in selected_modes(args.mode):
            results.append(benchmark_mode(shape, args.dtype, mode, args))

    output = {
        "config": {
            "mode": args.mode,
            "dtype": args.dtype,
            "shapes": [list(shape) for shape in args.shapes],
            "warmup": args.warmup,
            "iters": args.iters,
            "zstd_level": args.zstd_level,
            "stream": args.stream,
        },
        "results": results,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    print_table(results)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
