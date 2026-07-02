#!/usr/bin/env python3
"""Compare baseline and MemRift weight read time from disk."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable


BASELINE_WEIGHT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
}


def collect_baseline_weight_files(model_path: Path) -> list[Path]:
    if model_path.is_file():
        return [model_path]
    if not model_path.is_dir():
        raise FileNotFoundError(f"model path not found: {model_path}")

    files = [
        path
        for path in model_path.rglob("*")
        if path.is_file() and path.suffix in BASELINE_WEIGHT_SUFFIXES
    ]
    if not files:
        raise FileNotFoundError(f"no baseline weight files found under: {model_path}")
    return sorted(files)


def collect_memrift_files(comp_dir: Path) -> list[Path]:
    if not comp_dir.is_dir():
        raise FileNotFoundError(f"MemRift compressed directory not found: {comp_dir}")
    index_file = comp_dir / "index.json"
    if not index_file.is_file():
        raise FileNotFoundError(f"missing MemRift index: {index_file}")

    files = [path for path in comp_dir.rglob("*") if path.is_file()]
    if not files:
        raise FileNotFoundError(f"no MemRift files found under: {comp_dir}")
    return sorted(files)


def read_files_once(files: Iterable[Path], chunk_size: int) -> tuple[int, int]:
    total_bytes = 0
    checksum = 0
    for path in files:
        with path.open("rb", buffering=0) as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                checksum = (checksum + chunk[0] + chunk[-1]) & 0xFFFFFFFF
    return total_bytes, checksum


def measure_read(label: str, root: Path, files: list[Path], repeat: int, chunk_size: int) -> dict:
    samples = []
    total_bytes = 0
    checksum = 0

    for _ in range(repeat):
        t0 = time.perf_counter()
        total_bytes, checksum = read_files_once(files, chunk_size)
        elapsed = time.perf_counter() - t0
        samples.append(elapsed)

    best = min(samples)
    mean = sum(samples) / len(samples)
    return {
        "kind": label,
        "root": str(root),
        "num_files": len(files),
        "bytes_read": total_bytes,
        "gib_read": total_bytes / (1024**3),
        "read_seconds": best,
        "read_seconds_mean": mean,
        "read_seconds_samples": samples,
        "repeat": repeat,
        "chunk_size_bytes": chunk_size,
        "checksum_guard": checksum,
        "files": [str(path) for path in files],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--compressed-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--target-reduction", type=float, default=0.30)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    args = ap.parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    model_path = Path(args.model_path)
    comp_dir = Path(args.compressed_dir)

    baseline_files = collect_baseline_weight_files(model_path)
    memrift_files = collect_memrift_files(comp_dir)

    baseline = measure_read(
        "baseline_model_weights_disk_read",
        model_path,
        baseline_files,
        args.repeat,
        args.chunk_size,
    )
    memrift = measure_read(
        "memrift_compressed_weights_disk_read",
        comp_dir,
        memrift_files,
        args.repeat,
        args.chunk_size,
    )

    baseline_time = baseline["read_seconds"]
    memrift_time = memrift["read_seconds"]
    reduction = (
        (baseline_time - memrift_time) / baseline_time
        if baseline_time > 0
        else None
    )

    data = {
        "metric": "load_time_reduction",
        "model_key": args.model_key,
        "model_name": args.model_name,
        "criterion": "MemRift compressed-weight disk read time >= 30% lower than baseline model weight disk read time",
        "measurement": (
            "This metric measures inference-time model weight loading as disk read time only. "
            "Baseline reads raw model weight files from MODEL_PATH; MemRift reads files from "
            "MEMRIFT_WEIGHT_DIR, including index.json and compressed payload files. It does not "
            "instantiate the model and does not run first forward."
        ),
        "comparison_time_field": (
            "baseline_model_weights.read_seconds vs "
            "memrift_compressed_weights.read_seconds"
        ),
        "baseline_model_weights": baseline,
        "memrift_compressed_weights": memrift,
        "reduction_fraction": reduction,
        "reduction_percent": reduction * 100.0 if reduction is not None else None,
        "target_reduction_fraction": args.target_reduction,
        "pass": bool(reduction is not None and reduction >= args.target_reduction),
        "cache_note": (
            "The measured time can be affected by the OS page cache. For cold-cache results, "
            "run on a clean machine or clear page cache according to the host policy before each branch."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
