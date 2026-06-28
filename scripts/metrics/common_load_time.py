#!/usr/bin/env python3
"""Compare baseline model load time with MemRift compressed-weight load time."""
from __future__ import annotations

import argparse
import json
import os
import struct
import time
from pathlib import Path


def load_memrift_dir(comp_dir: Path, pin_memory: bool = False) -> dict:
    import torch

    index_file = comp_dir / "index.json"
    if not index_file.is_file():
        raise FileNotFoundError(f"missing MemRift index: {index_file}")
    index = json.loads(index_file.read_text(encoding="utf-8"))
    total_bytes = 0
    t0 = time.perf_counter()
    loaded = []
    for entry in index:
        fpath = comp_dir / entry["file"]
        with fpath.open("rb") as f:
            raw = f.read()
        total_bytes += len(raw)
        if len(raw) >= 8:
            numel = struct.unpack("<Q", raw[:8])[0]
            sm_size = numel if entry.get("dtype") == "bfloat16" else numel * 3
            sm = bytearray(raw[8 : 8 + sm_size])
            exp = raw[8 + sm_size :]
            if pin_memory and torch.cuda.is_available():
                t = torch.empty(len(sm), dtype=torch.uint8, pin_memory=True)
                t.copy_(torch.frombuffer(sm, dtype=torch.uint8))
                loaded.append((t, exp))
            else:
                loaded.append((sm, exp))
        else:
            loaded.append(raw)
    elapsed = time.perf_counter() - t0
    del loaded
    return {
        "kind": "memrift_compressed_dir",
        "path": str(comp_dir),
        "num_entries": len(index),
        "bytes_read": total_bytes,
        "load_seconds": elapsed,
    }


def load_baseline_model(model_path: str, device: str, first_forward_tokens: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=Path(model_path).exists(),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        local_files_only=Path(model_path).exists(),
    )
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()
    t_loaded = time.perf_counter()
    vocab = getattr(model.config, "vocab_size", 32000)
    ids = torch.randint(0, min(vocab, 50000), (1, first_forward_tokens), device=next(model.parameters()).device)
    with torch.no_grad():
        model(ids)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()
    t_ready = time.perf_counter()
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "kind": "baseline_hf_model",
        "path": model_path,
        "load_seconds": t_loaded - t0,
        "load_to_first_forward_seconds": t_ready - t0,
        "first_forward_tokens": first_forward_tokens,
        "peak_allocated_bytes": peak,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--compressed-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--first-forward-tokens", type=int, default=8)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--target-reduction", type=float, default=0.30)
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    baseline = None if args.skip_baseline else load_baseline_model(
        args.model_path, args.device, args.first_forward_tokens
    )
    memrift = load_memrift_dir(Path(args.compressed_dir))
    baseline_time = baseline["load_to_first_forward_seconds"] if baseline else None
    memrift_time = memrift["load_seconds"]
    reduction = (
        (baseline_time - memrift_time) / baseline_time
        if baseline_time and baseline_time > 0
        else None
    )
    data = {
        "baseline": baseline,
        "memrift": memrift,
        "comparison_time_field": "baseline.load_to_first_forward_seconds vs memrift.load_seconds",
        "reduction_fraction": reduction,
        "reduction_percent": reduction * 100.0 if reduction is not None else None,
        "target_reduction_fraction": args.target_reduction,
        "pass": bool(reduction is not None and reduction >= args.target_reduction),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
