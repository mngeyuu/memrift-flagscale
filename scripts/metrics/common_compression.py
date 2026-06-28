#!/usr/bin/env python3
"""Compute MemRift compressed-size ratio against a BF16 model-size estimate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def disk_usage_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def bf16_size_from_safetensors_index(model_path: Path) -> int | None:
    index_file = model_path / "model.safetensors.index.json"
    config_file = model_path / "model.safetensors"
    total = 0
    if index_file.is_file():
        data = json.loads(index_file.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map", {})
        files = sorted({model_path / name for name in weight_map.values()})
    elif config_file.is_file():
        files = [config_file]
    else:
        files = sorted(model_path.glob("*.safetensors"))
    if not files:
        return None
    try:
        from safetensors import safe_open
    except Exception:
        return None
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                total += t.numel() * 2
    return total


def bf16_size_from_transformers(model_path: str) -> int | None:
    try:
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="meta",
            trust_remote_code=True,
            local_files_only=Path(model_path).exists(),
        )
        total = sum(p.numel() for p in model.parameters()) * 2
        del model
        return int(total)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--compressed-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-saving", type=float, default=0.30)
    args = ap.parse_args()

    model_path = Path(args.model_path)
    comp_dir = Path(args.compressed_dir)
    bf16_bytes = bf16_size_from_safetensors_index(model_path)
    if bf16_bytes is None:
        bf16_bytes = bf16_size_from_transformers(args.model_path)
    compressed_bytes = disk_usage_bytes(comp_dir)
    ratio = (compressed_bytes / bf16_bytes) if bf16_bytes and compressed_bytes else None
    saving = (1.0 - ratio) if ratio is not None else None
    data = {
        "model_path": args.model_path,
        "compressed_dir": args.compressed_dir,
        "bf16_reference_bytes": bf16_bytes,
        "compressed_bytes": compressed_bytes,
        "compressed_over_bf16": ratio,
        "saving_fraction": saving,
        "saving_percent": saving * 100.0 if saving is not None else None,
        "target_saving_fraction": args.target_saving,
        "pass": bool(saving is not None and saving >= args.target_saving),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
