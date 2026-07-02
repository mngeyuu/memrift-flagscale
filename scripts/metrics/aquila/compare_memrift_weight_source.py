#!/usr/bin/env python3
"""Compare Aquila HF, Megatron checkpoint, and MemRift compressed weights.

This is a diagnostic script for accuracy-loss debugging. It samples a few
representative tensors and reports whether the compressed MemRift source matches
the HF weights and converted Megatron checkpoint.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch
import zstandard as zstd
from safetensors.torch import load_file


def load_hf_index(model_dir: Path) -> dict[str, torch.Tensor]:
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    out: dict[str, torch.Tensor] = {}
    cache: dict[str, dict[str, torch.Tensor]] = {}
    for name, file_name in idx["weight_map"].items():
        if file_name not in cache:
            cache[file_name] = load_file(model_dir / file_name, device="cpu")
        out[name] = cache[file_name][name]
    return out


def load_mcore_model(ckpt_dir: Path) -> dict:
    ckpt = torch.load(
        ckpt_dir / "iter_0000001" / "mp_rank_00" / "model_optim_rng.pt",
        map_location="cpu",
        weights_only=False,
    )
    return ckpt["model"]


def flatten(d: dict, prefix: str = ""):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            yield from flatten(v, key)
        else:
            yield key, v


def load_compressed_tensor(comp_dir: Path, hf_name: str) -> torch.Tensor:
    idx = json.loads((comp_dir / "index.json").read_text())
    by_name = {e["name"]: e for e in idx}
    e = by_name[hf_name]
    with (comp_dir / e["file"]).open("rb") as f:
        numel = struct.unpack("<Q", f.read(8))[0]
        sm_size = numel * (1 if e["dtype"] == "bfloat16" else 3)
        sm = f.read(sm_size)
        exp_comp = f.read()
    exp = zstd.ZstdDecompressor().decompress(exp_comp, max_output_size=numel)
    if e["dtype"] != "bfloat16":
        raise NotImplementedError(f"only bf16 diagnostics are implemented, got {e['dtype']}")
    sm_i32 = torch.frombuffer(bytearray(sm), dtype=torch.uint8).to(torch.int32)
    exp_i32 = torch.frombuffer(bytearray(exp), dtype=torch.uint8).to(torch.int32)
    bits = (((sm_i32 & 0x80) * 256) | (exp_i32 * 128) | (sm_i32 & 0x7F)).to(torch.uint16)
    return bits.view(torch.bfloat16).reshape(e["shape"])


def stats(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    a32 = a.float()
    b32 = b.float()
    diff = (a32 - b32).abs()
    denom = b32.abs().mean().clamp_min(1e-12)
    return {
        "name": name,
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "relative_mean_abs_diff": float(diff.mean() / denom),
        "equal": bool(torch.equal(a.cpu(), b.cpu())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="/share/project/mengyc/models/Aquila2-7B")
    ap.add_argument("--mcore-ckpt", default="/share/project/mengyc/models/Aquila2-7B-mcore-tp1")
    ap.add_argument("--memrift-dir", default="/share/project/mengyc/code/memrift-flagscale/memrift_weights/aquila2_7b_level18")
    ap.add_argument("--out", default="/share/project/mengyc/code/memrift-flagscale/output/metrics/aquila_ckpt_compare/weight_source_diff.json")
    args = ap.parse_args()

    hf = load_hf_index(Path(args.hf_model))
    mg_flat = dict(flatten(load_mcore_model(Path(args.mcore_ckpt))))
    comp_dir = Path(args.memrift_dir)

    pairs = [
        (
            "embed",
            "model.embed_tokens.weight",
            "embedding.word_embeddings.weight",
            lambda x: x[:143973],
        ),
        (
            "layer0_qkv",
            ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight"),
            "decoder.layers.0.self_attention.linear_qkv.weight",
            lambda x: x,
        ),
        (
            "layer0_proj",
            "model.layers.0.self_attn.o_proj.weight",
            "decoder.layers.0.self_attention.linear_proj.weight",
            lambda x: x,
        ),
        (
            "layer0_fc1",
            ("model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight"),
            "decoder.layers.0.mlp.linear_fc1.weight",
            lambda x: x,
        ),
        (
            "layer0_fc2",
            "model.layers.0.mlp.down_proj.weight",
            "decoder.layers.0.mlp.linear_fc2.weight",
            lambda x: x,
        ),
        (
            "layer31_qkv",
            ("model.layers.31.self_attn.q_proj.weight", "model.layers.31.self_attn.k_proj.weight", "model.layers.31.self_attn.v_proj.weight"),
            "decoder.layers.31.self_attention.linear_qkv.weight",
            lambda x: x,
        ),
        (
            "final_norm",
            "model.norm.weight",
            "decoder.final_layernorm.weight",
            lambda x: x,
        ),
        (
            "lm_head",
            "lm_head.weight",
            "output_layer.weight",
            lambda x: x[:143973],
        ),
    ]

    results = []
    for label, hf_names, mg_name, trim in pairs:
        if isinstance(hf_names, tuple):
            hf_t = torch.cat([hf[n] for n in hf_names], dim=0)
            comp_t = torch.cat([load_compressed_tensor(comp_dir, n) for n in hf_names], dim=0)
        else:
            hf_t = hf[hf_names]
            comp_t = load_compressed_tensor(comp_dir, hf_names)
        mg_t = trim(mg_flat[mg_name])
        hf_t = trim(hf_t)
        comp_t = trim(comp_t)
        results.append(stats(f"{label}: compressed_vs_hf", comp_t, hf_t))
        results.append(stats(f"{label}: mcore_vs_hf", mg_t, hf_t))
        results.append(stats(f"{label}: compressed_vs_mcore", comp_t, mg_t))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
