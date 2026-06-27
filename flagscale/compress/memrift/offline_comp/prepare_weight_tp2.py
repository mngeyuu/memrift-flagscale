"""
Direct HF → MemRift shard-mode compressed weights for TP>1.

Converts HF Mistral-7B (or similar GQA model) weights directly into
MemRift's shard-mode format without going through Megatron checkpoint
conversion.

Output structure:
    <outdir>/
      tp0_pp0/index.json   <- rank 0 weights
      tp0_pp0/000000.bin
      ...
      tp1_pp0/index.json   <- rank 1 weights
      tp1_pp0/000000.bin
      ...

Usage:
    python prepare_weight_tp2.py \\
        --model-dir /path/to/Mistral-7B \\
        --outdir    ./memrift_weights/mistral_7b_tp2_level3 \\
        --tp-size   2 \\
        --level     3

Supported architectures: Mistral-7B style (GQA, SwiGLU, RoPE)
  - num_attention_heads / num_key_value_heads (GQA)
  - SwiGLU: gate_proj + up_proj → linear_fc1
  - untie_embeddings_and_output_weights=True
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import zstandard as zstd

# ── path setup ────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", "..", ".."))
sys.path.insert(0, _root)
# ──────────────────────────────────────────────────────────────────────────────

MiB = 1024 * 1024


def parse_args():
    p = argparse.ArgumentParser(
        description="HF → MemRift TP-sharded compression (no Megatron required)"
    )
    p.add_argument("--model-dir", required=True,
                   help="HF model directory (contains config.json + safetensors/bin)")
    p.add_argument("--outdir", required=True,
                   help="Base output dir.  Per-rank: <outdir>/tp{N}_pp0/")
    p.add_argument("--tp-size", type=int, default=2,
                   help="Tensor parallel size (default: 2)")
    p.add_argument("--level", type=int, default=3,
                   help="Zstd compression level 1-22 (default: 3)")
    p.add_argument("--min-numel", type=int, default=512,
                   help="Skip params with fewer elements (default: 512)")
    p.add_argument("--bf16", action="store_true", default=True,
                   help="Cast weights to bf16 before compressing (default: True)")
    return p.parse_args()


# ── Weight loading ─────────────────────────────────────────────────────────────

def load_hf_weights(model_dir: str) -> Dict[str, torch.Tensor]:
    """Load all HF weights into a name→tensor dict (CPU, bf16)."""
    model_dir = Path(model_dir)

    # Try safetensors first (preferred)
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if safetensor_files:
        try:
            from safetensors.torch import load_file
            state = {}
            for f in safetensor_files:
                print(f"  Loading {f.name} ...", flush=True)
                state.update(load_file(str(f), device="cpu"))
            return state
        except ImportError:
            print("safetensors not installed, falling back to .bin", flush=True)

    # Fallback: pytorch_model*.bin
    bin_files = sorted(model_dir.glob("pytorch_model*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No weight files found in {model_dir}")
    state = {}
    for f in bin_files:
        print(f"  Loading {f.name} ...", flush=True)
        state.update(torch.load(str(f), map_location="cpu"))
    return state


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


# ── TP sharding helpers ────────────────────────────────────────────────────────

def col_shard(t: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Column-parallel shard: split along dim 0 (output features)."""
    chunk = t.shape[0] // tp_size
    return t[tp_rank * chunk: (tp_rank + 1) * chunk].contiguous()


def row_shard(t: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Row-parallel shard: split along dim 1 (input features)."""
    chunk = t.shape[1] // tp_size
    return t[:, tp_rank * chunk: (tp_rank + 1) * chunk].contiguous()


def vocab_shard(t: torch.Tensor, tp_rank: int, tp_size: int,
                padded_vocab_size: int) -> torch.Tensor:
    """Vocab-parallel shard: split along vocab dim 0."""
    chunk = padded_vocab_size // tp_size
    shard = t[tp_rank * chunk: (tp_rank + 1) * chunk]
    return shard.contiguous()


# ── Compression ───────────────────────────────────────────────────────────────

def compress_tensor(
    t: torch.Tensor,
    name: str,
    fs_sp,
    cpr: zstd.ZstdCompressor,
    outdir: str,
    file_idx: int,
    index: list,
    min_numel: int,
) -> int:
    """Compress one tensor.  Returns 1 if written, 0 if skipped."""
    if t.numel() < min_numel:
        return 0

    t = t.contiguous()
    if t.dtype == torch.float32:
        t = t.to(torch.bfloat16)
    if t.dtype not in (torch.bfloat16, torch.float16):
        return 0

    t_gpu = t.cuda()
    stm = torch.cuda.current_stream()
    with torch.cuda.stream(stm):
        cpu_exp, sm_bits = fs_sp.split(t_gpu, stm.cuda_stream)
        evt = stm.record_event()
    evt.synchronize()
    del t_gpu

    sm_cpu = sm_bits.cpu().contiguous()
    del sm_bits

    fn = f"{file_idx:06d}.bin"
    fpath = os.path.join(outdir, fn)
    numel = t.numel()

    with open(fpath, "wb") as fp:
        fp.write(struct.pack("<Q", numel))
        fp.write(sm_cpu.numpy().tobytes())
        exp_bytes = cpr.compress(cpu_exp.numpy().tobytes())
        fp.write(exp_bytes)

    dtype_str = str(t.dtype).replace("torch.", "")
    index.append(dict(
        name=name,
        file=fn,
        shape=list(t.shape),
        dtype=dtype_str,
        scheme="split_zstd",
    ))

    del cpu_exp, sm_cpu
    return 1


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load CUDA extension
    try:
        from flagscale.compress.float_split_stride_pin import (
            float_split_stride_pin as fs_sp,
        )
        assert fs_sp.is_available(), "CUDA extension not available"
    except Exception as e:
        raise RuntimeError(
            f"MemRift CUDA extension not found: {e}\n"
            "Build it: cd flagscale/compress/float_split_stride_pin && pip install -e ."
        )

    cfg = load_config(args.model_dir)
    num_layers       = cfg["num_hidden_layers"]
    hidden_size      = cfg["hidden_size"]
    ffn_hidden_size  = cfg["intermediate_size"]
    num_heads        = cfg["num_attention_heads"]
    num_kv_heads     = cfg["num_key_value_heads"]
    head_dim         = hidden_size // num_heads
    vocab_size       = cfg["vocab_size"]
    tp_size          = args.tp_size

    # Pad vocab_size to be divisible by tp_size (Megatron's make_vocab_size_divisible_by=64
    # with TP=2 means divisible by 128; Mistral-7B vocab=32000 already satisfies this)
    make_vocab_div_by = 64
    padded_vocab = ((vocab_size + tp_size * make_vocab_div_by - 1)
                    // (tp_size * make_vocab_div_by)) * (tp_size * make_vocab_div_by)

    print(f"Config: layers={num_layers}, hidden={hidden_size}, ffn={ffn_hidden_size}, "
          f"heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, "
          f"vocab={vocab_size} (padded={padded_vocab}), tp_size={tp_size}", flush=True)

    print("\nLoading HF weights ...", flush=True)
    state = load_hf_weights(args.model_dir)
    print(f"Loaded {len(state)} tensors.\n", flush=True)

    cpr = zstd.ZstdCompressor(level=args.level, write_checksum=False)
    t_start = time.perf_counter()

    for tp_rank in range(tp_size):
        rank_dir = os.path.join(args.outdir, f"tp{tp_rank}_pp0")
        os.makedirs(rank_dir, exist_ok=True)
        index: List[dict] = []
        file_idx = 0
        written = 0

        print(f"=== tp_rank={tp_rank} → {rank_dir} ===", flush=True)

        def _compress(t: torch.Tensor, mg_name: str) -> None:
            nonlocal file_idx, written
            n = compress_tensor(t, mg_name, fs_sp, cpr, rank_dir,
                                file_idx, index, args.min_numel)
            file_idx += n
            written  += n

        # ── Non-layer weights ──────────────────────────────────────────────

        # Embedding (vocab-parallel, col shard along vocab dim)
        if "model.embed_tokens.weight" in state:
            emb = state["model.embed_tokens.weight"].to(torch.bfloat16)
            if emb.shape[0] < padded_vocab:
                # Pad vocab dim with zeros
                pad = torch.zeros(padded_vocab - emb.shape[0], emb.shape[1], dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=0)
            emb_shard = vocab_shard(emb, tp_rank, tp_size, padded_vocab)
            _compress(emb_shard, "embedding.word_embeddings.weight")
            print(f"  [embed] {emb.shape} → shard {emb_shard.shape}", flush=True)

        # Final layernorm (replicated)
        if "model.norm.weight" in state:
            _compress(state["model.norm.weight"].to(torch.bfloat16),
                      "decoder.final_layernorm.weight")

        # Output layer (vocab-parallel, same split as embedding)
        if "lm_head.weight" in state:
            lm = state["lm_head.weight"].to(torch.bfloat16)
            if lm.shape[0] < padded_vocab:
                pad = torch.zeros(padded_vocab - lm.shape[0], lm.shape[1], dtype=lm.dtype)
                lm = torch.cat([lm, pad], dim=0)
            lm_shard = vocab_shard(lm, tp_rank, tp_size, padded_vocab)
            _compress(lm_shard, "output_layer.weight")

        # ── Per-layer weights ──────────────────────────────────────────────
        for i in range(num_layers):
            prefix = f"model.layers.{i}"
            mg_prefix = f"decoder.layers.{i}"

            # ── Attention ────────────────────────────────────────────────

            # QKV (column-parallel, merged after sharding)
            q = state.get(f"{prefix}.self_attn.q_proj.weight", None)
            k = state.get(f"{prefix}.self_attn.k_proj.weight", None)
            v = state.get(f"{prefix}.self_attn.v_proj.weight", None)
            if q is not None and k is not None and v is not None:
                q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
                q_s = col_shard(q, tp_rank, tp_size)
                k_s = col_shard(k, tp_rank, tp_size)
                v_s = col_shard(v, tp_rank, tp_size)
                qkv_s = torch.cat([q_s, k_s, v_s], dim=0)
                _compress(qkv_s, f"{mg_prefix}.self_attention.linear_qkv.weight")
                if i == 0:
                    print(f"  [layer 0 QKV] q{q.shape}+k{k.shape}+v{v.shape}"
                          f" → shard {qkv_s.shape}", flush=True)

            # Input layernorm — TE fused into linear_qkv; replicated across TP
            ln_in = state.get(f"{prefix}.input_layernorm.weight", None)
            if ln_in is not None:
                _compress(ln_in.to(torch.bfloat16),
                          f"{mg_prefix}.self_attention.linear_qkv.layer_norm_weight")

            # O-proj (row-parallel)
            o = state.get(f"{prefix}.self_attn.o_proj.weight", None)
            if o is not None:
                o_s = row_shard(o.to(torch.bfloat16), tp_rank, tp_size)
                _compress(o_s, f"{mg_prefix}.self_attention.linear_proj.weight")
                if i == 0:
                    print(f"  [layer 0 o_proj] {o.shape} → shard {o_s.shape}", flush=True)

            # ── MLP ──────────────────────────────────────────────────────

            # FC1 = gate + up concatenated (column-parallel, each half sharded)
            gate = state.get(f"{prefix}.mlp.gate_proj.weight", None)
            up   = state.get(f"{prefix}.mlp.up_proj.weight",   None)
            if gate is not None and up is not None:
                gate, up = gate.to(torch.bfloat16), up.to(torch.bfloat16)
                gate_s = col_shard(gate, tp_rank, tp_size)
                up_s   = col_shard(up,   tp_rank, tp_size)
                fc1_s  = torch.cat([gate_s, up_s], dim=0)
                _compress(fc1_s, f"{mg_prefix}.mlp.linear_fc1.weight")
                if i == 0:
                    print(f"  [layer 0 FC1] gate{gate.shape}+up{up.shape}"
                          f" → shard {fc1_s.shape}", flush=True)

            # Post-attention layernorm — TE fused into linear_fc1; replicated
            ln_post = state.get(f"{prefix}.post_attention_layernorm.weight", None)
            if ln_post is not None:
                _compress(ln_post.to(torch.bfloat16),
                          f"{mg_prefix}.mlp.linear_fc1.layer_norm_weight")

            # FC2 = down_proj (row-parallel)
            down = state.get(f"{prefix}.mlp.down_proj.weight", None)
            if down is not None:
                down_s = row_shard(down.to(torch.bfloat16), tp_rank, tp_size)
                _compress(down_s, f"{mg_prefix}.mlp.linear_fc2.weight")
                if i == 0:
                    print(f"  [layer 0 FC2] {down.shape} → shard {down_s.shape}", flush=True)

        # ── Write index.json ──────────────────────────────────────────────
        idx_path = os.path.join(rank_dir, "index.json")
        with open(idx_path, "w") as f:
            json.dump(index, f, indent=2)

        elapsed = time.perf_counter() - t_start
        print(f"  tp{tp_rank}: {written} params written → {idx_path}  [{elapsed:.1f}s]\n",
              flush=True)

    print("=== All ranks done ===")
    total = time.perf_counter() - t_start
    print(f"Total time: {total:.1f}s")

    # Quick sanity check
    print("\nOutput summary:")
    for tp_rank in range(tp_size):
        idx_path = os.path.join(args.outdir, f"tp{tp_rank}_pp0", "index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                idx = json.load(f)
            print(f"  tp{tp_rank}_pp0/index.json  {len(idx)} entries")
        else:
            print(f"  tp{tp_rank}_pp0/index.json  MISSING!")


if __name__ == "__main__":
    main()
