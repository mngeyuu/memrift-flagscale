"""
Distributed offline compression for Megatron TP/PP checkpoints.

Each rank compresses its own weight shards and saves them to:
    <outdir>/tp{tp_rank}_pp{pp_rank}/

Run with torchrun so each rank handles its own slice:

    torchrun --nnodes 1 --nproc_per_node <TP_SIZE> \\
        flagscale/compress/memrift/offline_comp/compress_megatron_tp.py \\
        --flagscale-config examples/mistral_7b/conf/train/7b.yaml \\
        --outdir /path/to/comp_weights \\
        --level 3

For single-GPU (TP=1, PP=1) this is equivalent to prepare_weight.py but
outputs in the shard-mode format (tp0_pp0/index.json) understood by
MegatronDynamicLoader in shard mode.

The script:
  1. Initialises Megatron with the same YAML config used for training.
  2. Builds the model with the target TP/PP partition.
  3. Each rank compresses its own parameters and writes tp{N}_pp{M}/.
  4. index.json uses Megatron-format parameter names.

Compressed file format (same as prepare_weight.py):
    [8 bytes little-endian uint64: numel]
    [numel bytes: sign+mantissa (1 B/elem for bf16)]
    [zstd-compressed exponent bytes]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import zstandard as zstd

# ── path setup ────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", "..", "..", ".."))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "flagscale", "train"))
# ──────────────────────────────────────────────────────────────────────────────

MiB = 1024 * 1024


def parse_args():
    p = argparse.ArgumentParser(
        description="Distributed offline MemRift compression for Megatron TP/PP"
    )
    p.add_argument(
        "--flagscale-config",
        required=True,
        help="Path to FlagScale training YAML (used to init Megatron + build model)",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Base output directory.  Per-rank output goes to <outdir>/tp{N}_pp{M}/",
    )
    p.add_argument(
        "--level",
        type=int,
        default=3,
        help="Zstd compression level (1-22, default: 3)",
    )
    p.add_argument(
        "--skip-non-bf16",
        action="store_true",
        default=False,
        help="Skip parameters that are not bf16 / fp16 (e.g. RMSNorm weights)",
    )
    p.add_argument(
        "--min-numel",
        type=int,
        default=512,
        help="Skip parameters with fewer than this many elements (default: 512)",
    )
    return p.parse_args()


def _compress_param(
    param: torch.Tensor,
    name: str,
    fs_sp,
    cpr: zstd.ZstdCompressor,
    outdir: str,
    file_idx: int,
    index: list,
    skip_non_bf16: bool,
    min_numel: int,
) -> bool:
    """Compress one parameter.  Returns True if written, False if skipped."""
    if param.numel() < min_numel:
        return False

    if param.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False

    if skip_non_bf16 and param.dtype == torch.float32:
        return False

    t = param.data.contiguous()
    if t.dtype == torch.float32:
        t = t.to(torch.bfloat16)

    stm = torch.cuda.current_stream()
    with torch.cuda.stream(stm):
        cpu_exp, sm_bits = fs_sp.split(t, stm.cuda_stream)
        evt = stm.record_event()
    evt.synchronize()

    sm_cpu = sm_bits.cpu().contiguous()
    del sm_bits

    fn = f"{file_idx:06d}.bin"
    fpath = os.path.join(outdir, fn)
    numel = t.numel()

    with open(fpath, "wb") as f:
        f.write(struct.pack("<Q", numel))
        f.write(sm_cpu.numpy().tobytes())
        t0 = time.time()
        exp_bytes = cpr.compress(cpu_exp.numpy().tobytes())
        comp_ms = (time.time() - t0) * 1000

    dtype_str = str(t.dtype).replace("torch.", "")
    index.append(
        dict(
            name=name,
            file=fn,
            shape=list(t.shape),
            dtype=dtype_str,
            scheme="split_zstd",
        )
    )

    sm_mb = sm_cpu.numel() / MiB
    exp_mb = len(exp_bytes) / MiB
    raw_mb = numel / MiB  # 1 byte/elem for bf16 exponent
    ratio = raw_mb / (sm_mb + exp_mb) if (sm_mb + exp_mb) > 0 else 0

    del cpu_exp, sm_cpu
    return True


def main():
    args = parse_args()

    # ── Initialise Megatron + distributed ────────────────────────────────────
    # We reuse the same initialisation path as the training script so that
    # TP / PP groups are set up identically.
    from megatron.training.initialize import initialize_megatron
    from megatron.training import get_args as megatron_get_args
    from megatron.core import parallel_state as ps

    # Inject the YAML config path so Megatron's hydra/argparse picks it up
    os.environ.setdefault("MEGATRON_CONFIG", args.flagscale_config)

    initialize_megatron(
        args_defaults={
            "train_iters": 1,
            "eval_iters": 0,
            "micro_batch_size": 1,
            "global_batch_size": 1,
            "lr": 1e-4,
            "min_lr": 1e-5,
            "lr_decay_style": "cosine",
            "lr_warmup_iters": 0,
            "mock_data": True,
            "split": "1,0,0",
            "no_load_optim": True,
            "no_load_rng": True,
        }
    )

    mg_args = megatron_get_args()
    tp_rank = ps.get_tensor_model_parallel_rank()
    tp_size = ps.get_tensor_model_parallel_world_size()
    pp_rank = ps.get_pipeline_model_parallel_rank()
    pp_size = ps.get_pipeline_model_parallel_world_size()
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    rank_outdir = os.path.join(args.outdir, f"tp{tp_rank}_pp{pp_rank}")
    os.makedirs(rank_outdir, exist_ok=True)

    print(
        f"[rank{global_rank}] tp={tp_rank}/{tp_size} pp={pp_rank}/{pp_size} "
        f"→ {rank_outdir}",
        flush=True,
    )

    # ── Build model (same as training) ───────────────────────────────────────
    from functools import partial
    from megatron.training import get_model
    from megatron.core.enums import ModelType

    sys.path.insert(0, os.path.join(_root, "flagscale", "train"))
    from model_provider import model_provider
    from gpt_builders import gpt_builder

    model_list = get_model(
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
    )
    model = model_list[0]
    model.eval()

    # Unwrap Float16Module / DDP wrappers
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module

    # ── Load CUDA extension ──────────────────────────────────────────────────
    try:
        from flagscale.compress.float_split_stride_pin import (
            float_split_stride_pin as fs_sp,
        )
        assert fs_sp.is_available(), "CUDA extension not available"
    except Exception as e:
        print(f"[rank{global_rank}] ERROR: {e}", flush=True)
        raise

    cpr = zstd.ZstdCompressor(level=args.level, write_checksum=False)
    index: list = []
    file_idx = 0
    skipped = 0
    written = 0
    total_sm_mb = 0.0
    total_exp_mb = 0.0
    total_raw_mb = 0.0

    t_start = time.perf_counter()

    for param_name, param in unwrapped.named_parameters():
        if param.numel() < args.min_numel:
            skipped += 1
            continue
        if param.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            skipped += 1
            continue
        if args.skip_non_bf16 and param.dtype == torch.float32:
            skipped += 1
            continue

        t = param.data.contiguous()
        if t.dtype == torch.float32:
            t = t.to(torch.bfloat16)

        stm = torch.cuda.current_stream()
        with torch.cuda.stream(stm):
            cpu_exp, sm_bits = fs_sp.split(t, stm.cuda_stream)
            evt = stm.record_event()
        evt.synchronize()

        sm_cpu = sm_bits.cpu().contiguous()
        del sm_bits

        fn = f"{file_idx:06d}.bin"
        fpath = os.path.join(rank_outdir, fn)
        numel = t.numel()

        with open(fpath, "wb") as fp:
            fp.write(struct.pack("<Q", numel))
            fp.write(sm_cpu.numpy().tobytes())
            exp_bytes = cpr.compress(cpu_exp.numpy().tobytes())
            fp.write(exp_bytes)

        dtype_str = str(t.dtype).replace("torch.", "")
        index.append(
            dict(
                name=param_name,
                file=fn,
                shape=list(t.shape),
                dtype=dtype_str,
                scheme="split_zstd",
            )
        )

        sm_mb = sm_cpu.numel() / MiB
        exp_mb = len(exp_bytes) / MiB
        raw_mb = numel / MiB
        total_sm_mb += sm_mb
        total_exp_mb += exp_mb
        total_raw_mb += raw_mb

        if global_rank == 0:
            ratio = raw_mb / (sm_mb + exp_mb) if (sm_mb + exp_mb) > 0 else 0
            print(
                f"  [{file_idx:4d}] {param_name}: {list(t.shape)} "
                f"sm={sm_mb:.1f}MB exp={exp_mb:.1f}MB ratio={ratio:.2f}x",
                flush=True,
            )

        del cpu_exp, sm_cpu
        file_idx += 1
        written += 1

    # ── Write index.json ─────────────────────────────────────────────────────
    index_path = os.path.join(rank_outdir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    elapsed = time.perf_counter() - t_start
    orig_mb = total_sm_mb + total_raw_mb  # sm + uncompressed exp ≈ 2 B/elem for bf16
    comp_mb = total_sm_mb + total_exp_mb
    ratio = orig_mb / comp_mb if comp_mb > 0 else 0

    print(
        f"[rank{global_rank}] Done: {written} params, {skipped} skipped, "
        f"{elapsed:.1f}s\n"
        f"  bf16 size ≈ {total_raw_mb*2:.0f} MB → compressed {comp_mb:.0f} MB "
        f"(ratio {ratio:.2f}x)\n"
        f"  index: {index_path}",
        flush=True,
    )

    # All ranks must finish before exit
    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
