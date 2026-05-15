"""
MemRift TP/PP-specific forward hooks.

Supplements megatron_dynamic_loader.py with hooks for layers that live
*outside* the standard decoder stack:

  VocabParallelEmbedding:
    - Each TP rank holds vocab_size//TP rows of the embedding table.
    - Compressed on CPU; materialized before embedding lookup, released after.

  PP first/last stage extras:
    - Stage 0 (first): embedding + positional encoding
    - Stage N-1 (last): final layernorm + output projection (lm_head)

Usage:
    from flagscale.compress.memrift.megatron_tp_hooks import (
        install_vocab_embedding_hooks,
        install_output_layer_hooks,
    )

    install_vocab_embedding_hooks(model, loader, async_compressor)
    install_output_layer_hooks(model, loader, async_compressor)
"""

from __future__ import annotations

import time
from typing import Optional, Any

import torch
import torch.nn as nn

from flagscale.compress.memrift.megatron_dynamic_loader import (
    CompressedParam,
    MegatronDynamicLoader,
    _trace,
)


def _find_module(model: nn.Module, *attr_paths: str) -> Optional[nn.Module]:
    """Try multiple dot-separated attribute paths; return first that exists."""
    for path in attr_paths:
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _compress_param(param: nn.Parameter, device, zstd_level: int = 3) -> CompressedParam:
    """Compress a single GPU parameter to CPU, return CompressedParam."""
    from flagscale.compress.memrift.async_compressor import (
        get_compression_ctx,
    )
    try:
        from flagscale.compress.float_split_stride_pin import (
            float_split_stride_pin as fs_sp,
        )
    except ImportError as e:
        raise RuntimeError(f"MemRift CUDA extension not found: {e}")

    t = param.data.contiguous()
    if t.dtype not in (torch.bfloat16, torch.float16):
        t = t.to(torch.bfloat16)

    d2h = torch.cuda.Stream()
    d2h.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(d2h):
        cpu_exp, sm_bits = fs_sp.split(t, d2h.cuda_stream)
        evt = d2h.record_event()
    evt.synchronize()

    import numpy as np
    arr = cpu_exp.numpy()
    cctx = get_compression_ctx(zstd_level)
    exp_bytes = cctx.compress(arr)

    cp = CompressedParam(
        orig_shape=tuple(param.shape),
        sm_cpu=sm_bits,
        exp_mv=exp_bytes,
        dtype=param.dtype,
        device=device,
    )
    cp.layer_idx = -1
    return cp


def _make_embedding_pre_hook(cp: CompressedParam, async_comp):
    """Returns a forward_pre_hook that materializes the embedding weight."""
    def _hook(module, args):
        if cp._bf16 is not None:
            return  # already materialized (shouldn't happen, but safe)

        if async_comp and cp._prefetch_future is not None:
            result = cp._prefetch_future.result()
            cp._prefetch_future = None
            if isinstance(result, tuple):
                bf16, evt, cpu_exp = result
                evt.synchronize()
                del cpu_exp
                cp._bf16 = bf16
            else:
                cp._bf16 = result
        else:
            cp._sm_on_gpu = cp.sm_cpu.to(cp._device, non_blocking=False)
            cp.materialize(sync=True)

        with torch.no_grad():
            module.weight.data = cp._bf16
        _trace("vocab_embed: materialized")

    return _hook


def _make_embedding_post_hook(cp: CompressedParam):
    """Returns a forward_hook that releases the embedding weight after use."""
    def _hook(module, args, output):
        cp.release()
        with torch.no_grad():
            module.weight.data = torch.empty(0, dtype=cp._dtype, device=cp._device)
        _trace("vocab_embed: released")

    return _hook


def install_vocab_embedding_hooks(
    model: nn.Module,
    loader: MegatronDynamicLoader,
    async_comp: Optional[Any] = None,
    zstd_level: int = 3,
) -> bool:
    """
    Compress and install forward hooks on VocabParallelEmbedding.

    The embedding weight (vocab_size//TP × hidden) is compressed to CPU and
    materialised only during the embedding lookup forward, then released.

    Returns True if hooks were successfully installed.
    """
    device = loader.device

    embed_module = _find_module(
        model,
        "embedding.word_embeddings",
        "language_model.embedding.word_embeddings",
        "model.embed_tokens",
    )
    if embed_module is None:
        if loader.print_debug:
            print("[MemRift] install_vocab_embedding_hooks: embedding not found")
        return False

    if not hasattr(embed_module, "weight") or embed_module.weight is None:
        return False

    if embed_module.weight.numel() == 0:
        # Already released (shouldn't happen at install time)
        return False

    cp = _compress_param(embed_module.weight, device, zstd_level)

    # Free GPU weight immediately
    with torch.no_grad():
        embed_module.weight.data = torch.empty(0, dtype=cp._dtype, device=device)

    embed_module.register_forward_pre_hook(_make_embedding_pre_hook(cp, async_comp))
    embed_module.register_forward_hook(_make_embedding_post_hook(cp))

    if loader.print_debug:
        shape = cp.orig_shape
        sm_mb = cp.sm_cpu.numel() / 1024 ** 2
        exp_mb = len(cp.exp_mv) / 1024 ** 2
        print(
            f"[MemRift] VocabEmbedding hook installed: "
            f"shape={shape} sm={sm_mb:.1f}MB exp={exp_mb:.1f}MB"
        )
    return True


def install_output_layer_hooks(
    model: nn.Module,
    loader: MegatronDynamicLoader,
    async_comp: Optional[Any] = None,
    zstd_level: int = 3,
) -> bool:
    """
    Compress and install forward hooks on the output projection (lm_head).

    Only relevant on the last PP stage.  If the weight is tied to the
    embedding it should NOT be double-freed; this function skips tied weights.

    Returns True if hooks were successfully installed.
    """
    device = loader.device

    out_module = _find_module(
        model,
        "output_layer",
        "language_model.output_layer",
        "lm_head",
    )
    if out_module is None:
        return False

    if not hasattr(out_module, "weight") or out_module.weight is None:
        return False

    if out_module.weight.numel() == 0:
        return False  # already released or tied+released by embed hook

    # Check for tied weights (same storage as embedding)
    embed = _find_module(
        model,
        "embedding.word_embeddings",
        "language_model.embedding.word_embeddings",
    )
    if embed is not None and hasattr(embed, "weight"):
        if embed.weight.data_ptr() == out_module.weight.data_ptr():
            if loader.print_debug:
                print("[MemRift] output_layer weight is tied to embedding — skipping separate hook")
            return False

    cp = _compress_param(out_module.weight, device, zstd_level)
    with torch.no_grad():
        out_module.weight.data = torch.empty(0, dtype=cp._dtype, device=device)

    out_module.register_forward_pre_hook(_make_embedding_pre_hook(cp, async_comp))
    out_module.register_forward_hook(_make_embedding_post_hook(cp))

    if loader.print_debug:
        print(
            f"[MemRift] OutputLayer hook installed: shape={cp.orig_shape}"
        )
    return True
