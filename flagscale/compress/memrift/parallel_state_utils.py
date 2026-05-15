"""
Megatron parallel state utilities for MemRift.

Safe wrappers around megatron.core.parallel_state that fall back to 0/1
when Megatron is not initialized or TP/PP is disabled.
"""

from __future__ import annotations
from typing import Tuple


def get_tp_rank() -> int:
    """Tensor parallel rank of the current process (0 for non-TP)."""
    try:
        from megatron.core import parallel_state as ps
        return ps.get_tensor_model_parallel_rank()
    except Exception:
        return 0


def get_tp_size() -> int:
    """Tensor parallel world size (1 for non-TP)."""
    try:
        from megatron.core import parallel_state as ps
        return ps.get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def get_pp_rank() -> int:
    """Pipeline parallel rank of the current process (0 for non-PP)."""
    try:
        from megatron.core import parallel_state as ps
        return ps.get_pipeline_model_parallel_rank()
    except Exception:
        return 0


def get_pp_size() -> int:
    """Pipeline parallel world size (1 for non-PP)."""
    try:
        from megatron.core import parallel_state as ps
        return ps.get_pipeline_model_parallel_world_size()
    except Exception:
        return 1


def get_tp_pp_rank() -> Tuple[int, int]:
    """Return (tp_rank, pp_rank) tuple."""
    return get_tp_rank(), get_pp_rank()


def get_tp_pp_size() -> Tuple[int, int]:
    """Return (tp_size, pp_size) tuple."""
    return get_tp_size(), get_pp_size()


def pp_layer_offset(total_layers: int, pp_size: int, pp_rank: int) -> int:
    """
    Global layer index of the first layer on this PP rank.

    Megatron distributes layers evenly: PP rank r owns
    [r * (L // PP), (r+1) * (L // PP)).
    """
    if pp_size <= 1:
        return 0
    layers_per_stage = total_layers // pp_size
    return pp_rank * layers_per_stage


def resolve_comp_dir(base_dir: str, tp_rank: int, pp_rank: int) -> str:
    """
    Find the compressed-weight directory for this TP/PP rank.

    Search priority (first that exists wins):
      1. <base_dir>/tp<N>_pp<M>/   — per-rank shard directory
      2. <base_dir>/tp<N>/         — TP-only directory (PP=0 assumed)
      3. <base_dir>/               — single-GPU legacy directory

    Raises FileNotFoundError if none of the candidates exist.
    """
    import os

    candidates = [
        os.path.join(base_dir, f"tp{tp_rank}_pp{pp_rank}"),
        os.path.join(base_dir, f"tp{tp_rank}"),
        base_dir,
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"[MemRift] No compressed-weight directory for tp={tp_rank} pp={pp_rank}. "
        f"Tried: {candidates}"
    )


def is_shard_index(index_path: str) -> bool:
    """
    Return True if the index.json at `index_path` uses Megatron-format
    parameter names (shard mode) rather than HuggingFace names (HF mode).

    Heuristic: shard-mode names contain 'self_attention' or 'linear_qkv'
    or start with 'decoder.layers'; HF-mode names use 'self_attn', 'q_proj', etc.
    """
    import json

    try:
        with open(index_path) as f:
            index = json.load(f)
        if not index:
            return False
        sample_name = index[0].get("name", "")
        shard_keywords = ("self_attention", "linear_qkv", "linear_fc1",
                          "linear_fc2", "linear_proj")
        hf_keywords = ("self_attn", "q_proj", "k_proj", "v_proj",
                       "gate_proj", "up_proj", "down_proj")
        if any(kw in sample_name for kw in shard_keywords):
            return True
        if any(kw in sample_name for kw in hf_keywords):
            return False
        # Fallback: check for 'decoder.layers'
        return "decoder.layers" in sample_name and "self_attention" in sample_name
    except Exception:
        return False
