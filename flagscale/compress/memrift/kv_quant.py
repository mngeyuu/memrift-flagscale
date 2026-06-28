"""
8-bit KV cache for Megatron static-batching inference (monkey-patch, no source change).

Stores the KV cache in int8 (per-token asymmetric quantization, KIVI-style) to halve
KV-cache memory and extend the maximum prefillable context.

Why this saves memory without a custom attention kernel
-------------------------------------------------------
Megatron's ``Attention._adjust_key_value_for_inference`` writes fresh K/V into a bf16
cache buffer and returns a *view* of that buffer to flash-attn. During PREFILL every
token is new, so the returned view equals the freshly-computed K/V. We therefore:

  * allocate the cache buffer in int8 (+ small fp16 scale/zero)  -> half the bytes
  * during prefill, return the freshly-computed bf16 K/V for attention (no dequant)
  * during decode, dequantize the int8 cache back to bf16 (single short step)

With ``--max-new-tokens 1`` the decode loop never runs, so the int8 cache is
written but never read back -> the prefill peak drops by ~half the KV cache.

8-bit per-token asymmetric quantization is near-lossless (well under 1% on KV).
"""
from typing import Optional, Tuple

import torch

_PATCHED = False


# ──────────────────────────────────────────────────────────────────────────────
# int8 per-token asymmetric quant/dequant
#   x: [seq, batch, heads, head_dim]; scale/zero computed per (seq,batch,heads) row
#   over the head_dim axis.
# ──────────────────────────────────────────────────────────────────────────────
def _quant_int8_per_token(x: torch.Tensor):
    xmin = x.amin(dim=-1, keepdim=True)
    xmax = x.amax(dim=-1, keepdim=True)
    scale = (xmax - xmin).clamp_(min=1e-6) / 255.0
    q = ((x - xmin) / scale).round_().clamp_(0, 255).to(torch.uint8)
    return q, scale.to(torch.float16), xmin.to(torch.float16)


def _dequant_int8_per_token(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, dtype):
    return (q.to(torch.float16) * scale + zero).to(dtype)


def install_int8_kv_cache(print_debug: bool = False):
    """Monkey-patch Attention._adjust_key_value_for_inference for int8 KV cache."""
    global _PATCHED
    if _PATCHED:
        return
    from megatron.core.transformer.attention import Attention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.utils import deprecate_inference_params

    def _adjust_int8(
        self,
        inference_context,
        query,
        key,
        value,
        rotary_pos_emb,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        inference_context = deprecate_inference_params(inference_context, inference_params)
        attn_mask_type = self.attn_mask_type
        if inference_context is None:
            return query, key, value, rotary_pos_emb, attn_mask_type, None

        # Only the static-batching path (generate_gpt.py) is supported here; anything
        # else falls back to the original method.
        if not inference_context.is_static_batching():
            return _ORIG_ADJUST(
                self, inference_context, query, key, value, rotary_pos_emb,
                rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin, sequence_len_offset,
            )
        assert not self.config.flash_decode, "int8 KV cache: flash_decode not supported"

        ln = self.layer_number
        # ── Pre-allocate int8 cache (key/value uint8 + fp16 scale/zero) ──
        if ln not in inference_context.key_value_memory_dict:
            max_seq = inference_context.max_sequence_length
            max_bs = inference_context.max_batch_size
            ng = self.num_query_groups_per_partition
            kd = key.size(-1)
            vd = value.size(-1)
            dev = key.device
            def _z(d, dt):
                return torch.empty(max_seq, max_bs, ng, d, dtype=dt, device=dev)
            inference_context.key_value_memory_dict[ln] = {
                "kq": _z(kd, torch.uint8), "ks": _z(1, torch.float16), "kz": _z(1, torch.float16),
                "vq": _z(vd, torch.uint8), "vs": _z(1, torch.float16), "vz": _z(1, torch.float16),
            }
        c = inference_context.key_value_memory_dict[ln]

        # past-the-prompt: turn off masking (matches original)
        if inference_context.sequence_len_offset > 0 and (
            not self.training or not _IS_TE_22
        ):
            attn_mask_type = AttnMaskType.no_mask

        batch_start = inference_context.batch_size_offset
        batch_end = batch_start + key.size(1)
        seq_start = inference_context.sequence_len_offset
        seq_end = seq_start + key.size(0)
        assert seq_end <= c["kq"].size(0), "sequence longer than inference_max_seq_length"

        # ── Adjust rotary embeddings (identical to original) ──
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            q_pos_emb = q_pos_emb[seq_start:seq_end, :, :, :]
            k_pos_emb = k_pos_emb[:seq_end, :, :, :]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        # ── Quantize fresh K/V into int8 cache ──
        kq, ks, kz = _quant_int8_per_token(key)
        vq, vs, vz = _quant_int8_per_token(value)
        sl = slice(seq_start, seq_end)
        bl = slice(batch_start, batch_end)
        c["kq"][sl, bl, ...] = kq; c["ks"][sl, bl, ...] = ks; c["kz"][sl, bl, ...] = kz
        c["vq"][sl, bl, ...] = vq; c["vs"][sl, bl, ...] = vs; c["vz"][sl, bl, ...] = vz

        if seq_start == 0:
            # PREFILL: all tokens new -> return the fresh bf16 K/V directly (no dequant).
            return query, key, value, rotary_pos_emb, attn_mask_type, None
        else:
            # DECODE: dequantize the full cache up to seq_end for attention.
            kfull = _dequant_int8_per_token(
                c["kq"][:seq_end, bl], c["ks"][:seq_end, bl], c["kz"][:seq_end, bl], key.dtype)
            vfull = _dequant_int8_per_token(
                c["vq"][:seq_end, bl], c["vs"][:seq_end, bl], c["vz"][:seq_end, bl], value.dtype)
            return query, kfull, vfull, rotary_pos_emb, attn_mask_type, None

    global _ORIG_ADJUST, _IS_TE_22
    _ORIG_ADJUST = Attention._adjust_key_value_for_inference
    try:
        from megatron.core.utils import is_te_min_version
        _IS_TE_22 = is_te_min_version("2.2.0")
    except Exception:
        _IS_TE_22 = False
    Attention._adjust_key_value_for_inference = _adjust_int8
    _PATCHED = True
    if print_debug:
        print("[KVQuant] int8 per-token KV cache patch installed", flush=True)
