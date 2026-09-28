"""
TransformerEngine ctx-weight release patch for MemRift.

Problem
-------
TE's custom autograd Functions (_LayerNormMLP, _LayerNormLinear, _Linear) store
materialized weight tensors directly on the autograd ctx via attribute assignment
(e.g. `ctx.fc1_weight = fc1_weight`, `ctx.weight = weight`, `ctx.weight_object = weight`).

These bypass PyTorch's `saved_tensors_hooks`, so MemRift's WeightPlaceholder cannot
intercept them. The ctx object is destroyed only when the autograd engine releases
the corresponding backward node, which for LoRA + frozen base happens AFTER all
backward gradients are computed. Result: all 32 decoder layers' bf16 weights stay
pinned on GPU throughout backward = ~15 GB leak.

Fix
---
Monkey-patch each TE Function's `backward` method to clear ctx attributes
that hold weight references right before returning. This ensures the weight ref
is dropped as soon as this layer's backward finishes, freeing the GPU memory
for the next layer's prefetch/materialize.

Usage
-----
Call `patch_te_ctx_release()` once after `import transformer_engine`.
"""
from __future__ import annotations

import os

_PATCHED = False
_TRACE = os.environ.get("MEMRIFT_TE_PATCH_TRACE", "0") == "1"
_CTX_PROBE = os.environ.get("MEMRIFT_TE_CTX_PROBE", "0") == "1"


def _describe_ctx_attrs(ctx, attrs, name):
    if not _CTX_PROBE:
        return
    parts = []
    for a in attrs:
        if not hasattr(ctx, a):
            continue
        try:
            obj = getattr(ctx, a)
        except Exception as exc:
            parts.append(f"{a}=<read_error:{exc}>")
            continue

        try:
            import torch

            if isinstance(obj, torch.Tensor):
                ptr = int(obj.data_ptr()) if obj.is_cuda and obj.numel() > 0 else -1
                nbytes = obj.numel() * obj.element_size()
                parts.append(
                    f"{a}=Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, "
                    f"cuda={obj.is_cuda}, ptr={ptr}, MB={nbytes/1024**2:.1f}, "
                    f"requires_grad={obj.requires_grad})"
                )
            elif isinstance(obj, (list, tuple)):
                tensors = []
                for idx, item in enumerate(obj):
                    if isinstance(item, torch.Tensor):
                        ptr = int(item.data_ptr()) if item.is_cuda and item.numel() > 0 else -1
                        tensors.append(
                            f"{idx}:shape={tuple(item.shape)},dtype={item.dtype},"
                            f"cuda={item.is_cuda},ptr={ptr}"
                        )
                parts.append(f"{a}={type(obj).__name__}(len={len(obj)}, tensors={tensors[:8]})")
            else:
                parts.append(f"{a}={type(obj).__name__}")
        except Exception as exc:
            parts.append(f"{a}=<describe_error:{exc}>")
    if parts:
        print(f"[MEMRIFT_TE_CTX_PROBE] {name}.ctx " + " | ".join(parts), flush=True)


def _clear_ctx_attrs(ctx, attrs):
    for a in attrs:
        if hasattr(ctx, a):
            try:
                setattr(ctx, a, None)
            except Exception:
                pass


def _wrap_backward(cls, attrs_to_clear, name):
    if not hasattr(cls, "backward"):
        return False
    orig_backward = cls.backward
    if getattr(orig_backward, "_memrift_patched", False):
        return False

    def patched_backward(ctx, *grad_outputs):
        try:
            out = orig_backward(ctx, *grad_outputs)
        finally:
            _describe_ctx_attrs(ctx, attrs_to_clear + ("tensor_objects",), name)
            _clear_ctx_attrs(ctx, attrs_to_clear)
            # tensor_objects holds Python wrappers around saved tensors; clear too.
            _clear_ctx_attrs(ctx, ("tensor_objects",))
            if _TRACE:
                print(f"[MEMRIFT_TE_PATCH] cleared {name}.ctx attrs", flush=True)
        return out

    patched_backward._memrift_patched = True  # type: ignore[attr-defined]
    # torch.autograd.Function.backward is a staticmethod
    cls.backward = staticmethod(patched_backward)
    return True


def patch_te_ctx_release() -> int:
    """Patch TE Function classes to release ctx-pinned weights after backward.

    Returns the number of classes patched. Safe to call multiple times.
    """
    global _PATCHED
    if _PATCHED:
        return 0

    patched = 0
    try:
        from transformer_engine.pytorch.module.layernorm_mlp import _LayerNormMLP
        if _wrap_backward(_LayerNormMLP, ("fc1_weight", "fc2_weight"), "_LayerNormMLP"):
            patched += 1
    except ImportError:
        pass

    try:
        from transformer_engine.pytorch.module.layernorm_linear import _LayerNormLinear
        if _wrap_backward(_LayerNormLinear, ("weight",), "_LayerNormLinear"):
            patched += 1
    except ImportError:
        pass

    try:
        from transformer_engine.pytorch.module.linear import _Linear
        if _wrap_backward(_Linear, ("weight_object",), "_Linear"):
            patched += 1
    except ImportError:
        pass

    _PATCHED = True
    return patched
