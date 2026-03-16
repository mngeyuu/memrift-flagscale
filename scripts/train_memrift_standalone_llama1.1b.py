#!/usr/bin/env python3
"""
Standalone MemRift training for TinyLlama 1.1B (Llama 1.1B).

Uses FlagScale's MemRift code (compress/memrift) without the FlagScale/Megatron
training framework. Trains with HuggingFace Transformers + PEFT LoRA, loads
compressed base weights from prepare_weight output, and records peak GPU memory.

Prerequisites:
  1. Prepare compressed weights:
     python -m flagscale.compress.memrift.offline_comp.prepare_weight \
       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --outdir ./memrift_weights/tinyllama_1b_level18 --level 18

  2. Build CUDA extension (required for weight decompression):
     cd flagscale/compress/float_split_stride_pin && pip install -e .

Usage:
  python scripts/train_memrift_standalone_llama1.1b.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --compressed_weights ./memrift_weights/tinyllama_1b_level18 \
    --steps 5 --max_length 512

  # With activation compression and memory profile:
  python scripts/train_memrift_standalone_llama1.1b.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --compressed_weights ./memrift_weights/tinyllama_1b_level18 \
    --steps 5 --activation --profile_memory
"""

import argparse
import gc
import json
import os
import struct
import sys

import numpy as np
import torch

# Add repo root for imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# FlagScale MemRift (no Megatron dependency for this script)
from flagscale.compress.memrift.megatron_dynamic_loader import CompressedParam
from flagscale.compress.memrift.async_compressor import AsyncCompressor
from flagscale.compress.memrift.activation_compression import DecoderLayerWrapper


MB = 1024.0 * 1024.0


def get_layer_name_from_param(param_name: str):
    """e.g. model.layers.0.self_attn.q_proj.weight -> model.layers.0"""
    parts = param_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                int(parts[i + 1])
                return ".".join(parts[: i + 2])
            except ValueError:
                pass
    return None


def get_layer_name_from_module_name(module_name: str):
    """e.g. base_model.model.model.layers.0.self_attn.q_proj -> base_model.model.model.layers.0"""
    parts = module_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                int(parts[i + 1])
                return ".".join(parts[: i + 2])
            except ValueError:
                pass
    return None


def resolve_weight_owner(mod, attr="weight"):
    """Resolve PEFT wrappers to the module that actually owns `attr`."""
    if mod is None:
        return None
    # Prefer unwrapped inner modules first. PEFT wrappers may expose `weight`
    # as a property while the real Parameter used in compute lives in base_layer.
    for unwrap_attr in ("base_layer", "to_wrap", "module"):
        inner = getattr(mod, unwrap_attr, None)
        if inner is not None and hasattr(inner, attr):
            return inner
    if hasattr(mod, attr):
        return mod
    return None


def load_compressed_weights_into_model(model, comp_dir, device):
    """
    Load FlagScale prepare_weight output (index.json + *.bin) into HF model.
    Replaces each matching parameter with a CompressedParam; returns layer2cps
    for installing hooks.
    """
    index_path = os.path.join(comp_dir, "index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Missing {index_path}. Run prepare_weight first.")
    with open(index_path) as f:
        index = json.load(f)

    layer2cps = {}
    name_to_modules = {n: mod for n, mod in model.named_modules()}
    # PEFT wraps base model: param names become base_model.model.model.layers.0...
    peft_prefix = "base_model.model.model." if any(n.startswith("base_model.") for n in name_to_modules) else ""

    for it in index:
        if it.get("scheme") != "split_zstd":
            continue
        name = it["name"]
        # Match HF param name: index has "model.layers.0...", PEFT model has "base_model.model.model.layers.0..."
        mod_name = name.rsplit(".", 1)[0]  # strip .weight or .bias
        if mod_name not in name_to_modules and peft_prefix:
            # index: "model.layers.0..."; PEFT: "base_model.model.model.layers.0..."
            suffix = name.replace("model.", "", 1).rsplit(".", 1)[0]
            mod_name = peft_prefix + suffix
        if mod_name not in name_to_modules:
            continue
        attr = "weight" if name.endswith(".weight") else "bias"
        mod = resolve_weight_owner(name_to_modules[mod_name], attr)
        if mod is None:
            continue
        if not hasattr(mod, attr):
            continue

        file_path = os.path.join(comp_dir, it["file"])
        with open(file_path, "rb") as f:
            numel = struct.unpack("<Q", f.read(8))[0]
            sm_size = numel * (1 if it["dtype"] == "bfloat16" else 3)
            sm_bytes = np.frombuffer(f.read(sm_size), dtype=np.uint8)
            exp_bytes = f.read()

        sm_gpu = torch.as_tensor(sm_bytes, dtype=torch.uint8, device=device)
        dtype = torch.bfloat16 if it["dtype"] == "bfloat16" else torch.float32
        cp = CompressedParam(it["shape"], sm_gpu, exp_bytes, dtype)
        cp.target_module = mod
        cp.target_attr = attr
        cp.hf_name = name
        cp.layer_idx = -1

        if getattr(mod, "_parameters", None) is not None:
            # Write directly to _parameters to support PEFT wrappers that expose
            # weight/bias via properties and may reject register_parameter().
            mod._parameters[attr] = cp
        else:
            setattr(mod, attr, cp)

        layer_name = get_layer_name_from_module_name(mod_name)
        if layer_name is not None:
            layer2cps.setdefault(layer_name, []).append(cp)
            cp.layer_idx = int(layer_name.split(".")[-1])

    return layer2cps


def materialize_non_layer_compressed_params(model, comp_dir, device):
    """Materialize embed / lm_head etc. once (they are not per-layer released)."""
    index_path = os.path.join(comp_dir, "index.json")
    with open(index_path) as f:
        index = json.load(f)
    name_to_modules = {n: mod for n, mod in model.named_modules()}
    peft_prefix = "base_model.model.model." if any(n.startswith("base_model.") for n in name_to_modules) else ""

    for it in index:
        if it.get("scheme") != "split_zstd":
            continue
        name = it["name"]
        mod_name = name.rsplit(".", 1)[0]
        if mod_name not in name_to_modules and peft_prefix:
            suffix = name.replace("model.", "", 1).rsplit(".", 1)[0]
            mod_name = peft_prefix + suffix
        if mod_name not in name_to_modules or get_layer_name_from_module_name(mod_name) is not None:
            continue
        attr = "weight" if name.endswith(".weight") else "bias"
        mod = resolve_weight_owner(name_to_modules[mod_name], attr)
        if mod is None:
            continue
        param = getattr(mod, attr, None)
        if isinstance(param, CompressedParam):
            param.materialize(sync=True)
            param.wait_ready()
            if getattr(mod, "_parameters", None) is not None and attr in mod._parameters:
                mod._parameters[attr] = param._bf16
            else:
                setattr(mod, attr, param._bf16)


def install_weight_hooks(model, layer2cps, device):
    """Install demo-aligned fwd/bwd hooks to materialize/release per layer."""
    name_to_module = {n: mod for n, mod in model.named_modules()}
    ordered_layers = sorted(
        layer2cps.items(),
        key=lambda kv: int(kv[0].split(".")[-1]),
    )

    def _materialize_cps(cps):
        for cp in cps:
            cp.materialize(sync=True)
            cp.wait_ready()
            mod, attr = cp.target_module, cp.target_attr
            if getattr(mod, "_parameters", None) is not None:
                mod._parameters[attr] = cp._bf16
            else:
                setattr(mod, attr, cp._bf16)

    def _release_cps(cps):
        for cp in cps:
            mod, attr = cp.target_module, cp.target_attr
            if getattr(mod, "_parameters", None) is not None:
                mod._parameters[attr] = cp
            else:
                setattr(mod, attr, cp)
            cp.data = torch.empty(0, dtype=cp._dtype, device=device)
            cp.release()

    for idx, (layer_name, cp_list) in enumerate(ordered_layers):
        if layer_name not in name_to_module:
            continue
        layer_module = name_to_module[layer_name]
        is_last = idx == len(ordered_layers) - 1

        def make_pre(cps):
            def _pre(module, inp):
                _materialize_cps(cps)
                return None
            return _pre

        def make_fwd_post(cps, skip_release=False):
            def _post(module, inp, out):
                if not skip_release:
                    _release_cps(cps)
                return None
            return _post

        def make_bwd_pre(cps):
            def _pre(module, grad_out):
                _materialize_cps(cps)
                return None
            return _pre

        def make_bwd_post(cps):
            def _post(module, grad_in, grad_out):
                _release_cps(cps)
                return None
            return _post

        layer_module.register_forward_pre_hook(make_pre(cp_list))
        layer_module.register_forward_hook(make_fwd_post(cp_list, skip_release=is_last))
        layer_module.register_full_backward_pre_hook(make_bwd_pre(cp_list))
        layer_module.register_full_backward_hook(make_bwd_post(cp_list))

    return layer2cps


def install_activation_compression(model, compressor, act_async=True, empty_interval=10):
    """Wrap decoder layers with DecoderLayerWrapper for activation compression."""
    container = model
    # Align with memrift_demo traversal: handle PeftModel / HF wrappers.
    if hasattr(container, "model"):
        container = container.model
    if hasattr(container, "model"):
        container = container.model
    if hasattr(container, "transformer"):
        container = container.transformer

    if hasattr(container, "layers"):
        layers = container.layers
    elif hasattr(container, "h"):  # GPT-style fallback
        layers = container.h
    else:
        return 0

    skip_ptrs = set()
    for name, param in model.named_parameters():
        if "lora" in name.lower() or "adapter" in name.lower():
            try:
                skip_ptrs.add(param.untyped_storage().data_ptr())
            except Exception:
                pass

    wrapped = 0
    for i, layer in enumerate(layers):
        if isinstance(layer, DecoderLayerWrapper):
            continue
        wrapper = DecoderLayerWrapper(
            layer,
            compressor=compressor,
            use_async=act_async,
            release_after_unpack=True,
            skip_storage_ptrs=skip_ptrs or None,
            do_empty=(i % empty_interval == 1),
        )
        layers[i] = wrapper
        wrapped += 1
    return wrapped


def main():
    parser = argparse.ArgumentParser(description="Standalone MemRift train on Llama 1.1B + record peak memory")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF model id or path")
    parser.add_argument("--compressed_weights", default="", help="Path to prepare_weight output (index.json + .bin). Omit when using --lora_only")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--activation", action="store_true", help="Enable activation compression")
    parser.add_argument("--profile_memory", action="store_true", help="Print memory breakdown and peak")
    parser.add_argument("--profile_time", action="store_true", help="Print per-phase iteration time: forward, backward, optimizer.step, zero_grad")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--output_peak", type=str, default="", help="Write peak_mb to this JSON file")
    parser.add_argument("--lora_only", action="store_true", help="Run pure LoRA training (no MemRift); do not pass --compressed_weights")
    args = parser.parse_args()

    if not args.lora_only and not args.compressed_weights:
        parser.error("--compressed_weights is required unless --lora_only is set.")

    if args.lora_only:
        print("Mode: 纯 LoRA 训练 (无 MemRift 权重压缩)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA required for this script.", file=sys.stderr)
        sys.exit(1)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    mem_after_load = torch.cuda.memory_allocated(device) / MB
    print(f"  Model loaded: {mem_after_load:.1f} MB allocated")

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.train()
    mem_after_lora = torch.cuda.memory_allocated(device) / MB
    print(f"  After LoRA: {mem_after_lora:.1f} MB allocated")

    if not args.lora_only:
        print("Loading compressed base weights into model...")
        layer2cps = load_compressed_weights_into_model(model, args.compressed_weights, device)
        materialize_non_layer_compressed_params(model, args.compressed_weights, device)
        install_weight_hooks(model, layer2cps, device)
        torch.cuda.empty_cache()
        mem_after_memrift = torch.cuda.memory_allocated(device) / MB
        print(f"  After MemRift weight injection: {mem_after_memrift:.1f} MB allocated ({len(layer2cps)} layers with CP)")
    else:
        mem_after_memrift = mem_after_lora

    if args.activation:
        print("Enabling activation compression...")
        compressor = AsyncCompressor(
            compress_workers=4,
            decode_workers=4,
            concurrency_limit=4,
            zstd_level=18,
            enable_async=True,
        )
        n_wrapped = install_activation_compression(model, compressor, act_async=True)
        print(f"  Wrapped {n_wrapped} decoder layers with DecoderLayerWrapper")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5,
    )

    # Align measurement with memrift_demo: report training-time peak, not model-load peak.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Dummy batch
    def get_batch():
        inp = tokenizer(
            "The quick brown fox jumps over the lazy dog. " * 50,
            return_tensors="pt",
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
        )
        return {k: v.to(device) for k, v in inp.items()}

    print(f"Running {args.steps} training steps (max_length={args.max_length})...")
    step_peaks = []
    for step in range(args.steps):
        if args.profile_memory:
            torch.cuda.empty_cache()
            # Keep parity with demo-side host cache cleanup when available.
            try:
                torch._C._host_emptyCache()
            except Exception:
                pass
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad()
        batch = get_batch()
        baseline_mb = torch.cuda.memory_allocated(device) / MB if args.profile_memory else 0.0
        out = model(**batch, labels=batch["input_ids"])
        loss = out.loss
        if args.profile_memory:
            torch.cuda.synchronize()
            fwd_alloc_mb = torch.cuda.memory_allocated(device) / MB
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_peak_mb = torch.cuda.max_memory_allocated(device) / MB
        step_peaks.append(step_peak_mb)
        if args.profile_memory:
            bwd_alloc_mb = torch.cuda.memory_allocated(device) / MB
            fwd_activation_mb = fwd_alloc_mb - baseline_mb
            grad_misc_mb = max(0.0, step_peak_mb - fwd_alloc_mb)
            print(f"  [Profile step {step + 1}] baseline={baseline_mb:.1f} MB  fwd_end={fwd_alloc_mb:.1f} MB  bwd_end={bwd_alloc_mb:.1f} MB  peak={step_peak_mb:.1f} MB  fwd_act={fwd_activation_mb:.1f} MB  bwd_tmp={grad_misc_mb:.1f} MB")
        if (step + 1) % 2 == 0 or step == 0:
            print(f"  Step {step + 1}/{args.steps}  loss={loss.item():.4f}  peak_allocated={step_peak_mb:.1f} MB")

    peak_mb = max(step_peaks) if step_peaks else (torch.cuda.max_memory_allocated(device) / MB)
    current_mb = torch.cuda.memory_allocated(device) / MB

    print("\n" + "=" * 60)
    print("显存峰值 (Peak GPU memory)")
    print("=" * 60)
    print(f"  max_memory_allocated:  {peak_mb:.2f} MB")
    print(f"  current allocated:     {current_mb:.2f} MB")
    if args.profile_memory:
        print(f"  (after load):          {mem_after_load:.2f} MB")
        print(f"  (after LoRA):         {mem_after_lora:.2f} MB")
        if not args.lora_only:
            print(f"  (after MemRift inj):  {mem_after_memrift:.2f} MB")
    print("=" * 60)

    if args.output_peak:
        with open(args.output_peak, "w") as f:
            json.dump({"peak_memory_mb": round(peak_mb, 2), "steps": args.steps}, f, indent=2)
        print(f"  Peak written to {args.output_peak}")

    return peak_mb


if __name__ == "__main__":
    main()
