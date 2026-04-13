#!/usr/bin/env python3
"""
MemRift vs Pure LoRA 对比实验
使用 TinyLlama 1.1B，batch size=1，seq len=2048，5 steps，仅 MLP LoRA
"""
import os
import sys
import json
import subprocess
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_command(cmd, description):
    """运行命令并返回输出"""
    print(f"\n{'='*70}")
    print(f"运行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, shell=True, cwd=REPO_ROOT, capture_output=False)
    return result.returncode

def main():
    # 配置参数
    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    compressed_weights = "./memrift_weights/tinyllama_1b_level18"
    steps = 5
    max_length = 2048
    batch_size = 1
    lora_r = 16  # 默认值
    lora_alpha = 32

    output_lora = "output/peak_lora.json"
    output_memrift = "output/peak_memrift.json"

    # 确保输出目录存在
    os.makedirs(os.path.join(REPO_ROOT, "output"), exist_ok=True)

    # 创建一个临时脚本来修改 target_modules
    script_content = '''#!/usr/bin/env python3
"""临时实验脚本 - 仅 MLP LoRA"""
import argparse
import gc
import json
import os
import struct
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from flagscale.compress.memrift.megatron_dynamic_loader import CompressedParam
from flagscale.compress.memrift.async_compressor import AsyncCompressor
from flagscale.compress.memrift.activation_compression import DecoderLayerWrapper

MB = 1024.0 * 1024.0

def get_layer_name_from_param(param_name: str):
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
    if mod is None:
        return None
    for unwrap_attr in ("base_layer", "to_wrap", "module"):
        inner = getattr(mod, unwrap_attr, None)
        if inner is not None and hasattr(inner, attr):
            return inner
    if hasattr(mod, attr):
        return mod
    return None

def load_compressed_weights_into_model(model, comp_dir, device):
    index_path = os.path.join(comp_dir, "index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Missing {index_path}. Run prepare_weight first.")
    with open(index_path) as f:
        index = json.load(f)

    layer2cps = {}
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
            mod._parameters[attr] = cp
        else:
            setattr(mod, attr, cp)

        layer_name = get_layer_name_from_module_name(mod_name)
        if layer_name is not None:
            layer2cps.setdefault(layer_name, []).append(cp)
            cp.layer_idx = int(layer_name.split(".")[-1])

    return layer2cps

def materialize_non_layer_compressed_params(model, comp_dir, device):
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

def main():
    parser = argparse.ArgumentParser(description="Standalone MemRift train on Llama 1.1B + record peak memory")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF model id or path")
    parser.add_argument("--compressed_weights", default="", help="Path to prepare_weight output (index.json + .bin). Omit when using --lora_only")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--activation", action="store_true", help="Enable activation compression")
    parser.add_argument("--profile_memory", action="store_true", help="Print memory breakdown and peak")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--output_peak", type=str, default="", help="Write peak_mb to this JSON file")
    parser.add_argument("--lora_only", action="store_true", help="Run pure LoRA training (no MemRift); do not pass --compressed_weights")
    args = parser.parse_args()

    if not args.lora_only and not args.compressed_weights:
        parser.error("--compressed_weights is required unless --lora_only is set.")

    if args.lora_only:
        print("Mode: 纯 LoRA 训练 (仅 MLP LoRA)")
    else:
        print("Mode: MemRift + LoRA 训练 (仅 MLP LoRA)")

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

    print("Applying LoRA (仅 MLP)...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["gate_proj", "up_proj", "down_proj"],  # 仅使用 MLP LoRA
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.train()
    mem_after_lora = torch.cuda.memory_allocated(device) / MB
    print(f"  After LoRA: {mem_after_lora:.1f} MB allocated")
    model.print_trainable_parameters()

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

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5,
    )

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Mock data
    def get_batch():
        inp = tokenizer(
            "The quick brown fox jumps over the lazy dog. " * 100,
            return_tensors="pt",
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
        )
        return {k: v.to(device) for k, v in inp.items()}

    print(f"Running {args.steps} training steps (max_length={args.max_length})...")
    step_peaks = []
    for step in range(args.steps):
        optimizer.zero_grad()
        batch = get_batch()
        out = model(**batch, labels=batch["input_ids"])
        loss = out.loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_peak_mb = torch.cuda.max_memory_allocated(device) / MB
        step_peaks.append(step_peak_mb)
        if (step + 1) % 2 == 0 or step == 0:
            print(f"  Step {step + 1}/{args.steps}  loss={loss.item():.4f}  peak_allocated={step_peak_mb:.1f} MB")

    peak_mb = max(step_peaks) if step_peaks else (torch.cuda.max_memory_allocated(device) / MB)
    current_mb = torch.cuda.memory_allocated(device) / MB

    print("\\n" + "=" * 60)
    print("显存峰值 (Peak GPU memory)")
    print("=" * 60)
    print(f"  max_memory_allocated:  {peak_mb:.2f} MB")
    print(f"  current allocated:     {current_mb:.2f} MB")
    print("=" * 60)

    if args.output_peak:
        with open(args.output_peak, "w") as f:
            json.dump({"peak_memory_mb": round(peak_mb, 2), "steps": args.steps}, f, indent=2)
        print(f"  Peak written to {args.output_peak}")

    return peak_mb

if __name__ == "__main__":
    main()
'''

    temp_script = os.path.join(REPO_ROOT, "scripts/temp_experiment.py")
    with open(temp_script, "w") as f:
        f.write(script_content)
    os.chmod(temp_script, 0o755)

    # 1. 运行纯 LoRA
    cmd_lora = f"python scripts/temp_experiment.py --model {model} --lora_only --steps {steps} --max_length {max_length} --batch_size {batch_size} --lora_r {lora_r} --lora_alpha {lora_alpha} --output_peak {output_lora}"
    run_command(cmd_lora, "纯 LoRA 训练")

    # 2. 运行 MemRift + LoRA
    cmd_memrift = f"python scripts/temp_experiment.py --model {model} --compressed_weights {compressed_weights} --steps {steps} --max_length {max_length} --batch_size {batch_size} --lora_r {lora_r} --lora_alpha {lora_alpha} --output_peak {output_memrift}"
    run_command(cmd_memrift, "MemRift + LoRA 训练")

    # 3. 读取结果并展示
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)

    peak_lora = None
    peak_memrift = None

    if os.path.exists(os.path.join(REPO_ROOT, output_lora)):
        with open(os.path.join(REPO_ROOT, output_lora)) as f:
            data = json.load(f)
            peak_lora = data["peak_memory_mb"]

    if os.path.exists(os.path.join(REPO_ROOT, output_memrift)):
        with open(os.path.join(REPO_ROOT, output_memrift)) as f:
            data = json.load(f)
            peak_memrift = data["peak_memory_mb"]

    # 构建表格
    print("\n" + "-"*80)
    print(f"{'配置项':<30} {'纯 LoRA':>15} {'MemRift + LoRA':>15} {'差异':>15}")
    print("-"*80)
    print(f"{'模型':<30} {'TinyLlama 1.1B':>15} {'TinyLlama 1.1B':>15} {'-':>15}")
    print(f"{'Batch Size':<30} {batch_size:>15} {batch_size:>15} {'-':>15}")
    print(f"{'Sequence Length':<30} {max_length:>15} {max_length:>15} {'-':>15}")
    print(f"{'训练轮数':<30} {steps:>15} {steps:>15} {'-':>15}")
    print(f"{'LoRA 类型':<30} {'仅 MLP':>15} {'仅 MLP':>15} {'-':>15}")
    print("-"*80)

    if peak_lora and peak_memrift:
        saved_mb = peak_lora - peak_memrift
        saved_pct = (saved_mb / peak_lora) * 100
        print(f"{'显存峰值 (MB)':<30} {peak_lora:>15.2f} {peak_memrift:>15.2f} {saved_mb:>+15.2f}")
        print(f"{'显存峰值 (GB)':<30} {peak_lora/1024:>15.2f} {peak_memrift/1024:>15.2f} {saved_mb/1024:>+15.2f}")
        print(f"{'显存节省比例':<30} {'-':>15} {'-':>15} {saved_pct:>+14.1f}%")
    else:
        print("无法读取完整结果")
    print("-"*80)

    # 清理临时文件
    os.unlink(temp_script)

    print("\n实验完成！")

if __name__ == "__main__":
    main()
