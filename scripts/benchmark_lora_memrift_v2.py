#!/usr/bin/env python
"""
Benchmark LoRA vs MemRift+LoRA
Direct comparison using HuggingFace models

Models:
- TinyLlama-1.1B
- Llama-3.2-3B-Instruct
- Mistral-7B-Instruct-v0.3
- Llama-3.1-8B-Instruct
"""
import os
import sys
import gc
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add FlagScale to path
sys.path.insert(0, '/share/project/mengyc/flagScale/FlagScale')


@dataclass
class ModelSpec:
    name: str
    hf_path: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    
    @property
    def weight_memory_gb(self):
        """Estimate model weight memory in GB (bf16)."""
        # Embedding + lm_head
        embed = self.vocab_size * self.hidden_size * 2
        
        # Per layer: QKV + O_proj + MLP (gate+up+down) + LayerNorms
        head_dim = self.hidden_size // self.num_heads
        qkv_size = self.hidden_size + 2 * (self.num_kv_heads * head_dim)
        layer_params = (
            qkv_size * self.hidden_size +  # QKV
            self.hidden_size * self.hidden_size +  # O_proj
            self.intermediate_size * self.hidden_size * 3 +  # MLP
            self.hidden_size * 2 * 2  # LayerNorms
        )
        
        total = embed * 2 + layer_params * self.num_layers
        return total * 2 / 1024**3  # bf16


MODELS = {
    "tinyllama": ModelSpec(
        name="TinyLlama-1.1B",
        hf_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_layers=22, hidden_size=2048, intermediate_size=5632,
        num_heads=32, num_kv_heads=4, vocab_size=32000,
    ),
    "llama-3.2-3b": ModelSpec(
        name="Llama-3.2-3B",
        hf_path="meta-llama/Llama-3.2-3B-Instruct",
        num_layers=28, hidden_size=3072, intermediate_size=8192,
        num_heads=24, num_kv_heads=8, vocab_size=128256,
    ),
    "mistral-7b": ModelSpec(
        name="Mistral-7B",
        hf_path="mistralai/Mistral-7B-Instruct-v0.3",
        num_layers=32, hidden_size=4096, intermediate_size=14336,
        num_heads=32, num_kv_heads=8, vocab_size=32768,
    ),
    "llama-3.1-8b": ModelSpec(
        name="Llama-3.1-8B",
        hf_path="meta-llama/Llama-3.1-8B-Instruct",
        num_layers=32, hidden_size=4096, intermediate_size=14336,
        num_heads=32, num_kv_heads=8, vocab_size=128256,
    ),
}


def reset_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# Global override set via --comp-dir CLI arg
_COMP_DIR_OVERRIDE: Optional[str] = None

def get_compressed_weight_dir(model_key: str) -> str:
    """Get compressed weight directory for a model."""
    if _COMP_DIR_OVERRIDE:
        return _COMP_DIR_OVERRIDE
    base_dir = "/share/project/mengyc/code/memrift_v1/weight_comp"
    if model_key == "tinyllama":
        return os.path.join(base_dir, "test")  # Existing TinyLlama weights
    return os.path.join(base_dir, model_key)


def prepare_compressed_weights(model_key: str, model_spec: ModelSpec) -> bool:
    """Prepare compressed weights for a model."""
    comp_dir = get_compressed_weight_dir(model_key)
    index_file = os.path.join(comp_dir, "index.json")
    
    if os.path.exists(index_file):
        print(f"  Compressed weights exist: {comp_dir}")
        return True
    
    print(f"  Preparing compressed weights for {model_spec.name}...")
    os.makedirs(comp_dir, exist_ok=True)
    
    try:
        from flagscale.compress.memrift.offline_comp.prepare_weight import prepare_weight
        prepare_weight(
            model_path=model_spec.hf_path,
            output_dir=comp_dir,
            dtype="bfloat16",
        )
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def run_lora_baseline(model_spec: ModelSpec, batch_size: int, seq_len: int, 
                      num_steps: int, lora_rank: int) -> Dict[str, Any]:
    """Run standard LoRA training (full weights in GPU)."""
    print(f"\n{'='*60}")
    print(f"LORA BASELINE: {model_spec.name}")
    print(f"{'='*60}")
    
    device = torch.device('cuda')
    reset_cuda()
    
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_spec.hf_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    mem_after_load = torch.cuda.memory_allocated(device)
    print(f"Memory after load: {mem_after_load / 1024**3:.2f} GB")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup
    print("Warmup...")
    input_ids = torch.randint(0, model_spec.vocab_size, (batch_size, seq_len), device=device)
    outputs = model(input_ids, labels=input_ids)
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    del outputs, input_ids
    
    reset_cuda()
    
    # Benchmark
    print(f"Running {num_steps} steps...")
    step_times = []
    losses = []
    
    for step in range(num_steps):
        input_ids = torch.randint(0, model_spec.vocab_size, (batch_size, seq_len), device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        step_times.append(elapsed)
        losses.append(loss.item())
        print(f"  Step {step+1}: loss={loss.item():.4f}, time={elapsed*1000:.0f}ms")
        
        del outputs, input_ids
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    avg_time = sum(step_times) / len(step_times)
    
    print(f"\nResults:")
    print(f"  Peak memory: {peak_mem / 1024**3:.2f} GB")
    print(f"  Avg step time: {avg_time*1000:.0f} ms")
    
    del model, optimizer
    reset_cuda()
    
    return {
        "peak_memory_gb": peak_mem / 1024**3,
        "avg_step_time_ms": avg_time * 1000,
        "step_times_ms": [t * 1000 for t in step_times],
        "losses": losses,
    }


def run_memrift(model_key: str, model_spec: ModelSpec, batch_size: int, seq_len: int,
                num_steps: int, lora_rank: int) -> Dict[str, Any]:
    """Run MemRift+LoRA training (compressed weights, on-demand loading)."""
    print(f"\n{'='*60}")
    print(f"MEMRIFT + LORA: {model_spec.name}")
    print(f"{'='*60}")
    
    device = torch.device('cuda')
    reset_cuda()
    
    comp_dir = get_compressed_weight_dir(model_key)
    if not os.path.exists(os.path.join(comp_dir, "index.json")):
        return {"error": f"Compressed weights not found at {comp_dir}"}
    
    from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader
    
    # Build model structure matching MemRift loader expectations
    head_dim = model_spec.hidden_size // model_spec.num_heads
    qkv_size = model_spec.hidden_size + 2 * (model_spec.num_kv_heads * head_dim)
    
    class MemRiftLayer(nn.Module):
        def __init__(self):
            super().__init__()
            h = model_spec.hidden_size
            ffn = model_spec.intermediate_size
            
            self.input_layernorm = nn.LayerNorm(h, dtype=torch.bfloat16, device=device)
            self.post_attention_layernorm = nn.LayerNorm(h, dtype=torch.bfloat16, device=device)
            
            # Empty base weights (managed by MemRift)
            self.self_attention = nn.Module()
            self.self_attention.linear_qkv = nn.Module()
            self.self_attention.linear_qkv.weight = nn.Parameter(
                torch.empty(0, dtype=torch.bfloat16, device=device), requires_grad=False)
            self.self_attention.linear_proj = nn.Module()
            self.self_attention.linear_proj.weight = nn.Parameter(
                torch.empty(0, dtype=torch.bfloat16, device=device), requires_grad=False)
            
            self.mlp = nn.Module()
            self.mlp.linear_fc1 = nn.Module()
            self.mlp.linear_fc1.weight = nn.Parameter(
                torch.empty(0, dtype=torch.bfloat16, device=device), requires_grad=False)
            self.mlp.linear_fc2 = nn.Module()
            self.mlp.linear_fc2.weight = nn.Parameter(
                torch.empty(0, dtype=torch.bfloat16, device=device), requires_grad=False)
            
            # LoRA adapters
            scale = 32.0 / lora_rank
            self.lora_qkv_A = nn.Parameter(torch.randn(lora_rank, h, dtype=torch.bfloat16, device=device) * 0.01)
            self.lora_qkv_B = nn.Parameter(torch.zeros(qkv_size, lora_rank, dtype=torch.bfloat16, device=device))
            self.lora_proj_A = nn.Parameter(torch.randn(lora_rank, h, dtype=torch.bfloat16, device=device) * 0.01)
            self.lora_proj_B = nn.Parameter(torch.zeros(h, lora_rank, dtype=torch.bfloat16, device=device))
            self.lora_fc1_A = nn.Parameter(torch.randn(lora_rank, h, dtype=torch.bfloat16, device=device) * 0.01)
            self.lora_fc1_B = nn.Parameter(torch.zeros(ffn * 2, lora_rank, dtype=torch.bfloat16, device=device))
            self.lora_fc2_A = nn.Parameter(torch.randn(lora_rank, ffn, dtype=torch.bfloat16, device=device) * 0.01)
            self.lora_fc2_B = nn.Parameter(torch.zeros(h, lora_rank, dtype=torch.bfloat16, device=device))
            self.lora_scale = scale
        
        def forward(self, x):
            h = self.input_layernorm(x)
            
            # Attention with LoRA
            qkv_w = self.self_attention.linear_qkv.weight
            if qkv_w.numel() > 0:
                qkv = F.linear(h, qkv_w) + F.linear(F.linear(h, self.lora_qkv_A), self.lora_qkv_B) * self.lora_scale
            else:
                qkv = F.linear(F.linear(h, self.lora_qkv_A), self.lora_qkv_B) * self.lora_scale
            
            proj_w = self.self_attention.linear_proj.weight
            if proj_w.numel() > 0:
                proj = F.linear(h, proj_w) + F.linear(F.linear(h, self.lora_proj_A), self.lora_proj_B) * self.lora_scale
            else:
                proj = F.linear(F.linear(h, self.lora_proj_A), self.lora_proj_B) * self.lora_scale
            
            x = x + proj
            h = self.post_attention_layernorm(x)
            
            # MLP with LoRA
            fc1_w = self.mlp.linear_fc1.weight
            if fc1_w.numel() > 0:
                fc1 = F.linear(h, fc1_w) + F.linear(F.linear(h, self.lora_fc1_A), self.lora_fc1_B) * self.lora_scale
            else:
                fc1 = F.linear(F.linear(h, self.lora_fc1_A), self.lora_fc1_B) * self.lora_scale
            
            gate, up = fc1.chunk(2, dim=-1)
            h = F.silu(gate) * up
            
            fc2_w = self.mlp.linear_fc2.weight
            if fc2_w.numel() > 0:
                fc2 = F.linear(h, fc2_w) + F.linear(F.linear(h, self.lora_fc2_A), self.lora_fc2_B) * self.lora_scale
            else:
                fc2 = F.linear(F.linear(h, self.lora_fc2_A), self.lora_fc2_B) * self.lora_scale
            
            return x + fc2
    
    class MemRiftModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(model_spec.vocab_size, model_spec.hidden_size,
                                         dtype=torch.bfloat16, device=device)
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([MemRiftLayer() for _ in range(model_spec.num_layers)])
            self.decoder.final_layernorm = nn.LayerNorm(model_spec.hidden_size, 
                                                        dtype=torch.bfloat16, device=device)
            self.lm_head = nn.Linear(model_spec.hidden_size, model_spec.vocab_size,
                                    bias=False, dtype=torch.bfloat16, device=device)
        
        def forward(self, input_ids, labels=None):
            x = self.embedding(input_ids)
            for layer in self.decoder.layers:
                x = layer(x)
            x = self.decoder.final_layernorm(x)
            logits = self.lm_head(x)
            
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return type('Output', (), {'loss': loss, 'logits': logits})()
    
    print("Creating model structure...")
    model = MemRiftModel()
    
    mem_before = torch.cuda.memory_allocated(device)
    print(f"Memory (empty base): {mem_before / 1024**3:.2f} GB")
    
    print("Setting up MemRift...")
    loader = MegatronDynamicLoader(
        model=model,
        comp_dir=comp_dir,
        device=device,
        prefetch_layers=2,
        print_debug=False,
    )
    loader.load_weights()
    loader.build_param_mapping()

    # AsyncCompressor: decompresses layer N+1 in background thread pool
    # while GPU computes layer N.  Without this, every layer decompress
    # blocks the main thread (~4-40ms/layer → 10x slowdown).
    from flagscale.compress.memrift.async_compressor import AsyncCompressor
    async_comp = AsyncCompressor(
        compress_workers=4,
        decode_workers=8,
        concurrency_limit=8,   # allow all 7 components per layer to decode concurrently
        zstd_level=1,
        enable_async=True,
    )
    loader.install_hooks(async_compressor=async_comp)
    loader.prefetch_initial_layers()
    
    mem_after = torch.cuda.memory_allocated(device)
    stats = loader.get_memory_stats()
    print(f"Memory after MemRift: {mem_after / 1024**3:.2f} GB")
    print(f"  sm_gpu resident: {stats['sm_gpu_mb'] / 1024:.2f} GB")
    
    lora_params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in lora_params)
    print(f"Trainable params: {trainable / 1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
    
    # Warmup
    print("Warmup...")
    input_ids = torch.randint(0, model_spec.vocab_size, (batch_size, seq_len), device=device)
    outputs = model(input_ids, labels=input_ids)
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    del outputs, input_ids
    
    reset_cuda()
    
    # Benchmark
    print(f"Running {num_steps} steps...")
    step_times = []
    losses = []
    
    for step in range(num_steps):
        input_ids = torch.randint(0, model_spec.vocab_size, (batch_size, seq_len), device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        step_times.append(elapsed)
        losses.append(loss.item())
        print(f"  Step {step+1}: loss={loss.item():.4f}, time={elapsed*1000:.0f}ms")
        
        del outputs, input_ids
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    avg_time = sum(step_times) / len(step_times)
    
    print(f"\nResults:")
    print(f"  Peak memory: {peak_mem / 1024**3:.2f} GB")
    print(f"  Avg step time: {avg_time*1000:.0f} ms")
    
    async_comp.shutdown()
    del model, loader, optimizer
    reset_cuda()
    
    return {
        "peak_memory_gb": peak_mem / 1024**3,
        "avg_step_time_ms": avg_time * 1000,
        "step_times_ms": [t * 1000 for t in step_times],
        "losses": losses,
        "sm_gpu_gb": stats['sm_gpu_mb'] / 1024,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["tinyllama"],
                       choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--prepare-weights", action="store_true")
    parser.add_argument("--lora-only", action="store_true")
    parser.add_argument("--memrift-only", action="store_true")
    parser.add_argument("--comp-dir", type=str, default=None,
                        help="Override compressed weight directory (for nvcomp_lz4 vs zstd comparison).")
    parser.add_argument("--compression", choices=["zstd", "nvcomp_lz4", "compare"], default="zstd",
                        help="Compression format for weights. 'compare' runs both zstd and nvcomp_lz4.")
    args = parser.parse_args()
    
    global _COMP_DIR_OVERRIDE
    if args.comp_dir:
        _COMP_DIR_OVERRIDE = args.comp_dir
        print(f"  [override] comp_dir = {args.comp_dir}")

    model_keys = list(MODELS.keys()) if "all" in args.models else args.models
    
    print("=" * 70)
    print("BENCHMARK: LoRA vs MemRift+LoRA")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Models: {[MODELS[k].name for k in model_keys]}")
    print(f"Config: bs={args.batch_size}, seq={args.seq_len}, steps={args.num_steps}, rank={args.lora_rank}")
    
    results = {}
    
    for model_key in model_keys:
        spec = MODELS[model_key]
        print(f"\n{'#'*70}")
        print(f"# {spec.name} (estimated {spec.weight_memory_gb:.1f} GB)")
        print(f"{'#'*70}")
        
        results[model_key] = {"model": spec.name, "estimated_weight_gb": spec.weight_memory_gb}
        
        # Prepare weights if needed
        if args.prepare_weights and not args.lora_only:
            prepare_compressed_weights(model_key, spec)
        
        # LoRA baseline
        if not args.memrift_only:
            try:
                results[model_key]["lora"] = run_lora_baseline(
                    spec, args.batch_size, args.seq_len, args.num_steps, args.lora_rank)
            except Exception as e:
                print(f"LoRA failed: {e}")
                import traceback
                traceback.print_exc()
                results[model_key]["lora"] = {"error": str(e)}
        
        # MemRift
        if not args.lora_only:
            if args.compression == "compare":
                # Run both compression formats and report separately
                for comp_label, comp_dir_suffix in [("memrift_zstd", "test"),
                                                     ("memrift_nvcomp_lz4", f"test_nvcomp_lz4")]:
                    _COMP_DIR_OVERRIDE = os.path.join(
                        "/share/project/mengyc/code/memrift_v1/weight_comp", comp_dir_suffix)
                    if not os.path.exists(os.path.join(_COMP_DIR_OVERRIDE, "index.json")):
                        print(f"  Skipping {comp_label}: no weights at {_COMP_DIR_OVERRIDE}")
                        results[model_key][comp_label] = {"error": "weights not found"}
                        continue
                    try:
                        results[model_key][comp_label] = run_memrift(
                            model_key, spec, args.batch_size, args.seq_len, args.num_steps, args.lora_rank)
                    except Exception as e:
                        print(f"MemRift ({comp_label}) failed: {e}")
                        results[model_key][comp_label] = {"error": str(e)}
                _COMP_DIR_OVERRIDE = None
            else:
                try:
                    results[model_key]["memrift"] = run_memrift(
                        model_key, spec, args.batch_size, args.seq_len, args.num_steps, args.lora_rank)
                except Exception as e:
                    print(f"MemRift failed: {e}")
                    import traceback
                    traceback.print_exc()
                    results[model_key]["memrift"] = {"error": str(e)}
    
    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Model':<20} {'Est.Weight':>10} {'LoRA Peak':>12} {'MemRift Peak':>12} {'Saved':>10} {'LoRA Time':>12} {'MemRift Time':>12}")
    print("-" * 90)
    
    for mk in model_keys:
        r = results.get(mk, {})
        est = r.get("estimated_weight_gb", 0)
        lora_peak = r.get("lora", {}).get("peak_memory_gb", float('nan'))
        mr_peak = r.get("memrift", {}).get("peak_memory_gb", float('nan'))
        lora_time = r.get("lora", {}).get("avg_step_time_ms", float('nan'))
        mr_time = r.get("memrift", {}).get("avg_step_time_ms", float('nan'))
        
        if lora_peak and mr_peak and not (lora_peak != lora_peak):  # not nan
            saved = lora_peak - mr_peak
            saved_str = f"{saved:+.2f} GB"
        else:
            saved_str = "N/A"
        
        print(f"{MODELS[mk].name:<20} {est:>8.1f} GB {lora_peak:>10.2f} GB {mr_peak:>10.2f} GB {saved_str:>10} {lora_time:>10.0f} ms {mr_time:>10.0f} ms")
    
    print("-" * 90)
    
    # Save
    out_file = f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
