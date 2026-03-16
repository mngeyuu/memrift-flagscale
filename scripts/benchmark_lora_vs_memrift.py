#!/usr/bin/env python
"""
Benchmark LoRA vs MemRift+LoRA on multiple models.
Measures: Peak GPU memory, Step time

Models:
- TinyLlama-1.1B
- Llama-3.2-3B-Instruct
- Mistral-7B-Instruct-v0.3
- Llama-3.1-8B-Instruct

Requirements:
- Compressed weights prepared for each model
- FlagScale with MemRift integration
"""
import os
import sys
import gc
import json
import time
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


@dataclass
class ModelConfig:
    name: str
    hf_path: str
    comp_dir: Optional[str]
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int


# Model configurations
MODELS = {
    "tinyllama-1.1b": ModelConfig(
        name="TinyLlama-1.1B",
        hf_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        comp_dir="/share/project/mengyc/code/memrift_v1/weight_comp/tinyllama-1.1b",
        num_layers=22,
        hidden_size=2048,
        intermediate_size=5632,
        num_attention_heads=32,
        num_key_value_heads=4,
    ),
    "llama-3.2-3b": ModelConfig(
        name="Llama-3.2-3B-Instruct",
        hf_path="meta-llama/Llama-3.2-3B-Instruct",
        comp_dir="/share/project/mengyc/code/memrift_v1/weight_comp/llama-3.2-3b",
        num_layers=28,
        hidden_size=3072,
        intermediate_size=8192,
        num_attention_heads=24,
        num_key_value_heads=8,
    ),
    "mistral-7b": ModelConfig(
        name="Mistral-7B-Instruct-v0.3",
        hf_path="mistralai/Mistral-7B-Instruct-v0.3",
        comp_dir="/share/project/mengyc/code/memrift_v1/weight_comp/mistral-7b",
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
    "llama-3.1-8b": ModelConfig(
        name="Llama-3.1-8B-Instruct",
        hf_path="meta-llama/Llama-3.1-8B-Instruct",
        comp_dir="/share/project/mengyc/code/memrift_v1/weight_comp/llama-3.1-8b",
        num_layers=32,
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
}


def reset_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def prepare_compressed_weights(model_key: str, model_config: ModelConfig):
    """Prepare compressed weights for a model using MemRift."""
    comp_dir = model_config.comp_dir
    if os.path.exists(os.path.join(comp_dir, "index.json")):
        print(f"[{model_config.name}] Compressed weights already exist at {comp_dir}")
        return True
    
    print(f"[{model_config.name}] Preparing compressed weights...")
    os.makedirs(comp_dir, exist_ok=True)
    
    # Use the prepare_weight script
    sys.path.insert(0, '/share/project/mengyc/flagScale/FlagScale')
    from flagscale.compress.memrift.offline_comp.prepare_weight import prepare_weight
    
    try:
        prepare_weight(
            model_path=model_config.hf_path,
            output_dir=comp_dir,
            dtype="bfloat16",
        )
        return True
    except Exception as e:
        print(f"[{model_config.name}] Failed to prepare weights: {e}")
        return False


def run_lora_baseline(model_config: ModelConfig, batch_size: int, seq_len: int, num_steps: int, lora_rank: int = 16):
    """Run LoRA training baseline (full weights in GPU)."""
    print(f"\n{'='*70}")
    print(f"BASELINE LoRA: {model_config.name}")
    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, lora_rank={lora_rank}")
    print(f"{'='*70}")
    
    device = torch.device('cuda')
    reset_cuda()
    
    # Load model with LoRA-style setup
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.hf_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    
    # Freeze base weights
    for param in model.parameters():
        param.requires_grad = False
    
    # Add simple LoRA adapters to attention layers
    lora_params = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and 'q_proj' in name:
            in_f = module.weight.shape[1]
            out_f = module.weight.shape[0]
            lora_A = nn.Parameter(torch.randn(lora_rank, in_f, dtype=torch.bfloat16, device=device) * 0.01)
            lora_B = nn.Parameter(torch.zeros(out_f, lora_rank, dtype=torch.bfloat16, device=device))
            module.lora_A = lora_A
            module.lora_B = lora_B
            lora_params.extend([lora_A, lora_B])
    
    mem_model = torch.cuda.memory_allocated(device)
    print(f"Model memory: {mem_model / 1024**3:.2f} GB")
    print(f"LoRA params: {sum(p.numel() for p in lora_params) / 1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
    
    # Warmup
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    reset_cuda()
    
    # Benchmark
    step_times = []
    for step in range(num_steps):
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        step_times.append(elapsed)
        
        print(f"  Step {step+1}: loss={loss.item():.4f}, time={elapsed*1000:.0f}ms")
    
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
    }


def run_memrift_lora(model_config: ModelConfig, batch_size: int, seq_len: int, num_steps: int, lora_rank: int = 16):
    """Run MemRift+LoRA training."""
    print(f"\n{'='*70}")
    print(f"MEMRIFT + LoRA: {model_config.name}")
    print(f"Config: batch_size={batch_size}, seq_len={seq_len}, lora_rank={lora_rank}")
    print(f"{'='*70}")
    
    device = torch.device('cuda')
    reset_cuda()
    
    # Check compressed weights
    if not os.path.exists(os.path.join(model_config.comp_dir, "index.json")):
        print(f"Compressed weights not found at {model_config.comp_dir}")
        print("Please run with --prepare-weights first")
        return None
    
    sys.path.insert(0, '/share/project/mengyc/flagScale/FlagScale')
    from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader
    
    # Create minimal model structure for MemRift
    print("Creating model structure...")
    
    config = AutoConfig.from_pretrained(model_config.hf_path, trust_remote_code=True)
    
    class MemRiftLayer(nn.Module):
        def __init__(self, hidden_size, intermediate_size, num_heads, num_kv_heads, device, lora_rank):
            super().__init__()
            head_dim = hidden_size // num_heads
            qkv_size = hidden_size + 2 * (num_kv_heads * head_dim)  # Q + K + V
            
            self.input_layernorm = nn.LayerNorm(hidden_size, dtype=torch.bfloat16, device=device)
            self.post_attention_layernorm = nn.LayerNorm(hidden_size, dtype=torch.bfloat16, device=device)
            
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
            
            # LoRA adapters (trainable)
            self.lora_qkv_A = nn.Parameter(torch.randn(lora_rank, hidden_size, dtype=torch.bfloat16, device=device) * 0.01)
            self.lora_qkv_B = nn.Parameter(torch.zeros(qkv_size, lora_rank, dtype=torch.bfloat16, device=device))
            self.lora_proj_A = nn.Parameter(torch.randn(lora_rank, hidden_size, dtype=torch.bfloat16, device=device) * 0.01)
            self.lora_proj_B = nn.Parameter(torch.zeros(hidden_size, lora_rank, dtype=torch.bfloat16, device=device))
            self.lora_fc1_A = nn.Parameter(torch.randn(lora_rank, hidden_size, dtype=torch.bfloat16, device=device) * 0.01)
            self.lora_fc1_B = nn.Parameter(torch.zeros(intermediate_size * 2, lora_rank, dtype=torch.bfloat16, device=device))
            self.lora_fc2_A = nn.Parameter(torch.randn(lora_rank, intermediate_size, dtype=torch.bfloat16, device=device) * 0.01)
            self.lora_fc2_B = nn.Parameter(torch.zeros(hidden_size, lora_rank, dtype=torch.bfloat16, device=device))
            
            self.hidden_size = hidden_size
            self.lora_scale = 32.0 / lora_rank
        
        def forward(self, x):
            h = self.input_layernorm(x)
            
            qkv_w = self.self_attention.linear_qkv.weight
            proj_w = self.self_attention.linear_proj.weight
            
            if qkv_w.numel() > 0:
                qkv = F.linear(h, qkv_w) + F.linear(F.linear(h, self.lora_qkv_A), self.lora_qkv_B) * self.lora_scale
            else:
                qkv = F.linear(F.linear(h, self.lora_qkv_A), self.lora_qkv_B) * self.lora_scale
            
            if proj_w.numel() > 0:
                proj = F.linear(h, proj_w) + F.linear(F.linear(h, self.lora_proj_A), self.lora_proj_B) * self.lora_scale
            else:
                proj = F.linear(F.linear(h, self.lora_proj_A), self.lora_proj_B) * self.lora_scale
            
            x = x + proj
            h = self.post_attention_layernorm(x)
            
            fc1_w = self.mlp.linear_fc1.weight
            fc2_w = self.mlp.linear_fc2.weight
            
            if fc1_w.numel() > 0:
                fc1 = F.linear(h, fc1_w) + F.linear(F.linear(h, self.lora_fc1_A), self.lora_fc1_B) * self.lora_scale
            else:
                fc1 = F.linear(F.linear(h, self.lora_fc1_A), self.lora_fc1_B) * self.lora_scale
            
            gate, up = fc1.chunk(2, dim=-1)
            h = F.silu(gate) * up
            
            if fc2_w.numel() > 0:
                fc2 = F.linear(h, fc2_w) + F.linear(F.linear(h, self.lora_fc2_A), self.lora_fc2_B) * self.lora_scale
            else:
                fc2 = F.linear(F.linear(h, self.lora_fc2_A), self.lora_fc2_B) * self.lora_scale
            
            return x + fc2
    
    class MemRiftModel(nn.Module):
        def __init__(self, config, model_config, device, lora_rank):
            super().__init__()
            self.embedding = nn.Embedding(config.vocab_size, model_config.hidden_size, 
                                          dtype=torch.bfloat16, device=device)
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([
                MemRiftLayer(model_config.hidden_size, model_config.intermediate_size,
                            model_config.num_attention_heads, model_config.num_key_value_heads,
                            device, lora_rank)
                for _ in range(model_config.num_layers)
            ])
            self.decoder.final_layernorm = nn.LayerNorm(model_config.hidden_size, 
                                                        dtype=torch.bfloat16, device=device)
            self.lm_head = nn.Linear(model_config.hidden_size, config.vocab_size, 
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
    
    model = MemRiftModel(config, model_config, device, lora_rank)
    
    mem_before = torch.cuda.memory_allocated(device)
    print(f"Model memory (empty base): {mem_before / 1024**3:.2f} GB")
    
    # Setup MemRift
    print("Setting up MemRift...")
    loader = MegatronDynamicLoader(
        model=model,
        comp_dir=model_config.comp_dir,
        device=device,
        prefetch_layers=1,
        print_debug=False,
    )
    loader.load_weights()
    loader.build_param_mapping()
    loader.install_hooks()
    loader.prefetch_initial_layers()
    
    mem_after = torch.cuda.memory_allocated(device)
    stats = loader.get_memory_stats()
    print(f"After MemRift setup: {mem_after / 1024**3:.2f} GB")
    print(f"  sm_gpu resident: {stats['sm_gpu_mb'] / 1024:.2f} GB")
    
    # Count trainable params
    lora_params = [p for p in model.parameters() if p.requires_grad]
    print(f"LoRA params: {sum(p.numel() for p in lora_params) / 1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
    
    # Warmup
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    reset_cuda()
    
    # Benchmark
    step_times = []
    for step in range(num_steps):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        step_times.append(elapsed)
        
        print(f"  Step {step+1}: loss={loss.item():.4f}, time={elapsed*1000:.0f}ms")
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    avg_time = sum(step_times) / len(step_times)
    
    print(f"\nResults:")
    print(f"  Peak memory: {peak_mem / 1024**3:.2f} GB")
    print(f"  Avg step time: {avg_time*1000:.0f} ms")
    
    del model, loader, optimizer
    reset_cuda()
    
    return {
        "peak_memory_gb": peak_mem / 1024**3,
        "avg_step_time_ms": avg_time * 1000,
        "step_times_ms": [t * 1000 for t in step_times],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark LoRA vs MemRift+LoRA")
    parser.add_argument("--models", nargs="+", default=["tinyllama-1.1b"],
                        choices=list(MODELS.keys()) + ["all"],
                        help="Models to benchmark")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--prepare-weights", action="store_true",
                        help="Prepare compressed weights before benchmarking")
    parser.add_argument("--lora-only", action="store_true",
                        help="Only run LoRA baseline")
    parser.add_argument("--memrift-only", action="store_true",
                        help="Only run MemRift+LoRA")
    args = parser.parse_args()
    
    if "all" in args.models:
        model_keys = list(MODELS.keys())
    else:
        model_keys = args.models
    
    print("=" * 70)
    print("BENCHMARK: LoRA vs MemRift+LoRA")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Models: {model_keys}")
    print(f"Config: batch_size={args.batch_size}, seq_len={args.seq_len}, "
          f"num_steps={args.num_steps}, lora_rank={args.lora_rank}")
    
    results = {}
    
    for model_key in model_keys:
        model_config = MODELS[model_key]
        print(f"\n{'#' * 70}")
        print(f"Model: {model_config.name}")
        print(f"{'#' * 70}")
        
        results[model_key] = {"model_name": model_config.name}
        
        # Prepare weights if needed
        if args.prepare_weights:
            prepare_compressed_weights(model_key, model_config)
        
        # Run LoRA baseline
        if not args.memrift_only:
            try:
                lora_results = run_lora_baseline(
                    model_config, args.batch_size, args.seq_len, 
                    args.num_steps, args.lora_rank
                )
                results[model_key]["lora"] = lora_results
            except Exception as e:
                print(f"LoRA baseline failed: {e}")
                results[model_key]["lora"] = {"error": str(e)}
        
        # Run MemRift+LoRA
        if not args.lora_only:
            try:
                memrift_results = run_memrift_lora(
                    model_config, args.batch_size, args.seq_len,
                    args.num_steps, args.lora_rank
                )
                results[model_key]["memrift"] = memrift_results
            except Exception as e:
                print(f"MemRift+LoRA failed: {e}")
                results[model_key]["memrift"] = {"error": str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'LoRA Peak':>12} {'MemRift Peak':>12} {'Memory Saved':>12} {'LoRA Time':>12} {'MemRift Time':>12}")
    print("-" * 85)
    
    for model_key in model_keys:
        r = results.get(model_key, {})
        model_name = MODELS[model_key].name
        
        lora_peak = r.get("lora", {}).get("peak_memory_gb", float('nan'))
        memrift_peak = r.get("memrift", {}).get("peak_memory_gb", float('nan'))
        lora_time = r.get("lora", {}).get("avg_step_time_ms", float('nan'))
        memrift_time = r.get("memrift", {}).get("avg_step_time_ms", float('nan'))
        
        if lora_peak and memrift_peak:
            saved = lora_peak - memrift_peak
            saved_str = f"{saved:+.2f} GB"
        else:
            saved_str = "N/A"
        
        print(f"{model_name:<25} {lora_peak:>10.2f} GB {memrift_peak:>10.2f} GB {saved_str:>12} "
              f"{lora_time:>10.0f} ms {memrift_time:>10.0f} ms")
    
    print("-" * 85)
    
    # Save results
    output_file = f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
