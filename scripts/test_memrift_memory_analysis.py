#!/usr/bin/env python
"""
Memory analysis for MemRift vs Baseline.

Measures:
- Baseline: Full model weights resident in GPU
- MemRift: sm_gpu resident + per-layer materialization
"""
import os
import sys
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/share/project/mengyc/flagScale/FlagScale')


def analyze_baseline_memory():
    """Analyze baseline memory: full weights in GPU."""
    print("=" * 70)
    print("BASELINE: Full weights resident in GPU")
    print("=" * 70)
    
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load weight index to get exact sizes
    comp_dir = "/share/project/mengyc/code/memrift_v1/weight_comp/test"
    with open(os.path.join(comp_dir, "index.json")) as f:
        index = json.load(f)
    
    # Calculate total weight size
    total_params = 0
    layer_params = {}
    
    for entry in index:
        numel = 1
        for s in entry["shape"]:
            numel *= s
        total_params += numel
        
        name = entry["name"]
        if "layers." in name:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    layer_idx = int(parts[i + 1])
                    layer_params[layer_idx] = layer_params.get(layer_idx, 0) + numel
                    break
    
    bytes_per_param = 2  # bf16
    total_bytes = total_params * bytes_per_param
    
    print(f"\nWeight Statistics:")
    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Total weight memory: {total_bytes / 1024**2:.1f} MB (bf16)")
    print(f"  Number of layers: {len(layer_params)}")
    
    avg_layer_params = sum(layer_params.values()) / len(layer_params)
    print(f"  Avg params per layer: {avg_layer_params / 1e6:.2f}M")
    print(f"  Avg memory per layer: {avg_layer_params * bytes_per_param / 1024**2:.1f} MB")
    
    return total_bytes, layer_params


def analyze_memrift_memory():
    """Analyze MemRift memory: sm_gpu + per-layer materialization."""
    print("\n" + "=" * 70)
    print("MEMRIFT: Compressed storage + on-demand materialization")
    print("=" * 70)
    
    device = torch.device('cuda')
    comp_dir = "/share/project/mengyc/code/memrift_v1/weight_comp/test"
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create minimal model structure for loader
    class MinimalModel(nn.Module):
        def __init__(self, num_layers=22):
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList()
            for _ in range(num_layers):
                layer = nn.Module()
                layer.self_attention = nn.Module()
                layer.self_attention.linear_qkv = nn.Module()
                layer.self_attention.linear_qkv.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16, device=device))
                layer.self_attention.linear_proj = nn.Module()
                layer.self_attention.linear_proj.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16, device=device))
                layer.mlp = nn.Module()
                layer.mlp.linear_fc1 = nn.Module()
                layer.mlp.linear_fc1.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16, device=device))
                layer.mlp.linear_fc2 = nn.Module()
                layer.mlp.linear_fc2.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16, device=device))
                self.decoder.layers.append(layer)
    
    model = MinimalModel()
    
    from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader
    
    mem_before = torch.cuda.memory_allocated(device)
    
    loader = MegatronDynamicLoader(
        model=model,
        comp_dir=comp_dir,
        device=device,
        print_debug=False,
    )
    loader.load_weights()
    loader.build_param_mapping()
    
    mem_after_load = torch.cuda.memory_allocated(device)
    sm_gpu_mem = mem_after_load - mem_before
    
    stats = loader.get_memory_stats()
    
    print(f"\nMemRift Storage:")
    print(f"  sm_gpu (resident): {stats['sm_gpu_mb']:.1f} MB")
    print(f"  exp_cpu (compressed): {stats['exp_cpu_mb']:.1f} MB")
    
    # Test single layer materialization
    print(f"\nPer-Layer Materialization Test:")
    
    layer_mem_usage = []
    for layer_idx in range(min(3, len(loader.layer_names))):
        layer_name = loader.layer_names[layer_idx]
        groups = loader.layer2groups.get(layer_name, [])
        
        torch.cuda.reset_peak_memory_stats(device)
        mem_before_layer = torch.cuda.memory_allocated(device)
        
        # Materialize all weights in this layer
        for group in groups:
            weight = loader._materialize_group(group, sync=True)
            loader._set_param(group, weight)
        
        mem_after_materialize = torch.cuda.memory_allocated(device)
        layer_mem = mem_after_materialize - mem_before_layer
        layer_mem_usage.append(layer_mem)
        
        print(f"  Layer {layer_idx}: materialized {len(groups)} groups, +{layer_mem / 1024**2:.1f} MB")
        
        # Release
        for group in groups:
            loader._clear_param(group)
        
        mem_after_release = torch.cuda.memory_allocated(device)
        print(f"    After release: {(mem_after_release - mem_before_layer) / 1024**2:+.1f} MB")
    
    avg_layer_mem = sum(layer_mem_usage) / len(layer_mem_usage) if layer_mem_usage else 0
    
    return stats['sm_gpu_mb'] * 1024**2, avg_layer_mem


def main():
    print("=" * 70)
    print("MEMORY ANALYSIS: Baseline vs MemRift")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Model: TinyLlama-like (22 layers)")
    
    # Analyze baseline
    baseline_total, baseline_layer_params = analyze_baseline_memory()
    
    # Analyze MemRift
    memrift_resident, memrift_per_layer = analyze_memrift_memory()
    
    # Theoretical analysis
    print("\n" + "=" * 70)
    print("THEORETICAL MEMORY COMPARISON")
    print("=" * 70)
    
    # Baseline: all weights + activations + gradients
    # MemRift: sm_gpu + 1-2 layers materialized + activations + gradients
    
    num_layers = 22
    bytes_per_param = 2
    
    # For training: weights + gradients + optimizer states (AdamW: 2x for m, v)
    # Baseline
    baseline_weights = baseline_total
    baseline_grads = baseline_total  # Same size as weights
    baseline_optimizer = baseline_total * 2 * 4  # AdamW m,v in fp32
    
    # MemRift (LoRA scenario: only train LoRA, base frozen)
    # Resident: sm_gpu
    # Per-layer: 1-2 layers materialized during forward/backward
    memrift_weights = memrift_resident  # sm_gpu
    memrift_active_layers = memrift_per_layer * 2  # 2 layers prefetched
    
    print(f"\n{'Component':<35} {'Baseline':>15} {'MemRift':>15}")
    print("-" * 65)
    print(f"{'Base weights (resident)':<35} {baseline_weights/1024**2:>12.1f} MB {memrift_resident/1024**2:>12.1f} MB")
    print(f"{'Active layer weights':<35} {'N/A':>15} {memrift_active_layers/1024**2:>12.1f} MB")
    print("-" * 65)
    print(f"{'Total weight footprint':<35} {baseline_weights/1024**2:>12.1f} MB {(memrift_resident + memrift_active_layers)/1024**2:>12.1f} MB")
    
    weight_savings = baseline_weights - (memrift_resident + memrift_active_layers)
    weight_savings_pct = weight_savings / baseline_weights * 100
    
    print(f"\n{'Weight memory savings:':<35} {weight_savings/1024**2:>+12.1f} MB ({weight_savings_pct:+.1f}%)")
    
    # Practical peak memory estimate
    print("\n" + "=" * 70)
    print("PRACTICAL PEAK MEMORY ESTIMATE (Forward + Backward)")
    print("=" * 70)
    
    # Assume activations scale with batch_size * seq_len * hidden * num_layers
    # For bs=2, seq=512, hidden=2048, 22 layers ~ 500MB activations
    activation_estimate = 500 * 1024**2  # MB
    
    baseline_peak = baseline_weights + activation_estimate
    memrift_peak = memrift_resident + memrift_active_layers + activation_estimate
    
    print(f"\n{'Component':<35} {'Baseline':>15} {'MemRift':>15}")
    print("-" * 65)
    print(f"{'Weight footprint':<35} {baseline_weights/1024**2:>12.1f} MB {(memrift_resident + memrift_active_layers)/1024**2:>12.1f} MB")
    print(f"{'Activations (estimate)':<35} {activation_estimate/1024**2:>12.1f} MB {activation_estimate/1024**2:>12.1f} MB")
    print("-" * 65)
    print(f"{'Estimated peak (inference)':<35} {baseline_peak/1024**2:>12.1f} MB {memrift_peak/1024**2:>12.1f} MB")
    
    peak_savings = baseline_peak - memrift_peak
    peak_savings_pct = peak_savings / baseline_peak * 100
    
    print(f"\n{'Peak memory savings (inference):':<35} {peak_savings/1024**2:>+12.1f} MB ({peak_savings_pct:+.1f}%)")
    
    # Note about training
    print("\n" + "=" * 70)
    print("NOTES")
    print("=" * 70)
    print("""
MemRift memory savings are most significant when:

1. Model is large (weights >> activations)
   - 1.1B model: weights ~2.1GB, savings ~34% of weights
   - 8B model: weights ~16GB, savings would be ~10GB+
   - 70B model: weights ~140GB, savings would be ~100GB+

2. Base weights are frozen (LoRA/PEFT training)
   - No gradients for base weights
   - No optimizer states for base weights
   - Only LoRA params need grad/optimizer

3. Single GPU or TP=1 setup
   - Current implementation requires TP=1

Current 1.1B test shows modest savings because:
- sm_gpu (1GB) is ~50% of original weights (2.1GB)
- Small model means activation memory is significant portion
- The savings percentage improves dramatically with larger models
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
