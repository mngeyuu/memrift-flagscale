#!/usr/bin/env python
"""
Debug hooks behavior to understand why peak memory isn't reduced.
"""
import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/share/project/mengyc/flagScale/FlagScale')


def main():
    print("=" * 70)
    print("DEBUG: MemRift Hooks Behavior")
    print("=" * 70)
    
    device = torch.device('cuda')
    gc.collect()
    torch.cuda.empty_cache()
    
    from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader
    
    comp_dir = "/share/project/mengyc/code/memrift_v1/weight_comp/test"
    
    # Create minimal model
    class DebugModel(nn.Module):
        def __init__(self, num_layers, device):
            super().__init__()
            self.embedding = nn.Embedding(32000, 2048, dtype=torch.bfloat16, device=device)
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList()
            
            for i in range(num_layers):
                layer = nn.Module()
                layer.idx = i
                layer.input_layernorm = nn.LayerNorm(2048, dtype=torch.bfloat16, device=device)
                layer.post_attention_layernorm = nn.LayerNorm(2048, dtype=torch.bfloat16, device=device)
                
                layer.self_attention = nn.Module()
                layer.self_attention.linear_qkv = nn.Module()
                layer.self_attention.linear_qkv.weight = nn.Parameter(
                    torch.empty(0, dtype=torch.bfloat16, device=device), requires_grad=False)
                layer.self_attention.linear_proj = nn.Module()
                layer.self_attention.linear_proj.weight = nn.Parameter(
                    torch.empty(0, dtype=torch.bfloat16, device=device), requires_grad=False)
                
                layer.mlp = nn.Module()
                layer.mlp.linear_fc1 = nn.Module()
                layer.mlp.linear_fc1.weight = nn.Parameter(
                    torch.empty(0, dtype=torch.bfloat16, device=device), requires_grad=False)
                layer.mlp.linear_fc2 = nn.Module()
                layer.mlp.linear_fc2.weight = nn.Parameter(
                    torch.empty(0, dtype=torch.bfloat16, device=device), requires_grad=False)
                
                self.decoder.layers.append(layer)
            
            self.decoder.final_layernorm = nn.LayerNorm(2048, dtype=torch.bfloat16, device=device)
    
    model = DebugModel(22, device)
    
    print(f"Initial memory: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
    
    loader = MegatronDynamicLoader(
        model=model,
        comp_dir=comp_dir,
        device=device,
        prefetch_layers=1,
        print_debug=True,  # Enable debug
    )
    
    loader.load_weights()
    loader.build_param_mapping()
    loader.install_hooks()
    
    print(f"\nAfter setup (before prefetch): {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
    
    loader.prefetch_initial_layers()
    
    print(f"After prefetch_initial_layers: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
    
    # Check which layers have materialized weights
    print("\nWeight status per layer:")
    for i, layer in enumerate(model.decoder.layers):
        qkv_numel = layer.self_attention.linear_qkv.weight.numel()
        proj_numel = layer.self_attention.linear_proj.weight.numel()
        fc1_numel = layer.mlp.linear_fc1.weight.numel()
        fc2_numel = layer.mlp.linear_fc2.weight.numel()
        total_mb = (qkv_numel + proj_numel + fc1_numel + fc2_numel) * 2 / 1024**2
        if total_mb > 0.1:
            print(f"  Layer {i}: {total_mb:.1f} MB materialized")
    
    # Manual forward through one layer to test hook trigger
    print("\n--- Manual forward through layers ---")
    x = model.embedding(torch.randint(0, 32000, (2, 64), device=device))
    
    for i in range(min(5, len(model.decoder.layers))):
        layer = model.decoder.layers[i]
        
        print(f"\nBefore layer {i} forward:")
        print(f"  GPU memory: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
        
        # Check weight status
        qkv_w = layer.self_attention.linear_qkv.weight
        print(f"  linear_qkv.weight shape: {qkv_w.shape}, numel: {qkv_w.numel()}")
        
        # Simple forward
        h = layer.input_layernorm(x)
        if qkv_w.numel() > 0:
            _ = F.linear(h, qkv_w)
        
        proj_w = layer.self_attention.linear_proj.weight
        if proj_w.numel() > 0:
            x = x + F.linear(h, proj_w)
        
        h = layer.post_attention_layernorm(x)
        fc1_w = layer.mlp.linear_fc1.weight
        if fc1_w.numel() > 0:
            fc1_out = F.linear(h, fc1_w)
            gate, up = fc1_out.chunk(2, dim=-1)
            h = F.silu(gate) * up
        
        fc2_w = layer.mlp.linear_fc2.weight
        if fc2_w.numel() > 0:
            x = x + F.linear(h, fc2_w)
        
        print(f"After layer {i} forward:")
        print(f"  GPU memory: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
    
    # Check final state
    print("\n--- Final weight status ---")
    total_materialized = 0
    for i, layer in enumerate(model.decoder.layers):
        qkv_numel = layer.self_attention.linear_qkv.weight.numel()
        proj_numel = layer.self_attention.linear_proj.weight.numel()
        fc1_numel = layer.mlp.linear_fc1.weight.numel()
        fc2_numel = layer.mlp.linear_fc2.weight.numel()
        total = qkv_numel + proj_numel + fc1_numel + fc2_numel
        total_materialized += total
        if total > 0:
            print(f"  Layer {i}: {total * 2 / 1024**2:.1f} MB still materialized")
    
    print(f"\nTotal materialized weights: {total_materialized * 2 / 1024**2:.1f} MB")
    print(f"sm_gpu resident: {loader.get_memory_stats()['sm_gpu_mb']:.1f} MB")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
