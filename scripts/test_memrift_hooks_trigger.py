#!/usr/bin/env python
"""
Test that hooks are properly triggered during forward pass.
The key is to call module(input) to trigger forward_pre_hook.
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
    print("TEST: Hooks trigger during forward pass")
    print("=" * 70)
    
    device = torch.device('cuda')
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader
    
    comp_dir = "/share/project/mengyc/code/memrift_v1/weight_comp/test"
    
    # Create model where layers have proper forward methods
    class TransformerLayer(nn.Module):
        """Layer with proper forward that uses all weights."""
        def __init__(self, hidden_size, ffn_hidden_size, device):
            super().__init__()
            qkv_size = hidden_size + hidden_size // 8 + hidden_size // 8  # 2560
            
            self.input_layernorm = nn.LayerNorm(hidden_size, dtype=torch.bfloat16, device=device)
            self.post_attention_layernorm = nn.LayerNorm(hidden_size, dtype=torch.bfloat16, device=device)
            
            # These will be managed by MemRift (start empty)
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
        
        def forward(self, x):
            # Attention
            h = self.input_layernorm(x)
            
            qkv_w = self.self_attention.linear_qkv.weight
            proj_w = self.self_attention.linear_proj.weight
            
            if qkv_w.numel() > 0:
                qkv = F.linear(h, qkv_w)  # [B, S, qkv_size]
            
            if proj_w.numel() > 0:
                attn_out = F.linear(h, proj_w)
                x = x + attn_out
            
            # MLP
            h = self.post_attention_layernorm(x)
            
            fc1_w = self.mlp.linear_fc1.weight
            fc2_w = self.mlp.linear_fc2.weight
            
            if fc1_w.numel() > 0:
                fc1_out = F.linear(h, fc1_w)
                gate, up = fc1_out.chunk(2, dim=-1)
                h = F.silu(gate) * up
                
                if fc2_w.numel() > 0:
                    x = x + F.linear(h, fc2_w)
            
            return x
    
    class MockModel(nn.Module):
        def __init__(self, num_layers, hidden_size, ffn_hidden_size, device):
            super().__init__()
            self.embedding = nn.Embedding(32000, hidden_size, dtype=torch.bfloat16, device=device)
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([
                TransformerLayer(hidden_size, ffn_hidden_size, device)
                for _ in range(num_layers)
            ])
            self.decoder.final_layernorm = nn.LayerNorm(hidden_size, dtype=torch.bfloat16, device=device)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.decoder.layers:
                x = layer(x)  # This triggers forward_pre_hook!
            return self.decoder.final_layernorm(x)
    
    model = MockModel(22, 2048, 5632, device)
    
    mem_init = torch.cuda.memory_allocated(device)
    print(f"Initial memory (empty weights): {mem_init / 1024**2:.1f} MB")
    
    # Setup MemRift
    loader = MegatronDynamicLoader(
        model=model,
        comp_dir=comp_dir,
        device=device,
        prefetch_layers=1,
        print_debug=True,
    )
    
    loader.load_weights()
    loader.build_param_mapping()
    loader.install_hooks()
    loader.prefetch_initial_layers()
    
    mem_setup = torch.cuda.memory_allocated(device)
    print(f"\nAfter MemRift setup: {mem_setup / 1024**2:.1f} MB")
    print(f"  sm_gpu: {loader.get_memory_stats()['sm_gpu_mb']:.1f} MB")
    
    # Check initial state
    print("\nInitial weight status:")
    for i, layer in enumerate(model.decoder.layers[:5]):
        qkv = layer.self_attention.linear_qkv.weight.numel()
        print(f"  Layer {i}: qkv.numel={qkv}")
    
    # Forward pass - this should trigger hooks
    print("\n" + "=" * 70)
    print("FORWARD PASS (should trigger hooks)")
    print("=" * 70)
    
    input_ids = torch.randint(0, 32000, (1, 32), device=device)
    
    print(f"\nMemory before forward: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
    
    with torch.no_grad():
        output = model(input_ids)
    
    print(f"Memory after forward: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
    print(f"Peak memory: {torch.cuda.max_memory_allocated(device) / 1024**2:.1f} MB")
    
    # Check final state
    print("\nFinal weight status:")
    materialized_count = 0
    for i, layer in enumerate(model.decoder.layers):
        qkv = layer.self_attention.linear_qkv.weight.numel()
        if qkv > 0:
            materialized_count += 1
            print(f"  Layer {i}: qkv.numel={qkv} (materialized)")
    print(f"\nTotal layers with materialized weights: {materialized_count}/22")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
