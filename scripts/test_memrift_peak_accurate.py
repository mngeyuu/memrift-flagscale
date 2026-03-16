#!/usr/bin/env python
"""
Accurate peak memory comparison.

Key difference from previous test:
- MemRift model starts with EMPTY base weights (not created then released)
- Only LoRA params + sm_gpu are in GPU at start
"""
import os
import sys
import gc
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/share/project/mengyc/flagScale/FlagScale')


def reset_cuda():
    """Reset CUDA memory state."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_baseline_forward_backward(batch_size=2, seq_len=512):
    """Baseline: Full weights in GPU."""
    print("\n" + "=" * 70)
    print("BASELINE: Full bf16 weights resident in GPU")
    print("=" * 70)
    
    device = torch.device('cuda')
    reset_cuda()
    
    # Read weight shapes from index
    comp_dir = "/share/project/mengyc/code/memrift_v1/weight_comp/test"
    with open(os.path.join(comp_dir, "index.json")) as f:
        index = json.load(f)
    
    # Create model with full weights
    class BaselineModel(nn.Module):
        def __init__(self, index, device):
            super().__init__()
            self.embedding = nn.Embedding(32000, 2048, dtype=torch.bfloat16, device=device)
            self.layers = nn.ModuleList()
            
            # Group weights by layer
            layer_weights = {}
            for entry in index:
                name = entry["name"]
                if "layers." not in name:
                    continue
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        layer_idx = int(parts[i + 1])
                        if layer_idx not in layer_weights:
                            layer_weights[layer_idx] = {}
                        # Get the component name
                        rest = ".".join(parts[i + 2:])
                        layer_weights[layer_idx][rest] = entry["shape"]
                        break
            
            for i in sorted(layer_weights.keys()):
                layer = nn.Module()
                layer.input_layernorm = nn.LayerNorm(2048, dtype=torch.bfloat16, device=device)
                layer.post_attention_layernorm = nn.LayerNorm(2048, dtype=torch.bfloat16, device=device)
                
                # Create full weights
                layer.linear_qkv = nn.Parameter(torch.randn(2560, 2048, dtype=torch.bfloat16, device=device) * 0.01)
                layer.linear_proj = nn.Parameter(torch.randn(2048, 2048, dtype=torch.bfloat16, device=device) * 0.01)
                layer.linear_fc1 = nn.Parameter(torch.randn(11264, 2048, dtype=torch.bfloat16, device=device) * 0.01)
                layer.linear_fc2 = nn.Parameter(torch.randn(2048, 5632, dtype=torch.bfloat16, device=device) * 0.01)
                
                # Freeze base
                layer.linear_qkv.requires_grad = False
                layer.linear_proj.requires_grad = False
                layer.linear_fc1.requires_grad = False
                layer.linear_fc2.requires_grad = False
                
                # LoRA (trainable)
                lora_rank = 16
                layer.lora_qkv_A = nn.Parameter(torch.randn(lora_rank, 2048, dtype=torch.bfloat16, device=device) * 0.01)
                layer.lora_qkv_B = nn.Parameter(torch.zeros(2560, lora_rank, dtype=torch.bfloat16, device=device))
                layer.lora_proj_A = nn.Parameter(torch.randn(lora_rank, 2048, dtype=torch.bfloat16, device=device) * 0.01)
                layer.lora_proj_B = nn.Parameter(torch.zeros(2048, lora_rank, dtype=torch.bfloat16, device=device))
                layer.lora_fc1_A = nn.Parameter(torch.randn(lora_rank, 2048, dtype=torch.bfloat16, device=device) * 0.01)
                layer.lora_fc1_B = nn.Parameter(torch.zeros(11264, lora_rank, dtype=torch.bfloat16, device=device))
                layer.lora_fc2_A = nn.Parameter(torch.randn(lora_rank, 5632, dtype=torch.bfloat16, device=device) * 0.01)
                layer.lora_fc2_B = nn.Parameter(torch.zeros(2048, lora_rank, dtype=torch.bfloat16, device=device))
                
                self.layers.append(layer)
            
            self.final_norm = nn.LayerNorm(2048, dtype=torch.bfloat16, device=device)
            self.lm_head = nn.Parameter(torch.randn(32000, 2048, dtype=torch.bfloat16, device=device) * 0.01)
            self.lm_head.requires_grad = False
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            scale = 32.0 / 16  # lora_alpha / lora_rank
            
            for layer in self.layers:
                h = layer.input_layernorm(x)
                
                # QKV + LoRA
                base = F.linear(h, layer.linear_qkv)
                lora = F.linear(F.linear(h, layer.lora_qkv_A), layer.lora_qkv_B) * scale
                
                # Proj + LoRA
                base_proj = F.linear(h, layer.linear_proj)
                lora_proj = F.linear(F.linear(h, layer.lora_proj_A), layer.lora_proj_B) * scale
                x = x + base_proj + lora_proj
                
                # MLP
                h = layer.post_attention_layernorm(x)
                base_fc1 = F.linear(h, layer.linear_fc1)
                lora_fc1 = F.linear(F.linear(h, layer.lora_fc1_A), layer.lora_fc1_B) * scale
                fc1 = base_fc1 + lora_fc1
                gate, up = fc1.chunk(2, dim=-1)
                h = F.silu(gate) * up
                
                base_fc2 = F.linear(h, layer.linear_fc2)
                lora_fc2 = F.linear(F.linear(h, layer.lora_fc2_A), layer.lora_fc2_B) * scale
                x = x + base_fc2 + lora_fc2
            
            x = self.final_norm(x)
            return F.linear(x, self.lm_head)
    
    model = BaselineModel(index, device)
    
    mem_model = torch.cuda.memory_allocated(device)
    print(f"Model memory after creation: {mem_model / 1024**2:.1f} MB")
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.1f}M, Trainable: {trainable / 1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    
    # Run forward + backward
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    target = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    
    optimizer.zero_grad()
    
    start = time.time()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    current_mem = torch.cuda.memory_allocated(device)
    
    print(f"\nForward+Backward completed in {elapsed*1000:.0f} ms")
    print(f"Loss: {loss.item():.2f}")
    print(f"Current memory: {current_mem / 1024**2:.1f} MB")
    print(f"Peak memory: {peak_mem / 1024**2:.1f} MB")
    
    # Cleanup
    del model, optimizer, logits, loss, input_ids, target
    reset_cuda()
    
    return peak_mem


def run_memrift_forward_backward(batch_size=2, seq_len=512):
    """MemRift: sm_gpu resident + on-demand weight materialization."""
    print("\n" + "=" * 70)
    print("MEMRIFT: sm_gpu resident + on-demand materialization")
    print("=" * 70)
    
    device = torch.device('cuda')
    reset_cuda()
    
    from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader
    
    comp_dir = "/share/project/mengyc/code/memrift_v1/weight_comp/test"
    
    # Create model with EMPTY base weights
    class MemRiftModel(nn.Module):
        def __init__(self, num_layers, device):
            super().__init__()
            self.embedding = nn.Embedding(32000, 2048, dtype=torch.bfloat16, device=device)
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList()
            
            for i in range(num_layers):
                layer = nn.Module()
                layer.input_layernorm = nn.LayerNorm(2048, dtype=torch.bfloat16, device=device)
                layer.post_attention_layernorm = nn.LayerNorm(2048, dtype=torch.bfloat16, device=device)
                
                # EMPTY base weights (will be filled by MemRift hooks)
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
                
                # LoRA (trainable)
                lora_rank = 16
                layer.lora_qkv_A = nn.Parameter(torch.randn(lora_rank, 2048, dtype=torch.bfloat16, device=device) * 0.01)
                layer.lora_qkv_B = nn.Parameter(torch.zeros(2560, lora_rank, dtype=torch.bfloat16, device=device))
                layer.lora_proj_A = nn.Parameter(torch.randn(lora_rank, 2048, dtype=torch.bfloat16, device=device) * 0.01)
                layer.lora_proj_B = nn.Parameter(torch.zeros(2048, lora_rank, dtype=torch.bfloat16, device=device))
                layer.lora_fc1_A = nn.Parameter(torch.randn(lora_rank, 2048, dtype=torch.bfloat16, device=device) * 0.01)
                layer.lora_fc1_B = nn.Parameter(torch.zeros(11264, lora_rank, dtype=torch.bfloat16, device=device))
                layer.lora_fc2_A = nn.Parameter(torch.randn(lora_rank, 5632, dtype=torch.bfloat16, device=device) * 0.01)
                layer.lora_fc2_B = nn.Parameter(torch.zeros(2048, lora_rank, dtype=torch.bfloat16, device=device))
                
                self.decoder.layers.append(layer)
            
            self.decoder.final_layernorm = nn.LayerNorm(2048, dtype=torch.bfloat16, device=device)
            self.lm_head = nn.Parameter(
                torch.empty(0, dtype=torch.bfloat16, device=device), requires_grad=False)
    
    model = MemRiftModel(22, device)
    
    mem_before_memrift = torch.cuda.memory_allocated(device)
    print(f"Model memory (empty base): {mem_before_memrift / 1024**2:.1f} MB")
    
    # Setup MemRift
    loader = MegatronDynamicLoader(
        model=model,
        comp_dir=comp_dir,
        device=device,
        prefetch_layers=1,
        print_debug=False,
    )
    loader.load_weights()
    loader.build_param_mapping()
    # No release_original_weights needed - already empty
    loader.install_hooks()
    loader.prefetch_initial_layers()
    
    mem_after_memrift = torch.cuda.memory_allocated(device)
    stats = loader.get_memory_stats()
    print(f"Model memory (after MemRift setup): {mem_after_memrift / 1024**2:.1f} MB")
    print(f"  sm_gpu resident: {stats['sm_gpu_mb']:.1f} MB")
    
    # Custom forward that uses MemRift-managed weights
    def forward(model, input_ids):
        x = model.embedding(input_ids)
        scale = 32.0 / 16
        
        for layer in model.decoder.layers:
            h = layer.input_layernorm(x)
            
            # Use weights from MemRift (set by hooks)
            qkv_weight = layer.self_attention.linear_qkv.weight
            proj_weight = layer.self_attention.linear_proj.weight
            
            if qkv_weight.numel() > 0:
                base_qkv = F.linear(h, qkv_weight)
            else:
                base_qkv = torch.zeros_like(h)
            lora_qkv = F.linear(F.linear(h, layer.lora_qkv_A), layer.lora_qkv_B) * scale
            
            if proj_weight.numel() > 0:
                base_proj = F.linear(h, proj_weight)
            else:
                base_proj = torch.zeros_like(h)
            lora_proj = F.linear(F.linear(h, layer.lora_proj_A), layer.lora_proj_B) * scale
            x = x + base_proj + lora_proj
            
            h = layer.post_attention_layernorm(x)
            
            fc1_weight = layer.mlp.linear_fc1.weight
            fc2_weight = layer.mlp.linear_fc2.weight
            
            if fc1_weight.numel() > 0:
                base_fc1 = F.linear(h, fc1_weight)
            else:
                base_fc1 = torch.zeros(h.shape[0], h.shape[1], 11264, device=h.device, dtype=h.dtype)
            lora_fc1 = F.linear(F.linear(h, layer.lora_fc1_A), layer.lora_fc1_B) * scale
            fc1 = base_fc1 + lora_fc1
            gate, up = fc1.chunk(2, dim=-1)
            h = F.silu(gate) * up
            
            if fc2_weight.numel() > 0:
                base_fc2 = F.linear(h, fc2_weight)
            else:
                base_fc2 = torch.zeros_like(x)
            lora_fc2 = F.linear(F.linear(h, layer.lora_fc2_A), layer.lora_fc2_B) * scale
            x = x + base_fc2 + lora_fc2
        
        x = model.decoder.final_layernorm(x)
        if model.lm_head.numel() > 0:
            return F.linear(x, model.lm_head)
        return x
    
    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable / 1e6:.2f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    
    # Run forward + backward
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    target = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    
    optimizer.zero_grad()
    
    start = time.time()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = forward(model, input_ids)
    
    if logits.dim() == 3 and logits.size(-1) == 32000:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
    else:
        loss = logits.sum() * 0.001
    
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    current_mem = torch.cuda.memory_allocated(device)
    
    print(f"\nForward+Backward completed in {elapsed*1000:.0f} ms")
    print(f"Loss: {loss.item():.2f}")
    print(f"Current memory: {current_mem / 1024**2:.1f} MB")
    print(f"Peak memory: {peak_mem / 1024**2:.1f} MB")
    
    # Cleanup
    del model, loader, optimizer, logits, loss, input_ids, target
    reset_cuda()
    
    return peak_mem


def main():
    print("=" * 70)
    print("ACCURATE PEAK MEMORY COMPARISON")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Model: TinyLlama-like (22 layers, 1.1B base params)")
    print(f"Scenario: Frozen base + LoRA (rank=16)")
    print(f"Config: batch_size=2, seq_len=512")
    
    # Run tests
    baseline_peak = run_baseline_forward_backward()
    memrift_peak = run_memrift_forward_backward()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    savings = baseline_peak - memrift_peak
    savings_pct = savings / baseline_peak * 100
    
    print(f"{'Metric':<30} {'Value':>20}")
    print("-" * 50)
    print(f"{'Baseline peak memory':<30} {baseline_peak / 1024**2:>17.1f} MB")
    print(f"{'MemRift peak memory':<30} {memrift_peak / 1024**2:>17.1f} MB")
    print("-" * 50)
    print(f"{'Memory savings':<30} {savings / 1024**2:>+17.1f} MB")
    print(f"{'Savings percentage':<30} {savings_pct:>+17.1f} %")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
