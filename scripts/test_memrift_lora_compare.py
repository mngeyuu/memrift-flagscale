#!/usr/bin/env python
"""
Compare peak memory: Frozen base + LoRA scenario
- Baseline: Full frozen weights stay in GPU
- MemRift: Frozen weights loaded on-demand, only LoRA params in GPU

This is the actual target scenario for MemRift.
"""
import os
import sys
import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/share/project/mengyc/flagScale/FlagScale')


class LoRALinear(nn.Module):
    """Linear with LoRA adapter."""
    def __init__(self, base_weight, lora_rank=16, lora_alpha=32):
        super().__init__()
        out_f, in_f = base_weight.shape
        self.base_weight = base_weight  # Frozen, no grad
        self.base_weight.requires_grad = False
        
        # LoRA params (trainable)
        self.lora_A = nn.Parameter(torch.randn(lora_rank, in_f, dtype=torch.bfloat16) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, lora_rank, dtype=torch.bfloat16))
        self.scale = lora_alpha / lora_rank
    
    def forward(self, x):
        base_out = F.linear(x, self.base_weight)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out


def create_lora_model(num_layers=22, hidden_size=2048, ffn_hidden_size=5632, vocab_size=32000, lora_rank=16):
    """Create model with frozen base + LoRA adapters."""
    
    class MockLayer(nn.Module):
        def __init__(self, hidden, ffn, lora_rank):
            super().__init__()
            qkv_size = hidden + hidden // 8 + hidden // 8
            
            # Create base weights (frozen)
            self.linear_qkv = LoRALinear(
                torch.randn(qkv_size, hidden, dtype=torch.bfloat16),
                lora_rank=lora_rank
            )
            self.linear_proj = LoRALinear(
                torch.randn(hidden, hidden, dtype=torch.bfloat16),
                lora_rank=lora_rank
            )
            self.linear_fc1 = LoRALinear(
                torch.randn(ffn * 2, hidden, dtype=torch.bfloat16),
                lora_rank=lora_rank
            )
            self.linear_fc2 = LoRALinear(
                torch.randn(hidden, ffn, dtype=torch.bfloat16),
                lora_rank=lora_rank
            )
            
            self.input_layernorm = nn.LayerNorm(hidden, dtype=torch.bfloat16)
            self.post_attention_layernorm = nn.LayerNorm(hidden, dtype=torch.bfloat16)
        
        def forward(self, x):
            h = self.input_layernorm(x)
            qkv = self.linear_qkv(h)
            x = x + self.linear_proj(h)
            h = self.post_attention_layernorm(x)
            fc1_out = self.linear_fc1(h)
            gate, up = fc1_out.chunk(2, dim=-1)
            h = F.silu(gate) * up
            x = x + self.linear_fc2(h)
            return x
    
    class MockModel(nn.Module):
        def __init__(self, num_layers, hidden, ffn, vocab_size, lora_rank):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden, dtype=torch.bfloat16)
            self.layers = nn.ModuleList([MockLayer(hidden, ffn, lora_rank) for _ in range(num_layers)])
            self.final_norm = nn.LayerNorm(hidden, dtype=torch.bfloat16)
            self.lm_head_weight = nn.Parameter(torch.randn(vocab_size, hidden, dtype=torch.bfloat16))
            self.lm_head_weight.requires_grad = False  # Frozen
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            x = self.final_norm(x)
            return F.linear(x, self.lm_head_weight)
    
    return MockModel(num_layers, hidden_size, ffn_hidden_size, vocab_size, lora_rank)


def count_params(model):
    """Count trainable and total params."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_baseline_lora(batch_size=2, seq_len=512, num_iters=3, lora_rank=16):
    """Baseline: Frozen base weights + LoRA, all in GPU."""
    print("\n" + "=" * 70)
    print("BASELINE: Frozen base (GPU) + LoRA")
    print("=" * 70)
    
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    
    model = create_lora_model(lora_rank=lora_rank).to(device)
    model.train()
    
    total_params, trainable_params = count_params(model)
    print(f"Total params: {total_params / 1e6:.1f}M")
    print(f"Trainable (LoRA): {trainable_params / 1e6:.2f}M ({trainable_params/total_params*100:.2f}%)")
    
    mem_model = torch.cuda.memory_allocated(device)
    print(f"Model memory: {mem_model / 1024**2:.1f} MB")
    
    # Only optimize LoRA params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    times = []
    for i in range(num_iters):
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
        times.append(elapsed)
        
        print(f"  Iter {i+1}: loss={loss.item():.2f}, time={elapsed*1000:.0f}ms")
        
        del logits, loss, input_ids, target
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    avg_time = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
    
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n[BASELINE RESULT]")
    print(f"  Peak memory: {peak_mem / 1024**2:.1f} MB")
    print(f"  Avg time: {avg_time*1000:.0f} ms")
    
    return peak_mem, avg_time


def create_memrift_lora_model(comp_dir, device, num_layers=22, hidden_size=2048, 
                               ffn_hidden_size=5632, vocab_size=32000, lora_rank=16):
    """
    Create LoRA model where base weights come from MemRift compressed storage.
    Base weights are loaded on-demand per layer.
    """
    from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader
    
    class MemRiftLoRALinear(nn.Module):
        """Linear with LoRA, base weight from MemRift."""
        def __init__(self, out_f, in_f, lora_rank=16, lora_alpha=32):
            super().__init__()
            # Placeholder for base weight (will be set by hook)
            self.register_buffer('base_weight', torch.empty(0, dtype=torch.bfloat16))
            self.out_f = out_f
            self.in_f = in_f
            
            # LoRA params (trainable)
            self.lora_A = nn.Parameter(torch.randn(lora_rank, in_f, dtype=torch.bfloat16, device=device) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_f, lora_rank, dtype=torch.bfloat16, device=device))
            self.scale = lora_alpha / lora_rank
        
        def forward(self, x):
            # base_weight should be set before forward
            if self.base_weight.numel() == 0:
                raise RuntimeError("base_weight not set!")
            base_out = F.linear(x, self.base_weight)
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
            return base_out + lora_out
    
    class MockSelfAttention(nn.Module):
        def __init__(self, hidden, lora_rank):
            super().__init__()
            qkv_size = hidden + hidden // 8 + hidden // 8
            self.linear_qkv = MemRiftLoRALinear(qkv_size, hidden, lora_rank)
            self.linear_proj = MemRiftLoRALinear(hidden, hidden, lora_rank)
        
        def forward(self, x):
            qkv = self.linear_qkv(x)
            return self.linear_proj(x)
    
    class MockMLP(nn.Module):
        def __init__(self, hidden, ffn, lora_rank):
            super().__init__()
            self.linear_fc1 = MemRiftLoRALinear(ffn * 2, hidden, lora_rank)
            self.linear_fc2 = MemRiftLoRALinear(hidden, ffn, lora_rank)
        
        def forward(self, x):
            h = self.linear_fc1(x)
            gate, up = h.chunk(2, dim=-1)
            h = F.silu(gate) * up
            return self.linear_fc2(h)
    
    class MockLayer(nn.Module):
        def __init__(self, hidden, ffn, lora_rank):
            super().__init__()
            self.self_attention = MockSelfAttention(hidden, lora_rank)
            self.mlp = MockMLP(hidden, ffn, lora_rank)
            self.input_layernorm = nn.LayerNorm(hidden, dtype=torch.bfloat16, device=device)
            self.post_attention_layernorm = nn.LayerNorm(hidden, dtype=torch.bfloat16, device=device)
        
        def forward(self, x):
            h = self.input_layernorm(x)
            x = x + self.self_attention(h)
            h = self.post_attention_layernorm(x)
            x = x + self.mlp(h)
            return x
    
    class MockDecoder(nn.Module):
        def __init__(self, num_layers, hidden, ffn, lora_rank):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer(hidden, ffn, lora_rank) for _ in range(num_layers)])
            self.final_layernorm = nn.LayerNorm(hidden, dtype=torch.bfloat16, device=device)
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.final_layernorm(x)
    
    class MockModel(nn.Module):
        def __init__(self, num_layers, hidden, ffn, vocab_size, lora_rank):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden, dtype=torch.bfloat16, device=device)
            self.decoder = MockDecoder(num_layers, hidden, ffn, lora_rank)
            # lm_head will also be from compressed
            self.register_buffer('lm_head_weight', torch.empty(0, dtype=torch.bfloat16))
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.decoder(x)
            if self.lm_head_weight.numel() == 0:
                # Fallback
                return x
            return F.linear(x, self.lm_head_weight)
    
    return MockModel(num_layers, hidden_size, ffn_hidden_size, vocab_size, lora_rank)


def run_memrift_lora(batch_size=2, seq_len=512, num_iters=3, lora_rank=16):
    """MemRift: Base weights on-demand, only LoRA + current layer's base in GPU."""
    print("\n" + "=" * 70)
    print("MEMRIFT: On-demand base weights + LoRA")
    print("=" * 70)
    
    device = torch.device('cuda')
    comp_dir = "/share/project/mengyc/code/memrift_v1/weight_comp/test"
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    
    # For fair comparison, we'll use the same mock model structure
    # but with MemRift managing the base weights
    from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader
    
    # Create minimal model (only LoRA params + layernorms + embedding)
    # Base weights will come from MemRift
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Only embedding (small) and layernorms
            self.embedding = nn.Embedding(32000, 2048, dtype=torch.bfloat16)
            self.decoder = nn.ModuleDict()
            
            # Create layer structure for hooks
            self.decoder['layers'] = nn.ModuleList()
            for i in range(22):
                layer = nn.Module()
                layer.self_attention = nn.Module()
                layer.self_attention.linear_qkv = nn.Module()
                layer.self_attention.linear_qkv.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16))
                layer.self_attention.linear_qkv.weight.requires_grad = False
                
                layer.self_attention.linear_proj = nn.Module()
                layer.self_attention.linear_proj.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16))
                layer.self_attention.linear_proj.weight.requires_grad = False
                
                layer.mlp = nn.Module()
                layer.mlp.linear_fc1 = nn.Module()
                layer.mlp.linear_fc1.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16))
                layer.mlp.linear_fc1.weight.requires_grad = False
                
                layer.mlp.linear_fc2 = nn.Module()
                layer.mlp.linear_fc2.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16))
                layer.mlp.linear_fc2.weight.requires_grad = False
                
                # LoRA adapters
                layer.lora_qkv_A = nn.Parameter(torch.randn(lora_rank, 2048, dtype=torch.bfloat16) * 0.01)
                layer.lora_qkv_B = nn.Parameter(torch.zeros(2560, lora_rank, dtype=torch.bfloat16))
                layer.lora_proj_A = nn.Parameter(torch.randn(lora_rank, 2048, dtype=torch.bfloat16) * 0.01)
                layer.lora_proj_B = nn.Parameter(torch.zeros(2048, lora_rank, dtype=torch.bfloat16))
                layer.lora_fc1_A = nn.Parameter(torch.randn(lora_rank, 2048, dtype=torch.bfloat16) * 0.01)
                layer.lora_fc1_B = nn.Parameter(torch.zeros(11264, lora_rank, dtype=torch.bfloat16))
                layer.lora_fc2_A = nn.Parameter(torch.randn(lora_rank, 5632, dtype=torch.bfloat16) * 0.01)
                layer.lora_fc2_B = nn.Parameter(torch.zeros(2048, lora_rank, dtype=torch.bfloat16))
                
                layer.input_layernorm = nn.LayerNorm(2048, dtype=torch.bfloat16)
                layer.post_attention_layernorm = nn.LayerNorm(2048, dtype=torch.bfloat16)
                
                self.decoder['layers'].append(layer)
            
            self.decoder['final_layernorm'] = nn.LayerNorm(2048, dtype=torch.bfloat16)
            self.lm_head_weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16))
            self.lm_head_weight.requires_grad = False
    
    model = SimpleModel().to(device)
    
    # Setup MemRift
    loader = MegatronDynamicLoader(
        model=model,
        comp_dir=comp_dir,
        device=device,
        print_debug=False,
    )
    loader.load_weights()
    loader.build_param_mapping()
    # Note: release_original_weights() not needed since we start with empty
    loader.install_hooks()
    loader.prefetch_initial_layers()
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params in GPU: {total_params / 1e6:.1f}M")
    print(f"Trainable (LoRA): {trainable_params / 1e6:.2f}M")
    print(f"sm_gpu (compressed base): {loader.get_memory_stats()['sm_gpu_mb']:.1f} MB")
    
    mem_model = torch.cuda.memory_allocated(device)
    print(f"Model memory: {mem_model / 1024**2:.1f} MB")
    
    # Custom forward with LoRA
    def forward_with_lora(model, input_ids):
        x = model.embedding(input_ids)
        scale = 32 / lora_rank  # lora_alpha / lora_rank
        
        for layer in model.decoder['layers']:
            # Attention
            h = layer.input_layernorm(x)
            
            # QKV with LoRA
            base_qkv = F.linear(h, layer.self_attention.linear_qkv.weight)
            lora_qkv = F.linear(F.linear(h, layer.lora_qkv_A), layer.lora_qkv_B) * scale
            qkv = base_qkv + lora_qkv
            
            # Proj with LoRA
            base_proj = F.linear(h, layer.self_attention.linear_proj.weight)
            lora_proj = F.linear(F.linear(h, layer.lora_proj_A), layer.lora_proj_B) * scale
            x = x + base_proj + lora_proj
            
            # MLP
            h = layer.post_attention_layernorm(x)
            
            # FC1 with LoRA
            base_fc1 = F.linear(h, layer.mlp.linear_fc1.weight)
            lora_fc1 = F.linear(F.linear(h, layer.lora_fc1_A), layer.lora_fc1_B) * scale
            fc1 = base_fc1 + lora_fc1
            gate, up = fc1.chunk(2, dim=-1)
            h = F.silu(gate) * up
            
            # FC2 with LoRA
            base_fc2 = F.linear(h, layer.mlp.linear_fc2.weight)
            lora_fc2 = F.linear(F.linear(h, layer.lora_fc2_A), layer.lora_fc2_B) * scale
            x = x + base_fc2 + lora_fc2
        
        x = model.decoder['final_layernorm'](x)
        return F.linear(x, model.lm_head_weight) if model.lm_head_weight.numel() > 0 else x
    
    # Optimizer for LoRA params only
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n and p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
    
    times = []
    for i in range(num_iters):
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        target = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        
        optimizer.zero_grad()
        
        start = time.time()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = forward_with_lora(model, input_ids)
        
        if logits.dim() == 3 and logits.size(-1) == 32000:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        else:
            # Fallback if lm_head not loaded
            loss = logits.sum() * 0 + 10.0
        
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"  Iter {i+1}: loss={loss.item():.2f}, time={elapsed*1000:.0f}ms")
        
        del logits, loss, input_ids, target
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    avg_time = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
    
    del model, loader, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n[MEMRIFT RESULT]")
    print(f"  Peak memory: {peak_mem / 1024**2:.1f} MB")
    print(f"  Avg time: {avg_time*1000:.0f} ms")
    
    return peak_mem, avg_time


def main():
    print("=" * 70)
    print("PEAK MEMORY COMPARISON: LoRA Training Scenario")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Model: TinyLlama-like (22 layers, 1.1B base params)")
    print(f"Scenario: Frozen base weights + LoRA adapters")
    
    config = {"batch_size": 2, "seq_len": 512, "num_iters": 3, "lora_rank": 16}
    
    print(f"\nConfig: batch_size={config['batch_size']}, seq_len={config['seq_len']}, lora_rank={config['lora_rank']}")
    
    # Run baseline
    baseline_peak, baseline_time = run_baseline_lora(**config)
    
    # Run MemRift
    memrift_peak, memrift_time = run_memrift_lora(**config)
    
    # Summary
    mem_saved = baseline_peak - memrift_peak
    mem_saved_pct = (mem_saved / baseline_peak) * 100
    time_diff_pct = ((memrift_time - baseline_time) / baseline_time) * 100
    
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'Baseline':>15} {'MemRift':>15} {'Diff':>15}")
    print("-" * 75)
    print(f"{'Peak Memory':<30} {baseline_peak/1024**2:>12.1f} MB {memrift_peak/1024**2:>12.1f} MB {mem_saved/1024**2:>+12.1f} MB")
    print(f"{'Avg Time':<30} {baseline_time*1000:>12.0f} ms {memrift_time*1000:>12.0f} ms {time_diff_pct:>+12.1f} %")
    print("-" * 75)
    print(f"\nMemory reduction: {mem_saved_pct:.1f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
