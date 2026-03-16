#!/usr/bin/env python
"""
Compare peak memory usage between baseline (no MemRift) and MemRift enabled.

This script measures:
1. Baseline: Standard training with full weights in GPU
2. MemRift: On-demand weight materialization with compressed storage
"""
import os
import sys
import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/share/project/mengyc/flagScale/FlagScale')


def create_mock_megatron_model(num_layers=22, hidden_size=2048, ffn_hidden_size=5632, vocab_size=32000):
    """Create a mock Megatron-style model matching TinyLlama structure."""
    
    class MockLinear(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_f, in_f, dtype=torch.bfloat16))
        
        def forward(self, x):
            return F.linear(x, self.weight)
    
    class MockSelfAttention(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            qkv_size = hidden + hidden // 8 + hidden // 8  # 2560
            self.linear_qkv = MockLinear(hidden, qkv_size)
            self.linear_proj = MockLinear(hidden, hidden)
        
        def forward(self, x):
            qkv = self.linear_qkv(x)
            return self.linear_proj(x)
    
    class MockMLP(nn.Module):
        def __init__(self, hidden, ffn):
            super().__init__()
            self.linear_fc1 = MockLinear(hidden, ffn * 2)
            self.linear_fc2 = MockLinear(ffn, hidden)
        
        def forward(self, x):
            h = self.linear_fc1(x)
            gate, up = h.chunk(2, dim=-1)
            h = F.silu(gate) * up
            return self.linear_fc2(h)
    
    class MockLayer(nn.Module):
        def __init__(self, hidden, ffn):
            super().__init__()
            self.self_attention = MockSelfAttention(hidden)
            self.mlp = MockMLP(hidden, ffn)
            self.input_layernorm = nn.LayerNorm(hidden, dtype=torch.bfloat16)
            self.post_attention_layernorm = nn.LayerNorm(hidden, dtype=torch.bfloat16)
        
        def forward(self, x):
            h = self.input_layernorm(x)
            x = x + self.self_attention(h)
            h = self.post_attention_layernorm(x)
            x = x + self.mlp(h)
            return x
    
    class MockDecoder(nn.Module):
        def __init__(self, num_layers, hidden, ffn):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer(hidden, ffn) for _ in range(num_layers)])
            self.final_layernorm = nn.LayerNorm(hidden, dtype=torch.bfloat16)
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.final_layernorm(x)
    
    class MockModel(nn.Module):
        def __init__(self, num_layers, hidden, ffn, vocab_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden, dtype=torch.bfloat16)
            self.decoder = MockDecoder(num_layers, hidden, ffn)
            self.lm_head = MockLinear(hidden, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.decoder(x)
            return self.lm_head(x)
    
    return MockModel(num_layers, hidden_size, ffn_hidden_size, vocab_size)


def run_baseline(batch_size=2, seq_len=512, num_iters=3):
    """Run baseline training without MemRift and measure peak memory."""
    print("\n" + "=" * 70)
    print("BASELINE (No MemRift) - Full weights in GPU")
    print("=" * 70)
    
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    
    # Create model
    model = create_mock_megatron_model().to(device)
    model.train()
    
    mem_model = torch.cuda.memory_allocated(device)
    print(f"Model memory: {mem_model / 1024**2:.1f} MB")
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Training loop
    times = []
    for i in range(num_iters):
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        target = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        
        start = time.time()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"  Iter {i+1}: loss={loss.item():.2f}, time={elapsed*1000:.0f}ms")
        
        model.zero_grad(set_to_none=True)
        del logits, loss, input_ids, target
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    avg_time = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n[BASELINE RESULT]")
    print(f"  Peak memory: {peak_mem / 1024**2:.1f} MB")
    print(f"  Avg time (excl warmup): {avg_time*1000:.0f} ms")
    
    return peak_mem, avg_time


def run_memrift(batch_size=2, seq_len=512, num_iters=3):
    """Run training with MemRift and measure peak memory."""
    print("\n" + "=" * 70)
    print("MEMRIFT - On-demand weight materialization")
    print("=" * 70)
    
    device = torch.device('cuda')
    comp_dir = "/share/project/mengyc/code/memrift_v1/weight_comp/test"
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    
    # Create model
    model = create_mock_megatron_model().to(device)
    model.train()
    
    mem_model = torch.cuda.memory_allocated(device)
    print(f"Model memory (before MemRift): {mem_model / 1024**2:.1f} MB")
    
    # Setup MemRift
    from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader
    
    loader = MegatronDynamicLoader(
        model=model,
        comp_dir=comp_dir,
        device=device,
        tp_rank=0,
        tp_size=1,
        prefetch_layers=1,
        print_debug=False,
    )
    
    loader.load_weights()
    loader.build_param_mapping()
    loader.release_original_weights()
    loader.install_hooks()
    loader.prefetch_initial_layers()
    
    mem_after_setup = torch.cuda.memory_allocated(device)
    print(f"Model memory (after MemRift setup): {mem_after_setup / 1024**2:.1f} MB")
    print(f"  sm_gpu (resident): {loader.get_memory_stats()['sm_gpu_mb']:.1f} MB")
    
    # Training loop
    times = []
    for i in range(num_iters):
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        target = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        
        start = time.time()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"  Iter {i+1}: loss={loss.item():.2f}, time={elapsed*1000:.0f}ms")
        
        model.zero_grad(set_to_none=True)
        del logits, loss, input_ids, target
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    avg_time = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
    
    # Cleanup
    del model, loader
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n[MEMRIFT RESULT]")
    print(f"  Peak memory: {peak_mem / 1024**2:.1f} MB")
    print(f"  Avg time (excl warmup): {avg_time*1000:.0f} ms")
    
    return peak_mem, avg_time


def main():
    print("=" * 70)
    print("PEAK MEMORY COMPARISON: Baseline vs MemRift")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Model: TinyLlama-like (22 layers, 1.1B params)")
    
    # Test configurations
    configs = [
        {"batch_size": 2, "seq_len": 256, "num_iters": 3},
        {"batch_size": 2, "seq_len": 512, "num_iters": 3},
        {"batch_size": 4, "seq_len": 512, "num_iters": 3},
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n{'#' * 70}")
        print(f"Config: batch_size={cfg['batch_size']}, seq_len={cfg['seq_len']}")
        print(f"{'#' * 70}")
        
        # Run baseline
        baseline_peak, baseline_time = run_baseline(**cfg)
        
        # Run MemRift
        memrift_peak, memrift_time = run_memrift(**cfg)
        
        # Calculate savings
        mem_saved = baseline_peak - memrift_peak
        mem_saved_pct = (mem_saved / baseline_peak) * 100 if baseline_peak > 0 else 0
        time_overhead = ((memrift_time - baseline_time) / baseline_time) * 100 if baseline_time > 0 else 0
        
        results.append({
            'config': cfg,
            'baseline_peak': baseline_peak,
            'memrift_peak': memrift_peak,
            'mem_saved': mem_saved,
            'mem_saved_pct': mem_saved_pct,
            'baseline_time': baseline_time,
            'memrift_time': memrift_time,
            'time_overhead': time_overhead,
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<25} {'Baseline Peak':>15} {'MemRift Peak':>15} {'Saved':>15} {'Time Overhead':>15}")
    print("-" * 85)
    
    for r in results:
        cfg_str = f"bs={r['config']['batch_size']}, seq={r['config']['seq_len']}"
        print(f"{cfg_str:<25} {r['baseline_peak']/1024**2:>12.1f} MB {r['memrift_peak']/1024**2:>12.1f} MB "
              f"{r['mem_saved']/1024**2:>+12.1f} MB {r['time_overhead']:>+13.1f}%")
    
    print("-" * 85)
    
    # Overall
    avg_saved = sum(r['mem_saved_pct'] for r in results) / len(results)
    avg_overhead = sum(r['time_overhead'] for r in results) / len(results)
    print(f"\nAverage memory reduction: {avg_saved:.1f}%")
    print(f"Average time overhead: {avg_overhead:+.1f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
