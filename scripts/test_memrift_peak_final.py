#!/usr/bin/env python
"""
Final peak memory comparison with proper hooks triggering.
"""
import os
import sys
import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/share/project/mengyc/flagScale/FlagScale')


def reset_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


class TransformerLayer(nn.Module):
    """Transformer layer with full forward implementation."""
    def __init__(self, hidden_size, ffn_hidden_size, device, init_weights=True):
        super().__init__()
        qkv_size = hidden_size + hidden_size // 8 + hidden_size // 8  # 2560
        
        self.input_layernorm = nn.LayerNorm(hidden_size, dtype=torch.bfloat16, device=device)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, dtype=torch.bfloat16, device=device)
        
        if init_weights:
            # Full weights (for baseline)
            self.linear_qkv = nn.Parameter(torch.randn(qkv_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.01)
            self.linear_proj = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device=device) * 0.01)
            self.linear_fc1 = nn.Parameter(torch.randn(ffn_hidden_size * 2, hidden_size, dtype=torch.bfloat16, device=device) * 0.01)
            self.linear_fc2 = nn.Parameter(torch.randn(hidden_size, ffn_hidden_size, dtype=torch.bfloat16, device=device) * 0.01)
        else:
            # Empty weights (for MemRift)
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
        
        self.init_weights = init_weights
    
    def forward(self, x):
        h = self.input_layernorm(x)
        
        if self.init_weights:
            # Baseline path
            qkv = F.linear(h, self.linear_qkv)
            attn_out = F.linear(h, self.linear_proj)
            x = x + attn_out
            
            h = self.post_attention_layernorm(x)
            fc1_out = F.linear(h, self.linear_fc1)
            gate, up = fc1_out.chunk(2, dim=-1)
            h = F.silu(gate) * up
            x = x + F.linear(h, self.linear_fc2)
        else:
            # MemRift path
            qkv_w = self.self_attention.linear_qkv.weight
            proj_w = self.self_attention.linear_proj.weight
            fc1_w = self.mlp.linear_fc1.weight
            fc2_w = self.mlp.linear_fc2.weight
            
            if qkv_w.numel() > 0:
                qkv = F.linear(h, qkv_w)
            if proj_w.numel() > 0:
                x = x + F.linear(h, proj_w)
            
            h = self.post_attention_layernorm(x)
            if fc1_w.numel() > 0:
                fc1_out = F.linear(h, fc1_w)
                gate, up = fc1_out.chunk(2, dim=-1)
                h = F.silu(gate) * up
                if fc2_w.numel() > 0:
                    x = x + F.linear(h, fc2_w)
        
        return x


def run_baseline(batch_size=2, seq_len=512, num_layers=22):
    """Baseline: Full weights in GPU."""
    print("\n" + "=" * 70)
    print("BASELINE: Full bf16 weights resident in GPU")
    print("=" * 70)
    
    device = torch.device('cuda')
    reset_cuda()
    
    class BaselineModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32000, 2048, dtype=torch.bfloat16, device=device)
            self.layers = nn.ModuleList([
                TransformerLayer(2048, 5632, device, init_weights=True)
                for _ in range(num_layers)
            ])
            self.final_norm = nn.LayerNorm(2048, dtype=torch.bfloat16, device=device)
            self.lm_head = nn.Parameter(torch.randn(32000, 2048, dtype=torch.bfloat16, device=device) * 0.01)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            x = self.final_norm(x)
            return F.linear(x, self.lm_head)
    
    model = BaselineModel()
    model.eval()
    
    mem_model = torch.cuda.memory_allocated(device)
    print(f"Model memory: {mem_model / 1024**2:.1f} MB")
    
    # Forward pass
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    
    start = time.time()
    with torch.no_grad():
        output = model(input_ids)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    
    print(f"Forward time: {elapsed*1000:.0f} ms")
    print(f"Peak memory: {peak_mem / 1024**2:.1f} MB")
    
    del model, output, input_ids
    reset_cuda()
    
    return peak_mem, elapsed


def run_memrift(batch_size=2, seq_len=512, num_layers=22):
    """MemRift: On-demand weight materialization."""
    print("\n" + "=" * 70)
    print("MEMRIFT: On-demand weight materialization")
    print("=" * 70)
    
    device = torch.device('cuda')
    reset_cuda()
    
    from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader
    
    comp_dir = "/share/project/mengyc/code/memrift_v1/weight_comp/test"
    
    class MemRiftModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32000, 2048, dtype=torch.bfloat16, device=device)
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([
                TransformerLayer(2048, 5632, device, init_weights=False)
                for _ in range(num_layers)
            ])
            self.decoder.final_layernorm = nn.LayerNorm(2048, dtype=torch.bfloat16, device=device)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.decoder.layers:
                x = layer(x)
            return self.decoder.final_layernorm(x)
    
    model = MemRiftModel()
    model.eval()
    
    mem_before = torch.cuda.memory_allocated(device)
    print(f"Model memory (empty weights): {mem_before / 1024**2:.1f} MB")
    
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
    loader.install_hooks()
    loader.prefetch_initial_layers()
    
    mem_after = torch.cuda.memory_allocated(device)
    stats = loader.get_memory_stats()
    print(f"After MemRift setup: {mem_after / 1024**2:.1f} MB")
    print(f"  sm_gpu resident: {stats['sm_gpu_mb']:.1f} MB")
    
    # Forward pass
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    
    start = time.time()
    with torch.no_grad():
        output = model(input_ids)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    
    print(f"Forward time: {elapsed*1000:.0f} ms")
    print(f"Peak memory: {peak_mem / 1024**2:.1f} MB")
    
    del model, loader, output, input_ids
    reset_cuda()
    
    return peak_mem, elapsed


def main():
    print("=" * 70)
    print("PEAK MEMORY COMPARISON: Baseline vs MemRift")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Model: TinyLlama-like (22 layers, 1.1B params)")
    
    configs = [
        {"batch_size": 1, "seq_len": 256},
        {"batch_size": 2, "seq_len": 512},
        {"batch_size": 4, "seq_len": 512},
        {"batch_size": 4, "seq_len": 1024},
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n{'#' * 70}")
        print(f"Config: batch_size={cfg['batch_size']}, seq_len={cfg['seq_len']}")
        print(f"{'#' * 70}")
        
        baseline_peak, baseline_time = run_baseline(**cfg)
        memrift_peak, memrift_time = run_memrift(**cfg)
        
        results.append({
            'config': cfg,
            'baseline_peak': baseline_peak,
            'memrift_peak': memrift_peak,
            'baseline_time': baseline_time,
            'memrift_time': memrift_time,
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'Baseline Peak':>15} {'MemRift Peak':>15} {'Saved':>15} {'Time Ratio':>12}")
    print("-" * 77)
    
    for r in results:
        cfg = f"bs={r['config']['batch_size']}, seq={r['config']['seq_len']}"
        saved = r['baseline_peak'] - r['memrift_peak']
        saved_pct = saved / r['baseline_peak'] * 100
        time_ratio = r['memrift_time'] / r['baseline_time']
        
        print(f"{cfg:<20} {r['baseline_peak']/1024**2:>12.1f} MB {r['memrift_peak']/1024**2:>12.1f} MB "
              f"{saved/1024**2:>+12.1f} MB {time_ratio:>11.1f}x")
    
    print("-" * 77)
    
    # Average
    total_baseline = sum(r['baseline_peak'] for r in results)
    total_memrift = sum(r['memrift_peak'] for r in results)
    avg_saved_pct = (total_baseline - total_memrift) / total_baseline * 100
    
    print(f"\nAverage memory reduction: {avg_saved_pct:.1f}%")
    print("\nNote: MemRift trades time for memory. Time overhead is due to CPU decompression.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
