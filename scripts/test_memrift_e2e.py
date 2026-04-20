#!/usr/bin/env python
"""
End-to-end test for MemRift integration with FlagScale training.

This script tests:
1. Model creation with MemRift hooks
2. Forward pass with on-demand weight materialization
3. Backward pass with weight release
4. Memory usage tracking
"""
import os
import sys
import gc
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add FlagScale to path
sys.path.insert(0, '/share/project/mengyc/flagScale/FlagScale')


def create_mock_megatron_model(num_layers=22, hidden_size=2048, ffn_hidden_size=5632, vocab_size=32000):
    """
    Create a mock Megatron-style model that matches TinyLlama structure.
    """
    
    class MockLinear(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_f, in_f, dtype=torch.bfloat16))
        
        def forward(self, x):
            return F.linear(x, self.weight)
    
    class MockSelfAttention(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            # QKV merged: hidden -> (q=hidden + k=hidden//8 + v=hidden//8)
            qkv_size = hidden + hidden // 8 + hidden // 8  # 2048 + 256 + 256 = 2560
            self.linear_qkv = MockLinear(hidden, qkv_size)
            self.linear_proj = MockLinear(hidden, hidden)
        
        def forward(self, x):
            qkv = self.linear_qkv(x)
            # Simplified: just use proj output
            return self.linear_proj(x)
    
    class MockMLP(nn.Module):
        def __init__(self, hidden, ffn):
            super().__init__()
            # gate + up merged: hidden -> ffn*2
            self.linear_fc1 = MockLinear(hidden, ffn * 2)
            self.linear_fc2 = MockLinear(ffn, hidden)
        
        def forward(self, x):
            h = self.linear_fc1(x)
            # Split gate/up, apply activation
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
            # Pre-norm style
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


def test_memrift_e2e():
    """End-to-end test of MemRift with forward/backward pass."""
    print("=" * 70)
    print("MemRift End-to-End Training Test")
    print("=" * 70)
    
    device = torch.device('cuda')
    comp_dir = "/share/project/mengyc/code/memrift_v1/weight_comp/test"
    
    # Check compressed weights exist
    if not os.path.exists(os.path.join(comp_dir, "index.json")):
        print(f"ERROR: Compressed weights not found at {comp_dir}")
        return False
    
    print(f"\n[Config]")
    print(f"  Compressed weights: {comp_dir}")
    print(f"  Device: {device}")
    print(f"  CUDA: {torch.cuda.get_device_name()}")
    
    # Initial memory
    torch.cuda.empty_cache()
    gc.collect()
    mem_init = torch.cuda.memory_allocated(device)
    print(f"\n[Memory] Initial: {mem_init / 1024**2:.1f} MB")
    
    # Create model
    print(f"\n[Step 1] Creating mock Megatron model...")
    model = create_mock_megatron_model(
        num_layers=22,
        hidden_size=2048,
        ffn_hidden_size=5632,
        vocab_size=32000
    ).to(device)
    
    mem_after_model = torch.cuda.memory_allocated(device)
    print(f"  Model created: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"[Memory] After model: {mem_after_model / 1024**2:.1f} MB (+{(mem_after_model - mem_init) / 1024**2:.1f} MB)")
    
    # Import and setup MemRift
    print(f"\n[Step 2] Setting up MemRift loader...")
    try:
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
        
        # Load weights
        loader.load_weights()
        print(f"  Loaded {len(loader.all_cps)} compressed params")
        
        # Build mapping
        loader.build_param_mapping()
        print(f"  Mapped {sum(len(g) for g in loader.layer2groups.values())} weight groups")
        
        # Release original weights
        loader.release_original_weights()
        
        mem_after_release = torch.cuda.memory_allocated(device)
        print(f"[Memory] After release: {mem_after_release / 1024**2:.1f} MB ({(mem_after_release - mem_after_model) / 1024**2:+.1f} MB)")
        
        # Install hooks
        loader.install_hooks()
        print(f"  Hooks installed for {len(loader.layer_names)} layers")
        
        # Prefetch initial layers for TE compatibility
        loader.prefetch_initial_layers()
        
        mem_after_prefetch = torch.cuda.memory_allocated(device)
        print(f"[Memory] After prefetch: {mem_after_prefetch / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Forward pass test
    print(f"\n[Step 3] Testing forward pass...")
    try:
        batch_size = 2
        seq_len = 128
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        
        model.train()
        
        start_time = time.time()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids)
        torch.cuda.synchronize()
        fwd_time = time.time() - start_time
        
        print(f"  Input: {input_ids.shape}")
        print(f"  Output: {logits.shape}")
        print(f"  Forward time: {fwd_time*1000:.1f} ms")
        
        mem_after_fwd = torch.cuda.memory_allocated(device)
        print(f"[Memory] After forward: {mem_after_fwd / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"  ERROR in forward: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Backward pass test
    print(f"\n[Step 4] Testing backward pass...")
    try:
        # Simple loss
        target = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        print(f"  Loss: {loss.item():.4f}")
        
        start_time = time.time()
        loss.backward()
        torch.cuda.synchronize()
        bwd_time = time.time() - start_time
        
        print(f"  Backward time: {bwd_time*1000:.1f} ms")
        
        # Check some gradients exist (for non-MemRift params)
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        print(f"  Params with grad: {grad_count}")
        
        mem_after_bwd = torch.cuda.memory_allocated(device)
        print(f"[Memory] After backward: {mem_after_bwd / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"  ERROR in backward: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Memory stats
    print(f"\n[Step 5] Memory summary...")
    stats = loader.get_memory_stats()
    print(f"  sm_gpu (resident compressed): {stats['sm_gpu_mb']:.1f} MB")
    print(f"  exp_cpu (zstd compressed): {stats['exp_cpu_mb']:.1f} MB")
    print(f"  CUDA allocated: {stats['cuda_allocated_mb']:.1f} MB")
    
    # Cleanup
    print(f"\n[Step 6] Cleanup...")
    del model, loader, logits, loss
    gc.collect()
    torch.cuda.empty_cache()
    
    mem_final = torch.cuda.memory_allocated(device)
    print(f"[Memory] Final: {mem_final / 1024**2:.1f} MB")
    
    print(f"\n" + "=" * 70)
    print("END-TO-END TEST PASSED!")
    print("=" * 70)
    
    return True


def test_multiple_iterations():
    """Test multiple training iterations to check memory stability."""
    print("\n" + "=" * 70)
    print("MemRift Multi-Iteration Stability Test")
    print("=" * 70)
    
    device = torch.device('cuda')
    comp_dir = "/share/project/mengyc/code/memrift_v1/weight_comp/test"
    
    # Create model and setup MemRift
    model = create_mock_megatron_model().to(device)
    
    from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader
    
    loader = MegatronDynamicLoader(
        model=model,
        comp_dir=comp_dir,
        device=device,
        print_debug=False,
    )
    loader.load_weights()
    loader.build_param_mapping()
    loader.release_original_weights()
    loader.install_hooks()
    loader.prefetch_initial_layers()
    
    print(f"\nRunning 5 iterations...")
    mem_samples = []
    
    for i in range(5):
        # Forward
        input_ids = torch.randint(0, 32000, (2, 128), device=device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids)
        
        # Loss & backward
        target = torch.randint(0, 32000, (2, 128), device=device)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        loss.backward()
        
        mem = torch.cuda.memory_allocated(device)
        mem_samples.append(mem)
        print(f"  Iter {i+1}: loss={loss.item():.4f}, mem={mem/1024**2:.1f} MB")
        
        # Zero grad
        model.zero_grad(set_to_none=True)
        del logits, loss
    
    # Check memory stability
    mem_diff = max(mem_samples) - min(mem_samples)
    print(f"\nMemory variation: {mem_diff/1024**2:.1f} MB")
    
    if mem_diff < 500 * 1024 * 1024:  # < 500MB variation
        print("STABILITY TEST PASSED!")
        return True
    else:
        print("WARNING: Large memory variation detected")
        return False


if __name__ == "__main__":
    success = True
    
    success = test_memrift_e2e() and success
    
    if success:
        success = test_multiple_iterations() and success
    
    sys.exit(0 if success else 1)
