#!/usr/bin/env python
"""
Basic test for MemRift integration.

This script tests:
1. CUDA extension availability
2. Compressed weight loading
3. HF->Megatron mapping
4. Weight materialization and param binding
5. Release and memory tracking
"""
import os
import sys
import json
import torch
import torch.nn as nn

# Add FlagScale to path
sys.path.insert(0, '/share/project/mengyc/flagScale/FlagScale')

def test_cuda_extension():
    """Test CUDA extension availability."""
    print("=" * 60)
    print("Test 1: CUDA Extension")
    print("=" * 60)
    
    try:
        from flagscale.compress.float_split_stride_pin import float_split_stride_pin as fs_sp
        available = fs_sp.is_available()
        print(f"  CUDA extension available: {available}")
        
        if available:
            # Quick functional test
            t = torch.randn(16, 16, dtype=torch.bfloat16, device='cuda')
            stream = torch.cuda.current_stream()
            exp, sm = fs_sp.split(t, stream.cuda_stream)
            print(f"  Split test: input shape {t.shape}, exp shape {exp.shape}, sm shape {sm.shape}")
            
            # Merge back
            merged = fs_sp.merge(exp, sm, list(t.shape), list(t.stride()), 0, torch.bfloat16, stream.cuda_stream)
            stream.synchronize()
            
            diff = (t - merged).abs().max().item()
            print(f"  Merge test: max diff = {diff}")
            assert diff < 1e-5, f"Merge mismatch: {diff}"
            print("  PASS: Split/merge roundtrip OK")
        
        return available
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zstandard():
    """Test zstandard availability."""
    print("\n" + "=" * 60)
    print("Test 2: Zstandard")
    print("=" * 60)
    
    try:
        import zstandard as zstd
        print(f"  zstandard version: {zstd.__version__}")
        
        # Quick compress/decompress test
        data = b"Hello MemRift! " * 100
        cctx = zstd.ZstdCompressor(level=18)
        compressed = cctx.compress(data)
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(compressed)
        
        print(f"  Compress test: {len(data)} -> {len(compressed)} bytes")
        assert data == decompressed
        print("  PASS: Compress/decompress OK")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_compressed_weight_loading():
    """Test loading compressed weights."""
    print("\n" + "=" * 60)
    print("Test 3: Compressed Weight Loading")
    print("=" * 60)
    
    comp_dir = "/share/project/mengyc/code/memrift_v1/weight_comp/test"
    index_path = os.path.join(comp_dir, "index.json")
    
    if not os.path.exists(index_path):
        print(f"  SKIP: Test weights not found at {comp_dir}")
        return False
    
    try:
        with open(index_path) as f:
            index = json.load(f)
        
        print(f"  Loaded index with {len(index)} entries")
        
        # Count layers
        layers = set()
        for entry in index:
            name = entry["name"]
            if "layers." in name:
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i+1 < len(parts):
                        try:
                            layers.add(int(parts[i+1]))
                        except:
                            pass
        
        print(f"  Found {len(layers)} transformer layers")
        print(f"  Layer indices: {sorted(layers)[:5]}...{sorted(layers)[-3:]}")
        
        # Test loading one weight
        entry = index[1]  # First layer weight
        print(f"\n  Loading test weight: {entry['name']}")
        print(f"    Shape: {entry['shape']}, dtype: {entry['dtype']}")
        
        import struct
        file_path = os.path.join(comp_dir, entry["file"])
        with open(file_path, "rb") as f:
            numel = struct.unpack("<Q", f.read(8))[0]
            sm_size = numel * 1  # bfloat16 = 1 byte sm
            sm_bytes = f.read(sm_size)
            exp_bytes = f.read()
        
        print(f"    numel: {numel}, sm_bytes: {len(sm_bytes)}, exp_bytes: {len(exp_bytes)}")
        
        # Decompress exponent
        import zstandard as zstd
        import numpy as np
        
        dctx = zstd.ZstdDecompressor()
        exp_np = np.frombuffer(dctx.decompress(exp_bytes), dtype=np.uint8)
        print(f"    Decompressed exp: {len(exp_np)} bytes")
        
        # Load to GPU
        sm_gpu = torch.as_tensor(
            np.frombuffer(sm_bytes, dtype=np.uint8), 
            dtype=torch.uint8, 
            device='cuda'
        )
        exp_host = torch.as_tensor(exp_np, dtype=torch.uint8)
        exp_host_pinned = torch.empty_like(exp_host, pin_memory=True)
        exp_host_pinned.copy_(exp_host)
        
        print(f"    sm_gpu: {sm_gpu.shape}, exp_host: {exp_host_pinned.shape}")
        
        # Merge
        from flagscale.compress.float_split_stride_pin import float_split_stride_pin as fs_sp
        
        shape = entry["shape"]
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i+1] * shape[i+1]
        
        stream = torch.cuda.current_stream()
        merged = fs_sp.merge(
            exp_host_pinned, sm_gpu,
            shape, strides, 0,
            torch.bfloat16, stream.cuda_stream
        )
        stream.synchronize()
        
        print(f"    Merged tensor: shape={merged.shape}, dtype={merged.dtype}")
        print(f"    Sample values: {merged[0,:5].tolist()}")
        
        print("  PASS: Weight loading OK")
        return True
        
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_megatron_dynamic_loader():
    """Test MegatronDynamicLoader with a mock model."""
    print("\n" + "=" * 60)
    print("Test 4: MegatronDynamicLoader")
    print("=" * 60)
    
    comp_dir = "/share/project/mengyc/code/memrift_v1/weight_comp/test"
    
    if not os.path.exists(os.path.join(comp_dir, "index.json")):
        print(f"  SKIP: Test weights not found")
        return False
    
    try:
        from flagscale.compress.memrift.megatron_dynamic_loader import (
            MegatronDynamicLoader, CompressedParam, MergedWeightGroup
        )
        
        # Create a minimal mock model structure
        class MockLinear(nn.Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(out_f, in_f, dtype=torch.bfloat16, device='cuda'))
        
        class MockSelfAttention(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                # QKV merged: (q_dim + k_dim + v_dim, hidden)
                self.linear_qkv = MockLinear(hidden, hidden + 256 + 256)  # 2048 + 256 + 256 = 2560
                self.linear_proj = MockLinear(hidden, hidden)
        
        class MockMLP(nn.Module):
            def __init__(self, hidden, ffn):
                super().__init__()
                # gate + up merged
                self.linear_fc1 = MockLinear(hidden, ffn * 2)
                self.linear_fc2 = MockLinear(ffn, hidden)
        
        class MockLayer(nn.Module):
            def __init__(self, hidden, ffn):
                super().__init__()
                self.self_attention = MockSelfAttention(hidden)
                self.mlp = MockMLP(hidden, ffn)
        
        class MockDecoder(nn.Module):
            def __init__(self, num_layers, hidden, ffn):
                super().__init__()
                self.layers = nn.ModuleList([MockLayer(hidden, ffn) for _ in range(num_layers)])
        
        class MockModel(nn.Module):
            def __init__(self, num_layers=22, hidden=2048, ffn=5632):
                super().__init__()
                self.decoder = MockDecoder(num_layers, hidden, ffn)
        
        # Create mock model
        device = torch.device('cuda')
        model = MockModel().to(device)
        
        print(f"  Created mock model with {len(model.decoder.layers)} layers")
        
        # Get initial memory
        mem_before = torch.cuda.memory_allocated(device)
        print(f"  Memory before loader: {mem_before / 1024**2:.1f} MB")
        
        # Create loader
        loader = MegatronDynamicLoader(
            model=model,
            comp_dir=comp_dir,
            device=device,
            tp_rank=0,
            tp_size=1,
            prefetch_layers=1,
            print_debug=True,
        )
        
        # Load weights
        print("\n  Loading compressed weights...")
        loader.load_weights()
        
        print(f"\n  Loaded {len(loader.all_cps)} compressed params")
        print(f"  Merged groups: {sum(len(g) for g in loader.merged_groups.values())}")
        
        # Build param mapping
        print("\n  Building param mapping...")
        loader.build_param_mapping()
        
        print(f"  layer2groups: {len(loader.layer2groups)} layers")
        for layer_name, groups in list(loader.layer2groups.items())[:2]:
            print(f"    {layer_name}: {[g.megatron_target for g in groups]}")
        
        # Test materialization of one group
        if loader.layer2groups:
            first_layer = list(loader.layer2groups.keys())[0]
            groups = loader.layer2groups[first_layer]
            if groups:
                group = groups[0]
                print(f"\n  Testing materialization of {first_layer}/{group.megatron_target}")
                
                weight = loader._materialize_group(group, sync=True)
                print(f"    Materialized weight shape: {weight.shape}")
                
                # Test set_param
                loader._set_param(group, weight)
                if group.target_module:
                    actual = getattr(group.target_module, group.target_attr)
                    print(f"    Param after set: shape={actual.shape}")
                
                # Test clear_param
                loader._clear_param(group)
                if group.target_module:
                    actual = getattr(group.target_module, group.target_attr)
                    print(f"    Param after clear: shape={actual.shape}")
        
        # Memory stats
        stats = loader.get_memory_stats()
        print(f"\n  Memory stats:")
        print(f"    sm_gpu (resident): {stats['sm_gpu_mb']:.1f} MB")
        print(f"    exp_cpu (compressed): {stats['exp_cpu_mb']:.1f} MB")
        print(f"    cuda_allocated: {stats['cuda_allocated_mb']:.1f} MB")
        
        print("\n  PASS: MegatronDynamicLoader OK")
        return True
        
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("MemRift Basic Test Suite")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    results = {}
    
    results['cuda_ext'] = test_cuda_extension()
    results['zstandard'] = test_zstandard()
    
    if results['cuda_ext'] and results['zstandard']:
        results['weight_loading'] = test_compressed_weight_loading()
        results['dynamic_loader'] = test_megatron_dynamic_loader()
    else:
        results['weight_loading'] = False
        results['dynamic_loader'] = False
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
