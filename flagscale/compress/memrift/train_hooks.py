"""
Training injection hooks for MemRift.

Main entry point for integrating MemRift into FlagScale training.

Usage:
    from flagscale.compress.memrift.train_hooks import inject_memrift_if_configured
    
    # In get_model(), after PEFT but before DDP:
    inject_memrift_if_configured(model, args)

Activation compression is injected per-layer (like memrift_demo): each decoder layer
is wrapped in DecoderLayerWrapper; saved_tensors_hooks run only inside that layer's
forward; tokens/futures are cleared in that layer's backward_hook.
"""
import os
from contextlib import nullcontext
from typing import Optional, List, Any, Set

import torch
import torch.nn as nn


# Module-level list of all MemRift loaders created during inject. Used by
# external probes (e.g. train.py memory probes) to introspect loader state.
_LOADERS: list = []


def _check_cuda_extension():
    """Check if CUDA extension is available and raise clear error if not."""
    try:
        from flagscale.compress.float_split_stride_pin import float_split_stride_pin as fs_sp
        if not fs_sp.is_available():
            raise RuntimeError(
                "CUDA extension float_split_stride_pin is not available.\n"
                "MemRift requires this extension for weight decompression.\n"
                "Please build it:\n"
                "  cd flagscale/compress/float_split_stride_pin\n"
                "  pip install -e .\n"
                "If build fails, ensure CUDA toolkit is installed and matches PyTorch CUDA version."
            )
        return True
    except ImportError as e:
        raise RuntimeError(
            f"CUDA extension float_split_stride_pin not found: {e}\n"
            "MemRift requires this extension for weight decompression.\n"
            "Please build it:\n"
            "  cd flagscale/compress/float_split_stride_pin\n"
            "  pip install -e ."
        )


def _check_zstandard():
    """Check if zstandard is available."""
    try:
        import zstandard
        return True
    except ImportError:
        raise RuntimeError(
            "zstandard library not found.\n"
            "MemRift requires zstandard for weight decompression.\n"
            "Please install it: pip install zstandard"
        )


def inject_memrift_if_configured(
    model: nn.Module | List[nn.Module],
    args: Any,
) -> None:
    """
    Inject MemRift if configured in args.
    
    This is the main entry point for MemRift integration. Call this in
    get_model() after PEFT/LoRA injection but before DDP wrapping.
    
    Args:
        model: Model or list of model chunks (for virtual pipeline)
        args: Training arguments (from get_args())
    
    Configuration (via args):
        memrift_enable: bool - Master enable switch
        memrift_weight_enable: bool - Enable weight compression
        memrift_activation_enable: bool - Enable activation compression
        memrift_compressed_weight_dir: str - Path to compressed weights
        memrift_zstd_level: int - Zstd compression level
        memrift_prefetch_layers: int - Number of layers to prefetch
        memrift_weight_async: bool - Enable async weight decompression
        memrift_act_async: bool - Enable async activation compression
        memrift_decode_pool_workers: int - Decode thread pool size
        memrift_compress_pool_workers: int - Compress thread pool size
        memrift_print_debug: bool - Print debug messages
    
    Note:
        MemRift in this package is intentionally single-GPU training only.
    """
    # Check if MemRift is enabled
    memrift_enable = getattr(args, "memrift_enable", False)
    if not memrift_enable:
        return

    # Patch TE autograd Functions to release ctx-pinned weights after backward.
    # Without this, ~15 GB of decoder weights stay pinned through entire backward.
    _rank0 = (getattr(args, "rank", 0) == 0)
    try:
        from flagscale.compress.memrift.te_ctx_patch import patch_te_ctx_release
        n = patch_te_ctx_release()
        if _rank0:
            print(f"[MemRift] TE ctx-weight release patch applied to {n} TE Function classes")
    except Exception as e:
        if _rank0:
            print(f"[MemRift] Warning: failed to patch TE ctx release: {e}")
    
    # Get configuration
    weight_enable = getattr(args, "memrift_weight_enable", False)
    activation_enable = getattr(args, "memrift_activation_enable", False)
    compressed_weight_dir = getattr(args, "memrift_compressed_weight_dir", None)
    zstd_level = getattr(args, "memrift_zstd_level", 6)
    act_zstd_level = getattr(args, "memrift_act_zstd_level", 3)
    prefetch_layers = getattr(args, "memrift_prefetch_layers", 4) # 实际上未生效，在代码里是硬编码的
    weight_async = getattr(args, "memrift_weight_async", False)
    act_async = getattr(args, "memrift_act_async", True)
    decode_workers = getattr(args, "memrift_decode_pool_workers", 16)
    compress_workers = getattr(args, "memrift_compress_pool_workers", 8)
    print_debug = getattr(args, "memrift_print_debug", False)
    
    # Check dependencies first
    _check_zstandard()
    if weight_enable:
        _check_cuda_extension()
    
    # Validate configuration
    if weight_enable and not compressed_weight_dir:
        raise ValueError(
            "memrift_weight_enable=True requires memrift_compressed_weight_dir to be set"
        )
    
    if weight_enable and compressed_weight_dir:
        if not os.path.isdir(compressed_weight_dir):
            raise ValueError(
                f"memrift_compressed_weight_dir does not exist: {compressed_weight_dir}"
            )
        index_path = os.path.join(compressed_weight_dir, "index.json")
        if not os.path.isfile(index_path):
            raise ValueError(
                f"MemRift single-GPU training expects index.json at: {index_path}"
            )

    tp_size = getattr(args, "tensor_model_parallel_size", 1)
    pp_size = getattr(args, "pipeline_model_parallel_size", 1)
    if tp_size != 1 or pp_size != 1:
        raise ValueError(
            "MemRift only supports single-GPU training in this build "
            "(tensor_model_parallel_size=1 and pipeline_model_parallel_size=1)."
        )
    
    # Convert model to list if needed
    if not isinstance(model, list):
        model_chunks = [model]
    else:
        model_chunks = model
    
    rank = getattr(args, "rank", 0)
    if print_debug:
        print(f"[MemRift][rank{rank}] Initializing with:")
        print(f"  weight_enable={weight_enable}")
        print(f"  activation_enable={activation_enable}")
        print(f"  compressed_weight_dir={compressed_weight_dir}")
        print("  parallelism=single_gpu")
        print(f"  prefetch_layers={prefetch_layers}")
        print(f"  weight_async={weight_async}")
        print(f"  num_attention_heads={getattr(args, 'num_attention_heads', None)}")
        print(f"  num_query_groups={getattr(args, 'num_query_groups', None)}")
        print(f"  group_query_attention={getattr(args, 'group_query_attention', False)}")
    
    # Create async compressor if needed
    async_compressor = None
    if weight_async or act_async:
        try:
            from flagscale.compress.memrift.async_compressor import AsyncCompressor
            async_compressor = AsyncCompressor(
                compress_workers=compress_workers,
                decode_workers=decode_workers,
                concurrency_limit=4, # 完全硬编码这个参数了
                # This compressor's level governs ONLY activation compression
                # (weight prefetch is decompress-only). Use the activation level.
                zstd_level=act_zstd_level,
                enable_async=True,
            )
            if print_debug and rank == 0:
                print("[MemRift] AsyncCompressor created")
        except Exception as e:
            if rank == 0:
                print(f"[MemRift] Warning: Failed to create AsyncCompressor: {e}")
            async_compressor = None
    
    # Inject weight compression
    if weight_enable:
        _inject_weight_compression(
            model_chunks=model_chunks,
            compressed_weight_dir=compressed_weight_dir,
            prefetch_layers=prefetch_layers,
            async_compressor=async_compressor,
            print_debug=print_debug,
            args=args,
            rank=rank,
        )
    
    # Inject activation compression (or weight placeholder hooks if only weight enabled).
    # When weight_enable is True we MUST install DecoderLayerWrapper even without
    # activation compression, so saved_tensors_hooks can intercept autograd-saved
    # weights and replace them with WeightPlaceholder; otherwise materialized bf16
    # weights are pinned in the autograd graph and ~15 GB of decoder weights leak.
    if activation_enable or weight_enable:
        _inject_activation_compression(
            model_chunks=model_chunks,
            async_compressor=async_compressor,
            act_async=act_async,
            print_debug=print_debug,
            rank=rank,
            compress_activations=activation_enable,
            act_zstd_level=act_zstd_level,
        )
    
    # Memory profiler (optional, for activation memory breakdown)
    profile_memory = getattr(args, "memrift_profile_memory", False)
    if profile_memory:
        try:
            from flagscale.compress.memrift.memory_profiler import install_memory_profiler
            num_layers = getattr(args, "num_layers", 22)
            profile_iters = int(os.environ.get("MEMRIFT_PROFILE_ITERS", "3"))
            install_memory_profiler(
                model_chunks=model_chunks,
                num_layers=num_layers,
                profile_iters=profile_iters,
                rank=rank,
            )
            if rank == 0:
                print(f"[MemRift] Memory profiler enabled, will report after {profile_iters} iterations")
        except Exception as e:
            if rank == 0:
                print(f"[MemRift] Memory profiler failed to install: {e}")

    if print_debug and rank == 0:
        print("[MemRift] Injection complete")


def _inject_weight_compression(
    model_chunks: List[nn.Module],
    compressed_weight_dir: str,
    prefetch_layers: int,
    async_compressor: Optional[Any],
    print_debug: bool,
    args: Any,
    rank: int = 0,
) -> None:
    """
    Inject weight compression into model chunks.
    
    Steps for each chunk:
    1. Load compressed weights from disk
    2. Build cp -> (target_module, target_attr) mapping
    3. Release original weights to free GPU memory
    4. Install forward/backward hooks
    5. Pre-materialize first K layers for TE compatibility
    """
    try:
        from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader
    except ImportError as e:
        if rank == 0:
            print(f"[MemRift] Failed to import MegatronDynamicLoader: {e}")
        return
    
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    allowed_targets = None
    peft_type = getattr(args, "peft_type", None)
    lora_targets = set(getattr(args, "lora_target_modules", []) or [])
    if peft_type == "lora" and lora_targets:
        mapping = {
            "linear_qkv": "self_attention.linear_qkv",
            "linear_proj": "self_attention.linear_proj",
            "linear_fc1": "mlp.linear_fc1",
            "linear_fc2": "mlp.linear_fc2",
        }
        # 保留lora映射megatron的模块
        allowed_targets = {mg for k, mg in mapping.items() if k in lora_targets}
        if print_debug and rank == 0:
            print(f"[MemRift] LoRA target filter enabled: {sorted(allowed_targets)}")

    for chunk_idx, chunk in enumerate(model_chunks):
        if print_debug and rank == 0:
            print(f"[MemRift] Processing chunk {chunk_idx}")
        
        # Unwrap Float16Module if needed
        unwrapped = chunk
        if hasattr(chunk, "module"):
            unwrapped = chunk.module
        
        try:
            # Create loader
            loader = MegatronDynamicLoader(
                model=unwrapped,
                comp_dir=compressed_weight_dir,
                device=device,
                prefetch_layers=prefetch_layers,
                print_debug=print_debug,
                allowed_targets=allowed_targets,
                num_attention_heads=getattr(args, "num_attention_heads", None),
                num_query_groups=getattr(args, "num_query_groups", None),
                hidden_size=getattr(args, "hidden_size", None),
                kv_channels=getattr(args, "kv_channels", None),
                group_query_attention=getattr(args, "group_query_attention", False),
            )
            
            # Step 1: Load compressed weights
            loader.load_weights()
            
            # Step 2: Build param mapping (cp -> target module/attr)
            loader.build_param_mapping()

            # Step 3: Release original weights to free GPU memory
            loader.release_original_weights()
            
            # Step 4: Install forward/backward hooks
            loader.install_hooks(async_compressor=async_compressor)
            
            # Step 5: Pre-materialize first K layers for TE compatibility
            loader.prefetch_initial_layers()

            _LOADERS.append(loader)

            # Log memory stats
            if print_debug and rank == 0:
                stats = loader.get_memory_stats()
                print(f"[MemRift] Chunk {chunk_idx} memory stats:")
                print(f"  sm_gpu (resident): {stats['sm_gpu_mb']:.1f} MB")
                print(f"  exp_cpu (compressed): {stats['exp_cpu_mb']:.1f} MB")
                print(f"  cuda_allocated: {stats['cuda_allocated_mb']:.1f} MB")
                print(f"  layers: {stats['num_layers']}, weight_groups: {stats['num_weight_groups']}")
                
        except Exception as e:
            if rank == 0:
                print(f"[MemRift] Error processing chunk {chunk_idx}: {e}")
                import traceback
                traceback.print_exc()


def _get_decoder_layers(module: nn.Module):
    """Return the decoder ModuleList (same pattern as megatron_dynamic_loader)."""
    for _name, m in module.named_modules():
        if hasattr(m, "layers") and isinstance(m.layers, nn.ModuleList):
            return m.layers
    if hasattr(module, "decoder") and hasattr(module.decoder, "layers"):
        return module.decoder.layers
    if hasattr(module, "language_model") and hasattr(module.language_model, "decoder"):
        return module.language_model.decoder.layers
    return None


def _inject_activation_compression(
    model_chunks: List[nn.Module],
    async_compressor: Optional[Any],
    act_async: bool,
    print_debug: bool,
    rank: int = 0,
    compress_activations: bool = True,
    act_zstd_level: int = 3,
) -> None:
    """
    Enable activation compression per-layer (aligned with memrift_demo).

    Each decoder layer is wrapped in DecoderLayerWrapper: saved_tensors_hooks run
    only inside that layer's forward; tokens/futures are stored on the wrapper and
    cleared in backward_hook. backward_pre_hook decompresses before the layer's backward.
    The training loop still uses _memrift_activation_context; we set it to a no-op
    so no global saved_tensors_hooks are applied.
    """
    try:
        from flagscale.compress.memrift.activation_compression import (
            DecoderLayerWrapper,
        )
    except ImportError as e:
        if rank == 0:
            print(f"[MemRift] Activation compression not available: {e}")
        return

    zstd_level = act_zstd_level
    compressor = async_compressor
    if compressor is None:
        try:
            from flagscale.compress.memrift.async_compressor import AsyncCompressor
            compressor = AsyncCompressor(
                compress_workers=2,
                decode_workers=2,
                concurrency_limit=4,
                zstd_level=zstd_level,
                enable_async=False,
            )
        except Exception as e:
            if rank == 0:
                print(f"[MemRift] Failed to create activation compressor: {e}")
            return

    skip_storage_ptrs: Set[int] = set()
    for chunk in model_chunks:
        unwrapped = chunk.module if hasattr(chunk, "module") else chunk
        for _name, p in unwrapped.named_parameters():
            if p.requires_grad:
                try:
                    skip_storage_ptrs.add(p.untyped_storage().data_ptr())
                except Exception:
                    pass
    if print_debug and rank == 0 and skip_storage_ptrs:
        print(f"[MemRift] LoRA: skip compressing {len(skip_storage_ptrs)} adapter storage(s) in activation hooks")

    wrapped_count = 0
    for chunk in model_chunks:
        unwrapped = chunk.module if hasattr(chunk, "module") else chunk
        decoder_layers = _get_decoder_layers(unwrapped)
        if decoder_layers is None:
            if rank == 0:
                print("[MemRift] Activation: no decoder.layers found in chunk, skipping")
            continue
        empty_cache_interval = int(os.environ.get("MEMRIFT_ACT_EMPTY_INTERVAL", "10"))
        for i in range(len(decoder_layers)):
            layer = decoder_layers[i]
            if isinstance(layer, DecoderLayerWrapper):
                continue
            wrapper = DecoderLayerWrapper(
                layer,
                compressor=compressor,
                use_async=act_async,
                release_after_unpack=True,
                skip_storage_ptrs=skip_storage_ptrs if skip_storage_ptrs else None,
                do_empty=(i % empty_cache_interval == 1),
                compress_activations=compress_activations,
            )
            decoder_layers[i] = wrapper
            wrapped_count += 1

    # No-op context: compression is per-layer now; train_step still calls
    # _get_memrift_activation_context(model) and must get a valid context.
    def _noop_context():
        return nullcontext()

    for chunk in model_chunks:
        chunk._memrift_activation_context = _noop_context

    if print_debug and rank == 0:
        print(f"[MemRift] Activation compression enabled (per-layer, {wrapped_count} layers wrapped)")


def get_memrift_status(args: Any) -> dict:
    """
    Get MemRift configuration status for logging.
    
    Args:
        args: Training arguments
        
    Returns:
        Dictionary with MemRift configuration
    """
    return {
        "memrift_enable": getattr(args, "memrift_enable", False),
        "memrift_weight_enable": getattr(args, "memrift_weight_enable", False),
        "memrift_activation_enable": getattr(args, "memrift_activation_enable", False),
        "memrift_compressed_weight_dir": getattr(args, "memrift_compressed_weight_dir", None),
        "memrift_zstd_level": getattr(args, "memrift_zstd_level", 6),
        "memrift_act_zstd_level": getattr(args, "memrift_act_zstd_level", 3),
        "memrift_prefetch_layers": getattr(args, "memrift_prefetch_layers", 4),
        "memrift_weight_async": getattr(args, "memrift_weight_async", False),
        "memrift_act_async": getattr(args, "memrift_act_async", True),
    }
