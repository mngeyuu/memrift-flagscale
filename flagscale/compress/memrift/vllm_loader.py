"""
MemRift vLLM Model Loader
=========================
Extends DefaultModelLoader to enable layer-by-layer weight streaming:
  1. Load model normally (all weights → GPU via DefaultModelLoader)
  2. Compress each decoder layer's weights: GPU bf16 → CPU sm + zstd-exp
  3. Install forward pre/post hooks for on-demand decompression
  4. GPU peak ≈ 1 decoder layer (~400 MB) instead of full model (13.5 GB)

Usage:
    from vllm import LLM
    from flagscale.compress.memrift.vllm_loader import MemRiftModelLoader

    llm = LLM(
        model="/path/to/model",
        load_format=MemRiftModelLoader,
        enforce_eager=True,   # required: torch.compile interferes with hooks
        dtype="bfloat16",
    )

Env vars:
    MEMRIFT_ZSTD_LEVEL        zstd compression level (default: 3)
    MEMRIFT_DECODE_WORKERS    decompression thread pool size (default: 16)
    MEMRIFT_PREFETCH_LAYERS   layers to prefetch ahead (default: 1)
"""

from __future__ import annotations

import os
import time
import threading
import concurrent.futures as fut
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig, LoadConfig
from vllm.model_executor.model_loader.loader import DefaultModelLoader

try:
    import zstandard as zstd
    _ZSTD_AVAILABLE = True
except ImportError:
    _ZSTD_AVAILABLE = False

# Thread-local zstd contexts
_tls = threading.local()


def _get_cctx(level: int):
    if not hasattr(_tls, "cctx") or _tls.cctx_level != level:
        _tls.cctx = zstd.ZstdCompressor(level=level, write_checksum=False)
        _tls.cctx_level = level
    return _tls.cctx


def _get_dctx():
    if not hasattr(_tls, "dctx"):
        _tls.dctx = zstd.ZstdDecompressor()
    return _tls.dctx


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CompressedParam:
    """One compressed weight tensor belonging to a decoder layer."""
    param_ref: nn.Parameter    # the live Parameter object in the model
    sm_cpu: torch.Tensor       # pinned CPU  [numel], dtype=uint8, sign+mantissa
    exp_mv: bytes              # zstd-compressed exponent bytes
    orig_shape: tuple
    orig_dtype: torch.dtype
    numel: int
    # runtime scratch (filled by hooks)
    _sm_on_gpu: Optional[torch.Tensor] = field(default=None, repr=False)
    _bf16_gpu:  Optional[torch.Tensor] = field(default=None, repr=False)
    _future:    Any             = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Compression helpers (bf16 → sm_cpu + exp_zstd)
# ─────────────────────────────────────────────────────────────────────────────

def _compress_tensor_bf16(t: torch.Tensor, zstd_level: int,
                           fs_sp) -> Tuple[torch.Tensor, bytes, int]:
    """
    Split bf16 tensor into sign+mantissa (pinned CPU) and zstd-compressed exponent.
    Returns (sm_cpu, exp_bytes, numel).
    """
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    t = t.contiguous().cuda()

    d2h = torch.cuda.Stream()
    d2h.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(d2h):
        cpu_exp, sm_bits = fs_sp.split(t, d2h.cuda_stream)
        evt = d2h.record_event()
    evt.synchronize()

    numel = cpu_exp.numel()
    arr = cpu_exp.numpy()
    exp_bytes = _get_cctx(zstd_level).compress(arr)

    # sm_bits is already pinned CPU from the CUDA extension
    return sm_bits, exp_bytes, numel


def _decompress_param(cp: _CompressedParam, h2d: torch.cuda.Stream,
                      fs_sp, semaphore: threading.Semaphore) -> torch.Tensor:
    """Synchronously decompress one parameter to GPU (no thread pool)."""
    semaphore.acquire()
    try:
        cpu_exp = torch.empty(cp.numel, dtype=torch.uint8, pin_memory=True)
        dctx = _get_dctx()
        with dctx.stream_reader(memoryview(cp.exp_mv)) as reader:
            nread = reader.readinto(memoryview(cpu_exp.numpy()))
            assert nread == cp.numel, "decompress size mismatch"

        strides = _c_strides(cp.orig_shape)
        with torch.cuda.stream(h2d):
            bf16 = fs_sp.merge(
                cpu_exp, cp._sm_on_gpu,
                list(cp.orig_shape), list(strides), 0,
                cp.orig_dtype, h2d.cuda_stream,
            )
            cp._sm_on_gpu.record_stream(h2d)
        evt = h2d.record_event()
        evt.synchronize()
        del cpu_exp
        return bf16
    finally:
        semaphore.release()


def _c_strides(shape):
    s = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 2, -1, -1):
        running *= shape[i + 1]
        s[i] = running
    return tuple(s)


# ─────────────────────────────────────────────────────────────────────────────
# Layer state
# ─────────────────────────────────────────────────────────────────────────────

class _LayerState:
    """Holds compressed params for one decoder layer; manages GPU lifecycle."""

    def __init__(self, idx: int, params: List[_CompressedParam],
                 device, h2d: torch.cuda.Stream, fs_sp,
                 semaphore: threading.Semaphore,
                 decode_pool: fut.ThreadPoolExecutor,
                 prefetch: bool = True):
        self.idx = idx
        self.params = params
        self.device = device
        self.h2d = h2d
        self.fs_sp = fs_sp
        self.semaphore = semaphore
        self.decode_pool = decode_pool
        self.prefetch = prefetch
        self._prefetch_futures: List[fut.Future] = []

    # ── prefetch (async) ──────────────────────────────────────────────────

    def start_prefetch(self):
        """Submit decompression jobs to thread pool (non-blocking)."""
        if self._prefetch_futures:
            return  # already in flight
        for cp in self.params:
            cp._sm_on_gpu = cp.sm_cpu.to(self.device, non_blocking=False)
        for cp in self.params:
            f = self.decode_pool.submit(self._decomp_one, cp)
            self._prefetch_futures.append(f)

    def _decomp_one(self, cp: _CompressedParam) -> torch.Tensor:
        self.semaphore.acquire()
        try:
            cpu_exp = torch.empty(cp.numel, dtype=torch.uint8, pin_memory=True)
            dctx = _get_dctx()
            with dctx.stream_reader(memoryview(cp.exp_mv)) as reader:
                nread = reader.readinto(memoryview(cpu_exp.numpy()))
                assert nread == cp.numel, "decompress size mismatch"

            strides = _c_strides(cp.orig_shape)
            with torch.cuda.stream(self.h2d):
                bf16 = self.fs_sp.merge(
                    cpu_exp, cp._sm_on_gpu,
                    list(cp.orig_shape), list(strides), 0,
                    cp.orig_dtype, self.h2d.cuda_stream,
                )
                cp._sm_on_gpu.record_stream(self.h2d)
            evt = self.h2d.record_event()

            # Release semaphore before GPU sync so other workers can proceed
            self.semaphore.release()
            evt.synchronize()
            del cpu_exp
            return bf16
        except Exception:
            self.semaphore.release()
            raise

    def wait_and_restore(self):
        """Wait for prefetch futures and restore param.data to GPU tensors."""
        if self._prefetch_futures:
            for cp, f in zip(self.params, self._prefetch_futures):
                cp._bf16_gpu = f.result()
            self._prefetch_futures.clear()
        else:
            # Cold (no prefetch): decompress synchronously
            for cp in self.params:
                cp._sm_on_gpu = cp.sm_cpu.to(self.device, non_blocking=False)
                cp._bf16_gpu = _decompress_param(
                    cp, self.h2d, self.fs_sp, self.semaphore)

        for cp in self.params:
            cp.param_ref.data = cp._bf16_gpu

    def release(self):
        """Free GPU tensors and set param.data to empty CPU placeholder."""
        for cp in self.params:
            cp._bf16_gpu = None
            cp._sm_on_gpu = None
            # Use a 0-element CPU tensor so the Parameter still exists
            cp.param_ref.data = torch.empty(
                0, dtype=cp.orig_dtype, device="cpu")
        self._prefetch_futures.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Manager: compresses model + installs hooks
# ─────────────────────────────────────────────────────────────────────────────

class VLLMMemRiftManager:
    """
    Compresses vLLM decoder layers and installs forward hooks for
    layer-by-layer weight streaming.
    """

    def __init__(
        self,
        zstd_level: int = 3,
        decode_workers: int = 16,
        prefetch_layers: int = 1,
        concurrency_limit: int = 4,
    ):
        if not _ZSTD_AVAILABLE:
            raise RuntimeError("zstandard not installed: pip install zstandard")

        try:
            from flagscale.compress.float_split_stride_pin import (
                float_split_stride_pin as _fs_sp,
            )
            if not _fs_sp.is_available():
                raise RuntimeError("float_split_stride_pin CUDA ext not available")
            self.fs_sp = _fs_sp
        except ImportError as e:
            raise RuntimeError(f"MemRift CUDA extension not found: {e}")

        self.zstd_level = zstd_level
        self.decode_workers = decode_workers
        self.prefetch_layers = prefetch_layers
        self.concurrency_limit = concurrency_limit

    def compress_and_hook(self, model: nn.Module, device) -> None:
        """
        Main entry: find decoder layers, compress weights, install hooks.
        Modifies `model` in-place (no return value).
        """
        layers = self._find_decoder_layers(model)
        if not layers:
            raise RuntimeError(
                "[MemRift-vLLM] Could not find decoder layers. "
                "Supported architectures: LLaMA, Mistral, Qwen2, Falcon."
            )

        print(f"[MemRift-vLLM] Compressing {len(layers)} decoder layers "
              f"(zstd={self.zstd_level}, workers={self.decode_workers}, "
              f"prefetch={self.prefetch_layers}) …")

        h2d = torch.cuda.Stream()
        semaphore = threading.Semaphore(self.concurrency_limit)
        decode_pool = fut.ThreadPoolExecutor(max_workers=self.decode_workers)

        t0 = time.perf_counter()
        layer_states: List[_LayerState] = []
        for i, layer in enumerate(layers):
            params = self._compress_layer(layer, i)
            state = _LayerState(
                idx=i, params=params, device=device,
                h2d=h2d, fs_sp=self.fs_sp,
                semaphore=semaphore, decode_pool=decode_pool,
                prefetch=self.prefetch_layers > 0,
            )
            layer_states.append(state)

        elapsed = time.perf_counter() - t0
        print(f"[MemRift-vLLM] Compressed {len(layers)} layers in {elapsed:.1f}s")

        self._install_hooks(layers, layer_states)

        # Pre-fetch the first layer so layer 0 is ready at the first forward
        layer_states[0].start_prefetch()

        # Store on model for inspection / lifecycle management
        model._memrift_layer_states = layer_states
        model._memrift_decode_pool = decode_pool

    # ── internal helpers ──────────────────────────────────────────────────

    def _find_decoder_layers(self, model: nn.Module):
        """
        Returns the nn.ModuleList of decoder layers.
        Tries common attribute paths used by LLaMA, Mistral, Qwen2, Falcon, etc.
        """
        candidates = [
            # vLLM model structure (ForCausalLM wraps inner model)
            lambda m: m.model.layers,
            lambda m: m.model.model.layers,
            lambda m: m.transformer.h,
            lambda m: m.model.transformer.h,
            # Fallback: named modules scan
        ]
        for fn in candidates:
            try:
                layers = fn(model)
                if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                    return list(layers)
            except AttributeError:
                continue
        # Named-modules fallback
        for name, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 8:
                # Heuristic: long ModuleList = decoder layers
                first = next(iter(mod))
                if any(hasattr(first, a) for a in
                       ("self_attn", "attention", "attn", "self_attention")):
                    return list(mod)
        return []

    def _compress_layer(self, layer: nn.Module,
                        idx: int) -> List[_CompressedParam]:
        """Compress all bf16/fp16 weights in one decoder layer."""
        compressed = []
        for pname, param in layer.named_parameters(recurse=True):
            if param.dtype not in (torch.bfloat16, torch.float16):
                continue
            if not param.is_cuda:
                continue
            if param.numel() < 512:   # skip tiny norm weights etc.
                continue

            sm_cpu, exp_mv, numel = _compress_tensor_bf16(
                param.data, self.zstd_level, self.fs_sp)

            cp = _CompressedParam(
                param_ref=param,
                sm_cpu=sm_cpu,
                exp_mv=exp_mv,
                orig_shape=tuple(param.shape),
                orig_dtype=param.dtype,
                numel=numel,
            )
            compressed.append(cp)
            # Free GPU memory immediately after compression
            param.data = torch.empty(0, dtype=param.dtype, device="cpu")

        return compressed

    def _install_hooks(
        self,
        layers: List[nn.Module],
        states: List[_LayerState],
    ) -> None:
        """Register forward pre/post hooks on each decoder layer."""
        n = len(layers)
        prefetch_span = self.prefetch_layers

        for i, (layer, cur_state) in enumerate(zip(layers, states)):
            # Gather next-layer states for prefetch
            nxt_states = [states[j] for j in
                          range(i + 1, min(n, i + 1 + prefetch_span))]

            def _pre(module, args,
                     _cur=cur_state, _nxt=nxt_states):
                # Ensure current layer weights are on GPU
                _cur.wait_and_restore()
                # Kick off prefetch for upcoming layers
                for ns in _nxt:
                    ns.start_prefetch()

            def _post(module, args, output,
                      _cur=cur_state):
                _cur.release()

            layer.register_forward_pre_hook(_pre)
            layer.register_forward_hook(_post)

        print(f"[MemRift-vLLM] Installed hooks on {n} layers "
              f"(prefetch_span={prefetch_span})")


# ─────────────────────────────────────────────────────────────────────────────
# MemRiftModelLoader — the vLLM load_format plugin
# ─────────────────────────────────────────────────────────────────────────────

class MemRiftModelLoader(DefaultModelLoader):
    """
    vLLM ModelLoader that enables MemRift layer-by-layer weight streaming.

    Pass as load_format to vLLM's LLM():
        llm = LLM(model=..., load_format=MemRiftModelLoader, enforce_eager=True)

    Configuration via env vars:
        MEMRIFT_ZSTD_LEVEL       (default: 3)
        MEMRIFT_DECODE_WORKERS   (default: 16)
        MEMRIFT_PREFETCH_LAYERS  (default: 1)
    """

    def __init__(self, load_config: LoadConfig):
        # Temporarily override load_format so super().__init__ doesn't reject
        # a non-string load_format value (some vLLM versions check this).
        from vllm.config import LoadFormat
        orig = load_config.load_format
        load_config.load_format = LoadFormat.AUTO
        super().__init__(load_config)
        load_config.load_format = orig

        self._zstd_level     = int(os.getenv("MEMRIFT_ZSTD_LEVEL", "3"))
        self._decode_workers = int(os.getenv("MEMRIFT_DECODE_WORKERS", "16"))
        self._prefetch       = int(os.getenv("MEMRIFT_PREFETCH_LAYERS", "1"))

    def load_model(self, vllm_config: VllmConfig) -> nn.Module:
        """Load model weights, then compress decoder layers to CPU."""
        t0 = time.perf_counter()

        # Step 1: Normal load (all weights → GPU)
        model = super().load_model(vllm_config)

        load_time = time.perf_counter() - t0
        peak_before = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[MemRift-vLLM] Normal load done in {load_time:.1f}s, "
              f"GPU peak = {peak_before:.2f} GB")

        # Step 2: Compress + install hooks
        device = torch.device(vllm_config.device_config.device)
        torch.cuda.reset_peak_memory_stats()

        manager = VLLMMemRiftManager(
            zstd_level=self._zstd_level,
            decode_workers=self._decode_workers,
            prefetch_layers=self._prefetch,
        )
        manager.compress_and_hook(model, device)

        peak_after = torch.cuda.max_memory_allocated() / 1024**3
        total = time.perf_counter() - t0
        print(f"[MemRift-vLLM] Ready. GPU peak after offload = {peak_after:.2f} GB "
              f"(total setup {total:.1f}s)")

        return model
