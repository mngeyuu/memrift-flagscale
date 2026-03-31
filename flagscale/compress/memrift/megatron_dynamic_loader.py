"""
Megatron Dynamic Weight Loader for MemRift.

Handles:
- HF to Megatron weight name mapping (TP=1 only in v1)
- Merged weight assembly (qkv -> linear_qkv, gate+up -> linear_fc1)
- cp -> (target_module, target_attr) binding
- Forward/backward hooks for dynamic load/release
- Original weight release to save GPU memory
- TE compatibility (pre-decompress first K layers)

Note: This version only supports TP=1 (tensor_model_parallel_size=1).
"""
import json
import os
import struct
import threading
import concurrent.futures as fut
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from contextlib import nullcontext

try:
    _memrift_nvtx_range = torch.cuda.nvtx.range
except AttributeError:

    def _memrift_nvtx_range(_name: str):
        return nullcontext()

try:
    from flagscale.compress.memrift import time_profiler as layer_time_profiler
except ImportError:
    layer_time_profiler = None

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

try:
    import nvidia.nvcomp as _nvcomp_lib
    NVCOMP_AVAILABLE = True
except ImportError:
    _nvcomp_lib = None
    NVCOMP_AVAILABLE = False

# Magic bytes to detect GPU nvCOMP-compressed exponent bytes.
# LZ4: prepare_weight.py --compression nvcomp_lz4 (may be re-encoded to ANS at load).
# ANS: prepare_weight.py --compression nvcomp_ans (preferred; smaller, no recompress).
NVCOMP_LZ4_MAGIC = b"NVL4"
NVCOMP_ANS_MAGIC = b"NVAN"

# Thread-local decompressor
_tls = threading.local()
_PTR2GROUP: Dict[int, "MergedWeightGroup"] = {}
# Prefer stream/event-based dependency by default to maximize overlap.
# Set MEMRIFT_DEEP_ASYNC=0 to fall back to host-side synchronize behavior.
_DEEP_ASYNC = os.environ.get("MEMRIFT_DEEP_ASYNC", "1") == "1"


def _trace(msg: str):
    if os.environ.get("MEMRIFT_TRACE", "0") != "1":
        return
    print(f"[MemRiftTrace][weights] {msg}", flush=True)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _get_dctx():
    if not hasattr(_tls, "dctx"):
        _tls.dctx = zstd.ZstdDecompressor()
    return _tls.dctx


def _c_contiguous_strides(shape):
    """Calculate C-contiguous strides for a shape."""
    strides = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 2, -1, -1):
        running *= shape[i + 1]
        strides[i] = running
    return tuple(strides)


class CompressedParam(nn.Parameter):
    """
    A Parameter subclass that holds compressed weight data.
    
    The actual bf16/fp32 data is materialized on-demand during forward/backward,
    then released after use to save GPU memory.
    
    Key attributes for param binding:
        target_module: The nn.Module whose weight this cp represents
        target_attr: The attribute name ('weight' or 'bias')
        hf_name: Original HF parameter name
        megatron_target: Megatron target (e.g., 'self_attention.linear_qkv')
        merge_key: For merged weights, the component key (e.g., 'q', 'k', 'v')
    """
    
    def __new__(cls, orig_shape, sm_gpu, exp_mv, dtype):
        # Create a dummy empty tensor
        dummy = torch.empty(0, dtype=dtype, device=sm_gpu.device)
        return super().__new__(cls, dummy, requires_grad=False)
    
    def __init__(self, orig_shape, sm_gpu, exp_mv, dtype):
        super().__init__()
        self.orig_shape = tuple(orig_shape)
        self.sm_gpu = sm_gpu
        self.exp_mv = exp_mv
        self._dtype = dtype
        self._bf16 = None
        self._ready_event = threading.Event()
        self._CtoD_evt = None
        self._exp_host = None

        # For nvcomp_lz4: persistent pinned buffer holding the compressed data.
        # Allocated once on the main thread at load time (pinned alloc from the
        # main thread is ~0.02ms vs ~10ms from thread-pool threads whose per-thread
        # pinned-memory caches are cold).  Reused every iteration; safe because
        # all H2D ops on h2d_stream are serialized and the GPU reads the buffer
        # before the next submission.
        self.comp_pinned: Optional[torch.Tensor] = None
        
        # Param binding (set by loader)
        self.target_module: Optional[nn.Module] = None
        self.target_attr: str = "weight"
        self.hf_name: str = ""
        self.megatron_target: str = ""
        self.merge_key: Optional[str] = None  # 'q'/'k'/'v' or 'gate'/'up'
        self.layer_idx: int = -1
        
        # Async prefetch
        self._prefetch_future = None
    
    def materialize(self, sync: bool = True):
        """Decompress and matieralize the full tensor."""
        import os as _os2
        _mat_dbg = _os2.environ.get("MEMRIFT_MAT_DBG19", "0") == "1" and self.layer_idx == 19
        if _mat_dbg:
            _pf_state = "NONE" if self._prefetch_future is None else (
                "DONE" if self._prefetch_future.done() else "PENDING"
            )
            print(f"[MAT_DBG L19] enter: _bf16={'SET' if self._bf16 is not None else 'NONE'}"
                  f" _prefetch_future={_pf_state}", flush=True)

        if self._bf16 is not None:
            if _mat_dbg:
                print(f"[MAT_DBG L19] fast-path: _bf16 already set, returning", flush=True)
            return self._bf16
        
        # Consume async prefetch if present (demo-style: wait for async, never decompress on main thread).
        # When prefetch was submitted, we always wait for it so decompress stays in thread pool and GPU
        # can stay busy with merge on h2d_stream while main thread blocks in result().
        if self._prefetch_future is not None:
            pref = self._prefetch_future
            self._prefetch_future = None
            try:
                if not pref.done():
                    _trace(f"materialize: prefetch not ready (layer={self.layer_idx}), wait for async (demo-style)")
                # Nsight: 若此处耗时明显，说明预取未盖住计算（加深 prefetch / 增 workers）
                with _memrift_nvtx_range(
                    f"MemRift/wait_prefetch_future L{self.layer_idx}"
                ):
                    _t_wait0 = time.perf_counter()
                    pref_out = pref.result()
                    _wait_ms = (time.perf_counter() - _t_wait0) * 1000.0
                if layer_time_profiler is not None and layer_time_profiler.is_enabled():
                    layer_time_profiler.add_time(
                        f"decoder.layers.{self.layer_idx}"
                        if self.layer_idx >= 0
                        else "__memrift_prefetch__",
                        "prefetch_future_result_wait_ms",
                        _wait_ms,
                    )
                if isinstance(pref_out, tuple) and len(pref_out) == 3:
                    # (bf16, evt, cpu_exp): thread exited without GPU sync.
                    # cpu_exp must stay alive until the merge DMA completes.
                    self._bf16, self._CtoD_evt, self._exp_host = pref_out
                elif isinstance(pref_out, tuple) and len(pref_out) == 2:
                    self._bf16, self._CtoD_evt = pref_out
                else:
                    self._bf16 = pref_out
                self._ready_event.set()
                _trace(f"materialize: consumed prefetched tensor (layer={self.layer_idx})")
                if sync and self._CtoD_evt is not None and not _DEEP_ASYNC:
                    # Non-deep-async: CPU-side sync; safe to free cpu_exp after this.
                    self._CtoD_evt.synchronize()
                    if self._exp_host is not None:
                        del self._exp_host
                        self._exp_host = None
                # _DEEP_ASYNC path: _exp_host freed in cp.release() after use.
                return self._bf16
            except fut.CancelledError:
                pass  # fall through to sync path below
            except Exception as e:
                _trace(f"materialize: prefetch failed (layer={self.layer_idx}): {type(e).__name__}")
                # fall through to sync path
        
        # Import CUDA extension
        try:
            from flagscale.compress.float_split_stride_pin import float_split_stride_pin as fs_sp
            if not fs_sp.is_available():
                raise RuntimeError(
                    "CUDA extension float_split_stride_pin not available. "
                    "Please build it first: cd flagscale/compress/float_split_stride_pin && pip install -e ."
                )
        except ImportError as e:
            raise RuntimeError(
                f"CUDA extension not importable: {e}. "
                "MemRift requires the CUDA extension for weight decompression."
            )
        
        numel = int(np.prod(self.orig_shape))
        strides = _c_contiguous_strides(self.orig_shape)
        stream = torch.cuda.current_stream()

        # ── GPU path: nvCOMP LZ4 ──────────────────────────────────────────
        # Activated when exp_mv is prefixed with NVCOMP_LZ4_MAGIC ("NVL4").
        # Eliminates CPU zstd blocking: CPU only does a fast memcpy to pinned
        # staging, then all H2D/decode/merge run asynchronously on GPU.
        if (NVCOMP_AVAILABLE
                and isinstance(self.exp_mv, (bytes, bytearray, memoryview))
                and len(self.exp_mv) >= 4
                and bytes(self.exp_mv[:4]) == NVCOMP_LZ4_MAGIC):

            # 1. Use persistent pinned buffer (pre-allocated on main thread at load
            #    time to avoid ~10ms cold-cache alloc from thread-pool threads).
            #    comp_pinned is safe to reuse: H2D ops on the same stream are
            #    serialized, so the GPU finishes reading before the next submission.
            if self.comp_pinned is not None:
                comp_pinned = self.comp_pinned  # reuse persistent buffer
            else:
                comp_data = self.exp_mv[4:]
                comp_pinned = torch.empty(len(comp_data), dtype=torch.uint8, pin_memory=True)
                comp_pinned.numpy()[:] = np.frombuffer(comp_data, dtype=np.uint8)

            # 2. H2D DMA compressed bytes → GPU (async, on current stream)
            with torch.cuda.stream(stream):
                comp_gpu = comp_pinned.to(
                    self.sm_gpu.device, non_blocking=True)

            # 3. GPU nvCOMP LZ4 decode (queued on stream; runs after H2D)
            nvcomp_codec = _nvcomp_lib.Codec(
                algorithm='lz4', cuda_stream=stream.cuda_stream)
            comp_arr = _nvcomp_lib.as_array(comp_gpu.view(torch.int8))
            decomp = nvcomp_codec.decode(comp_arr)

            # 4. Convert decoded output to a proper PyTorch-owned tensor.
            #    .clone() runs after decode (same stream) and breaks the
            #    DLPack lifetime dependency on decomp to avoid double-free.
            decomp_raw = torch.from_dlpack(
                decomp.to_dlpack()).view(torch.uint8)
            with torch.cuda.stream(stream):
                exp_gpu = decomp_raw[:numel].clone()
            exp_gpu.record_stream(stream)

            # 5. GPU merge: exp_gpu + sm_gpu → bf16
            with torch.cuda.stream(stream):
                self._bf16 = fs_sp.merge(
                    exp_gpu, self.sm_gpu,
                    list(self.orig_shape), list(strides), 0,
                    self._dtype, stream.cuda_stream,
                )
            ev = stream.record_event()
            self._CtoD_evt = ev
            # Keep staging alive until caller syncs ev:
            #   comp_pinned: persistent (not freed), GPU DMA source – included to
            #     document the lifetime but NOT deleted in release().
            #   comp_gpu: decode source; decomp_raw: DLPack ref; exp_gpu: merge input.
            self._exp_host = (comp_gpu, nvcomp_codec, decomp, decomp_raw, exp_gpu)
            self._ready_event.set()

            if sync and not _DEEP_ASYNC:
                ev.synchronize()
                self._exp_host = None

            return self._bf16

        # ── GPU path: nvCOMP ANS ──────────────────────────────────────────
        # prepare_weight.py --compression nvcomp_ans (magic "NVAN").
        if (NVCOMP_AVAILABLE
                and isinstance(self.exp_mv, (bytes, bytearray, memoryview))
                and len(self.exp_mv) >= 4
                and bytes(self.exp_mv[:4]) == NVCOMP_ANS_MAGIC):

            if _mat_dbg:
                import threading as _thr
                print(f"[MAT_DBG L19] SYNC-ANS-decode path (no prefetch future!) thread={_thr.current_thread().name}", flush=True)

            if self.comp_pinned is not None:
                comp_pinned = self.comp_pinned
            else:
                comp_data = self.exp_mv[4:]
                comp_pinned = torch.empty(len(comp_data), dtype=torch.uint8, pin_memory=True)
                comp_pinned.numpy()[:] = np.frombuffer(comp_data, dtype=np.uint8)

            with torch.cuda.stream(stream):
                comp_gpu = comp_pinned.to(
                    self.sm_gpu.device, non_blocking=True)

            nvcomp_codec = _nvcomp_lib.Codec(
                algorithm='ans', cuda_stream=stream.cuda_stream)
            comp_arr = _nvcomp_lib.as_array(comp_gpu.view(torch.int8))
            decomp = nvcomp_codec.decode(comp_arr)

            decomp_raw = torch.from_dlpack(decomp).view(torch.uint8)
            decomp_raw.record_stream(stream)
            with torch.cuda.stream(stream):
                self._bf16 = fs_sp.merge(
                    decomp_raw, self.sm_gpu,
                    list(self.orig_shape), list(strides), 0,
                    self._dtype, stream.cuda_stream,
                )
            ev = stream.record_event()
            self._CtoD_evt = ev
            self._exp_host = (comp_gpu, nvcomp_codec, decomp, decomp_raw)
            self._ready_event.set()

            if sync and not _DEEP_ASYNC:
                ev.synchronize()
                self._exp_host = None

            return self._bf16

        # ── CPU path: zstd (backward compat / old compressed files) ──────
        self._exp_host = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
        dctx = _get_dctx()
        with dctx.stream_reader(memoryview(self.exp_mv)) as reader:
            view = memoryview(self._exp_host.numpy())
            nread = reader.readinto(view)
            assert nread == numel, f"decompress size mismatch: {nread} vs {numel}"

        with torch.cuda.stream(stream):
            self._bf16 = fs_sp.merge(
                self._exp_host, self.sm_gpu,
                list(self.orig_shape), list(strides), 0,
                self._dtype, stream.cuda_stream
            )
        ev = stream.record_event()
        self._CtoD_evt = ev
        self._ready_event.set()

        if sync and not _DEEP_ASYNC:
            ev.synchronize()

        return self._bf16
    
    def wait_ready(self):
        """Wait for materialization to complete."""
        self._ready_event.wait()
        if self._CtoD_evt and not _DEEP_ASYNC:
            self._CtoD_evt.synchronize()
    
    def release(self):
        """Release the materialized tensor to free GPU memory."""
        if self._bf16 is None:
            return
        
        try:
            from flagscale.compress.float_split_stride_pin import float_split_stride_pin as fs_sp
            if fs_sp.is_available():
                fs_sp.release_cuda(self._bf16)
        except:
            pass
        
        self._bf16 = None
        self._ready_event.clear()
        self._CtoD_evt = None
        
        if self._exp_host is not None:
            del self._exp_host
            self._exp_host = None
    
    def release_compressed(self):
        """Release compressed data (when no longer needed)."""
        self.exp_mv = None
        self.sm_gpu = None


@dataclass
class MergedWeightGroup:
    """
    Group of CompressedParams that need to be merged into one Megatron weight.
    
    For TP=1:
    - qkv: concat(q, k, v) along dim 0
    - fc1: concat(gate, up) along dim 0
    """
    megatron_target: str  # e.g., 'self_attention.linear_qkv'
    layer_idx: int
    components: Dict[str, CompressedParam] = field(default_factory=dict)
    target_module: Optional[nn.Module] = None
    target_attr: str = "weight"
    target_shape: Optional[Tuple[int, ...]] = None
    
    def is_complete(self) -> bool:
        """Check if all components are present."""
        if "linear_qkv" in self.megatron_target:
            return all(k in self.components for k in ["q", "k", "v"])
        elif "linear_fc1" in self.megatron_target:
            return all(k in self.components for k in ["gate", "up"])
        return True
    
    def get_merged_shape(self) -> Tuple[int, ...]:
        """Get the shape of the merged weight (TP=1)."""
        if "linear_qkv" in self.megatron_target:
            # qkv: concat along dim 0
            q_shape = self.components["q"].orig_shape
            k_shape = self.components["k"].orig_shape
            v_shape = self.components["v"].orig_shape
            return (q_shape[0] + k_shape[0] + v_shape[0], q_shape[1])
        elif "linear_fc1" in self.megatron_target:
            # fc1: concat along dim 0
            gate_shape = self.components["gate"].orig_shape
            up_shape = self.components["up"].orig_shape
            return (gate_shape[0] + up_shape[0], gate_shape[1])
        else:
            # Single weight
            cp = list(self.components.values())[0]
            return cp.orig_shape


def get_group_by_tensor_ptr(ptr: int) -> Optional[MergedWeightGroup]:
    """Lookup merged weight group by currently materialized tensor ptr."""
    return _PTR2GROUP.get(int(ptr))


def _materialize_group_tensor(group: MergedWeightGroup, sync: bool = True) -> torch.Tensor:
    """Materialize and assemble one merged weight group.

    Fast path: if the target param already holds a valid (non-empty) weight, return it
    directly without re-decompressing.  This prevents redundant work for layers that were
    pre-materialized by prefetch_initial_layers(): their param.data is filled, but
    cp._bf16 was already cleared after torch.cat, so a naive call to cp.materialize()
    would trigger a full re-decompress even though the weight is already on GPU.

    After torch.cat, the individual component tensors (cp._bf16) are released
    immediately so that only the merged result stays on GPU.  This avoids a 2x
    memory peak that would otherwise occur (components + merged copy).
    cp.release() is idempotent so _clear_param can still call it safely later.
    """
    # Fast path: weight already resident in param.data – skip decompression.
    if group.target_module is not None:
        _param = getattr(group.target_module, group.target_attr, None)
        if _param is not None and _param.data.numel() > 0:
            import os as _os3
            if _os3.environ.get("MEMRIFT_MAT_DBG19", "0") == "1" and group.layer_idx == 19:
                print(f"[MAT_DBG L19] _materialize_group_tensor fast-path: param.data already set, numel={_param.data.numel()}", flush=True)
            return _param.data

    if "linear_qkv" in group.megatron_target:
        q_cp = group.components.get("q")
        k_cp = group.components.get("k")
        v_cp = group.components.get("v")
        if q_cp is None or k_cp is None or v_cp is None:
            raise RuntimeError(f"Incomplete QKV group: {list(group.components.keys())}")
        q = q_cp.materialize(sync=sync)
        k = k_cp.materialize(sync=sync)
        v = v_cp.materialize(sync=sync)
        if sync and not _DEEP_ASYNC:
            q_cp.wait_ready()
            k_cp.wait_ready()
            v_cp.wait_ready()
        else:
            # DEEP_ASYNC / event-based: make compute stream wait for h2d_stream
            # before torch.cat so the merge DMA is visible on the compute stream.
            cur = torch.cuda.current_stream()
            for _cp in (q_cp, k_cp, v_cp):
                if _cp._CtoD_evt is not None:
                    cur.wait_event(_cp._CtoD_evt)
        # Some HF checkpoints use GQA-style QKV shapes (e.g. q=H, k=v=H/8),
        # while Megatron target may expect dense-style (q=k=v=H) when
        # group_query_attention is disabled. If target shape is known and rows
        # mismatch, expand K/V rows to match target before concatenation.
        target_rows = None
        if group.target_shape is not None and len(group.target_shape) >= 1:
            target_rows = int(group.target_shape[0])
        if target_rows is not None:
            merged_rows = int(q.shape[0] + k.shape[0] + v.shape[0])
            if merged_rows != target_rows:
                q_rows = int(q.shape[0])
                kv_rows_total = target_rows - q_rows
                if kv_rows_total > 0 and kv_rows_total % 2 == 0:
                    expect_k_rows = kv_rows_total // 2
                    expect_v_rows = kv_rows_total // 2
                    rk_ok = (expect_k_rows % int(k.shape[0]) == 0)
                    rv_ok = (expect_v_rows % int(v.shape[0]) == 0)
                    if rk_ok and rv_ok:
                        rk = expect_k_rows // int(k.shape[0])
                        rv = expect_v_rows // int(v.shape[0])
                        if rk > 1:
                            k = k.repeat_interleave(rk, dim=0)
                        if rv > 1:
                            v = v.repeat_interleave(rv, dim=0)
                    else:
                        raise RuntimeError(
                            f"QKV row mismatch cannot be adapted: target_rows={target_rows}, "
                            f"q={q.shape[0]}, k={k.shape[0]}, v={v.shape[0]}"
                        )
        merged = torch.cat([q, k, v], dim=0)
        # Release component tensors immediately to avoid 2x peak memory.
        q_cp.release()
        k_cp.release()
        v_cp.release()
        return merged

    if "linear_fc1" in group.megatron_target:
        gate_cp = group.components.get("gate")
        up_cp = group.components.get("up")
        if gate_cp is None or up_cp is None:
            raise RuntimeError(f"Incomplete FC1 group: {list(group.components.keys())}")
        gate = gate_cp.materialize(sync=sync)
        up = up_cp.materialize(sync=sync)
        if sync and not _DEEP_ASYNC:
            gate_cp.wait_ready()
            up_cp.wait_ready()
        else:
            cur = torch.cuda.current_stream()
            for _cp in (gate_cp, up_cp):
                if _cp._CtoD_evt is not None:
                    cur.wait_event(_cp._CtoD_evt)
        merged = torch.cat([gate, up], dim=0)
        gate_cp.release()
        up_cp.release()
        return merged

    cp = group.components.get("single")
    if cp is None:
        cp = list(group.components.values())[0]
    weight = cp.materialize(sync=sync)
    if sync and not _DEEP_ASYNC:
        cp.wait_ready()
    return weight


def ensure_group_param_materialized(group: MergedWeightGroup) -> Optional[torch.Tensor]:
    """Ensure target param data for group is present and return it."""
    if group.target_module is None:
        return None
    param = getattr(group.target_module, group.target_attr, None)
    if param is None:
        return None
    if param.data.numel() == 0:
        weight = _materialize_group_tensor(group, sync=True)
        with torch.no_grad():
            if weight.device == param.device and weight.dtype == param.dtype:
                param.data = weight
            else:
                param.data = weight.to(device=param.device, dtype=param.dtype)
        _PTR2GROUP[int(param.data_ptr())] = group
    return param.data


class MegatronDynamicLoader:
    """
    Dynamic weight loader for Megatron models (TP=1 only).
    
    Key features:
    1. HF -> Megatron name mapping
    2. Merged weight assembly (qkv, fc1)
    3. cp -> param binding and write-back
    4. Original weight release
    5. TE-compatible pre-decompression
    6. Forward/backward hooks
    
    Usage:
        loader = MegatronDynamicLoader(
            model=model,
            comp_dir="/path/to/compressed",
            device=torch.device("cuda:0"),
        )
        loader.load_weights()
        loader.build_param_mapping()
        loader.release_original_weights()
        loader.install_hooks()
        loader.prefetch_initial_layers()  # For TE compatibility
    """
    
    # HF to Megatron name mapping
    HF_TO_MEGATRON = {
        "q_proj": ("self_attention.linear_qkv", "q"),
        "k_proj": ("self_attention.linear_qkv", "k"),
        "v_proj": ("self_attention.linear_qkv", "v"),
        "o_proj": ("self_attention.linear_proj", None),
        "gate_proj": ("mlp.linear_fc1", "gate"),
        "up_proj": ("mlp.linear_fc1", "up"),
        "down_proj": ("mlp.linear_fc2", None),
    }
    
    # Megatron target weights
    MEGATRON_WEIGHT_TARGETS = [
        "self_attention.linear_qkv",
        "self_attention.linear_proj",
        "mlp.linear_fc1",
        "mlp.linear_fc2",
    ]
    
    def __init__(
        self,
        model: nn.Module,
        comp_dir: str,
        device: torch.device,
        tp_rank: int = 0,
        tp_size: int = 1,
        prefetch_layers: int = 1,
        print_debug: bool = False,
        allowed_targets: Optional[Set[str]] = None,
    ):
        """
        Initialize the loader.
        
        Args:
            model: Megatron model (unwrapped, should be the decoder/GPTModel)
            comp_dir: Path to compressed weights directory
            device: Target CUDA device
            tp_rank: Tensor parallel rank (must be 0 for v1)
            tp_size: Tensor parallel world size (must be 1 for v1)
            prefetch_layers: Number of layers to prefetch
            print_debug: Print debug messages
            allowed_targets: Optional Megatron target filter. If set, only these
                targets are dynamically managed (others are skipped).
        """
        # Validate TP=1 constraint
        if tp_size != 1:
            raise ValueError(
                f"MegatronDynamicLoader v1 only supports TP=1, got tp_size={tp_size}. "
                "Please set tensor_model_parallel_size: 1 in your config."
            )
        
        self.model = model
        self.comp_dir = comp_dir
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        # Respect the configured prefetch span so large models can trade a bit of
        # overlap for lower steady-state memory.
        self.prefetch_layers = max(prefetch_layers, 1)
        self.print_debug = print_debug
        self.allowed_targets = allowed_targets
        # Relax hook-side hard sync to reduce main-thread stalls.
        # Default to relaxed sync for better overlap; force strict sync with
        # MEMRIFT_WEIGHT_SYNC=1. Keep MEMRIFT_WEIGHT_RELAX_SYNC for compatibility.
        strict_sync = os.environ.get("MEMRIFT_WEIGHT_SYNC", "0") == "1"
        if os.environ.get("MEMRIFT_WEIGHT_RELAX_SYNC", "") == "1":
            strict_sync = False
        self._hook_materialize_sync = strict_sync
        
        # Load index
        with open(os.path.join(comp_dir, "index.json")) as f:
            self.index = json.load(f)
        
        # Data structures
        self.num_layers = 0
        self.layer_names: List[str] = []
        
        # layer_idx -> megatron_target -> MergedWeightGroup
        self.merged_groups: Dict[int, Dict[str, MergedWeightGroup]] = defaultdict(dict)
        
        # layer_name -> list of MergedWeightGroup (for hooks)
        self.layer2groups: Dict[str, List[MergedWeightGroup]] = defaultdict(list)
        
        # All CompressedParams (for memory tracking)
        self.all_cps: List[CompressedParam] = []
        
        # Non-layer params (embed, lm_head, etc.)
        self.non_layer_cps: List[CompressedParam] = []
        
        # Track original weights that were released
        self.released_params: Set[int] = set()  # id(param) set
        
        # Async compressor reference
        self.async_compressor = None
        self._hook_warn_ms = _env_float("MEMRIFT_HOOK_WARN_MS", 800.0)
        
        # Backward empty_cache counter (aligned with memrift_demo)
        self._bwd_counter = 0
        self._bwd_empty_step = int(_env_float("MEMRIFT_BWD_EMPTY_STEP", 5))
        self._prefetch_submit_mode = os.environ.get(
            "MEMRIFT_WEIGHT_PREFETCH_MODE", "on_stream"
        )
        self._prefetch_submit_mode_fwd = os.environ.get(
            "MEMRIFT_WEIGHT_PREFETCH_MODE_FWD",
            self._prefetch_submit_mode,
        )
        self._prefetch_submit_mode_bwd = os.environ.get(
            "MEMRIFT_WEIGHT_PREFETCH_MODE_BWD",
            self._prefetch_submit_mode,
        )
        self._windowed_prefetch = os.environ.get("MEMRIFT_WINDOWED_PREFETCH", "0") == "1"
        self._windowed_prefetch_fwd = os.environ.get(
            "MEMRIFT_WINDOWED_PREFETCH_FWD",
            "1" if self._windowed_prefetch else "0",
        ) == "1"
        self._windowed_prefetch_bwd = os.environ.get(
            "MEMRIFT_WINDOWED_PREFETCH_BWD",
            "1" if self._windowed_prefetch else "0",
        ) == "1"
        self._fwd_prefetch_cursor = 0
        self._bwd_prefetch_cursor = -1

    def _count_pending_prefetch(self, groups: List[MergedWeightGroup]) -> int:
        pending = 0
        for group in groups:
            for cp in group.components.values():
                fut_obj = cp._prefetch_future
                if fut_obj is not None and not fut_obj.done():
                    pending += 1
        return pending

    def _submit_prefetch_layers(self, layer_names: List[str], async_comp, submit_mode: str) -> int:
        if async_comp is None or not layer_names:
            return 0
        to_prefetch = []
        for layer_name in layer_names:
            for group in self.layer2groups.get(layer_name, []):
                for cp in group.components.values():
                    if cp._bf16 is None and cp._prefetch_future is None:
                        to_prefetch.append(cp)
        if not to_prefetch:
            return 0
        if submit_mode == "async_bg":
            futures = async_comp.prefetch_batch_async(to_prefetch)
        else:
            futures = async_comp.prefetch_batch_on_stream(to_prefetch)
        for cp, fut_obj in futures.items():
            cp._prefetch_future = fut_obj
        return len(to_prefetch)

    def _schedule_forward_prefetch(self, cur_idx: int, span: int, async_comp) -> int:
        """Schedule forward prefetch with batched submit."""
        if async_comp is None or self._fwd_prefetch_cursor >= self.num_layers:
            return 0
        if cur_idx < self._fwd_prefetch_cursor - span:
            return 0

        end_idx = min(self.num_layers, self._fwd_prefetch_cursor + span)
        layer_names = self.layer_names[self._fwd_prefetch_cursor:end_idx]

        # Collect all components to prefetch from all layers in one go
        to_prefetch = []
        for layer_name in layer_names:
            for group in self.layer2groups.get(layer_name, []):
                for cp in group.components.values():
                    if cp._bf16 is None and cp._prefetch_future is None:
                        to_prefetch.append(cp)

        if not to_prefetch:
            self._fwd_prefetch_cursor = end_idx
            return 0

        # Submit ALL components in a single prefetch_batch_on_stream call
        if self._prefetch_submit_mode_fwd == "async_bg":
            futures = async_comp.prefetch_batch_async(to_prefetch)
        else:
            futures = async_comp.prefetch_batch_on_stream(to_prefetch)

        for cp, fut_obj in futures.items():
            cp._prefetch_future = fut_obj

        self._fwd_prefetch_cursor = end_idx
        return len(to_prefetch)

    def _schedule_backward_prefetch(self, cur_idx: int, span: int, async_comp) -> int:
        """Schedule backward prefetch with batched submit."""
        if async_comp is None or self._bwd_prefetch_cursor < 0:
            return 0
        if cur_idx > self._bwd_prefetch_cursor + (span - 1):
            return 0

        start_idx = max(0, self._bwd_prefetch_cursor - span + 1)
        layer_names = [self.layer_names[i] for i in range(self._bwd_prefetch_cursor, start_idx - 1, -1)]

        # Collect all components to prefetch from all layers in one go
        to_prefetch = []
        for layer_name in layer_names:
            for group in self.layer2groups.get(layer_name, []):
                for cp in group.components.values():
                    if cp._bf16 is None and cp._prefetch_future is None:
                        to_prefetch.append(cp)

        if not to_prefetch:
            self._bwd_prefetch_cursor = start_idx - 1
            return 0

        # Submit ALL components in a single prefetch_batch_on_stream call
        if self._prefetch_submit_mode_bwd == "async_bg":
            futures = async_comp.prefetch_batch_async(to_prefetch)
        else:
            futures = async_comp.prefetch_batch_on_stream(to_prefetch)

        for cp, fut_obj in futures.items():
            cp._prefetch_future = fut_obj

        self._bwd_prefetch_cursor = start_idx - 1
        return len(to_prefetch)
    
    def _get_layer_idx(self, hf_name: str) -> Optional[int]:
        """Extract layer index from HF parameter name."""
        parts = hf_name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None
    
    def _map_hf_to_megatron(self, hf_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Map HF parameter name to Megatron target and merge key.
        
        Returns:
            (megatron_target, merge_key) or (None, None) if not mapped
        """
        for hf_suffix, (mg_target, merge_key) in self.HF_TO_MEGATRON.items():
            if f".{hf_suffix}." in hf_name or hf_name.endswith(f".{hf_suffix}.weight"):
                return mg_target, merge_key
        return None, None
    
    def load_weights(self):
        """
        Load compressed weights from disk and organize into MergedWeightGroups.
        """
        if self.print_debug:
            print(f"[MemRift] Loading weights from {self.comp_dir}")
        
        # First pass: count layers
        for entry in self.index:
            layer_idx = self._get_layer_idx(entry["name"])
            if layer_idx is not None:
                self.num_layers = max(self.num_layers, layer_idx + 1)
        
        self.layer_names = [f"decoder.layers.{i}" for i in range(self.num_layers)]
        
        if self.print_debug:
            print(f"[MemRift] Found {self.num_layers} layers")
        
        # Second pass: load weights and organize
        for entry in self.index:
            if entry["scheme"] not in (
                "split_zstd",
                "split_nvcomp_lz4",
                "split_nvcomp_ans",
            ):
                if self.print_debug:
                    print(f"[MemRift] Skipping non-split: {entry['name']} ({entry['scheme']})")
                continue
            
            hf_name = entry["name"]
            layer_idx = self._get_layer_idx(hf_name)
            megatron_target, merge_key = self._map_hf_to_megatron(hf_name)
            
            # Read compressed data
            file_path = os.path.join(self.comp_dir, entry["file"])
            with open(file_path, "rb") as f:
                numel = struct.unpack("<Q", f.read(8))[0]
                sm_size = numel * (1 if entry["dtype"] == "bfloat16" else 3)
                sm_bytes = np.frombuffer(f.read(sm_size), dtype=np.uint8)
                exp_bytes = f.read()
            
            # Create tensors: use pinned host memory for sm so H2D is fast (DMA).
            # as_tensor(..., device=device) from numpy uses pageable memory -> slow H2D.
            sm_pinned = torch.from_numpy(sm_bytes).pin_memory()
            sm_gpu = sm_pinned.to(self.device, non_blocking=True)
            dtype = torch.bfloat16 if entry["dtype"] == "bfloat16" else torch.float32
            
            # Create CompressedParam
            cp = CompressedParam(entry["shape"], sm_gpu, exp_bytes, dtype)
            cp.hf_name = hf_name
            cp.megatron_target = megatron_target or ""
            cp.merge_key = merge_key
            cp.layer_idx = layer_idx if layer_idx is not None else -1

            # Pre-allocate persistent pinned buffer on the main thread for nvcomp payloads.
            # Thread-pool threads have cold per-thread caches → torch.empty(..., pin_memory=True)
            # takes ~10ms there vs ~0.02ms on the (warm) main thread.  Reusing a main-thread-
            # allocated buffer eliminates that 10ms × 308-calls bottleneck per training step.
            if NVCOMP_AVAILABLE and len(exp_bytes) >= 4:
                head = exp_bytes[:4]
                if head == NVCOMP_LZ4_MAGIC or head == NVCOMP_ANS_MAGIC:
                    comp_data = exp_bytes[4:]
                    cp.comp_pinned = torch.empty(
                        len(comp_data), dtype=torch.uint8, pin_memory=True)
                    cp.comp_pinned.numpy()[:] = np.frombuffer(comp_data, dtype=np.uint8)
                    # Offline ANS: same buffer is what batch decode reads (ANS codec).
                    if head == NVCOMP_ANS_MAGIC:
                        cp.ans_pinned = cp.comp_pinned
            
            self.all_cps.append(cp)
            
            if layer_idx is not None and megatron_target:
                # Layer parameter: add to merged group
                if megatron_target not in self.merged_groups[layer_idx]:
                    self.merged_groups[layer_idx][megatron_target] = MergedWeightGroup(
                        megatron_target=megatron_target,
                        layer_idx=layer_idx,
                    )
                
                group = self.merged_groups[layer_idx][megatron_target]
                if merge_key:
                    group.components[merge_key] = cp
                else:
                    # Single weight (proj, fc2)
                    group.components["single"] = cp
                
                if self.print_debug:
                    print(f"[MemRift] Loaded {hf_name} -> layer {layer_idx} / {megatron_target} / {merge_key or 'single'}")
            else:
                # Non-layer parameter
                self.non_layer_cps.append(cp)
                if self.print_debug:
                    print(f"[MemRift] Non-layer param: {hf_name}")
        
        # Validate merged groups
        for layer_idx, groups in self.merged_groups.items():
            for target, group in groups.items():
                if not group.is_complete():
                    missing = []
                    if "linear_qkv" in target:
                        missing = [k for k in ["q", "k", "v"] if k not in group.components]
                    elif "linear_fc1" in target:
                        missing = [k for k in ["gate", "up"] if k not in group.components]
                    print(f"[MemRift] Warning: Incomplete group layer {layer_idx} / {target}, missing: {missing}")
        
        if self.print_debug:
            print(f"[MemRift] Loaded {len(self.all_cps)} compressed params")

        # Re-compress LZ4 exponent bytes → ANS at startup for faster H2D + decode.
        # ANS achieves ~2.8x compression on real bf16 exponent bytes (vs LZ4's 1.0x),
        # reducing H2D time from ~4ms to ~1.7ms and decode from ~8.5ms to ~2.4ms.
        # This one-time GPU operation takes ~100ms for a 1B model.
        self._recompress_lz4_to_ans()

    def _recompress_lz4_to_ans(self):

        if not NVCOMP_AVAILABLE or _nvcomp_lib is None:
            return

        NVCOMP_ANS_MAGIC = b"NVAN"

        lz4_cps = [
            cp for cp in self.all_cps
            if (cp.comp_pinned is not None
                and getattr(cp, 'exp_mv', None) is not None
                and isinstance(cp.exp_mv, (bytes, bytearray, memoryview))
                and len(cp.exp_mv) >= 4
                and bytes(cp.exp_mv[:4]) == NVCOMP_LZ4_MAGIC)
        ]

        if not lz4_cps:
            return

        try:
            lz4_codec = _nvcomp_lib.Codec(algorithm='lz4')
            ans_codec = _nvcomp_lib.Codec(algorithm='ans')
        except Exception:
            return

        BATCH = 1  # process 1 weight at a time to avoid large GPU allocations
        total_saved_mb = 0.0
        n_converted = 0

        for i in range(0, len(lz4_cps), BATCH):
            batch = lz4_cps[i: i + BATCH]

            # H2D compressed data (LZ4) to GPU
            gpu_bufs = []
            for cp in batch:
                cg = cp.comp_pinned.to(self.device, non_blocking=True)
                gpu_bufs.append(cg)
            torch.cuda.synchronize(self.device)

            # Decode with LZ4
            lz4_arrs = [_nvcomp_lib.as_array(cg.view(torch.int8)) for cg in gpu_bufs]
            decoded_list = lz4_codec.decode(lz4_arrs)
            del lz4_arrs

            # Re-encode each with ANS, then D2H to pinned memory
            for cp, cg, decoded in zip(batch, gpu_bufs, decoded_list):
                raw_tensor = torch.from_dlpack(decoded).view(torch.uint8).clone()
                torch.cuda.synchronize(self.device)
                del decoded

                ans_arr = _nvcomp_lib.as_array(raw_tensor.view(torch.int8))
                ans_comp = ans_codec.encode(ans_arr)
                torch.cuda.synchronize(self.device)
                del raw_tensor, ans_arr

                # buffer_size = actual ANS bitstream bytes (not the full DLPack tensor
                # which may include padding). Use only the valid prefix.
                ans_valid_bytes = int(ans_comp.buffer_size)
                ans_comp_tensor = torch.from_dlpack(ans_comp).view(torch.uint8)
                ans_data = ans_comp_tensor[:ans_valid_bytes].clone()
                torch.cuda.synchronize(self.device)
                del ans_comp_tensor, ans_comp

                lz4_size = cp.comp_pinned.numel()
                total_saved_mb += (lz4_size - ans_valid_bytes) / 1e6

                # D2H to pinned memory (only valid bytes)
                ans_pinned = torch.empty(ans_valid_bytes, dtype=torch.uint8, pin_memory=True)
                ans_pinned.copy_(ans_data)
                torch.cuda.synchronize(self.device)
                del ans_data, cg
                cp.ans_pinned = ans_pinned
                n_converted += 1

            del gpu_bufs, decoded_list

        if self.print_debug:
            print(f"[MemRift] Re-compressed {n_converted} LZ4→ANS weights, "
                  f"saved {total_saved_mb:.1f}MB CPU pinned memory.")

    def build_param_mapping(self):
        """
        Build mapping from MergedWeightGroup to actual model parameters.
        
        This finds the target nn.Module and attribute for each group,
        so hooks can write materialized weights back to the model.
        """
        if self.print_debug:
            print("[MemRift] Building param mapping...")
        
        # Find decoder layers
        decoder_layers = None
        for name, module in self.model.named_modules():
            # Try common patterns
            if hasattr(module, "layers") and isinstance(module.layers, nn.ModuleList):
                decoder_layers = module.layers
                if self.print_debug:
                    print(f"[MemRift] Found decoder layers at: {name}.layers")
                break
        
        if decoder_layers is None:
            # Prefer explicit paths so we bind to the same structure used at runtime
            if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
                decoder_layers = self.model.decoder.layers
            elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "decoder"):
                decoder_layers = self.model.language_model.decoder.layers
            elif hasattr(self.model, "module"):
                # DDP wrapped: bind against unwrapped model
                unwrapped = self.model.module
                self.model = unwrapped
                return self.build_param_mapping()
        
        if decoder_layers is None:
            print("[MemRift] Warning: Could not find decoder layers, param binding may fail")
            return
        
        # Map each group to its target module
        for layer_idx, groups in self.merged_groups.items():
            if layer_idx >= len(decoder_layers):
                print(f"[MemRift] Warning: layer_idx {layer_idx} >= num_layers {len(decoder_layers)}")
                continue
            
            layer = decoder_layers[layer_idx]
            layer_name = self.layer_names[layer_idx]
            
            for target, group in groups.items():
                if self.allowed_targets is not None and target not in self.allowed_targets:
                    if self.print_debug:
                        print(f"[MemRift] Skip target by filter: layer {layer_idx} / {target}")
                    continue
                # Navigate to target module
                target_module = layer
                parts = target.split(".")
                for part in parts[:-1]:
                    if hasattr(target_module, part):
                        target_module = getattr(target_module, part)
                    else:
                        target_module = None
                        break
                
                if target_module is not None:
                    final_attr = parts[-1]
                    if hasattr(target_module, final_attr):
                        linear_module = getattr(target_module, final_attr)
                        # Resolve PEFT wrappers to an inner module that owns `.weight`.
                        bind_module = linear_module
                        if not (hasattr(bind_module, "weight") and getattr(bind_module, "weight") is not None):
                            for unwrap_attr in ("to_wrap", "base_layer", "module"):
                                inner = getattr(bind_module, unwrap_attr, None)
                                if isinstance(inner, nn.Module) and hasattr(inner, "weight"):
                                    if getattr(inner, "weight") is not None:
                                        bind_module = inner
                                        break
                        if not (hasattr(bind_module, "weight") and getattr(bind_module, "weight") is not None):
                            if self.print_debug:
                                print(
                                    f"[MemRift] Warning: skip target without weight: "
                                    f"layer {layer_idx} / {target} ({type(linear_module).__name__})"
                                )
                            continue
                        group.target_module = bind_module
                        group.target_attr = "weight"
                        if hasattr(bind_module, "weight") and getattr(bind_module, "weight") is not None:
                            try:
                                group.target_shape = tuple(bind_module.weight.shape)
                            except Exception:
                                group.target_shape = None
                        
                        # Also set on component cps for easier access
                        for cp in group.components.values():
                            cp.target_module = bind_module
                            cp.target_attr = "weight"
                        
                        if self.print_debug:
                            print(f"[MemRift] Mapped layer {layer_idx} / {target} -> {type(linear_module).__name__}")
                    else:
                        print(f"[MemRift] Warning: {final_attr} not found in {type(target_module).__name__}")
                else:
                    print(f"[MemRift] Warning: Could not navigate to {target} in layer {layer_idx}")
                
                # Add to layer2groups
                self.layer2groups[layer_name].append(group)
        
        if self.print_debug:
            total_mapped = sum(1 for g in self.layer2groups.values() for _ in g)
            print(f"[MemRift] Mapped {total_mapped} weight groups")
    
    def release_original_weights(self):
        """
        Release original weight tensors to free GPU memory.
        
        Replaces each mapped parameter's .data with torch.empty(0, ...) so the
        allocator can reclaim memory. Must be called after build_param_mapping().
        After this, TE would see (0,) shape on first forward, so call
        prefetch_initial_layers() after install_hooks() to pre-fill first K layers.
        """
        if self.print_debug:
            mem_before = torch.cuda.memory_allocated(self.device)
            print(f"[MemRift] Memory before release: {mem_before / 1024**2:.1f} MB")
        
        released_count = 0
        for layer_name, groups in self.layer2groups.items():
            for group in groups:
                if group.target_module is not None:
                    param = getattr(group.target_module, group.target_attr, None)
                    if param is not None and id(param) not in self.released_params:
                        dtype = param.dtype
                        dev = param.device
                        with torch.no_grad():
                            empty = torch.empty(0, dtype=dtype, device=dev)
                            if hasattr(param, "data"):
                                param.data = empty
                        self.released_params.add(id(param))
                        released_count += 1
        
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        
        if self.print_debug:
            mem_after = torch.cuda.memory_allocated(self.device)
            print(f"[MemRift] Released {released_count} original weights")
            print(f"[MemRift] Memory after release: {mem_after / 1024**2:.1f} MB")
            print(f"[MemRift] Memory saved: {(mem_before - mem_after) / 1024**2:.1f} MB")
    
    def _materialize_group(self, group: MergedWeightGroup, sync: bool = True):
        """
        Materialize all components of a group and assemble merged weight.
        
        For TP=1, merged weights are:
        - qkv: concat(q, k, v) along dim 0
        - fc1: concat(gate, up) along dim 0
        """
        return _materialize_group_tensor(group, sync=sync)
    
    def _set_param(self, group: MergedWeightGroup, weight: torch.Tensor, wait_metric: Optional[str] = None):
        """
        Write materialized weight back to the model parameter.
        
        Forward/backward use the parameter's .data; replacing param.data ensures
        the next forward/backward on this layer uses the decompressed weight.
        
        Avoids .to() when device/dtype already match to prevent an extra copy.
        """
        if group.target_module is None:
            return
        param = getattr(group.target_module, group.target_attr, None)
        if param is None:
            if self.print_debug:
                print(f"[MemRift] set_param skip: no param at {group.target_attr}")
            return
        with torch.no_grad():
            if weight.is_cuda:
                # Tell the caching allocator this tensor is now used on the compute
                # stream so it won't be reclaimed for h2d_stream allocations while
                # forward/backward kernels are still reading it.  Mirrors demo's
                # set_param(): self._bf16.record_stream(current_stream).
                weight.record_stream(torch.cuda.current_stream(weight.device))
            if weight.device == param.device and weight.dtype == param.dtype:
                param.data = weight
            else:
                param.data = weight.to(device=param.device, dtype=param.dtype)
        # Avoid host blocking: always attach producer events to current stream.
        # This is a no-op when tensors are already ready, and preserves correctness
        # for relaxed-sync mode without forcing synchronize() on host.
        cur_stream = torch.cuda.current_stream(param.device)
        wait_start_evt = wait_end_evt = None
        if (
            wait_metric
            and layer_time_profiler is not None
            and layer_time_profiler.is_enabled()
        ):
            wait_start_evt = torch.cuda.Event(enable_timing=True)
            wait_end_evt = torch.cuda.Event(enable_timing=True)
            wait_start_evt.record(cur_stream)
        for cp in group.components.values():
            if cp._CtoD_evt is not None:
                cur_stream.wait_event(cp._CtoD_evt)
        if wait_end_evt is not None and wait_start_evt is not None:
            wait_end_evt.record(cur_stream)
            wait_end_evt.synchronize()
            try:
                layer_time_profiler.add_time(
                    self.layer_names[group.layer_idx] if group.layer_idx >= 0 else "unknown",
                    wait_metric,
                    float(wait_start_evt.elapsed_time(wait_end_evt)),
                )
            except Exception:
                pass
        _PTR2GROUP[int(param.data_ptr())] = group
        if self.print_debug:
            print(f"[MemRift] set_param: layer {group.layer_idx} / {group.megatron_target} shape={weight.shape}")
    
    def _clear_param(self, group: MergedWeightGroup):
        """Clear parameter data to free memory."""
        if group.target_module is not None:
            param = getattr(group.target_module, group.target_attr, None)
            if param is not None:
                old_ptr = int(param.data_ptr()) if param.data.numel() > 0 else -1
                dtype = param.dtype
                device = param.device
                with torch.no_grad():
                    param.data = torch.empty(0, dtype=dtype, device=device)
                if old_ptr >= 0:
                    _PTR2GROUP.pop(old_ptr, None)
                if self.print_debug:
                    print(f"[MemRift] clear_param: layer {group.layer_idx} / {group.megatron_target}")
        
        # Also release component cps
        for cp in group.components.values():
            cp.release()
    
    def _materialize_and_set_layer(self, layer_name: str):
        """Materialize all groups in a layer and set params."""
        for group in self.layer2groups.get(layer_name, []):
            weight = self._materialize_group(group, sync=True)
            self._set_param(group, weight)
    
    def _release_layer(self, layer_name: str):
        """Release all groups in a layer."""
        for group in self.layer2groups.get(layer_name, []):
            self._clear_param(group)
    
    def prefetch_initial_layers(self):
        """
        Pre-materialize first K layers and write back to param.data (TE compatibility).
        
        After release_original_weights(), all param.data are empty; TE would fail
        on first forward due to shape checks. This fills param.data for the first
        prefetch_layers+1 layers so the first forward sees valid shapes.
        """
        k = min(self.prefetch_layers + 1, self.num_layers)
        if self.print_debug:
            print(f"[MemRift] Pre-materializing first {k} layers (param.data write-back for TE)")
        
        for i in range(k):
            if i < len(self.layer_names):
                layer_name = self.layer_names[i]
                _t0 = time.perf_counter()
                self._materialize_and_set_layer(layer_name)
                if layer_time_profiler is not None and layer_time_profiler.is_enabled():
                    layer_time_profiler.add_time(
                        layer_name,
                        "startup_weight_materialize_ms",
                        1000.0 * (time.perf_counter() - _t0),
                    )
        
        torch.cuda.synchronize(self.device)
        
        # Verify first K layers have non-empty params
        ok = 0
        for i in range(k):
            if i >= len(self.layer_names):
                break
            for group in self.layer2groups.get(self.layer_names[i], []):
                if group.target_module is not None:
                    param = getattr(group.target_module, group.target_attr, None)
                    if param is not None and param.data.numel() > 0:
                        ok += 1
        if self.print_debug:
            mem = torch.cuda.memory_allocated(self.device)
            print(f"[MemRift] Prefetch done: {ok} params filled, memory {mem / 1024**2:.1f} MB")
    
    def install_hooks(self, async_compressor=None):
        """
        Install forward/backward hooks for dynamic weight loading.
        
        Hooks:
        - forward_pre: materialize current layer (consuming prefetch future if any),
                       write to param via _set_param; then prefetch next layer (store future on cp).
        - forward_post: release current layer (except last)
        - backward_pre: materialize current layer, prefetch prev
        - backward_post: release current layer
        
        Async prefetch: materialize_async() future is stored on cp._prefetch_future.
        When the next layer runs, cp.materialize() consumes the future and returns
        the tensor; _materialize_group merges and _set_param writes to model param,
        so the prefetched result does land on the real parameter.
        
        Args:
            async_compressor: Optional AsyncCompressor for async prefetch
        """
        self.async_compressor = async_compressor

        # Per-iteration timing log (controlled by MEMRIFT_ITER_LOG env var).
        # Prints one line per layer per iteration so fast/slow iterations can be
        # compared side by side.  Only rank 0, only the first N iters
        # (MEMRIFT_ITER_LOG_ITERS, default 8).
        self._iter_log_enabled = os.environ.get("MEMRIFT_ITER_LOG", "0") == "1"
        self._iter_log_max_iters = int(os.environ.get("MEMRIFT_ITER_LOG_ITERS", "8"))
        # Diagnostic: always print env var values so we can verify propagation.
        print(f"[MemRift] install_hooks: MEMRIFT_ITER_LOG={os.environ.get('MEMRIFT_ITER_LOG','<unset>')} "
              f"_iter_log_enabled={self._iter_log_enabled} num_layers={len(self.layer_names)}", flush=True)
        self._fwd_iter_count = 0   # incremented when layer-0 fwd_pre fires
        self._bwd_iter_count = 0   # incremented when last-layer bwd_pre fires
        self._fwd_iter_wall_t0: float = 0.0   # wall-clock start of current fwd
        self._bwd_iter_wall_t0: float = 0.0   # wall-clock start of current bwd
        self._bwd_l0_abs_end: float = 0.0     # abs wall-clock when bwd_pre layer-0 ends

        # Find layer modules
        name2layer = {}
        for name, module in self.model.named_modules():
            name2layer[name] = module
        
        # Also try to find by decoder.layers pattern
        decoder_layers = None
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
            decoder_layers = self.model.decoder.layers
        elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "decoder"):
            decoder_layers = self.model.language_model.decoder.layers
        
        if decoder_layers is not None:
            for i, layer in enumerate(decoder_layers):
                name2layer[f"decoder.layers.{i}"] = layer
        
        if self.print_debug:
            found = [n for n in self.layer_names if n in name2layer]
            print(f"[MemRift] Found {len(found)}/{len(self.layer_names)} layer modules for hooks")
            print(f"[MemRift] Hook materialize sync={self._hook_materialize_sync}")
            print(
                f"[MemRift] Prefetch submit mode fwd={self._prefetch_submit_mode_fwd} "
                f"bwd={self._prefetch_submit_mode_bwd}"
            )
            print(
                f"[MemRift] Windowed prefetch fwd={self._windowed_prefetch_fwd} "
                f"bwd={self._windowed_prefetch_bwd}"
            )

        self._fwd_prefetch_cursor = min(self.prefetch_layers + 1, self.num_layers)
        self._bwd_prefetch_cursor = self.num_layers - 2
        
        # Forward hooks
        for i in range(len(self.layer_names)):
            cur = self.layer_names[i]
            span = max(1, int(self.prefetch_layers))
            nxt_names = self.layer_names[i + 1 : min(len(self.layer_names), i + 1 + span)]
            
            if cur not in name2layer:
                continue
            
            layer_module = name2layer[cur]
            cur_groups = self.layer2groups.get(cur, [])
            nxt_groups_list = [self.layer2groups.get(nm, []) for nm in nxt_names]
            
            # Capture variables
            def make_fwd_pre(cur_groups, nxt_groups_list, cur_name, cur_idx, async_comp):
                _fwd_hook_fired_once = [False]
                def _hook(mod, inp):
                    t0 = time.perf_counter()
                    if not _fwd_hook_fired_once[0]:
                        _fwd_hook_fired_once[0] = True
                        print(f"[MemRift] fwd_pre FIRST FIRE layer={cur_name} iter_log={self._iter_log_enabled}", flush=True)
                    _trace(f"fwd_pre: enter layer={cur_name}, groups={len(cur_groups)}")
                    pending_cur = self._count_pending_prefetch(cur_groups)
                    if pending_cur > 0:
                        _trace(f"fwd_pre: layer={cur_name} pending_prefetch_cur={pending_cur}")

                    # 1) Submit prefetch for NEXT K layers FIRST so CPU decompression
                    #    can overlap with current layer materialize + forward compute.
                    #    K is controlled by prefetch_layers.
                    _t_pref0 = time.perf_counter()
                    if async_comp and nxt_groups_list:
                        if self._windowed_prefetch_fwd:
                            n_prefetched = self._schedule_forward_prefetch(cur_idx, span, async_comp)
                        else:
                            # Collect all components that need prefetch, then batch-decode
                            # in a single codec.decode([...]) call (2x faster than per-call).
                            # Skip groups whose param.data is already resident (pre-materialized
                            # by prefetch_initial_layers) to avoid clogging the bg queue with
                            # wasted work that delays truly needed prefetches.
                            to_prefetch = []
                            for nxt_groups in nxt_groups_list:
                                for group in nxt_groups:
                                    if group.target_module is not None:
                                        _nxt_param = getattr(group.target_module, group.target_attr, None)
                                        if _nxt_param is not None and _nxt_param.data.numel() > 0:
                                            continue  # already resident, skip to avoid bg queue waste
                                    for cp in group.components.values():
                                        if cp._bf16 is None and cp._prefetch_future is None:
                                            to_prefetch.append(cp)
                            n_prefetched = len(to_prefetch)
                            if to_prefetch:
                                if self._prefetch_submit_mode_fwd == "async_bg":
                                    futures = async_comp.prefetch_batch_async(to_prefetch)
                                else:
                                    futures = async_comp.prefetch_batch_on_stream(to_prefetch)
                                for cp, fut_obj in futures.items():
                                    cp._prefetch_future = fut_obj
                    else:
                        n_prefetched = 0
                    _t_pref1 = time.perf_counter()
                    if layer_time_profiler is not None and layer_time_profiler.is_enabled():
                        layer_time_profiler.add_time(
                            cur_name,
                            "forward_prefetch_submit_ms",
                            1000.0 * (_t_pref1 - _t_pref0),
                        )
                    if os.environ.get("MEMRIFT_DBG2", "0") == "1":
                        print(f"[H] {cur_name}: n_pref={n_prefetched} pref={1000*(_t_pref1-_t_pref0):.1f}ms", flush=True)

                    # 2) Materialize current layer and write back to model param.
                    #    If the layer was pre-fetched, cp.materialize() consumes the future.
                    #    Fast path in _materialize_group_tensor: if param.data is already
                    #    valid (from prefetch_initial_layers) it returns it directly.
                    materialize_ms = 0.0
                    for group in cur_groups:
                        tg = time.perf_counter()
                        weight = self._materialize_group(
                            group, sync=self._hook_materialize_sync
                        )
                        self._set_param(group, weight, wait_metric="forward_current_wait_gpu_ms")
                        group_ms = (time.perf_counter() - tg) * 1000.0
                        materialize_ms += group_ms
                        if group_ms > self._hook_warn_ms:
                            _trace(
                                f"WARN fwd_pre: slow group materialize {group_ms:.1f} ms "
                                f"(layer={cur_name}, target={group.megatron_target})"
                            )
                    if layer_time_profiler is not None and layer_time_profiler.is_enabled():
                        layer_time_profiler.add_time(
                            cur_name,
                            "forward_weight_materialize_ms",
                            materialize_ms,
                        )
                        setattr(mod, "_memrift_fwd_compute_start", time.perf_counter())

                    hook_ms = (time.perf_counter() - t0) * 1000.0
                    if hook_ms > self._hook_warn_ms:
                        _trace(f"WARN fwd_pre: slow hook {hook_ms:.1f} ms (layer={cur_name})")
                    _trace(f"fwd_pre: done layer={cur_name}")
                    # ── Iteration timing log (MEMRIFT_ITER_LOG=1) ──────────────
                    if self._iter_log_enabled:
                        if cur_idx == 0:
                            self._fwd_iter_count += 1
                            self._fwd_iter_wall_t0 = t0
                            # Print gap from previous bwd_pre_L0 end to this fwd_pre_L0 start
                            if self._bwd_l0_abs_end > 0.0:
                                inter_iter_ms = (t0 - self._bwd_l0_abs_end) * 1000.0
                                print(
                                    f"[ITER_LOG] <<< inter_iter gap to iter={self._fwd_iter_count:3d}:"
                                    f"  {inter_iter_ms:8.1f}ms  (bwd_L0_end → fwd_L0_start)",
                                    flush=True,
                                )
                        last_fwd_idx = len(self.layer_names) - 1
                        if cur_idx == last_fwd_idx:
                            # Record when the last fwd hook finishes so we can compute
                            # the "silence" between fwd hooks end and bwd hooks start.
                            self._fwd_hooks_duration_ms = (time.perf_counter() - self._fwd_iter_wall_t0) * 1000.0
                        if self._fwd_iter_count <= self._iter_log_max_iters:
                            elapsed = (time.perf_counter() - self._fwd_iter_wall_t0) * 1000.0
                            print(
                                f"[ITER_LOG] fwd iter={self._fwd_iter_count:3d}"
                                f" layer={cur_idx:3d} mat={materialize_ms:7.1f}ms"
                                f" hook={hook_ms:7.1f}ms  t_since_fwd0={elapsed:8.1f}ms",
                                flush=True,
                            )
                return _hook
            
            def make_fwd_post(cur_groups, is_last, cur_name):
                def _hook(mod, inp, out):
                    if layer_time_profiler is not None and layer_time_profiler.is_enabled():
                        t_start = getattr(mod, "_memrift_fwd_compute_start", None)
                        if t_start is not None:
                            layer_time_profiler.add_time(
                                cur_name,
                                "forward_compute_ms",
                                1000.0 * (time.perf_counter() - t_start),
                            )
                            try:
                                delattr(mod, "_memrift_fwd_compute_start")
                            except Exception:
                                setattr(mod, "_memrift_fwd_compute_start", None)
                    # Release current layer (but not on last layer in forward)
                    if not is_last:
                        for group in cur_groups:
                            self._clear_param(group)
                return _hook
            
            is_last_layer = (i == len(self.layer_names) - 1)
            layer_module.register_forward_pre_hook(
                make_fwd_pre(cur_groups, nxt_groups_list, cur, i, async_compressor)
            )
            layer_module.register_forward_hook(
                make_fwd_post(cur_groups, is_last_layer, cur)
            )
        
        # Backward hooks (reverse order)
        for i in range(len(self.layer_names) - 1, -1, -1):
            cur = self.layer_names[i]
            span = max(1, int(self.prefetch_layers))
            prv_names = self.layer_names[max(0, i - span) : i][::-1]
            
            if cur not in name2layer:
                continue
            
            layer_module = name2layer[cur]
            cur_groups = self.layer2groups.get(cur, [])
            prv_groups_list = [self.layer2groups.get(nm, []) for nm in prv_names]
            
            def make_bwd_pre(cur_groups, prv_groups_list, cur_name, cur_idx, async_comp):
                def _hook(mod, grad_out):
                    t0 = time.perf_counter()
                    _trace(f"bwd_pre: enter layer={cur_name}, groups={len(cur_groups)}")
                    pending_cur = self._count_pending_prefetch(cur_groups)
                    if pending_cur > 0:
                        _trace(f"bwd_pre: layer={cur_name} pending_prefetch_cur={pending_cur}")

                    # 1) Submit prefetch for PREVIOUS K layers FIRST so CPU decompression
                    #    can overlap with current layer materialize + backward compute.
                    #    K is controlled by prefetch_layers.
                    prefetch_ms = 0.0
                    if async_comp and prv_groups_list:
                        if self._windowed_prefetch_bwd:
                            _tp0 = time.perf_counter()
                            self._schedule_backward_prefetch(cur_idx, span, async_comp)
                            prefetch_ms = 1000.0 * (time.perf_counter() - _tp0)
                        else:
                            # Batch-decode all components for previous layers at once.
                            # Skip groups whose param.data is already resident to avoid
                            # clogging the bg queue with wasted work.
                            to_prefetch = []
                            _bwd_pref_dbg = os.environ.get("MEMRIFT_MAT_DBG19", "0") == "1"
                            for prv_groups in prv_groups_list:
                                for group in prv_groups:
                                    _skip_reason = None
                                    if group.target_module is not None:
                                        _prv_param = getattr(group.target_module, group.target_attr, None)
                                        if _prv_param is not None and _prv_param.data.numel() > 0:
                                            _skip_reason = f"param_resident(numel={_prv_param.data.numel()})"
                                            if _bwd_pref_dbg and group.layer_idx == 19:
                                                print(f"[BWD_PREF_DBG] cur_idx={cur_idx} L19 group={group.megatron_target} SKIP: {_skip_reason}", flush=True)
                                            continue  # already resident, skip
                                    for cp in group.components.values():
                                        _cp_skip = None
                                        if cp._bf16 is not None:
                                            _cp_skip = f"_bf16=SET"
                                        elif cp._prefetch_future is not None:
                                            _cp_skip = f"_prefetch_future={'DONE' if cp._prefetch_future.done() else 'PENDING'}"
                                        if _cp_skip:
                                            if _bwd_pref_dbg and group.layer_idx == 19:
                                                print(f"[BWD_PREF_DBG] cur_idx={cur_idx} L19 group={group.megatron_target} cp={cp.merge_key} SKIP: {_cp_skip}", flush=True)
                                        else:
                                            to_prefetch.append(cp)
                                            if _bwd_pref_dbg and group.layer_idx == 19:
                                                print(f"[BWD_PREF_DBG] cur_idx={cur_idx} L19 group={group.megatron_target} cp={cp.merge_key} ADDED to prefetch", flush=True)
                            if to_prefetch:
                                _tp0 = time.perf_counter()
                                if self._prefetch_submit_mode_bwd == "async_bg":
                                    futures = async_comp.prefetch_batch_async(to_prefetch)
                                else:
                                    futures = async_comp.prefetch_batch_on_stream(to_prefetch)
                                for cp, fut_obj in futures.items():
                                    cp._prefetch_future = fut_obj
                                prefetch_ms = 1000.0 * (time.perf_counter() - _tp0)
                    if layer_time_profiler is not None and layer_time_profiler.is_enabled():
                        layer_time_profiler.add_time(
                            cur_name,
                            "backward_prefetch_submit_ms",
                            prefetch_ms,
                        )

                    # 2) Materialize current layer and write back to param (backward uses it).
                    materialize_ms = 0.0
                    for group in cur_groups:
                        tg = time.perf_counter()
                        weight = self._materialize_group(
                            group, sync=self._hook_materialize_sync
                        )
                        self._set_param(group, weight, wait_metric="backward_current_wait_gpu_ms")
                        group_ms = (time.perf_counter() - tg) * 1000.0
                        materialize_ms += group_ms
                        if group_ms > self._hook_warn_ms:
                            _trace(
                                f"WARN bwd_pre: slow group materialize {group_ms:.1f} ms "
                                f"(layer={cur_name}, target={group.megatron_target})"
                            )
                    if layer_time_profiler is not None and layer_time_profiler.is_enabled():
                        layer_time_profiler.add_time(
                            cur_name,
                            "backward_weight_materialize_ms",
                            materialize_ms,
                        )
                        setattr(mod, "_memrift_bwd_compute_start", time.perf_counter())

                    hook_ms = (time.perf_counter() - t0) * 1000.0
                    if hook_ms > self._hook_warn_ms:
                        _trace(f"WARN bwd_pre: slow hook {hook_ms:.1f} ms (layer={cur_name})")
                    _trace(f"bwd_pre: done layer={cur_name}")
                    # ── Iteration timing log (MEMRIFT_ITER_LOG=1) ──────────────
                    if self._iter_log_enabled:
                        last_bwd_idx = len(self.layer_names) - 1
                        if cur_idx == last_bwd_idx:  # first bwd hook = last layer
                            self._bwd_iter_count += 1
                            self._bwd_iter_wall_t0 = t0
                            fwd_to_bwd_ms = (t0 - self._fwd_iter_wall_t0) * 1000.0
                            fwd_hooks_to_bwd_ms = fwd_to_bwd_ms - getattr(self, "_fwd_hooks_duration_ms", 0.0)
                            print(
                                f"[ITER_LOG] >>> bwd_start iter={self._bwd_iter_count:3d}"
                                f"  fwd→bwd_gap={fwd_to_bwd_ms:8.1f}ms"
                                f"  (after_fwd_hooks+{fwd_hooks_to_bwd_ms:6.1f}ms)",
                                flush=True,
                            )
                        if self._bwd_iter_count <= self._iter_log_max_iters:
                            elapsed = (time.perf_counter() - self._bwd_iter_wall_t0) * 1000.0
                            print(
                                f"[ITER_LOG] bwd iter={self._bwd_iter_count:3d}"
                                f" layer={cur_idx:3d} mat={materialize_ms:7.1f}ms"
                                f" hook={hook_ms:7.1f}ms  t_since_bwd0={elapsed:8.1f}ms",
                                flush=True,
                            )
                            if cur_idx == 0:
                                # record abs time when last bwd hook ends
                                self._bwd_l0_abs_end = time.perf_counter()
                return _hook
            
            def make_bwd_post(cur_groups, cur_name):
                def _hook(mod, grad_in, grad_out):
                    if layer_time_profiler is not None and layer_time_profiler.is_enabled():
                        t_start = getattr(mod, "_memrift_bwd_compute_start", None)
                        if t_start is not None:
                            layer_time_profiler.add_time(
                                cur_name,
                                "backward_compute_ms",
                                1000.0 * (time.perf_counter() - t_start),
                            )
                            try:
                                delattr(mod, "_memrift_bwd_compute_start")
                            except Exception:
                                setattr(mod, "_memrift_bwd_compute_start", None)
                    for group in cur_groups:
                        self._clear_param(group)
                    self._bwd_counter += 1
                    if self._bwd_counter % self._bwd_empty_step == 0:
                        torch.cuda.empty_cache()
                return _hook
            
            layer_module.register_full_backward_pre_hook(
                make_bwd_pre(cur_groups, prv_groups_list, cur, i, async_compressor)
            )
            layer_module.register_full_backward_hook(
                make_bwd_post(cur_groups, cur)
            )

            # TE safety: some linear backward paths run before layer-level backward_pre
            # sees a fully-restored parameter. Add per-linear guard hooks to ensure
            # empty weights are materialized before GEMM in backward.
            seen_modules = set()
            linear_modules = []
            for group in cur_groups:
                tm = getattr(group, "target_module", None)
                if tm is None:
                    continue
                mid = id(tm)
                if mid in seen_modules:
                    continue
                seen_modules.add(mid)
                linear_modules.append(tm)

            def make_linear_bwd_pre(cur_groups):
                def _hook(mod, grad_out):
                    try:
                        w = getattr(mod, "weight", None)
                        if w is not None and w.data.numel() > 0:
                            return
                    except Exception:
                        pass
                    for group in cur_groups:
                        weight = self._materialize_group(
                            group, sync=self._hook_materialize_sync
                        )
                        self._set_param(group, weight, wait_metric="backward_current_wait_gpu_ms")
                return _hook

            linear_bwd_pre = make_linear_bwd_pre(cur_groups)
            for tm in linear_modules:
                tm.register_full_backward_pre_hook(linear_bwd_pre)
        
        if self.print_debug:
            print(f"[MemRift] Installed hooks for {len(self.layer_names)} layers")
    
    def reset(self):
        if self.async_compressor is not None:
            self.async_compressor.reset()
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics for monitoring."""
        sm_gpu_bytes = sum(cp.sm_gpu.numel() for cp in self.all_cps if cp.sm_gpu is not None)
        exp_bytes = sum(len(cp.exp_mv) for cp in self.all_cps if cp.exp_mv is not None)
        
        return {
            "sm_gpu_bytes": sm_gpu_bytes,
            "sm_gpu_mb": sm_gpu_bytes / 1024**2,
            "exp_cpu_bytes": exp_bytes,
            "exp_cpu_mb": exp_bytes / 1024**2,
            "num_layers": self.num_layers,
            "num_weight_groups": sum(len(g) for g in self.layer2groups.values()),
            "cuda_allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
        }
