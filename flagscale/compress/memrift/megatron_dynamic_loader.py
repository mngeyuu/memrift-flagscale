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
import weakref

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


# Thread-local decompressor
_tls = threading.local()
_PTR2GROUP: Dict[int, "MergedWeightGroup"] = {}


class WeightPlaceholder:
    """Lightweight placeholder for autograd-saved materialized weights.

    Holds a weakref to the MergedWeightGroup. The actual bf16 tensor is NOT held,
    allowing _clear_param to free it after forward. During backward, the
    corresponding layer's bwd_pre re-materializes the weight, and _unpack
    retrieves the fresh bf16 via group.target_module.weight.data.
    """
    __slots__ = ("group_ref", "shape", "stride")

    def __init__(self, group: "MergedWeightGroup", shape, stride):
        self.group_ref = weakref.ref(group)
        self.shape = tuple(shape)
        self.stride = tuple(stride)


def _lookup_weight_group(data_ptr: int) -> Optional["MergedWeightGroup"]:
    return _PTR2GROUP.get(int(data_ptr))
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
    
    def __new__(cls, orig_shape, sm_cpu, exp_mv, dtype, device):
        # Create a dummy empty tensor on the target CUDA device
        dummy = torch.empty(0, dtype=dtype, device=device)
        return super().__new__(cls, dummy, requires_grad=False)

    def __init__(self, orig_shape, sm_cpu, exp_mv, dtype, device):
        super().__init__()
        self.orig_shape = tuple(orig_shape)
        self._sm_gpu = sm_cpu.to(device)   # sign matrix lives on GPU permanently
        self.exp_mv = exp_mv
        self._dtype = dtype
        self._device = device  # target CUDA device
        self._bf16 = None
        self._ready_event = threading.Event()
        self._CtoD_evt = None
        self._exp_host = None
        
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
        """Decompress and materialize the full tensor."""
        if self._bf16 is not None:
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
                pref_out = pref.result()
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
        
        # Decompress exponent to pinned memory
        self._exp_host = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
        dctx = _get_dctx()
        with dctx.stream_reader(memoryview(self.exp_mv)) as reader:
            view = memoryview(self._exp_host.numpy())
            nread = reader.readinto(view)
            assert nread == numel, f"decompress size mismatch: {nread} vs {numel}"
        
        # Merge on GPU: sm is already resident on GPU
        strides = _c_contiguous_strides(self.orig_shape)
        stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            self._bf16 = fs_sp.merge(
                self._exp_host, self._sm_gpu,
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
        self._sm_gpu = None


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
            return _param.data

    if "linear_qkv" in group.megatron_target:
        # Shard mode: QKV already merged into a single tensor.
        single_cp = group.components.get("single")
        if single_cp is not None:
            return single_cp.materialize(sync=sync)

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
        # Shard mode: gate+up already merged into a single tensor.
        single_cp = group.components.get("single")
        if single_cp is not None:
            return single_cp.materialize(sync=sync)

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


# Counts on-demand re-materializations triggered from _unpack during backward.
_BWD_REMATERIALIZE_COUNT = [0]


def unpack_weight_for_backward(group: "MergedWeightGroup") -> torch.Tensor:
    """Materialize a group's weight on demand from saved_tensors_hooks._unpack.

    Runs INSIDE the TE fused autograd Function's backward, at the exact moment
    the weight is consumed (zero race). If a prior release freed the weight,
    this re-materializes it (self-healing).

    Returns the materialized weight tensor (param.data), or an empty tensor if
    materialization failed (caller raises).
    """
    if group.target_module is None:
        return torch.empty(0)
    param = getattr(group.target_module, group.target_attr, None)
    if param is not None and param.data.numel() > 0:
        return param.data
    if param is None:
        return torch.empty(0)
    # Empty -> materialize now (perfectly timed for TE backward consumption).
    _BWD_REMATERIALIZE_COUNT[0] += 1
    weight = ensure_group_param_materialized(group)
    return weight if weight is not None else torch.empty(0)


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

    # HF global (non-layer) param name → candidate Megatron paths (model-relative).
    # Tried in order; first match wins.  Handles both wrapped (language_model.*) and
    # bare Megatron GPT model structures.
    NON_LAYER_HF_TO_MEGATRON_HINTS: Dict[str, List[str]] = {
        "model.embed_tokens.weight": [
            "embedding.word_embeddings.weight",
            "language_model.embedding.word_embeddings.weight",
        ],
        "model.norm.weight": [
            "decoder.final_layernorm.weight",
            "language_model.decoder.final_layernorm.weight",
        ],
        "lm_head.weight": [
            "output_layer.weight",
            "language_model.output_layer.weight",
        ],
    }

    # Per-layer HF norm suffix → candidate Megatron paths (layer-relative).
    # TE (Transformer Engine) fuses layer norms into the adjacent linear module;
    # the non-TE fallback paths are also listed.
    LAYER_NORM_SUFFIX_TO_MEGATRON: Dict[str, List[str]] = {
        "input_layernorm.weight": [
            "self_attention.linear_qkv.layer_norm_weight",   # TE fused
            "input_layernorm.weight",                         # non-TE
        ],
        "post_attention_layernorm.weight": [
            "mlp.linear_fc1.layer_norm_weight",               # TE fused
            "post_attention_layernorm.weight",                 # non-TE
        ],
    }
    
    def __init__(
        self,
        model: nn.Module,
        comp_dir: str,
        device: torch.device,
        tp_rank: int = 0,
        tp_size: int = 1,
        pp_rank: int = 0,
        pp_size: int = 1,
        total_layers: Optional[int] = None,
        prefetch_layers: int = 1,
        print_debug: bool = False,
        allowed_targets: Optional[Set[str]] = None,
    ):
        """
        Initialize the loader.

        Args:
            model: Megatron model (unwrapped, should be the decoder/GPTModel)
            comp_dir: Base path to compressed weights directory.
                      For multi-GPU: comp_dir/tp{N}_pp{M}/ is used when it exists.
                      Falls back to comp_dir/tp{N}/ then comp_dir/ (single-GPU legacy).
            device: Target CUDA device
            tp_rank: Tensor parallel rank of this process
            tp_size: Tensor parallel world size
            pp_rank: Pipeline parallel rank of this process
            pp_size: Pipeline parallel world size
            total_layers: Total number of transformer layers in the full model
                          (used to compute PP layer offset).  Required when pp_size > 1.
            prefetch_layers: Number of layers to prefetch ahead
            print_debug: Print debug messages
            allowed_targets: Optional Megatron target filter.
        """
        self.model = model
        self.comp_dir = comp_dir
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.total_layers = total_layers
        self.prefetch_layers = prefetch_layers
        self.print_debug = print_debug
        self.allowed_targets = allowed_targets

        # PP layer offset: local layer i on this rank = global layer (pp_offset + i)
        if pp_size > 1 and total_layers is not None:
            self.pp_layer_offset = pp_rank * (total_layers // pp_size)
        else:
            self.pp_layer_offset = 0

        # Resolve effective compressed-weight directory for this TP/PP rank
        from flagscale.compress.memrift.parallel_state_utils import resolve_comp_dir
        self._effective_comp_dir = resolve_comp_dir(comp_dir, tp_rank, pp_rank)

        # Detect shard mode: index uses Megatron-format names (already merged & sharded)
        from flagscale.compress.memrift.parallel_state_utils import is_shard_index
        _idx_path = os.path.join(self._effective_comp_dir, "index.json")
        self._shard_mode: bool = is_shard_index(_idx_path)

        # Relax hook-side hard sync to reduce main-thread stalls.
        strict_sync = os.environ.get("MEMRIFT_WEIGHT_SYNC", "0") == "1"
        if os.environ.get("MEMRIFT_WEIGHT_RELAX_SYNC", "") == "1":
            strict_sync = False
        self._hook_materialize_sync = strict_sync

        # Load index from the effective (rank-specific) directory
        with open(_idx_path) as f:
            self.index = json.load(f)

        if print_debug:
            mode = "shard" if self._shard_mode else "HF"
            print(
                f"[MemRift] Init: tp={tp_rank}/{tp_size} pp={pp_rank}/{pp_size} "
                f"offset={self.pp_layer_offset} mode={mode} "
                f"dir={self._effective_comp_dir}"
            )

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
        self.released_params: Set[int] = set()

        # Async compressor reference
        self.async_compressor = None
        self._hook_warn_ms = _env_float("MEMRIFT_HOOK_WARN_MS", 800.0)

        # Backward empty_cache counter (aligned with memrift_demo)
        self._bwd_counter = 0
        self._bwd_empty_step = int(_env_float("MEMRIFT_BWD_EMPTY_STEP", 5))

    def _count_pending_prefetch(self, groups: List[MergedWeightGroup]) -> int:
        pending = 0
        for group in groups:
            for cp in group.components.values():
                fut_obj = cp._prefetch_future
                if fut_obj is not None and not fut_obj.done():
                    pending += 1
        return pending
    
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

        Shard mode  (comp_dir/tp{N}_pp{M}/index.json):
          - Names are Megatron-format: "decoder.layers.{i}.self_attention.linear_qkv.weight"
          - Weights are already merged (QKV cat'd, FC1 gate+up cat'd) and TP-sharded.
          - Layer indices are LOCAL to this PP rank (0 … L/PP-1).
          - No HF→Megatron mapping needed.

        HF mode (legacy, comp_dir/index.json):
          - Names are HuggingFace-format: "model.layers.{i}.self_attn.q_proj.weight"
          - QKV and FC1 components loaded separately and merged at materialize time.
          - TP=1, PP=1 only.
        """
        if self.print_debug:
            print(f"[MemRift] Loading weights from {self._effective_comp_dir} "
                  f"(shard_mode={self._shard_mode})")

        if self._shard_mode:
            self._load_weights_shard_mode()
        else:
            self._load_weights_hf_mode()

    def _load_weights_hf_mode(self):
        """HF-mode loading: legacy single-GPU path with HF→Megatron name mapping."""
        # First pass: count layers
        for entry in self.index:
            layer_idx = self._get_layer_idx(entry["name"])
            if layer_idx is not None:
                self.num_layers = max(self.num_layers, layer_idx + 1)

        self.layer_names = [f"decoder.layers.{i}" for i in range(self.num_layers)]

        if self.print_debug:
            print(f"[MemRift] HF mode: {self.num_layers} layers")

        for entry in self.index:
            if entry["scheme"] != "split_zstd":
                continue

            hf_name = entry["name"]
            layer_idx = self._get_layer_idx(hf_name)
            megatron_target, merge_key = self._map_hf_to_megatron(hf_name)

            file_path = os.path.join(self._effective_comp_dir, entry["file"])
            cp = self._read_compressed_file(file_path, entry)
            cp.hf_name = hf_name
            cp.megatron_target = megatron_target or ""
            cp.merge_key = merge_key
            cp.layer_idx = layer_idx if layer_idx is not None else -1

            self.all_cps.append(cp)

            if layer_idx is not None and megatron_target:
                if megatron_target not in self.merged_groups[layer_idx]:
                    self.merged_groups[layer_idx][megatron_target] = MergedWeightGroup(
                        megatron_target=megatron_target,
                        layer_idx=layer_idx,
                    )
                group = self.merged_groups[layer_idx][megatron_target]
                group.components[merge_key if merge_key else "single"] = cp
                if self.print_debug:
                    print(f"[MemRift] HF {hf_name} → layer {layer_idx} / "
                          f"{megatron_target} / {merge_key or 'single'}")
            else:
                self.non_layer_cps.append(cp)

        # Validate
        for layer_idx, groups in self.merged_groups.items():
            for target, group in groups.items():
                if not group.is_complete():
                    missing = (
                        [k for k in ["q", "k", "v"] if k not in group.components]
                        if "linear_qkv" in target else
                        [k for k in ["gate", "up"] if k not in group.components]
                    )
                    print(f"[MemRift] Warning: Incomplete group layer {layer_idx} "
                          f"/ {target}, missing: {missing}")

        if self.print_debug:
            print(f"[MemRift] Loaded {len(self.all_cps)} compressed params (HF mode)")

    def _load_weights_shard_mode(self):
        """
        Shard-mode loading: Megatron-format names, weights already merged & TP-sharded.

        Name format: "decoder.layers.{local_i}.{megatron_target}.weight"
        e.g.  "decoder.layers.0.self_attention.linear_qkv.weight"
              "decoder.layers.0.mlp.linear_fc1.weight"

        Each entry is a single 'single'-component MergedWeightGroup — no merging needed.
        """
        # First pass: count local layers on this PP rank
        for entry in self.index:
            layer_idx = self._get_layer_idx(entry["name"])
            if layer_idx is not None:
                self.num_layers = max(self.num_layers, layer_idx + 1)

        self.layer_names = [f"decoder.layers.{i}" for i in range(self.num_layers)]

        if self.print_debug:
            print(f"[MemRift] Shard mode: {self.num_layers} local layers "
                  f"(global offset {self.pp_layer_offset})")

        for entry in self.index:
            if entry["scheme"] != "split_zstd":
                continue

            param_name = entry["name"]  # Megatron-format
            layer_idx = self._get_layer_idx(param_name)

            file_path = os.path.join(self._effective_comp_dir, entry["file"])
            cp = self._read_compressed_file(file_path, entry)
            cp.hf_name = param_name   # reuse hf_name field for storage
            cp.layer_idx = layer_idx if layer_idx is not None else -1

            self.all_cps.append(cp)

            if layer_idx is not None:
                # Extract megatron_target: everything between "decoder.layers.{i}."
                # and ".weight" (or end of string)
                prefix = f"decoder.layers.{layer_idx}."
                if param_name.startswith(prefix):
                    remainder = param_name[len(prefix):]
                    # Strip trailing ".weight" or ".bias"
                    for suffix in (".weight", ".bias"):
                        if remainder.endswith(suffix):
                            remainder = remainder[: -len(suffix)]
                            break
                    megatron_target = remainder
                else:
                    megatron_target = param_name

                cp.megatron_target = megatron_target
                cp.merge_key = None

                if megatron_target not in self.merged_groups[layer_idx]:
                    self.merged_groups[layer_idx][megatron_target] = MergedWeightGroup(
                        megatron_target=megatron_target,
                        layer_idx=layer_idx,
                    )
                self.merged_groups[layer_idx][megatron_target].components["single"] = cp

                if self.print_debug:
                    print(f"[MemRift] Shard {param_name} → layer {layer_idx} / {megatron_target}")
            else:
                # Non-layer param (embed, norm, lm_head)
                self.non_layer_cps.append(cp)

        if self.print_debug:
            print(f"[MemRift] Loaded {len(self.all_cps)} compressed params (shard mode)")

    def _read_compressed_file(self, file_path: str, entry: dict) -> "CompressedParam":
        """Read one split_zstd file and return an unbound CompressedParam."""
        with open(file_path, "rb") as f:
            numel = struct.unpack("<Q", f.read(8))[0]
            sm_size = numel * (1 if entry["dtype"] == "bfloat16" else 3)
            sm_bytes = np.frombuffer(f.read(sm_size), dtype=np.uint8)
            exp_bytes = f.read()
        sm_gpu = torch.tensor(sm_bytes, dtype=torch.uint8, device=self.device)
        dtype = torch.bfloat16 if entry["dtype"] == "bfloat16" else torch.float32
        return CompressedParam(entry["shape"], sm_gpu, exp_bytes, dtype, self.device)
    
    def build_param_mapping(self):
        """
        Build mapping from MergedWeightGroup to actual model parameters.

        In shard mode the index uses LOCAL layer indices (0 … L/PP-1) which map
        directly to decoder_layers[i] on this PP rank.  No offset translation
        is needed inside this method — the index was generated against the same
        local model.

        In HF mode (TP=1, PP=1) behaviour is identical to the original.
        """
        if self.print_debug:
            print("[MemRift] Building param mapping...")

        # Find decoder layers
        decoder_layers = None
        for name, module in self.model.named_modules():
            if hasattr(module, "layers") and isinstance(module.layers, nn.ModuleList):
                decoder_layers = module.layers
                if self.print_debug:
                    print(f"[MemRift] Found decoder layers at: {name}.layers "
                          f"({len(decoder_layers)} layers on this PP rank)")
                break

        if decoder_layers is None:
            if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
                decoder_layers = self.model.decoder.layers
            elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "decoder"):
                decoder_layers = self.model.language_model.decoder.layers
            elif hasattr(self.model, "module"):
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

        # Non-layer params (embed, lm_head, norm) are never dynamically loaded via
        # hooks — their weights stay in param.data permanently.  Free their compressed
        # form (_sm_gpu + exp_mv) since it will never be used for decompression.
        for cp in self.non_layer_cps:
            cp.release_compressed()

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
    
    def _set_param(self, group: MergedWeightGroup, weight: torch.Tensor):
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
        for cp in group.components.values():
            if cp._CtoD_evt is not None:
                cur_stream.wait_event(cp._CtoD_evt)
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

    def release_all_layers(self):
        """Force-release every layer's materialized weights and cp._bf16.

        Used as post-backward cleanup: per-linear bwd_pre fallback hooks can
        re-materialize weights AFTER layer bwd_post clears them, leaving 32
        layers pinned at end of iteration. Calling this after the autograd
        graph is released frees them all.
        """
        for layer_name in self.layer_names:
            self._release_layer(layer_name)

    # ── Non-layer weight helpers ─────────────────────────────────────────────

    def _navigate(self, root: nn.Module, path: str):
        """Navigate to a sub-module or parameter by dot-separated path."""
        obj = root
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    def _get_decoder_layers_for_norms(self) -> Optional[nn.ModuleList]:
        """Return the decoder ModuleList (used for per-layer norm writes)."""
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
            return self.model.decoder.layers
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "decoder"):
            return self.model.language_model.decoder.layers
        for _name, m in self.model.named_modules():
            if hasattr(m, "layers") and isinstance(m.layers, nn.ModuleList):
                return m.layers
        return None

    def materialize_non_layer_weights(self) -> int:
        """
        Decompress and write all non-layer weights into the model.

        This includes:
        1. Global params: embed_tokens, final norm, lm_head
        2. Per-layer norms: input_layernorm, post_attention_layernorm
           (both TE-fused and non-TE paths are tried)

        Call this **before** release_original_weights() so that the model has
        correct weights for inference — the compressed directory is the sole
        weight source when no Megatron checkpoint is loaded via --load.

        Returns: number of parameters successfully written.
        """
        written = 0
        decoder_layers = self._get_decoder_layers_for_norms()

        for cp in self.non_layer_cps:
            hf_name = cp.hf_name
            written_this = False

            # 1) Global non-layer mapping (embed_tokens, final norm, lm_head)
            hints = self.NON_LAYER_HF_TO_MEGATRON_HINTS.get(hf_name)
            if hints:
                for hint_path in hints:
                    target = self._navigate(self.model, hint_path)
                    if target is not None and isinstance(target, torch.Tensor):
                        weight = cp.materialize(sync=True)
                        with torch.no_grad():
                            target.data = weight.to(device=target.device, dtype=target.dtype)
                        cp.release()
                        written += 1
                        written_this = True
                        _trace(f"non-layer: {hf_name} → {hint_path}")
                        break

            # 2) Per-layer norm mapping
            if not written_this and decoder_layers is not None:
                layer_idx = cp.layer_idx
                if 0 <= layer_idx < len(decoder_layers):
                    layer = decoder_layers[layer_idx]
                    for norm_suffix, mg_paths in self.LAYER_NORM_SUFFIX_TO_MEGATRON.items():
                        if not hf_name.endswith(norm_suffix):
                            continue
                        for mg_path in mg_paths:
                            parts = mg_path.rsplit(".", 1)
                            attr = parts[-1]
                            mod = self._navigate(layer, parts[0]) if len(parts) == 2 else layer
                            if mod is None:
                                continue
                            target = getattr(mod, attr, None)
                            if isinstance(target, torch.Tensor):
                                weight = cp.materialize(sync=True)
                                with torch.no_grad():
                                    target.data = weight.to(
                                        device=target.device, dtype=target.dtype
                                    )
                                cp.release()
                                written += 1
                                written_this = True
                                _trace(
                                    f"layer norm: {hf_name} → "
                                    f"decoder.layers.{layer_idx}.{mg_path}"
                                )
                                break
                        if written_this:
                            break

            if not written_this and self.print_debug:
                print(f"[MemRift] materialize_non_layer: no target found for {hf_name!r}")

        if self.print_debug:
            print(
                f"[MemRift] materialize_non_layer_weights: "
                f"wrote {written}/{len(self.non_layer_cps)} params"
            )
        return written

    def prefetch_initial_layers(self):
        """
        Pre-materialize first K layers and write back to param.data (TE compatibility).
        
        After release_original_weights(), all param.data are empty; TE would fail
        on first forward due to shape checks. This fills param.data for the first
        prefetch_layers+1 layers so the first forward sees valid shapes.
        """
        # Only pre-fill layer 0: TE needs non-empty weight shapes before the first
        # forward, but materializing prefetch_layers+1 layers at once wastes GPU memory.
        k = 1
        if self.print_debug:
            print(f"[MemRift] Pre-materializing first {k} layer(s) (param.data write-back for TE)")
        
        for i in range(k):
            if i < len(self.layer_names):
                layer_name = self.layer_names[i]
                self._materialize_and_set_layer(layer_name)
        
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
        
        # Forward hooks
        for i in range(len(self.layer_names)):
            cur = self.layer_names[i]
            span = 1  # prefetch only 1 layer ahead to bound peak memory
            nxt_names = self.layer_names[i + 1 : min(len(self.layer_names), i + 1 + span)]
            
            if cur not in name2layer:
                continue
            
            layer_module = name2layer[cur]
            cur_groups = self.layer2groups.get(cur, [])
            nxt_groups_list = [self.layer2groups.get(nm, []) for nm in nxt_names]
            
            # Capture variables
            def make_fwd_pre(cur_groups, nxt_groups_list, cur_name, async_comp):
                def _hook(mod, inp):
                    t0 = time.perf_counter()
                    _trace(f"fwd_pre: enter layer={cur_name}, groups={len(cur_groups)}")
                    pending_cur = self._count_pending_prefetch(cur_groups)
                    if pending_cur > 0:
                        _trace(f"fwd_pre: layer={cur_name} pending_prefetch_cur={pending_cur}")

                    # 1) Submit prefetch for NEXT K layers FIRST so CPU decompression
                    #    can overlap with current layer materialize + forward compute.
                    #    K is controlled by prefetch_layers.
                    if async_comp and nxt_groups_list:
                        for nxt_groups in nxt_groups_list:
                            for group in nxt_groups:
                                for cp in group.components.values():
                                    if cp._bf16 is None and cp._prefetch_future is None:
                                        cp._prefetch_future = async_comp.materialize_async(
                                            cp.exp_mv, cp._sm_gpu, cp.orig_shape, cp._dtype
                                        )

                    # 2) Materialize current layer and write back to model param.
                    #    If the layer was pre-fetched, cp.materialize() consumes the future.
                    #    Fast path in _materialize_group_tensor: if param.data is already
                    #    valid (from prefetch_initial_layers) it returns it directly.
                    for group in cur_groups:
                        tg = time.perf_counter()
                        weight = self._materialize_group(
                            group, sync=self._hook_materialize_sync
                        )
                        self._set_param(group, weight)
                        group_ms = (time.perf_counter() - tg) * 1000.0
                        if group_ms > self._hook_warn_ms:
                            _trace(
                                f"WARN fwd_pre: slow group materialize {group_ms:.1f} ms "
                                f"(layer={cur_name}, target={group.megatron_target})"
                            )

                    hook_ms = (time.perf_counter() - t0) * 1000.0
                    if hook_ms > self._hook_warn_ms:
                        _trace(f"WARN fwd_pre: slow hook {hook_ms:.1f} ms (layer={cur_name})")
                    _trace(f"fwd_pre: done layer={cur_name}")
                return _hook
            
            def make_fwd_post(cur_groups):
                def _hook(mod, inp, out):
                    # Always release; backward_pre re-materializes on demand
                    for group in cur_groups:
                        self._clear_param(group)
                return _hook

            layer_module.register_forward_pre_hook(
                make_fwd_pre(cur_groups, nxt_groups_list, cur, async_compressor)
            )
            layer_module.register_forward_hook(
                make_fwd_post(cur_groups)
            )
        
        # Backward hooks (reverse order)
        for i in range(len(self.layer_names) - 1, -1, -1):
            cur = self.layer_names[i]
            span = 1  # prefetch only 1 layer behind to bound peak memory
            prv_names = self.layer_names[max(0, i - span) : i][::-1]
            
            if cur not in name2layer:
                continue
            
            layer_module = name2layer[cur]
            cur_groups = self.layer2groups.get(cur, [])
            prv_groups_list = [self.layer2groups.get(nm, []) for nm in prv_names]
            
            def make_bwd_pre(cur_groups, prv_groups_list, cur_name, async_comp):
                def _hook(mod, grad_out):
                    if os.environ.get("MEMRIFT_HOOK_ORDER", "0") == "1":
                        print(f"[HOOK_ORDER] bwd_pre  layer={cur_name}", flush=True)
                    t0 = time.perf_counter()
                    _trace(f"bwd_pre: enter layer={cur_name}, groups={len(cur_groups)}")
                    pending_cur = self._count_pending_prefetch(cur_groups)
                    if pending_cur > 0:
                        _trace(f"bwd_pre: layer={cur_name} pending_prefetch_cur={pending_cur}")

                    # 1) Submit prefetch for PREVIOUS K layers FIRST so CPU decompression
                    #    can overlap with current layer materialize + backward compute.
                    #    K is controlled by prefetch_layers.
                    if async_comp and prv_groups_list:
                        for prv_groups in prv_groups_list:
                            for group in prv_groups:
                                for cp in group.components.values():
                                    if cp._bf16 is None and cp._prefetch_future is None:
                                        cp._prefetch_future = async_comp.materialize_async(
                                            cp.exp_mv, cp._sm_gpu, cp.orig_shape, cp._dtype
                                        )

                    # 2) Materialize current layer and write back to param (backward uses it).
                    for group in cur_groups:
                        tg = time.perf_counter()
                        weight = self._materialize_group(
                            group, sync=self._hook_materialize_sync
                        )
                        self._set_param(group, weight)
                        group_ms = (time.perf_counter() - tg) * 1000.0
                        if group_ms > self._hook_warn_ms:
                            _trace(
                                f"WARN bwd_pre: slow group materialize {group_ms:.1f} ms "
                                f"(layer={cur_name}, target={group.megatron_target})"
                            )

                    hook_ms = (time.perf_counter() - t0) * 1000.0
                    if hook_ms > self._hook_warn_ms:
                        _trace(f"WARN bwd_pre: slow hook {hook_ms:.1f} ms (layer={cur_name})")
                    _trace(f"bwd_pre: done layer={cur_name}")
                return _hook
            
            def make_bwd_post(cur_groups, cur_name=cur):
                def _hook(mod, grad_in, grad_out):
                    if os.environ.get("MEMRIFT_HOOK_ORDER", "0") == "1":
                        print(f"[HOOK_ORDER] bwd_post layer={cur_name}", flush=True)
                    for group in cur_groups:
                        self._clear_param(group)
                    self._bwd_counter += 1
                    if self._bwd_counter % self._bwd_empty_step == 0:
                        torch.cuda.empty_cache()
                return _hook
            
            layer_module.register_full_backward_pre_hook(
                make_bwd_pre(cur_groups, prv_groups_list, cur, async_compressor)
            )
            layer_module.register_full_backward_hook(
                make_bwd_post(cur_groups)
            )

            # Per-linear hooks bound to a single group: ensures the materialized
            # weight is released as soon as THAT linear's backward truly finishes,
            # not when the layer module's full_backward_hook fires (which can
            # precede RowParallelLinear's deferred all-reduce backward Function).
            # Module->group: deduplicate by id(target_module) since multiple
            # weight groups might map to the same linear module (rare but safe).
            mod2group = {}
            for group in cur_groups:
                tm = getattr(group, "target_module", None)
                if tm is None:
                    continue
                mod2group.setdefault(id(tm), (tm, group))

            # TE fuses fc1+fc2 (and self-attn qkv+proj) into single autograd
            # Functions (e.g. _LayerNormMLP). Backward of these Functions reads
            # BOTH weights simultaneously. So when ANY RowParallel linear in the
            # layer fires lin_bwd_pre, we must materialize ALL the layer's groups,
            # and the corresponding lin_bwd_post clears them all. ColumnParallel
            # linears never independently fire their hooks (no standalone autograd
            # Node), so registering on them is a harmless no-op.
            def make_linear_bwd_pre_all(cur_groups, cur_name=cur):
                def _hook(mod, grad_out):
                    if os.environ.get("MEMRIFT_HOOK_ORDER", "0") == "1":
                        empty = True
                        try:
                            w = getattr(mod, "weight", None)
                            if w is not None and w.data.numel() > 0:
                                empty = False
                        except Exception:
                            pass
                        print(f"[HOOK_ORDER] lin_bwd_pre layer={cur_name} mod={type(mod).__name__} empty={empty}", flush=True)
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
                        self._set_param(group, weight)
                return _hook

            def make_linear_bwd_post_all(cur_groups, cur_name=cur):
                def _hook(mod, grad_in, grad_out):
                    if os.environ.get("MEMRIFT_HOOK_ORDER", "0") == "1":
                        print(f"[HOOK_ORDER] lin_bwd_post layer={cur_name} mod={type(mod).__name__}", flush=True)
                    for group in cur_groups:
                        self._clear_param(group)
                return _hook

            if os.environ.get("MEMRIFT_DISABLE_LINEAR_BWD_PRE", "0") != "1":
                lpre = make_linear_bwd_pre_all(cur_groups)
                lpost = make_linear_bwd_post_all(cur_groups)
                for tm, _g in mod2group.values():
                    tm.register_full_backward_pre_hook(lpre)
                    tm.register_full_backward_hook(lpost)
        
        if self.print_debug:
            print(f"[MemRift] Installed hooks for {len(self.layer_names)} layers")

    def install_inference_hooks(self, async_compressor=None):
        """
        Forward-only hooks for inference (no backward pass, no activation compression).

        Weight lifecycle per token:
          forward_pre  → decompress current layer (consumes prefetch future or sync),
                         write to param.data, submit prefetch for next layer
          forward_post → release current layer (GPU memory freed immediately)

        GPU peak: ~1 layer resident at a time (~400 MB for Mistral-7B), identical
        to the training forward pattern but without backward re-materialization.
        """
        self.async_compressor = async_compressor

        # ── find layer modules (same logic as install_hooks) ──
        name2layer = {}
        for name, module in self.model.named_modules():
            name2layer[name] = module

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
            print(f"[MemRift] install_inference_hooks: found {len(found)}/{len(self.layer_names)} layers")

        # ── forward hooks only (no backward hooks) ──
        for i in range(len(self.layer_names)):
            cur = self.layer_names[i]
            span = 1
            nxt_names = self.layer_names[i + 1 : min(len(self.layer_names), i + 1 + span)]

            if cur not in name2layer:
                continue

            layer_module = name2layer[cur]
            cur_groups = self.layer2groups.get(cur, [])
            nxt_groups_list = [self.layer2groups.get(nm, []) for nm in nxt_names]

            def make_fwd_pre_infer(cur_groups, nxt_groups_list, cur_name, async_comp):
                def _hook(mod, inp):
                    t0 = time.perf_counter()
                    _trace(f"infer fwd_pre: layer={cur_name}")

                    # 1) Submit prefetch for next layer before materializing current,
                    #    so CPU decompression overlaps with GPU compute.
                    if async_comp and nxt_groups_list:
                        for nxt_groups in nxt_groups_list:
                            for group in nxt_groups:
                                for cp in group.components.values():
                                    if cp._bf16 is None and cp._prefetch_future is None:
                                        cp._prefetch_future = async_comp.materialize_async(
                                            cp.exp_mv, cp._sm_gpu, cp.orig_shape, cp._dtype
                                        )

                    # 2) Materialize current layer (consume prefetch or sync decompress).
                    for group in cur_groups:
                        weight = self._materialize_group(
                            group, sync=self._hook_materialize_sync
                        )
                        self._set_param(group, weight)

                    hook_ms = (time.perf_counter() - t0) * 1000.0
                    if hook_ms > self._hook_warn_ms:
                        _trace(f"WARN infer fwd_pre: slow {hook_ms:.1f} ms (layer={cur_name})")
                return _hook

            def make_fwd_post_infer(cur_groups):
                def _hook(mod, inp, out):
                    # Release current layer immediately — next token's forward_pre
                    # will re-materialize (or consume the prefetch future).
                    for group in cur_groups:
                        self._clear_param(group)
                return _hook

            layer_module.register_forward_pre_hook(
                make_fwd_pre_infer(cur_groups, nxt_groups_list, cur, async_compressor)
            )
            layer_module.register_forward_hook(
                make_fwd_post_infer(cur_groups)
            )
            # No backward hooks registered.

        if self.print_debug:
            print(f"[MemRift] Installed inference-only hooks for {len(self.layer_names)} layers")

    def reset(self):
        if self.async_compressor is not None:
            self.async_compressor.reset()
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics for monitoring."""
        sm_gpu_bytes = sum(cp._sm_gpu.numel() for cp in self.all_cps if cp._sm_gpu is not None)
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
