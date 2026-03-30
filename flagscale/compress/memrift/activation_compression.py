from contextlib import contextmanager
from typing import Optional, Set, Any
import functools
import os
import weakref
import time

import torch
import torch.nn as nn

from flagscale.compress.memrift import act_profile
from flagscale.compress.memrift import time_profiler as layer_time_profiler
from flagscale.compress.memrift.async_compressor import PlaceHolderToken, AsyncCompressor


def _should_compress_activation(t: torch.Tensor, skip_storage_ptrs: Optional[Set[int]] = None) -> bool:
    if not t.is_cuda or t.numel() == 0:
        return False
    if t.dtype not in (torch.float32, torch.bfloat16):
        return False
    if t.is_leaf or (not t.requires_grad):
        return False
    if skip_storage_ptrs:
        try:
            if t.untyped_storage().data_ptr() in skip_storage_ptrs:
                return False
        except Exception:
            pass
    return True


def _unpack(tok: Any, compressor: AsyncCompressor):
    if isinstance(tok, PlaceHolderToken):
        tok_act_async = bool(getattr(tok, "act_async", False))
        if tok_act_async and compressor.enable_async:
            if tok.decomped_data is not None and tok.CtoD_copy_evt is None:
                return tok.decomped_data
            # Fallback: if bwd_pre_hook did not schedule async decode yet, do it here.
            if tok.decomped_data is None and getattr(tok, "future", None) is not None:
                compressor.decompress_async(tok, tok.future)
                tok.future = None
            import os as _os
            _dbg = _os.environ.get("MEMRIFT_ACT_TIME", "0") == "1"
            _prof = act_profile.is_enabled()

            # 优化：优先使用 GPU 事件同步，避免 CPU 侧阻塞
            training_stream = torch.cuda.current_stream()
            if tok.CtoD_copy_evt is not None and tok.decomped_data is not None:
                # 直接使用 GPU 事件等待，不阻塞 CPU
                training_stream.wait_event(tok.CtoD_copy_evt)
                tok.decomped_data.record_stream(training_stream)
                tok._clear_after_recover()
                return tok.decomped_data

            # 进一步优化：检查是否可以非阻塞地检查 ready_evt，
            # 如果不行，则让 GPU 事件来处理同步
            use_gpu_only_wait = _os.environ.get("MEMRIFT_GPU_ONLY_WAIT", "1") == "1"
            if use_gpu_only_wait and tok.CtoD_copy_evt is not None:
                # 直接跳过 CPU 等待，完全依赖 GPU 事件同步
                # 这样 CPU 可以继续执行其他任务
                training_stream.wait_event(tok.CtoD_copy_evt)
                if tok.decomped_data is not None:
                    tok.decomped_data.record_stream(training_stream)
                    tok._clear_after_recover()
                    return tok.decomped_data

            # 只有在没有 CtoD_copy_evt 时才等待 ready_evt
            _w0 = time.perf_counter()
            tok.ready_evt.wait()
            _w1 = time.perf_counter()
            if _prof:
                act_profile.add_ms("unpack_ready_evt_wait_ms", 1000.0 * (_w1 - _w0))
            if layer_time_profiler.is_enabled():
                layer_time_profiler.add_time(
                    getattr(tok, "layer_name", "unknown"),
                    "activation_wait_ms",
                    1000.0 * (_w1 - _w0),
                )
            # GPU-side ordering: training stream waits for H2D to complete.
            if tok.CtoD_copy_evt is not None:
                _wu0 = time.perf_counter()
                training_stream.wait_event(tok.CtoD_copy_evt)
                _wu1 = time.perf_counter()
                if _prof:
                    act_profile.add_ms("unpack_wait_event_cpu_ms", 1000.0 * (_wu1 - _wu0))
            if tok.decomped_data is None:
                raise RuntimeError(
                    f"MemRift activation decode did not materialize tensor for shape={tok.shape}. "
                    "Consider reducing activation decode concurrency or disabling look-ahead decode."
                )
            tok.decomped_data.record_stream(training_stream)
            tok._clear_after_recover()
            if _prof:
                act_profile.maybe_log_rolling("act")
            if _dbg and (_w1 - _w0) > 0.001:
                print(f"[ACT_UNPACK_WAIT] {1000*(_w1-_w0):.2f}ms shape={tok.shape}", flush=True)
        else:
            if tok.decomped_data is None:
                compressor.decompress_sync(tok)
                tok.decomped_data.record_stream(torch.cuda.current_stream())
        return tok.decomped_data
    return tok


class DecoderLayerWrapper(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        compressor: AsyncCompressor,
        use_async: bool,
        release_after_unpack: bool = True,
        skip_storage_ptrs: Optional[Set[int]] = None,
        do_empty: bool = False,
        layer_name: str = "",
    ):
        super().__init__()
        self.layer = layer
        self.comp = compressor
        self.use_async = use_async
        self.release_after_unpack = release_after_unpack
        self.skip_storage_ptrs = skip_storage_ptrs or set()
        self.tokens = []
        self.futures = []
        self.do_empty = do_empty
        self.layer_name = layer_name or type(layer).__name__
        self._bwd_start_evt = None
        self._bwd_wall_start = None

        self.layer.register_full_backward_pre_hook(self._bwd_pre_hook)
        self.layer.register_full_backward_hook(self._bwd_hook)
        # Register with compressor for global H2D pre-submit
        self.comp._layer_wrappers.append(self)

    def forward(self, *inp, **kw):
        self.tokens.clear()
        self.futures.clear()
        seen = {}

        def _pack(t):
            if not _should_compress_activation(t, self.skip_storage_ptrs):
                return t

            key = (t.data_ptr(), t.nbytes)
            if key in seen:
                tok_ref, t_ref = seen[key]
                # 检查原始张量是否仍然是同一个对象（避免内存地址重用导致的冲突）
                if t_ref() is t:
                    tok = tok_ref()
                    if tok is not None:
                        return tok

            tok = PlaceHolderToken(t.dtype, t.shape, tuple(t.stride()), t.storage_offset())
            tok.act_async = bool(self.use_async)
            tok.layer_name = self.layer_name
            # 保存弱引用：弱引用不会阻止垃圾回收，同时可以检测重复的激活张量
            seen[key] = (weakref.ref(tok), weakref.ref(t))
            if self.use_async and self.comp.enable_async:
                fut = self.comp.kickoff_async(tok, t)
                self.futures.append(fut)
                tok.fut_id = len(self.futures) - 1
                tok.future = fut
            else:
                self.comp.kickoff_sync(tok, t)
            self.tokens.append(weakref.ref(tok))
            return tok

        unpack_fn = functools.partial(_unpack, compressor=self.comp)
        with torch.autograd.graph.saved_tensors_hooks(_pack, unpack_fn):
            fwd_start_evt = None
            if layer_time_profiler.is_enabled():
                fwd_start_evt = torch.cuda.Event(enable_timing=True)
                fwd_start_evt.record(torch.cuda.current_stream())
            out = self.layer(*inp, **kw)
            if fwd_start_evt is not None:
                fwd_end_evt = torch.cuda.Event(enable_timing=True)
                fwd_end_evt.record(torch.cuda.current_stream())
                fwd_end_evt.synchronize()
                layer_time_profiler.add_time(
                    self.layer_name,
                    "forward_compute_ms",
                    float(fwd_start_evt.elapsed_time(fwd_end_evt)),
                )

        # 关键优化：前向传播结束后立即清理 seen 字典，移除对 tok 的引用
        seen.clear()
        del seen

        # 优化：避免频繁调用 empty_cache()，这会导致 GPU 空闲，降低性能
        # if self.do_empty:
        #     torch.cuda.empty_cache()
        return out

    def _bwd_pre_hook(self, _mod, _grad_in):
        if not (self.use_async and self.comp.enable_async):
            return

        _prof = act_profile.is_enabled()
        _layer_prof = layer_time_profiler.is_enabled()
        _t_hook0 = time.perf_counter() if _prof else 0.0

        def _submit_for_wrapper(w):
            for tok_ptr in w.tokens[::-1]:
                tok = tok_ptr()
                if tok is None or tok.decomped_data is not None:
                    continue
                fut_idx = tok.fut_id
                if fut_idx < 0 or fut_idx >= len(w.futures) or w.futures[fut_idx] is None:
                    continue
                if _prof or _layer_prof:
                    # Training stream: mark the moment we enqueue activation restore work.
                    setattr(tok, "_prof_hook_evt", torch.cuda.current_stream().record_event())
                w.comp.decompress_async(tok, w.futures[fut_idx])
                w.futures[fut_idx] = None
                tok.future = None

        # 1. Submit H2D for THIS layer FIRST (highest priority for copy engine).
        _submit_for_wrapper(self)

        # 2. Optionally look ahead multiple backward layers to overlap decode with compute.
        # Large models can reduce this to lower decode-time memory pressure.
        if os.environ.get("MEMRIFT_ACT_LOOKAHEAD", "1") == "1":
            try:
                idx = self.comp._layer_wrappers.index(self)
                # 优化：根据 Nsight 分析结果，增加预取层数到默认4层
                # 这样可以更早开始解压，更好地重叠计算和解压
                lookahead_layers = int(os.environ.get("MEMRIFT_ACT_LOOKAHEAD_LAYERS", "4"))
                for i in range(1, lookahead_layers + 1):
                    if idx - i >= 0:
                        _submit_for_wrapper(self.comp._layer_wrappers[idx - i])
            except (ValueError, IndexError):
                pass

        if layer_time_profiler.is_enabled():
            self._bwd_start_evt = torch.cuda.Event(enable_timing=True)
            self._bwd_start_evt.record(torch.cuda.current_stream())
            self._bwd_wall_start = time.perf_counter()

        if _prof:
            act_profile.add_ms("bwd_pre_hook_total_ms", 1000.0 * (time.perf_counter() - _t_hook0))

    def _bwd_hook(self, _mod, _gin, _gout):
        if layer_time_profiler.is_enabled() and self._bwd_wall_start is not None:
            layer_time_profiler.add_time(
                self.layer_name,
                "backward_compute_ms",
                1000.0 * (time.perf_counter() - self._bwd_wall_start),
            )
            self._bwd_start_evt = None
            self._bwd_wall_start = None
        for tok_ptr in self.tokens:
            tok = tok_ptr()
            if tok is not None:
                del tok
        self.tokens.clear()
        self.futures.clear()


@contextmanager
def activation_compression_context(
    compressor: Optional[AsyncCompressor] = None,
    use_async: bool = False,
    release_after_unpack: bool = True,
    zstd_level: int = 18,
    skip_storage_ptrs: Optional[Set[int]] = None,
):
    if compressor is None:
        compressor = AsyncCompressor(
            compress_workers=2,
            decode_workers=2,
            concurrency_limit=4,
            zstd_level=zstd_level,
            enable_async=False,
        )

    seen = {}

    def pack(t):
        if not _should_compress_activation(t, skip_storage_ptrs):
            return t

        key = (t.data_ptr(), t.nbytes)
        if key in seen:
            tok_ref, t_ref = seen[key]
            if t_ref() is not None:
                tok = tok_ref()
                if tok is not None:
                    return tok

        tok = PlaceHolderToken(t.dtype, t.shape, tuple(t.stride()), t.storage_offset())
        tok.act_async = bool(use_async)
        seen[key] = (weakref.ref(tok), weakref.ref(t))
        if use_async and compressor.enable_async:
            fut = compressor.kickoff_async(tok, t)
            tok.fut_id = 0
            tok.future = fut
        else:
            compressor.kickoff_sync(tok, t)
        return tok

    unpack = functools.partial(_unpack, compressor=compressor)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield compressor
