import threading
import concurrent.futures as fut
from contextlib import nullcontext
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field

import numpy as np
import torch

try:
    _memrift_nvtx_range = torch.cuda.nvtx.range
except AttributeError:

    def _memrift_nvtx_range(name: str):
        return nullcontext()

try:
    from flagscale.compress.memrift import act_profile as _act_profile
except ImportError:
    _act_profile = None

try:
    from flagscale.compress.memrift import time_profiler as _layer_time_profiler
except ImportError:
    _layer_time_profiler = None


def _ap_add(name: str, ms: float) -> None:
    if _act_profile is None:
        return
    _act_profile.add_ms(name, ms)


def _ap_gpu_elapsed(start_evt: torch.cuda.Event, end_evt: torch.cuda.Event, name: str) -> None:
    if _act_profile is None or not _act_profile.is_enabled():
        return
    try:
        _ap_add(name, float(start_evt.elapsed_time(end_evt)))
    except Exception:
        pass


def _lt_add(layer_name: str, metric: str, ms: float) -> None:
    if _layer_time_profiler is None or not _layer_time_profiler.is_enabled():
        return
    try:
        _layer_time_profiler.add_time(layer_name, metric, ms)
    except Exception:
        pass


def _lt_enabled() -> bool:
    return _layer_time_profiler is not None and _layer_time_profiler.is_enabled()


def _c_contiguous_strides(shape):
    """Calculate C-contiguous strides for a shape."""
    strides = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 2, -1, -1):
        running *= shape[i + 1]
        strides[i] = running
    return tuple(strides)


def _get_prefetch_merge_meta(cp):
    """Cache per-parameter merge metadata used on every prefetch."""
    cached = getattr(cp, "_memrift_prefetch_merge_meta", None)
    if cached is not None:
        return cached
    shape_list = list(cp.orig_shape)
    stride_list = list(_c_contiguous_strides(cp.orig_shape))
    cached = (shape_list, stride_list)
    setattr(cp, "_memrift_prefetch_merge_meta", cached)
    return cached

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

# Format magic for nvCOMP LZ4 compressed exponent bytes.
# Prepended during offline compression (prepare_weight.py --compression nvcomp_lz4).
NVCOMP_LZ4_MAGIC = b"NVL4"

# Format magic for nvCOMP ANS compressed exponent bytes.
# ANS (Asymmetric Numeral Systems) achieves ~2.8x compression on real weight
# exponent bytes (vs LZ4's 1.0x) and GPU-decodes in ~0.7ms vs LZ4's 8.5ms.
# Weights are re-compressed at startup from LZ4 to ANS by MegatronDynamicLoader.
NVCOMP_ANS_MAGIC = b"NVAN"

# ---------------------------------------------------------------------------
# 设计说明：DLPack vs 静态 GPU 解码缓冲（Llama 3.1 8B 等固定 Shape）
# ---------------------------------------------------------------------------
# 当前 ``nvidia.nvcomp`` Python 绑定里 ``Codec.decode`` 返回自管 GPU 缓冲，
# 需 ``torch.from_dlpack`` 才能接到 PyTorch；该步骤会在 CPU 侧与 decode 流对齐。
# 若要 **彻底去掉 from_dlpack** 并实现「解压直写预分配 tensor.data_ptr()」的
# Zero-CPU-Sync 路径，需要下列之一：
#   (a) nvCOMP 暴露 decode-into-user-buffer 的 Python API；或
#   (b) 自定义 C++/CUDA 扩展：持有 ``torch.empty(..., device='cuda')`` 池，
#       调用 nvCOMP C API 写入 ``tensor.data_ptr()``，再交给 ``fs_sp.merge``。
# 在此之前，优化集中在：批量 decode、独立流、以及监控 ``pref.result()`` 等待。
# ---------------------------------------------------------------------------

_MEMRIFT_WEIGHT_NON_NVCOMP_ERR = (
    "MemRift: weight exponent payload is not nvCOMP GPU format (expected NVAN or NVL4 prefix). "
    "Use: python -m flagscale.compress.memrift.offline_comp.prepare_weight "
    "--compression nvcomp_ans (or nvcomp_lz4). "
    "Legacy CPU zstd split weights: set MEMRIFT_ALLOW_CPU_WEIGHT_ZSTD=1."
)


def memrift_allow_cpu_weight_zstd() -> bool:
    import os
    return os.environ.get("MEMRIFT_ALLOW_CPU_WEIGHT_ZSTD", "0") == "1"


def memrift_act_store_compressed_on_cpu() -> bool:
    """After GPU ANS encode, copy compressed activation to pinned CPU and free GPU blob (default on).

    Set ``MEMRIFT_ACT_STORE_CPU=0`` to keep compressed tensors on GPU (old behavior, higher VRAM).
    """
    import os
    return os.environ.get("MEMRIFT_ACT_STORE_CPU", "1") == "1"


def memrift_require_nvcomp_weight_payload(exp_mv: Any) -> None:
    """When nvCOMP is installed, reject raw zstd exponent blobs unless opt-in."""
    if not NVCOMP_AVAILABLE or memrift_allow_cpu_weight_zstd():
        return
    if exp_mv is None:
        raise RuntimeError(_MEMRIFT_WEIGHT_NON_NVCOMP_ERR)
    if not isinstance(exp_mv, (bytes, bytearray, memoryview)) or len(exp_mv) < 4:
        raise RuntimeError(_MEMRIFT_WEIGHT_NON_NVCOMP_ERR)
    if bytes(exp_mv[:4]) not in (NVCOMP_LZ4_MAGIC, NVCOMP_ANS_MAGIC):
        raise RuntimeError(_MEMRIFT_WEIGHT_NON_NVCOMP_ERR)


# Thread-local storage for compressor/decompressor contexts
_tls = threading.local()


def get_compression_ctx(level: int = 18):
    """Get thread-local zstd compression context."""
    if not hasattr(_tls, "cctx") or _tls.cctx_level != level:
        _tls.cctx = zstd.ZstdCompressor(level=level, write_checksum=False)
        _tls.cctx_level = level
    return _tls.cctx


def get_decompression_ctx():
    """Get thread-local zstd decompression context."""
    if not hasattr(_tls, "dctx"):
        _tls.dctx = zstd.ZstdDecompressor()
    return _tls.dctx


@dataclass
class PlaceHolderToken:
    """
    Placeholder for compressed activation tensor.
    
    Used with torch.autograd.graph.saved_tensors_hooks to replace
    large activation tensors with compressed representations.
    """
    dtype: torch.dtype
    shape: torch.Size
    stride: tuple
    offset: int
    
    # Runtime fields (filled during compression/decompression)
    sm_bits: Optional[torch.Tensor] = None
    comped_cpu_exp: Optional[bytes] = None
    numel: int = 0
    decomped_data: Optional[torch.Tensor] = None
    ready_evt: threading.Event = field(default_factory=threading.Event)
    CtoD_copy_evt: Optional[torch.cuda.Event] = None
    fut_id: int = -1
    
    def _clear_after_recover(self):
        """Clean up after decompression is complete."""
        if hasattr(self, "fut_id"):
            del self.fut_id
        self.ready_evt.clear()
        if hasattr(self, "CtoD_copy_evt") and self.CtoD_copy_evt is not None:
            del self.CtoD_copy_evt
            self.CtoD_copy_evt = None


class AsyncCompressor:
    """
    Async compression/decompression manager.

    Manages:
    - Thread pools for background encode/decode work
    - CUDA streams for H2D / nvCOMP / merge (weights) and activation ANS encode/decode
    - Semaphore for concurrency control

    Weight decompression (materialize / prefetch):
    - GPU nvCOMP when exponent payload has NVL4/NVAN magic (see ``materialize_async``).
    - Optional CPU zstd for legacy weights if ``MEMRIFT_ALLOW_CPU_WEIGHT_ZSTD=1``.

    Activation async path (``kickoff_async`` / ``decompress_async``):
    - GPU nvCOMP ANS only (requires ``nvidia.nvcomp``). No CPU pinned offload.

    Usage:
        compressor = AsyncCompressor(
            compress_workers=8,
            decode_workers=4,
            concurrency_limit=4,
            zstd_level=18
        )

        # Async compression
        fut = compressor.kickoff_async(token, tensor)

        # Async decompression
        compressor.decompress_async(token, fut)
    """

    def __init__(
        self,
        compress_workers: int = 16,
        decode_workers: int = 8,
        zstd_workers: int = -1,
        concurrency_limit: int = 16,  # 优化：提高默认并发限制
        zstd_level: int = 3,  # 保持较低的压缩级别以提高速度
        enable_async: bool = True,
    ):
        """
        Initialize AsyncCompressor.
        
        Args:
            compress_workers: Number of compression thread pool workers
            decode_workers: Number of decompression thread pool workers
            zstd_workers: CPU zstd weight path only; -1 = same as decode_workers.
                Use a smaller pool to isolate zstd from nvCOMP submit threads.
            concurrency_limit: Max concurrent operations (semaphore limit)
            zstd_level: Zstd compression level (1-22)
            enable_async: If False, use synchronous operations
        """
        if not ZSTD_AVAILABLE:
            raise RuntimeError("zstandard not available. Install with: pip install zstandard")
        
        self.zstd_level = zstd_level
        self.enable_async = enable_async
        import os as _os
        if zstd_workers < 0:
            _zw = _os.environ.get("MEMRIFT_ZSTD_POOL_WORKERS")
            if _zw is not None:
                try:
                    zstd_workers = int(_zw)
                except ValueError:
                    zstd_workers = decode_workers
            else:
                zstd_workers = decode_workers
        self._zstd_workers = max(1, int(zstd_workers))
        
        if enable_async:
            self.compress_pool = fut.ThreadPoolExecutor(compress_workers)
            self.decode_pool = fut.ThreadPoolExecutor(decode_workers)
            # 与 nvCOMP / GPU 提交线程池分离，避免 zstd 占满 decode_pool
            self.zstd_pool = fut.ThreadPoolExecutor(self._zstd_workers)
            self.decomp_semaphore = threading.Semaphore(value=concurrency_limit)
        else:
            self.zstd_pool = None
            self.cctx = zstd.ZstdCompressor(level=zstd_level, write_checksum=False)
            self.dctx = zstd.ZstdDecompressor()
        
        # CUDA streams – three-stream pipeline for GPU weight decompression:
        #   h2d_stream  : ONLY H2D transfers (compressed data pinned→GPU)
        #   decode_stream: ONLY nvCOMP decode (waits for H2D via event)
        #   merge_stream : clone + float-split-merge (waits for decode via from_dlpack CPU sync)
        # Each stream has minimal pending work, eliminating the chain of
        # implicit synchronisations that made the single-stream design slow.
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()
        self.decode_stream = torch.cuda.Stream()
        self.merge_stream = torch.cuda.Stream()

        # CUDA streams for GPU activation compression/decompression.
        # act_comp_stream: shared stream for GPU ANS encode (all encode workers serialize here;
        #   encode runs during forward when GPU is idle between layers, so serialization is OK).
        # act_decomp_stream: SENTINEL only – decode uses thread-local streams (_act_dec_tls)
        #   so that each decode worker thread uses its own CUDA stream. This prevents the
        #   "cascade sync" bug where dec_stream.synchronize() for one worker waits for all
        #   12+ other tasks queued by concurrent workers on the same stream.
        self.act_comp_stream = torch.cuda.Stream()   # sentinel / fallback
        self.act_decomp_stream = torch.cuda.Stream() # sentinel / fallback
        # Thread-local storage: each encode/decode worker thread gets its own
        # CUDA stream and codec instance created lazily on first use.
        self._act_enc_tls = threading.local()
        self._act_dec_tls = threading.local()
        
        # nvCOMP codecs for GPU-side decompression.
        # _nvcomp_codec: LZ4, bound to h2d_stream (fallback / materialize_on_stream).
        # _nvcomp_decode_codec: ANS, bound to decode_stream (prefetch_batch_on_stream).
        # _nvcomp_ans_codec: ANS, bound to h2d_stream (startup re-compression + decode).
        # _nvcomp_act_enc_codec: ANS encode for activations (act_comp_stream).
        # _nvcomp_act_dec_codec: ANS decode for activations (act_decomp_stream).
        # Note: actual encode/decode for activations uses per-task codecs (thread-safe);
        #       these sentinel objects only check whether the nvCOMP path is available.
        self._nvcomp_codec = None
        self._nvcomp_decode_codec = None
        self._nvcomp_ans_codec = None
        self._nvcomp_act_enc_codec = None
        self._nvcomp_act_dec_codec = None
        self._nvcomp_lock = threading.Lock()  # guard codec creation
        if NVCOMP_AVAILABLE:
            try:
                self._nvcomp_codec = _nvcomp_lib.Codec(
                    algorithm='lz4',
                    cuda_stream=self.h2d_stream.cuda_stream,
                )
                self._nvcomp_decode_codec = _nvcomp_lib.Codec(
                    algorithm='ans',
                    cuda_stream=self.decode_stream.cuda_stream,
                )
                self._nvcomp_ans_codec = _nvcomp_lib.Codec(
                    algorithm='ans',
                    cuda_stream=self.h2d_stream.cuda_stream,
                )
                # Sentinel objects (algorithm=ans) – signals GPU-ANS activation path available.
                self._nvcomp_act_enc_codec = _nvcomp_lib.Codec(
                    algorithm='ans',
                    cuda_stream=self.act_comp_stream.cuda_stream,
                )
                self._nvcomp_act_dec_codec = _nvcomp_lib.Codec(
                    algorithm='ans',
                    cuda_stream=self.act_decomp_stream.cuda_stream,
                )
            except Exception:
                self._nvcomp_codec = None
                self._nvcomp_decode_codec = None
                self._nvcomp_ans_codec = None
                self._nvcomp_act_enc_codec = None
                self._nvcomp_act_dec_codec = None
        
        # Try to import CUDA extension
        try:
            from flagscale.compress.float_split_stride_pin import float_split_stride_pin as fs_sp
            self._fs_sp = fs_sp if fs_sp.is_available() else None
        except ImportError:
            self._fs_sp = None

        # Background decode threads (started lazily on first use).
        # Uses dedicated CUDA streams / nvcomp codec per thread so no stream races.
        # Controlled by MEMRIFT_BG_DECODE_THREADS (default=1).
        self._bg_num_threads: int = max(1, int(
            __import__('os').environ.get("MEMRIFT_BG_DECODE_THREADS", "1")
        ))
        self._bg_queue: Optional[Any] = None
        self._bg_threads: list = []

        # Registry for DecoderLayerWrapper instances (look-ahead decode submit in bwd_pre_hook).
        self._layer_wrappers: list = []

        # GPU staging buffer pool: reuse GPU uint8 tensors for H2D copies to
        # avoid repeated cudaMallocAsync.  Keyed by exact byte size.
        # Thread-safe via _gpu_pool_lock.
        self._gpu_pool: Dict[tuple, list] = {}
        self._gpu_pool_lock = threading.Lock()
        self._gpu_pool_enabled = __import__('os').environ.get(
            "MEMRIFT_GPU_STAGING_POOL", "1") == "1"

    # -- GPU staging buffer pool helpers --

    def _acquire_gpu_staging(self, size: int, device: torch.device) -> torch.Tensor:
        """Get a GPU uint8 tensor from the pool (or allocate a new one)."""
        if not self._gpu_pool_enabled:
            return torch.empty(size, dtype=torch.uint8, device=device)
        key = (device.index, size)
        with self._gpu_pool_lock:
            pool = self._gpu_pool.get(key)
            if pool:
                return pool.pop()
        return torch.empty(size, dtype=torch.uint8, device=device)

    def _return_gpu_staging(self, buf: torch.Tensor) -> None:
        """Return a buffer to the pool for future reuse."""
        if not self._gpu_pool_enabled:
            return
        key = (buf.device.index, buf.numel())
        with self._gpu_pool_lock:
            pool = self._gpu_pool.setdefault(key, [])
            if len(pool) < 64:
                pool.append(buf)

    def _build(self):
        """Rebuild pools and streams (used after reset)."""
        if self.enable_async:
            self.compress_pool = fut.ThreadPoolExecutor(16)  # activation GPU-encode workers
            self.decode_pool = fut.ThreadPoolExecutor(4)     # weight materialize + act decode workers
            self.zstd_pool = fut.ThreadPoolExecutor(max(1, self._zstd_workers))
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()
        self.decode_stream = torch.cuda.Stream()
        self.merge_stream = torch.cuda.Stream()
        self.act_comp_stream = torch.cuda.Stream()
        self.act_decomp_stream = torch.cuda.Stream()
        # Recreate nvcomp codecs bound to the new streams
        self._nvcomp_codec = None
        self._nvcomp_decode_codec = None
        self._nvcomp_ans_codec = None
        self._nvcomp_act_enc_codec = None
        self._nvcomp_act_dec_codec = None
        if NVCOMP_AVAILABLE:
            try:
                self._nvcomp_codec = _nvcomp_lib.Codec(
                    algorithm='lz4',
                    cuda_stream=self.h2d_stream.cuda_stream,
                )
                self._nvcomp_decode_codec = _nvcomp_lib.Codec(
                    algorithm='ans',
                    cuda_stream=self.decode_stream.cuda_stream,
                )
                self._nvcomp_ans_codec = _nvcomp_lib.Codec(
                    algorithm='ans',
                    cuda_stream=self.h2d_stream.cuda_stream,
                )
                self._nvcomp_act_enc_codec = _nvcomp_lib.Codec(
                    algorithm='ans',
                    cuda_stream=self.act_comp_stream.cuda_stream,
                )
                self._nvcomp_act_dec_codec = _nvcomp_lib.Codec(
                    algorithm='ans',
                    cuda_stream=self.act_decomp_stream.cuda_stream,
                )
            except Exception:
                pass
        # Reset bg thread fields (will be re-created lazily if needed)
        self._bg_queue = None
        self._bg_threads = []

    def reset(self):
        """Reset pools and streams (call between training rounds)."""
        if self.enable_async:
            self.compress_pool.shutdown(wait=True)
            self.decode_pool.shutdown(wait=True)
            if getattr(self, "zstd_pool", None) is not None:
                self.zstd_pool.shutdown(wait=True)
            del self.compress_pool
            del self.decode_pool
            del self.zstd_pool
        # Gracefully stop existing bg decode threads before rebuilding.
        if self._bg_queue is not None:
            for _ in self._bg_threads:
                self._bg_queue.put(None)
        for t in self._bg_threads:
            t.join(timeout=5.0)
        self._build()
        torch.cuda.reset_peak_memory_stats()
    
    def shutdown(self):
        """Shutdown all pools and background thread."""
        if self.enable_async:
            self.compress_pool.shutdown(wait=True)
            self.decode_pool.shutdown(wait=True)
            if getattr(self, "zstd_pool", None) is not None:
                self.zstd_pool.shutdown(wait=True)
        # Stop background decode threads (one sentinel per thread)
        if self._bg_queue is not None:
            for _ in self._bg_threads:
                self._bg_queue.put(None)
        for t in self._bg_threads:
            t.join(timeout=5.0)
        self._bg_threads = []

    # -------------------------------------------------------------------------
    #  Background decode thread (non-blocking prefetch)
    # -------------------------------------------------------------------------

    def _ensure_bg_threads(self):
        """Lazily start N background decode threads on first use.

        All threads share one SimpleQueue.  SimpleQueue.get() is thread-safe for
        multiple concurrent consumers so work is distributed automatically.
        Each thread owns its private CUDA streams and nvcomp codec to avoid races.

        NOTE on N > 1: GPU ANS decode is bottlenecked by HBM bandwidth.  Concurrent
        decode streams from multiple threads DO NOT achieve proportional speedup —
        empirically 4 threads yield only 0.22–0.49× the throughput of a single
        thread because the GPU serialises the decode kernels.  Keep
        MEMRIFT_BG_DECODE_THREADS=1 unless experimenting with non-ANS codecs or
        future GPU hardware with improved concurrency.  The per-layer queue-split
        path (enabled when N > 1) is still correct and useful if that changes.
        """
        # Fast-path: all threads already running.
        if self._bg_threads and all(t.is_alive() for t in self._bg_threads):
            return
        import queue as _queue
        self._bg_queue = _queue.SimpleQueue()
        self._bg_threads = []
        for i in range(self._bg_num_threads):
            t = threading.Thread(
                target=self._bg_decode_worker,
                daemon=True,
                name=f"memrift-prefetch-{i}",
            )
            t.start()
            self._bg_threads.append(t)

    def _bg_decode_worker(self):
        """Background thread: process decode requests sequentially.

        Uses private CUDA streams and nvcomp codec to avoid stream races
        with the main thread.  Each request is processed in order.
        """
        # Dedicated streams (this thread only – no races with main thread).
        bg_h2d = torch.cuda.Stream()
        bg_dec = torch.cuda.Stream()
        bg_merge = torch.cuda.Stream()

        bg_codec = None
        if NVCOMP_AVAILABLE and _nvcomp_lib is not None:
            try:
                bg_codec = _nvcomp_lib.Codec(
                    algorithm='ans',
                    cuda_stream=bg_dec.cuda_stream,
                )
            except Exception:
                bg_codec = None

        fs_sp = self._fs_sp

        while True:
            item = self._bg_queue.get()
            if item is None:
                break  # shutdown sentinel
            nvcomp_cps, pf_dict, merge_evt = item
            try:
                if bg_codec is None or fs_sp is None:
                    raise RuntimeError("bg codec/fs_sp unavailable")
                self._do_bg_batch_decode(
                    nvcomp_cps, pf_dict, merge_evt,
                    bg_h2d, bg_dec, bg_merge, bg_codec, fs_sp,
                )
            except Exception:
                # Record event to prevent training_stream from blocking forever.
                try:
                    bg_merge.record_event(merge_evt)
                except Exception:
                    pass
                for cp in nvcomp_cps:
                    if cp in pf_dict:
                        pf_dict[cp].set_result(None)

    def _do_bg_batch_decode(self, nvcomp_cps, pf_dict, merge_evt,
                             h2d_stream, decode_stream, merge_stream,
                             codec, fs_sp):
        """Core ANS batch decode on bg-thread's dedicated streams.

        Mirror of ``prefetch_batch_on_stream`` logic but uses bg-thread
        streams and calls ``PrefetchFuture.set_result()`` when done.
        """
        import time as _time, os as _os2
        _ta = _time.perf_counter()

        prof_enabled = _lt_enabled()

        # Step 1: H2D on h2d_stream
        h2d_start_evt = h2d_end_evt = None
        if prof_enabled:
            h2d_start_evt = torch.cuda.Event(enable_timing=True)
            h2d_end_evt = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(h2d_stream):
                h2d_start_evt.record(h2d_stream)
        comp_gpu_list = []
        _t_h2d_prepare = 0.0
        _t_h2d_copy = 0.0
        for cp in nvcomp_cps:
            _th0 = _time.perf_counter()
            ans_pinned = getattr(cp, 'ans_pinned', None)
            comp_pinned = getattr(cp, 'comp_pinned', None)
            buf = ans_pinned if ans_pinned is not None else comp_pinned
            if buf is None:
                raw = cp.exp_mv[4:]
                buf = torch.empty(len(raw), dtype=torch.uint8, pin_memory=True)
                buf.numpy()[:] = np.frombuffer(raw, dtype=np.uint8)
            _th1 = _time.perf_counter()
            _n = buf.numel()
            cg = self._acquire_gpu_staging(_n, cp.sm_gpu.device)
            with torch.cuda.stream(h2d_stream):
                cg.copy_(buf, non_blocking=True)
            _th2 = _time.perf_counter()
            comp_gpu_list.append(cg)
            _t_h2d_prepare += _th1 - _th0
            _t_h2d_copy += _th2 - _th1
        h2d_done = h2d_stream.record_event()
        if h2d_end_evt is not None:
            with torch.cuda.stream(h2d_stream):
                h2d_end_evt.record(h2d_stream)
        _tb = _time.perf_counter()

        # Step 2: Wrap as nvcomp.Array (CPU-only)
        comp_arrs = []
        _t_as_array = 0.0
        _t_record_stream = 0.0
        for cg in comp_gpu_list:
            _tw0 = _time.perf_counter()
            comp_arrs.append(_nvcomp_lib.as_array(cg.view(torch.int8)))
            _tw1 = _time.perf_counter()
            cg.record_stream(decode_stream)
            _tw2 = _time.perf_counter()
            _t_as_array += _tw1 - _tw0
            _t_record_stream += _tw2 - _tw1
        del comp_gpu_list
        _tc = _time.perf_counter()

        # Step 3: Decode on decode_stream (waits for H2D via event)
        decode_start_evt = decode_end_evt = None
        if prof_enabled:
            decode_start_evt = torch.cuda.Event(enable_timing=True)
            decode_end_evt = torch.cuda.Event(enable_timing=True)
        with _memrift_nvtx_range("memrift_weight_decode_stream"):
            with torch.cuda.stream(decode_stream):
                decode_stream.wait_event(h2d_done)
                if decode_start_evt is not None:
                    decode_start_evt.record(decode_stream)
            decoded_list = codec.decode(comp_arrs)   # blocks CPU ~0.7ms (ANS decode submit)
            with torch.cuda.stream(decode_stream):
                if decode_end_evt is not None:
                    decode_end_evt.record(decode_stream)
        del comp_arrs
        _td = _time.perf_counter()

        # Step 4: from_dlpack + merge on merge_stream
        bf16_list = []
        _t_meta = _t_dlpack = _t_record_tensor = _t_merge = 0.0
        merge_start_evt = merge_end_evt = None
        if prof_enabled:
            merge_start_evt = torch.cuda.Event(enable_timing=True)
            merge_end_evt = torch.cuda.Event(enable_timing=True)
        merge_started = False
        for cp, decoded in zip(nvcomp_cps, decoded_list):
            _tx0 = _time.perf_counter()
            shape_list, stride_list = _get_prefetch_merge_meta(cp)
            _txm = _time.perf_counter()
            decomp_raw = torch.from_dlpack(decoded).view(torch.uint8)
            _tx1 = _time.perf_counter()
            decomp_raw.record_stream(decode_stream)
            decomp_raw.record_stream(merge_stream)
            _txr = _time.perf_counter()
            with torch.cuda.stream(merge_stream):
                if merge_start_evt is not None and not merge_started:
                    merge_start_evt.record(merge_stream)
                    merge_started = True
                bf16 = fs_sp.merge(
                    decomp_raw, cp.sm_gpu,
                    shape_list, stride_list, 0,
                    cp._dtype, merge_stream.cuda_stream,
                )
            _tx2 = _time.perf_counter()
            del decomp_raw
            bf16_list.append(bf16)
            _t_meta += _txm - _tx0
            _t_dlpack += _tx1 - _txm
            _t_record_tensor += _txr - _tx1
            _t_merge += _tx2 - _txr
        del decoded_list
        _te = _time.perf_counter()

        # Step 5: Record merge_evt THEN signal PrefetchFutures.
        # Order matters: event must be recorded before set_result() so that
        # training_stream.wait_event() sees a valid (recorded) event.
        if merge_end_evt is not None:
            with torch.cuda.stream(merge_stream):
                merge_end_evt.record(merge_stream)
        merge_stream.record_event(merge_evt)
        _tf = _time.perf_counter()
        _lt_add("__weight_prefetch_async_bg__", "submit_h2d_ms", 1000.0 * (_tb - _ta))
        _lt_add("__weight_prefetch_async_bg__", "submit_h2d_prepare_ms", 1000.0 * _t_h2d_prepare)
        _lt_add("__weight_prefetch_async_bg__", "submit_h2d_copy_ms", 1000.0 * _t_h2d_copy)
        _lt_add("__weight_prefetch_async_bg__", "submit_wrap_array_ms", 1000.0 * (_tc - _tb))
        _lt_add("__weight_prefetch_async_bg__", "submit_as_array_ms", 1000.0 * _t_as_array)
        _lt_add("__weight_prefetch_async_bg__", "submit_record_stream_ms", 1000.0 * _t_record_stream)
        _lt_add("__weight_prefetch_async_bg__", "submit_decode_ms", 1000.0 * (_td - _tc))
        _lt_add("__weight_prefetch_async_bg__", "submit_merge_meta_ms", 1000.0 * _t_meta)
        _lt_add("__weight_prefetch_async_bg__", "submit_dlpack_ms", 1000.0 * _t_dlpack)
        _lt_add("__weight_prefetch_async_bg__", "submit_tensor_record_stream_ms", 1000.0 * _t_record_tensor)
        _lt_add("__weight_prefetch_async_bg__", "submit_merge_ms", 1000.0 * _t_merge)
        _lt_add("__weight_prefetch_async_bg__", "submit_record_event_ms", 1000.0 * (_tf - _te))
        _lt_add("__weight_prefetch_async_bg__", "submit_total_ms", 1000.0 * (_tf - _ta))
        if prof_enabled and merge_end_evt is not None:
            _prof_sync = _os2.environ.get("MEMRIFT_PROF_GPU_TIMING_SYNC", "0") == "1"
            if _prof_sync:
                try:
                    merge_end_evt.synchronize()
                    if h2d_start_evt is not None and h2d_end_evt is not None:
                        _lt_add("__weight_prefetch_async_bg__", "gpu_h2d_ms", float(h2d_start_evt.elapsed_time(h2d_end_evt)))
                    if decode_start_evt is not None and decode_end_evt is not None:
                        _lt_add("__weight_prefetch_async_bg__", "gpu_decode_ms", float(decode_start_evt.elapsed_time(decode_end_evt)))
                    if merge_start_evt is not None:
                        _lt_add("__weight_prefetch_async_bg__", "gpu_merge_ms", float(merge_start_evt.elapsed_time(merge_end_evt)))
                except Exception:
                    pass
        for cp, bf16 in zip(nvcomp_cps, bf16_list):
            if cp in pf_dict:
                pf_dict[cp].set_result(bf16)

    def prefetch_batch_async(self, cp_list) -> dict:
        """Non-blocking prefetch: submit to background threads, return immediately.

        When multiple bg threads are configured (MEMRIFT_BG_DECODE_THREADS > 1),
        components are grouped by layer_idx and each layer is submitted as a
        *separate* queue item so that N threads can decode N layers concurrently.
        With a single thread the old "one item for all cps" path is kept to avoid
        any per-layer overhead.

        Returns a dict mapping each ``cp`` → ``PrefetchFuture``.
        Zstd-compressed components fall back to ``materialize_on_stream()``.
        """
        import time as _time
        _t0 = _time.perf_counter()
        self._ensure_bg_threads()

        nvcomp_cps = []
        result = {}
        for cp in cp_list:
            exp_mv = cp.exp_mv
            is_ans = getattr(cp, 'ans_pinned', None) is not None
            is_lz4 = (self._nvcomp_codec is not None
                      and isinstance(exp_mv, (bytes, bytearray, memoryview))
                      and len(exp_mv) >= 4
                      and bytes(exp_mv[:4]) == NVCOMP_LZ4_MAGIC)
            if is_ans or is_lz4:
                nvcomp_cps.append(cp)
            else:
                result[cp] = self.materialize_on_stream(
                    cp.exp_mv, cp.sm_gpu, cp.orig_shape, cp._dtype,
                    comp_pinned=getattr(cp, "comp_pinned", None),
                )

        if not nvcomp_cps:
            return result

        if self._bg_num_threads > 1:
            # ── Multi-thread path: one queue item per layer for max parallelism ──
            # Group cps by layer_idx so each group is decoded by a separate thread.
            from collections import defaultdict as _defaultdict
            by_layer: dict = _defaultdict(list)
            for cp in nvcomp_cps:
                by_layer[getattr(cp, "layer_idx", -1)].append(cp)

            pf_dict: dict = {}
            for layer_idx, layer_cps in sorted(by_layer.items()):
                layer_merge_evt = torch.cuda.Event()
                layer_prof = (f"decoder.layers.{layer_idx}"
                              if layer_idx >= 0 else "__unknown_weight_layer__")
                layer_pf = {
                    cp: PrefetchFuture(layer_merge_evt, prof_name=layer_prof)
                    for cp in layer_cps
                }
                self._bg_queue.put((layer_cps, layer_pf, layer_merge_evt))
                pf_dict.update(layer_pf)
        else:
            # ── Single-thread path: one item for the whole batch (original logic) ──
            merge_evt = torch.cuda.Event()
            pf_dict = {
                cp: PrefetchFuture(
                    merge_evt,
                    prof_name=(f"decoder.layers.{cp.layer_idx}"
                               if getattr(cp, "layer_idx", -1) >= 0
                               else "__unknown_weight_layer__"),
                )
                for cp in nvcomp_cps
            }
            self._bg_queue.put((nvcomp_cps, pf_dict, merge_evt))

        _lt_add("__weight_prefetch_async_bg__", "enqueue_total_ms", 1000.0 * (_time.perf_counter() - _t0))

        result.update(pf_dict)
        return result

    # -------------------------------------------------------------------------
    #  Synchronous compression/decompression
    # -------------------------------------------------------------------------
    
    def kickoff_sync(self, tok: PlaceHolderToken, t: torch.Tensor):
        """Synchronous compression (blocking)."""
        if self._fs_sp is None:
            raise RuntimeError("CUDA extension not available for sync compression")
        
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.d2h_stream):
            cpu_exp, sm_bits = self._fs_sp.split(t, self.d2h_stream.cuda_stream)
            evt = self.d2h_stream.record_event()
        evt.synchronize()
        
        tok.sm_bits = sm_bits
        arr = cpu_exp.numpy()
        cctx = get_compression_ctx(self.zstd_level) if not getattr(self, "cctx", None) else self.cctx
        comped_bytes = cctx.compress(arr)
        tok.comped_cpu_exp = comped_bytes
        tok.numel = arr.size
    
    def decompress_sync(self, tok: PlaceHolderToken):
        """Synchronous decompression (blocking)."""
        if self._fs_sp is None:
            raise RuntimeError("CUDA extension not available for sync decompression")
        
        dctx = get_decompression_ctx() if not getattr(self, "dctx", None) else self.dctx
        # Decompress exponent
        cpu_exp = torch.empty(tok.numel, dtype=torch.uint8, pin_memory=True)
        with dctx.stream_reader(memoryview(tok.comped_cpu_exp)) as reader:
            view = memoryview(cpu_exp.numpy())
            nread = reader.readinto(view)
            assert nread == tok.numel, "decompress size mismatch"
        
        # Merge on GPU
        stream = self.h2d_stream
        with torch.cuda.stream(stream):
            rst = self._fs_sp.merge(
                cpu_exp, tok.sm_bits,
                list(tok.shape), list(tok.stride), tok.offset,
                tok.dtype, stream.cuda_stream
            )
        evt = stream.record_event()
        evt.synchronize()
        
        tok.decomped_data = rst
    
    # 异步编解码

    # 设计说明：ANS 激活压缩默认在 GPU 上 encode，再将压缩体 **异步 D2H 到 pinned CPU** 释放显存；
    # 反向按需 **H2D + GPU decode**（见 ``nvcomp_ans_cpu_pinned``）。``MEMRIFT_ACT_STORE_CPU=0`` 时
    # 压缩体留在 GPU（``nvcomp_ans_gpu_only``），省 PCIe 但占 VRAM。

    def _kickoff_async_gpu(self, tok: PlaceHolderToken, t: torch.Tensor) -> fut.Future:
        """GPU ANS encode + optional D2H of compressed blob (default: store on CPU).

        Main thread 只提交 Future；encode/D2H 在 compress_pool 线程中完成，不阻塞训练流。
        """
        tok.dtype = t.dtype
        tok.shape = t.shape
        tok.stride = t.stride()
        tok.numel = t.numel()
        tok.offset = 0
        tok.sm_bits = None   # not used in GPU-ANS path
        store_cpu = memrift_act_store_compressed_on_cpu()
        tok.comp_format = "nvcomp_ans_cpu_pinned" if store_cpu else "nvcomp_ans_gpu_only"
        tok._memrift_act_cuda_device_index = int(t.device.index) if t.is_cuda else 0

        numel_bytes = t.numel() * t.element_size()

        # Main thread (fast, non-blocking):
        training_evt = torch.cuda.current_stream().record_event()
        t.record_stream(self.act_comp_stream)

        _act_enc_tls = self._act_enc_tls

        import os as _os, time as _time
        _dbg_act = _os.environ.get("MEMRIFT_ACT_DBG", "0") == "1"

        # 关键优化：在主线程先获取 is_contiguous 信息，这样后台线程
        # 可以更早释放原始张量的引用
        is_contiguous = t.is_contiguous()

        def _encode_worker(t_ref, training_evt, numel_bytes, is_contiguous, device_index, store_cpu):
            torch.cuda.set_device(device_index)
            # Lazily create a CUDA stream + codec for this encode worker thread.
            if not hasattr(_act_enc_tls, 'stream'):
                _act_enc_tls.stream = torch.cuda.Stream()
                _act_enc_tls.codec = _nvcomp_lib.Codec(
                    algorithm='ans', cuda_stream=_act_enc_tls.stream.cuda_stream)
            tl_stream = _act_enc_tls.stream
            tl_codec  = _act_enc_tls.codec

            _t0 = _time.perf_counter()
            # Insert GPU-side ordering: tl_stream waits for training fwd event
            tl_stream.wait_event(training_evt)

            t_cont = None
            act_int8 = None

            # Check if already contiguous before creating a copy
            if is_contiguous:
                act_int8 = t_ref.view(torch.int8)
            else:
                # Only make contiguous if necessary
                with torch.cuda.stream(tl_stream):
                    t_cont = t_ref.contiguous()
                t_cont.record_stream(tl_stream)
                act_int8 = t_cont.view(torch.int8)

            # 立即释放原始张量引用，这是显存释放的关键！
            del t_ref

            comp_arr = _nvcomp_lib.as_array(act_int8)
            _t1 = _time.perf_counter()
            # encode() internally syncs tl_stream (only THIS thread blocks)
            comp_output = tl_codec.encode(comp_arr)
            _t2 = _time.perf_counter()
            n = int(comp_output.buffer_size)
            with torch.cuda.stream(tl_stream):
                comp_tensor = torch.from_dlpack(comp_output).view(torch.uint8)
                # Clone to keep only the valid portion and break DLPack dependency
                comp_tensor_gpu = comp_tensor[:n].clone()
                comp_tensor_gpu.record_stream(tl_stream)

            _t3 = _time.perf_counter()
            _ap_add("gpu_act_enc_setup_cpu_ms", 1000.0 * (_t1 - _t0))
            _ap_add("gpu_act_enc_encode_cpu_ms", 1000.0 * (_t2 - _t1))
            _ap_add("gpu_act_enc_gpu_only_ms", 1000.0 * (_t3 - _t2))
            if _dbg_act:
                print(
                    f"[ACT_ENC_GPU] mb={numel_bytes//1024}KB setup={1000*(_t1-_t0):.1f}ms "
                    f"encode={1000*(_t2-_t1):.1f}ms gpu_clone={1000*(_t3-_t2):.1f}ms "
                    f"store_cpu={store_cpu}",
                    flush=True,
                )

            if t_cont is not None:
                del t_cont
            del act_int8, comp_arr, comp_output, comp_tensor

            if store_cpu:
                # 按需：GPU 压缩完成后 D2H 到 pinned CPU，释放压缩体占用的显存
                cpu_pin = torch.empty(n, dtype=torch.uint8, pin_memory=True)
                _td2h0 = _time.perf_counter()
                with torch.cuda.stream(tl_stream):
                    cpu_pin.copy_(comp_tensor_gpu, non_blocking=True)
                d2h_evt = tl_stream.record_event()
                d2h_evt.synchronize()
                _td2h1 = _time.perf_counter()
                _ap_add("gpu_act_enc_d2h_compressed_ms", 1000.0 * (_td2h1 - _td2h0))
                del comp_tensor_gpu
                if _dbg_act:
                    print(
                        f"[ACT_ENC_D2H] n={n}B wall={1000*(_td2h1-_td2h0):.2f}ms",
                        flush=True,
                    )
                return (cpu_pin, numel_bytes)

            return (comp_tensor_gpu, numel_bytes)

        device_index = int(tok._memrift_act_cuda_device_index)
        return self.compress_pool.submit(
            _encode_worker, t, training_evt, numel_bytes, is_contiguous, device_index, store_cpu
        )

    def _decompress_async_gpu(self, tok: PlaceHolderToken, future: fut.Future) -> None:
        """GPU-only ANS activation decompression (NO H2D), fully on GPU.

        Flow (in decode_pool background thread, ALL GPU):
          1. future.result() → (gpu_compressed_tensor, numel_bytes_original)
          2. GPU ANS decode on tl_stream (compressed data already on GPU)
          3. from_dlpack inside tl_stream context → clone
          4. tl_stream.synchronize() – waits ONLY for THIS thread's work
          5. Sets tok.decomped_data + tok.CtoD_copy_evt + tok.ready_evt

        Fully GPU-only: no H2D/D2H overhead, compressed data stays on GPU the entire time.
        """
        _act_dec_tls = self._act_dec_tls

        import os as _os2, time as _time2
        _dbg_act2 = _os2.environ.get("MEMRIFT_ACT_DBG", "0") == "1"

        def _decode(tok, future):
            tok.ready_evt.clear()
            try:
                # Lazily create a CUDA stream for this worker thread.
                if not hasattr(_act_dec_tls, 'stream'):
                    _act_dec_tls.stream = torch.cuda.Stream()
                    _act_dec_tls.codec = _nvcomp_lib.Codec(
                        algorithm='ans', cuda_stream=_act_dec_tls.stream.cuda_stream)
                tl_stream = _act_dec_tls.stream
                tl_codec  = _act_dec_tls.codec

                _td0 = _time2.perf_counter()
                # Get compressed tensor directly from GPU (no CPU bytes!)
                gpu_comp, numel_bytes = future.result()
                _td1 = _time2.perf_counter()

                # No H2D needed - data is already on GPU!
                gpu_comp.record_stream(tl_stream)

                comp_arr = _nvcomp_lib.as_array(gpu_comp.view(torch.int8))
                _td2 = _time2.perf_counter()
                # decode() internally syncs tl_stream (only THIS thread blocks)
                with _memrift_nvtx_range("memrift_act_decode_tl_stream"):
                    decomp_out = tl_codec.decode(comp_arr)
                _td3 = _time2.perf_counter()
                # from_dlpack inside tl_stream context
                with _memrift_nvtx_range("memrift_act_dlpack_clone_tl_stream"):
                    with torch.cuda.stream(tl_stream):
                        decomp_raw = torch.from_dlpack(decomp_out).view(torch.uint8)
                        decomp_raw.record_stream(tl_stream)
                        result = decomp_raw[:numel_bytes].view(tok.dtype).reshape(tok.shape)
                        result = result.clone()
                        # 优化：在同一个 stream 上直接记录事件，避免显式同步
                        copy_evt = tl_stream.record_event()
                # 优化：移除不必要的同步，因为我们已经记录了事件
                # tl_stream.synchronize()
                _td4 = _time2.perf_counter()

                _ap_add("gpu_act_dec_wait_future_cpu_ms", 1000.0 * (_td1 - _td0))
                _ap_add("gpu_act_dec_gpu_ready_ms", 1000.0 * (_td2 - _td1))
                _ap_add("gpu_act_dec_decode_cpu_ms", 1000.0 * (_td3 - _td2))
                _ap_add("gpu_act_dec_clone_sync_cpu_ms", 1000.0 * (_td4 - _td3))
                if _layer_time_profiler is not None and _layer_time_profiler.is_enabled():
                    try:
                        layer_name = getattr(tok, "layer_name", "unknown")
                        _layer_time_profiler.add_time(
                            layer_name,
                            "activation_decode_thread_ms",
                            1000.0 * (_td4 - _td0),
                        )
                        _layer_time_profiler.add_time(
                            layer_name,
                            "activation_decode_only_ms",
                            1000.0 * (_td3 - _td2),
                        )
                    except Exception:
                        pass
                hook_ev = getattr(tok, "_prof_hook_evt", None)
                if hook_ev is not None:
                    _ap_gpu_elapsed(hook_ev, copy_evt, "gpu_ms_hook_to_act_decode_done")
                    if _layer_time_profiler is not None and _layer_time_profiler.is_enabled():
                        try:
                            _layer_time_profiler.add_time(
                                getattr(tok, "layer_name", "unknown"),
                                "activation_decode_gpu_ms",
                                float(hook_ev.elapsed_time(copy_evt)),
                            )
                        except Exception:
                            pass
                    try:
                        delattr(tok, "_prof_hook_evt")
                    except Exception:
                        tok._prof_hook_evt = None  # type: ignore[attr-defined]

                if _dbg_act2:
                    print(f"[ACT_DEC_GPU] wait_enc={1000*(_td1-_td0):.1f}ms gpu_ready={1000*(_td2-_td1):.1f}ms decode={1000*(_td3-_td2):.1f}ms clone={1000*(_td4-_td3):.1f}ms total={1000*(_td4-_td0):.1f}ms", flush=True)

                tok.decomped_data = result
                tok.CtoD_copy_evt = copy_evt
                del gpu_comp, comp_arr, decomp_out, decomp_raw
            except Exception:
                import traceback; traceback.print_exc()
                tok.decomped_data = None
                tok.CtoD_copy_evt = None
            finally:
                tok.ready_evt.set()

        self.decode_pool.submit(_decode, tok, future)

    def _decompress_async_from_cpu_pinned(self, tok: PlaceHolderToken, future: fut.Future) -> None:
        """Backward: H2D pinned CPU compressed blob, then GPU ANS decode (decode_pool thread, ordered on tl_stream)."""
        _act_dec_tls = self._act_dec_tls
        import os as _os2, time as _time2
        _dbg_act2 = _os2.environ.get("MEMRIFT_ACT_DBG", "0") == "1"

        def _decode(tok, future):
            tok.ready_evt.clear()
            try:
                if not hasattr(_act_dec_tls, 'stream'):
                    _act_dec_tls.stream = torch.cuda.Stream()
                    _act_dec_tls.codec = _nvcomp_lib.Codec(
                        algorithm='ans', cuda_stream=_act_dec_tls.stream.cuda_stream)
                tl_stream = _act_dec_tls.stream
                tl_codec = _act_dec_tls.codec

                di = int(getattr(tok, "_memrift_act_cuda_device_index", 0))
                torch.cuda.set_device(di)

                _td0 = _time2.perf_counter()
                cpu_pin, numel_bytes = future.result()
                _td1 = _time2.perf_counter()

                n = int(cpu_pin.numel())
                gpu_comp = torch.empty(n, dtype=torch.uint8, device=f"cuda:{di}")
                with torch.cuda.stream(tl_stream):
                    gpu_comp.copy_(cpu_pin, non_blocking=True)
                _td2 = _time2.perf_counter()

                gpu_comp.record_stream(tl_stream)
                comp_arr = _nvcomp_lib.as_array(gpu_comp.view(torch.int8))
                _td3 = _time2.perf_counter()
                with _memrift_nvtx_range("memrift_act_decode_tl_stream"):
                    decomp_out = tl_codec.decode(comp_arr)
                _td4 = _time2.perf_counter()
                with _memrift_nvtx_range("memrift_act_dlpack_clone_tl_stream"):
                    with torch.cuda.stream(tl_stream):
                        decomp_raw = torch.from_dlpack(decomp_out).view(torch.uint8)
                        decomp_raw.record_stream(tl_stream)
                        result = decomp_raw[:numel_bytes].view(tok.dtype).reshape(tok.shape)
                        result = result.clone()
                        copy_evt = tl_stream.record_event()
                _td5 = _time2.perf_counter()

                _ap_add("gpu_act_dec_wait_future_cpu_ms", 1000.0 * (_td1 - _td0))
                _ap_add("gpu_act_dec_h2d_compressed_ms", 1000.0 * (_td2 - _td1))
                _ap_add("gpu_act_dec_decode_cpu_ms", 1000.0 * (_td4 - _td3))
                _ap_add("gpu_act_dec_clone_sync_cpu_ms", 1000.0 * (_td5 - _td4))
                if _layer_time_profiler is not None and _layer_time_profiler.is_enabled():
                    try:
                        layer_name = getattr(tok, "layer_name", "unknown")
                        _layer_time_profiler.add_time(
                            layer_name,
                            "activation_decode_thread_ms",
                            1000.0 * (_td5 - _td0),
                        )
                        _layer_time_profiler.add_time(
                            layer_name,
                            "activation_decode_only_ms",
                            1000.0 * (_td4 - _td3),
                        )
                    except Exception:
                        pass
                hook_ev = getattr(tok, "_prof_hook_evt", None)
                if hook_ev is not None:
                    _ap_gpu_elapsed(hook_ev, copy_evt, "gpu_ms_hook_to_act_decode_done")
                    if _layer_time_profiler is not None and _layer_time_profiler.is_enabled():
                        try:
                            _layer_time_profiler.add_time(
                                getattr(tok, "layer_name", "unknown"),
                                "activation_decode_gpu_ms",
                                float(hook_ev.elapsed_time(copy_evt)),
                            )
                        except Exception:
                            pass
                    try:
                        delattr(tok, "_prof_hook_evt")
                    except Exception:
                        tok._prof_hook_evt = None  # type: ignore[attr-defined]

                if _dbg_act2:
                    print(
                        f"[ACT_DEC_CPU_PIN] wait_fut={1000*(_td1-_td0):.1f}ms h2d={1000*(_td2-_td1):.1f}ms "
                        f"decode={1000*(_td4-_td3):.1f}ms clone={1000*(_td5-_td4):.1f}ms",
                        flush=True,
                    )

                tok.decomped_data = result
                tok.CtoD_copy_evt = copy_evt
                del gpu_comp, comp_arr, decomp_out, decomp_raw
            except Exception:
                import traceback
                traceback.print_exc()
                tok.decomped_data = None
                tok.CtoD_copy_evt = None
            finally:
                tok.ready_evt.set()

        self.decode_pool.submit(_decode, tok, future)

    # -------------------------------------------------------------------------
    #  Asynchronous compression/decompression
    # -------------------------------------------------------------------------

    def kickoff_async(self, tok: PlaceHolderToken, t: torch.Tensor):
        """
        Start async activation compression for saved-tensors hooks.

        GPU nvCOMP ANS encode; by default compressed bytes are copied to pinned CPU (``nvcomp_ans_cpu_pinned``).
        Matching decode in ``decompress_async``. Set ``MEMRIFT_ACT_STORE_CPU=0`` to keep blobs on GPU.

        Raises:
            RuntimeError: if the float_split extension or nvCOMP ANS activation codecs are unavailable.
        """
        if self._fs_sp is None:
            raise RuntimeError("CUDA extension not available for async compression")

        if not (NVCOMP_AVAILABLE and self._nvcomp_act_enc_codec is not None):
            raise RuntimeError(
                "MemRift activation compression requires nvidia.nvcomp (GPU ANS). "
                "Install a matching nvidia-nvcomp-cu* package, or disable activation compression "
                "(e.g. train.system.memrift_activation_enable=false)."
            )
        return self._kickoff_async_gpu(tok, t)
    
    def decompress_async(self, tok: PlaceHolderToken, future: fut.Future):
        """
        Start async activation restore (GPU-only path preferred).

        Routes by ``tok.comp_format``:
        - ``nvcomp_ans_cpu_pinned`` (default): H2D compressed blob then GPU ANS decode
        - ``nvcomp_ans_gpu_only``: compressed tensor already on GPU → ANS decode only
        - ``nvcomp_ans_full``: same as GPU-only decode path
        - Legacy zstd + float_split if activations used ``kickoff_sync`` (CPU decompress).

        The token's ready_evt will be set when the tensor is ready on GPU.
        """
        if self._fs_sp is None:
            raise RuntimeError("CUDA extension not available for async decompression")

        comp_format = getattr(tok, 'comp_format', 'zstd')
        if comp_format == 'nvcomp_ans_cpu_pinned' and self._nvcomp_act_dec_codec is not None:
            self._decompress_async_from_cpu_pinned(tok, future)
            return

        # GPU-resident compressed blob
        if (comp_format == 'nvcomp_ans_gpu_only' or comp_format == 'nvcomp_ans_full') \
                and self._nvcomp_act_dec_codec is not None:
            self._decompress_async_gpu(tok, future)
            return

        # CPU path (zstd fallback) - should not be used in GPU-only mode
        # Use act_decomp_stream instead of h2d_stream so activation merge does NOT
        # contend with weight decompression H2D (which uses h2d_stream).
        fs_sp = self._fs_sp
        act_h2d_stream = self.act_decomp_stream
        semaphore = self.decomp_semaphore

        def _decode(tok, future):
            semaphore.acquire()
            comped_bytes, numel = None, None
            cpu_exp = None
            try:
                result = future.result()
                if isinstance(result[0], torch.Tensor) and result[0].is_cuda:
                    # GPU tensor fallback - use GPU decode path
                    self._decompress_async_gpu(tok, future)
                    return

                comped_bytes, numel = result

                cpu_exp = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
                dctx = get_decompression_ctx()
                with dctx.stream_reader(memoryview(comped_bytes)) as reader:
                    view = memoryview(cpu_exp.numpy())
                    nread = reader.readinto(view)
                    assert nread == numel, "decompress size mismatch"

                with torch.cuda.stream(act_h2d_stream):
                    rst = fs_sp.merge(
                        cpu_exp, tok.sm_bits,
                        list(tok.shape), list(tok.stride), tok.offset,
                        tok.dtype, act_h2d_stream.cuda_stream
                    )
                    tok.sm_bits.record_stream(act_h2d_stream)

                evt = act_h2d_stream.record_event()
                evt.synchronize()
                tok.CtoD_copy_evt = evt
                tok.ready_evt.set()
                tok.decomped_data = rst

            finally:
                semaphore.release()
                del future
                if comped_bytes is not None:
                    del comped_bytes
                if tok.sm_bits is not None:
                    fs_sp.release_cuda(tok.sm_bits)
                if cpu_exp is not None:
                    del cpu_exp

        self.decode_pool.submit(_decode, tok, future)
    
    def materialize_async(
        self,
        exp_mv: bytes,
        sm_gpu: torch.Tensor,
        orig_shape: tuple,
        dtype: torch.dtype,
        callback: Optional[Callable[[torch.Tensor], None]] = None,
        comp_pinned: Optional[torch.Tensor] = None,
    ):
        """
        Async weight materialization (decompression + merge).
        
        Supports two decompression paths selected by exp_mv format:
        - GPU path (NVCOMP_LZ4_MAGIC prefix): CPU thread submits H2D +
          nvcomp GPU decode + merge, returns in ~0.3ms. No CPU blocking.
        - CPU path (zstd, backward compat): CPU thread does zstd
          decompression (~4-6ms), then submits GPU merge.
        
        Args:
            exp_mv: Compressed exponent bytes
            sm_gpu: Sign+mantissa tensor on GPU
            orig_shape: Original tensor shape
            dtype: Target dtype
            callback: Optional callback with decompressed tensor
            comp_pinned: Pre-allocated persistent pinned buffer (from CompressedParam.comp_pinned).
                When provided, skips the ~10ms cold-cache torch.empty(pin_memory=True) that
                thread-pool threads would otherwise incur on every call.
        """
        if self._fs_sp is None:
            raise RuntimeError("CUDA extension not available")
        
        fs_sp = self._fs_sp
        h2d_stream = self.h2d_stream
        semaphore = self.decomp_semaphore
        nvcomp_codec = self._nvcomp_codec  # capture for closure
        nvcomp_ans = self._nvcomp_ans_codec
        # Capture the pre-allocated pinned buffer to avoid closure over 'comp_pinned'
        # parameter being rebound by later calls.
        _comp_pinned_persistent = comp_pinned

        def _c_contiguous_strides(shape):
            strides = [1] * len(shape)
            running = 1
            for i in range(len(shape) - 2, -1, -1):
                running *= shape[i + 1]
                strides[i] = running
            return tuple(strides)

        def _materialize():
            semaphore.acquire()
            _sem_released = False
            _success = False
            keepalive = None  # tensors that must stay alive until evt sync
            try:
                numel = int(np.prod(orig_shape))
                strides = _c_contiguous_strides(orig_shape)

                # ── GPU path: nvCOMP LZ4 ──────────────────────────────────
                # Detected when exp_mv is prefixed with NVCOMP_LZ4_MAGIC.
                # CPU work: H2D submit (~0.05ms) + kernel submits (~0.2ms).
                # Returns immediately; all H2D/decode/merge run async on h2d_stream.
                if (nvcomp_codec is not None
                        and isinstance(exp_mv, (bytes, bytearray, memoryview))
                        and len(exp_mv) >= 4
                        and bytes(exp_mv[:4]) == NVCOMP_LZ4_MAGIC):

                    # 1. Use persistent pinned buffer to avoid per-call alloc overhead.
                    #    Thread-pool threads have cold per-thread pinned-memory caches
                    #    (~10ms/alloc vs ~0.02ms for main-thread warm cache).
                    if _comp_pinned_persistent is not None:
                        comp_pinned_buf = _comp_pinned_persistent  # pre-filled, reuse
                    else:
                        comp_data = exp_mv[4:]
                        comp_pinned_buf = torch.empty(
                            len(comp_data), dtype=torch.uint8, pin_memory=True)
                        comp_pinned_buf.numpy()[:] = np.frombuffer(
                            comp_data, dtype=np.uint8)

                    # 2. H2D DMA: compressed bytes → GPU (async, h2d_stream)
                    _n = comp_pinned_buf.numel()
                    comp_gpu = self._acquire_gpu_staging(_n, sm_gpu.device)
                    with torch.cuda.stream(h2d_stream):
                        comp_gpu.copy_(comp_pinned_buf, non_blocking=True)
                    comp_gpu.record_stream(h2d_stream)

                    # 3. GPU nvCOMP LZ4 decode (queued on h2d_stream via codec)
                    #    Executes after H2D because same stream.
                    #    Create per-task codec so concurrent thread-pool calls don't
                    #    share a single Codec object (thread-safety issue if decoded
                    #    from multiple threads simultaneously).  Codec creation is fast
                    #    (~0.12ms) and the GPU ops are still serialized by h2d_stream.
                    task_codec = _nvcomp_lib.Codec(algorithm='lz4', cuda_stream=h2d_stream.cuda_stream)
                    comp_arr = _nvcomp_lib.as_array(comp_gpu.view(torch.int8))
                    # nvcomp返回自有缓冲对象
                    decomp = task_codec.decode(comp_arr)

                    # 4. Convert decoded output to a proper PyTorch-owned tensor.
                    #    .clone() on h2d_stream: (a) runs after decode because
                    #    both are on h2d_stream; (b) breaks DLPack lifetime
                    #    dependency on decomp, eliminating potential double-free.
                    decomp_raw = torch.from_dlpack(
                        decomp.to_dlpack()).view(torch.uint8)
                    with torch.cuda.stream(h2d_stream):
                        exp_gpu = decomp_raw[:numel].clone()
                    exp_gpu.record_stream(h2d_stream)

                    # 5. GPU merge: exp_gpu + sm_gpu → bf16 (on h2d_stream)
                    with torch.cuda.stream(h2d_stream):
                        bf16 = fs_sp.merge(
                            exp_gpu, sm_gpu,
                            list(orig_shape), list(strides), 0,
                            dtype, h2d_stream.cuda_stream,
                        )
                    evt = h2d_stream.record_event()

                    # Keep alive until caller syncs evt:
                    #   comp_gpu   – decode kernel source (H2D dest)
                    #   decomp     – nvcomp.Array owns decoded GPU memory
                    #   decomp_raw – DLPack ref keeps nvcomp output alive
                    #   exp_gpu    – clone output; merge reads it
                    # NOTE: comp_pinned_buf is either the persistent cp.comp_pinned
                    # (not freed here) or a freshly allocated buffer (kept alive below).
                    # task_codec must stay alive until evt syncs (decode kernel uses it)
                    if _comp_pinned_persistent is not None:
                        keepalive = (task_codec, comp_gpu, decomp, decomp_raw, exp_gpu)
                    else:
                        keepalive = (comp_pinned_buf, task_codec, comp_gpu, decomp, decomp_raw, exp_gpu)

                    semaphore.release()
                    _sem_released = True
                    _success = True

                    if callback:
                        evt.synchronize()
                        keepalive = None
                        _success = False
                        callback(bf16)
                        return bf16

                    return (bf16, evt, keepalive)

                # ── GPU path: nvCOMP ANS (prepare_weight --compression nvcomp_ans) ─
                if (nvcomp_ans is not None
                        and isinstance(exp_mv, (bytes, bytearray, memoryview))
                        and len(exp_mv) >= 4
                        and bytes(exp_mv[:4]) == NVCOMP_ANS_MAGIC):

                    if _comp_pinned_persistent is not None:
                        comp_pinned_buf = _comp_pinned_persistent
                    else:
                        comp_data = exp_mv[4:]
                        comp_pinned_buf = torch.empty(
                            len(comp_data), dtype=torch.uint8, pin_memory=True)
                        comp_pinned_buf.numpy()[:] = np.frombuffer(
                            comp_data, dtype=np.uint8)

                    _n = comp_pinned_buf.numel()
                    comp_gpu = self._acquire_gpu_staging(_n, sm_gpu.device)
                    with torch.cuda.stream(h2d_stream):
                        comp_gpu.copy_(comp_pinned_buf, non_blocking=True)
                    comp_gpu.record_stream(h2d_stream)

                    task_codec = _nvcomp_lib.Codec(
                        algorithm='ans', cuda_stream=h2d_stream.cuda_stream)
                    comp_arr = _nvcomp_lib.as_array(comp_gpu.view(torch.int8))
                    decomp = task_codec.decode(comp_arr)

                    decomp_raw = torch.from_dlpack(decomp).view(torch.uint8)
                    decomp_raw.record_stream(h2d_stream)
                    with torch.cuda.stream(h2d_stream):
                        bf16 = fs_sp.merge(
                            decomp_raw, sm_gpu,
                            list(orig_shape), list(strides), 0,
                            dtype, h2d_stream.cuda_stream,
                        )
                    evt = h2d_stream.record_event()

                    if _comp_pinned_persistent is not None:
                        keepalive = (task_codec, comp_gpu, decomp, decomp_raw)
                    else:
                        keepalive = (
                            comp_pinned_buf, task_codec, comp_gpu, decomp, decomp_raw)

                    semaphore.release()
                    _sem_released = True
                    _success = True

                    if callback:
                        evt.synchronize()
                        keepalive = None
                        _success = False
                        callback(bf16)
                        return bf16

                    return (bf16, evt, keepalive)

                # ── CPU path: zstd (legacy / backward compat) ─────────────
                memrift_require_nvcomp_weight_payload(exp_mv)
                cpu_exp = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
                dctx = get_decompression_ctx()
                with dctx.stream_reader(memoryview(exp_mv)) as reader:
                    view = memoryview(cpu_exp.numpy())
                    nread = reader.readinto(view)
                    assert nread == numel, "decompress size mismatch"

                with torch.cuda.stream(h2d_stream):
                    bf16 = fs_sp.merge(
                        cpu_exp, sm_gpu,
                        list(orig_shape), list(strides), 0,
                        dtype, h2d_stream.cuda_stream
                    )
                evt = h2d_stream.record_event()

                semaphore.release()
                _sem_released = True
                _success = True
                keepalive = cpu_exp

                if callback:
                    evt.synchronize()
                    del keepalive
                    _success = False
                    callback(bf16)
                    return bf16

                # Non-callback (prefetch) path: return immediately.
                # cpu_exp is included in keepalive to keep DMA src alive.
                return (bf16, evt, keepalive)

            finally:
                if not _sem_released:
                    semaphore.release()
                if not _success and keepalive is not None:
                    del keepalive

        submit_pool = self.decode_pool
        if self.enable_async and self._materialize_path_is_cpu_zstd(
            exp_mv, nvcomp_codec, nvcomp_ans
        ):
            submit_pool = self.zstd_pool
        return submit_pool.submit(_materialize)

    @staticmethod
    def _materialize_path_is_cpu_zstd(
        exp_mv: Any,
        nvcomp_codec: Any,
        nvcomp_ans: Any,
    ) -> bool:
        """True if ``materialize_async`` will take the CPU zstd branch (not nvCOMP GPU)."""
        if not isinstance(exp_mv, (bytes, bytearray, memoryview)) or len(exp_mv) < 4:
            return True
        m = bytes(exp_mv[:4])
        if nvcomp_codec is not None and m == NVCOMP_LZ4_MAGIC:
            return False
        if nvcomp_ans is not None and m == NVCOMP_ANS_MAGIC:
            return False
        return True

    def prefetch_batch_on_stream(self, cp_list) -> dict:
        """
        Batch-decode multiple compressed weight components in a single GPU call.

        All components are decoded with ONE ``codec.decode([...])`` call, which
        amortises the per-call overhead (~5 ms × N → ~2 ms total for 7 comps).

        Args:
            cp_list: list of ``CompressedParam`` objects whose compressed
                     exponent bytes should be decoded.  nvCOMP paths (offline
                     ANS or LZ4 plus ``ans_pinned``) are batched; zstd falls
                     back to ``materialize_on_stream()`` individually.

        Returns:
            dict mapping each cp in ``cp_list`` → ``StreamFuture``.
        """
        if self._fs_sp is None:
            raise RuntimeError("CUDA extension not available")

        fs_sp = self._fs_sp
        h2d_stream = self.h2d_stream

        # Split into nvcomp (LZ4 or ANS) and zstd components.
        # ANS weights (NVCOMP_ANS_MAGIC) are stored in cp.ans_pinned and decoded
        # with _nvcomp_decode_codec (ANS, decode_stream). LZ4 weights fall back to
        # _nvcomp_codec (LZ4, h2d_stream) via materialize_on_stream.
        nvcomp_cps = []
        zstd_cps = []
        for cp in cp_list:
            exp_mv = cp.exp_mv
            is_ans = (self._nvcomp_decode_codec is not None
                      and getattr(cp, 'ans_pinned', None) is not None)
            is_lz4 = (self._nvcomp_codec is not None
                      and isinstance(exp_mv, (bytes, bytearray, memoryview))
                      and len(exp_mv) >= 4
                      and bytes(exp_mv[:4]) == NVCOMP_LZ4_MAGIC)
            if is_ans or is_lz4:
                nvcomp_cps.append(cp)
            else:
                zstd_cps.append(cp)

        result = {}

        # ── zstd fallback: one-by-one (blocked when nvCOMP is on unless opt-in) ─
        for cp in zstd_cps:
            memrift_require_nvcomp_weight_payload(cp.exp_mv)
            result[cp] = self.materialize_on_stream(
                cp.exp_mv, cp.sm_gpu, cp.orig_shape, cp._dtype,
                comp_pinned=getattr(cp, "comp_pinned", None),
            )
            try:
                result[cp]._prof_name = f"decoder.layers.{cp.layer_idx}" if getattr(cp, "layer_idx", -1) >= 0 else "__unknown_weight_layer__"
            except Exception:
                pass

        if not nvcomp_cps:
            return result

        # ── nvCOMP batch path (three-stream pipeline) ───────────────────────
        # Stream layout:
        #   h2d_stream   : H2D transfers only  → h2d_done_event (fires after H2D)
        #   decode_stream: nvCOMP decode only   → CPU synced via from_dlpack
        #   merge_stream : clone + float-split-merge → merge_event for training
        # Weights may be LZ4-compressed (comp_pinned) or ANS-compressed (ans_pinned).
        # ANS gives ~2.8x smaller buffers → faster H2D + faster decode.
        import time as _time, os as _os
        _dbg = _os.environ.get("MEMRIFT_DBG2", "0") == "1"
        _ta = _time.perf_counter()
        decode_stream = self.decode_stream
        merge_stream = self.merge_stream
        decode_codec = self._nvcomp_decode_codec

        prof_enabled = _lt_enabled()

        # ── Step 1: H2D compressed data on h2d_stream ──────────────────────
        h2d_start_evt = h2d_end_evt = None
        if prof_enabled:
            h2d_start_evt = torch.cuda.Event(enable_timing=True)
            h2d_end_evt = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(h2d_stream):
                h2d_start_evt.record(h2d_stream)
        comp_gpu_list = []
        pinned_extras = []
        _t_h2d_prepare = 0.0
        _t_h2d_copy = 0.0
        for cp in nvcomp_cps:
            _th0 = _time.perf_counter()
            # Prefer ans_pinned (ANS-compressed, smaller), fall back to comp_pinned (LZ4).
            ans_pinned = getattr(cp, 'ans_pinned', None)
            comp_pinned = getattr(cp, 'comp_pinned', None)
            if ans_pinned is not None:
                comp_pinned_buf = ans_pinned
                pinned_extras.append(None)
            elif comp_pinned is not None:
                comp_pinned_buf = comp_pinned
                pinned_extras.append(None)
            else:
                comp_data = cp.exp_mv[4:]
                comp_pinned_buf = torch.empty(
                    len(comp_data), dtype=torch.uint8, pin_memory=True)
                comp_pinned_buf.numpy()[:] = np.frombuffer(comp_data, dtype=np.uint8)
                pinned_extras.append(comp_pinned_buf)

            _th1 = _time.perf_counter()
            # Use GPU staging pool to avoid repeated cudaMallocAsync.
            _n = comp_pinned_buf.numel()
            cg = self._acquire_gpu_staging(_n, cp.sm_gpu.device)
            with torch.cuda.stream(h2d_stream):
                cg.copy_(comp_pinned_buf, non_blocking=True)
            _th2 = _time.perf_counter()
            comp_gpu_list.append(cg)
            _t_h2d_prepare += _th1 - _th0
            _t_h2d_copy += _th2 - _th1

        # h2d_done_event fires when H2D completes; h2d_stream has ONLY this H2D.
        h2d_done_event = h2d_stream.record_event()
        if h2d_end_evt is not None:
            with torch.cuda.stream(h2d_stream):
                h2d_end_evt.record(h2d_stream)

        _tb = _time.perf_counter()

        # ── Step 2: Wrap as nvcomp.Array (CPU-only) ──────────────────────────
        comp_arrs = []
        _t_as_array = 0.0
        _t_record_stream = 0.0
        for cg in comp_gpu_list:
            _tw0 = _time.perf_counter()
            comp_arrs.append(_nvcomp_lib.as_array(cg.view(torch.int8)))
            _tw1 = _time.perf_counter()
            cg.record_stream(decode_stream)
            _tw2 = _time.perf_counter()
            _t_as_array += _tw1 - _tw0
            _t_record_stream += _tw2 - _tw1
        del comp_gpu_list

        _tc = _time.perf_counter()

        # ── Step 3: Decode on decode_stream ──────────────────────────────────
        # GPU: decode_stream waits for h2d_done_event, then runs decode kernels.
        # CPU: codec.decode() syncs decode_stream (waits only for H2D: ~1.68ms ANS
        #      vs ~4.17ms LZ4). decode_stream has no prior pending work.
        decode_start_evt = decode_end_evt = None
        if prof_enabled:
            decode_start_evt = torch.cuda.Event(enable_timing=True)
            decode_end_evt = torch.cuda.Event(enable_timing=True)
        with _memrift_nvtx_range("memrift_weight_decode_stream"):
            with torch.cuda.stream(decode_stream):
                decode_stream.wait_event(h2d_done_event)
                if decode_start_evt is not None:
                    decode_start_evt.record(decode_stream)
            decoded_list = decode_codec.decode(comp_arrs)
            with torch.cuda.stream(decode_stream):
                if decode_end_evt is not None:
                    decode_end_evt.record(decode_stream)
        del comp_arrs

        _td = _time.perf_counter()

        # ── Step 4: from_dlpack + merge ───────────────────────────────────────
        # 长期：用「按 Shape 预分配 CUDA uint8 缓冲 + nvCOMP C API 直写 data_ptr」
        # 替换 DLPack，见文件头部设计说明。
        # from_dlpack() syncs decode_stream (current decode only: ~0.87ms ANS).
        # merge are submitted to merge_stream without extra CPU blocking beyond dlpack.
        # ANS decode produces output with exactly numel(orig_shape) elements,
        # contiguous – no intermediate clone needed; pass decomp_raw directly to merge.
        bf16_list = []
        _t_meta = _t_dlpack = _t_record_tensor = _t_merge = 0.0
        merge_start_evt = merge_end_evt = None
        if prof_enabled:
            merge_start_evt = torch.cuda.Event(enable_timing=True)
            merge_end_evt = torch.cuda.Event(enable_timing=True)
        merge_started = False
        for cp, decoded in zip(nvcomp_cps, decoded_list):
            _tx0 = _time.perf_counter()
            shape_list, stride_list = _get_prefetch_merge_meta(cp)
            _txm = _time.perf_counter()
            decomp_raw = torch.from_dlpack(decoded).view(torch.uint8)
            _tx1 = _time.perf_counter()
            decomp_raw.record_stream(decode_stream)
            decomp_raw.record_stream(merge_stream)
            _txr = _time.perf_counter()
            with torch.cuda.stream(merge_stream):
                if merge_start_evt is not None and not merge_started:
                    merge_start_evt.record(merge_stream)
                    merge_started = True
                bf16 = fs_sp.merge(
                    decomp_raw, cp.sm_gpu,
                    shape_list, stride_list, 0,
                    cp._dtype, merge_stream.cuda_stream,
                )
            _tx2 = _time.perf_counter()
            del decomp_raw
            bf16_list.append(bf16)
            _t_meta += _txm - _tx0
            _t_dlpack += _tx1 - _txm
            _t_record_tensor += _txr - _tx1
            _t_merge += _tx2 - _txr

        del decoded_list

        # ── Step 5: ONE event on merge_stream after all merges ───────────────
        _te = _time.perf_counter()
        if merge_end_evt is not None:
            with torch.cuda.stream(merge_stream):
                merge_end_evt.record(merge_stream)
        evt = merge_stream.record_event()
        _tf = _time.perf_counter()
        _lt_add("__weight_prefetch_batch__", "submit_h2d_ms", 1000.0 * (_tb - _ta))
        _lt_add("__weight_prefetch_batch__", "submit_h2d_prepare_ms", 1000.0 * _t_h2d_prepare)
        _lt_add("__weight_prefetch_batch__", "submit_h2d_copy_ms", 1000.0 * _t_h2d_copy)
        _lt_add("__weight_prefetch_batch__", "submit_wrap_array_ms", 1000.0 * (_tc - _tb))
        _lt_add("__weight_prefetch_batch__", "submit_as_array_ms", 1000.0 * _t_as_array)
        _lt_add("__weight_prefetch_batch__", "submit_record_stream_ms", 1000.0 * _t_record_stream)
        _lt_add("__weight_prefetch_batch__", "submit_decode_ms", 1000.0 * (_td - _tc))
        _lt_add("__weight_prefetch_batch__", "submit_merge_meta_ms", 1000.0 * _t_meta)
        _lt_add("__weight_prefetch_batch__", "submit_dlpack_ms", 1000.0 * _t_dlpack)
        _lt_add("__weight_prefetch_batch__", "submit_tensor_record_stream_ms", 1000.0 * _t_record_tensor)
        _lt_add("__weight_prefetch_batch__", "submit_merge_ms", 1000.0 * _t_merge)
        _lt_add("__weight_prefetch_batch__", "submit_record_event_ms", 1000.0 * (_tf - _te))
        _lt_add("__weight_prefetch_batch__", "submit_total_ms", 1000.0 * (_tf - _ta))
        if prof_enabled and merge_end_evt is not None:
            _prof_sync = _os.environ.get("MEMRIFT_PROF_GPU_TIMING_SYNC", "0") == "1"
            if _prof_sync:
                try:
                    merge_end_evt.synchronize()
                    if h2d_start_evt is not None and h2d_end_evt is not None:
                        _lt_add("__weight_prefetch_batch__", "gpu_h2d_ms", float(h2d_start_evt.elapsed_time(h2d_end_evt)))
                    if decode_start_evt is not None and decode_end_evt is not None:
                        _lt_add("__weight_prefetch_batch__", "gpu_decode_ms", float(decode_start_evt.elapsed_time(decode_end_evt)))
                    if merge_start_evt is not None:
                        _lt_add("__weight_prefetch_batch__", "gpu_merge_ms", float(merge_start_evt.elapsed_time(merge_end_evt)))
                except Exception:
                    pass
        if _dbg:
            print(f"[BatchDec] n={len(nvcomp_cps)} h2d={1000*(_tb-_ta):.1f}ms as_arr={1000*(_tc-_tb):.1f}ms dec={1000*(_td-_tc):.1f}ms dlpack={1000*_t_dlpack:.1f}ms merge={1000*_t_merge:.1f}ms evt={1000*(_tf-_te):.1f}ms total={1000*(_tf-_ta):.1f}ms", flush=True)

        # Keepalive: only freshly-allocated CPU pinned bufs (must outlive H2D DMA).
        extra_ka = tuple(p for p in pinned_extras if p is not None)

        for i, cp in enumerate(nvcomp_cps):
            prof_name = f"decoder.layers.{cp.layer_idx}" if getattr(cp, "layer_idx", -1) >= 0 else "__unknown_weight_layer__"
            if i == 0:
                result[cp] = StreamFuture(bf16_list[i], evt, extra_ka, prof_name=prof_name)
            else:
                result[cp] = StreamFuture(bf16_list[i], evt, (), prof_name=prof_name)

        return result

    def materialize_on_stream(
        self,
        exp_mv: bytes,
        sm_gpu: torch.Tensor,
        orig_shape: tuple,
        dtype: torch.dtype,
        comp_pinned: Optional[torch.Tensor] = None,
    ) -> 'StreamFuture':
        """
        Submit decompression on the calling (main) thread via CUDA streams.

        Eliminates the thread-pool overhead (~18-23 ms/component) that was the
        primary bottleneck in the nvCOMP path.  All CUDA commands are issued
        to h2d_stream from the calling thread; the GPU executes them
        asynchronously while the training stream continues.

        Returns a StreamFuture immediately.  Consuming it via .result() inserts
        a GPU-side wait_event so the training stream does not read the weight
        before decompression is complete – the CPU never blocks.

        Supports the same decompression paths as materialize_async():
        - nvCOMP LZ4 (``NVL4``): pure-GPU path on ``h2d_stream``
        - nvCOMP ANS (``NVAN``, e.g. ``prepare_weight --compression nvcomp_ans``)
        - zstd CPU path (legacy / backward compat)

        Args:
            exp_mv:       Compressed exponent bytes (``NVL4`` / ``NVAN`` / raw zstd).
            sm_gpu:       Sign+mantissa tensor already on GPU.
            orig_shape:   Original weight tensor shape.
            dtype:        Target dtype (usually bfloat16).
            comp_pinned:  Persistent pinned buffer (cp.comp_pinned).  When
                          provided the H2D copy is zero-copy from the perspective
                          of pinned-memory allocation overhead.
        """
        if self._fs_sp is None:
            raise RuntimeError("CUDA extension not available")

        fs_sp = self._fs_sp
        h2d_stream = self.h2d_stream
        numel = int(np.prod(orig_shape))
        strides = _c_contiguous_strides(orig_shape)

        # ── GPU path: nvCOMP LZ4 ─────────────────────────────────────────
        if (self._nvcomp_codec is not None
                and isinstance(exp_mv, (bytes, bytearray, memoryview))
                and len(exp_mv) >= 4
                and bytes(exp_mv[:4]) == NVCOMP_LZ4_MAGIC):

            # 1. Pinned source buffer (persistent or freshly allocated).
            #    On the main thread, torch.empty(pin_memory=True) is ~0.02 ms
            #    (warm per-thread cache), vs ~10 ms from a cold thread-pool worker.
            if comp_pinned is not None:
                comp_pinned_buf = comp_pinned
                ka_extra = ()
            else:
                comp_data = exp_mv[4:]
                comp_pinned_buf = torch.empty(
                    len(comp_data), dtype=torch.uint8, pin_memory=True)
                comp_pinned_buf.numpy()[:] = np.frombuffer(comp_data, dtype=np.uint8)
                ka_extra = (comp_pinned_buf,)

            # 2. H2D DMA: compressed bytes → GPU (async on h2d_stream)
            _n = comp_pinned_buf.numel()
            comp_gpu = self._acquire_gpu_staging(_n, sm_gpu.device)
            with torch.cuda.stream(h2d_stream):
                comp_gpu.copy_(comp_pinned_buf, non_blocking=True)

            # 3. GPU nvCOMP LZ4 decode (queued on h2d_stream via codec)
            comp_arr = _nvcomp_lib.as_array(comp_gpu.view(torch.int8))
            decomp = self._nvcomp_codec.decode(comp_arr)
            comp_gpu.record_stream(h2d_stream)
            del comp_gpu, comp_arr

            # 4. DLPack → tensor; record_stream defers nvcomp GPU free until clone done.
            decomp_raw = torch.from_dlpack(decomp.to_dlpack()).view(torch.uint8)
            del decomp
            decomp_raw.record_stream(h2d_stream)
            with torch.cuda.stream(h2d_stream):
                exp_gpu = decomp_raw[:numel].clone()
            del decomp_raw

            # 5. GPU merge: exp_gpu + sm_gpu → bf16 (on h2d_stream)
            with torch.cuda.stream(h2d_stream):
                bf16 = fs_sp.merge(
                    exp_gpu, sm_gpu,
                    list(orig_shape), list(strides), 0,
                    dtype, h2d_stream.cuda_stream,
                )
            exp_gpu.record_stream(h2d_stream)
            del exp_gpu

            evt = h2d_stream.record_event()
            # Only keep freshly-allocated CPU pinned buf; GPU intermediates use record_stream.
            return StreamFuture(bf16, evt, ka_extra)

        # ── GPU path: nvCOMP ANS ────────────────────────────────────────────
        if (self._nvcomp_ans_codec is not None
                and isinstance(exp_mv, (bytes, bytearray, memoryview))
                and len(exp_mv) >= 4
                and bytes(exp_mv[:4]) == NVCOMP_ANS_MAGIC):

            if comp_pinned is not None:
                comp_pinned_buf = comp_pinned
                ka_extra = ()
            else:
                comp_data = exp_mv[4:]
                comp_pinned_buf = torch.empty(
                    len(comp_data), dtype=torch.uint8, pin_memory=True)
                comp_pinned_buf.numpy()[:] = np.frombuffer(comp_data, dtype=np.uint8)
                ka_extra = (comp_pinned_buf,)

            _n = comp_pinned_buf.numel()
            comp_gpu = self._acquire_gpu_staging(_n, sm_gpu.device)
            with torch.cuda.stream(h2d_stream):
                comp_gpu.copy_(comp_pinned_buf, non_blocking=True)

            comp_arr = _nvcomp_lib.as_array(comp_gpu.view(torch.int8))
            decomp = self._nvcomp_ans_codec.decode(comp_arr)
            comp_gpu.record_stream(h2d_stream)
            del comp_gpu, comp_arr

            decomp_raw = torch.from_dlpack(decomp).view(torch.uint8)
            del decomp
            decomp_raw.record_stream(h2d_stream)
            with torch.cuda.stream(h2d_stream):
                bf16 = fs_sp.merge(
                    decomp_raw, sm_gpu,
                    list(orig_shape), list(strides), 0,
                    dtype, h2d_stream.cuda_stream,
                )
            del decomp_raw

            evt = h2d_stream.record_event()
            return StreamFuture(bf16, evt, ka_extra)

        # ── CPU path: zstd (legacy / backward compat) ─────────────────────
        memrift_require_nvcomp_weight_payload(exp_mv)
        cpu_exp = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
        dctx = get_decompression_ctx()
        with dctx.stream_reader(memoryview(exp_mv)) as reader:
            view = memoryview(cpu_exp.numpy())
            nread = reader.readinto(view)
            assert nread == numel, "decompress size mismatch"

        with torch.cuda.stream(h2d_stream):
            bf16 = fs_sp.merge(
                cpu_exp, sm_gpu,
                list(orig_shape), list(strides), 0,
                dtype, h2d_stream.cuda_stream,
            )
        evt = h2d_stream.record_event()
        return StreamFuture(bf16, evt, cpu_exp)


class StreamFuture:
    """Future-like wrapper for CUDA-stream-based decompression results.

    Allows decompression results already submitted to a CUDA stream to be
    consumed by code that expects the ``concurrent.futures.Future`` interface
    (i.e. ``.done()`` / ``.result()``).

    ``.result()`` inserts a GPU-side ``wait_event`` on the *current* CUDA
    stream so that downstream GPU kernels automatically wait for decode to
    finish – the CPU thread never blocks.
    """

    def __init__(self, bf16: torch.Tensor, event: torch.cuda.Event, keepalive, prof_name: str = ""):
        self._bf16 = bf16
        self._event = event
        self._keepalive = keepalive
        self._prof_name = prof_name

    def done(self) -> bool:
        """Always True – CUDA commands are already submitted."""
        return True

    def result(self):
        """Insert GPU-side wait and return (bf16, event, keepalive)."""
        import time as _time
        _t0 = _time.perf_counter()
        with _memrift_nvtx_range("MemRift/StreamFuture.wait_event"):
            torch.cuda.current_stream().wait_event(self._event)
        _t1 = _time.perf_counter()
        if self._prof_name:
            _lt_add(self._prof_name, "prefetch_future_consume_ms", 1000.0 * (_t1 - _t0))
            _lt_add(self._prof_name, "prefetch_future_wait_event_cpu_ms", 1000.0 * (_t1 - _t0))
        ret = (self._bf16, self._event, self._keepalive)
        # Free keepalive immediately: GPU intermediates were managed by record_stream;
        # CPU pinned bufs can be freed once H2D DMA has submitted (evt already fired
        # or the stream dependency ensures correctness).
        self._keepalive = ()
        return ret

    def cancel(self) -> bool:
        return False


class PrefetchFuture:
    """Thread-safe future for background-thread decode results.

    The background decode thread fills ``_bf16`` and records ``_merge_evt``
    on its private ``bg_merge_stream``, then calls ``set_result()`` which
    sets the Python ``threading.Event`` (``_ready``).

    The main thread calls ``result()`` which:
      1. Briefly waits on ``_ready`` (CPU threading.Event – near-zero latency
         when the bg thread has finished ahead of the training stream).
      2. Inserts a GPU-side ``training_stream.wait_event(_merge_evt)`` so the
         training kernels automatically wait for the bg merge to complete.
      3. Returns ``(bf16, merge_evt, ())``.

    This keeps the main hook thread non-blocking (~0.01 ms per hook) while
    the GPU still gets the correct ordering via CUDA events.
    """

    def __init__(self, merge_evt: torch.cuda.Event, prof_name: str = ""):
        self._merge_evt = merge_evt
        self._bf16: Optional[torch.Tensor] = None
        self._ready = threading.Event()
        self._prof_name = prof_name

    def done(self) -> bool:
        return self._ready.is_set()

    def set_result(self, bf16: Optional[torch.Tensor]):
        """Called by bg thread after merge is submitted and merge_evt recorded."""
        self._bf16 = bf16
        self._ready.set()

    def result(self):
        """Wait for bg submission (CPU), insert GPU-side wait, return (bf16, evt, ())."""
        import time as _time, os as _os
        _t0 = _time.perf_counter()
        _pf_dbg = _os.environ.get("MEMRIFT_MAT_DBG19", "0") == "1" and "L19" in self._prof_name
        if _pf_dbg:
            print(f"[PF_DBG L19] result() enter: _ready.is_set()={self._ready.is_set()}", flush=True)
        with _memrift_nvtx_range("MemRift/PrefetchFuture.ready_wait"):
            self._ready.wait()  # CPU wait – >0 说明 bg decode 慢于消费，考虑加深 prefetch
        _t1 = _time.perf_counter()
        if _pf_dbg:
            print(f"[PF_DBG L19] _ready.wait() took {(_t1-_t0)*1000:.1f}ms", flush=True)
        with _memrift_nvtx_range("MemRift/PrefetchFuture.wait_merge_evt"):
            torch.cuda.current_stream().wait_event(self._merge_evt)
        _t2 = _time.perf_counter()
        if self._prof_name:
            _lt_add(self._prof_name, "prefetch_future_consume_ms", 1000.0 * (_t2 - _t0))
            _lt_add(self._prof_name, "prefetch_future_ready_wait_ms", 1000.0 * (_t1 - _t0))
            _lt_add(self._prof_name, "prefetch_future_wait_event_cpu_ms", 1000.0 * (_t2 - _t1))
        return (self._bf16, self._merge_evt, ())

    def cancel(self) -> bool:
        return False
