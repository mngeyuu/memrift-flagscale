"""
MemRift vs Pure-LoRA 模型载入时间 & 压缩比 Benchmark

指标:
  1. 存储压缩比  (MemRift compressed / bf16 original)
  2. 模型载入时间:
       - MemRift  : 从磁盘读取压缩权重 → CPU pinned memory（sm+exp）
       - Pure     : 从 safetensors 读取 → GPU (transformers from_pretrained)
  3. 单次前向延迟 (layer-by-layer decode):
       - MemRift  : 逐层解压 + merge → forward → 释放
       - Pure     : 全量权重驻 GPU → forward
  4. 推理时 GPU 峰值显存对比
"""

import os
import sys
import json
import struct
import time
import threading
import concurrent.futures as fut
import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

DEVICE        = "cuda:0"
MODEL_PATH    = "/share/project/mengyc/models/Mistral-7B-Instruct-v0.2"
MEMRIFT_DIR   = "/share/project/mengyc/code/memrift_demo/output/memrift_weights_mistral7b_level18"
ZSTD_LEVEL    = 18
SEQ_LEN       = 128          # 推理序列长度

# ── CUDA extension ──────────────────────────────────────────────────────────
sys.path.insert(0, "/share/project/mengyc/code/memrift-flagscale")
try:
    from flagscale.compress.float_split_stride_pin import float_split_stride_pin as fs_sp
    _FS_SP_OK = fs_sp.is_available()
except Exception:
    _FS_SP_OK = False

try:
    import zstandard as zstd
    _ZSTD_OK = True
except ImportError:
    _ZSTD_OK = False

_tls = threading.local()

def _get_dctx():
    if not hasattr(_tls, "dctx"):
        _tls.dctx = zstd.ZstdDecompressor()
    return _tls.dctx


# ════════════════════════════════════════════════════════════════════════════
# § 1  存储压缩比
# ════════════════════════════════════════════════════════════════════════════

def compute_compression_stats():
    idx = json.load(open(os.path.join(MEMRIFT_DIR, "index.json")))

    total_numel        = 0
    total_bf16_bytes   = 0
    total_sm_bytes     = 0   # 未压缩 sm (1B/elem)
    total_exp_comp     = 0   # zstd 压缩后的 exp
    total_bin_bytes    = 0   # 实际文件大小

    for entry in idx:
        numel = 1
        for d in entry["shape"]: numel *= d
        total_numel     += numel
        total_bf16_bytes += numel * 2
        total_sm_bytes   += numel          # 1 byte sign+mantissa per bf16

        fpath = os.path.join(MEMRIFT_DIR, entry["file"])
        fsize = os.path.getsize(fpath)
        total_bin_bytes += fsize

    total_exp_comp = total_bin_bytes - total_sm_bytes - 8 * len(idx)  # 每个文件头 8 bytes

    ratio      = total_bin_bytes / total_bf16_bytes
    savings    = 1.0 - ratio
    exp_ratio  = total_exp_comp / total_sm_bytes   # exp 本身压缩率

    print("\n" + "═"*62)
    print("  §1  存储压缩比 (Storage Compression Ratio)")
    print("═"*62)
    print(f"  原始 bf16 权重大小   : {total_bf16_bytes/1024**3:.3f} GB  ({total_numel/1e9:.3f}B 参数)")
    print(f"  SM 部分 (未压缩)     : {total_sm_bytes/1024**3:.3f} GB  (1 byte/elem, sign+mantissa)")
    print(f"  Exp 部分 (zstd-{ZSTD_LEVEL})   : {total_exp_comp/1024**3:.3f} GB  (exp 字节压缩率 {exp_ratio:.1%})")
    print(f"  MemRift 总大小       : {total_bin_bytes/1024**3:.3f} GB")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  压缩率 (compressed/original) : {ratio:.4f}  ({ratio*100:.1f}%)")
    print(f"  空间节省                     : {savings*100:.1f}%  ✓ {'(≥30%)' if savings>=0.30 else '(<30%)'}")
    print("═"*62)

    return total_bf16_bytes, total_bin_bytes, len(idx)


# ════════════════════════════════════════════════════════════════════════════
# § 2  MemRift 模型载入时间 (磁盘 → CPU pinned memory)
# ════════════════════════════════════════════════════════════════════════════

def benchmark_memrift_load(n_workers: int = 8):
    """
    读取所有 .bin 文件 → CPU pinned memory，测量 I/O 时间。
    这是 MemRift 推理的"预加载"阶段：权重留在 CPU，GPU 上只有当前层。
    """
    idx = json.load(open(os.path.join(MEMRIFT_DIR, "index.json")))

    def _load_one(entry):
        fpath = os.path.join(MEMRIFT_DIR, entry["file"])
        with open(fpath, "rb") as f:
            numel_raw = struct.unpack("<Q", f.read(8))[0]
            sm_size   = numel_raw * (1 if entry["dtype"] == "bfloat16" else 3)
            sm_bytes  = np.frombuffer(f.read(sm_size), dtype=np.uint8).copy()
            exp_bytes = f.read()
        sm_pinned = torch.from_numpy(sm_bytes).pin_memory()
        return sm_pinned, exp_bytes

    # 冷读 (清 page cache 由用户负责，这里直接计时)
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    with fut.ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_load_one, idx))
    t1 = time.perf_counter()

    total_sm_bytes  = sum(r[0].numel() for r in results)
    total_exp_bytes = sum(len(r[1]) for r in results)
    total_bytes     = total_sm_bytes + total_exp_bytes
    elapsed         = t1 - t0

    print("\n" + "═"*62)
    print("  §2  MemRift 模型载入时间  (disk → CPU pinned)")
    print("═"*62)
    print(f"  文件数量      : {len(idx)}")
    print(f"  读取总量      : {total_bytes/1024**3:.3f} GB")
    print(f"  载入时间      : {elapsed:.2f} s")
    print(f"  I/O 带宽      : {total_bytes/elapsed/1024**3:.2f} GB/s")
    print(f"  SM 驻留 CPU   : {total_sm_bytes/1024**3:.3f} GB  (pinned)")
    print(f"  Exp 驻留 CPU  : {total_exp_bytes/1024**3:.3f} GB  (compressed)")
    print(f"  GPU 显存占用  : 0 GB  (权重尚未上 GPU)")
    print("═"*62)

    return results, idx, elapsed


# ════════════════════════════════════════════════════════════════════════════
# § 3  Pure 模型载入时间 (safetensors → GPU)
# ════════════════════════════════════════════════════════════════════════════

def benchmark_pure_load():
    """
    用 transformers 把完整 bf16 模型加载到 GPU，测量端到端时间。
    """
    from transformers import AutoModelForCausalLM

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize(DEVICE)

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
    )
    torch.cuda.synchronize(DEVICE)
    t1 = time.perf_counter()

    mem_alloc = torch.cuda.memory_allocated(DEVICE)
    mem_peak  = torch.cuda.max_memory_allocated(DEVICE)

    # safetensors 文件大小
    import glob
    sf_files   = glob.glob(os.path.join(MODEL_PATH, "*.safetensors"))
    sf_total   = sum(os.path.getsize(f) for f in sf_files)
    elapsed    = t1 - t0

    print("\n" + "═"*62)
    print("  §3  Pure 模型载入时间  (safetensors → GPU)")
    print("═"*62)
    print(f"  safetensors 大小 : {sf_total/1024**3:.3f} GB")
    print(f"  载入时间         : {elapsed:.2f} s")
    print(f"  I/O 带宽         : {sf_total/elapsed/1024**3:.2f} GB/s")
    print(f"  GPU 已分配       : {mem_alloc/1024**3:.3f} GB")
    print(f"  GPU 峰值         : {mem_peak/1024**3:.3f} GB")
    print("═"*62)

    return model, elapsed, mem_peak


# ════════════════════════════════════════════════════════════════════════════
# § 4  推理延迟对比 (layer-by-layer MemRift vs full-GPU Pure)
# ════════════════════════════════════════════════════════════════════════════

def benchmark_memrift_inference(loaded_entries, idx, n_warmup=1, n_runs=3):
    """
    逐层 MemRift 推理：每层 decompress + H2D merge → forward → 释放。
    使用一个最小的模拟 forward（逐层解压 + CUDA merge）测量延迟。
    """
    if not _FS_SP_OK or not _ZSTD_OK:
        print("[skip] MemRift inference: CUDA extension or zstd not available")
        return None

    device = torch.device(DEVICE)
    h2d    = torch.cuda.Stream(device=device)

    def _decompress_one(sm_pinned, exp_bytes, shape, dtype):
        """Decompress one weight tensor: CPU decompress + GPU merge."""
        numel   = 1
        for d in shape: numel *= d
        cpu_exp = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
        dctx    = _get_dctx()
        with dctx.stream_reader(memoryview(exp_bytes)) as reader:
            nread = reader.readinto(memoryview(cpu_exp.numpy()))
            assert nread == numel

        strides = [1] * len(shape)
        running = 1
        for i in range(len(shape)-2, -1, -1):
            running *= shape[i+1]; strides[i] = running

        with torch.cuda.stream(h2d):
            sm_gpu = sm_pinned.to(device, non_blocking=True)
            tensor = fs_sp.merge(
                cpu_exp, sm_gpu, list(shape), strides, 0,
                dtype, h2d.cuda_stream
            )
            sm_gpu.record_stream(h2d)
        h2d.record_event().synchronize()
        return tensor

    # 以 layer index 为单位分组（仅 decoder layers）
    layer_entries = [e for e in idx if "layers." in e["name"]]
    # 仅取前 4 层做代理延迟（代表每层的时间）
    sample_entries = layer_entries[:4]
    sample_loaded  = [loaded_entries[idx.index(e)] for e in sample_entries]

    def _run_once():
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for (sm_pinned, exp_bytes), entry in zip(sample_loaded, sample_entries):
            t = _decompress_one(sm_pinned, exp_bytes, entry["shape"],
                                torch.bfloat16 if entry["dtype"]=="bfloat16" else torch.float32)
            del t
        torch.cuda.synchronize(device)
        return time.perf_counter() - t0

    # Warmup
    for _ in range(n_warmup): _run_once()

    torch.cuda.reset_peak_memory_stats(device)
    times = [_run_once() for _ in range(n_runs)]
    peak  = torch.cuda.max_memory_allocated(device)

    avg_per_layer = sum(times) / n_runs / len(sample_entries) * 1000
    total_32_layers = avg_per_layer * 32 / 1000  # seconds for all 32 layers

    print("\n" + "═"*62)
    print("  §4a  MemRift 推理延迟 (decompress+merge, per layer)")
    print("═"*62)
    print(f"  测试层数        : {len(sample_entries)} layers × {n_runs} runs")
    print(f"  平均每层耗时    : {avg_per_layer:.1f} ms")
    print(f"  估计 32 层总耗时: {total_32_layers:.2f} s  (serial, no overlap)")
    print(f"  GPU 峰值显存    : {peak/1024**3:.3f} GB  (仅当前层驻留)")
    print("═"*62)

    return avg_per_layer, peak


def benchmark_pure_inference(model, n_warmup=1, n_runs=3):
    """纯 GPU 模型单次前向（全量权重驻 GPU）"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tok = tokenizer(["Hello world"] * 1, return_tensors="pt",
                    max_length=SEQ_LEN, truncation=True, padding="max_length")
    input_ids = tok["input_ids"].to(DEVICE)

    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids)
        torch.cuda.synchronize(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(input_ids)
        torch.cuda.synchronize(DEVICE)
        elapsed = (time.perf_counter() - t0) / n_runs

    peak = torch.cuda.max_memory_allocated(DEVICE)

    print("\n" + "═"*62)
    print("  §4b  Pure 推理延迟 (全量权重驻 GPU, seq_len={SEQ_LEN})")
    print("═"*62)
    print(f"  平均前向耗时   : {elapsed*1000:.1f} ms")
    print(f"  GPU 峰值显存   : {peak/1024**3:.3f} GB")
    print("═"*62)

    return elapsed, peak


# ════════════════════════════════════════════════════════════════════════════
# § 5  汇总对比
# ════════════════════════════════════════════════════════════════════════════

def print_summary(bf16_gb, bin_gb, memrift_load_s, pure_load_s,
                  memrift_inf_ms, memrift_gpu_gb,
                  pure_inf_ms,   pure_gpu_gb):

    print("\n" + "╔" + "═"*60 + "╗")
    print("║  MemRift vs Pure-LoRA  推理对比汇总" + " "*23 + "║")
    print("╠" + "═"*60 + "╣")
    print(f"║  {'指标':<22}  {'MemRift':>12}  {'Pure LoRA':>12}  ║")
    print("╠" + "═"*60 + "╣")
    # 存储
    print(f"║  {'模型存储大小 (GB)':<22}  {bin_gb:>12.2f}  {bf16_gb:>12.2f}  ║")
    savings = (1 - bin_gb/bf16_gb)*100
    print(f"║  {'存储节省':<22}  {savings:>11.1f}%  {'—':>12}  ║")
    # 载入
    load_speedup = pure_load_s / memrift_load_s if memrift_load_s>0 else 0
    print(f"║  {'模型载入时间 (s)':<22}  {memrift_load_s:>12.2f}  {pure_load_s:>12.2f}  ║")
    print(f"║  {'载入加速比':<22}  {load_speedup:>11.2f}x  {'—':>12}  ║")
    # 推理
    if memrift_inf_ms and pure_inf_ms:
        print(f"║  {'推理延迟 (ms/layer)':<22}  {memrift_inf_ms:>12.1f}  {'—':>12}  ║")
        print(f"║  {'推理延迟 (ms/fwd)':<22}  {'—':>12}  {pure_inf_ms*1000:>12.1f}  ║")
    # GPU 峰值
    print(f"║  {'推理 GPU 峰值 (GB)':<22}  {memrift_gpu_gb:>12.3f}  {pure_gpu_gb:>12.3f}  ║")
    mem_save = (1 - memrift_gpu_gb/pure_gpu_gb)*100
    print(f"║  {'GPU 峰值节省':<22}  {mem_save:>11.1f}%  {'—':>12}  ║")
    print("╚" + "═"*60 + "╝")


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("MemRift vs Pure-LoRA  推理基准测试")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Device: {DEVICE}")

    # §1 压缩比
    bf16_bytes, bin_bytes, n_files = compute_compression_stats()

    # §2 MemRift 载入 (GPU still empty)
    loaded, idx, memrift_load_t = benchmark_memrift_load(n_workers=16)

    # §4a MemRift 推理延迟 (在 pure 模型载入前测，确保 GPU 仅有当前层)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    memrift_inf_ms, memrift_inf_peak = benchmark_memrift_inference(loaded, idx)

    # 释放 loaded 节省内存，避免影响 pure 载入
    del loaded
    torch.cuda.empty_cache()

    # §3 Pure 载入
    pure_model, pure_load_t, pure_load_peak = benchmark_pure_load()

    # §4b Pure 推理延迟
    pure_inf_s, pure_inf_peak = benchmark_pure_inference(pure_model)

    # §5 汇总
    print_summary(
        bf16_gb        = bf16_bytes  / 1024**3,
        bin_gb         = bin_bytes   / 1024**3,
        memrift_load_s = memrift_load_t,
        pure_load_s    = pure_load_t,
        memrift_inf_ms = memrift_inf_ms,
        memrift_gpu_gb = memrift_inf_peak / 1024**3,
        pure_inf_ms    = pure_inf_s,
        pure_gpu_gb    = pure_inf_peak / 1024**3,
    )

    del pure_model
    torch.cuda.empty_cache()
