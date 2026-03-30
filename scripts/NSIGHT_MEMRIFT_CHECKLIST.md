# Nsight Systems MemRift Profile 检查清单

用 Nsight Systems 打开 `benchmark_profile_artifacts/nsight/memrift_llama11b_nsight.nsys-rep` 后，按下面三项在 timeline 里重点检查。

---

## 1. 数据是否每 step 从 CPU 传 GPU

**在 Timeline 里看：**  
**CUDA Memory** → **HtoD memcpy**（Host-to-Device）

- 点击某个 HtoD memcpy，看 **Size**。
- **若单次 >100MB**：说明每 step 都在搬运大块数据，可能有不必要的整层/整块 CPU→GPU 拷贝。

**当前 profile 的统计（248 机器）：**

| Operation              | Total (MB) | Count | Avg (MB) | Max (MB) |
|------------------------|------------|-------|----------|----------|
| HtoD memcpy            | 1100.07    | 371   | 2.97     | **65.54**|
| D2D memcpy             | 23.43      | 222   | 0.11     | 0.36     |
| D2H memcpy             | 0.001      | 19    | ~0       | ~0       |

- 单次 HtoD **最大约 65.5 MB**（未到 100MB，但 371 次合计 1.1GB，需在 timeline 看是否集中在每 step 内）。
- **建议**：在 timeline 里按 step 看 HtoD 的分布；若每 step 都有多次几十 MB 的 HtoD，仍算“每 step 大块数据从 CPU 到 GPU”，需要从数据/prefetch 设计上优化。

---

## 2. 是否没有 prefetch（copy 与 compute 是否重叠）

**看 timeline 上 copy 与 compute 的先后关系：**

- **Prefetch 正常**：应是  
  **copy copy copy**（预取下一批）  
  **compute**（当前算）  
  **copy copy copy**  
  **compute**  
  → copy 与 compute 有重叠，下一批在算当前时已在传。

- **Prefetch 缺失/无效**：  
  **copy → compute → copy → compute**  
  → 串行，没有重叠，GPU 会等 copy 完再算。

**操作**：在 Nsight 里同时看 **CUDA Memory (HtoD)** 与 **GPU Compute** 轨道，确认是“成批 copy + 成批 compute”还是“一次 copy 一次 compute”。

---

## 3. MemRift decode 是否阻塞

**在 CPU timeline 里找：**

- **decompress** / **decode** / **zstd** 等与解压相关的活动。

**判断：**

- 若 **decode 时间 > compute 时间**：说明 CPU 解压成为瓶颈，GPU 在等数据，属于 runtime pipeline 设计问题（应尽量异步/重叠 decode 与 compute）。
- 若 decode 与 compute 在时间上重叠良好，且 decode 总时长小于 compute，则 pipeline 较健康。

**操作**：在 Nsight 的 **CPU** 视图中搜索/筛选与 zstd、decompress、decode 相关的 API 或名称，量一下总时长和与 GPU compute 的相对位置。

---

## 快速复现统计

在 248 机器上对当前 profile 重新生成内存统计：

```bash
cd /share/project/mengyc/flagScale/FlagScale
nsys stats benchmark_profile_artifacts/nsight/memrift_llama11b_nsight.nsys-rep --report cuda_gpu_mem_size_sum
```

看 **HtoD** 的 Count / Total / Max 是否与上表一致，以及是否新跑 profile 后变化。
