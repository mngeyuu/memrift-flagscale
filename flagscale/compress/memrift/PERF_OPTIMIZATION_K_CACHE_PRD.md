# MemRift 性能优化 PR 说明（显存预算 K 层 GPU 缓存 + 同步瘦身）

## 1. 背景与问题陈述

Nsight 等 profile 显示存在：

- 主路径或预取路径上 **`cudaStreamSynchronize` / CPU 侧等待** 与 GPU 空档对齐，overlap 不足；
- **`cudaMallocAsync`（或等价临时 GPU 缓冲）** 调用频繁，部分来自解压/物化临时缓冲；
- **Host→Device 传输次数或累计字节** 偏高，与「层用完后释放、下一轮再次全链路物化」的行为一致。

本 PR 系列**不追求**脱离 MemRift「以 CPU/压缩换显存」的定位；在**给定显存预算**下，用**有界缓存**与**事件驱动序**减少重复工作与无谓阻塞。

## 2. 目标（可验收）

| 指标 | 说明 | 验收方式 |
|------|------|----------|
| G1 | 在固定 `K` 与固定训练配置下，**迭代 wall time** 相对基线下降（目标由实验给出，**不设全局 2～3× 承诺**） | 同机同配置 ≥3 次取中位数 |
| G2 | **峰值 `torch.cuda.max_memory_allocated()`** 不超过基线 + **预算 `B` MB**（`B` 由 `K` 与模型推导上界后写入配置） | 每 step 或每 iter 记录峰值 |
| G3 | **loss/grad 数值**：与基线对齐（允许严格 fp 下的微小噪声，需在 PR 中写明阈值） | 同 seed 短跑对比或 bit-wise 策略说明 |
| G4 | Nsight：**`wait_prefetch` 墙钟**或 **主线程在 Future 上的阻塞**相对缩短；**重复 H2D**（同层同 iter 内）减少 | 前后各一份 `.nsys-rep` + `nsys stats` |

## 3. 非目标与约束

- **不**以「关闭所有同步」为手段；**不**引入无法证明正确性的全异步。
- **不**默认无界缓存整网权重导致 MemRift **失去省显存意义**。
- **不**将 **TF32 / PCIe Gen4** 等列为本 PR 的硬 KPI（可与 Megatron 既有精度配置并存，单独 PR 讨论）。
- 仅保证 **TP=1** 等 MemRift v1 已声明约束；多卡行为不在首版范围。

## 4. 方案概述

### 4.1 有界 GPU 权重缓存（核心与显存预算绑定）

- **语义**：最近使用过的、已物化到 GPU 的 **权重张量（或 merge 后的 group）** 在 `release` 时不立即归还分配器，而进入 **容量为 `K` 的 LRU**（或简单 FIFO，实现阶段二选一，PR 中固定一种）。
- **参数**：
  - `MEMRIFT_GPU_WEIGHT_CACHE_LAYERS`（或 Hydra `memrift_gpu_weight_cache_layers`）：**非负整数**，`0` 表示关闭（与当前行为一致）。
- **逐出**：第 `K+1` 个不同 key 进入时，逐出最久未使用项，**显式释放** GPU 张量，保证峰值有上界。
- **Key 设计**：必须能唯一标识「同一逻辑权重」；建议 `(layer_idx, merge_group_id, version)` 或等价，避免错误复用。

### 4.2 同步：从「CPU 阻塞」迁到「流事件」

- **原则**：仅当 **消费侧**（训练 default stream 或 TE 提交 kernel 的流）能通过 **`wait_event`** 建立与 **H2D/decode/merge** 的序时，才删除对应 **CPU `synchronize` / `result()` 阻塞**。
- **范围**：按调用点逐笔修改（`CompressedParam.materialize` / `wait_ready` / `PrefetchFuture.result` 等），**每一笔**需注释「序由哪一 event 保证」。
- **禁止**：全局环境变量「禁用所有同步」。

### 4.3 临时缓冲复用（可选第二阶段）

- 在 **不改变 nvCOMP 语义** 的前提下，对 `materialize_on_stream` / `prefetch_batch_on_stream` 中 **固定 shape** 的路径做 **per-stream 或 per-thread 缓冲池**（上限字节数可配），减少 `cudaMallocAsync` 抖动。
- 若 Python nvCOMP **无法 decode-into-user-buffer**，本项**不**承诺消除 DLPack 路径的全部分配。

## 5. 实现清单（建议拆 PR）

| PR | 内容 | 风险 |
|----|------|------|
| PR-A | 仅 **事件化同步** + NVTX/计数字段；**不**加缓存 | 中（正确性） |
| PR-B | **`K` 层 LRU**，单元测试 + 8B 冒烟 | 中高（显存、生命周期） |
| PR-C | **缓冲复用**（有上限） | 中 |

首版可只合并 PR-A，达标后再 PR-B。

## 6. 必测项（合并前）

1. **TinyLlama / 或小模型**：`K=0` 与 `K=1,2` 对比，迭代时间与峰值显存。
2. **Llama 8B MemRift weight-only**（或与用户主线一致的一条配置）：`K=0` vs `K=用户选定`，**G2 不超标**。
3. **Nsight**：各一条短 trace，对比 `wait_prefetch`、HtoD 次数或总字节（注明是否同 iter 数）。
4. **回归**：无 MemRift、纯 LoRA 配置 **零行为变化**（不 import 缓存模块或 `K=0` 默认）。

## 7. 回滚条件

- 任一 **正确性** 失败（loss 异常、NaN、checkpoint 不一致）。
- **峰值显存** 超过基线 + 预算 `B`（配置错误 or LRU key 错误导致重复驻留）。
- **性能回退**：在相同 `K` 下迭代时间 **劣于基线 >5%**（可调）且 Nsight 无明确解释——**默认关缓存 `K=0`**，保留事件同步改进若独立可留。

## 8. 文档与配置

- `train.system` / `arguments_fs.py`：增加 `--memrift-gpu-weight-cache-layers`（默认 `0`）。
- `get_memrift_status`：打印当前 `K` 与缓存命中统计（若实现统计）。
- 本文档路径：`flagscale/compress/memrift/PERF_OPTIMIZATION_K_CACHE_PRD.md`。

## 9. 与先前「四大问题」方案的差异（摘要）

- **不用**「禁用所有显式同步」；改用 **可证明的 event 序**。
- **不用**无界 GPU 权重缓存；改用 **`K` + LRU + 峰值预算 `B`**。
- **不把** TF32/PCIe 与 2～3× 加速写进本 PR 承诺；**以 G1–G4 实测为准**。
