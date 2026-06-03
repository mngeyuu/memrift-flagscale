# MemRift 速度回归修复 — 激活压缩 zstd level 与权重解耦

日期：2026-06-03
状态：待审

## 1. 背景与根因（已用实验+代码核实）

权重路径修复后（见 `2026-06-02-memrift-weight-te-backward-design.md`），memrift 训练正确且省显存，
但**慢**。单变量实验（TinyLlama-1.1B bs1，env myc，TP=1）定位根因：

| 配置 | 每 iter |
|---|---|
| weight-only memrift（关激活压缩） | **1.8s** |
| full memrift @ 激活 zstd level **18** | **117.8s** |
| full memrift @ 激活 zstd level **1**（仅改 level） | **~3.7s**（32×） |

**根因：激活压缩使用了 zstd level 18。** level 18 是离线权重压缩级别（压缩吞吐仅 ~10MB/s），
用于在线压缩每层大激活灾难性慢。权重路径本身快（1.8s）。

代码核实（为什么 `--memrift-zstd-level 18` 会作用到激活）：
- `train_hooks.py:119` 读 `memrift_zstd_level` → `:185` 创建**唯一共享 `AsyncCompressor(zstd_level=18)`**，
  同实例传给权重预取(`:197`)与激活(`:217`)。
- `AsyncCompressor.zstd_level` 类内**只**用于 `kickoff_sync:176` / `kickoff_async:233`（=激活压缩）；
  权重预取 `materialize_async`（297-387）只 `get_decompression_ctx()`（解压，level 无关），**不碰 zstd_level**。
- 训练期 `memrift_zstd_level` 无其他权重运行时用途（权重从磁盘预压加载；`megatron_tp_hooks._compress_param`
  仅 inference 的 embedding/output hook 用，训练不调用；`train_hooks.py:499` 属 inference 注入）。

结论：训练期 `--memrift-zstd-level` 实际只决定激活压缩速度，与权重无关。这是耦合错误。

## 2. 目标

- 把激活压缩 level 从权重 level 解耦，给激活一个低默认，消除 ~32× 拖慢。
- 不改变权重路径与显存行为。
- 最小改动。

## 3. 方案（Approach A）

新增独立参数 `--memrift-act-zstd-level`（int，默认 **3**），仅驱动激活压缩；
`--memrift-zstd-level` 保留原语义（匹配离线 prepare_weight 的 level、供 inference/TP 路径），
不再喂给运行时激活压缩器。

默认 3 的理由：zstd 1~3 压缩吞吐均 >100MB/s（实测 level 1 → 3.7s/iter），3 比 1 压缩比略好仍很快；可调。

## 4. 改动清单

| 文件 | 改动 |
|---|---|
| `flagscale/train/megatron/training/arguments_fs.py` | 在 `--memrift-zstd-level`(:954-959) 旁新增 `--memrift-act-zstd-level`（type=int, default=3, help 说明仅用于激活压缩） |
| `flagscale/compress/memrift/train_hooks.py` | `inject_memrift_if_configured`：新增读 `act_zstd_level = getattr(args, "memrift_act_zstd_level", 3)`；创建共享 `AsyncCompressor` 时（:185）改用 `zstd_level=act_zstd_level`（该 compressor 的 level 只governs激活压缩）；`_inject_activation_compression` 的 fallback 默认（:396 `zstd_level = 18`）改为接收/使用 `act_zstd_level`；`get_memrift_status` 增列该项 |

注：权重预取共享同一 compressor 实例无碍（它不使用 level）；无需拆分实例。

## 5. 验证（本次用 TinyLlama；Mistral-7B 留作之后）

诊断脚本 `scripts/diag_tinyllama_weight_mem.sh`（已支持 `ZLEVEL` 覆盖，可临时验证；本设计落地后默认即低）：
1. TinyLlama full memrift @ 默认（act-level 3）：每 iter **~3.7s**（vs level18 的 117.8s），无崩溃、loss 正常。
2. 峰值仍 ~3739 MB（不变；权重主导），weight-only 仍 ~1.8s（不受影响）。
3. 确认未显式传 `--memrift-act-zstd-level` 时取默认 3；显式传 18 可复现旧慢速（开关有效）。

Mistral-7B-v0.2 复测（act-level 3 vs LoRA 峰值/速度）：**之后单独进行**（用户指定）。

## 6. 风险

- 低 level → 激活压缩比下降 → 激活内存省得少。TinyLlama 实测峰值不受 level 影响（权重主导）；
  长 seq / 大 batch 下激活才主导，届时若需更高压缩比可调高 `--memrift-act-zstd-level`（速度/内存权衡留给用户）。
- `--memrift-zstd-level` 语义变化（不再影响激活）需在 help 文案中讲清，避免误解。

## 7. 非目标

- 不引入新编码器（lz4）——作为后续可选项（Approach B）。
- 不改激活压缩的 overlap/调度机制。
- 不动权重路径。
