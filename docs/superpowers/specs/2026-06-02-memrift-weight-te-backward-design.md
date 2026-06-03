# MemRift 权重处理与 Megatron/TE backward 时机冲突 — 修复设计

日期：2026-06-02
状态：待审

## 1. 背景

memrift-demo（native PyTorch + HF safetensors）的权重处理思路：

- 离线把每个 bf16 权重用 CUDA kernel `float_split_stride_pin` 拆成 `sm`（符号+尾数，1 B/elem，GPU 常驻）+ `exp`（指数，1 B/elem，zstd 压缩存 CPU）。
- 在线用 `CompressedParam`（0 元素占位）+ 逐 `DecoderLayer` 的 forward/backward hook 预取/物化/释放，使任一时刻仅少数层的完整 bf16 权重驻留 GPU。

该思路已移植进 memrift-flagscale（`flagscale/compress/memrift/`，约 4200 行）。本设计只解决移植后与 Megatron/TransformerEngine(TE) 的一个核心冲突。

## 2. 问题（已用实验坐实）

环境：TinyLlama-1.1B，TP=1，seq=2048，mbs=1，`transformer_impl=transformer_engine`，LoRA（4 个 target 全开）。诊断脚本：`scripts/diag_tinyllama_weight_mem.sh`。

| 指标 | 纯 LoRA | memrift(canonical) |
|---|---|---|
| inject 后静态显存 | 2304 MB | 1236 MB（压缩生效，省 ~1 GB） |
| iter0 峰值 | 5674 MB | backward 崩溃，到不了峰值 |
| 结果 | 跑完 3 iter | `RuntimeError: WeightPlaceholder: weight not materialized at backward time for mlp.linear_fc1 (layer 21)` |

`MEMRIFT_HOOK_ORDER=1` 实测 backward 触发序列（最后一层 21）：

```
bwd_pre  layer=21                                  # 模块级 full_backward_pre_hook：materialize 本层
bwd_post layer=21                                  # 模块级 full_backward_hook：立刻 CLEAR 本层
lin_bwd_pre layer=21 mod=TERowParallelLinear empty=True   # TE 融合 Function 此刻才真正 backward，权重已空
lin_bwd_post layer=21
```

## 3. 根因

`activation_compression.py:_unpack` 处理 `WeightPlaceholder` 时（第 42–47 行）读取
`group.target_module.weight.data`，**要求权重在 backward 消费的那一刻已被 re-materialize**。

但 Megatron+TE 把整层计算融进自定义 autograd Function（`_LayerNormMLP`、`_LayerNormLinear`、`_Linear`），其 backward **真正消费权重的时机**与 PyTorch module 级 `full_backward_pre_hook` / `full_backward_hook` 的触发时机**解耦**：module hook 在该 module 的梯度 I/O 边界成对、紧挨着触发（materialize→clear），发生在 TE 融合 Function 真正 backward 之前。

demo（native HF）成立，是因为 HF 的 module backward hook 恰好在该层 backward 完成时触发；TE 打破了这个前提。靠外部 hook "事先 re-materialize" 是在与 TE 融合 autograd 抢时序：

- TinyLlama TP=1：抢输 → **崩溃**（本文复现）。
- mistral / llama8b：per-linear 兜底偶尔抢到 → 不崩，但干净的逐层释放被破坏，权重反复重物化又不及时释放 → **累积、显存没省、峰值反超纯 LoRA**（用户最初观察到的症状）。

两种症状同一根因。`megatron_dynamic_loader.py:1037` 的 `release_all_layers()` 注释已自曝该隐患，但它只在 iteration 末兜底，对 backward 过程中的峰值无效。

## 4. 设计目标

1. 正确性：消除 `WeightPlaceholder: weight not materialized` 崩溃。
2. 内存：backward 阶段权重驻留有界（≈ 1 层），iter0 峰值显著低于纯 LoRA 的 5674 MB。
3. 最小改动、不引入新的时序竞速；保留已生效的 `sm` 压缩与 forward 路径。

## 5. 方案：由 `_unpack` 驱动的按需物化 + 释放滞后一层

核心洞察：**`saved_tensors_hooks` 的 `_unpack` 正好在 TE backward 真正消费权重的那一刻运行**——这是唯一可靠、零竞速的时机。让权重生命周期由 `_unpack` 驱动，而非外部 module hook 抢跑。

### 5.1 按需物化（修崩溃）

`activation_compression.py:_unpack` 中处理 `WeightPlaceholder`：当
`group.target_module.weight.data` 为空时，**当场**调用
`megatron_dynamic_loader.ensure_group_param_materialized(group)`
（该函数已存在，第 412 行：materialize group → 写回 param.data → 注册 `_PTR2GROUP` → 返回 data），随后返回权重。再为空才抛错（真失败）。

无依赖外部 hook，时机精准，单线程同步即可（`sync=True`），避免与 backward 计算流竞争。

### 5.2 释放滞后一层（修峰值）

backward 按层逆序执行（21→0）。当 `_unpack` 为层 L 物化权重时，所有层号 > L 的层均已 backward 完毕，可安全释放。

在 `megatron_dynamic_loader.py` 增设一个轻量全局追踪与 helper：

- `_BWD_ACTIVE_LAYERS`：记录 backward 期间被物化的 (loader, layer_idx) 集合。
- `release_layers_above(layer_idx)`：释放所有层号 > layer_idx 且仍 materialized 的 group（`_clear_param`）。

`_unpack` 物化层 L 后调用 `release_layers_above(L)`。这样 backward 任一时刻权重驻留 ≈ 1 层（外加 TE 融合 Function 内 qkv/proj/fc1/fc2 同层多组，属同层，无碍）。forward 路径已逐层释放，不改。

#### 5.2.1 自愈性质（5.1 兜底 5.2）

5.1 与 5.2 的职责分离是本方案成立的关键：

> **5.1 保证正确性**：无论权重此刻在不在 GPU，`_unpack` 要用就当场物化并返回。
> **5.2 只负责把显存压到 ≈ 1 层**，是纯优化。

因此即便 5.2 "释放过早"（见下 5.2.2），下一次 `_unpack` 发现 `param.data` 为空会**当场再物化一次**返回——**不崩，只是多一次解压（性能代价，非正确性错误）**。等于把现状"抢不赢就崩"换成"抢不准最多多解压一次"。

同层多次 `_unpack` 不会重复物化：首次物化写回 `param.data` 后，后续走快路径（`_materialize_group_tensor:312-315` 见 `param.data.numel()>0` 直接返回；`CompressedParam.materialize:138` 见 `_bf16 is not None` 直接返回）。所以常规逐层逆序下**每层只物化一次、释放一次，无 thrashing**。

#### 5.2.2 释放窗口可调，用实测重物化次数定参

唯一可能触发重复物化的是 TE 的**延迟消费**节点（如 `megatron_dynamic_loader.py:1372` 注释提到的 RowParallelLinear deferred all-reduce backward）：该层被 5.2 释放后，其延迟节点又来 `_unpack`，触发一次重物化。

释放滞后层数 `K`（`release_layers_above(L - (K-1))`）作为可调参数：

- `K=1`（滞后 1 层）：峰值最低，接受极少数 deferred 节点的偶发重物化。
- `K=2`：几乎消除重物化，峰值略升 1 层。
- 关闭 5.2、仅靠 iteration 末 `release_all_layers()` 兜底：零重物化，但 backward 峰值≈全层（退化）。

落地策略：**先 `K=1` 跑验证**，加一条"重物化计数"日志（`_unpack` 走到当场物化分支时计数），量化重物化频率与开销，再决定 `K`。

### 5.3 移除/中和制造竞速的 backward 物化 hook

`install_hooks()` 中的：

- 模块级 `make_bwd_pre`（materialize）+ `make_bwd_post`（clear）——产生 materialize→clear churn，删除其"物化"职责。
- per-linear `make_linear_bwd_pre_all` / `make_linear_bwd_post_all`——已有 `MEMRIFT_DISABLE_LINEAR_BWD_PRE=1` 开关，默认关闭之。

backward 期间的物化/释放全部交给 5.1/5.2。`release_all_layers()` 保留为 iteration 末兜底。

### 5.4 保留项

- `te_ctx_patch.py`：仍需要——处理 TE 经 ctx 属性（非 save_for_backward）持有的权重引用，使其 backward 后及时释放。
- forward 的逐层 materialize/release hook：可靠，保留。
- `sm` GPU 常驻：保留（demo 一致；后续可作独立优化项评估是否改回 pinned-CPU）。

## 6. 改动清单

| 文件 | 改动 |
|---|---|
| `flagscale/compress/memrift/activation_compression.py` | `_unpack` 的 WeightPlaceholder 分支：空权重时调 `ensure_group_param_materialized`（当场物化分支处计数重物化次数）；物化后调 `release_layers_above(L-(K-1))` |
| `flagscale/compress/memrift/megatron_dynamic_loader.py` | 新增 `_BWD_ACTIVE_LAYERS` 追踪 + `release_layers_above()` + 重物化计数器；释放滞后层数 `K`（env `MEMRIFT_BWD_RELEASE_LAG`，默认 1）；`ensure_group_param_materialized` 中登记追踪；`install_hooks()` 移除 backward 物化职责、默认关 per-linear bwd hook |

## 7. 风险与对策

- **ctx 属性路径**：若某些权重 TE 仅经 ctx 属性（绕过 save_for_backward）消费，则不会进 `_unpack`。当前崩溃证明 fc1 走 save_for_backward；其余由 `te_ctx_patch` 覆盖。验证时确认无 CUDA illegal access。
- **同层多组并发**：TE 融合 Function 一次 backward 读同层多权重，均同层 L，`release_layers_above(L)` 不会误删 → 安全。
- **GQA / qkv 合并**：物化走既有 `_materialize_group_tensor`（含 q/k/v cat、GQA 行扩展），逻辑不变。
- **TP/PP**：本次只在 TP=1 复现与验证；释放逻辑按 (loader, local layer_idx) 追踪，多 rank 各自独立，理论兼容，留作后续验证。
- **释放过早 / 重复物化**：常规逐层逆序下仅释放层号 > L 的层，L 自身保留至其所有组 unpack 完成，安全且无重物化。唯一例外是 TE 延迟消费节点（如 RowParallel deferred all-reduce）——此时被 5.1 自愈为"一次重物化"，不崩，仅性能代价。释放滞后层数 `K` 可调（见 5.2.2），先 `K=1` 实测重物化频率再定参。

## 8. 验证计划

用 `scripts/diag_tinyllama_weight_mem.sh`（TinyLlama TP=1）：

1. `MODE=memrift`：跑完 3 iter 不崩（修正确性）。
2. 对比峰值：memrift iter0 峰值（MEM_PROBE K）显著 < 纯 LoRA 的 5674 MB。
3. `MEM_PROBE H_state`：backward 后 materialized 组数 ≈ 0（或仅 1 层）；`H2` 释放后无大幅下降（说明无残留累积）。
4. `MEMRIFT_HOOK_ORDER` 不再出现 `lin_bwd_pre ... empty=True` 导致的失败路径。
5. loss 有限且与纯 LoRA 同量级（不引入数值错误）。
6. 重物化计数：记录 `_unpack` 当场物化分支命中次数。若远大于"层数×组数"（说明 deferred 节点频繁触发重物化），调大 `K` 重测，权衡峰值与重物化开销。

通过后再上 Mistral-7B / Llama-3.1-8B 复测峰值与训练稳定性。

## 9. 非目标（本次不做）

- `sm` 是否改回 pinned-CPU 以进一步降显存（独立优化）。
- TP>1 / PP>1 的系统性验证。
- 激活压缩本身的优化。
