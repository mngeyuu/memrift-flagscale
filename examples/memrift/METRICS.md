# MemRift 指标评测配置说明

本文档说明 `examples/memrift` 下保留的 YAML 配置如何服务
`scripts/metrics` 中的验收指标脚本。

当前仅保留两个模型的单卡训练配置：

- LLaMA-3.1-8B：`conf/train_llama31_8b_mock.yaml`
- Aquila2-7B：`conf/train_aquila2_7b_mock.yaml`

历史 demo、推理、TP、Guanaco、TinyLlama、Mistral、Llama-3.2、MLP-only、
占位配置均已移除。指标脚本会通过 Hydra 覆盖默认配置，统一使用 Alpaca
Megatron indexed dataset、单卡、MemRift 权重+激活异步。

## 1. 全局基线：一律 LoRA

- 所有**精度对比**（PPL、下游任务准确率）的基线均为：**纯 LoRA**（冻结底座 + LoRA，`memrift` 关闭），与线上一致的 rank/alpha/target 模块。
- **不得**用「无 LoRA 全参」或「单独 BF16 全量推理」作为精度基线。
- **MemRift+LoRA** 分支：在相同 LoRA 超参下开启 MemRift 权重（及可选激活）路径。

## 2. PPL

- 在固定验证集上计算因果语言建模困惑度。
- **门控**：`(PPL_mem+lora - PPL_lora) / PPL_lora <= 0.01`（相对劣化不超过 1%）。
- 建议验证集：`wikitext-2-raw-v1` 的 `validation` 子集（可截断样本数以控制时间）；需在结果 JSONL 中记录 `dataset` 与 `max_samples`/`max_length`。

## 3. 下游任务（与 PPL 并列验收）

- 默认实现：**BoolQ** 验证集子集，用「续写 ` yes` / ` no` 的负对数似然」二选一作为预测（无生成解码，便于批跑）。
- **门控**：`(acc_lora - acc_mem+lora) / acc_lora <= 0.01`（当 `acc_lora > 0`）；若 `acc_lora` 接近 0，则补充绝对下降不超过 1 个百分点。

## 4. 压缩率（相对默认 MemRift 档 r0）

- `r` = MemRift 压缩产物目录在磁盘上的总字节数（含 `index.json` 与全部 `*.bin` 等） / **参考体积**。
- **参考体积**优先取：同模型 **BF16 全参权重**总字节数的估计值（参数元素数 × 2），与压缩对象（冻结底座）一致。
- **r0**：团队约定的**基准压缩档**，例如 `prepare_weight --level 18` 生成目录的 `r`。
- **目标**：在满足第 2、3 节门控前提下，`r <= 0.7 * r0`（相对 r0 再缩小约 30% 占用）。若业务用压缩倍数，则等价于 `(1/r) >= 1.3 * (1/r0)`。

## 5. 训练上下文 +20%（阶段一）

- 设当前训练配置中的基线长度为 `L0`（如 yaml 中 `seq_length`），目标 `L1 = ceil(1.2 * L0)` 或 `round(1.2 * L0)`（需在结果中写死所用取整方式）。
- **仅训练场景**：在 MemRift+LoRA 下用 Hydra 覆盖 `train.model.seq_length`（及数据管线一致配置），验证可跑通、OOM 情况，并报告 PPL/任务相对纯 LoRA 是否仍满足 1% 门控。
- 对应脚本：
  - [scripts/metrics/llama8b/train_context_gain.sh](/share/project/mengyc/code/memrift-flagscale/scripts/metrics/llama8b/train_context_gain.sh)
  - [scripts/metrics/aquila/train_context_gain.sh](/share/project/mengyc/code/memrift-flagscale/scripts/metrics/aquila/train_context_gain.sh)

## 6. 推理权重读盘时间

- baseline 读取原始模型目录中的权重文件，如 `*.safetensors`、`*.bin`。
- MemRift 分支读取压缩权重目录中的 `index.json` 与压缩 payload 文件。
- 该指标只测推理载入阶段的磁盘读取耗时，不实例化模型，不跑首个 forward。
- **门控**：`(read_time_baseline - read_time_memrift) / read_time_baseline >= 0.30`。

## 7. 自动化采集

- 驱动脚本：[scripts/memrift_metrics_matrix.py](/share/project/mengyc/code/memrift-flagscale/scripts/memrift_metrics_matrix.py)  
  输出 **JSONL**，每行一条记录，便于合并与画图。
- 显存/步时对比可继续用：[scripts/benchmark_lora_vs_memrift.py](/share/project/mengyc/code/memrift-flagscale/scripts/benchmark_lora_vs_memrift.py)（需将其中硬编码 `sys.path` 改为本仓库根目录）。
