# MemRift 四模型评测指标定义（执行用）

本文档与计划「MemRift 四模型指标评测」对齐，作为验收口径与跑数说明。

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
- 辅助脚本：[scripts/run_train_context_plus20.sh](/share/project/mengyc/code/memrift-flagscale/scripts/run_train_context_plus20.sh)。

## 6. 推理上下文 +20% 与载入时间 ?30%（阶段二）

- **不列入阶段一必做**；后续在 METRICS 中补跑。
- **载入时间**：从「开始加载权重」到「完成首次 forward」的 wall time；基线 = 同模型同 LoRA、非 MemRift；对比 = MemRift 压缩底座 + LoRA 等价路径。

## 7. 自动化采集

- 驱动脚本：[scripts/memrift_metrics_matrix.py](/share/project/mengyc/code/memrift-flagscale/scripts/memrift_metrics_matrix.py)  
  输出 **JSONL**，每行一条记录，便于合并与画图。
- 显存/步时对比可继续用：[scripts/benchmark_lora_vs_memrift.py](/share/project/mengyc/code/memrift-flagscale/scripts/benchmark_lora_vs_memrift.py)（需将其中硬编码 `sys.path` 改为本仓库根目录）。

## 8. 智源自研模型（占位）

- 配置占位：[examples/memrift/conf/train/zhiyuan_placeholder.yaml](/share/project/mengyc/code/memrift-flagscale/examples/memrift/conf/train/zhiyuan_placeholder.yaml)  
  填入 HF id 或本地路径后，与四模型共用同一套 `memrift_metrics_matrix.py` 与 `run.py` 流程。
