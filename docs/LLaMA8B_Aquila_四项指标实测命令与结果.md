# LLaMA-3.1-8B 与 Aquila2-7B 四项指标实测命令与结果

本文档只覆盖本次验收指定的两个模型：

- 业界主流模型：LLaMA-3.1-8B（`llama8b`）；
- 智源自研模型：Aquila2-7B（`aquila`）。

测试日期为 2026-09-23，测试硬件为 NVIDIA A800-SXM4-80GB。所有指标均已实际运行，并由脚本按照验收门槛自动计算、自动输出 `PASS`/`FAIL`，当前两个模型均为 `4/4 PASS`。

## 一、首次运行前准备

进入项目并安装一次兼容依赖。依赖安装在项目的 `output/runtime_deps` 下，不修改共享 conda 环境；后续指标脚本会自动加载该目录。

```bash
cd /share/project/mengyc/code/memrift-flagscale

source /root/miniconda3/etc/profile.d/conda.sh
conda activate flagscale-train

python -m pip install --upgrade --no-deps \
  --target "$PWD/output/runtime_deps/hf_compat" \
  huggingface-hub==0.36.0
```

当前两个模型的 Megatron TP=1 检查点已准备好。如重新部署或检查点被删除，只需分别执行一次：

```bash
CONDA_ENV=flagscale-train \
  bash scripts/metrics/llama8b/convert_hf_to_mcore_tp1.sh

CONDA_ENV=flagscale-train \
  bash scripts/metrics/aquila/convert_hf_to_mcore_tp1.sh
```

## 二、逐项启动命令

以下命令可以一项一项、串行执行。每个脚本结束前都会在终端最后一屏打印完整验收表、计算公式、阈值和最终判定。

### 2.1 LLaMA-3.1-8B

模型压缩比例：

```bash
CONDA_ENV=flagscale-train CUDA_VISIBLE_DEVICES=0 \
  bash scripts/metrics/llama8b/compression_ratio.sh
```

Loss 损失：

```bash
CONDA_ENV=flagscale-train CUDA_VISIBLE_DEVICES=0 \
  bash scripts/metrics/llama8b/accuracy_loss.sh
```

训练最大上下文长度：

```bash
CONDA_ENV=flagscale-train CUDA_VISIBLE_DEVICES=0 \
  bash scripts/metrics/llama8b/train_context_gain.sh
```

推理模型载入时间：

```bash
CONDA_ENV=flagscale-train CUDA_VISIBLE_DEVICES=0 \
  bash scripts/metrics/llama8b/load_time_reduction.sh
```

### 2.2 Aquila2-7B

模型压缩比例：

```bash
CONDA_ENV=flagscale-train CUDA_VISIBLE_DEVICES=0 \
  bash scripts/metrics/aquila/compression_ratio.sh
```

Loss 损失：

```bash
CONDA_ENV=flagscale-train CUDA_VISIBLE_DEVICES=0 \
  bash scripts/metrics/aquila/accuracy_loss.sh
```

训练最大上下文长度：

```bash
CONDA_ENV=flagscale-train CUDA_VISIBLE_DEVICES=0 \
  bash scripts/metrics/aquila/train_context_gain.sh
```

推理模型载入时间：

```bash
CONDA_ENV=flagscale-train CUDA_VISIBLE_DEVICES=0 \
  bash scripts/metrics/aquila/load_time_reduction.sh
```

## 三、本次实测结果

| 模型 | 性能指标 | 基线实测值 | MemRift 实测值 | 比例结果 | 验收门槛 | 判定 |
|---|---|---:|---:|---:|---:|---|
| LLaMA-3.1-8B | 模型压缩比例 | BF16 14.9575 GiB | 9.9499 GiB | 节省 33.4793% | 节省率 ≥ 30% | PASS |
| LLaMA-3.1-8B | Loss 损失 | L0=5.375886 | L1=5.377508 | ΔLoss=+0.0302% | ΔLoss ≤ 1% | PASS |
| LLaMA-3.1-8B | 训练最大上下文 | L0=12,160 tokens | L1=15,872 tokens | 增长 30.5263% | L1/L0 ≥ 1.2 | PASS |
| LLaMA-3.1-8B | 推理权重载入时间 | T0=26.7551s | T1=18.1012s | 降低 32.3446% | T1/T0 ≤ 0.7 | PASS |
| Aquila2-7B | 模型压缩比例 | BF16 14.2598 GiB | 9.4671 GiB | 节省 33.6101% | 节省率 ≥ 30% | PASS |
| Aquila2-7B | Loss 损失 | L0=5.659653 | L1=5.652042 | ΔLoss=-0.1345% | ΔLoss ≤ 1% | PASS |
| Aquila2-7B | 训练最大上下文 | L0=12,544 tokens | L1=16,589 tokens | 增长 32.2465% | L1/L0 ≥ 1.2 | PASS |
| Aquila2-7B | 推理权重载入时间 | T0=50.7803s | T1=17.3987s | 降低 65.7373% | T1/T0 ≤ 0.7 | PASS |

两个模型的总体判定：

```text
LLaMA-3.1-8B：总体最终判定：符合指标 (PASS)（4/4 项量化指标通过）
Aquila2-7B：总体最终判定：符合指标 (PASS)（4/4 项量化指标通过）
```

## 四、指标口径

### 4.1 模型压缩比例

```text
存储节省率 = (BF16 参数参考大小 - MemRift 压缩目录大小) / BF16 参数参考大小 × 100%
验收要求：存储节省率 >= 30%
```

### 4.2 Loss 损失

```text
相对 Loss 损耗 = (MemRift 最终 lm loss - Pure LoRA 最终 lm loss) / Pure LoRA 最终 lm loss × 100%
验收要求：相对 Loss 损耗 <= 1%
```

该项比较 Pure LoRA 与 MemRift 权重压缩+LoRA，使用相同模型、数据、batch size、训练步数和其他关键配置。此项关闭激活压缩，避免将上下文显存优化项混入权重压缩精度损失口径。

### 4.3 训练最大上下文长度

```text
上下文长度收益 = (MemRift 最大可运行长度 - Pure LoRA 最大可运行长度) / Pure LoRA 最大可运行长度 × 100%
验收要求：MemRift 最大长度 / Pure LoRA 最大长度 >= 1.2
```

脚本会自动进行粗粒度和细粒度边界搜索。搜索日志中出现 CUDA OOM 表示脚本正在探测不可运行上界；脚本会回退到最后一个完整跑完两次训练迭代并产生有效 `lm loss` 的长度，最终以最后一屏的判定为准。

本脚本当前测量训练侧最大上下文；没有声称覆盖推理侧最大上下文。

### 4.4 推理模型载入时间

```text
载入时间下降比例 = (原始权重磁盘读取时间 - 压缩权重磁盘读取时间) / 原始权重磁盘读取时间 × 100%
验收要求：压缩权重读取时间 / 原始权重读取时间 <= 0.7
```

当前口径是相同读取实现下的权重文件磁盘读取时间，不包含模型实例化和首轮前向。录制正式验收视频时，应保持存储设备和 OS 缓存条件一致。

## 五、结果文件

单项验收证据保存在：

```text
output/metrics/llama8b/<指标名>/result.json
output/metrics/aquila/<指标名>/result.json
```

四项汇总证据保存在：

```text
output/metrics/llama8b/summary.json
output/metrics/aquila/summary.json
```

其中 `<指标名>` 分别为：

```text
compression_ratio
accuracy_loss
train_context_gain
load_time_reduction
```
