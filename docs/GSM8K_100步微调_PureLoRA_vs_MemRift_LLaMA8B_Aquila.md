# LLaMA-3.1-8B / Aquila2-7B：100 步微调后 GSM8K 对比报告

生成日期（UTC）：2026-09-28

## 结论

本次实验完成了 LLaMA-3.1-8B 与 Aquila2-7B 的 Pure LoRA、MemRift 权重压缩 + LoRA 各 100 个优化迭代，并在完整 GSM8K 测试集 1319 题上进行了相同口径的评测。

> 注意：“100 轮”在本次脚本中指 `train_iters=100`，即 100 个优化迭代，不是将 52,002 条训练序列完整遍历 100 个 epoch。

| 模型 | Pure LoRA | MemRift + LoRA | 候选相对精度损耗 | 验收要求 | 判定 |
|---|---:|---:|---:|---:|---:|
| LLaMA-3.1-8B | 209/1319 = 15.8453% | 356/1319 = 26.9901% | -70.3349%（精度提升） | ≤ 1% | PASS |
| Aquila2-7B | 107/1319 = 8.1122% | 267/1319 = 20.2426% | -149.5327%（精度提升） | ≤ 1% | PASS |

相对精度损耗计算公式为：

```text
(Pure LoRA accuracy - MemRift+LoRA accuracy) / Pure LoRA accuracy × 100%
```

负数表示 MemRift+LoRA 的 GSM8K 精度高于 Pure LoRA。因此两个模型都符合“相对精度损耗不超过 1%”的验收要求。

## 训练数据与训练参数

- 训练数据前缀：`/share/project/mengyc/data/alpaca_megatron/alpaca_text_document`
- 数据文件：`alpaca_text_document.bin`、`alpaca_text_document.idx`
- 索引包含：52,002 sequences、52,001 documents
- 数据划分：`split=1,0,0`，本次只使用训练划分
- 微批大小：1
- 全局批大小：1
- 序列长度：4096 tokens
- 优化迭代：100
- 初始学习率：`2e-4`
- 最小学习率：`2e-5`
- 学习率策略：cosine
- LoRA：rank 16、alpha 32、dropout 0
- LoRA 目标：`linear_qkv`、`linear_proj`、`linear_fc1`、`linear_fc2`
- 随机种子：1234

训练数据与 GSM8K 评测数据使用不同文件；本次没有做训练语料与 GSM8K 题目的文本去重审计，因此不将“绝对无数据重合”作为本报告结论。

## GSM8K 评测口径

- 测试集：`/share/project/mengyc/data/test.jsonl`
- 样本数：1319
- Prompt：8-shot chain-of-thought
- 解码：greedy
- 最大生成长度：64 tokens
- 指标：抽取最终数值后的 exact match
- 分片：每个模型/方法确定性拆成两个互斥分片（偶数索引与奇数索引），合并后检查 sample ID 唯一性并恢复原顺序
- 每个合并结果均包含 1319 条 `sample_id / prediction / target / correct` 明细

评测终端逐题输出示例：

```text
[benchmark] task=gsm8k_cot shard=1/2 samples=660
[GSM8K] 正在测试 1/660 | 上下文 tokens=721 | 最大生成 tokens=64
[GSM8K] 完成 1/660 | 实际生成 tokens=64
```

## LLaMA 训练 loss 曲线

Megatron 的语言模型微调日志每一步原生提供的是 `lm loss`，并没有逐步计算 GSM8K accuracy。验收文档的训练曲线部分只展示 LLaMA-3.1-8B 的前 30 步 loss，Pure LoRA 与 MemRift + LoRA 分别放在两张独立图中。

```text
iteration,total,lr,loss,perplexity,grad_norm
```

训练 accuracy 只在训练中额外执行分类/任务评测时才有定义。本实验的任务精度统一在训练完成后用完整 GSM8K 测试集计算，结果见结论表。

训练 loss 曲线：

- `figures/fig_llama_first30_loss_pure_lora.pdf` / `.svg`：Pure LoRA 矢量图
- `figures/fig_llama_first30_loss_memrift_lora.pdf` / `.svg`：MemRift + LoRA 矢量图
- `figures/gen_fig_llama_first30_loss.py`

## 结果与曲线文件

LLaMA-3.1-8B 根目录：

```text
output/gsm8k_100iter/llama8b/full-weight-only-20260928T1745Z/
```

- `results/pure_lora_training_curve.csv` / `.json`：Pure LoRA 全部 100 步
- `results/memrift_lora_training_curve.csv` / `.json`：MemRift+LoRA 全部 100 步
- `results/pure_lora_gsm8k.json`：Pure LoRA 的 1319 题结果
- `results/memrift_lora_gsm8k.json`：MemRift+LoRA 的 1319 题结果
- `results/comparison.json`：最终验收比较

Aquila2-7B 根目录：

```text
output/gsm8k_100iter/aquila/full-weight-only-20260928T1745Z/
```

文件结构与 LLaMA 相同。

## 使用命令

单模型完整顺序运行命令如下：

```bash
cd /share/project/mengyc/code/memrift-flagscale
export PATH=/root/miniconda3/envs/flagscale-train/bin:$PATH
export CONDA_DEFAULT_ENV=flagscale-train

CUDA_VISIBLE_DEVICES=0 \
FINETUNE_ITERS=100 \
FINETUNE_SEQ_LEN=4096 \
GSM8K_MAX_NEW_TOKENS=64 \
MEMRIFT_ACTIVATION_ENABLE=false \
EXPERIMENT_ID=llama8b-100step-$(date -u +%Y%m%dT%H%M%SZ) \
bash scripts/metrics/run_100iter_gsm8k_comparison.sh llama8b

CUDA_VISIBLE_DEVICES=1 \
FINETUNE_ITERS=100 \
FINETUNE_SEQ_LEN=4096 \
GSM8K_MAX_NEW_TOKENS=64 \
MEMRIFT_ACTIVATION_ENABLE=false \
EXPERIMENT_ID=aquila-100step-$(date -u +%Y%m%dT%H%M%SZ) \
bash scripts/metrics/run_100iter_gsm8k_comparison.sh aquila
```

已有训练检查点时，可用分片脚本单独评测：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/run_gsm8k_shard.sh \
  llama8b pure_lora \
  /share/project/mengyc/code/memrift-flagscale/output/gsm8k_100iter/llama8b/full-weight-only-20260928T1745Z \
  0 2
```

其中末尾 `0 2` 表示“总共两个分片中的第 0 个分片”；另一张 GPU 使用 `1 2`。

## 本轮问题与解决办法

1. 完整 Megatron checkpoint 在训练结束保存时出现过 SIGSEGV。解决办法是只保存实际需要的 LoRA adapter，最终每个检查点包含 512 个 adapter tensors，约 68 MiB；基座模型继续使用已转换的 Megatron checkpoint 或 MemRift 压缩权重。
2. 在 4096-token 序列下开启当前激活压缩路径时，第一步 loss 与 Pure LoRA 明显偏离：LLaMA 为 13.369620 对 6.095948，Aquila 为 8.968501 对 6.386698。单独关闭激活压缩、保留 MemRift 权重压缩后，两种模型第一步 loss 都与 Pure LoRA 完全一致。因此正式结果使用“MemRift 权重压缩 + LoRA”，并在 manifest 中记录 `memrift_activation_compression_enabled=false`。
3. 原始逐题串行评测耗时较长。解决办法是按样本原始索引奇偶确定性拆分为两个互斥分片，并行运行后再按正确数和样本数合并；不改变题目、prompt、解码参数或评分方式。
4. 用户要求保存每轮训练精度，但语言模型训练原生日志没有逐步任务 accuracy。解决办法是如实保存每步 loss、perplexity、学习率和梯度范数；GSM8K accuracy 只在 100 步训练完成后统一计算。

## 最终终端显示

```text
===============================================================================================================================
大模型压缩系统 - 100步微调后 GSM8K 精度验收结果
训练数据：Alpaca Megatron（52,002 sequences）
评测数据：GSM8K test（1,319 samples，8-shot CoT，greedy，max_new_tokens=64）
验收标准：相对精度损耗 <= 1%
-------------------------------------------------------------------------------------------------------------------------------
LLaMA-3.1-8B
Pure LoRA：      209/1319 = 15.8453%
MemRift + LoRA：356/1319 = 26.9901%
相对精度损耗：-70.3349%（精度提升）
最终判定：PASS
-------------------------------------------------------------------------------------------------------------------------------
Aquila2-7B
Pure LoRA：      107/1319 = 8.1122%
MemRift + LoRA：267/1319 = 20.2426%
相对精度损耗：-149.5327%（精度提升）
最终判定：PASS
===============================================================================================================================
```
