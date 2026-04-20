#!/bin/bash
# ============================================
# TinyLlama 1.1B MemRift 训练一键脚本
# ============================================
# 1) 若未准备压缩权重，则从 HuggingFace 模型生成 MemRift 压缩权重
# 2) 使用 mock 数据启动 MemRift+LoRA 训练（无需准备 data_path）
#
# 用法（在仓库根目录 FlagScale 下执行）:
#   ./scripts/run_tinyllama_memrift_train.sh
#
# 可选环境变量:
#   TINYLLAMA_MODEL     - HuggingFace 模型名或本地路径，默认 TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   MEMRIFT_WEIGHT_DIR  - 压缩权重输出/已有目录，默认 ./memrift_weights/tinyllama_1b_level18
#   TRAIN_ITERS         - 训练迭代数，默认 50（冒烟）/ 可改为 1000+
#   SKIP_PREPARE        - 若已准备好压缩权重，设 1 可跳过 prepare_weight
#
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TINYLLAMA_MODEL="${TINYLLAMA_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$ROOT_DIR/memrift_weights/tinyllama_1b_level18}"
TRAIN_ITERS="${TRAIN_ITERS:-50}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

CONFIG_PATH="examples/memrift/conf"
CONFIG_NAME="train"

echo "============================================"
echo "TinyLlama 1.1B MemRift 训练"
echo "============================================"
echo "ROOT_DIR:            $ROOT_DIR"
echo "TINYLLAMA_MODEL:     $TINYLLAMA_MODEL"
echo "MEMRIFT_WEIGHT_DIR:  $MEMRIFT_WEIGHT_DIR"
echo "TRAIN_ITERS:         $TRAIN_ITERS"
echo "============================================"

# Step 1: 准备 MemRift 压缩权重（若无）
if [ "$SKIP_PREPARE" != "1" ]; then
  if [ ! -f "$MEMRIFT_WEIGHT_DIR/index.json" ]; then
    echo ">> 正在生成 MemRift 压缩权重..."
    python -m flagscale.compress.memrift.offline_comp.prepare_weight \
      --model "$TINYLLAMA_MODEL" \
      --outdir "$MEMRIFT_WEIGHT_DIR" \
      --level 18
    echo ">> 压缩权重已写入: $MEMRIFT_WEIGHT_DIR"
  else
    echo ">> 检测到已有压缩权重: $MEMRIFT_WEIGHT_DIR (跳过 prepare_weight)"
  fi
else
  echo ">> SKIP_PREPARE=1，跳过 prepare_weight"
fi

# Step 2: 使用 mock 数据配置跑训练（tinyllama_1b_lora_memrift_mock）
# tokenizer_model: Megatron HuggingFaceTokenizer 使用此参数
# lr_warmup_iters: 必须小于 train_iters
echo ">> 启动 MemRift + LoRA 训练 (mock 数据, ${TRAIN_ITERS} iters)..."

python run.py \
  --config-path="$CONFIG_PATH" \
  --config-name=train_mock \
  action=run \
  train.system.memrift_compressed_weight_dir="$MEMRIFT_WEIGHT_DIR" \
  train.model.tokenizer_path="$TINYLLAMA_MODEL" \
  train.model.tokenizer_model="$TINYLLAMA_MODEL" \
  train.trainer.train_iters="$TRAIN_ITERS" \
  train.trainer.lr_warmup_iters=10

echo ">> 训练结束"
