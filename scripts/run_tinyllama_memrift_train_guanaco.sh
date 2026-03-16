#!/bin/bash
# ============================================
# TinyLlama 1.1B MemRift+LoRA 真实数据训练（openassistant-guanaco）
# ============================================
# 用法（在仓库根目录 FlagScale 下执行）:
#   ./scripts/run_tinyllama_memrift_train_guanaco.sh
#
# 可选环境变量:
#   TINYLLAMA_MODEL     - HuggingFace 模型名，默认 TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   MEMRIFT_WEIGHT_DIR  - 压缩权重目录，默认 ./memrift_weights/tinyllama_1b_level18
#   TRAIN_ITERS         - 训练迭代数，默认 200
#   SKIP_PREPARE        - 已准备好压缩权重时设为 1 跳过 prepare_weight
#
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TINYLLAMA_MODEL="${TINYLLAMA_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$ROOT_DIR/memrift_weights/tinyllama_1b_level18}"
TRAIN_ITERS="${TRAIN_ITERS:-200}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

CONFIG_PATH="examples/memrift/conf"

echo "============================================"
echo "TinyLlama 1.1B MemRift+LoRA 真实数据训练 (guanaco)"
echo "============================================"
echo "ROOT_DIR:            $ROOT_DIR"
echo "TINYLLAMA_MODEL:     $TINYLLAMA_MODEL"
echo "MEMRIFT_WEIGHT_DIR:  $MEMRIFT_WEIGHT_DIR"
echo "TRAIN_ITERS:         $TRAIN_ITERS"
echo "data_path:           timdettmers/openassistant-guanaco"
echo "============================================"

if [ "$SKIP_PREPARE" != "1" ] && [ ! -f "$MEMRIFT_WEIGHT_DIR/index.json" ]; then
  echo ">> 正在生成 MemRift 压缩权重..."
  python -m flagscale.compress.memrift.offline_comp.prepare_weight \
    --model "$TINYLLAMA_MODEL" \
    --outdir "$MEMRIFT_WEIGHT_DIR" \
    --level 18
  echo ">> 压缩权重已写入: $MEMRIFT_WEIGHT_DIR"
elif [ "$SKIP_PREPARE" = "1" ]; then
  echo ">> SKIP_PREPARE=1，跳过 prepare_weight"
else
  echo ">> 使用已有压缩权重: $MEMRIFT_WEIGHT_DIR"
fi

echo ">> 启动 MemRift+LoRA 训练（真实数据 openassistant-guanaco，${TRAIN_ITERS} iters）..."

python run.py \
  --config-path="$CONFIG_PATH" \
  --config-name=train_guanaco \
  action=run \
  train.system.memrift_compressed_weight_dir="$MEMRIFT_WEIGHT_DIR" \
  train.model.tokenizer_path="$TINYLLAMA_MODEL" \
  train.model.tokenizer_model="$TINYLLAMA_MODEL" \
  train.trainer.train_iters="$TRAIN_ITERS" \
  train.trainer.lr_warmup_iters=50

echo ">> 训练结束"
