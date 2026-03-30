#!/bin/bash
# ============================================================
# Llama-3.1-8B — 纯 LoRA 基线
# seq=2048, bs=1，用于与 MemRift GPU-Only 对比
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LLAMA31_MODEL="${LLAMA31_MODEL:-meta-llama/Llama-3.1-8B}"
TRAIN_ITERS="${TRAIN_ITERS:-30}"

echo "============================================================"
echo " Llama-3.1-8B 纯 LoRA 基线"
echo " seq=2048  bs=1  train_iters=$TRAIN_ITERS"
echo "============================================================"
echo " ROOT_DIR:      $ROOT_DIR"
echo " LLAMA31_MODEL: $LLAMA31_MODEL"
echo "============================================================"

LOG_DIR="$ROOT_DIR/outputs/llama31_8b_seq2048_bs1_lora_only"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/flagscale:$ROOT_DIR/flagscale/train:${PYTHONPATH:-}"

echo ">> 启动训练，日志写入: $LOG_FILE"

python run.py \
  --config-path="examples/memrift/conf" \
  --config-name="train_llama31_8b_seq2048_bs1_lora_only" \
  action=run \
  train.model.tokenizer_path="$LLAMA31_MODEL" \
  train.model.tokenizer_model="$LLAMA31_MODEL" \
  train.trainer.train_iters="$TRAIN_ITERS" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "训练结束，日志: $LOG_FILE"
echo "迭代时间: grep 'elapsed time per iteration' \"$LOG_FILE\""
echo "显存峰值: grep 'max allocated' \"$LOG_FILE\""
echo "============================================================"
