#!/bin/bash
# ============================================================
# Llama-3.1-8B — 纯 LoRA（无 MemRift）
# seq=4096, bs=1，用于与 MemRift GPU-Only 对比
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LLAMA31_MODEL="${LLAMA31_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
TRAIN_ITERS="${TRAIN_ITERS:-30}"
MICRO_BS="${MICRO_BS:-1}"

echo "============================================================"
echo " Llama-3.1-8B 纯 LoRA（无 MemRift）"
echo " seq=4096  micro_bs=$MICRO_BS  train_iters=$TRAIN_ITERS"
echo "============================================================"
echo " ROOT_DIR:           $ROOT_DIR"
echo " LLAMA31_MODEL:      $LLAMA31_MODEL"
echo "============================================================"

# ── Step 2: 训练 ─────────────────────────────────────────────
LOG_DIR="$ROOT_DIR/outputs/llama31_8b_seq4096_bs${MICRO_BS}_lora_only"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo ">> 启动训练，日志写入: $LOG_FILE"
echo ">> 显存与迭代时间每步（log_interval=1）打印到日志"

# 环境变量优化
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/flagscale:$ROOT_DIR/flagscale/train:${PYTHONPATH}"

OUTPUT_NAME="llama31_8b_seq4096_bs${MICRO_BS}_lora_only"

# 运行训练
python run.py \
  --config-path="examples/memrift/conf" \
  --config-name="train_llama31_8b_seq4096_bs1_lora_only" \
  action=run \
  experiment.exp_name="$OUTPUT_NAME" \
  train.model.tokenizer_path="$LLAMA31_MODEL" \
  train.model.tokenizer_model="$LLAMA31_MODEL" \
  train.trainer.train_iters="$TRAIN_ITERS" \
  train.data.micro_batch_size="$MICRO_BS" \
  train.data.global_batch_size="$MICRO_BS" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo " 训练结束，日志: $LOG_FILE"
echo " 从日志中提取迭代时间："
echo "   grep 'elapsed time per iteration' $LOG_FILE"
echo " 从日志中提取显存峰值："
echo "   grep 'max allocated' $LOG_FILE"
echo "============================================================"
