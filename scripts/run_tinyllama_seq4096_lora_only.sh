#!/bin/bash
# ============================================================
# TinyLlama 1.1B — 纯 LoRA（无 MemRift）
# seq=4096, bs=1，用于与 MemRift GPU-Only 对比
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TINYLLAMA_MODEL="${TINYLLAMA_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
TRAIN_ITERS="${TRAIN_ITERS:-30}"

echo "============================================================"
echo " TinyLlama 1.1B 纯 LoRA（无 MemRift）"
echo " seq=4096  bs=1  train_iters=$TRAIN_ITERS"
echo "============================================================"
echo " ROOT_DIR:           $ROOT_DIR"
echo " TINYLLAMA_MODEL:    $TINYLLAMA_MODEL"
echo "============================================================"

# ── Step 2: 训练 ─────────────────────────────────────────────
LOG_DIR="$ROOT_DIR/outputs/tinyllama_seq4096_bs1_lora_only_30iters"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo ">> 启动训练，日志写入: $LOG_FILE"
echo ">> 显存与迭代时间每步（log_interval=1）打印到日志"

# 环境变量优化
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/flagscale:$ROOT_DIR/flagscale/train:${PYTHONPATH}"

# 清理旧输出
rm -rf "$ROOT_DIR/outputs/tinyllama_seq4096_bs1_lora_only_30iters"

# 运行训练
python run.py \
  --config-path="examples/memrift/conf" \
  --config-name="train_tinyllama_seq4096_bs1_lora_only" \
  action=run \
  train.model.tokenizer_path="$TINYLLAMA_MODEL" \
  train.model.tokenizer_model="$TINYLLAMA_MODEL" \
  train.trainer.train_iters="$TRAIN_ITERS" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo " 训练结束，日志: $LOG_FILE"
echo " 从日志中提取迭代时间："
echo "   grep 'elapsed time per iteration' $LOG_FILE"
echo " 从日志中提取显存峰值："
echo "   grep 'max allocated' $LOG_FILE"
echo "============================================================"
