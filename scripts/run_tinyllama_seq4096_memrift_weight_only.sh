#!/bin/bash
# ============================================================
# TinyLlama 1.1B — MemRift 仅权重压缩 + 权重异步解压（prefetch）
# seq=4096, bs=1，记录显存分解（memrift_profile_memory）与每步迭代时间
# ============================================================
# 用法（在 FlagScale 根目录执行）:
#   ./scripts/run_tinyllama_seq4096_memrift_weight_only.sh
#
# 依赖: 已按 FlagScale README 安装 Megatron-LM-FL / megatron-core（可 import megatron.core）
#
# 可选环境变量:
#   MEMRIFT_WEIGHT_DIR - 压缩权重目录，默认 ./memrift_weights/tinyllama_1b_level18
#   TRAIN_ITERS        - 训练步数，默认 20
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$ROOT_DIR/memrift_weights/tinyllama_1b_level18}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
TINYLLAMA_MODEL="${TINYLLAMA_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"

echo "============================================================"
echo " TinyLlama 1.1B MemRift（仅权重压缩异步，无激活压缩）"
echo " seq=4096  bs=1  train_iters=$TRAIN_ITERS"
echo "============================================================"

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/flagscale:$ROOT_DIR/flagscale/train:${PYTHONPATH}"

LOG_DIR="$ROOT_DIR/outputs/memrift_tinyllama_seq4096_bs1_weight_only"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

python run.py \
  --config-path="examples/memrift/conf" \
  --config-name="train_tinyllama_seq4096_bs1_memrift_weight_only" \
  action=run \
  train.system.memrift_compressed_weight_dir="$MEMRIFT_WEIGHT_DIR" \
  train.model.tokenizer_path="$TINYLLAMA_MODEL" \
  train.model.tokenizer_model="$TINYLLAMA_MODEL" \
  train.trainer.train_iters="$TRAIN_ITERS" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo " 日志: $LOG_FILE"
echo " 迭代时间: grep -E 'iteration|elapsed time per iteration' \"$LOG_FILE\""
echo " MemRift 显存峰值报告: grep -E 'MemProfiler|整轮 peak|max_memory' \"$LOG_FILE\""
echo "============================================================"
