#!/bin/bash
# ============================================================
# Llama-3.1-8B — MemRift GPU-Only 优化版
# seq=2048, bs=1，目标接近纯 LoRA 训练时长
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LLAMA31_MODEL="${LLAMA31_MODEL:-meta-llama/Llama-3.1-8B}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$ROOT_DIR/memrift_weights/llama31_8b_nvcomp_ans}"
TRAIN_ITERS="${TRAIN_ITERS:-30}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

echo "============================================================"
echo " Llama-3.1-8B MemRift GPU-Only 优化版"
echo " seq=2048  bs=1  train_iters=$TRAIN_ITERS"
echo "============================================================"
echo " ROOT_DIR:           $ROOT_DIR"
echo " LLAMA31_MODEL:      $LLAMA31_MODEL"
echo " MEMRIFT_WEIGHT_DIR: $MEMRIFT_WEIGHT_DIR"
echo "============================================================"

if [ ! -f "$MEMRIFT_WEIGHT_DIR/index.json" ]; then
  if [ "$SKIP_PREPARE" = "1" ]; then
    echo "错误：压缩权重不存在: $MEMRIFT_WEIGHT_DIR/index.json"
    exit 1
  fi

  echo ">> 未检测到 NVAN 压缩权重，开始离线压缩..."
  mkdir -p "$MEMRIFT_WEIGHT_DIR"
  python -m flagscale.compress.memrift.offline_comp.prepare_weight \
    --model "$LLAMA31_MODEL" \
    --outdir "$MEMRIFT_WEIGHT_DIR" \
    --compression nvcomp_ans \
    --level 18
fi

LOG_DIR="$ROOT_DIR/outputs/llama31_8b_seq2048_bs1_memrift_gpu_only_optimized"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/flagscale:$ROOT_DIR/flagscale/train:${PYTHONPATH:-}"
export MEMRIFT_DEEP_ASYNC=1
export MEMRIFT_WEIGHT_RELAX_SYNC=1
export MEMRIFT_BWD_EMPTY_STEP=100000
export MEMRIFT_ACT_EMPTY_INTERVAL=0
export MEMRIFT_ACT_LOOKAHEAD=0

echo ">> 启动训练，日志写入: $LOG_FILE"

python run.py \
  --config-path="examples/memrift/conf" \
  --config-name="train_llama31_8b_seq2048_bs1_memrift_gpu_only_optimized" \
  action=run \
  train.system.memrift_compressed_weight_dir="$MEMRIFT_WEIGHT_DIR" \
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
