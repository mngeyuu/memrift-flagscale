#!/bin/bash
# ============================================================
# Llama-3.1-8B — 权重压缩 + 激活压缩 GPU-Only 优化版
# seq=4096, bs=1，完全走 GPU 路径
# ============================================================
# 用法（在 FlagScale 根目录执行）:
#   ssh 172.24.178.248 "conda run -n myc-flagscale \\\
#       bash /share/project/mengyc/flagScale/FlagScale/scripts/run_llama31_8b_seq4096_memrift_gpu_only_optimized.sh"
#
# 可选环境变量:
#   LLAMA31_MODEL       - HF 模型名或本地路径，默认 meta-llama/Llama-3.1-8B
#   MEMRIFT_WEIGHT_DIR - 压缩权重目录，默认 ./memrift_weights/llama31_8b_level18
#   TRAIN_ITERS        - 训练步数，默认 30
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LLAMA31_MODEL="${LLAMA31_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$ROOT_DIR/memrift_weights/llama31_8b_level18}"
TRAIN_ITERS="${TRAIN_ITERS:-30}"

echo "============================================================"
echo " Llama-3.1-8B MemRift GPU-Only 优化版"
echo " 权重+激活压缩，完全 GPU 路径"
echo " seq=4096  bs=1  train_iters=$TRAIN_ITERS"
echo "============================================================"
echo " ROOT_DIR:           $ROOT_DIR"
echo " LLAMA31_MODEL:      $LLAMA31_MODEL"
echo " MEMRIFT_WEIGHT_DIR: $MEMRIFT_WEIGHT_DIR"
echo "============================================================"

# 检查压缩权重
if [ ! -d "$MEMRIFT_WEIGHT_DIR" ]; then
  echo "错误：压缩权重不存在: $MEMRIFT_WEIGHT_DIR"
  exit 1
else
  echo ">> 已有压缩权重: $MEMRIFT_WEIGHT_DIR"
fi

# ── Step 2: 训练 ─────────────────────────────────────────────
LOG_DIR="$ROOT_DIR/outputs/memrift_llama31_8b_seq4096_bs1_gpu_only_optimized"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo ">> 启动训练，日志写入: $LOG_FILE"
echo ">> 显存与迭代时间每步（log_interval=1）打印到日志"

# 环境变量优化
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/flagscale:$ROOT_DIR/flagscale/train:${PYTHONPATH}"
# 深度异步，减少 CPU 同步
export MEMRIFT_DEEP_ASYNC=1
export MEMRIFT_WEIGHT_RELAX_SYNC=1
# 允许使用旧版 CPU zstd 权重格式
export MEMRIFT_ALLOW_CPU_WEIGHT_ZSTD=1

# 运行训练
python run.py \
  --config-path="examples/memrift/conf" \
  --config-name="train_llama31_8b_seq4096_bs1_memrift_gpu_only_optimized" \
  action=run \
  train.system.memrift_compressed_weight_dir="$MEMRIFT_WEIGHT_DIR" \
  train.model.tokenizer_path="$LLAMA31_MODEL" \
  train.model.tokenizer_model="$LLAMA31_MODEL" \
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
