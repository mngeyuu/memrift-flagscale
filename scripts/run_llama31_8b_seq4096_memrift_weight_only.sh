#!/bin/bash
# ============================================================
# Llama-3.1-8B — 仅权重压缩优化版
# seq=4096, bs=1，用于与全开配置对比
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LLAMA31_MODEL="${LLAMA31_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$ROOT_DIR/memrift_weights/llama31_8b_nvcomp_ans}"
TRAIN_ITERS="${TRAIN_ITERS:-30}"
PREFETCH_LAYERS="${PREFETCH_LAYERS:-1}"
MICRO_BS="${MICRO_BS:-1}"
PROFILE_OUT="${PROFILE_OUT:-}"

echo "============================================================"
echo " Llama-3.1-8B MemRift 仅权重压缩优化版"
echo " seq=4096  micro_bs=$MICRO_BS  train_iters=$TRAIN_ITERS"
echo "============================================================"
echo " ROOT_DIR:           $ROOT_DIR"
echo " LLAMA31_MODEL:      $LLAMA31_MODEL"
echo "============================================================"

OUTPUT_NAME="llama31_8b_seq4096_bs${MICRO_BS}_memrift_weight_only"

# 清理旧输出
rm -rf "$ROOT_DIR/outputs/$OUTPUT_NAME"

# ── Step 2: 训练 ─────────────────────────────────────────────
LOG_DIR="$ROOT_DIR/outputs/$OUTPUT_NAME"
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
# 真正启用后台线程异步预取（主线程不阻塞 codec.decode）
export MEMRIFT_WEIGHT_PREFETCH_MODE=async_bg
# BG decode 线程数（默认=1）：GPU ANS decode 受显存带宽限制，多线程并行实测反而更慢
# (benchmark: 4线程 speedup=0.22-0.49x)，保持1个线程为最优。
export MEMRIFT_BG_DECODE_THREADS=1
# 禁用 backward 中的 empty_cache（默认每 5 步调用一次，造成 GPU 停顿）
export MEMRIFT_BWD_EMPTY_STEP=100000

# 分层时间 profiling（若设置了 PROFILE_OUT 则启用，每 30s 写一次中间结果）
if [ -n "$PROFILE_OUT" ]; then
    mkdir -p "$(dirname "$PROFILE_OUT")"
    export MEMRIFT_LAYER_TIME_PROFILE_PATH="$PROFILE_OUT"
    export MEMRIFT_LAYER_TIME_PROFILE_LIVE_SEC=30
    echo ">> Profiling 已启用，输出: $PROFILE_OUT"
fi

# 运行训练
python run.py \
  --config-path="examples/memrift/conf" \
  --config-name="train_llama31_8b_seq4096_bs1_memrift_weight_only" \
  action=run \
  experiment.exp_name="$OUTPUT_NAME" \
  train.system.memrift_compressed_weight_dir="$MEMRIFT_WEIGHT_DIR" \
  train.model.tokenizer_path="$LLAMA31_MODEL" \
  train.model.tokenizer_model="$LLAMA31_MODEL" \
  train.trainer.train_iters="$TRAIN_ITERS" \
  train.system.memrift_prefetch_layers="$PREFETCH_LAYERS" \
  train.data.micro_batch_size="$MICRO_BS" \
  train.data.global_batch_size="$MICRO_BS" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo " 训练结束，日志: $LOG_FILE"
echo " 输出目录: $ROOT_DIR/outputs/$OUTPUT_NAME"
echo " 从日志中提取迭代时间："
echo "   grep 'elapsed time per iteration' $LOG_FILE"
echo " 从日志中提取显存峰值："
echo "   grep 'max allocated' $LOG_FILE"
echo "============================================================"
