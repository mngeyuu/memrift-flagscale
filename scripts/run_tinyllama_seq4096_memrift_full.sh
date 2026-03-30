#!/bin/bash
# ============================================================
# TinyLlama 1.1B — 权重压缩 + 激活压缩全开
# seq=4096, bs=1，记录显存与迭代时间
# ============================================================
# 用法（在 FlagScale 根目录执行）:
#   ssh 172.24.178.248 "conda run -n myc-flagscale \
#       bash /share/project/mengyc/flagScale/FlagScale/scripts/run_tinyllama_seq4096_memrift_full.sh"
#
# 可选环境变量:
#   TINYLLAMA_MODEL    - HF 模型名或本地路径，默认 TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   MEMRIFT_WEIGHT_DIR - 压缩权重目录，默认 ./memrift_weights/tinyllama_1b_level18
#   TRAIN_ITERS        - 训练步数，默认 20
#   SKIP_PREPARE       - 设 1 可跳过 prepare_weight（已有压缩权重时使用）
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TINYLLAMA_MODEL="${TINYLLAMA_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$ROOT_DIR/memrift_weights/tinyllama_1b_level18}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

echo "============================================================"
echo " TinyLlama 1.1B MemRift Full（权重+激活压缩）"
echo " seq=4096  bs=1  train_iters=$TRAIN_ITERS"
echo "============================================================"
echo " ROOT_DIR:           $ROOT_DIR"
echo " TINYLLAMA_MODEL:    $TINYLLAMA_MODEL"
echo " MEMRIFT_WEIGHT_DIR: $MEMRIFT_WEIGHT_DIR"
echo "============================================================"

# ── Step 1: 准备压缩权重 ──────────────────────────────────────
if [ "$SKIP_PREPARE" != "1" ]; then
  if [ ! -f "$MEMRIFT_WEIGHT_DIR/index.json" ]; then
    echo ">> 生成 MemRift 压缩权重..."
    python -m flagscale.compress.memrift.offline_comp.prepare_weight \
      --model "$TINYLLAMA_MODEL" \
      --outdir "$MEMRIFT_WEIGHT_DIR" \
      --level 18
    echo ">> 压缩权重写入: $MEMRIFT_WEIGHT_DIR"
  else
    echo ">> 已有压缩权重: $MEMRIFT_WEIGHT_DIR（跳过 prepare_weight）"
  fi
else
  echo ">> SKIP_PREPARE=1，跳过 prepare_weight"
fi

# ── Step 2: 训练 ─────────────────────────────────────────────
# 环境变量说明:
#   MEMRIFT_ACT_TIME=1  - 打印每层激活压缩/解压耗时
#   MEMRIFT_TRACE=1     - 打印权重 prefetch/materialize trace（详细）
LOG_FILE="$ROOT_DIR/outputs/memrift_tinyllama_seq4096_bs1/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

echo ">> 启动训练，日志写入: $LOG_FILE"
echo ">> 显存与迭代时间每步（log_interval=1）打印到日志"

# 修复1：禁用 flagcx 后端自动加载（避免 undefined symbol: cuDeviceGet）
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# 修复2：补全 PYTHONPATH，nohup 脚本内会再追加 $ROOT:$ROOT/flagscale/train
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/flagscale:$ROOT_DIR/flagscale/train:${PYTHONPATH}"

MEMRIFT_ACT_TIME=1 \
python run.py \
  --config-path="examples/memrift/conf" \
  --config-name="train_tinyllama_seq4096_full" \
  action=run \
  train.system.memrift_compressed_weight_dir="$MEMRIFT_WEIGHT_DIR" \
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
