#!/bin/bash
# ============================================
# 使用 Nsight Systems 对 MemRift 训练做 GPU/CUDA profile
# ============================================
# 用法（在 FlagScale 仓库根目录执行）:
#   ./scripts/run_nsight_profile.sh
#   ./scripts/run_nsight_profile.sh --iters 10 --output my_profile
#
# Llama-3.1-8B + MemRift 仅权重 + CPU/GPU/显存：见 run_nsight_llama31_8b_memrift_weight_only.sh
#
# 可选环境变量:
#   MEMRIFT_WEIGHT_DIR - 压缩权重目录，默认 ./memrift_weights/tinyllama_1b_level18
#   NSIGHT_ITERS       - profile 训练步数，默认 1（够看时间线，省时）
#   NSIGHT_OUTPUT      - 输出 .nsys-rep 文件名（不含扩展名），默认 memrift_llama11b_nsight
#
# 依赖: 已安装 Nsight Systems CLI (nsys)，且已准备压缩权重（或先跑 run_tinyllama_memrift_train.sh 生成）
# 生成后: 用 Nsight Systems 图形界面打开 <NSIGHT_OUTPUT>.nsys-rep 查看时间线。
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$ROOT_DIR/memrift_weights/tinyllama_1b_level18}"
NSIGHT_ITERS="${NSIGHT_ITERS:-1}"
NSIGHT_OUTPUT="${NSIGHT_OUTPUT:-memrift_llama11b_nsight}"

if ! command -v nsys &>/dev/null; then
  echo "错误: 未找到 nsys (Nsight Systems)。请安装 CUDA Toolkit / Nsight Systems 并确保 nsys 在 PATH 中。"
  exit 1
fi

echo "============================================"
echo "Nsight Systems Profile — MemRift TinyLlama 1.1B"
echo "============================================"
echo "ROOT_DIR:           $ROOT_DIR"
echo "MEMRIFT_WEIGHT_DIR: $MEMRIFT_WEIGHT_DIR"
echo "NSIGHT_ITERS:       $NSIGHT_ITERS"
echo "NSIGHT_OUTPUT:      $NSIGHT_OUTPUT"
echo "============================================"

python scripts/profile_memrift_nsight.py \
  --iters "$NSIGHT_ITERS" \
  --output "$NSIGHT_OUTPUT" \
  --compressed-weight-dir "$MEMRIFT_WEIGHT_DIR" \
  "$@"

echo ""
echo ">> 完成。用 Nsight Systems 打开: $NSIGHT_OUTPUT.nsys-rep"
