#!/usr/bin/env bash
# SIGABRT / 无 Python traceback 的 CUDA 崩溃：用同步启动 + C++ 栈缩小到具体 kernel/API。
#
# 在 GPU 机（如 172.24.178.248）:
#   source .../conda.sh && conda activate myc-flagscale
#   cd /share/project/mengyc/flagScale/FlagScale
#   bash scripts/run_llama31_memrift_cuda_debug.sh
#
# 环境变量（可选）:
#   TRAIN_ITERS       默认 2
#   TORCH_USE_CUDA_DSA  默认 1；PyTorch 过旧可设 0
#   MEMRIFT_*         与 run_llama31_8b_seq4096_memrift_weight_only.sh 相同
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 同步执行每个 kernel，错误会在「发起 cudaLaunch 的线程」上立即报出，便于对照 Python/C++ 栈
export CUDA_LAUNCH_BLOCKING=1
# PyTorch C++ 扩展栈（含部分 CUDA 调用路径）
export TORCH_SHOW_CPP_STACKTRACES=1
# 设备端断言更可读（需较新 PyTorch；不支持时无效果）
export TORCH_USE_CUDA_DSA="${TORCH_USE_CUDA_DSA:-1}"

export TRAIN_ITERS="${TRAIN_ITERS:-2}"

echo "============================================================"
echo " MemRift CUDA 调试模式"
echo " CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 TORCH_USE_CUDA_DSA=$TORCH_USE_CUDA_DSA"
echo " TRAIN_ITERS=$TRAIN_ITERS"
echo "============================================================"

exec bash "$ROOT_DIR/scripts/run_llama31_8b_seq4096_memrift_weight_only.sh"
