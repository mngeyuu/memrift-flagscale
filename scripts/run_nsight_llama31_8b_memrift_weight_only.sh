#!/usr/bin/env bash
# =============================================================================
# Nsight Systems：FlagScale + MemRift（仅权重压缩）+ Llama-3.1-8B + mock 数据
# =============================================================================
# 采集：
#   - GPU：CUDA API / kernel / NVTX（由 profile_memrift_nsight.py 的 nsys 参数）
#   - CPU：nsys --sample=cpu
#   - 显存：nsys --cuda-memory-usage=true（时间线上看峰值；见文末 nsys stats）
# 可选 Python↔CUDA 栈关联：--python-backtrace=cuda（默认开启，可用 NSIGHT_PYTHON_BT=0 关闭）
#
# 用法（在 FlagScale 仓库根目录）:
#   ./scripts/run_nsight_llama31_8b_memrift_weight_only.sh
#   NSIGHT_ITERS=2 MEMRIFT_WEIGHT_DIR=/path/to/comp ./scripts/run_nsight_llama31_8b_memrift_weight_only.sh
#
# 环境变量:
#   MEMRIFT_WEIGHT_DIR     压缩权重目录，默认 $ROOT/memrift_weights/llama31_8b_nvcomp_ans
#   LLAMA31_MODEL          tokenizer / 模型 ID 或本地路径（Hydra 覆盖）
#   NSIGHT_ITERS           train_iters，默认 3
#   NSIGHT_OUT             输出前缀（无扩展名），默认 outputs/nsight/flagscale_memrift_wonly_llama8b_<时间戳>
#   NSIGHT_DIR             NSIGHT_OUT 所在目录，默认 outputs/nsight
#   CUDA_MEMORY_USAGE      设为 0 可关闭显存采集（略省开销）
#   NSIGHT_PYTHON_BT       设为 0 关闭 --python-backtrace=cuda
#   MASTER_PORT            torchrun 端口，避免多任务冲突
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$ROOT_DIR/memrift_weights/llama31_8b_nvcomp_ans}"
# 默认用集群本地路径（无网环境）；有网可改为 meta-llama/Llama-3.1-8B
LLAMA31_MODEL="${LLAMA31_MODEL:-/share/project/mengyc/models/modelscope/LLM-Research/Meta-Llama-3-8B-Instruct}"
NSIGHT_ITERS="${NSIGHT_ITERS:-3}"
NSIGHT_DIR="${NSIGHT_DIR:-$ROOT_DIR/outputs/nsight}"
STAMP="$(date +%Y%m%d_%H%M%S)"
NSIGHT_OUT="${NSIGHT_OUT:-$NSIGHT_DIR/flagscale_memrift_wonly_llama8b_${STAMP}}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 3000))}"
export MASTER_PORT

mkdir -p "$(dirname "$NSIGHT_OUT")"

if ! command -v nsys &>/dev/null; then
  echo "错误: 未找到 nsys。请安装 Nsight Systems 并加入 PATH。"
  exit 1
fi

echo "============================================================"
echo " Nsight | FlagScale MemRift 仅权重 | Llama-3.1-8B"
echo " MEMRIFT_WEIGHT_DIR=$MEMRIFT_WEIGHT_DIR"
echo " LLAMA31_MODEL=$LLAMA31_MODEL"
echo " NSIGHT_ITERS=$NSIGHT_ITERS"
echo " NSIGHT_OUT=$NSIGHT_OUT"
echo " MASTER_PORT=$MASTER_PORT"
echo " CUDA_MEMORY_USAGE=${CUDA_MEMORY_USAGE:-1} (显存曲线)"
echo "============================================================"

export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/flagscale:$ROOT_DIR/flagscale/train:${PYTHONPATH:-}"
# 与 run_llama31_8b_seq4096_memrift_weight_only.sh 对齐的 MemRift 运行时开关
export MEMRIFT_DEEP_ASYNC="${MEMRIFT_DEEP_ASYNC:-1}"
export MEMRIFT_WEIGHT_RELAX_SYNC="${MEMRIFT_WEIGHT_RELAX_SYNC:-1}"
export MEMRIFT_ALLOW_CPU_WEIGHT_ZSTD="${MEMRIFT_ALLOW_CPU_WEIGHT_ZSTD:-1}"
export MEMRIFT_WEIGHT_PREFETCH_MODE="${MEMRIFT_WEIGHT_PREFETCH_MODE:-async_bg}"
export MEMRIFT_BG_DECODE_THREADS="${MEMRIFT_BG_DECODE_THREADS:-1}"
export MEMRIFT_BWD_EMPTY_STEP="${MEMRIFT_BWD_EMPTY_STEP:-100000}"

PY_BT_ARGS=()
if [ "${NSIGHT_PYTHON_BT:-1}" != "0" ]; then
  PY_BT_ARGS=(--python-backtrace-cuda)
fi

python scripts/profile_memrift_nsight.py \
  --config-name train_llama31_8b_seq4096_bs1_memrift_weight_only \
  --compressed-weight-dir "$MEMRIFT_WEIGHT_DIR" \
  --iters "$NSIGHT_ITERS" \
  --fast-profile \
  --output "$NSIGHT_OUT" \
  "${PY_BT_ARGS[@]}" \
  --hydra-override "train.model.tokenizer_path=$LLAMA31_MODEL" \
  --hydra-override "train.model.tokenizer_model=$LLAMA31_MODEL" \
  "$@"

echo ""
echo "完成。打开: ${NSIGHT_OUT}.nsys-rep"
echo "显存峰值：Nsight GUI → CUDA Memory 轨道；或:"
echo "  nsys stats --report cuda_gpu_mem_size_sum \"${NSIGHT_OUT}.nsys-rep\""
