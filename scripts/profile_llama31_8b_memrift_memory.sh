#!/bin/bash
# ============================================
# Llama-3.1-8B MemRift 训练 · 内存 Profiling
# ============================================
# 用 Llama-3.1-8B 验证 MemRift 在大模型上的显存优势
#
# 用法（在仓库根目录 FlagScale 下执行）:
#   ./scripts/profile_llama31_8b_memrift_memory.sh
#
# 可选环境变量:
#   LLAMA_MODEL         - HF 模型 ID，默认 meta-llama/Llama-3.1-8B
#   MEMRIFT_WEIGHT_DIR  - 压缩权重目录
#   TRAIN_ITERS         - 训练步数，默认 50
#   SKIP_PREPARE        - 设为 1 跳过压缩权重准备
#   CUDA_DEVICE         - 使用的 GPU 编号，默认 1
#
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LLAMA_MODEL="${LLAMA_MODEL:-meta-llama/Llama-3.1-8B}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$ROOT_DIR/memrift_weights/llama31_8b_level18}"
TRAIN_ITERS="${TRAIN_ITERS:-50}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
MEMORY_SNAPSHOT_PATH="${MEMORY_SNAPSHOT_PATH:-memrift_llama31_8b_memory_snapshot.pickle}"

CONFIG_PATH="examples/memrift/conf"
EXP_DIR="${EXP_DIR:-$ROOT_DIR/outputs/memrift_llama31_8b}"

echo "============================================"
echo "Llama-3.1-8B MemRift 训练 · 内存 Profiling"
echo "============================================"
echo "ROOT_DIR:            $ROOT_DIR"
echo "LLAMA_MODEL:         $LLAMA_MODEL"
echo "MEMRIFT_WEIGHT_DIR:  $MEMRIFT_WEIGHT_DIR"
echo "TRAIN_ITERS:         $TRAIN_ITERS"
echo "CUDA_DEVICE:         $CUDA_DEVICE"
echo "log_memory_to_tensorboard: true"
echo "record_memory_history:      true"
echo "============================================"

# 若未跳过且无压缩权重，先准备权重
if [ "$SKIP_PREPARE" != "1" ] && [ ! -f "$MEMRIFT_WEIGHT_DIR/index.json" ]; then
  echo ">> 正在生成 MemRift 压缩权重（Llama-3.1-8B，需几分钟）..."
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m flagscale.compress.memrift.offline_comp.prepare_weight \
    --model "$LLAMA_MODEL" \
    --outdir "$MEMRIFT_WEIGHT_DIR" \
    --level 18
  echo ">> 压缩权重已生成: $MEMRIFT_WEIGHT_DIR"
fi

OVERRIDES=(
  "train.system.memrift_compressed_weight_dir=$MEMRIFT_WEIGHT_DIR"
  "train.model.tokenizer_path=$LLAMA_MODEL"
  "train.model.tokenizer_model=$LLAMA_MODEL"
  "train.trainer.train_iters=$TRAIN_ITERS"
  "train.system.memrift_activation_enable=true"
  "+train.system.log_memory_to_tensorboard=true"
  "+train.system.record_memory_history=true"
  "+train.system.memory_snapshot_path=$MEMORY_SNAPSHOT_PATH"
)

echo ">> 启动训练（后台运行；看日志: tail -f $EXP_DIR/logs/host_0_localhost.output）..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python run.py \
  --config-path="$CONFIG_PATH" \
  --config-name=train_llama31_8b_mock \
  action=run \
  "${OVERRIDES[@]}"

echo ""
echo ">> 查看内存结果:"
echo "  1) 日志: $EXP_DIR/logs/host_0_localhost.output 中 report_memory"
echo "  2) TensorBoard: tensorboard --logdir=$EXP_DIR/tensorboard"
echo "  3) 内存快照: $ROOT_DIR/$MEMORY_SNAPSHOT_PATH"
