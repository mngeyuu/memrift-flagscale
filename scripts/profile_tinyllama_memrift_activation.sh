#!/bin/bash
# ============================================
# TinyLlama 1.1B MemRift + LoRA · 逐层激活显存 Profiling
# ============================================
# 在 FlagScale 中跑 MemRift + LoRA 训练，启用逐层显存 hook 追踪激活贡献。
#
# 用法: ./scripts/profile_tinyllama_memrift_activation.sh
#
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TINYLLAMA_MODEL="${TINYLLAMA_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$ROOT_DIR/memrift_weights/tinyllama_1b_level18}"
TRAIN_ITERS=5
SKIP_PREPARE="${SKIP_PREPARE:-0}"
CONFIG_PATH="examples/memrift/conf"
EXP_DIR="$ROOT_DIR/outputs/memrift_profile"

echo "============================================"
echo "TinyLlama MemRift + LoRA 激活显存 Profiling"
echo "============================================"
echo "ROOT_DIR:            $ROOT_DIR"
echo "MEMRIFT_WEIGHT_DIR:  $MEMRIFT_WEIGHT_DIR"
echo "TRAIN_ITERS:         $TRAIN_ITERS"
echo "EXP_DIR:             $EXP_DIR"
echo "============================================"

# 清理旧的 checkpoint 和日志
rm -rf "$EXP_DIR"

# 准备权重（如果未跳过）
if [ "$SKIP_PREPARE" != "1" ] && [ ! -f "$MEMRIFT_WEIGHT_DIR/index.json" ]; then
  echo ">> 正在生成 MemRift 压缩权重..."
  python -m flagscale.compress.memrift.offline_comp.prepare_weight \
    --model "$TINYLLAMA_MODEL" \
    --outdir "$MEMRIFT_WEIGHT_DIR" \
    --level 18
fi

export MEMRIFT_PROFILE_ITERS=3

OVERRIDES=(
  "train.system.memrift_compressed_weight_dir=$MEMRIFT_WEIGHT_DIR"
  "train.model.tokenizer_path=$TINYLLAMA_MODEL"
  "train.model.tokenizer_model=$TINYLLAMA_MODEL"
  "train.trainer.train_iters=$TRAIN_ITERS"
  "train.trainer.lr_warmup_iters=2"
  "train.system.memrift_activation_enable=true"
  "+train.system.memrift_profile_memory=true"
  "+train.system.log_memory_to_tensorboard=true"
  "experiment.exp_dir=$EXP_DIR"
)

echo ">> 启动训练..."
python run.py \
  --config-path="$CONFIG_PATH" \
  --config-name=train_mock \
  action=run \
  "${OVERRIDES[@]}"

echo ""
echo ">> 完成。查看日志获取显存分解报告:"
echo "  tail -100 $EXP_DIR/logs/host_0_localhost.output"
