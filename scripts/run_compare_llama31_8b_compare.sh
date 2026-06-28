#!/usr/bin/env bash
# LLaMA-3.1-8B comparison driven by examples/memrift/conf/train_llama31_8b_mock.yaml.
# Runs:
#   1. MemRift + LoRA (weight + activation)
#   2. Pure LoRA baseline
#
# Useful overrides:
#   CUDA_VISIBLE_DEVICES=1 TRAIN_ITERS=5 bash scripts/run_compare_llama31_8b_compare.sh
#   ACTION=dryrun bash scripts/run_compare_llama31_8b_compare.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-myc-fl312}"
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/flagscale:$ROOT_DIR/flagscale/train:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

CONFIG_PATH="${CONFIG_PATH:-examples/memrift/conf}"
CONFIG_NAME="${CONFIG_NAME:-train_llama31_8b_mock}"
ACTION="${ACTION:-run}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
MODEL_PATH="${MODEL_PATH:-/share/project/mengyc/models/Meta-Llama-3-8B-Instruct}"
MEMRIFT_WEIGHTS="${MEMRIFT_WEIGHTS:-$ROOT_DIR/memrift_weights/llama31_8b_level18}"
EXP_TAG="${EXP_TAG:-llama31_8b_compare}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/output/$EXP_TAG}"

WARMUP=$((TRAIN_ITERS / 5))
if [ "$WARMUP" -lt 1 ]; then WARMUP=1; fi
if [ "$WARMUP" -ge "$TRAIN_ITERS" ]; then WARMUP=$((TRAIN_ITERS - 1)); fi
if [ "$WARMUP" -lt 0 ]; then WARMUP=0; fi

run_yaml() {
  local name="$1"
  shift
  local exp_dir="$OUT_ROOT/$name"
  mkdir -p "$exp_dir"

  echo ""
  echo "=== [$name] YAML-driven run: action=$ACTION ==="
  python run.py \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    action="$ACTION" \
    experiment.exp_name="$name" \
    experiment.exp_dir="$exp_dir" \
    experiment.task.entrypoint=flagscale/train/megatron/train_gpt.py \
    experiment.runner.type=ssh \
    "~train.system.run_foreground" \
    train.model.tokenizer_path="$MODEL_PATH" \
    train.model.tokenizer_model="$MODEL_PATH" \
    train.trainer.train_iters="$TRAIN_ITERS" \
    train.trainer.lr_warmup_iters="$WARMUP" \
    train.system.logging.log_interval=1 \
    "$@" \
    2>&1 | tee "$exp_dir/stdout.log"
}

echo "=============================================="
echo " LLaMA-3.1-8B YAML comparison"
echo "=============================================="
echo " ROOT_DIR:             $ROOT_DIR"
echo " CONFIG:               $CONFIG_PATH/$CONFIG_NAME"
echo " ACTION:               $ACTION"
echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo " MODEL_PATH:           $MODEL_PATH"
echo " MEMRIFT_WEIGHTS:      $MEMRIFT_WEIGHTS"
echo " TRAIN_ITERS:          $TRAIN_ITERS"
echo " OUT_ROOT:             $OUT_ROOT"
echo "=============================================="

run_yaml "${EXP_TAG}_memrift" \
  train.system.memrift_enable=true \
  train.system.memrift_weight_enable=true \
  train.system.memrift_activation_enable=true \
  train.system.memrift_weight_async=true \
  train.system.memrift_act_async=true \
  train.system.memrift_compressed_weight_dir="$MEMRIFT_WEIGHTS"

run_yaml "${EXP_TAG}_pure_lora" \
  train.system.memrift_enable=false \
  train.system.memrift_weight_enable=false \
  train.system.memrift_activation_enable=false \
  +train.system.init_model_with_meta_device=false

echo ""
echo "=== Comparison runs submitted/completed ==="
echo "Artifacts: $OUT_ROOT"
