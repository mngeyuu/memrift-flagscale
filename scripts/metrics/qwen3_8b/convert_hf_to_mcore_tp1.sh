#!/usr/bin/env bash
# Convert Qwen3-8B Hugging Face weights to a Megatron-Core TP=1 checkpoint.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
activate_env
setup_common_env
source "$SCRIPT_DIR/model_env.sh"

HF_MODEL="${HF_MODEL:-$MODEL_PATH}"
SAVE_DIR="${MEGATRON_CKPT_DIR}"
LOG_DIR="$OUT_ROOT/convert_hf_to_mcore_tp1"
LOG_FILE="$LOG_DIR/convert.log"

mkdir -p "$SAVE_DIR" "$LOG_DIR"
cd "$REPO_ROOT/tools/checkpoint"

python convert.py \
  --model-type qwen3 \
  --loader transformers \
  --saver mcore \
  --load-dir "$HF_MODEL" \
  --save-dir "$SAVE_DIR" \
  --target-tensor-parallel-size 1 \
  --target-pipeline-parallel-size 1 \
  --target-expert-parallel-size 1 \
  --target-params-dtype bf16 \
  --true-vocab-size 151936 \
  --position-embedding-type rope \
  --megatron-path "$REPO_ROOT/flagscale/train" \
  2>&1 | tee "$LOG_FILE"
