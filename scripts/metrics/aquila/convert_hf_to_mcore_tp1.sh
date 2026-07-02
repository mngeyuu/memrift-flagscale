#!/usr/bin/env bash
# Convert Aquila2-7B HuggingFace weights to a Megatron-Core TP=1 checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"
activate_env
setup_common_env
# shellcheck source=model_env.sh
source "$SCRIPT_DIR/model_env.sh"

HF_MODEL="${HF_MODEL:-$MODEL_PATH}"
SAVE_DIR="${MEGATRON_CKPT_DIR:-/share/project/mengyc/models/Aquila2-7B-mcore-tp1}"
LOG_DIR="$REPO_ROOT/output/metrics/$MODEL_KEY/convert_hf_to_mcore_tp1"
LOG_FILE="$LOG_DIR/convert.log"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "=== Aquila2-7B HF -> Megatron-Core TP=1 ==="
echo "source: $HF_MODEL"
echo "target: $SAVE_DIR"
echo "log:    $LOG_FILE"

cd "$REPO_ROOT/tools/checkpoint"

python convert.py \
  --model-type aquila \
  --loader transformers \
  --saver mcore \
  --load-dir "$HF_MODEL" \
  --save-dir "$SAVE_DIR" \
  --target-tensor-parallel-size 1 \
  --target-pipeline-parallel-size 1 \
  --target-expert-parallel-size 1 \
  --target-params-dtype bf16 \
  --true-vocab-size 143973 \
  --position-embedding-type rope \
  --megatron-path "$REPO_ROOT/flagscale/train" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "done: $SAVE_DIR"
find "$SAVE_DIR" -maxdepth 3 -type f | sort
