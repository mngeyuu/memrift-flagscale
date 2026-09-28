#!/usr/bin/env bash
# Metric 1.1: final lm loss degradation <= 1%, Pure LoRA vs MemRift weight-compressed LoRA.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"
activate_env
setup_common_env
# shellcheck source=model_env.sh
source "$SCRIPT_DIR/model_env.sh"

METRIC_DIR="$OUT_ROOT/accuracy_loss"
BASE_DIR="$METRIC_DIR/pure_lora"
MEM_DIR="$METRIC_DIR/memrift_weight_only"
RESULT="$METRIC_DIR/result.json"
BASE_TRAIN_METRICS="$BASE_DIR/train_metrics.json"
MEM_TRAIN_METRICS="$MEM_DIR/train_metrics.json"

start_metric_result_display "$RESULT"
trap 'finish_metric_result_display "$?" accuracy_loss "$RESULT" "$METRIC_FRESHNESS_MARKER"' EXIT

ensure_memrift_weights "$MODEL_PATH" "$MEMRIFT_WEIGHT_DIR" "${MEMRIFT_PREPARE_LEVEL:-18}"

if [ ! -f "$MEGATRON_CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
  echo "[metrics] missing Aquila Megatron checkpoint: $MEGATRON_CKPT_DIR" >&2
  echo "[metrics] run scripts/metrics/aquila/convert_hf_to_mcore_tp1.sh first" >&2
  exit 2
fi

# FlagScale host logs are append-only. Use clean directories so stale failures
# and loss samples from earlier runs cannot affect this result.
rm -rf "$BASE_DIR" "$MEM_DIR"

DISABLE_TRAIN_CHECKPOINT=true MEMRIFT_DISABLE_FINAL_CHECKPOINT=1 \
  run_yaml_train pure_lora "$BASE_DIR" lora "$BASE_SEQ_LEN" "$TRAIN_ITERS" "$MAX_POSITION_EMBEDDINGS"
DISABLE_TRAIN_CHECKPOINT=true MEMRIFT_DISABLE_FINAL_CHECKPOINT=1 \
  MEMRIFT_ACTIVATION_ENABLE=false \
  MEMRIFT_KEEP_WEIGHTS_RESIDENT=1 \
  run_yaml_train memrift_weight_only "$MEM_DIR" memrift_async "$BASE_SEQ_LEN" "$TRAIN_ITERS" "$MAX_POSITION_EMBEDDINGS"

parse_train_log_json "$(host_log_for "$BASE_DIR")" "$BASE_TRAIN_METRICS" >/dev/null
parse_train_log_json "$(host_log_for "$MEM_DIR")" "$MEM_TRAIN_METRICS" >/dev/null

python3 scripts/metrics/common_accuracy_loss.py \
  --baseline "$BASE_TRAIN_METRICS" --candidate "$MEM_TRAIN_METRICS" --output "$RESULT" \
  --model-key "$MODEL_KEY" --model-name "$MODEL_NAME" \
  --threshold "${ACCURACY_THRESHOLD:-0.01}"
