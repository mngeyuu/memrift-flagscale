#!/usr/bin/env bash
# Metric 1.1: final lm loss degradation <= 1%, Pure LoRA vs MemRift+LoRA.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
activate_env
setup_common_env
source "$SCRIPT_DIR/model_env.sh"

METRIC_DIR="$OUT_ROOT/accuracy_loss"
BASE_DIR="$METRIC_DIR/pure_lora"
MEM_DIR="$METRIC_DIR/memrift_weight_act_async"
RESULT="$METRIC_DIR/result.json"
BASE_TRAIN_METRICS="$BASE_DIR/train_metrics.json"
MEM_TRAIN_METRICS="$MEM_DIR/train_metrics.json"

start_metric_result_display "$RESULT"
trap 'finish_metric_result_display "$?" accuracy_loss "$RESULT" "$METRIC_FRESHNESS_MARKER"' EXIT

ensure_memrift_weights "$MODEL_PATH" "$MEMRIFT_WEIGHT_DIR" "$MEMRIFT_PREPARE_LEVEL"

DISABLE_TRAIN_CHECKPOINT=true MEMRIFT_DISABLE_FINAL_CHECKPOINT=1 \
  run_yaml_train pure_lora "$BASE_DIR" lora "$BASE_SEQ_LEN" "$TRAIN_ITERS" "$MAX_POSITION_EMBEDDINGS"
DISABLE_TRAIN_CHECKPOINT=true MEMRIFT_DISABLE_FINAL_CHECKPOINT=1 \
  run_yaml_train memrift_weight_act_async "$MEM_DIR" memrift_async "$BASE_SEQ_LEN" "$TRAIN_ITERS" "$MAX_POSITION_EMBEDDINGS"

parse_train_log_json "$(host_log_for "$BASE_DIR")" "$BASE_TRAIN_METRICS" >/dev/null
parse_train_log_json "$(host_log_for "$MEM_DIR")" "$MEM_TRAIN_METRICS" >/dev/null

python3 scripts/metrics/common_accuracy_loss.py \
  --baseline "$BASE_TRAIN_METRICS" --candidate "$MEM_TRAIN_METRICS" --output "$RESULT" \
  --model-key "$MODEL_KEY" --model-name "$MODEL_NAME" \
  --threshold "${ACCURACY_THRESHOLD:-0.01}"
