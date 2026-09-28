#!/usr/bin/env bash
# Metric 1.2 inference load time: MemRift compressed-weight disk-read reduction >= 30%.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
activate_env
setup_common_env
source "$SCRIPT_DIR/model_env.sh"

METRIC_DIR="$OUT_ROOT/load_time_reduction"
RESULT="$METRIC_DIR/result.json"

start_metric_result_display "$RESULT"
trap 'finish_metric_result_display "$?" load_time_reduction "$RESULT" "$METRIC_FRESHNESS_MARKER"' EXIT

ensure_memrift_weights "$MODEL_PATH" "$MEMRIFT_WEIGHT_DIR" "$MEMRIFT_PREPARE_LEVEL"

python3 "$REPO_ROOT/scripts/metrics/common_load_time.py" \
  --model-path "$MODEL_PATH" \
  --compressed-dir "$MEMRIFT_WEIGHT_DIR" \
  --out "$RESULT" \
  --model-key "$MODEL_KEY" \
  --model-name "$MODEL_NAME" \
  --repeat "${LOAD_TIME_REPEAT:-1}" \
  --chunk-size "${LOAD_TIME_CHUNK_SIZE:-67108864}" \
  --target-reduction "${TARGET_LOAD_REDUCTION:-0.30}"
