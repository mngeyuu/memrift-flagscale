#!/usr/bin/env bash
# Metric 1.1 compression gate: compressed storage saving >= 30%.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
activate_env
setup_common_env
source "$SCRIPT_DIR/model_env.sh"

RESULT="$OUT_ROOT/compression_ratio/result.json"
start_metric_result_display "$RESULT"
trap 'finish_metric_result_display "$?" compression_ratio "$RESULT" "$METRIC_FRESHNESS_MARKER"' EXIT

ensure_memrift_weights "$MODEL_PATH" "$MEMRIFT_WEIGHT_DIR" "$MEMRIFT_PREPARE_LEVEL"

python3 "$REPO_ROOT/scripts/metrics/common_compression.py" \
  --model-path "$MODEL_PATH" \
  --compressed-dir "$MEMRIFT_WEIGHT_DIR" \
  --out "$RESULT" \
  --target-saving "${TARGET_SAVING:-0.30}"
