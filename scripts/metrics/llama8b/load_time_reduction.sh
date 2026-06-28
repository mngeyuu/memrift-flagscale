#!/usr/bin/env bash
# Metric 1.2 inference load time: MemRift load time reduction >= 30%.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
activate_env
setup_common_env
source "$SCRIPT_DIR/model_env.sh"

python3 "$REPO_ROOT/scripts/metrics/common_load_time.py" \
  --model-path "$MODEL_PATH" \
  --compressed-dir "$MEMRIFT_WEIGHT_DIR" \
  --out "$OUT_ROOT/load_time_reduction/result.json" \
  --device "${LOAD_DEVICE:-cuda:0}" \
  --first-forward-tokens "${FIRST_FORWARD_TOKENS:-8}" \
  ${SKIP_BASELINE_LOAD:+--skip-baseline} \
  --target-reduction "${TARGET_LOAD_REDUCTION:-0.30}"
