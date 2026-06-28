#!/usr/bin/env bash
# Metric 1.2 training context: MemRift+LoRA single-GPU context length gain vs pure LoRA.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
activate_env
setup_common_env
source "$SCRIPT_DIR/model_env.sh"

METRIC_DIR="$OUT_ROOT/train_context_gain"
BASE_DIR="$METRIC_DIR/pure_lora_L${BASE_SEQ_LEN}"
MEM_DIR="$METRIC_DIR/memrift_async_L${TARGET_SEQ_LEN}"
RESULT="$METRIC_DIR/result.json"
CONTEXT_TRAIN_ITERS="${CONTEXT_TRAIN_ITERS:-1}"

run_yaml_train "pure_lora_L${BASE_SEQ_LEN}" "$BASE_DIR" lora "$BASE_SEQ_LEN" "$CONTEXT_TRAIN_ITERS" "$MAX_POSITION_EMBEDDINGS"
run_yaml_train "memrift_async_L${TARGET_SEQ_LEN}" "$MEM_DIR" memrift_async "$TARGET_SEQ_LEN" "$CONTEXT_TRAIN_ITERS" "$MAX_POSITION_EMBEDDINGS"

BASE_LOG="$(host_log_for "$BASE_DIR")"
MEM_LOG="$(host_log_for "$MEM_DIR")"
parse_train_log_json "$BASE_LOG" "$BASE_DIR/metrics.json" >/dev/null
parse_train_log_json "$MEM_LOG" "$MEM_DIR/metrics.json" >/dev/null

python3 - "$BASE_DIR/metrics.json" "$MEM_DIR/metrics.json" "$RESULT" "$MODEL_KEY" "$MODEL_NAME" "$BASE_SEQ_LEN" "$TARGET_SEQ_LEN" <<'PY'
import json
import sys
from pathlib import Path

base = json.loads(Path(sys.argv[1]).read_text())
mem = json.loads(Path(sys.argv[2]).read_text())
out = Path(sys.argv[3])
base_len = int(sys.argv[6])
target_len = int(sys.argv[7])
gain = (target_len - base_len) / base_len
data = {
    "metric": "train_context_gain",
    "model_key": sys.argv[4],
    "model_name": sys.argv[5],
    "criterion": "MemRift+LoRA trains at >= 20% longer context than pure LoRA baseline",
    "baseline_context_tokens": base_len,
    "memrift_context_tokens": target_len,
    "context_gain_fraction": gain,
    "context_gain_percent": gain * 100,
    "baseline_pure_lora": base,
    "memrift_weight_act_async": mem,
    "pass": bool(gain >= 0.20 and not mem.get("failed_pattern_found") and mem.get("log_exists")),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(data, indent=2, ensure_ascii=False))
PY
