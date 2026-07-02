#!/usr/bin/env bash
# Metric 1.1 accuracy gate: MemRift+LoRA loss degradation <= 1% vs pure LoRA.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$SCRIPT_DIR/../common.sh"
activate_env
setup_common_env
# shellcheck source=model_env.sh
source "$SCRIPT_DIR/model_env.sh"

ensure_memrift_weights "$MODEL_PATH" "$MEMRIFT_WEIGHT_DIR" "$MEMRIFT_PREPARE_LEVEL"

METRIC_DIR="$OUT_ROOT/accuracy_loss"
BASE_DIR="$METRIC_DIR/pure_lora"
MEM_DIR="$METRIC_DIR/memrift_weight_act_async"
RESULT="$METRIC_DIR/result.json"

run_yaml_train pure_lora "$BASE_DIR" lora "$BASE_SEQ_LEN" "$TRAIN_ITERS" "$MAX_POSITION_EMBEDDINGS"
run_yaml_train memrift_weight_act_async "$MEM_DIR" memrift_async "$BASE_SEQ_LEN" "$TRAIN_ITERS" "$MAX_POSITION_EMBEDDINGS"

BASE_LOG="$(host_log_for "$BASE_DIR")"
MEM_LOG="$(host_log_for "$MEM_DIR")"
parse_train_log_json "$BASE_LOG" "$BASE_DIR/metrics.json" >/dev/null
parse_train_log_json "$MEM_LOG" "$MEM_DIR/metrics.json" >/dev/null

python3 - "$BASE_DIR/metrics.json" "$MEM_DIR/metrics.json" "$RESULT" "$MODEL_KEY" "$MODEL_NAME" <<'PY'
import json
import sys
from pathlib import Path

base = json.loads(Path(sys.argv[1]).read_text())
mem = json.loads(Path(sys.argv[2]).read_text())
out = Path(sys.argv[3])
base_loss = base.get("final_lm_loss")
mem_loss = mem.get("final_lm_loss")
degradation = ((mem_loss - base_loss) / base_loss) if base_loss and mem_loss is not None else None
data = {
    "metric": "accuracy_loss",
    "model_key": sys.argv[4],
    "model_name": sys.argv[5],
    "criterion": "relative loss degradation <= 1%",
    "baseline_pure_lora": base,
    "memrift_weight_act_async": mem,
    "relative_loss_degradation": degradation,
    "relative_loss_degradation_percent": degradation * 100 if degradation is not None else None,
    "pass": bool(degradation is not None and degradation <= 0.01 and not mem.get("failed_pattern_found")),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(data, indent=2, ensure_ascii=False))
PY
