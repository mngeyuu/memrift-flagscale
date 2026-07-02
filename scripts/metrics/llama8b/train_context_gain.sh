#!/usr/bin/env bash
# Metric 1.2 training context: single-GPU max runnable context, LoRA vs MemRift+LoRA.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
activate_env
setup_common_env
source "$SCRIPT_DIR/model_env.sh"

ensure_memrift_weights "$MODEL_PATH" "$MEMRIFT_WEIGHT_DIR" "$MEMRIFT_PREPARE_LEVEL"

METRIC_DIR="$OUT_ROOT/train_context_gain"
RESULT="$METRIC_DIR/result.json"
CONTEXT_TRAIN_ITERS="${CONTEXT_TRAIN_ITERS:-1}"
CONTEXT_SEARCH_START="${CONTEXT_SEARCH_START:-$BASE_SEQ_LEN}"
CONTEXT_SEARCH_STEP="${CONTEXT_SEARCH_STEP:-512}"
CONTEXT_SEARCH_CAP="${CONTEXT_SEARCH_CAP:-8192}"

mkdir -p "$METRIC_DIR"

LORA_MAX_FILE="$METRIC_DIR/pure_lora_max_context.txt"
MEMRIFT_MAX_FILE="$METRIC_DIR/memrift_max_context.txt"
find_max_context_len lora pure_lora "$METRIC_DIR/pure_lora_search" \
  "$CONTEXT_SEARCH_START" "$CONTEXT_SEARCH_CAP" "$CONTEXT_SEARCH_STEP" "$CONTEXT_TRAIN_ITERS" "$LORA_MAX_FILE"
find_max_context_len memrift_async memrift_async "$METRIC_DIR/memrift_search" \
  "$CONTEXT_SEARCH_START" "$CONTEXT_SEARCH_CAP" "$CONTEXT_SEARCH_STEP" "$CONTEXT_TRAIN_ITERS" "$MEMRIFT_MAX_FILE"

python3 - "$METRIC_DIR/pure_lora_search/attempts.jsonl" "$METRIC_DIR/memrift_search/attempts.jsonl" "$LORA_MAX_FILE" "$MEMRIFT_MAX_FILE" "$RESULT" "$MODEL_KEY" "$MODEL_NAME" "$CONTEXT_SEARCH_START" "$CONTEXT_SEARCH_STEP" "$CONTEXT_SEARCH_CAP" <<'PY'
import json
import sys
from pathlib import Path

def read_attempts(path):
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

lora_attempts = read_attempts(sys.argv[1])
mem_attempts = read_attempts(sys.argv[2])
lora_max = int(Path(sys.argv[3]).read_text().strip())
mem_max = int(Path(sys.argv[4]).read_text().strip())
out = Path(sys.argv[5])
cap = int(sys.argv[10])
gain = (mem_max - lora_max) / lora_max if lora_max else None
baseline_hit_cap = lora_max >= cap
memrift_hit_cap = mem_max >= cap

def attempt_at(attempts, seq):
    for item in reversed(attempts):
        if int(item["seq_len"]) == seq:
            return item
    return None

data = {
    "metric": "train_context_gain",
    "model_key": sys.argv[6],
    "model_name": sys.argv[7],
    "criterion": "single-GPU max runnable MemRift+LoRA context >= 20% longer than max runnable pure LoRA context",
    "search_start_tokens": int(sys.argv[8]),
    "search_step_tokens": int(sys.argv[9]),
    "search_cap_tokens": cap,
    "baseline_max_context_tokens": lora_max,
    "memrift_max_context_tokens": mem_max,
    "context_gain_fraction": gain,
    "context_gain_percent": gain * 100 if gain is not None else None,
    "baseline_hit_search_cap": baseline_hit_cap,
    "memrift_hit_search_cap": memrift_hit_cap,
    "boundary_resolved": not (baseline_hit_cap or memrift_hit_cap),
    "cap_note": (
        "At least one side reached CONTEXT_SEARCH_CAP; increase CONTEXT_SEARCH_CAP "
        "to measure the real max-context boundary."
        if baseline_hit_cap or memrift_hit_cap else ""
    ),
    "baseline_pure_lora_max_attempt": attempt_at(lora_attempts, lora_max),
    "memrift_weight_act_async_max_attempt": attempt_at(mem_attempts, mem_max),
    "baseline_attempts": lora_attempts,
    "memrift_attempts": mem_attempts,
    "pass": bool(gain is not None and gain >= 0.20),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(data, indent=2, ensure_ascii=False))
PY
