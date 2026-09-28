#!/usr/bin/env bash
# Metric 1.2 training context: single-GPU max runnable context, LoRA vs MemRift+LoRA.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
activate_env
setup_common_env
source "$SCRIPT_DIR/model_env.sh"

METRIC_DIR="$OUT_ROOT/train_context_gain"
RESULT="$METRIC_DIR/result.json"
METRICS_MD="${METRICS_MD:-/share/project/mengyc/MemRift_metrics.md}"
CONTEXT_TRAIN_ITERS="${CONTEXT_TRAIN_ITERS:-2}"
CONTEXT_SEARCH_START="${CONTEXT_SEARCH_START:-8192}"
CONTEXT_SEARCH_STEP="${CONTEXT_SEARCH_STEP:-1024}"
CONTEXT_SEARCH_FINE_STEP="${CONTEXT_SEARCH_FINE_STEP:-128}"
CONTEXT_SEARCH_CAP="${CONTEXT_SEARCH_CAP:-131072}"

mkdir -p "$METRIC_DIR"

start_metric_result_display "$RESULT"
trap 'finish_metric_result_display "$?" train_context_gain "$RESULT" "$METRIC_FRESHNESS_MARKER"' EXIT

LORA_MAX_FILE="$METRIC_DIR/pure_lora_max_context.txt"
MEMRIFT_MAX_FILE="$METRIC_DIR/memrift_max_context.txt"
MEMRIFT_SEARCH_START_FILE="$METRIC_DIR/memrift_search_start.txt"

# Context probes intentionally do not pass a checkpoint save directory.
find_max_context_len lora pure_lora "$METRIC_DIR/pure_lora_search" \
  "$CONTEXT_SEARCH_START" "$CONTEXT_SEARCH_CAP" "$CONTEXT_SEARCH_STEP" "$CONTEXT_TRAIN_ITERS" "$LORA_MAX_FILE"

MEMRIFT_SEARCH_START="$(python3 - "$LORA_MAX_FILE" "$CONTEXT_SEARCH_CAP" <<'PY'
import math
import sys
from pathlib import Path

lora_max = int(Path(sys.argv[1]).read_text(encoding="utf-8").strip())
cap = int(sys.argv[2])
if lora_max <= 0:
    raise SystemExit("LoRA context search did not find any runnable sequence length")
start = math.ceil(lora_max * 1.2)
print(min(start, cap))
PY
)"
printf '%s\n' "$MEMRIFT_SEARCH_START" > "$MEMRIFT_SEARCH_START_FILE"

find_max_context_len memrift_async memrift_async "$METRIC_DIR/memrift_search" \
  "$MEMRIFT_SEARCH_START" "$CONTEXT_SEARCH_CAP" "$CONTEXT_SEARCH_STEP" "$CONTEXT_TRAIN_ITERS" "$MEMRIFT_MAX_FILE"

python3 - "$METRIC_DIR/pure_lora_search/attempts.jsonl" "$METRIC_DIR/memrift_search/attempts.jsonl" "$LORA_MAX_FILE" "$MEMRIFT_MAX_FILE" "$RESULT" "$MODEL_KEY" "$MODEL_NAME" "$CONTEXT_SEARCH_START" "$CONTEXT_SEARCH_STEP" "$CONTEXT_SEARCH_CAP" "$MEMRIFT_SEARCH_START_FILE" "$METRICS_MD" <<'PY'
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
mem_start = int(Path(sys.argv[11]).read_text().strip())
metrics_md = Path(sys.argv[12])
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
    "memrift_search_start_tokens": mem_start,
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
metrics_md.parent.mkdir(parents=True, exist_ok=True)
metrics_md.write_text(
    "\n".join(
        [
            "# MemRift Metrics",
            "",
            "## train_context_gain",
            "",
            f"- model: {sys.argv[7]} ({sys.argv[6]})",
            f"- pure_lora_max_context_tokens: {lora_max}",
            f"- memrift_search_start_tokens: {mem_start}",
            f"- memrift_max_context_tokens: {mem_max}",
            f"- context_gain_percent: {data['context_gain_percent']}",
            f"- pass: {data['pass']}",
            f"- result_json: {out}",
            "",
        ]
    ),
    encoding="utf-8",
)
print(json.dumps(data, indent=2, ensure_ascii=False))
PY
