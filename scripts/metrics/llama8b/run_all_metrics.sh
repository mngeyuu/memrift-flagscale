#!/usr/bin/env bash
# Run all LLaMA-3.1-8B MemRift metrics.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
activate_env
setup_common_env
source "$SCRIPT_DIR/model_env.sh"

ensure_memrift_weights "$MODEL_PATH" "$MEMRIFT_WEIGHT_DIR" "$MEMRIFT_PREPARE_LEVEL"

bash "$SCRIPT_DIR/compression_ratio.sh"
bash "$SCRIPT_DIR/load_time_reduction.sh"
bash "$SCRIPT_DIR/accuracy_loss.sh"
bash "$SCRIPT_DIR/train_context_gain.sh"

python3 - "$OUT_ROOT/summary.json" \
  "$OUT_ROOT/compression_ratio/result.json" \
  "$OUT_ROOT/load_time_reduction/result.json" \
  "$OUT_ROOT/accuracy_loss/result.json" \
  "$OUT_ROOT/train_context_gain/result.json" <<'PY'
import json
import sys
from pathlib import Path

summary = {}
for path in map(Path, sys.argv[2:]):
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        summary[data.get("metric", path.parent.name)] = data
    else:
        summary[path.parent.name] = {"missing_result": str(path)}

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
