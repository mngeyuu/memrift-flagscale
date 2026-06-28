#!/usr/bin/env bash
# Shared helpers for MemRift metric scripts.

set -euo pipefail

METRICS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$METRICS_DIR/../.." && pwd)"

activate_env() {
  if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
    local env_name="${CONDA_ENV:-myc-fl312}"
    if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
      conda activate "$env_name"
    else
      echo "[metrics] conda env '$env_name' not found locally; continuing with current shell" >&2
    fi
  fi
}

setup_common_env() {
  cd "$REPO_ROOT"
  export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/flagscale:$REPO_ROOT/flagscale/train:$REPO_ROOT/flagscale/train/megatron:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export ALPACA_DATA_PATH="${ALPACA_DATA_PATH:-/share/project/mengyc/data/alpaca_megatron/alpaca_text_document}"
}

ceil_120_percent() {
  python3 - "$1" <<'PY'
import math
import sys
print(math.ceil(int(sys.argv[1]) * 1.2))
PY
}

host_log_for() {
  local exp_dir="$1"
  if [ -f "$exp_dir/logs/host_0_localhost.output" ]; then
    printf '%s\n' "$exp_dir/logs/host_0_localhost.output"
  elif [ -f "$exp_dir/logs/host.output" ]; then
    printf '%s\n' "$exp_dir/logs/host.output"
  else
    printf '%s\n' "$exp_dir/stdout.log"
  fi
}

run_yaml_train() {
  local variant="$1"
  local exp_dir="$2"
  local mode="$3"
  local seq_len="$4"
  local train_iters="$5"
  local max_pos="${6:-$seq_len}"

  mkdir -p "$exp_dir"

  local common_args=(
    "--config-path=examples/memrift/conf"
    "--config-name=$CONFIG_NAME"
    "action=${ACTION:-test}"
    "experiment.exp_name=${MODEL_KEY}_${variant}"
    "experiment.exp_dir=$exp_dir"
    "experiment.task.entrypoint=flagscale/train/megatron/train_gpt.py"
    "experiment.runner.type=ssh"
    "experiment.runner.nnodes=1"
    "experiment.runner.nproc_per_node=1"
    "++experiment.envs.CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
    "++experiment.envs.TORCH_DEVICE_BACKEND_AUTOLOAD=0"
    "++train.system.run_foreground=false"
    "train.system.tensor_model_parallel_size=1"
    "train.system.pipeline_model_parallel_size=1"
    "train.system.context_parallel_size=1"
    "train.model.tokenizer_path=$MODEL_PATH"
    "train.model.tokenizer_model=$MODEL_PATH"
    "train.model.seq_length=$seq_len"
    "train.model.max_position_embeddings=$max_pos"
    "train.data.mock_data=false"
    "train.data.data_path=$ALPACA_DATA_PATH"
    "train.data.micro_batch_size=1"
    "train.data.global_batch_size=1"
    "train.trainer.train_iters=$train_iters"
    "++train.trainer.eval_iters=0"
    "train.trainer.lr_warmup_iters=0"
    "train.system.logging.log_interval=1"
  )

  local mode_args=()
  case "$mode" in
    lora)
      mode_args=(
        "train.system.memrift_enable=false"
        "train.system.memrift_weight_enable=false"
        "train.system.memrift_activation_enable=false"
        "+train.system.init_model_with_meta_device=false"
      )
      ;;
    memrift_async)
      mode_args=(
        "train.system.memrift_enable=true"
        "train.system.memrift_weight_enable=true"
        "train.system.memrift_activation_enable=true"
        "train.system.memrift_weight_async=true"
        "train.system.memrift_act_async=true"
        "train.system.memrift_compressed_weight_dir=$MEMRIFT_WEIGHT_DIR"
      )
      ;;
    *)
      echo "unknown run mode: $mode" >&2
      return 2
      ;;
  esac

  echo "[$MODEL_KEY][$variant] mode=$mode seq=$seq_len iters=$train_iters gpu=${CUDA_VISIBLE_DEVICES:-0}"
  local t0 t1
  t0=$(python3 - <<'PY'
import time
print(time.time())
PY
)
  set +e
  python run.py "${common_args[@]}" "${mode_args[@]}" 2>&1 | tee "$exp_dir/stdout.log"
  local rc=${PIPESTATUS[0]}
  set -e
  t1=$(python3 - <<'PY'
import time
print(time.time())
PY
)
  python3 - "$exp_dir/wall_time.json" "$t0" "$t1" "$rc" <<'PY'
import json
import sys
out, t0, t1, rc = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
with open(out, "w", encoding="utf-8") as f:
    json.dump({"wall_seconds": t1 - t0, "launcher_return_code": rc}, f, indent=2)
    f.write("\n")
PY
  return "$rc"
}

parse_train_log_json() {
  local log_file="$1"
  local json_out="$2"
  python3 - "$log_file" "$json_out" <<'PY'
import json
import re
import sys
from pathlib import Path

log = Path(sys.argv[1])
out = Path(sys.argv[2])
text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""

losses = [float(x) for x in re.findall(r"lm loss:\s*([0-9.+\-Ee]+)", text)]
iter_ms = [float(x) for x in re.findall(r"elapsed time per iteration \(ms\):\s*([0-9.]+)", text)]
max_alloc = None
for m in re.finditer(r"max allocated:\s*([0-9.]+)", text):
    max_alloc = float(m.group(1))

failed = bool(re.search(r"(Traceback|FAILED|error:|OutOfMemory|out of memory|CUDA out of memory)", text, re.I))
data = {
    "source_log": str(log),
    "log_exists": log.is_file(),
    "failed_pattern_found": failed,
    "final_lm_loss": losses[-1] if losses else None,
    "lm_loss_samples": losses,
    "elapsed_ms_per_iter_mean": sum(iter_ms) / len(iter_ms) if iter_ms else None,
    "elapsed_ms_per_iter_last": iter_ms[-1] if iter_ms else None,
    "num_iter_time_samples": len(iter_ms),
    "max_allocated_mb": max_alloc,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(data, ensure_ascii=False))
PY
}

write_metric_result() {
  local out_file="$1"
  shift
  python3 - "$out_file" "$@" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
pairs = sys.argv[2:]
data = {}
for item in pairs:
    key, value = item.split("=", 1)
    try:
        data[key] = json.loads(value)
    except Exception:
        data[key] = value
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(out)
PY
}
