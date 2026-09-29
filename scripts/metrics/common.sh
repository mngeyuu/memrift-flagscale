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
  local runtime_deps="${METRICS_RUNTIME_DEPS:-$REPO_ROOT/output/runtime_deps/hf_compat}"
  if [ -d "$runtime_deps" ]; then
    export PYTHONPATH="$runtime_deps:$REPO_ROOT:$REPO_ROOT/flagscale:$REPO_ROOT/flagscale/train:$REPO_ROOT/flagscale/train/megatron:${PYTHONPATH:-}"
  else
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/flagscale:$REPO_ROOT/flagscale/train:$REPO_ROOT/flagscale/train/megatron:${PYTHONPATH:-}"
  fi
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export ALPACA_DATA_PATH="${ALPACA_DATA_PATH:-/share/project/mengyc/data/alpaca_megatron/alpaca_text_document}"
}

ensure_memrift_weights() {
  local model_path="$1"
  local weight_dir="$2"
  local level="${3:-18}"

  if [ -f "$weight_dir/index.json" ]; then
    echo "[metrics] found compressed weights: $weight_dir/index.json"
    return 0
  fi

  if [ "${PREPARE_MEMRIFT_WEIGHTS:-1}" = "0" ]; then
    echo "[metrics] missing compressed weights and PREPARE_MEMRIFT_WEIGHTS=0: $weight_dir/index.json" >&2
    return 2
  fi

  if [ ! -e "$model_path" ]; then
    echo "[metrics] model path not found: $model_path" >&2
    return 2
  fi

  echo "[metrics] compressed weights not found; preparing:"
  echo "  model:  $model_path"
  echo "  outdir: $weight_dir"
  echo "  level:  $level"
  mkdir -p "$weight_dir"
  python3 -m flagscale.compress.memrift.offline_comp.prepare_weight \
    --model "$model_path" \
    --outdir "$weight_dir" \
    --level "$level"
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
  local save_dir="${7:-}"

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
    "++experiment.envs.MEMRIFT_DISABLE_FINAL_CHECKPOINT=${MEMRIFT_DISABLE_FINAL_CHECKPOINT:-0}"
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
    "++train.model.train_iters=$train_iters"
    "++train.model.eval_interval=$train_iters"
    "++train.model.eval_iters=0"
    "++train.model.lr=2e-4"
    "++train.model.min_lr=2e-5"
    "++train.model.lr_warmup_iters=0"
    "++train.model.lr_decay_style=cosine"
    "++train.model.weight_decay=0.1"
    "++train.model.clip_grad=1.0"
    "++train.model.adam_beta1=0.9"
    "++train.model.adam_beta2=0.95"
    "++train.model.adam_eps=1e-8"
    "train.system.logging.log_interval=1"
  )

  if [ -n "${MEMRIFT_ADAPTER_SAVE_DIR:-}" ]; then
    common_args+=(
      "++experiment.envs.MEMRIFT_ADAPTER_SAVE_DIR=$MEMRIFT_ADAPTER_SAVE_DIR"
    )
  fi

  # The uncompressed LoRA baseline must start from the same pretrained model
  # as the MemRift candidate. The candidate obtains those weights from
  # MEMRIFT_WEIGHT_DIR, so load the converted checkpoint for the baseline only.
  if [ "$mode" = "lora" ] && [ -n "${MEGATRON_CKPT_DIR:-}" ]; then
    common_args+=(
      "+train.system.checkpoint.load=$MEGATRON_CKPT_DIR"
      "+train.system.checkpoint.ckpt_format=torch"
      "+train.system.checkpoint.exit_on_missing_checkpoint=true"
      "+train.system.no_load_optim=true"
      "+train.system.no_load_rng=true"
      "+train.system.finetune=true"
    )
  fi

  if [ -n "$save_dir" ]; then
    common_args+=(
      "+train.system.checkpoint.save=$save_dir"
      "train.system.checkpoint.save_interval=$train_iters"
    )
  elif [ "${DISABLE_TRAIN_CHECKPOINT:-false}" = "true" ]; then
    common_args+=(
      "+train.system.checkpoint.save=null"
      "train.system.checkpoint.save_interval=1000000000"
    )
    if [ "$mode" != "lora" ] || [ -z "${MEGATRON_CKPT_DIR:-}" ]; then
      common_args+=("+train.system.checkpoint.load=null")
    fi
  fi

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
        "train.system.memrift_activation_enable=${MEMRIFT_ACTIVATION_ENABLE:-true}"
        "train.system.memrift_weight_async=${MEMRIFT_WEIGHT_ASYNC:-true}"
        "train.system.memrift_act_async=${MEMRIFT_ACT_ASYNC:-true}"
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

run_benchmark_accuracy() {
  local variant="$1"
  local exp_dir="$2"
  local mode="$3"
  local checkpoint_dir="$4"
  local output_file="$5"

  mkdir -p "$exp_dir" "$(dirname "$output_file")"
  local common_args=(
    "--config-path=examples/memrift/conf"
    "--config-name=$CONFIG_NAME"
    "action=${ACTION:-test}"
    "experiment.exp_name=${MODEL_KEY}_${variant}_benchmark"
    "experiment.exp_dir=$exp_dir"
    "experiment.task.entrypoint=scripts/metrics/benchmark_accuracy.py"
    "experiment.runner.type=ssh"
    "experiment.runner.nnodes=1"
    "experiment.runner.nproc_per_node=1"
    "++experiment.envs.CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
    "++experiment.envs.TORCH_DEVICE_BACKEND_AUTOLOAD=0"
    "++experiment.envs.BENCHMARK_VARIANT=$variant"
    "++experiment.envs.BENCHMARK_MODE=$mode"
    "++experiment.envs.BENCHMARK_CHECKPOINT=$checkpoint_dir"
    "++experiment.envs.BENCHMARK_BASE_CHECKPOINT=${BENCHMARK_BASE_CHECKPOINT:-$MEGATRON_CKPT_DIR}"
    "++experiment.envs.BENCHMARK_OUTPUT=$output_file"
    "++experiment.envs.GSM8K_LIMIT=${GSM8K_LIMIT:-}"
    "++experiment.envs.HELLASWAG_LIMIT=${HELLASWAG_LIMIT:-}"
    "++experiment.envs.GSM8K_MAX_NEW_TOKENS=${GSM8K_MAX_NEW_TOKENS:-64}"
    "++experiment.envs.BENCHMARK_SHARD_INDEX=${BENCHMARK_SHARD_INDEX:-0}"
    "++experiment.envs.BENCHMARK_SHARD_COUNT=${BENCHMARK_SHARD_COUNT:-1}"
    "train.system.tensor_model_parallel_size=1"
    "train.system.pipeline_model_parallel_size=1"
    "train.system.context_parallel_size=1"
    "train.model.tokenizer_path=$MODEL_PATH"
    "train.model.tokenizer_model=$MODEL_PATH"
    "+train.system.checkpoint.load=$checkpoint_dir"
    "+train.system.checkpoint.ckpt_format=torch"
    "+train.system.no_load_optim=true"
    "+train.system.no_load_rng=true"
  )

  if [ "$mode" = "lora" ] || [ "$mode" = "lora_adapter_only" ]; then
    common_args+=(
      "train.system.memrift_enable=false"
      "train.system.memrift_weight_enable=false"
      "train.system.memrift_activation_enable=false"
      "+train.system.init_model_with_meta_device=false"
    )
  else
    common_args+=(
      "++experiment.envs.MEMRIFT_KEEP_WEIGHTS_RESIDENT=${MEMRIFT_KEEP_WEIGHTS_RESIDENT:-0}"
      "train.system.memrift_enable=true"
      "train.system.memrift_weight_enable=true"
      "train.system.memrift_activation_enable=${MEMRIFT_ACTIVATION_ENABLE:-true}"
      "train.system.memrift_weight_async=true"
      "train.system.memrift_act_async=${MEMRIFT_ACT_ASYNC:-true}"
      "train.system.memrift_compressed_weight_dir=$MEMRIFT_WEIGHT_DIR"
    )
  fi

  python run.py "${common_args[@]}" 2>&1 | tee "$exp_dir/stdout.log"
}

run_context_probe() {
  local label="$1"
  local mode="$2"
  local seq_len="$3"
  local train_iters="$4"
  local exp_dir="$5"

  rm -rf "$exp_dir"
  DISABLE_TRAIN_CHECKPOINT=true MEMRIFT_DISABLE_FINAL_CHECKPOINT=1 run_yaml_train "$label" "$exp_dir" "$mode" "$seq_len" "$train_iters" "$seq_len" || true

  local log_file metrics_file
  log_file="$(host_log_for "$exp_dir")"
  metrics_file="$exp_dir/metrics.json"
  parse_train_log_json "$log_file" "$metrics_file" >/dev/null || true
  local wall_time_file="$exp_dir/wall_time.json"
  python3 - "$metrics_file" "$wall_time_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
wall_time_path = Path(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
data = json.loads(path.read_text(encoding="utf-8"))
wall_time = json.loads(wall_time_path.read_text(encoding="utf-8")) if wall_time_path.is_file() else {}
data["launcher_return_code"] = wall_time.get("launcher_return_code")
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
ok = (
    data.get("log_exists")
    and data.get("launcher_return_code") == 0
    and not data.get("failed_pattern_found")
    and data.get("final_lm_loss") is not None
)
raise SystemExit(0 if ok else 1)
PY
}

find_max_context_len() {
  local mode="$1"
  local prefix="$2"
  local out_dir="$3"
  local start_len="$4"
  local cap_len="$5"
  local step_len="$6"
  local train_iters="$7"
  local result_file="${8:-}"
  local fine_step_len="${CONTEXT_SEARCH_FINE_STEP:-128}"

  mkdir -p "$out_dir"
  local attempts_jsonl="$out_dir/attempts.jsonl"
  : > "$attempts_jsonl"

  record_context_attempt() {
    local phase="$1"
    local seq="$2"
    local ok="$3"
    local exp_dir="$4"
    local log_file
    log_file="$(host_log_for "$exp_dir")"
    parse_train_log_json "$log_file" "$exp_dir/metrics.json" >/dev/null || true
    local wall_time_file="$exp_dir/wall_time.json"
    python3 - "$exp_dir/metrics.json" "$wall_time_file" <<'PY'
import json
import sys
from pathlib import Path

metrics_file = Path(sys.argv[1])
wall_time_file = Path(sys.argv[2])
if metrics_file.is_file() and wall_time_file.is_file():
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    wall_time = json.loads(wall_time_file.read_text(encoding="utf-8"))
    metrics["launcher_return_code"] = wall_time.get("launcher_return_code")
    metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
    python3 - "$attempts_jsonl" "$seq" "$ok" "$phase" "$exp_dir/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
seq = int(sys.argv[2])
ok = bool(int(sys.argv[3]))
phase = sys.argv[4]
metrics_file = Path(sys.argv[5])
metrics = json.loads(metrics_file.read_text()) if metrics_file.is_file() else {}
with out.open("a", encoding="utf-8") as f:
    f.write(json.dumps({"phase": phase, "seq_len": seq, "ok": ok, "metrics": metrics}, ensure_ascii=False) + "\n")
PY
  }

  local best=0
  local first_fail=0
  local seq="$start_len"

  while [ "$seq" -le "$cap_len" ]; do
    local exp_dir="$out_dir/${prefix}_L${seq}"
    echo "[$MODEL_KEY][$prefix] coarse linear probing seq=$seq cap=$cap_len step=$step_len mode=$mode" >&2
    local ok=0
    if run_context_probe "${prefix}_L${seq}" "$mode" "$seq" "$train_iters" "$exp_dir"; then
      ok=1
      best="$seq"
    else
      first_fail="$seq"
    fi
    record_context_attempt "coarse" "$seq" "$ok" "$exp_dir"
    if [ "$ok" -eq 0 ]; then
      break
    fi
    seq="$((seq + step_len))"
  done

  if [ "$best" -eq 0 ]; then
    if [ -n "$result_file" ]; then
      printf '0\n' > "$result_file"
    else
      echo "0"
    fi
    return 0
  fi

  if [ "$first_fail" -eq 0 ]; then
    if [ -n "$result_file" ]; then
      printf '%s\n' "$best" > "$result_file"
    else
      echo "$best"
    fi
    return 0
  fi

  local fine_low=0
  local fine_high="$(((first_fail - best) / fine_step_len))"
  while [ $((fine_high - fine_low)) -gt 1 ]; do
    local fine_index="$(((fine_low + fine_high) / 2))"
    local seq="$((best + fine_index * fine_step_len))"
    local exp_dir="$out_dir/${prefix}_L${seq}"
    echo "[$MODEL_KEY][$prefix] fine probing seq=$seq fine_index=$fine_index low=$fine_low high=$fine_high mode=$mode" >&2
    local ok=0
    if run_context_probe "${prefix}_L${seq}" "$mode" "$seq" "$train_iters" "$exp_dir"; then
      ok=1
      fine_low="$fine_index"
    else
      fine_high="$fine_index"
    fi
    record_context_attempt "fine" "$seq" "$ok" "$exp_dir"
  done

  local low="$((best + fine_low * fine_step_len))"
  if [ -n "$result_file" ]; then
    printf '%s\n' "$low" > "$result_file"
  else
    echo "$low"
  fi
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

fatal_patterns = (
    r"Traceback \(most recent call last\):",
    r"\bFAILED\b",
    r"\bRuntimeError:",
    r"\bValueError:",
    r"\bAssertionError:",
    r"OutOfMemory",
    r"out of memory",
    r"CUDA out of memory",
    r"NCCL error",
    r"DistBackendError",
    r"ncclUnhandledCudaError",
)
failed = any(re.search(pattern, text) for pattern in fatal_patterns)
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

start_metric_result_display() {
  local result_file="$1"
  METRIC_FRESHNESS_MARKER=""
  if [ "${METRICS_SUPPRESS_SINGLE_SUMMARY:-0}" = "1" ]; then
    return 0
  fi
  mkdir -p "$(dirname "$result_file")"
  METRIC_FRESHNESS_MARKER="$(mktemp "$(dirname "$result_file")/.metric_run.XXXXXX")"
}

finish_metric_result_display() {
  local command_rc="$1"
  local metric="$2"
  local result_file="$3"
  local freshness_marker="$4"
  trap - EXIT

  if [ "${METRICS_SUPPRESS_SINGLE_SUMMARY:-0}" = "1" ]; then
    exit "$command_rc"
  fi

  local summary_rc=0
  python3 "$REPO_ROOT/scripts/metrics/metrics_summary.py" \
    --model-key "$MODEL_KEY" \
    --model-name "$MODEL_NAME" \
    --only "$metric" \
    --freshness-marker "$freshness_marker" \
    "$result_file" || summary_rc=$?
  rm -f -- "$freshness_marker"

  if [ "$command_rc" -ne 0 ] || [ "$summary_rc" -ne 0 ]; then
    exit 1
  fi
  exit 0
}

run_all_metric_suite() {
  local model_script_dir="$1"
  local -a metric_names=(
    compression_ratio
    load_time_reduction
    accuracy_loss
    train_context_gain
  )
  local -a result_paths=()
  local metric

  mkdir -p "$OUT_ROOT"
  local freshness_marker
  freshness_marker="$(mktemp "$OUT_ROOT/.acceptance_run.XXXXXX")"
  for metric in "${metric_names[@]}"; do
    result_paths+=("$OUT_ROOT/$metric/result.json")
  done

  local suite_rc=0
  if ensure_memrift_weights "$MODEL_PATH" "$MEMRIFT_WEIGHT_DIR" "${MEMRIFT_PREPARE_LEVEL:-18}"; then
    for metric in "${metric_names[@]}"; do
      echo
      echo "======================================================================"
      echo "[acceptance] running $metric"
      echo "======================================================================"
      if METRICS_SUPPRESS_SINGLE_SUMMARY=1 bash "$model_script_dir/$metric.sh"; then
        echo "[acceptance] $metric script completed"
      else
        local metric_rc=$?
        echo "[acceptance] $metric script exited with code $metric_rc" >&2
        suite_rc=1
      fi
    done
  else
    local prepare_rc=$?
    echo "[acceptance] compressed-weight preparation failed with code $prepare_rc" >&2
    echo "[acceptance] metric runs skipped; the final table will show missing results" >&2
    suite_rc=1
  fi

  local summary_rc=0
  python3 "$REPO_ROOT/scripts/metrics/metrics_summary.py" \
    --model-key "$MODEL_KEY" \
    --model-name "$MODEL_NAME" \
    --summary-out "$OUT_ROOT/summary.json" \
    --freshness-marker "$freshness_marker" \
    "${result_paths[@]}" || summary_rc=$?
  rm -f -- "$freshness_marker"

  if [ "$summary_rc" -ne 0 ]; then
    suite_rc=1
  fi
  return "$suite_rc"
}
