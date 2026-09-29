#!/usr/bin/env bash
# Fine-tune Pure LoRA and MemRift+LoRA from the same pretrained model, then
# compare their GSM8K 8-shot CoT exact-match accuracy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL_SELECTOR="${1:-}"
case "$MODEL_SELECTOR" in
  llama8b|aquila) ;;
  *)
    echo "usage: $0 {llama8b|aquila}" >&2
    exit 2
    ;;
esac

activate_env
setup_common_env
source "$SCRIPT_DIR/$MODEL_SELECTOR/model_env.sh"

FINETUNE_ITERS="${FINETUNE_ITERS:-100}"
FINETUNE_SEQ_LEN="${FINETUNE_SEQ_LEN:-4096}"
GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-64}"
GSM8K_LIMIT="${GSM8K_LIMIT:-}"
MEMRIFT_ACTIVATION_ENABLE="${MEMRIFT_ACTIVATION_ENABLE:-false}"
SMOKE_MODE="${SMOKE_MODE:-0}"
EXPERIMENT_ID="${EXPERIMENT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${GSM8K_RUN_ROOT:-$REPO_ROOT/output/gsm8k_100iter/$MODEL_KEY/$EXPERIMENT_ID}"

if ! [[ "$FINETUNE_ITERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "FINETUNE_ITERS must be a positive integer: $FINETUNE_ITERS" >&2
  exit 2
fi
if ! [[ "$FINETUNE_SEQ_LEN" =~ ^[1-9][0-9]*$ ]]; then
  echo "FINETUNE_SEQ_LEN must be a positive integer: $FINETUNE_SEQ_LEN" >&2
  exit 2
fi
if [ -n "$GSM8K_LIMIT" ] && ! [[ "$GSM8K_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "GSM8K_LIMIT must be empty or a positive integer: $GSM8K_LIMIT" >&2
  exit 2
fi
if [ -e "$RUN_ROOT" ]; then
  echo "run directory already exists; choose another EXPERIMENT_ID or GSM8K_RUN_ROOT: $RUN_ROOT" >&2
  exit 2
fi

PURE_TRAIN_DIR="$RUN_ROOT/train/pure_lora"
MEM_TRAIN_DIR="$RUN_ROOT/train/memrift_lora"
PURE_CKPT_DIR="$RUN_ROOT/checkpoints/pure_lora"
MEM_CKPT_DIR="$RUN_ROOT/checkpoints/memrift_lora"
PURE_EVAL_DIR="$RUN_ROOT/eval/pure_lora"
MEM_EVAL_DIR="$RUN_ROOT/eval/memrift_lora"
PURE_RESULT="$RUN_ROOT/results/pure_lora_gsm8k.json"
MEM_RESULT="$RUN_ROOT/results/memrift_lora_gsm8k.json"
COMPARISON_RESULT="$RUN_ROOT/results/comparison.json"
PURE_CURVE_JSON="$RUN_ROOT/results/pure_lora_training_curve.json"
PURE_CURVE_CSV="$RUN_ROOT/results/pure_lora_training_curve.csv"
MEM_CURVE_JSON="$RUN_ROOT/results/memrift_lora_training_curve.json"
MEM_CURVE_CSV="$RUN_ROOT/results/memrift_lora_training_curve.csv"
MANIFEST="$RUN_ROOT/experiment_manifest.json"

mkdir -p "$RUN_ROOT/results"

ensure_memrift_weights "$MODEL_PATH" "$MEMRIFT_WEIGHT_DIR" "${MEMRIFT_PREPARE_LEVEL:-18}"
if [ ! -f "$MEGATRON_CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
  echo "missing pretrained Megatron checkpoint: $MEGATRON_CKPT_DIR" >&2
  echo "run scripts/metrics/$MODEL_KEY/convert_hf_to_mcore_tp1.sh first" >&2
  exit 2
fi
if [ ! -s /share/project/mengyc/data/test.jsonl ]; then
  echo "missing GSM8K test set: /share/project/mengyc/data/test.jsonl" >&2
  exit 2
fi

python3 - "$MANIFEST" "$MODEL_KEY" "$MODEL_NAME" "$FINETUNE_ITERS" "$FINETUNE_SEQ_LEN" "$GSM8K_LIMIT" "$GSM8K_MAX_NEW_TOKENS" "$ALPACA_DATA_PATH" "$MEGATRON_CKPT_DIR" "$MEMRIFT_WEIGHT_DIR" "$MEMRIFT_ACTIVATION_ENABLE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
data = {
    "model_key": sys.argv[2],
    "model_name": sys.argv[3],
    "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "finetune_iterations": int(sys.argv[4]),
    "finetune_sequence_length": int(sys.argv[5]),
    "training_data": sys.argv[8],
    "pretrained_megatron_checkpoint": sys.argv[9],
    "memrift_compressed_weights": sys.argv[10],
    "variants": ["pure_lora", "memrift_weight_compression_lora"],
    "memrift_activation_compression_enabled": sys.argv[11].lower() == "true",
    "optimizer": {
        "learning_rate": 2e-4,
        "minimum_learning_rate": 2e-5,
        "warmup_iterations": 0,
        "decay_style": "cosine",
        "weight_decay": 0.1,
        "global_batch_size": 1,
        "micro_batch_size": 1,
    },
    "evaluation": {
        "task": "gsm8k_cot",
        "dataset": "/share/project/mengyc/data/test.jsonl",
        "num_fewshot": 8,
        "decoding": "greedy",
        "metric": "exact_match",
        "sample_limit": int(sys.argv[6]) if sys.argv[6] else None,
        "max_new_tokens": int(sys.argv[7]),
    },
}
out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

echo "======================================================================"
echo "[$MODEL_KEY] stage 1/4: Pure LoRA fine-tuning, iterations=$FINETUNE_ITERS"
echo "======================================================================"
DISABLE_TRAIN_CHECKPOINT=true MEMRIFT_DISABLE_FINAL_CHECKPOINT=1 \
  MEMRIFT_ADAPTER_SAVE_DIR="$PURE_CKPT_DIR" \
  run_yaml_train gsm8k100_pure_lora "$PURE_TRAIN_DIR" lora \
    "$FINETUNE_SEQ_LEN" "$FINETUNE_ITERS" "$MAX_POSITION_EMBEDDINGS"
test -f "$PURE_CKPT_DIR/latest_checkpointed_iteration.txt"
python3 "$REPO_ROOT/scripts/metrics/extract_training_curve.py" \
  --log "$PURE_TRAIN_DIR/stdout.log" --json "$PURE_CURVE_JSON" --csv "$PURE_CURVE_CSV" \
  --expected-iterations "$FINETUNE_ITERS"

echo "======================================================================"
echo "[$MODEL_KEY] stage 2/4: MemRift+LoRA fine-tuning, iterations=$FINETUNE_ITERS"
echo "[$MODEL_KEY] MemRift weight compression=true, activation compression=$MEMRIFT_ACTIVATION_ENABLE"
echo "======================================================================"
DISABLE_TRAIN_CHECKPOINT=true MEMRIFT_DISABLE_FINAL_CHECKPOINT=1 \
  MEMRIFT_ADAPTER_SAVE_DIR="$MEM_CKPT_DIR" \
  run_yaml_train gsm8k100_memrift_lora "$MEM_TRAIN_DIR" memrift_async \
    "$FINETUNE_SEQ_LEN" "$FINETUNE_ITERS" "$MAX_POSITION_EMBEDDINGS"
test -f "$MEM_CKPT_DIR/latest_checkpointed_iteration.txt"
python3 "$REPO_ROOT/scripts/metrics/extract_training_curve.py" \
  --log "$MEM_TRAIN_DIR/stdout.log" --json "$MEM_CURVE_JSON" --csv "$MEM_CURVE_CSV" \
  --expected-iterations "$FINETUNE_ITERS"

echo "======================================================================"
echo "[$MODEL_KEY] stage 3/4: Pure LoRA GSM8K evaluation"
echo "======================================================================"
GSM8K_LIMIT="$GSM8K_LIMIT" HELLASWAG_LIMIT=0 \
  GSM8K_MAX_NEW_TOKENS="$GSM8K_MAX_NEW_TOKENS" \
  run_benchmark_accuracy pure_lora "$PURE_EVAL_DIR" lora_adapter_only "$PURE_CKPT_DIR" "$PURE_RESULT"
test -s "$PURE_RESULT"

echo "======================================================================"
echo "[$MODEL_KEY] stage 4/4: MemRift+LoRA GSM8K evaluation"
echo "======================================================================"
# Accuracy evaluation materializes the lossless compressed base weights once.
# Dynamic per-token disk decompression would change runtime, not predictions,
# and would make full GSM8K evaluation unnecessarily slow.
GSM8K_LIMIT="$GSM8K_LIMIT" HELLASWAG_LIMIT=0 \
  GSM8K_MAX_NEW_TOKENS="$GSM8K_MAX_NEW_TOKENS" \
  MEMRIFT_KEEP_WEIGHTS_RESIDENT=1 \
  run_benchmark_accuracy memrift_lora "$MEM_EVAL_DIR" memrift_async "$MEM_CKPT_DIR" "$MEM_RESULT"
test -s "$MEM_RESULT"

if [ "$SMOKE_MODE" = "1" ]; then
  python3 - "$PURE_RESULT" "$MEM_RESULT" <<'PY'
import json
import sys
from pathlib import Path
for path in map(Path, sys.argv[1:]):
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data["tasks"]["gsm8k_cot"]
    print(f"{data['variant']}: {result['num_correct']}/{result['num_samples']} = {result['accuracy']:.6f}")
print("SMOKE PASS: both trained checkpoints were loaded and produced GSM8K results")
PY
else
  python3 "$REPO_ROOT/scripts/metrics/benchmark_accuracy_result.py" \
    --baseline "$PURE_RESULT" \
    --candidate "$MEM_RESULT" \
    --output "$COMPARISON_RESULT" \
    --model-key "$MODEL_KEY" \
    --model-name "$MODEL_NAME" \
    --tasks gsm8k_cot \
    --threshold "${ACCURACY_THRESHOLD:-0.01}"

  python3 - "$COMPARISON_RESULT" <<'PY'
import json
import sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
r = d["tasks"]["gsm8k_cot"]
print("=" * 100)
print("100-step LoRA fine-tuning 后 GSM8K 对比结果")
print(f"测试模型：{d['model_name']} ({d['model_key']})")
print(f"Pure LoRA accuracy：{r['pure_lora_accuracy'] * 100:.4f}%")
print(f"MemRift+LoRA accuracy：{r['memrift_accuracy'] * 100:.4f}%")
print(f"绝对差值：{r['absolute_drop_percentage_points']:.4f} percentage points")
print(f"相对精度损耗：{r['relative_accuracy_drop_percent']:.4f}%")
print(f"最终判定：{'PASS' if r['pass'] else 'FAIL'}")
print("=" * 100)
PY
fi

printf '%s\n' "$RUN_ROOT" > "$REPO_ROOT/output/gsm8k_100iter/$MODEL_KEY/latest_run.txt"
echo "RUN_ROOT=$RUN_ROOT"
echo "MANIFEST=$MANIFEST"
echo "PURE_RESULT=$PURE_RESULT"
echo "MEMRIFT_RESULT=$MEM_RESULT"
if [ "$SMOKE_MODE" != "1" ]; then
  echo "COMPARISON_RESULT=$COMPARISON_RESULT"
fi
