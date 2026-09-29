#!/usr/bin/env bash
# Evaluate one deterministic GSM8K shard from an already-trained adapter checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL_SELECTOR="${1:?usage: $0 MODEL VARIANT RUN_ROOT SHARD_INDEX SHARD_COUNT}"
VARIANT="${2:?missing variant}"
RUN_ROOT="${3:?missing run root}"
SHARD_INDEX="${4:?missing shard index}"
SHARD_COUNT="${5:?missing shard count}"

case "$MODEL_SELECTOR" in
  llama8b|aquila) ;;
  *) echo "unsupported model: $MODEL_SELECTOR" >&2; exit 2 ;;
esac
case "$VARIANT" in
  pure_lora)
    MODE="lora_adapter_only"
    KEEP_RESIDENT=0
    ;;
  memrift_lora)
    MODE="memrift_async"
    KEEP_RESIDENT=1
    ;;
  *) echo "unsupported variant: $VARIANT" >&2; exit 2 ;;
esac

activate_env
setup_common_env
source "$SCRIPT_DIR/$MODEL_SELECTOR/model_env.sh"

CHECKPOINT_DIR="$RUN_ROOT/checkpoints/$VARIANT"
EVAL_DIR="$RUN_ROOT/eval/${VARIANT}_shard_${SHARD_INDEX}_of_${SHARD_COUNT}"
RESULT="$RUN_ROOT/results/${VARIANT}_gsm8k_shard_${SHARD_INDEX}_of_${SHARD_COUNT}.json"
test -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"

BENCHMARK_SHARD_INDEX="$SHARD_INDEX" BENCHMARK_SHARD_COUNT="$SHARD_COUNT" \
GSM8K_LIMIT="" HELLASWAG_LIMIT=0 GSM8K_MAX_NEW_TOKENS="${GSM8K_MAX_NEW_TOKENS:-64}" \
MEMRIFT_KEEP_WEIGHTS_RESIDENT="$KEEP_RESIDENT" MEMRIFT_ACTIVATION_ENABLE=false \
  run_benchmark_accuracy "$VARIANT" "$EVAL_DIR" "$MODE" "$CHECKPOINT_DIR" "$RESULT"

test -s "$RESULT"
echo "SHARD_RESULT=$RESULT"
