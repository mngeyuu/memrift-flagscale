#!/usr/bin/env bash
# Llama-3.1-8B: FlagScale MemRift+LoRA vs pure LoRA (mock data).
# All artifacts under ./output/llama31_8b_compare/ (logs, runs, summary JSON).
#
# Prerequisites: CUDA, conda env with FlagScale deps, float_split_stride_pin built,
#   HF token if needed for meta-llama/Llama-3.1-8B
#
# Usage (repo root):
#   bash scripts/run_llama31_8b_memrift_vs_lora_output.sh
# Use physical GPU 1 (process sees cuda:0):
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_llama31_8b_memrift_vs_lora_output.sh
# Env:
#   CUDA_VISIBLE_DEVICES (unset=all visible GPUs; "1" = only physical GPU 1)
#   TRAIN_ITERS (default 50), SKIP_PREPARE (1=skip prepare_weight),
#   SKIP_MEMRIFT (1=skip MemRift run; reuse memrift logs under OUT_ROOT and rerun pure LoRA + summary),
#   LLAMA31_MODEL (default meta-llama/Llama-3.1-8B),
#   MEMRIFT_WEIGHT_DIR (default $REPO/memrift_weights/llama31_8b_level18)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/flagscale:${REPO_ROOT}/flagscale/train:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"

TRAIN_ITERS="${TRAIN_ITERS:-50}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_MEMRIFT="${SKIP_MEMRIFT:-0}"
LLAMA31_MODEL="${LLAMA31_MODEL:-meta-llama/Llama-3.1-8B}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$REPO_ROOT/memrift_weights/llama31_8b_level18}"
CONFIG_PATH="examples/memrift/conf"
CONFIG_NAME="train_llama31_8b_mock"

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/output/llama31_8b_compare}"
MEMRIFT_EXP="$OUT_ROOT/memrift_run"
LORA_EXP="$OUT_ROOT/pure_lora_run"
LOG_DIR="$OUT_ROOT/logs"
SUMMARY_DIR="$OUT_ROOT/summary"

mkdir -p "$LOG_DIR" "$SUMMARY_DIR" "$MEMRIFT_EXP" "$LORA_EXP"

WARMUP=$((TRAIN_ITERS / 5))
if [ "$WARMUP" -lt 2 ]; then WARMUP=2; fi
if [ "$WARMUP" -ge "$TRAIN_ITERS" ]; then WARMUP=$((TRAIN_ITERS - 1)); fi

echo "=============================================="
echo " Llama-3.1-8B  MemRift+LoRA  vs  pure LoRA"
echo "=============================================="
echo " REPO_ROOT:          $REPO_ROOT"
echo " OUT_ROOT:           $OUT_ROOT"
echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset, all GPUs>}"
echo " MODEL:              $LLAMA31_MODEL"
echo " MEMRIFT_WEIGHT_DIR: $MEMRIFT_WEIGHT_DIR"
echo " TRAIN_ITERS:        $TRAIN_ITERS"
echo " SKIP_MEMRIFT:        $SKIP_MEMRIFT"
echo "=============================================="

if [ "$SKIP_PREPARE" != "1" ] && [ ! -f "$MEMRIFT_WEIGHT_DIR/index.json" ]; then
  echo ">> Preparing MemRift compressed weights..."
  mkdir -p "$MEMRIFT_WEIGHT_DIR"
  python -m flagscale.compress.memrift.offline_comp.prepare_weight \
    --model "$LLAMA31_MODEL" \
    --outdir "$MEMRIFT_WEIGHT_DIR" \
    --level 18
else
  echo ">> Using existing weights or SKIP_PREPARE=1"
fi

run_memrift() {
  echo ""
  echo ">> [1/2] MemRift + LoRA (mock) -> $MEMRIFT_EXP"
  rm -rf "$MEMRIFT_EXP" 2>/dev/null || true
  mkdir -p "$MEMRIFT_EXP"
  python run.py \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    action=run \
    experiment.exp_dir="$MEMRIFT_EXP" \
    experiment.exp_name=llama31_8b_memrift \
    train.system.memrift_enable=true \
    train.system.memrift_weight_enable=true \
    train.system.memrift_activation_enable=true \
    train.system.memrift_weight_async=true \
    train.system.memrift_act_async=true \
    train.system.memrift_compressed_weight_dir="$MEMRIFT_WEIGHT_DIR" \
    train.model.tokenizer_path="$LLAMA31_MODEL" \
    train.model.tokenizer_model="$LLAMA31_MODEL" \
    train.trainer.train_iters="$TRAIN_ITERS" \
    train.trainer.lr_warmup_iters="$WARMUP" \
    train.system.logging.log_interval=1 \
    2>&1 | tee "$LOG_DIR/run_memrift_stdout.log"
}

run_pure_lora() {
  echo ""
  echo ">> [2/2] Pure LoRA (MemRift off) -> $LORA_EXP"
  # ÕûÄ¿Â¼Çå¿Õ£¬±ÜÃâ²ÐÁô checkpoint / rerun ×´Ì¬µ¼ÖÂ load iter_0000030 µÈÓÄÁéÂ·¾¶
  rm -rf "$LORA_EXP" 2>/dev/null || true
  mkdir -p "$LORA_EXP"
  python run.py \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    action=run \
    experiment.exp_dir="$LORA_EXP" \
    experiment.exp_name=llama31_8b_pure_lora \
    train.system.memrift_enable=false \
    train.system.memrift_weight_enable=false \
    "+train.system.init_model_with_meta_device=false" \
    train.model.tokenizer_path="$LLAMA31_MODEL" \
    train.model.tokenizer_model="$LLAMA31_MODEL" \
    train.trainer.train_iters="$TRAIN_ITERS" \
    train.trainer.lr_warmup_iters="$WARMUP" \
    train.system.logging.log_interval=1 \
    2>&1 | tee "$LOG_DIR/run_pure_lora_stdout.log"
}

if [ "$SKIP_MEMRIFT" != "1" ]; then
  run_memrift
  MLOG="$MEMRIFT_EXP/logs/host_0_localhost.output"
  [ -f "$MLOG" ] || MLOG="$MEMRIFT_EXP/logs/host.output"
  cp -f "$MLOG" "$LOG_DIR/memrift_host_0_localhost.output" 2>/dev/null || true
else
  echo ""
  echo ">> SKIP_MEMRIFT=1: not rerunning MemRift; reusing host log from $MEMRIFT_EXP or $LOG_DIR"
  MLOG="$MEMRIFT_EXP/logs/host_0_localhost.output"
  [ -f "$MLOG" ] || MLOG="$MEMRIFT_EXP/logs/host.output"
  if [ -f "$MLOG" ]; then
    cp -f "$MLOG" "$LOG_DIR/memrift_host_0_localhost.output" 2>/dev/null || true
  elif [ -f "$LOG_DIR/memrift_host_0_localhost.output" ]; then
    echo "   (using existing $LOG_DIR/memrift_host_0_localhost.output)"
  else
    echo "   WARNING: no MemRift host log found; memrift_metrics.json may be empty"
  fi
fi

run_pure_lora
LLOG="$LORA_EXP/logs/host_0_localhost.output"
[ -f "$LLOG" ] || LLOG="$LORA_EXP/logs/host.output"
cp -f "$LLOG" "$LOG_DIR/pure_lora_host_0_localhost.output" 2>/dev/null || true

echo ""
echo ">> Parsing metrics..."
python3 "$REPO_ROOT/scripts/parse_flagscale_train_metrics.py" \
  "$LOG_DIR/memrift_host_0_localhost.output" \
  --json-out "$SUMMARY_DIR/memrift_metrics.json" || true
python3 "$REPO_ROOT/scripts/parse_flagscale_train_metrics.py" \
  "$LOG_DIR/pure_lora_host_0_localhost.output" \
  --json-out "$SUMMARY_DIR/pure_lora_metrics.json" || true

python3 "$REPO_ROOT/scripts/write_llama31_lora_comparison.py" \
  --out-root "$OUT_ROOT" \
  --train-iters "$TRAIN_ITERS" \
  --model "$LLAMA31_MODEL" \
  --memrift-weight-dir "$MEMRIFT_WEIGHT_DIR"

echo ""
echo "=============================================="
echo " Done. Artifacts:"
echo "   $OUT_ROOT/"
echo "   - memrift_run/   (Hydra + logs + checkpoints)"
echo "   - pure_lora_run/"
echo "   - logs/          (tee stdout + host log copies)"
echo "   - summary/       (memrift_metrics.json, pure_lora_metrics.json, comparison.json)"
echo "=============================================="
