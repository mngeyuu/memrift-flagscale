#!/usr/bin/env bash
# 实验脚本：对比 MemRift + LoRA vs 纯 LoRA（仅 MLP LoRA）
# 使用 TinyLlama 1.1B，batch size=1，seq len=2048，5 steps

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/flagscale:${REPO_ROOT}/flagscale/train:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"

TRAIN_ITERS=5
SKIP_PREPARE=1
TINYLLAMA_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MEMRIFT_WEIGHT_DIR="${REPO_ROOT}/memrift_weights/tinyllama_1b_level18"
CONFIG_PATH="examples/memrift/conf"
CONFIG_NAME="train_mlp_only_mock"

OUT_ROOT="${REPO_ROOT}/output/tinyllama_1b_mlp_only_compare"
MEMRIFT_EXP="$OUT_ROOT/memrift_run"
LORA_EXP="$OUT_ROOT/pure_lora_run"
LOG_DIR="$OUT_ROOT/logs"
SUMMARY_DIR="$OUT_ROOT/summary"

mkdir -p "$LOG_DIR" "$SUMMARY_DIR" "$MEMRIFT_EXP" "$LORA_EXP"

WARMUP=1

echo "=============================================="
echo " TinyLlama-1.1B  MemRift+LoRA  vs  pure LoRA"
echo " 配置: 仅 MLP LoRA, bs=1, seq=2048, steps=5"
echo "=============================================="
echo " REPO_ROOT:          $REPO_ROOT"
echo " OUT_ROOT:           $OUT_ROOT"
echo " MODEL:              $TINYLLAMA_MODEL"
echo " MEMRIFT_WEIGHT_DIR: $MEMRIFT_WEIGHT_DIR"
echo " TRAIN_ITERS:        $TRAIN_ITERS"
echo "=============================================="

if [ "$SKIP_PREPARE" != "1" ] && [ ! -f "$MEMRIFT_WEIGHT_DIR/index.json" ]; then
  echo ">> Preparing MemRift compressed weights..."
  mkdir -p "$MEMRIFT_WEIGHT_DIR"
  python -m flagscale.compress.memrift.offline_comp.prepare_weight \
    --model "$TINYLLAMA_MODEL" \
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
    experiment.exp_name=tinyllama_1b_memrift_mlp_only \
    train.system.memrift_enable=true \
    train.system.memrift_weight_enable=true \
    train.system.memrift_activation_enable=true \
    train.system.memrift_weight_async=true \
    train.system.memrift_act_async=true \
    train.system.memrift_compressed_weight_dir="$MEMRIFT_WEIGHT_DIR" \
    train.model.tokenizer_path="$TINYLLAMA_MODEL" \
    train.model.tokenizer_model="$TINYLLAMA_MODEL" \
    train.model.seq_length=2048 \
    data.micro_batch_size=1 \
    data.global_batch_size=1 \
    train.trainer.train_iters="$TRAIN_ITERS" \
    train.trainer.lr_warmup_iters="$WARMUP" \
    train.system.logging.log_interval=1 \
    2>&1 | tee "$LOG_DIR/run_memrift_stdout.log"
}

run_pure_lora() {
  echo ""
  echo ">> [2/2] Pure LoRA (MemRift off) -> $LORA_EXP"
  rm -rf "$LORA_EXP" 2>/dev/null || true
  mkdir -p "$LORA_EXP"
  python run.py \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    action=run \
    experiment.exp_dir="$LORA_EXP" \
    experiment.exp_name=tinyllama_1b_pure_lora_mlp_only \
    train.system.memrift_enable=false \
    train.system.memrift_weight_enable=false \
    "+train.system.init_model_with_meta_device=false" \
    train.model.tokenizer_path="$TINYLLAMA_MODEL" \
    train.model.tokenizer_model="$TINYLLAMA_MODEL" \
    train.model.seq_length=2048 \
    data.micro_batch_size=1 \
    data.global_batch_size=1 \
    train.trainer.train_iters="$TRAIN_ITERS" \
    train.trainer.lr_warmup_iters="$WARMUP" \
    train.system.logging.log_interval=1 \
    2>&1 | tee "$LOG_DIR/run_pure_lora_stdout.log"
}

run_memrift
MLOG="$MEMRIFT_EXP/logs/host_0_localhost.output"
[ -f "$MLOG" ] || MLOG="$MEMRIFT_EXP/logs/host.output"
cp -f "$MLOG" "$LOG_DIR/memrift_host_0_localhost.output" 2>/dev/null || true

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
  --model "$TINYLLAMA_MODEL" \
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
