#!/usr/bin/env bash
# 使用 FlagScale 框架跑 纯 LoRA vs MemRift+LoRA，LoRA 为 MLP-only，对比显存峰值
# 用法：先 ssh 172.24.178.248，conda activate myc-flagscale，在仓库根目录执行:
#   bash scripts/run_lora_vs_memrift_peak_flagscale.sh
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-myc-flagscale}"
if ! command -v torchrun >/dev/null 2>&1; then
  for _conda in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$_conda" ] && { . "$_conda"; conda activate "$CONDA_ENV_NAME"; break; }
  done
fi
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/flagscale:$REPO_ROOT/flagscale/train:${PYTHONPATH}"

MODEL="${TINYLLAMA_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$REPO_ROOT/memrift_weights/tinyllama_1b_level18}"
TRAIN_ITERS="${TRAIN_ITERS:-50}"
CONFIG_NAME="train_mlp_only_mock"
CONFIG_PATH="examples/memrift/conf"
EXP_DIR="$REPO_ROOT/outputs/memrift_example"
PID_FILE="$EXP_DIR/logs/pids/host_0_localhost.pid"

echo "=============================================="
echo "  FlagScale: 纯 LoRA vs MemRift+LoRA (MLP-only)"
echo "=============================================="
echo "  REPO_ROOT:    $REPO_ROOT"
echo "  MODEL:        $MODEL"
echo "  WEIGHT_DIR:   $WEIGHT_DIR"
echo "  TRAIN_ITERS:  $TRAIN_ITERS"
echo "  CONFIG:       $CONFIG_PATH / $CONFIG_NAME"
echo "=============================================="

# 准备压缩权重（仅 MemRift 跑需要，先统一准备好）
if [ ! -f "$WEIGHT_DIR/index.json" ]; then
  echo "[prepare] 生成 MemRift 压缩权重: $WEIGHT_DIR"
  mkdir -p "$WEIGHT_DIR"
  python -m flagscale.compress.memrift.offline_comp.prepare_weight \
    --model "$MODEL" \
    --outdir "$WEIGHT_DIR" \
    --level 18
else
  echo "[prepare] 使用已有压缩权重: $WEIGHT_DIR"
fi

_run_one() {
  local name="$1"
  local peak_log="$2"
  shift 2
  rm -f "$peak_log"
  rm -rf "$EXP_DIR/checkpoints" "$EXP_DIR/logs" 2>/dev/null || true
  mkdir -p "$(dirname "$peak_log")"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -l 2 > "$peak_log" &
  local nvpid=$!
  sleep 1
  echo ">> 启动: $name" >&2
  python run.py \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    action=run \
    train.model.tokenizer_path="$MODEL" \
    train.model.tokenizer_model="$MODEL" \
    train.trainer.train_iters="$TRAIN_ITERS" \
    train.trainer.lr_warmup_iters=10 \
    "$@"
  kill $nvpid 2>/dev/null || true
  if [ -f "$PID_FILE" ]; then
    local train_pid=$(cat "$PID_FILE")
    while kill -0 "$train_pid" 2>/dev/null; do sleep 5; done
  fi
  sleep 2
  # 只输出峰值数字，供 PEAK1=$(_run_one ...) 捕获
  if [ -s "$peak_log" ]; then
    awk -F',' 'NF>=2 { gsub(/ /,"",$2); v=$2+0; if (v>max) max=v } END { if (max=="") max=0; print max+0 }' "$peak_log"
  else
    echo "0"
  fi
}

PEAK_LORA="/tmp/peak_lora_$$.txt"
PEAK_MEMRIFT="/tmp/peak_memrift_$$.txt"
trap "rm -f $PEAK_LORA $PEAK_MEMRIFT" EXIT

# 1) 纯 LoRA（FlagScale 框架，MLP-only，不开启 MemRift）
echo ""
echo "[1/2] 纯 LoRA (MLP-only, memrift_enable=false)"
PEAK1=$(_run_one "纯 LoRA" "$PEAK_LORA" "train.system.memrift_enable=false")
echo "     峰值: ${PEAK1} MB"

# 2) MemRift+LoRA（MLP-only，开启 MemRift 权重压缩）
echo ""
echo "[2/2] MemRift+LoRA (MLP-only)"
PEAK2=$(_run_one "MemRift+LoRA" "$PEAK_MEMRIFT" \
  "train.system.memrift_enable=true" \
  "train.system.memrift_weight_enable=true" \
  "train.system.memrift_compressed_weight_dir=$WEIGHT_DIR")
echo "     峰值: ${PEAK2} MB"

# 3) 对比
echo ""
echo "=============================================="
echo "  显存峰值对比 (FlagScale, LoRA MLP-only)"
echo "=============================================="
printf "  纯 LoRA 峰值:       %s MB\n" "$PEAK1"
printf "  MemRift+LoRA 峰值:  %s MB\n" "$PEAK2"
SAVED=$(python3 -c "print(round(float('$PEAK1') - float('$PEAK2'), 2))" 2>/dev/null || echo "N/A")
printf "  节省:               %s MB\n" "$SAVED"
echo "=============================================="
