#!/usr/bin/env bash
# 在远程机 172.24.178.248 上运行（先 ssh 登录并 conda activate myc-flagscale）
# 对比 FlagScale MemRift Llama 1.1B 与 纯 LoRA 训练的显存峰值
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
WEIGHT_DIR="${WEIGHT_DIR:-$REPO_ROOT/memrift_weights/tinyllama_1b_level18}"
STEPS="${STEPS:-5}"
MAX_LEN="${MAX_LEN:-512}"

echo "=============================================="
echo "  LoRA vs MemRift 显存峰值对比 (Llama 1.1B)"
echo "=============================================="
echo "  REPO_ROOT: $REPO_ROOT"
echo "  MODEL: $MODEL"
echo "  COMPRESSED_WEIGHTS: $WEIGHT_DIR"
echo "  STEPS: $STEPS  MAX_LENGTH: $MAX_LEN"
echo "=============================================="

# 1. 若无压缩权重则先 prepare
if [[ ! -f "$WEIGHT_DIR/index.json" ]]; then
  echo "[1/3] 准备压缩权重: $WEIGHT_DIR"
  mkdir -p "$WEIGHT_DIR"
  python -m flagscale.compress.memrift.offline_comp.prepare_weight \
    --model "$MODEL" \
    --outdir "$WEIGHT_DIR" \
    --level 18
else
  echo "[1/3] 压缩权重已存在: $WEIGHT_DIR"
fi

# 2. 纯 LoRA 训练并记录峰值
echo ""
echo "[2/3] 运行 纯 LoRA 训练..."
python scripts/train_memrift_standalone_llama1.1b.py \
  --model "$MODEL" \
  --lora_only \
  --steps "$STEPS" \
  --max_length "$MAX_LEN" \
  --profile_memory \
  --output_peak /tmp/lora_peak.json

# 3. MemRift+LoRA 训练并记录峰值
echo ""
echo "[3/3] 运行 MemRift+LoRA 训练..."
python scripts/train_memrift_standalone_llama1.1b.py \
  --model "$MODEL" \
  --compressed_weights "$WEIGHT_DIR" \
  --steps "$STEPS" \
  --max_length "$MAX_LEN" \
  --profile_memory \
  --output_peak /tmp/memrift_peak.json

# 4. 对比结果
echo ""
echo "=============================================="
echo "  显存峰值对比结果"
echo "=============================================="
LORA_MB=$(python -c "import json; print(json.load(open('/tmp/lora_peak.json'))['peak_memory_mb'])")
MR_MB=$(python -c "import json; print(json.load(open('/tmp/memrift_peak.json'))['peak_memory_mb'])")
SAVED_MB=$(python -c "print(round($LORA_MB - $MR_MB, 2))")
echo "  纯 LoRA 峰值:     ${LORA_MB} MB"
echo "  MemRift+LoRA 峰值: ${MR_MB} MB"
echo "  节省:             ${SAVED_MB} MB"
echo "=============================================="
