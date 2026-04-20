#!/bin/bash
# 使用 FlagScale 重构的 MemRift 代码（不依赖 FlagScale 训练框架）跑 TinyLlama 1.1B 训练并记录显存峰值
#
# 用法:
#   ./scripts/run_memrift_standalone_llama1.1b.sh
#   COMPRESSED_WEIGHTS=/path/to/weights ./scripts/run_memrift_standalone_llama1.1b.sh
#
# 环境变量:
#   MODEL                  HF 模型 id，默认 TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   COMPRESSED_WEIGHTS     已准备好的压缩权重目录，若未设则先执行 prepare_weight
#   STEPS                  训练步数，默认 5
#   MAX_LENGTH             序列长度，默认 512
#   ACTIVATION             设为 1 开启激活压缩
#   OUTPUT_PEAK            写入峰值 JSON 的文件路径

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGSCALE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$FLAGSCALE_ROOT"

MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
STEPS="${STEPS:-5}"
MAX_LENGTH="${MAX_LENGTH:-512}"
WEIGHTS_DIR="${COMPRESSED_WEIGHTS:-$FLAGSCALE_ROOT/memrift_weights/tinyllama_1b_level18}"
OUTPUT_PEAK="${OUTPUT_PEAK:-$FLAGSCALE_ROOT/scripts/standalone_peak_memory.json}"
ACTIVATION="${ACTIVATION:-0}"

echo "=============================================="
echo "MemRift 独立训练 (TinyLlama 1.1B) - 显存峰值记录"
echo "=============================================="
echo "  MODEL:              $MODEL"
echo "  COMPRESSED_WEIGHTS: $WEIGHTS_DIR"
echo "  STEPS:              $STEPS"
echo "  MAX_LENGTH:         $MAX_LENGTH"
echo "  ACTIVATION:         $ACTIVATION"
echo "  OUTPUT_PEAK:        $OUTPUT_PEAK"
echo "=============================================="

if [ ! -f "$WEIGHTS_DIR/index.json" ]; then
  echo ">> 未检测到压缩权重，先执行 prepare_weight..."
  mkdir -p "$(dirname "$WEIGHTS_DIR")"
  python -m flagscale.compress.memrift.offline_comp.prepare_weight \
    --model "$MODEL" \
    --outdir "$WEIGHTS_DIR" \
    --level 18
  echo ">> 压缩权重已写入 $WEIGHTS_DIR"
fi

ACT_FLAG=""
[ "$ACTIVATION" = "1" ] && ACT_FLAG="--activation"

python scripts/train_memrift_standalone_llama1.1b.py \
  --model "$MODEL" \
  --compressed_weights "$WEIGHTS_DIR" \
  --steps "$STEPS" \
  --max_length "$MAX_LENGTH" \
  --profile_memory \
  --output_peak "$OUTPUT_PEAK" \
  $ACT_FLAG

echo ""
echo ">> 显存峰值已记录到: $OUTPUT_PEAK"
