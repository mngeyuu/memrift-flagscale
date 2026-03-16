#!/bin/bash
# ============================================
# MemRift 冒烟测试：从 run.py 跑 MemRift 训练若干 iter，验证集成是否完整
# ============================================
#
# 用法（在仓库根目录 FlagScale 下执行）:
#   # 1. 准备压缩权重、数据、tokenizer 后，通过环境变量传入路径并跑 5 个 iter
#   export MEMRIFT_WEIGHT_DIR=/path/to/memrift_weights/tinyllama_1b_level18
#   export DATA_PATH=/path/to/your/data
#   export TOKENIZER_PATH=/path/to/TinyLlama-1.1B-Chat-v1.0
#   ./scripts/run_memrift_smoke.sh
#
#   # 2. 仅打印将要执行的命令（dryrun）
#   ./scripts/run_memrift_smoke.sh dryrun
#
#   # 3. 自定义 iter 数（默认 5）
#   TRAIN_ITERS=10 ./scripts/run_memrift_smoke.sh
#
# 前置条件:
#   - 已用 prepare_weight 生成压缩权重目录（含 index.json 与 *.bin）
#   - data_path、tokenizer_path 可访问
#   - 已安装 float_split_stride_pin 扩展与依赖

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-}"
DATA_PATH="${DATA_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
DRYRUN="${1:-}"
ACTION="run"
[ "$DRYRUN" = "dryrun" ] && ACTION="dryrun"

CONFIG_PATH="examples/memrift/conf"
CONFIG_NAME="train"

# 构建 Hydra 覆盖：路径可由环境变量覆盖
OVERRIDES=()
OVERRIDES+=( "train.trainer.train_iters=${TRAIN_ITERS}" )
[ -n "$MEMRIFT_WEIGHT_DIR" ] && OVERRIDES+=( "train.system.memrift_compressed_weight_dir=${MEMRIFT_WEIGHT_DIR}" )
[ -n "$DATA_PATH" ]          && OVERRIDES+=( "train.data.data_path=${DATA_PATH}" )
[ -n "$TOKENIZER_PATH" ]     && OVERRIDES+=( "train.model.tokenizer_path=${TOKENIZER_PATH}" )

echo "============================================"
echo "MemRift 冒烟测试 (from run.py)"
echo "============================================"
echo "ROOT_DIR: $ROOT_DIR"
echo "CONFIG:   --config-path=${CONFIG_PATH} --config-name=${CONFIG_NAME}"
echo "TRAIN_ITERS: ${TRAIN_ITERS}"
echo "MEMRIFT_WEIGHT_DIR: ${MEMRIFT_WEIGHT_DIR:-'(未设置，使用 yaml 默认)'}"
echo "DATA_PATH:          ${DATA_PATH:-'(未设置，使用 yaml 默认)'}"
echo "TOKENIZER_PATH:     ${TOKENIZER_PATH:-'(未设置，使用 yaml 默认)'}"
echo "============================================"

# 若未设置路径且为正式 run，提示并可选退出
if [ "$ACTION" = "run" ]; then
  if [ -z "$MEMRIFT_WEIGHT_DIR" ] || [ -z "$DATA_PATH" ] || [ -z "$TOKENIZER_PATH" ]; then
    echo "提示: 未设置 MEMRIFT_WEIGHT_DIR / DATA_PATH / TOKENIZER_PATH 时，将使用 yaml 中的占位路径，可能报错。"
    echo "建议: export MEMRIFT_WEIGHT_DIR=... DATA_PATH=... TOKENIZER_PATH=... 后再执行。"
    read -p "是否继续? [y/N] " -n 1 -r; echo
    if [[ ! $REPLY =~ ^[yY]$ ]]; then
      exit 1
    fi
  fi
fi

python run.py \
  --config-path="$CONFIG_PATH" \
  --config-name="$CONFIG_NAME" \
  action="$ACTION" \
  "${OVERRIDES[@]}"
