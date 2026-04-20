#!/bin/bash
# 在 172.24.178.248 上运行：先激活 myc-flagscale，再跑通 MemRift TinyLlama 1.1B 训练
# 用法（在本地执行）:
#   ssh 172.24.178.248 'bash -s' < ./scripts/run_memrift_tinyllama_1b_remote.sh
# 或登录后执行:
#   ssh 172.24.178.248
#   source ~/miniconda3/etc/profile.d/conda.sh  # 或 anaconda3
#   conda activate myc-flagscale
#   cd /share/project/mengyc/flagScale/FlagScale && ./scripts/run_memrift_tinyllama_1b_remote.sh --run
#
# 若只做环境激活并进入 repo，不加 --run 即可。

set -e

FLAGSCALE_ROOT="${FLAGSCALE_ROOT:-/share/project/mengyc/flagScale/FlagScale}"
TINYLLAMA_MODEL="${TINYLLAMA_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$FLAGSCALE_ROOT/memrift_weights/tinyllama_1b_level18}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"

# 激活 conda（若当前未激活）
if ! command -v torchrun >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate myc-flagscale
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate myc-flagscale
  else
    echo "未找到 conda，请先安装或手动: conda activate myc-flagscale"
    exit 1
  fi
fi

cd "$FLAGSCALE_ROOT"
echo "FLAGSCALE_ROOT: $FLAGSCALE_ROOT"
echo "MEMRIFT_WEIGHT_DIR: $MEMRIFT_WEIGHT_DIR"
echo "TRAIN_ITERS: $TRAIN_ITERS"

# 若无压缩权重则先准备
if [ ! -f "$MEMRIFT_WEIGHT_DIR/index.json" ]; then
  echo ">> 准备 MemRift 压缩权重..."
  python -m flagscale.compress.memrift.offline_comp.prepare_weight \
    --model "$TINYLLAMA_MODEL" \
    --outdir "$MEMRIFT_WEIGHT_DIR" \
    --level 18
fi

# 仅当传入 --run 时执行训练
RUN_TRAIN=false
for arg in "$@"; do
  [ "$arg" = "--run" ] && RUN_TRAIN=true && break
done

if [ "$RUN_TRAIN" = true ]; then
  echo ">> 启动 MemRift TinyLlama 1.1B 训练 (${TRAIN_ITERS} iters)..."
  PID_FILE="$FLAGSCALE_ROOT/outputs/memrift_example/logs/pids/host_0_localhost.pid"
  TRAIN_ITERS="$TRAIN_ITERS" TINYLLAMA_MODEL="$TINYLLAMA_MODEL" MEMRIFT_WEIGHT_DIR="$MEMRIFT_WEIGHT_DIR" \
  python run.py \
    --config-path=examples/memrift/conf \
    --config-name=train_mock \
    action=run \
    train.system.memrift_compressed_weight_dir="$MEMRIFT_WEIGHT_DIR" \
    train.model.tokenizer_path="$TINYLLAMA_MODEL" \
    train.model.tokenizer_model="$TINYLLAMA_MODEL" \
    train.trainer.train_iters="$TRAIN_ITERS" \
    train.trainer.lr_warmup_iters=2

  # run.py 会生成脚本并用 nohup 后台启动训练，此处等待训练进程真正结束再退出
  echo ">> 等待训练进程启动..."
  for i in 1 2 3 4 5 6 7 8 9 10; do
    [ -f "$PID_FILE" ] && break
    sleep 2
  done
  if [ ! -f "$PID_FILE" ]; then
    echo ">> 未找到 pid 文件: $PID_FILE，训练可能未正确启动"
    exit 1
  fi
  TRAIN_PID=$(cat "$PID_FILE")
  echo ">> 训练已在后台启动 (pid=$TRAIN_PID)，等待其结束..."
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 15
  done
  echo ">> 训练结束"
else
  echo "仅完成环境检查与路径设置。要真正跑训练请加 --run，例如: $0 --run"
fi
