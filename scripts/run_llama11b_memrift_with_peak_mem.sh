#!/bin/bash
# 在 172.24.178.248 上：激活 myc-flagscale，跑 TinyLlama 1.1B 训练并记录显存峰值
# 用法: 在远程执行: bash scripts/run_llama11b_memrift_with_peak_mem.sh
# 或本地: ssh 172.24.178.248 'cd /share/project/mengyc/flagScale/FlagScale && bash scripts/run_llama11b_memrift_with_peak_mem.sh'

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 激活 conda
CONDA_ENV_NAME="myc-flagscale"
if ! command -v torchrun >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
  fi
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/flagscale/train:${PYTHONPATH}"

# 显存采样（每 2 秒）写临时文件
PEAK_LOG="/tmp/gpu_peak_$$.log"
PID_FILE="$REPO_ROOT/outputs/memrift_example/logs/pids/host_0_localhost.pid"
EXP_DIR="$REPO_ROOT/outputs/memrift_example"
HOST_LOG="$EXP_DIR/logs/host_0_localhost.output"
cleanup() { kill "$NVPID" 2>/dev/null; true; }
trap cleanup EXIT

# 避免旧 checkpoint/旧日志污染本次训练与峰值解析
rm -rf "$EXP_DIR/checkpoints" 2>/dev/null || true
rm -f "$HOST_LOG" "$PEAK_LOG" 2>/dev/null || true

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -l 2 > "$PEAK_LOG" &
NVPID=$!
sleep 1

echo "=== TinyLlama 1.1B MemRift 训练开始，显存采样中 (每 2s) ==="
python run.py --config-path=examples/memrift/conf --config-name=train_mock action=run \
  train.trainer.train_iters=50 \
  2>&1 | tee /tmp/memrift_train_main.log

# 训练脚本使用 nohup 后台执行，等待其结束再统计显存峰值
if [ -f "$PID_FILE" ]; then
  TRAIN_PID=$(cat "$PID_FILE")
  echo "等待训练进程 PID=$TRAIN_PID 结束..."
  while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 5; done
  echo "训练已结束。"
fi

echo ""
echo "===== 显存峰值 (nvidia-smi 每 2s 采样) ====="
if [ -s "$PEAK_LOG" ]; then
  awk -F',' 'NF>=2 { gsub(/ /,"",$2); v=$2+0; if (v>max) max=v } END { if (max=="") max=0; print "Peak GPU memory used: " max " MB" }' "$PEAK_LOG"
else
  echo "无 nvidia-smi 采样数据"
fi
echo "===== 训练日志中的显存报告 (report_memory) ====="
grep -E "max allocated|max reserved|memory \(MB\)" /tmp/memrift_train_main.log "$REPO_ROOT/outputs/memrift_example/logs/"*.output 2>/dev/null || echo "(未找到)"
