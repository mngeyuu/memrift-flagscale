#!/bin/bash
# 在 172.24.178.248 上结束卡住的 Nsight profile / torchrun / train_gpt 进程
# 用法：在已 SSH 到 172.24.178.248 的终端里执行: bash scripts/kill_nsight_profile.sh

set -e
echo "正在查找 nsys / torchrun / train_gpt 相关进程..."
pids=$(pgrep -f "nsys profile.*memrift_llama11b_nsight" 2>/dev/null || true)
pids="$pids $(pgrep -f "torchrun.*train_gpt.py.*memrift" 2>/dev/null || true)"
pids="$pids $(pgrep -f "train_gpt.py.*memrift-compressed-weight-dir" 2>/dev/null || true)"
pids=$(echo $pids | tr ' ' '\n' | sort -u)
if [ -z "$pids" ]; then
  echo "未找到相关进程。"
  exit 0
fi
echo "将结束以下 PID: $pids"
for p in $pids; do
  [ -n "$p" ] && kill -9 "$p" 2>/dev/null || true
done
echo "已发送 SIGKILL。可用 nvidia-smi 确认显存释放。"
