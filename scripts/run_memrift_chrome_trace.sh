#!/usr/bin/env bash
# MemRift 训练 → PyTorch Profiler → Chrome Trace JSON（与 flagscale runner 一致的 PYTHONPATH）
#
# 用法（在 FlagScale 仓库根目录）:
#   ./scripts/run_memrift_chrome_trace.sh
#   CHROME_DIR=/tmp/memrift_chrome ./scripts/run_memrift_chrome_trace.sh
#
# 依赖:
#   - Python 环境已安装 megatron-core（提供 megatron.core）及训练依赖
#   - 若 import torch 因 flagcx 等后端失败: 已默认 export TORCH_DEVICE_BACKEND_AUTOLOAD=0
#
# 查看: https://ui.perfetto.dev/ 或 Chrome chrome://tracing → Load chrome_trace_*.json

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/flagscale/train${PYTHONPATH:+:${PYTHONPATH}}"

exec python scripts/run_memrift_chrome_trace.py "$@"
