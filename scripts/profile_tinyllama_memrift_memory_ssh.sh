#!/bin/bash
# ============================================
# 通过 SSH 在 172.24.178.248 上：先 ssh、再激活 myc-flagscale 环境后跑
# TinyLlama MemRift 训练（带内存 profiling）
# ============================================
# 用法（在本地执行）:
#   ./scripts/profile_tinyllama_memrift_memory_ssh.sh
#
# 远端：conda 环境 myc-flagscale，执行 profile_tinyllama_memrift_memory.sh
#
# 可选环境变量:
#   SSH_HOST        - 目标主机，默认 172.24.178.248
#   REMOTE_FLAGSCALE_DIR - 远端 FlagScale 仓库根目录，默认 /share/project/mengyc/flagScale/FlagScale
#   TINYLLAMA_MODEL, MEMRIFT_WEIGHT_DIR, TRAIN_ITERS, SKIP_PREPARE - 同 profile_tinyllama_memrift_memory.sh
#   ENABLE_PYTORCH_PROFILER, PROFILE_STEP_START, PROFILE_STEP_END - 同 profile 脚本
#
set -e

SSH_HOST="${SSH_HOST:-172.24.178.248}"
REMOTE_DIR="${REMOTE_FLAGSCALE_DIR:-/share/project/mengyc/flagScale/FlagScale}"
CONDA_ENV="${CONDA_ENV:-myc-flagscale}"

TINYLLAMA_MODEL="${TINYLLAMA_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$REMOTE_DIR/memrift_weights/tinyllama_1b_level18}"
TRAIN_ITERS="${TRAIN_ITERS:-50}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
ENABLE_PYTORCH_PROFILER="${ENABLE_PYTORCH_PROFILER:-0}"
PROFILE_STEP_START="${PROFILE_STEP_START:-5}"
PROFILE_STEP_END="${PROFILE_STEP_END:-10}"

echo "============================================"
echo "SSH 至 $SSH_HOST，激活 $CONDA_ENV 后执行 TinyLlama MemRift（内存 Profiling）"
echo "============================================"
echo "REMOTE_DIR:          $REMOTE_DIR"
echo "MEMRIFT_WEIGHT_DIR:  $MEMRIFT_WEIGHT_DIR"
echo "TRAIN_ITERS:         $TRAIN_ITERS"
echo "SKIP_PREPARE:        $SKIP_PREPARE"
echo "============================================"

# 远端：source conda → activate myc-flagscale → cd → 执行 profile 脚本
ssh "$SSH_HOST" "bash -l -c \"
  set -e
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
  conda activate $CONDA_ENV
  cd '$REMOTE_DIR'
  export TINYLLAMA_MODEL='$TINYLLAMA_MODEL'
  export MEMRIFT_WEIGHT_DIR='$MEMRIFT_WEIGHT_DIR'
  export TRAIN_ITERS='$TRAIN_ITERS'
  export SKIP_PREPARE='$SKIP_PREPARE'
  export ENABLE_PYTORCH_PROFILER='$ENABLE_PYTORCH_PROFILER'
  export PROFILE_STEP_START='$PROFILE_STEP_START'
  export PROFILE_STEP_END='$PROFILE_STEP_END'
  ./scripts/profile_tinyllama_memrift_memory.sh
\""

echo ">> 远端 profile 训练脚本已结束"
