#!/bin/bash
# ============================================
# 通过 SSH 在 172.24.178.248 上：先 ssh、再激活 conda 环境后跑 TinyLlama MemRift 训练
# ============================================
# 用法（在本地执行）:
#   ./scripts/run_tinyllama_memrift_train_ssh.sh
#
# 远端环境：conda 环境名为 myc-flagscale（需先 source conda 再 activate）
#
# 可选环境变量:
#   SSH_HOST        - 目标主机，默认 172.24.178.248
#   REMOTE_FLAGSCALE_DIR - 远端 FlagScale 仓库根目录，默认 /share/project/mengyc/flagScale/FlagScale
#   TINYLLAMA_MODEL, MEMRIFT_WEIGHT_DIR, TRAIN_ITERS, SKIP_PREPARE - 同 run_tinyllama_memrift_train.sh
#
set -e

SSH_HOST="${SSH_HOST:-172.24.178.248}"
REMOTE_DIR="${REMOTE_FLAGSCALE_DIR:-/share/project/mengyc/flagScale/FlagScale}"
CONDA_ENV="${CONDA_ENV:-myc-flagscale}"

# 其余变量传给远端（可选）
TINYLLAMA_MODEL="${TINYLLAMA_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$REMOTE_DIR/memrift_weights/tinyllama_1b_level18}"
TRAIN_ITERS="${TRAIN_ITERS:-50}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

echo "============================================"
echo "SSH 至 $SSH_HOST，激活 flagscale-myc 后执行 TinyLlama MemRift 训练"
echo "============================================"
echo "REMOTE_DIR: $REMOTE_DIR"
echo "============================================"

# 在远端：先 source conda，再激活环境，再 cd 到仓库并执行训练脚本
ssh "$SSH_HOST" "bash -l -c \"
  set -e
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
  conda activate $CONDA_ENV
  cd '$REMOTE_DIR'
  export TINYLLAMA_MODEL='$TINYLLAMA_MODEL'
  export MEMRIFT_WEIGHT_DIR='$MEMRIFT_WEIGHT_DIR'
  export TRAIN_ITERS='$TRAIN_ITERS'
  export SKIP_PREPARE='$SKIP_PREPARE'
  ./scripts/run_tinyllama_memrift_train.sh
\""

echo ">> 远端训练脚本已结束"
