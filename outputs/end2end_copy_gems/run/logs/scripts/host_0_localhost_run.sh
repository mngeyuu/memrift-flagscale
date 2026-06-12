#!/bin/bash

CONDA_ENV_NAME="base"
if ! command -v torchrun >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
  fi
fi

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
mkdir -p /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/outputs/end2end_copy_gems/run/checkpoints
mkdir -p /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/outputs/end2end_copy_gems/run/checkpoints
mkdir -p /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/outputs/end2end_copy_gems/run/logs
mkdir -p /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/outputs/end2end_copy_gems/run/logs/pids
mkdir -p /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/outputs/end2end_copy_gems/run/logs/details
mkdir -p /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/outputs/end2end_copy_gems/run/tensorboard
mkdir -p /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/outputs/end2end_copy_gems/run/wandb

cd /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark

export PYTHONPATH=/root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark:/root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/flagscale/train:${PYTHONPATH}

cmd="TORCH_DEVICE_BACKEND_AUTOLOAD=0 CUDA_VISIBLE_DEVICES=0 GEMS_VENDOR=metax MEMRIFT_ACT_SPLIT_PATH=copy MEMRIFT_ACT_SPLIT_PROFILE=1 torchrun --nnodes 1 --nproc_per_node 1 --rdzv_id default --node_rank 0 --rdzv_backend c10d --rdzv_endpoint localhost:36327 --log_dir /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/outputs/end2end_copy_gems/run/logs/details/host_0_localhost/20260606_092220.024707 --tee 3 flagscale/train/train_gpt.py --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 --disable-bias-linear --use-flash-attn --bf16 --log-interval 1 --tensorboard-log-interval 10 --wandb-mode disabled --tensorboard-dir /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/outputs/end2end_copy_gems/run/tensorboard --wandb-save-dir /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/outputs/end2end_copy_gems/run/wandb --save-interval 500 --save /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/outputs/end2end_copy_gems/run/checkpoints --load /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/outputs/end2end_copy_gems/run/checkpoints --ckpt-format torch --memrift-enable --memrift-weight-enable --memrift-compressed-weight-dir /MXC550/share/myc/memrift-flagscale/memrift_weights/llama31_8b_zstd_level18 --memrift-zstd-level 3 --memrift-prefetch-layers 12 --memrift-weight-async --memrift-activation-enable --memrift-act-async --memrift-decode-pool-workers 48 --memrift-compress-pool-workers 8 --num-layers 32 --hidden-size 4096 --ffn-hidden-size 14336 --num-attention-heads 32 --num-query-groups 8 --seq-length 2048 --max-position-embeddings 131072 --tokenizer-type HuggingFaceTokenizer --tokenizer-path /MXC550/share/myc/model/llama-3.1-8b --tokenizer-model /MXC550/share/myc/model/llama-3.1-8b --make-vocab-size-divisible-by 64 --norm-init-weight 1.0 --use-rotary-position-embeddings --rotary-base 500000 --no-position-embedding --normalization RMSNorm --norm-epsilon 1e-05 --swiglu --untie-embeddings-and-output-weights --attention-dropout 0.0 --hidden-dropout 0.0 --init-method-std 0.02 --transformer-impl local --peft-type lora --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 --lora-dim 16 --lora-alpha 32 --lora-dropout 0.0 --split 1,0,0 --micro-batch-size 1 --global-batch-size 8 --mock-data --train-iters 1 --lr 0.0002 --min-lr 2e-05 --lr-warmup-iters 0 --lr-decay-style cosine --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-08"


bash -c "$cmd; sync" >> /root/.config/superpowers/worktrees/memrift-flagscale/memrift-split-copy-benchmark/outputs/end2end_copy_gems/run/logs/host_0_localhost.output 2>&1

