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
mkdir -p /home/secure/myc/memrift-flagscale/outputs/llama31_8b_context_limit_act_weight_nw0p/pure_lora/seq_3456/checkpoints
mkdir -p /home/secure/myc/memrift-flagscale/outputs/llama31_8b_context_limit_act_weight_nw0p/pure_lora/seq_3456/checkpoints
mkdir -p /home/secure/myc/memrift-flagscale/outputs/llama31_8b_context_limit_act_weight_nw0p/pure_lora/seq_3456/logs
mkdir -p /home/secure/myc/memrift-flagscale/outputs/llama31_8b_context_limit_act_weight_nw0p/pure_lora/seq_3456/logs/pids
mkdir -p /home/secure/myc/memrift-flagscale/outputs/llama31_8b_context_limit_act_weight_nw0p/pure_lora/seq_3456/logs/details
mkdir -p /home/secure/myc/memrift-flagscale/outputs/llama31_8b_context_limit_act_weight_nw0p/pure_lora/seq_3456/tensorboard
mkdir -p /home/secure/myc/memrift-flagscale/outputs/llama31_8b_context_limit_act_weight_nw0p/pure_lora/seq_3456/wandb

cd /home/secure/myc/memrift-flagscale

export PYTHONPATH=/home/secure/myc/memrift-flagscale:/home/secure/myc/memrift-flagscale/flagscale/train:${PYTHONPATH}

cmd="TORCH_DEVICE_BACKEND_AUTOLOAD=0 torchrun --nnodes 1 --nproc_per_node 1 --rdzv_id default --node_rank 0 --rdzv_backend c10d --rdzv_endpoint localhost:50973 --log_dir /home/secure/myc/memrift-flagscale/outputs/llama31_8b_context_limit_act_weight_nw0p/pure_lora/seq_3456/logs/details/host_0_localhost/20260529_183000.217486 --tee 3 flagscale/train/train_gpt.py --distributed-backend flagcx --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 --disable-bias-linear --use-flash-attn --no-rope-fusion --no-persist-layer-norm --bf16 --log-interval 1 --tensorboard-log-interval 10 --wandb-mode disabled --tensorboard-dir /home/secure/myc/memrift-flagscale/outputs/llama31_8b_context_limit_act_weight_nw0p/pure_lora/seq_3456/tensorboard --wandb-save-dir /home/secure/myc/memrift-flagscale/outputs/llama31_8b_context_limit_act_weight_nw0p/pure_lora/seq_3456/wandb --save-interval 500 --save /home/secure/myc/memrift-flagscale/outputs/llama31_8b_context_limit_act_weight_nw0p/pure_lora/seq_3456/checkpoints --load /home/secure/myc/memrift-flagscale/outputs/llama31_8b_context_limit_act_weight_nw0p/pure_lora/seq_3456/checkpoints --ckpt-format torch --memrift-compressed-weight-dir ./memrift_weights/llama31_8b_zstd_level1 --memrift-zstd-level 3 --memrift-prefetch-layers 4 --memrift-weight-async --memrift-activation-enable --memrift-act-async --memrift-decode-pool-workers 48 --memrift-compress-pool-workers 8 --num-workers 0 --num-layers 32 --hidden-size 4096 --ffn-hidden-size 14336 --num-attention-heads 32 --num-query-groups 8 --seq-length 3456 --max-position-embeddings 131072 --tokenizer-type HuggingFaceTokenizer --tokenizer-path /home/secure/myc/model/llama-3.1-8b --tokenizer-model /home/secure/myc/model/llama-3.1-8b --make-vocab-size-divisible-by 64 --norm-init-weight 1.0 --use-rotary-position-embeddings --rotary-base 500000 --no-position-embedding --normalization RMSNorm --norm-epsilon 1e-05 --swiglu --untie-embeddings-and-output-weights --attention-dropout 0.0 --hidden-dropout 0.0 --init-method-std 0.02 --transformer-impl local --peft-type lora --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 --lora-dim 16 --lora-alpha 32 --lora-dropout 0.0 --split 1,0,0 --micro-batch-size 1 --global-batch-size 8 --mock-data --train-iters 1 --lr 0.0002 --min-lr 2e-05 --lr-warmup-iters 0 --lr-decay-style cosine --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-08 --eval-iters 0 --eval-interval 2000000000"


bash -c "$cmd; sync" >> /home/secure/myc/memrift-flagscale/outputs/llama31_8b_context_limit_act_weight_nw0p/pure_lora/seq_3456/logs/host_0_localhost.output 2>&1

