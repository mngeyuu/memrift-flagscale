#!/bin/bash
set -e
cd /share/project/mengyc/code/memrift-flagscale
source /root/miniconda3/etc/profile.d/conda.sh
conda activate myc-flagscale

export PYTHONPATH=/share/project/mengyc/code/memrift-flagscale:/share/project/mengyc/code/memrift-flagscale/flagscale/train:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export WANDB_MODE=offline
export HF_HUB_DISABLE_XET=1

MODEL_PATH=/share/project/mengyc/models/Mistral-7B-Instruct-v0.2
MEMRIFT_WEIGHTS=/share/project/mengyc/code/memrift_demo/output/memrift_weights_mistral7b_level18
TAG=fixed_mistral7b

COMMON="--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --disable-bias-linear --use-flash-attn --bf16 --log-interval 1 --eval-iters 0 --wandb-mode disabled --save-interval 9999 --no-load-optim --no-load-rng --ckpt-format torch --tokenizer-type HuggingFaceTokenizer --tokenizer-path ${MODEL_PATH} --tokenizer-model ${MODEL_PATH} --mock-data --micro-batch-size 1 --global-batch-size 8 --split 1,0,0 --train-iters 5 --lr 2e-4 --min-lr 2e-5 --lr-warmup-iters 2 --lr-decay-style cosine --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --peft-type lora --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 --lora-dim 16 --lora-alpha 32 --lora-dropout 0.0 --transformer-impl transformer_engine --attention-dropout 0.0 --hidden-dropout 0.0 --init-method-std 0.02"

ARCH="--num-layers 32 --hidden-size 4096 --ffn-hidden-size 14336 --num-attention-heads 32 --num-query-groups 8 --seq-length 2048 --max-position-embeddings 131072 --norm-init-weight 1.0 --use-rotary-position-embeddings --rotary-base 1000000 --no-position-embedding --normalization RMSNorm --norm-epsilon 1e-05 --swiglu --untie-embeddings-and-output-weights --make-vocab-size-divisible-by 64"

mkdir -p outputs/${TAG}_memrift outputs/${TAG}_pure_lora

echo "=== [1/2] MemRift (fixed: sm_cpu, k=1, span=1, always-release) ==="
torchrun --nnodes 1 --nproc_per_node 1 flagscale/train/train_gpt.py \
  $COMMON $ARCH \
  --save outputs/${TAG}_memrift/ckpt \
  --memrift-enable --memrift-weight-enable \
  --memrift-compressed-weight-dir ${MEMRIFT_WEIGHTS} \
  --memrift-zstd-level 3 --memrift-prefetch-layers 12 \
  --memrift-weight-async --memrift-activation-enable --memrift-act-async \
  --memrift-decode-pool-workers 48 --memrift-compress-pool-workers 8 \
  2>&1 | tee outputs/${TAG}_memrift/train.log
echo "=== MemRift DONE ==="

echo "=== [2/2] 纯 LoRA ==="
torchrun --nnodes 1 --nproc_per_node 1 flagscale/train/train_gpt.py \
  $COMMON $ARCH \
  --save outputs/${TAG}_pure_lora/ckpt \
  2>&1 | tee outputs/${TAG}_pure_lora/train.log
echo "=== 纯LoRA DONE ==="
