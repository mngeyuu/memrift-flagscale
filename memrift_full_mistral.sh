#!/bin/bash
# MemRift 单卡全路径(权重+激活)端到端验证 —— Mistral-7B-Instruct-v0.2
# 基于 memrift_full.sh,替换为本环境可用的 Mistral 资源:
#   压缩权重: memrift_weights/mistral7b_v02_level18 (HF-mode split_zstd)
#   模型/分词器: models/Mistral-7B-Instruct-v0.2
# 用于 D2H/H2D cudaMemcpyAsync 改造的 baseline / after 对比(逐 iter loss)。
source /root/miniconda3/etc/profile.d/conda.sh
conda activate myc
cd /share/project/mengyc/code/memrift-flagscale

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH=/share/project/mengyc/code/memrift-flagscale:/share/project/mengyc/code/memrift-flagscale/flagscale/train:${PYTHONPATH}
export MEMRIFT_TE_PATCH_TRACE=0
export MEMRIFT_DISABLE_LINEAR_BWD_PRE=0
export MEMRIFT_HOOK_ORDER=1

MODEL_PATH=/share/project/mengyc/models/Mistral-7B-Instruct-v0.2

torchrun --nnodes 1 --nproc_per_node 1 --master_port ${MASTER_PORT:-29518} \
  flagscale/train/train_gpt.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --disable-bias-linear \
  --bf16 \
  --memrift-enable \
  --memrift-weight-enable \
  --memrift-compressed-weight-dir ./memrift_weights/mistral7b_v02_level18 \
  --memrift-zstd-level 3 \
  --memrift-prefetch-layers 12 \
  --memrift-weight-async \
  --memrift-activation-enable \
  --memrift-act-async \
  --memrift-decode-pool-workers 4 \
  --memrift-compress-pool-workers 4 \
  --memrift-print-debug \
  --num-layers 32 \
  --hidden-size 4096 \
  --ffn-hidden-size 14336 \
  --num-attention-heads 32 \
  --num-query-groups 8 \
  --seq-length 2048 \
  --max-position-embeddings 32768 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-path ${MODEL_PATH} \
  --tokenizer-model ${MODEL_PATH} \
  --make-vocab-size-divisible-by 64 \
  --use-rotary-position-embeddings \
  --rotary-base 1000000 \
  --no-position-embedding \
  --normalization RMSNorm \
  --norm-epsilon 1e-05 \
  --swiglu \
  --untie-embeddings-and-output-weights \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --transformer-impl transformer_engine \
  --peft-type lora \
  --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
  --lora-dim 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --micro-batch-size 1 \
  --global-batch-size 1 \
  --mock-data \
  --train-iters 1 \
  --lr 0.0002 \
  --min-lr 2e-05 \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-08 \
  --log-interval 1 \
  --ckpt-format torch
