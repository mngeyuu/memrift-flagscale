#!/bin/bash
# ============================================================
# MemRift 权重显存诊断：TinyLlama-1.1B TP=1
#   MODE=memrift  -> memrift weight-only（关闭 activation 压缩，隔离权重路径）
#   MODE=lora     -> 纯 LoRA 基线（memrift 全关，base 权重全 bf16 常驻）
# 两种模式都会在 iter 0 自动打印 [MEM_PROBE] G/H/H_big/H_state/H2/K
# 额外开 MEMRIFT_HOOK_ORDER=1 观察 backward hook 触发序列
# ============================================================
set -e
cd /share/project/mengyc/code/memrift-flagscale
source /root/miniconda3/etc/profile.d/conda.sh
conda activate myc

MODE=${MODE:-memrift}
export PYTHONPATH=/share/project/mengyc/code/memrift-flagscale:/share/project/mengyc/code/memrift-flagscale/flagscale/train
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MASTER_PORT=${MASTER_PORT:-29533}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export WANDB_MODE=offline
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEMRIFT_HOOK_ORDER=${MEMRIFT_HOOK_ORDER:-1}

TOK=TinyLlama/TinyLlama-1.1B-Chat-v1.0
MEMRIFT_WEIGHTS=./memrift_weights/tinyllama_1b_level18
OUTDIR=output/diag_tinyllama_${MODE}
mkdir -p ${OUTDIR}

COMMON="\
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --disable-bias-linear --use-flash-attn --bf16 \
  --log-interval 1 --eval-iters 0 \
  --save-interval 99999 --no-load-optim --no-load-rng --ckpt-format torch \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-path ${TOK} --tokenizer-model ${TOK} \
  --mock-data --micro-batch-size 1 --global-batch-size 1 --split 1,0,0 \
  --train-iters ${TRAIN_ITERS:-3} --lr 2e-4 --min-lr 2e-5 --lr-warmup-iters 1 --lr-decay-style cosine \
  --weight-decay 0.1 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
  --peft-type lora --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
  --lora-dim 16 --lora-alpha 32 --lora-dropout 0.0 \
  --transformer-impl transformer_engine \
  --attention-dropout 0.0 --hidden-dropout 0.0 --init-method-std 0.02"

ARCH="\
  --num-layers 22 --hidden-size 2048 --ffn-hidden-size 5632 \
  --num-attention-heads 32 --num-query-groups 4 \
  --seq-length 2048 --max-position-embeddings 2048 \
  --use-rotary-position-embeddings --rotary-base 10000 --no-position-embedding \
  --normalization RMSNorm --norm-epsilon 1e-5 --swiglu --make-vocab-size-divisible-by 1"

PREFETCH=${PREFETCH:-4}
if [ "$MODE" = "memrift" ]; then
  # canonical：开 activation + prefetch 4，复现"能跑通但显存没省"
  ACT_ARGS="--memrift-activation-enable --memrift-act-async"
  [ "${NO_ACT:-0}" = "1" ] && ACT_ARGS=""   # NO_ACT=1 隔离权重路径（可能触发崩溃）
  MEMRIFT_ARGS="\
    --memrift-enable --memrift-weight-enable \
    --memrift-compressed-weight-dir ${MEMRIFT_WEIGHTS} \
    --memrift-zstd-level ${ZLEVEL:-18} --memrift-prefetch-layers ${PREFETCH} --memrift-weight-async \
    --memrift-decode-pool-workers 16 --memrift-compress-pool-workers 8 ${ACT_ARGS}"
else
  MEMRIFT_ARGS=""   # 纯 LoRA：memrift 全关
fi

echo "=== DIAG MODE=${MODE} ==="
torchrun --nnodes 1 --nproc_per_node 1 --master_port ${MASTER_PORT} \
  flagscale/train/train_gpt.py \
  ${COMMON} ${ARCH} ${MEMRIFT_ARGS} \
  --save ${OUTDIR}/ckpt \
  2>&1 | tee ${OUTDIR}/train.log
echo "=== DONE MODE=${MODE} ==="
