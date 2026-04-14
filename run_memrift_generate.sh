#!/bin/bash
# run_memrift_generate.sh — MemRift layer-by-layer weight-streaming generation
#
# GPU peak ≈ 1 transformer layer (~400 MB) rather than full model (13.5 GB).
#
# Usage:
#   bash run_memrift_generate.sh                       # MemRift greedy
#   GREEDY=0 PROMPT="Hello" bash run_memrift_generate.sh  # sampling
#   MEMRIFT=0 bash run_memrift_generate.sh             # baseline (full model)

set -e
cd /share/project/mengyc/code/memrift-flagscale

source /root/miniconda3/etc/profile.d/conda.sh
conda activate myc-flagscale

export PYTHONPATH=/share/project/mengyc/code/memrift-flagscale:/share/project/mengyc/code/memrift-flagscale/flagscale/train:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HUB_DISABLE_XET=1

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH=/share/project/mengyc/models/Mistral-7B-Instruct-v0.2
MEMRIFT_WEIGHTS=/share/project/mengyc/code/memrift_demo/output/memrift_weights_mistral7b_level18

# ── Generation params ────────────────────────────────────────────────────────
PROMPT="${PROMPT:-Once upon a time}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"
GREEDY="${GREEDY:-1}"
MEMRIFT="${MEMRIFT:-1}"

# ── Mistral-7B architecture ──────────────────────────────────────────────────
ARCH="--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --num-layers 32 --hidden-size 4096 --ffn-hidden-size 14336 \
  --num-attention-heads 32 --num-query-groups 8 \
  --seq-length 2048 --max-position-embeddings 131072 \
  --norm-init-weight 1.0 --use-rotary-position-embeddings \
  --rotary-base 1000000 --no-position-embedding \
  --normalization RMSNorm --norm-epsilon 1e-05 --swiglu \
  --untie-embeddings-and-output-weights --make-vocab-size-divisible-by 64 \
  --transformer-impl transformer_engine \
  --use-flash-attn --bf16 --disable-bias-linear"

COMMON="--tokenizer-type HuggingFaceTokenizer \
  --tokenizer-path ${MODEL_PATH} --tokenizer-model ${MODEL_PATH} \
  --no-load-optim --no-load-rng"

# ── Build generation flags ────────────────────────────────────────────────────
GEN_FLAGS="--prompt \"${PROMPT}\" --max-new-tokens ${MAX_NEW_TOKENS}"
[[ "${GREEDY}" == "1" ]] && GEN_FLAGS="${GEN_FLAGS} --greedy"

# ── Build MemRift flags ───────────────────────────────────────────────────────
if [[ "${MEMRIFT}" == "1" ]]; then
  MEMRIFT_FLAGS="--memrift-enable --memrift-weight-enable \
    --memrift-compressed-weight-dir ${MEMRIFT_WEIGHTS} \
    --memrift-zstd-level 3 --memrift-prefetch-layers 1 \
    --memrift-weight-async --memrift-decode-pool-workers 48"
  TAG="memrift_generate"
else
  MEMRIFT_FLAGS=""
  TAG="baseline_generate"
fi

mkdir -p outputs/${TAG}

echo "============================================================"
echo "  MemRift GPT Generate"
echo "  MEMRIFT      : ${MEMRIFT}"
echo "  PROMPT       : ${PROMPT}"
echo "  MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "  GREEDY       : ${GREEDY}"
echo "============================================================"

eval torchrun --nnodes 1 --nproc_per_node 1 \
  flagscale/train/generate_gpt.py \
  ${ARCH} ${COMMON} ${GEN_FLAGS} ${MEMRIFT_FLAGS} \
  2>&1 | tee outputs/${TAG}/generate.log

echo "=== DONE — log saved to outputs/${TAG}/generate.log ==="
