#!/bin/bash
set -u
cd /share/project/mengyc/code/memrift-flagscale
source /root/miniconda3/etc/profile.d/conda.sh
conda activate myc-flagscale

export PYTHONPATH=/share/project/mengyc/code/memrift-flagscale:/share/project/mengyc/code/memrift-flagscale/flagscale/train:
export CUDA_VISIBLE_DEVICES=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export WANDB_MODE=offline
export HF_HUB_DISABLE_XET=1

MODEL_PATH=/share/project/mengyc/models/Meta-Llama-3-8B-Instruct
MEMRIFT_WEIGHTS=/share/project/mengyc/code/memrift-flagscale/memrift_weights/llama31_8b_level18
OUTDIR=output/bench_load
mkdir -p $OUTDIR

COMMON_ARGS="--num-layers 32 --hidden-size 4096 --ffn-hidden-size 14336 \
  --num-attention-heads 32 --num-query-groups 8 \
  --seq-length 2048 --max-position-embeddings 131072 \
  --norm-init-weight 1.0 --use-rotary-position-embeddings \
  --rotary-base 500000 --no-position-embedding \
  --normalization RMSNorm --norm-epsilon 1e-05 --swiglu \
  --untie-embeddings-and-output-weights --make-vocab-size-divisible-by 64 \
  --disable-bias-linear --use-flash-attn --bf16 \
  --transformer-impl transformer_engine \
  --attention-dropout 0.0 --hidden-dropout 0.0 --init-method-std 0.02 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-path ${MODEL_PATH} \
  --tokenizer-model ${MODEL_PATH} \
  --train-iters 1 --eval-iters 0 \
  --micro-batch-size 1 --global-batch-size 1 \
  --lr 1e-4 --min-lr 1e-5 --lr-decay-style cosine --lr-warmup-iters 0 \
  --mock-data --split 1,0,0 \
  --no-load-optim --no-load-rng \
  --wandb-mode disabled \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --prompt 'Once upon a time' --max-new-tokens 20"

echo "========================================"
echo "=== [1/2] Standard (Pure LoRA base) ==="
echo "========================================"
T_START=$(date +%s%3N)
echo "[BENCH] start_ms=${T_START}"
torchrun --nnodes 1 --nproc_per_node 1 flagscale/train/generate_gpt.py \
  ${COMMON_ARGS} \
  2>&1 | tee ${OUTDIR}/standard.log
T_END=$(date +%s%3N)
echo "[BENCH] end_ms=${T_END}"
STANDARD_MS=$((T_END - T_START))
echo "[BENCH] total_wall_ms=${STANDARD_MS}"

echo ""
echo "========================================"
echo "=== [2/2] MemRift ==="
echo "========================================"
T_START=$(date +%s%3N)
echo "[BENCH] start_ms=${T_START}"
torchrun --nnodes 1 --nproc_per_node 1 flagscale/train/generate_gpt.py \
  ${COMMON_ARGS} \
  --memrift-enable \
  --memrift-weight-enable \
  --memrift-compressed-weight-dir ${MEMRIFT_WEIGHTS} \
  --memrift-zstd-level 3 \
  --memrift-prefetch-layers 12 \
  --memrift-weight-async \
  --memrift-activation-enable \
  --memrift-act-async \
  --memrift-decode-pool-workers 48 \
  --memrift-compress-pool-workers 8 \
  2>&1 | tee ${OUTDIR}/memrift.log
T_END=$(date +%s%3N)
echo "[BENCH] end_ms=${T_END}"
MEMRIFT_MS=$((T_END - T_START))
echo "[BENCH] total_wall_ms=${MEMRIFT_MS}"

echo ""
echo "========================================"
echo "=== FINAL BENCH RESULTS ==="
python3 -c "
import re

def parse_log(path):
    txt = open(path).read()
    ttft  = re.search(r'Time to first token\s*:\s*([\d.]+)\s*ms', txt)
    total = re.search(r'Total time\s*:\s*([\d.]+)\s*ms', txt)
    return (float(ttft.group(1)) if ttft else None,
            float(total.group(1)) if total else None)

std_log  = '${OUTDIR}/standard.log'
mrf_log  = '${OUTDIR}/memrift.log'
std_wall = ${STANDARD_MS}
mrf_wall = ${MEMRIFT_MS}

std_ttft, std_gen = parse_log(std_log)
mrf_ttft, mrf_gen = parse_log(mrf_log)

std_load = std_wall - (std_gen or 0)
mrf_load = mrf_wall - (mrf_gen or 0)

print(f'Standard wall time   : {std_wall/1000:.1f} s')
print(f'  Generation time    : {(std_gen or 0)/1000:.1f} s')
print(f'  LOAD TIME          : {std_load/1000:.1f} s')
print()
print(f'MemRift wall time    : {mrf_wall/1000:.1f} s')
print(f'  Generation time    : {(mrf_gen or 0)/1000:.1f} s')
print(f'  LOAD TIME          : {mrf_load/1000:.1f} s')
print()
if std_load > 0 and mrf_load > 0:
    diff = (std_load - mrf_load) / std_load * 100
    print(f'MemRift load speedup : {diff:.1f}% faster')
"
echo "=== BENCH DONE ==="
