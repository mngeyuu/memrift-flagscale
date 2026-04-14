#!/bin/bash
# run_vllm_memrift.sh — MemRift + vLLM layer-by-layer weight streaming
#
# Usage:
#   bash run_vllm_memrift.sh            # MemRift (default)
#   MEMRIFT=0 bash run_vllm_memrift.sh  # baseline (normal vLLM, no offload)
#
# GPU requirement: ~1 GB (MemRift) vs ~14 GB (baseline)

set -e
cd /share/project/mengyc/code/memrift-flagscale

source /root/miniconda3/etc/profile.d/conda.sh
conda activate flagscale-inference

export PYTHONPATH=/share/project/mengyc/code/memrift-flagscale:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HUB_DISABLE_XET=1

# MemRift tuning
export MEMRIFT_ZSTD_LEVEL="${MEMRIFT_ZSTD_LEVEL:-3}"
export MEMRIFT_DECODE_WORKERS="${MEMRIFT_DECODE_WORKERS:-16}"
export MEMRIFT_PREFETCH_LAYERS="${MEMRIFT_PREFETCH_LAYERS:-1}"

MEMRIFT="${MEMRIFT:-1}"

if [[ "${MEMRIFT}" == "1" ]]; then
  TAG="vllm_memrift"
  SCRIPT="flagscale/inference/inference_llm_memrift.py"
else
  TAG="vllm_baseline"
  SCRIPT="flagscale/inference/inference_llm.py"
fi

mkdir -p outputs/${TAG}

echo "============================================================"
echo "  vLLM Inference: MEMRIFT=${MEMRIFT}"
echo "  ZSTD_LEVEL=${MEMRIFT_ZSTD_LEVEL}  WORKERS=${MEMRIFT_DECODE_WORKERS}"
echo "  PREFETCH=${MEMRIFT_PREFETCH_LAYERS}"
echo "============================================================"

python ${SCRIPT} \
  --config-path examples/memrift/conf/inference_mistral_7b.yaml \
  2>&1 | tee outputs/${TAG}/inference.log

echo "=== DONE — log: outputs/${TAG}/inference.log ==="
