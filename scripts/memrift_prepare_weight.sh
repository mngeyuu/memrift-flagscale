#!/bin/bash
# ============================================
# MemRift Weight Preparation Script
# ============================================
# This script compresses HuggingFace model weights to MemRift format
#
# Usage:
#   ./scripts/memrift_prepare_weight.sh /path/to/hf/model /path/to/output/dir [level]
#
# Arguments:
#   $1 - Path to HuggingFace model
#   $2 - Output directory for compressed weights
#   $3 - Zstd compression level (optional, default 18)
#
# Example:
#   ./scripts/memrift_prepare_weight.sh \
#       /models/TinyLlama-1.1B-Chat-v1.0 \
#       ./memrift_weights/tinyllama_1b_level18 \
#       18

set -e

MODEL_PATH=${1:?"Usage: $0 <model_path> <output_dir> [level]"}
OUTPUT_DIR=${2:?"Usage: $0 <model_path> <output_dir> [level]"}
LEVEL=${3:-18}

echo "============================================"
echo "MemRift Weight Preparation"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Compression Level: $LEVEL"
echo "============================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run compression
python -m flagscale.compress.memrift.offline_comp.prepare_weight \
    --model "$MODEL_PATH" \
    --outdir "$OUTPUT_DIR" \
    --level "$LEVEL"

echo "============================================"
echo "Compression complete!"
echo "Output directory: $OUTPUT_DIR"
echo "Files created:"
ls -lh "$OUTPUT_DIR"
echo "============================================"
