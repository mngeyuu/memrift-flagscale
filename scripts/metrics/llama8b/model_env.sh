#!/usr/bin/env bash

MODEL_KEY="llama8b"
MODEL_NAME="LLaMA-3.1-8B"
CONFIG_NAME="${CONFIG_NAME:-train_llama31_8b_mock}"
MODEL_PATH="${MODEL_PATH:-/share/project/mengyc/models/Llama-3.1-8B}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$REPO_ROOT/memrift_weights/llama31_8b_level18}"
MEMRIFT_PREPARE_LEVEL="${MEMRIFT_PREPARE_LEVEL:-18}"
ALPACA_DATA_PATH="${ALPACA_DATA_PATH:-/share/project/mengyc/data/alpaca_megatron/alpaca_text_document}"
BASE_SEQ_LEN="${BASE_SEQ_LEN:-4096}"
TARGET_SEQ_LEN="${TARGET_SEQ_LEN:-$(ceil_120_percent "$BASE_SEQ_LEN")}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-131072}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/output/metrics/$MODEL_KEY}"
