#!/usr/bin/env bash

MODEL_KEY="aquila"
MODEL_NAME="Aquila2-7B"
CONFIG_NAME="${CONFIG_NAME:-train_aquila2_7b_mock}"
MODEL_PATH="${MODEL_PATH:-/share/project/mengyc/models/Aquila2-7B}"
MEGATRON_CKPT_DIR="${MEGATRON_CKPT_DIR:-/share/project/mengyc/models/Aquila2-7B-mcore-tp1}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$REPO_ROOT/memrift_weights/aquila2_7b_level18}"
ALPACA_DATA_PATH="${ALPACA_DATA_PATH:-/share/project/mengyc/data/alpaca_megatron/alpaca_text_document}"
BASE_SEQ_LEN="${BASE_SEQ_LEN:-4096}"
TARGET_SEQ_LEN="${TARGET_SEQ_LEN:-$(ceil_120_percent "$BASE_SEQ_LEN")}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-$TARGET_SEQ_LEN}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/output/metrics/$MODEL_KEY}"
