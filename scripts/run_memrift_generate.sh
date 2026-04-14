#!/usr/bin/env bash
# run_memrift_generate.sh — MemRift layer-by-layer weight-streaming generation
#
# Runs generate_gpt.py with MemRift inference hooks:
#   - Frozen weights decompressed one layer at a time (CPU → GPU)
#   - GPU peak ≈ 1 transformer layer (~400 MB) rather than full model
#
# Usage:
#   # Mistral-7B (uses local tokenizer path set in the YAML)
#   MODEL=mistral_7b ./scripts/run_memrift_generate.sh
#
#   # Llama-3.1-8B
#   MODEL=llama31_8b ./scripts/run_memrift_generate.sh
#
#   # Override prompt / token count
#   MODEL=mistral_7b PROMPT="Tell me a story" MAX_NEW_TOKENS=100 \
#     ./scripts/run_memrift_generate.sh
#
#   # Dry-run (print command only, don't execute)
#   MODEL=mistral_7b ./scripts/run_memrift_generate.sh dryrun
#
# Required env vars (overrides YAML defaults):
#   MODEL                - mistral_7b | llama31_8b  (default: mistral_7b)
#   COMP_WEIGHT_DIR      - path to compressed weights (default: from YAML)
#   TOKENIZER_PATH       - HF tokenizer path / model id (default: from YAML)
#   PROMPT               - generation prompt (default: "Once upon a time")
#   MAX_NEW_TOKENS       - tokens to generate (default: 50)
#   GREEDY               - 1=greedy decode, 0=sampling (default: 1)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ── Defaults ────────────────────────────────────────────────────────────────
MODEL="${MODEL:-mistral_7b}"
PROMPT="${PROMPT:-Once upon a time}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"
GREEDY="${GREEDY:-1}"
ACTION="${1:-run}"

case "${MODEL}" in
  mistral_7b)
    CONFIG_NAME="generate_mistral_7b"
    DEFAULT_COMP_DIR="./memrift_weights/mistral_7b_level18"
    ;;
  llama31_8b)
    CONFIG_NAME="generate_llama31_8b"
    DEFAULT_COMP_DIR="./memrift_weights/llama31_8b_level18"
    ;;
  *)
    echo "Unknown MODEL=${MODEL}. Use mistral_7b or llama31_8b." >&2
    exit 1
    ;;
esac

COMP_WEIGHT_DIR="${COMP_WEIGHT_DIR:-${DEFAULT_COMP_DIR}}"

# ── Build run.py overrides ───────────────────────────────────────────────────
OVERRIDES=(
  "action=${ACTION}"
  "train.system.memrift_compressed_weight_dir=${COMP_WEIGHT_DIR}"
  "train.model.max_new_tokens=${MAX_NEW_TOKENS}"
  "train.model.prompt=${PROMPT}"
)

if [[ -n "${TOKENIZER_PATH:-}" ]]; then
  OVERRIDES+=(
    "train.model.tokenizer_path=${TOKENIZER_PATH}"
    "train.model.tokenizer_model=${TOKENIZER_PATH}"
  )
fi

if [[ "${GREEDY}" == "1" ]]; then
  OVERRIDES+=("train.model.greedy=true")
else
  OVERRIDES+=("train.model.greedy=false")
fi

# ── Execute ──────────────────────────────────────────────────────────────────
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/flagscale/train:${PYTHONPATH:-}"

CMD=(python run.py
  --config-path=examples/memrift/conf
  "--config-name=${CONFIG_NAME}"
  "${OVERRIDES[@]}"
)

echo "=== MemRift generate: MODEL=${MODEL} ==="
echo "    comp_dir  : ${COMP_WEIGHT_DIR}"
echo "    prompt    : ${PROMPT}"
echo "    max_tokens: ${MAX_NEW_TOKENS}"
echo ""
echo "CMD: ${CMD[*]}"
echo ""

if [[ "${ACTION}" == "dryrun" ]]; then
  echo "[dryrun] command printed above, not executed."
  exit 0
fi

exec "${CMD[@]}"
