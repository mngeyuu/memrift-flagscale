#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
L0="${L0:-2048}"
L1=$(python3 -c "print(int(${L0} * 1.2 + 0.5))")
CFG="${CONFIG_NAME:-train_mock}"
MAX_POS="${MAX_POS:-$L1}"
echo "config-name=$CFG  seq_length L0=$L0 -> L1=$L1  max_position_embeddings=$MAX_POS"
python run.py --config-path=examples/memrift/conf --config-name="$CFG" action=run \
  train.model.seq_length="$L1" \
  train.model.max_position_embeddings="$MAX_POS" \
  "$@"
