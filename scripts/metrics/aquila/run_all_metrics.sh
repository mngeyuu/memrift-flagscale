#!/usr/bin/env bash
# Run all Aquila2-7B MemRift metrics and print the final acceptance table.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"
activate_env
setup_common_env
source "$SCRIPT_DIR/model_env.sh"

run_all_metric_suite "$SCRIPT_DIR"
