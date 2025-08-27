#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: $0 <config.json> [OUTPUT_DIR]"
  exit 1
fi
CONFIG="$1"
ENGINE_DIR="${2:-build/$(basename "$CONFIG" .json)}"
mkdir -p "$ENGINE_DIR"
echo "[INFO] Building engine: $ENGINE_DIR from $CONFIG"
trtllm-build --config "$CONFIG" --output_dir "$ENGINE_DIR"
echo "[OK] Engine built at $ENGINE_DIR"

