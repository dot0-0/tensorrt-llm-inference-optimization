#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="build/llama2-7b-bf16"
PROMPTS="data/prompts.txt"
OUT_JSON="metrics_trtllm.json"

if [ ! -d "$ENGINE_DIR" ]; then
  echo "[ERR] Engine not found at $ENGINE_DIR. Build it first."
  exit 1
fi

echo "[INFO] Running TensorRT-LLM benchmark..."
trtllm-run --engine_dir $ENGINE_DIR --prompts_file $PROMPTS --max_output_len 256 --report_json $OUT_JSON

echo "[OK] Wrote $OUT_JSON"

