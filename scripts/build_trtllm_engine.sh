#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/llama2_7b_bf16.json"
ENGINE_DIR="build/llama2-7b-bf16"
mkdir -p "$ENGINE_DIR"

echo "[INFO] Building TensorRT-LLM engine to $ENGINE_DIR using $CONFIG"
trtllm-build --config $CONFIG --output_dir $ENGINE_DIR

echo "[OK] Engine built at $ENGINE_DIR"

