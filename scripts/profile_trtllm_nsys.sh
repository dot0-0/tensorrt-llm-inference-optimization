#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="build/llama2-7b-bf16"
PROMPTS="data/prompts.txt"
OUT_REP="profiles/trtllm_run.qdrep"
mkdir -p profiles

if [ ! -d "$ENGINE_DIR" ]; then
  echo "[ERR] Engine not found at $ENGINE_DIR. Build it first."
  exit 1
fi

echo "[INFO] Profiling TRT-LLM run with Nsight Systems..."
nsys profile -o $OUT_REP --force-overwrite=true trtllm-run \
  --engine_dir $ENGINE_DIR --prompts_file $PROMPTS --max_output_len 128 --report_json metrics_trtllm.json

echo "[OK] Profile saved to $OUT_REP"
