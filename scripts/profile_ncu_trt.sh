i#!/usr/bin/env bash
set -euo pipefail
ENGINE_DIR="${1:-build/llama2_7b_bf16}"
PROMPTS="data/prompts.txt"
OUT="profiles/ncu_fmha"
mkdir -p profiles
# Note: adjust metrics if your ncu version differs.
ncu --target-processes all \
    --kernel-name-base demangled \
    --kernel-name-pattern "fmha" \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv --log-file ${OUT}.csv \
    trtllm-run --engine_dir "$ENGINE_DIR" --prompts_file "$PROMPTS" --max_output_len 128 --report_json metrics_trtllm.json
echo "[OK] Wrote ${OUT}.csv"

