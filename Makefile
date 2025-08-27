# ---------- Makefile: TensorRT-LLM Inference Optimization ----------
SHELL := /bin/bash
.PHONY: help venv install baseline plots build-engine-bf16 trt trt-plots \
        build-engine-bf16-no-fmha ablation-compare profile-nsys parse-nsys plots-all \
        ncu-profile matrix mem-baseline mem-trt triton-bench dyn-batch clean docker-shell

# ---- Paths & Variables ----
PY := python
PIP := pip
VENV := .venv
PWD := $(shell pwd)

# Hugging Face / baseline
BASELINE_MODEL ?= TinyLlama/TinyLlama-1.1B-Chat-v1.0
PROMPTS := data/prompts.txt

# TRT-LLM docker image (adjust as needed)
DOCKER_IMAGE ?= nvcr.io/nvidia/trt-llm:24.07
DOCKER_RUN = docker run --gpus all -it --rm -v $(PWD):/workspace -w /workspace $(DOCKER_IMAGE)

# Configs & engines
CONFIG_BF16 := configs/llama2_7b_bf16.json
ENGINE_BF16 := build/llama2_7b_bf16

CONFIG_BF16_NO_FMHA := configs/llama2_7b_bf16_no_fmha.json
ENGINE_BF16_NO_FMHA := build/llama2_7b_bf16_no_fmha

CONFIG_TP2 := configs/llama2_7b_bf16_tp2.json
ENGINE_TP2 := build/llama2_7b_bf16_tp2

# Metrics & outputs
PT_JSON := metrics_pytorch.json
TRT_JSON := metrics_trtllm.json
NSYS_REP := profiles/trtllm_run.qdrep
NSYS_CSV := data/nsys_kernels.csv
PLOTS_DIR := plots

# ---- Help ----
help:
	@echo "Targets:"
	@echo "  venv                 Create local Python venv"
	@echo "  install              Install Python deps into venv"
	@echo "  baseline             Run PyTorch baseline benchmark"
	@echo "  plots                Render plots (baseline or with TRT if provided)"
	@echo "  build-engine-bf16    Build TRT engine (BF16 + FMHA)"
	@echo "  trt                  Run TRT benchmark on built engine"
	@echo "  trt-plots            Rebuild plots using PT + TRT metrics + Nsight CSV (if present)"
	@echo "  build-engine-bf16-no-fmha  Build TRT engine with FMHA disabled (ablation)"
	@echo "  ablation-compare     Print TRT vs PyTorch improvements"
	@echo "  profile-nsys         Capture Nsight Systems profile (.qdrep)"
	@echo "  parse-nsys           Convert .qdrep -> CSV and update plots"
	@echo "  ncu-profile          Collect Nsight Compute metrics for fmha kernels"
	@echo "  matrix               Sweep batch/prompt/max_new tokens across PT and TRT"
	@echo "  mem-baseline         Measure peak memory (PyTorch) via torch.cuda"
	@echo "  mem-trt              Measure peak memory (TRT) via nvidia-smi sampling"
	@echo "  triton-bench         Benchmark Triton toy SDPA vs PyTorch SDPA"
	@echo "  dyn-batch            Run dynamic batching simulator"
	@echo "  docker-shell         Open a shell in the TRT-LLM container (mounted repo)"
	@echo "  clean                Remove build artifacts and temporary files"

# ---- Python env ----
venv:
	python -m venv $(VENV)

install: venv
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt

# ---- Baseline (PyTorch) ----
baseline:
	. $(VENV)/bin/activate && \
	$(PY) scripts/benchmark_pytorch.py \
		--model "$(BASELINE_MODEL)" \
		--dtype bf16 \
		--batch_size 2 \
		--max_new_tokens 64 \
		--prompts_file $(PROMPTS) \
		--out $(PT_JSON)

plots:
	. $(VENV)/bin/activate && \
	$(PY) src/plot_metrics.py --pytorch $(PT_JSON)

# ---- TRT build & run ----
build-engine-bf16:
	$(DOCKER_RUN) bash scripts/build_engine.sh $(CONFIG_BF16) $(ENGINE_BF16)

trt:
	$(DOCKER_RUN) bash -lc 'trtllm-run --engine_dir $(ENGINE_BF16) --prompts_file $(PROMPTS) --max_output_len 256 --report_json $(TRT_JSON)'

trt-plots:
	. $(VENV)/bin/activate && \
	$(PY) src/plot_metrics.py --pytorch $(PT_JSON) --trt $(TRT_JSON) --nsys $(NSYS_CSV) --outdir $(PLOTS_DIR)

# ---- Ablation: FMHA off ----
build-engine-bf16-no-fmha:
	$(DOCKER_RUN) bash scripts/build_engine.sh $(CONFIG_BF16_NO_FMHA) $(ENGINE_BF16_NO_FMHA)

ablation-compare:
	. $(VENV)/bin/activate && \
	$(PY) scripts/compare_metrics.py --pytorch $(PT_JSON) --trt $(TRT_JSON)

# ---- Nsight Systems ----
profile-nsys:
	$(DOCKER_RUN) bash scripts/profile_trtllm_nsys.sh

parse-nsys:
	. $(VENV)/bin/activate && \
	$(PY) scripts/parse_nsys.py --rep $(NSYS_REP) --out $(NSYS_CSV) && \
	$(PY) src/plot_metrics.py --pytorch $(PT_JSON) --trt $(TRT_JSON) --nsys $(NSYS_CSV) --outdir $(PLOTS_DIR)

# ---- Nsight Compute (fmha kernels) ----
ncu-profile:
	$(DOCKER_RUN) bash scripts/profile_ncu_trt.sh

# ---- Matrix sweeps ----
matrix:
	. $(VENV)/bin/activate && \
	$(PY) scripts/run_matrix.py \
		--framework both \
		--model "$(BASELINE_MODEL)" \
		--engine_dir $(ENGINE_BF16) \
		--batch_sizes 1,2,4 \
		--prompt_lens 64,512,1024 \
		--max_new_tokens 32,64 \
		--batches_per_point 4 \
		--out_csv data/matrix_results.csv

# ---- Memory tracking ----
mem-baseline:
	. $(VENV)/bin/activate && \
	$(PY) scripts/memory_track_baseline.py --model "$(BASELINE_MODEL)" --prompts_file $(PROMPTS) > data/pt_peak.json && \
	echo "Wrote data/pt_peak.json"

mem-trt:
	$(DOCKER_RUN) bash scripts/peak_mem_watch.sh trtllm-run --engine_dir $(ENGINE_BF16) --prompts_file $(PROMPTS) --max_output_len 256 --report_json $(TRT_JSON)

# ---- Triton toy SDPA ----
triton-bench:
	. $(VENV)/bin/activate && \
	$(PY) scripts/bench_triton_sdp.py --S 512 --D 64

# ---- Dynamic batching sim ----
dyn-batch:
	. $(VENV)/bin/activate && \
	$(PY) scripts/run_dynamic_batch_sim.py --out_csv data/dyn_batch_sweep.csv

# ---- Docker convenience ----
docker-shell:
	$(DOCKER_RUN) /bin/bash

# ---- Clean ----
clean:
	rm -rf build profiles *.json *.csv $(PLOTS_DIR)/*.png data/matrix_results.csv data/dyn_batch_sweep.csv data/peak_memory.csv data/pt_peak.json
	@echo "[OK] Cleaned"

