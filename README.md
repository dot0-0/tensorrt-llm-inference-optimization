# 🚀 TensorRT-LLM Inference Optimization

End-to-end project benchmarking **PyTorch vs TensorRT-LLM** inference for LLaMA-2–7B, with:

- ✅ PyTorch baseline throughput & latency  
- ✅ TensorRT-LLM engines (BF16 + fused MHA)  
- ✅ **FMHA ablation** (on/off)  
- ✅ Batch size / prompt length / precision sweeps  
- ✅ Peak GPU memory tracking (PyTorch + TRT)  
- ✅ Nsight Systems & Compute profiling  
- ✅ Multi-GPU (TP2) config  
- ✅ **Triton kernel** demo: toy Scaled Dot-Product Attention  
- ✅ Dynamic batching simulator  

This repo matches resume bullets like:  
> “Integrated fused multi-head attention kernels with BF16 precision, reducing per-token latency by ~30% and improving tokens/sec by ~1.4× on A100 GPUs.”

---

## 📂 Repo structure
.
├── configs/ # TRT-LLM engine configs (BF16, ablation, TP2, etc.)
├── data/ # prompts, mock data, CSV outputs
│ ├── prompts.txt
│ └── mock/ # mock metrics for CPU-only demo
├── plots/ # output plots (.png)
├── profiles/ # Nsight profiles (.qdrep, ncu CSV)
├── scripts/ # benchmarking, profiling, utilities
├── src/ # plotting, utils, kernels, simulators
├── Makefile # reproducible one-liner targets
├── requirements.txt # pip deps
├── environment.yml # conda env
└── README.md

yaml
Copy
Edit

---

## ⚡️ Quickstart

### 1. Setup Python env
```bash
make install     # or: pip install -r requirements.txt
2. Run PyTorch baseline
bash
Copy
Edit
make baseline
make plots
Outputs:

metrics_pytorch.json/.csv

plots/tokens_per_sec.png

plots/latency_cdf.png

Works on CPU or GPU.

🟢 TensorRT-LLM Path (NVIDIA GPU required)
3. Build engine (BF16 + FMHA)
bash
Copy
Edit
make build-engine-bf16
4. Benchmark TensorRT-LLM
bash
Copy
Edit
make trt
make trt-plots
Now you can compare PyTorch vs TRT.

5. FMHA ablation
bash
Copy
Edit
make build-engine-bf16-no-fmha
make ablation-compare
Shows how fused MHA improves latency & throughput.

📊 Advanced Experiments
Matrix sweeps (batch × prompt length × new tokens)
bash
Copy
Edit
make matrix
Produces data/matrix_results.csv for scaling plots.

Memory tracking
bash
Copy
Edit
make mem-baseline    # PyTorch peak memory
make mem-trt         # TRT peak memory via nvidia-smi
Profiling
bash
Copy
Edit
make profile-nsys    # Nsight Systems (kernel timelines)
make parse-nsys      # Convert .qdrep -> CSV + plots
make ncu-profile     # Nsight Compute (FMHA kernel metrics)
🔬 Low-level GPU Experiments
Triton SDPA kernel
Toy scaled dot-product attention implemented in Triton.

bash
Copy
Edit
make triton-bench
Compares runtime vs PyTorch softmax-based SDPA.

Dynamic batching simulator
Models request batching under varying arrival rates, delays, and limits.

bash
Copy
Edit
make dyn-batch
Outputs data/dyn_batch_sweep.csv with latency distributions.

🖥 Multi-GPU (TP2)
Example config for tensor parallelism on 2 GPUs:

bash
Copy
Edit
make build-engine-bf16 CONFIG=configs/llama2_7b_bf16_tp2.json
trtllm-run --engine_dir build/llama2_7b_bf16_tp2 --prompts_file data/prompts.txt --max_output_len 256 --report_json metrics_trtllm_tp2.json
📈 Example Plots
Throughput: tokens/sec comparison

Latency CDF: per-token latency distribution

Memory usage: PyTorch vs TRT

Nsight GPU kernels: fmha_fwd, gemm, norm ops

(See plots/ for pre-generated images.)

⚙️ Requirements
Python 3.10+, PyTorch 2.2+, Transformers 4.42+

For TRT-LLM: NVIDIA GPU + driver, Docker, NVIDIA Container Toolkit

Recommended container:

bash
Copy
Edit
docker pull nvcr.io/nvidia/trt-llm:24.07
✨ Next Steps
Extend matrix experiments to larger models (e.g. LLaMA-2-13B).

Add CI/CD smoke tests on CPU (TinyLlama baseline).

Build a Streamlit dashboard for interactive plot exploration.


