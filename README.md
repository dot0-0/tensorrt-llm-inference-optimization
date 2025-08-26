# TensorRT-LLM Inference Optimization (LLaMA-2-7B)

End-to-end sample with PyTorch baseline vs TensorRT-LLM, Nsight profiling, and ready-made plots.

## Quickstart
1) `pip install -r requirements.txt`
2) `python scripts/benchmark_pytorch.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out metrics_pytorch.json`
3) (GPU) Use `scripts/build_trtllm_engine.sh` then `scripts/benchmark_trtllm.sh`
4) `python src/plot_metrics.py` to render plots from mock or real metrics.
