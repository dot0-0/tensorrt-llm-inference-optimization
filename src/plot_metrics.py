#!/usr/bin/env python3
import argparse, json, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def ecdf(data):
    x = np.sort(np.array(data))
    y = np.arange(1, len(x)+1) / float(len(x))
    return x, y

def per_token_latencies(per_batch):
    vals = []
    for b in per_batch:
        if b['new_tokens'] > 0 and b['seconds'] > 0:
            vals.append(b['seconds'] / b['new_tokens'])
    return vals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pytorch', default='metrics_pytorch.json')
    ap.add_argument('--trt', default=None)
    ap.add_argument('--nsys', default=None)  # optional CSV
    ap.add_argument('--mem', default=None)   # optional CSV with columns: system,gb
    ap.add_argument('--outdir', default='plots')
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    m_pt = load_json(args.pytorch)
    m_trt = load_json(args.trt) if args.trt else None

    # Throughput
    labels, tps = ['PyTorch'], [m_pt['tokens_per_sec']]
    if m_trt:
        labels.append('TensorRT-LLM')
        tps.append(m_trt['tokens_per_sec'])
    plt.figure()
    plt.bar(labels, tps)
    plt.ylabel('Tokens per second')
    plt.title('Throughput')
    plt.savefig(outdir / 'tokens_per_sec.png', bbox_inches='tight')
    plt.close()

    # Latency CDF
    lat_pt = per_token_latencies(m_pt['per_batch'])
    plt.figure()
    x1, y1 = ecdf(lat_pt)
    plt.plot(x1, y1, label='PyTorch')
    if m_trt:
        lat_trt = per_token_latencies(m_trt['per_batch'])
        x2, y2 = ecdf(lat_trt)
        plt.plot(x2, y2, label='TensorRT-LLM')
    plt.xlabel('Per-token latency (s/token)')
    plt.ylabel('CDF')
    plt.title('Latency Distribution (lower is better)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(outdir / 'latency_cdf.png', bbox_inches='tight')
    plt.close()

    # Memory usage (optional)
    if args.mem and pathlib.Path(args.mem).exists():
        mem = pd.read_csv(args.mem)
        plt.figure()
        plt.bar(mem['system'], mem['gb'])
        plt.ylabel('Peak GPU memory (GB)')
        plt.title('Memory usage')
        plt.savefig(outdir / 'gpu_memory.png', bbox_inches='tight')
        plt.close()

    # Nsight kernels (optional)
    if args.nsys and pathlib.Path(args.nsys).exists():
        nsys = pd.read_csv(args.nsys)
        name_col = 'Name' if 'Name' in nsys.columns else nsys.columns[0]
        time_col = None
        for cand in ['Total Time (ms)', 'Time (ms)', 'Total Time (ns)']:
            if cand in nsys.columns:
                time_col = cand
                break
        if time_col is None and len(nsys.columns) > 1:
            time_col = nsys.columns[1]
        if time_col:
            top = nsys.sort_values(by=time_col, ascending=False).head(8)
            plt.figure()
            plt.barh(top[name_col], top[time_col])
            plt.xlabel(time_col)
            plt.title('Nsight Systems: Top GPU Kernels')
            plt.gca().invert_yaxis()
            plt.savefig(outdir / 'nsys_kernel_time.png', bbox_inches='tight')
            plt.close()

    print('[OK] Wrote plots to', outdir)

if __name__ == '__main__':
    main()

