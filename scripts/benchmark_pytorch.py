#!/usr/bin/env python3
import argparse, json, time, pathlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd

def load_prompts(path):
    with open(path, 'r') as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    ap.add_argument('--dtype', default='bf16', choices=['bf16','fp16','fp32'])
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--prompts_file', default='data/prompts.txt')
    ap.add_argument('--out', default='metrics_pytorch.json')
    args = ap.parse_args()

    prompts = load_prompts(args.prompts_file)
    batches = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]

    dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    dtype = dtype_map[args.dtype]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'[INFO] Loading {args.model} on {device} ({args.dtype})')
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map='auto' if device=='cuda' else None
    )
    model.to(device)
    model.eval()

    total_new_tokens, total_time = 0, 0.0
    per_batch = []

    with torch.inference_mode():
        for batch in tqdm(batches, desc='Batches'):
            inputs = tok(batch, return_tensors='pt', padding=True).to(device)
            t0 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            dt = time.perf_counter() - t0
            new_tokens = (out.shape[-1] - inputs['input_ids'].shape[-1]) * len(batch)
            total_new_tokens += int(new_tokens)
            total_time += dt
            per_batch.append({'batch_size': len(batch), 'new_tokens': int(new_tokens), 'seconds': float(dt)})

    tps = (total_new_tokens / total_time) if total_time > 0 else 0.0
    metrics = {
        'framework': 'pytorch/transformers',
        'model': args.model,
        'dtype': args.dtype,
        'device': device,
        'batch_size': args.batch_size,
        'max_new_tokens': args.max_new_tokens,
        'total_new_tokens': int(total_new_tokens),
        'total_time_sec': float(total_time),
        'tokens_per_sec': float(tps),
        'per_batch': per_batch,
        'notes': 'Baseline PyTorch generation benchmark.'
    }

    out_path = pathlib.Path(args.out)
    out_path.write_text(json.dumps(metrics, indent=2))
    csv_path = out_path.with_suffix('.csv')
    pd.DataFrame(per_batch).to_csv(csv_path, index=False)
    print(f'[OK] Wrote {out_path} and {csv_path}')
    print(f'[RESULT] tokens/sec = {tps:.3f} (total_new_tokens={total_new_tokens}, total_time={total_time:.2f}s)')

if __name__ == '__main__':
    main()
