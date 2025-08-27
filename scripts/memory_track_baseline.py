#!/usr/bin/env python3
import argparse, json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--prompts_file', default='data/prompts.txt')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if device=='cuda' else torch.float32)
    model.to(device).eval()

    prompts = [ln.strip() for ln in open(args.prompts_file) if ln.strip()]
    inputs = tok(prompts, return_tensors='pt', padding=True).to(device)

    if device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    peak = torch.cuda.max_memory_allocated() if device=='cuda' else 0
    print(json.dumps({"device":device, "peak_bytes": int(peak), "peak_gb": float(peak)/1e9}, indent=2))

if __name__ == '__main__':
    main()

