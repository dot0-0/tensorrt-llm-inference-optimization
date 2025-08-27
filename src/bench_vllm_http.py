#!/usr/bin/env python3
import argparse, time, json, requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/generate")
    ap.add_argument("--prompts_file", default="data/prompts.txt")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    prompts = [ln.strip() for ln in open(args.prompts_file) if ln.strip()]
    t0 = time.time()
    total_new = 0
    for p in prompts:
        payload = {"prompt": p, "max_tokens": args.max_new_tokens}
        r = requests.post(args.url, json=payload, timeout=120)
        r.raise_for_status()
        out = r.json()
        txt = out.get("text") or out.get("generated_text") or ""
        total_new += len(txt.split())
    dt = time.time() - t0
    tps = total_new / dt if dt > 0 else 0.0
    print(json.dumps({"total_new_tokens": total_new, "total_time_sec": dt, "tokens_per_sec": tps}, indent=2))

if __name__ == "__main__":
    main()

